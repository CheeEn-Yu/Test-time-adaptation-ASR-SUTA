import os
import copy
import logging
import matplotlib.pyplot as plt
import torch
import hydra
from jiwer import wer
from omegaconf import OmegaConf
from whisper.normalizers import EnglishTextNormalizer
from datetime import datetime
from transformers import AutoProcessor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from suta import *
from data import *
import logging

logging.getLogger("accelerate.utils.modeling").setLevel(logging.ERROR)

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DECODER_STEP = 512

# Configure logging
def configure_logging(exp_name):
    # Create experiment directory if not exists
    os.makedirs(exp_name, exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'figs'), exist_ok=True)
    
    # Main logger configuration
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    
    # Results logger configuration
    result_logger = logging.getLogger('results')
    result_logger.setLevel(logging.INFO)
    result_logger.propagate = False  # Prevent propagation to root logger

    # File handlers
    log_file = os.path.join(exp_name, 'log.txt')
    result_file = os.path.join(exp_name, 'result.txt')

    # Formatters
    main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    result_formatter = logging.Formatter('%(message)s')

    # Handlers
    main_handler = logging.FileHandler(log_file)
    main_handler.setFormatter(main_formatter)
    
    result_handler = logging.FileHandler(result_file)
    result_handler.setFormatter(result_formatter)

    # Add handlers
    logger.addHandler(main_handler)
    result_logger.addHandler(result_handler)

    return logger, result_logger

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_experiment(args, logger):
    config_str = OmegaConf.to_yaml(args)
    start_time = datetime.now()
    logger.info(f'Experiment started at: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('Configuration:\n' + config_str)
    return start_time

def log_model_info(model, args, logger):
    params, names = hf_collect_params(model)
    logger.info('Model parameters:\n%s', '\n'.join(names))
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_ratio = trainable_params / total_params
    logger.info(f'Trainable parameter ratio: {train_ratio:.4f}')

def plot_losses(step_loss, p_loss_list, count, args):
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss', color='tab:red')
    ax.plot(step_loss, color='tab:red', marker='o')
    
    if 'p_loss' in args.objective_f:
        ax2 = ax.twinx()
        ax2.set_ylabel('P Loss', color='tab:blue')
        ax2.plot(p_loss_list, color='tab:blue', marker='o')
    
    plt.title(f'Loss Trajectory - Sample {count}')
    plt.savefig(f'{args.exp_name}/figs/suta_{count}.png')
    plt.close()

def find_topk_norm_layers(diff_dict, k=3):
    norms = {key: np.linalg.norm(vec) for key, vec in diff_dict.items()}
    sorted_layers = sorted(norms.items(), key=lambda x: x[1], reverse=True)
    
    topk_layers = sorted_layers[:k]
    topk_layers = {name: norm for name, norm in topk_layers}
    return topk_layers  # 回傳 [(layer_name, norm), ...]

def process_batch(batch, model, processor, normalizer, args, optimizer, scheduler, count, result_logger):
    c_loss_list, p_loss_list, step_loss = [], [], []
    lens, wavs, texts, files = batch
    
    result_logger.info(f'idx:{count}')
    label = normalizer(texts[0])
    result_logger.info(f'label:{label}')

    inputs = processor(wavs[0], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(model.device)

    # Original transcription
    teacher_token_list = model.decode(
        input_features, 
        forced_decoder_ids=processor.get_decoder_prompt_ids(
            language=args.asr_lang, 
            task=args.task
        )
    )
    transcription = processor.batch_decode(teacher_token_list, skip_special_tokens=True)[0]
    transcription = normalizer(transcription)
    error_metric = wer(label, transcription) if args.task == "transcribe" else sentence_bleu([label], transcription)
    result_logger.info(f'ori({error_metric:.5f}):{transcription}')

    # copy original parameters
    pre_adapt_state_dict = copy.deepcopy(model.state_dict())


    if args.tta:
        choose_layers_step = args.steps // 3
        for step in range(choose_layers_step):
            outputs, loss, e_loss, p_loss = model.AED_suta(
                input_features, args, optimizer,
                teacher_token_list=teacher_token_list,
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language=args.asr_lang,
                    task=args.task
                ),
                generate_text=False
            )
        post_adapt_state_dict = copy.deepcopy(model.state_dict())
        layer_norm_diff = {}
        relative_changes = {}
        for name in pre_adapt_state_dict:
            if 'layer_norm' in name:
                diff = post_adapt_state_dict[name] - pre_adapt_state_dict[name]
                layer_norm_diff[name] = diff.cpu().numpy().flatten()
                original = pre_adapt_state_dict[name].cpu().numpy().flatten()
                original_norm = np.linalg.norm(original)
                diff_norm = np.linalg.norm(layer_norm_diff[name])
                # Calculate relative change (avoid division by zero)
                if original_norm > 0:
                    relative_changes[name] = diff_norm / original_norm
                else:
                    relative_changes[name] = diff_norm  # Fallback to absolute change
        topk_layers = find_topk_norm_layers(relative_changes, k=3)
        # log topk_layers
        result_logger.info(f'Topk layers: {topk_layers}')
        # collect fine-tuned parameters with topk layers
        for param in model.parameters():
            param.requires_grad = False
        params, names = [], []
        for name, param in model.named_parameters():
            if name in topk_layers.keys():
                param.requires_grad = True
                params.append(param)
                names.append(name)
                result_logger.info(f'Choose layer {name}')
        optimizer, scheduler = setup_optimizer(
            args,
            params,
            args.opt,
            args.lr,
            weight_decay=1e-5,
            scheduler=args.scheduler
        )
            
        for step in range(choose_layers_step, args.steps):
            outputs, loss, e_loss, p_loss = model.AED_suta(
                input_features, args, optimizer,
                teacher_token_list=teacher_token_list,
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language=args.asr_lang,
                    task=args.task
                ),
                generate_text=(step % 3 == 0 or step == args.steps-1)
            )
            if step % 3 == 0 or step == args.steps-1:
                transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                transcription = normalizer(transcription)
                adapt_error = wer(label, transcription) if args.task == "transcribe" else sentence_bleu([label], transcription)
                result_logger.info(f'step{step}({adapt_error:.5f}): {transcription}')
            
            step_loss.append(loss.item())
            if 'p_loss' in args.objective_f:
                p_loss_list.append(p_loss.item())

        plot_losses(step_loss, p_loss_list, count, args)
        result_logger.info("=" * 40)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args):
    # Initialize logging
    args.exp_name = args.exp_name if args.exp_name else f'ex_data/{args.asr.split("/")[-1]}_{args.task}_{args.lang}'
    logger, result_logger = configure_logging(args.exp_name)
    set_seed(args.seed)
    start_time = setup_experiment(args, logger)
    
    
    # Load dataset and model
    normalizer = EnglishTextNormalizer()
    dataset = load_SUTAdataset(
        name=args.dataset_name,
        path=args.dataset_dir,
        batch_size=1,
        lang=args.lang,
        noise_dir=args.noise_dir,
        snr=args.snr
    )
    
    model = WhisperTTADecoder.from_pretrained(args.asr, device_map='auto')
    processor = AutoProcessor.from_pretrained(args.asr)
    
    # Log model information
    log_model_info(model, args, logger)
    
    # Process batches
    for count, batch in tqdm(enumerate(dataset), total=len(dataset)):
        if args.num_data and count > args.num_data:
            break
        model = WhisperTTADecoder.from_pretrained(args.asr, device_map='auto')
        # Initialize optimizer
        optimizer, scheduler = setup_optimizer(
            args,
            hf_collect_params(model, args.encoderLN, args.decoderLN)[0],
            args.opt,
            args.lr_scale * args.lr,
            weight_decay=1e-5,
            scheduler=args.scheduler
        )
        process_batch(
            batch, model, processor, normalizer,
            args, optimizer, scheduler, count, result_logger
        )

    # Add transcription processing after all batches
    try:
        wer_processor = transcriptionProcessor(task=args.task)
        wer_processor.process_file(f'{args.exp_name}/result.txt')
        wer_list = wer_processor.step_mean_wer()
        
        logger.info("Final Results:")
        for log_wer in wer_list:
            logger.info(log_wer)
            
    except Exception as e:
        logger.error("Failed to process results log: %s", str(e))

    # Finalize experiment
    end_time = datetime.now()
    logger.info(f'Experiment completed at: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Running time: {end_time - start_time}')
    
    # Close file handlers
    for handler in logger.handlers + result_logger.handlers:
        handler.close()
        logger.removeHandler(handler)
        result_logger.removeHandler(handler)

if __name__ == '__main__':
    main()