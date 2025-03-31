import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
import glob
import numpy as np
import os
import random
from tqdm import tqdm
from functools import partial
from corpus.audiolib import audioread, audiowrite, snr_mixer
import logging
import matplotlib.pyplot as plt
import hydra
from jiwer import wer
from omegaconf import OmegaConf
from whisper.normalizers import EnglishTextNormalizer
from datetime import datetime
from transformers import AutoProcessor
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from suta import *
from data import *

random.seed(42)


def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def cleanup_distributed():
    dist.destroy_process_group()

class Covost2Dataset(Dataset):
    def __init__(self, split, path, lang, noise_dir=None, snr=0.0):
        self.SNR = snr
        if noise_dir is None:
            self.noisefilenames = None
        elif noise_dir == 'Gaussian':
            self.noisefilenames = 'Gaussian'
        else:
            self.noisefilenames = glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
            self.noisefilenames.append("Gaussian")
        ds = load_dataset("covost2", f'{lang}_en', data_dir=path, split=split, trust_remote_code=True)
        ds = ds.remove_columns(['client_id', 'id'])
        self.ds = ds

    def __getitem__(self, index):
        audio_array = self.ds[index]['audio']['array']
        if self.noisefilenames is None:
            return len(audio_array) / 16000, audio_array, self.ds[index]['translation'], self.ds[index]['file']
        
        if self.noisefilenames == 'Gaussian':
            noise = np.random.randn(*audio_array.shape)
        else:
            noisefile = random.choice(self.noisefilenames)
            if noisefile == "Gaussian":
                noise = np.random.randn(*audio_array.shape)
            else:
                noise, fs = audioread(noisefile)
                while len(noise) < len(audio_array):
                    noise = np.append(noise, noise)
                noise = noise[:len(audio_array)]
        
        clean_snr, noise_snr, noisy_snr = snr_mixer(audio_array, noise, self.SNR)
        return len(audio_array) / 16000, noisy_snr, self.ds[index]['translation'], self.ds[index]['file']

    def __len__(self):
        return len(self.ds)


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

def plot_losses(step_loss, p_loss_list, count, args, rank=None, world_size=None):
    """Modified plot function for parallel processing"""
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss', color='tab:red')
    ax.plot(step_loss, color='tab:red', marker='o')
    
    if 'p_loss' in args.objective_f:
        ax2 = ax.twinx()
        ax2.set_ylabel('P Loss', color='tab:blue')
        ax2.plot(p_loss_list, color='tab:blue', marker='o')
    
    # Add rank info to filename
    idx = world_size*count+rank
    plt.title(f'Loss Trajectory - Sample {idx}')
    plt.savefig(f'{args.exp_name}/figs/suta_{idx}.png')
    plt.close()

def process_batch(batch, processor, normalizer, args, count, rank, world_size):
    model = WhisperTTADecoder.from_pretrained(args.asr, device_map=f'cuda:{rank}')
    optimizer, scheduler = setup_optimizer(
        args,
        hf_collect_params(model)[0],
        args.opt,
        args.lr,
        weight_decay=1e-5,
        scheduler=args.scheduler
    )
    c_loss_list, p_loss_list, step_loss = [], [], []
    lens, wavs, texts, files = batch
    
    label = normalizer(texts[0])
    result_str = f'idx:{world_size*count+rank}\nlabel:{label}\n'

    input_features = wavs.to(model.device)

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
    result_str += f'ori({error_metric:.5f}):{transcription}\n'

    if args.tta:
        for step in range(args.steps):
            if step % 3 == 0 or step == args.steps-1:
                outputs, loss, e_loss, p_loss = model.tf_suta(
                    input_features, args, optimizer,
                    teacher_token_list=teacher_token_list,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language=args.asr_lang,
                        task=args.task
                    ),
                    generate_text=True
                )
                transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                transcription = normalizer(transcription)
                adapt_error = wer(label, transcription) if args.task == "transcribe" else sentence_bleu([label], transcription)
                result_str += f'step{step}({adapt_error:.5f}): {transcription}\n'
            else:
                outputs, loss, e_loss, p_loss = model.AED_suta(
                    input_features, args, optimizer,
                    teacher_token_list=teacher_token_list,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language=args.asr_lang,
                        task=args.task
                    )
                )
            
            step_loss.append(loss.item())
            if "p_loss" in args.objective_f:
                p_loss_list.append(p_loss.item())


        plot_losses(step_loss, p_loss_list, count, args, rank, world_size)
        result_str += "="*40

    return result_str

def main_worker(rank, world_size, args):
    # Initialize logging
    args.exp_name = args.exp_name if args.exp_name else f'ex_data/{args.asr.split("/")[-1]}_{args.task}_{args.lang}'
    setup_distributed()
    if rank == 0:
        logger, result_logger = configure_logging(args.exp_name)
        start_time = setup_experiment(args, logger)
    
    processor = AutoProcessor.from_pretrained(args.asr)
    dataloader = load_SUTAdataset(
        name=args.dataset_name,
        path=args.dataset_dir,
        batch_size=1,
        lang=args.lang,
        noise_dir=args.noise_dir,
        snr=args.snr,
        task=args.task,
        is_multi_gpus=True,
        processor=processor
    )
    # dataset = Covost2Dataset(split='test', path=f'../TTA_LAS/covost2_{args.lang}', lang=args.lang, noise_dir="../res", snr=10.0)
    # my_collate_fn = partial(collate_fn, processor=processor)
    # sampler = DistributedSampler(dataset)
    # dataloader = DataLoader(
    #     dataset, 
    #     batch_size=1, 
    #     sampler=sampler, 
    #     num_workers=4, 
    #     pin_memory=True,
    #     collate_fn=my_collate_fn
    # )
    # Load dataset and model
    normalizer = EnglishTextNormalizer()

    # Process batches
    for count, batch in tqdm(enumerate(dataloader)):
        if args.num_data and count > args.num_data:
            break
        result_str = process_batch(batch, processor, normalizer, args, count, rank, world_size)
        result_list = [None] * world_size
        dist.all_gather_object(result_list, (count, result_str))
        if rank == 0:
            sorted_results = sorted(result_list, key=lambda x: x[0])
            with open(os.path.join(args.exp_name, 'result.txt'), 'a') as f:
                for _, res in sorted_results:
                    f.write(res + "\n")
        dist.barrier()

    if rank == 0:
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
    cleanup_distributed()
    
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args):
    set_seed(args.seed)
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    main_worker(rank, world_size, args)

if __name__ == "__main__":
    main()
