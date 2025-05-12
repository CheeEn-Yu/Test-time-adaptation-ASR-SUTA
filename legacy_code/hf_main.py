import os
import matplotlib.pyplot as plt
import torch
import hydra
# import evaluate
from jiwer import wer
from omegaconf import OmegaConf
from whisper.normalizers import EnglishTextNormalizer
from datetime import datetime
from transformers import AutoProcessor
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from suta import *
from utils.data import *


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args):
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # if args.task == "translation":
        # smoothing_function = SmoothingFunction().method1
        # bleu = evaluate.load('bleu')
    normalizer = EnglishTextNormalizer()
    
    dataset = load_SUTAdataset(name=args.dataset_name, path=args.dataset_dir, batch_size=1, lang=args.lang, noise_dir=args.noise_dir, snr=args.snr)
    os.makedirs(args.exp_name, exist_ok=True)
    os.makedirs(f'{args.exp_name}/figs', exist_ok=True)
    config_str = OmegaConf.to_yaml(args)
    with open(f'{args.exp_name}/log.txt', 'w') as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'start time: {current_time}\n')
        f.write(config_str)

    with open(f'{args.exp_name}/result.txt', 'a') as f:
        for count, batch in tqdm(enumerate(dataset)):

            if args.num_data:
                if count > args.num_data:
                    break

            c_loss_list, p_loss_list, step_loss, wers = [], [], [], []
            lens, wavs, texts, files = batch
            f.write(f'idx:{count}'+'\n')
            label = normalizer(texts[0])
            f.write('label:'+label+'\n')
            # load model
            if count == 0 or args.tta== True:
                processor = AutoProcessor.from_pretrained(args.asr)
                model = WhisperTTADecoder.from_pretrained(args.asr, device_map='auto')
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.asr_lang, task=args.task)

                params, names = hf_collect_params(model)
                optimizer, scheduler = setup_optimizer(args, params, args.opt, args.lr, weight_decay=1e-5, scheduler=args.scheduler)

            if count == 0:
                import pprint
                with open(f'{args.exp_name}/log.txt', 'a') as logfile:
                    pp = pprint.PrettyPrinter(stream=logfile)
                    pp.pprint(names)
                    total_params = sum(
                        param.numel() for param in model.parameters()
                    )
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    train_ratio = trainable_params / total_params
                    print(f'train_param ratio: {train_ratio}')
                    logfile.write(f'train_param ratio: {train_ratio}\n')
                
            inputs = processor(wavs[0],sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(model.device)

            # Original transcription
            teacher_token_list = model.decode(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(teacher_token_list, skip_special_tokens=True)[0]
            transcription = normalizer(transcription)
            ori_wer = wer(label, transcription) if args.task == "transcribe" else sentence_bleu([label], transcription)
            f.write(f'ori({ori_wer:.5f}):{transcription}\n')

            if args.tta:
                # Start TTA
                for step in range(args.steps):
                    if step % 3 == 0 or step == args.steps-1:
                        outputs, loss, e_loss, p_loss = model.AED_suta(input_features, args, optimizer, teacher_token_list=teacher_token_list, forced_decoder_ids=forced_decoder_ids, generate_text=True)
                        transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                        transcription = normalizer(transcription)
                        adapt_wer = wer(label, transcription) if args.task == "transcribe" else sentence_bleu([label], transcription)
                        f.write(f'step{step}({adapt_wer:.5f}): {transcription}\n')
                    else:
                        outputs, loss, e_loss, p_loss = model.AED_suta(input_features, args, optimizer, teacher_token_list=teacher_token_list, forced_decoder_ids=forced_decoder_ids)
                    try:
                        step_loss.append(loss.item())
                        p_loss_list.append(p_loss.item())
                    except:
                        step_loss.append(loss)
                        p_loss_list.append(p_loss)


                # 10 loss figure
                fig0, ax0 = plt.subplots(1,1)
                color = 'tab:red'
                ax0.set_xlabel('step')
                ax0.set_ylabel('loss', color=color)
                ax0.plot([loss for loss in step_loss], color=color, marker='o')
                ax0.tick_params(axis='y', labelcolor=color)

                ax2 = ax0.twinx()  # 共享 x 軸
                color = 'tab:blue'
                ax2.set_xlabel('step')
                ax2.set_ylabel('p_loss', color=color)
                ax2.plot([i for i in p_loss_list], color=color, marker='o')
                ax2.tick_params(axis='y', labelcolor=color)
                plt.title(f'idx:{count}')
                plt.savefig(f'{args.exp_name}/figs/suta_{count}.png')
                plt.close()

                f.write("=======================================\n")

    processor = transcriptionProcessor(task=args.task)
    processor.process_file(f'{args.exp_name}/result.txt')
    wer_list = processor.step_mean_wer()
    with open(f'{args.exp_name}/log.txt', 'a') as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'end time: {current_time}\n')
        for log_wer in wer_list:
            f.write(log_wer)
            f.write('\n')

if __name__ == '__main__':
    main()