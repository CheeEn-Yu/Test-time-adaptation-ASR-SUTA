import os
import matplotlib.pyplot as plt
import torch
import hydra
from jiwer import wer
from omegaconf import OmegaConf
from whisper.normalizers import EnglishTextNormalizer
from datetime import datetime

import whisper
from whisper.audio import (
    log_mel_spectrogram,
    pad_or_trim,
)
from tqdm import tqdm
from suta import *
from data import *
# from timeit import default_timer


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args):
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    normalizer = EnglishTextNormalizer()
    
    dataset = load_SUTAdataset(name=args.dataset_name, path=args.dataset_dir, batch_size=1, noise_dir=args.noise_dir, snr=args.snr)
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

            model = whisper.load_model(args.asr).to(DEVICE)
            params, names = whisper_collect_params(model, args.encoderLN, args.decoderLN, train_feature=args.train_feature)
            options = whisper.DecodingOptions(language=args.lang, without_timestamps=True)
            c_loss_list, p_loss_list, step_loss, wers = [], [], [], []
            lens, wavs, texts, files = batch
            f.write(f'idx:{count}'+'\n')
            f.write('label:'+normalizer(texts[0])+'\n')
            model.eval()
            if args.asr == 'large' or args.asr == 'large_v2' or args.asr == 'large_v3':
                mel = log_mel_spectrogram(pad_or_trim(wavs[0]), n_mels=128).unsqueeze(0)
            else:
                if args.dataset_name.lower() == "multilibri":
                    mel = log_mel_spectrogram(pad_or_trim(wavs[0].float())).unsqueeze(0)
                else:
                    mel = log_mel_spectrogram(pad_or_trim(wavs[0])).unsqueeze(0)
            mel = mel.to(DEVICE)
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
                
            optimizer, scheduler = setup_optimizer(args, params, args.opt, args.lr, weight_decay=1e-4, scheduler=args.scheduler)
            decode_obj = TTADecode(model, options)

            # get words before TTA
            with torch.no_grad():
                _, ori_text = decode_obj.run(mel, max_decoder_step=args.max_decoder_step)
                label = normalizer(texts[0]) if args.lang == "en" else texts[0]
                ori_text = normalizer(ori_text[0]) if args.lang == "en" else ori_text[0]
                ori_wer = wer(label, ori_text)
                wers.append(ori_wer)
                f.write(f'ori({ori_wer}):{ori_text}\n')

            # Start TTA
            for step in range(args.steps):
                if step % 3 == 0 or step == args.steps-1:
                    adapt_text, loss, c_loss, p_loss = decode_obj.adapt(mel, args, optimizer, scheduler, generate_text=True)
                    adapt_text = normalizer(adapt_text[0]) if args.lang == "en" else adapt_text[0]
                    adapt_wer = wer(label, adapt_text)
                    wers.append(adapt_wer)
                    f.write(f'step{step}({adapt_wer}): {adapt_text}\n')
                else:
                    adapt_text, loss, c_loss, p_loss = decode_obj.adapt(mel, args, optimizer, scheduler)
                step_loss.append(loss)
                c_loss_list.append(c_loss)
                p_loss_list.append(p_loss)

            # 10 loss figure
            if count < 5 or count >2930:
                fig0, ax0 = plt.subplots(1,1)
                color = 'tab:red'
                ax0.set_xlabel('step')
                ax0.set_ylabel('loss', color=color)
                ax0.plot([loss for loss in step_loss], color=color, marker='o')
                ax0.tick_params(axis='y', labelcolor=color)

                ax2 = ax0.twinx()  # 共享 x 軸
                color = 'tab:blue'
                ax2.set_xlabel('step')
                ax2.set_ylabel('c_loss', color=color)
                ax2.plot([i for i in c_loss_list], color=color, marker='o')
                ax2.tick_params(axis='y', labelcolor=color)
                plt.title(f'idx:{count}')
                plt.savefig(f'{args.exp_name}/figs/suta_{count}.png')
                plt.close()

            f.write("=======================================\n")

    processor = transcriptionProcessor()
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