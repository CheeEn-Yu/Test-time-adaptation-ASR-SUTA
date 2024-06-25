import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import whisper
from whisper.decoding import DecodingTask
from whisper.tokenizer import get_tokenizer
from whisper.audio import (
    log_mel_spectrogram,
    pad_or_trim,
    load_audio,
)

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import jiwer
from jiwer import wer

from data import *
from suta import *
from omegaconf import OmegaConf
args = OmegaConf.load("config.yaml")

dataset = load_dataset(name='noisy', path='./noisy_LibriSpeech', batch_size=1)

from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()
options = whisper.DecodingOptions(language="en", without_timestamps=True)
tokenizer = get_tokenizer(True)
exp_name = 'ex_data/suta_weighted_low'
with open(f'{exp_name}/{args.asr}_noise.txt', 'a') as f:

    for count, batch in tqdm(enumerate(dataset)):
        if count > 50:
            break
        # load model
        model = whisper.load_model(args.asr)
        model.eval()
        task = DecodingTask(model, options)

        # set training param
        params, names = whisper_collect_params(model, args.encoderLN, args.decoderLN, train_feature=args.train_feature)
        if count == 0:
            print(f'training parameter: {names}')
        optimizer, scheduler = setup_optimizer(args, params, args.opt, args.lr, weight_decay=1e-4, scheduler=args.scheduler)

        # unzip batch
        lens, wavs, texts, files = batch
        f.write(f'idx:{count}'+'\n')
        f.write('label:'+normalizer(texts[0])+'\n')

        # preprocess data
        if args.asr == 'large' or args.asr == 'large_v2' or args.asr == 'large_v3': # the code is for batch size = 1
            mel = log_mel_spectrogram(pad_or_trim(wavs[0]), n_mels=128).unsqueeze(0).to(DEVICE)
        else:
            mel = log_mel_spectrogram(pad_or_trim(wavs[0])).unsqueeze(0).to(DEVICE)

        step_loss,wers,word_changes = [],[],[0]
        # original whisper output
        with torch.no_grad():
            result = model.decode(mel, options)[0]
            teacher_tokens = result.tokens
            ori_text = normalizer(result.text)
            ori_wer = wer(normalizer(texts[0]), ori_text)
            wers.append(ori_wer)
        f.write(f'ori({ori_wer}):{ori_text}\n')
        del result
        torch.cuda.empty_cache()
        
        options = whisper.DecodingOptions(language="en", without_timestamps=True)
        task = DecodingTask(model, options)
        if options.beam_size is not None:
            n_batch = options.beam_size
        else:
            n_batch = 1

        # TTA
        for step in range(args.steps):
            model.zero_grad()
            audio_features = task._get_audio_features(mel)
            tokens = torch.tensor([task.initial_tokens]).repeat(1, 1).to(device=audio_features.device)
            tokens = tokens.repeat_interleave(task.n_group, dim=0).to(audio_features.device)
            sum_logprobs = torch.zeros(n_batch, device=audio_features.device)

            loss = 0
            entropy_list = None
            negative_loss = 0
            try:
                for i in range(task.sample_len):
                    logits = task.inference.logits(tokens, audio_features) # (1,2,51864)
                    
                    logits = logits[:, -1]
                    for logit_filter in task.logit_filters:
                        logit_filter.apply(logits, tokens)
                    tokens, completed = task.decoder.update(tokens, logits, sum_logprobs)
                    if completed or tokens.shape[-1] > task.n_ctx:
                        break
                    
                    logits = torch.topk(logits, k=30).values

                    e_loss = softmax_entropy(logits.unsqueeze(0) / args.temp).mean(0).mean()
                    c_loss = mcc_loss(logits.unsqueeze(0) / args.temp, class_num=args.topk)
                    word_loss = e_loss * args.em_coef + c_loss * (1 - args.em_coef)
                    if entropy_list is None:
                        entropy_list = word_loss.unsqueeze(0)
                    else:
                        entropy_list = torch.cat((word_loss.unsqueeze(0), entropy_list), dim=-1)
            finally:
                pass


            # weighted entropy loss (focus on high entropy token)
            weight = 1/(1+100*torch.exp(-entropy_list))
            loss = (weight*entropy_list).sum()/ weight.sum()
            # fix token under mean
            if step == 0:
                avg = entropy_list.mean()
                adapted_idx = torch.nonzero(entropy_list > avg).squeeze().cpu()

            # weighted entropy loss (focus on low entropy token)
            weight = 1/(1-100*torch.exp(-entropy_list))
            loss = (weight*entropy_list).sum()/ weight.sum()
            # fix token under mean
            if step == 0:
                avg = entropy_list.mean()
                adapted_idx = torch.nonzero(entropy_list > avg).squeeze().cpu()

            # high mean
            # mean = entropy_list.mean()
            # entropy_list = entropy_list[entropy_list > mean]
            # loss = entropy_list.mean()

            # lpf_entropy_list = torch.fft.fft(entropy_list)
            # n = len(lpf_entropy_list)
            # center = n // 2
            # low_pass_filter = torch.ones_like(lpf_entropy_list)
            # low_pass_filter[center:] = 0
            # lpf_entropy_list = lpf_entropy_list *low_pass_filter
            # lpf_entropy_list = torch.fft.ifft(lpf_entropy_list).real
            # mse_loss = torch.nn.MSELoss()
            # loss = mse_loss(entropy_list, lpf_entropy_list)

            if step==0:
                step_loss.append(loss)
            step_loss.append(loss)
            loss.backward()
            optimizer.step()
            task.inference.cleanup_caching()
            if args.scheduler is not None:
                scheduler.step()

            if step == 0:
                fig0, ax0 = plt.subplots(1,2,figsize=(15,6))
                color = 'tab:red'
                ax0[0].set_xlabel('token')
                ax0[0].set_ylabel('entropy', color=color)
                ax0[0].plot([frame.cpu().detach() for frame in entropy_list], color=color)
                ax0[0].tick_params(axis='y', labelcolor=color)
            if step == args.steps - 1:
                color = 'tab:blue'
                ax0[0].set_xlabel('token')
                ax0[0].set_ylabel('entropy', color=color)
                ax0[0].plot([frame.cpu().detach() for frame in entropy_list], color=color, alpha=0.5, linewidth=2)
                ax0[0].tick_params(axis='y', labelcolor=color)
                plt.title(f'entropy of idx:{count}')


            # output after adaptation
            check_list = [0,2,4,9,12,15,19]
            if step in check_list:
                options = whisper.DecodingOptions(language="en", without_timestamps=True)
                task = DecodingTask(model, options)
                with torch.no_grad():
                    adapted_tokens = model.decode(mel, options)[0].tokens
                for i in adapted_idx:
                    if i < len(teacher_tokens) and i < len(adapted_tokens) and teacher_tokens[i] != adapted_tokens[i]:
                        teacher_tokens[i] = adapted_tokens[i]
                # decode token
                after_text = tokenizer.decode(teacher_tokens).strip()
                normalized_label = normalizer(texts[0])
                after_text = normalizer(after_text)
                after_wer = wer(normalized_label, after_text)
                result = jiwer.compute_measures(ori_text, after_text)
                word_change = result['substitutions']+result['insertions']+result['deletions']
                f.write(f'step{step}({after_wer}): {after_text}\n')
                word_changes.append(word_change)
                wers.append(after_wer)
        del logits
        torch.cuda.empty_cache()

        # plot loss curve and wer
        color = 'tab:red'
        ax0[1].set_xlabel('step')
        ax0[1].set_ylabel('loss', color=color)
        ax0[1].plot([loss.cpu().detach() for loss in step_loss], color=color)
        ax0[1].tick_params(axis='y', labelcolor=color)

        # plot wers
        ax2 = ax0[1].twinx()  # 共享 x 軸
        color = 'tab:blue'
        ax2.set_xlabel('step')
        ax2.set_ylabel('wer', color=color)
        if args.steps == 20:
            check_list = [0,1,3,5,10,13,16,20]
        else:
            check_list = [0,1,3,5,10]
        ax2.plot(check_list, wers, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # plot number of word change
        ax3 = ax0[1].twinx()  # 共享 x 軸
        color = 'tab:green'
        ax3.set_ylabel('num of word change', color=color)
        if args.steps == 20:
            check_list = [0,1,3,5,10,13,16,20]
        else:
            check_list = [0,1,3,5,10]
        ax3.plot(check_list, word_changes, color=color)
        ax3.tick_params(axis='y', labelcolor=color)

        plt.title(f'idx:{count}')
        plt.savefig(f'{exp_name}/suta_{count}.png')
        plt.close()

        f.write("=======================================\n")
        