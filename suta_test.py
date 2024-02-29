import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import whisper
from dataclasses import dataclass, field, replace
from whisper.decoding import DecodingTask
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

from jiwer import wer

from data import *
from suta import *
from omegaconf import OmegaConf
args = OmegaConf.load("config.yaml")

dataset = load_dataset(name='noisy', path='./noisy_LibriSpeech', batch_size=1)
# dataset = load_dataset(['test-other'], 'librispeech', 'LibriSpeech', 1, extra_noise=args.extra_noise)

teacher_tokens = []
from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()
options = whisper.DecodingOptions(language="en", without_timestamps=True)
exp_name = 'ex_data/suta_ex_sure'
with open(f'{exp_name}/base_no_noise.txt', 'a') as f:

    for count, batch in tqdm(enumerate(dataset)):
        if count > 100:
            break
        # load model
        model = whisper.load_model(args.asr)
        model.eval()
        task = DecodingTask(model, options)

        # set training param
        params, names = whisper_collect_params(model, args.encoderLN, args.decoderLN, train_feature=args.train_feature)
        optimizer, scheduler = setup_optimizer(params, args.opt, 2e-5, scheduler=None)

        # unzip batch
        lens, wavs, texts, files = batch
        f.write(f'idx:{count}'+'\n')
        f.write('label:'+normalizer(texts[0])+'\n')

        # preprocess data
        if args.asr == 'large' or args.asr == 'large_v2' or args.asr == 'large_v3': # the code is for batch size = 1
            mel = log_mel_spectrogram(pad_or_trim(wavs[0]), n_mels=128).unsqueeze(0).to(DEVICE)
        else:
            mel = log_mel_spectrogram(pad_or_trim(wavs[0])).unsqueeze(0).to(DEVICE)

        step_loss = []
        wers = []
        # original whisper output
        with torch.no_grad():
            result = model.decode(mel, options)[0]
            teacher_tokens = result.tokens
            text = result.text
            ori_wer = wer(normalizer(texts[0]), normalizer(text))
            wers.append(ori_wer)
        f.write(f'ori({ori_wer}):{normalizer(text)}\n')
        del result
        torch.cuda.empty_cache()
        
        # teacher forcing to get logit
        options = whisper.DecodingOptions(language="en", without_timestamps=True)
        task = DecodingTask(model, options)
        if options.beam_size is not None:
            n_batch = options.beam_size
        else:
            n_batch = 1

        # SGEM or SUTA
        for step in range(args.steps):
            model.zero_grad()
            audio_features = task._get_audio_features(mel)
            tokens = torch.tensor([task.initial_tokens]).repeat(1, 1).to(device=audio_features.device)
            tokens = tokens.repeat_interleave(task.n_group, dim=0).to(audio_features.device)
            sum_logprobs = torch.zeros(n_batch, device=audio_features.device)

            loss = 0
            entropy_list = None
            negative_loss = 0
            for i in range(len(teacher_tokens)):
                added_token = torch.Tensor([[teacher_tokens[i]]]).long().expand(tokens.shape[0], 1).to(DEVICE)
                tokens = torch.cat((tokens, added_token), dim=1)
                logits = task.inference.logits(tokens, audio_features) # (1,2,51864)
                
                logits = logits[:, -1]
                for logit_filter in task.logit_filters:
                    logit_filter.apply(logits, tokens)
                
                logits = torch.topk(logits, k=30).values

                # SUTA
                e_loss = softmax_entropy(logits.unsqueeze(0) / args.temp).mean(0).mean()
                c_loss = mcc_loss(logits.unsqueeze(0) / args.temp, class_num=args.topk)
                loss = e_loss * args.em_coef + c_loss * (1 - args.em_coef)
                if entropy_list is None:
                    entropy_list = loss.unsqueeze(0)
                else:
                    entropy_list = torch.cat((loss.unsqueeze(0), entropy_list), dim=-1)

                # # GEM
                # entropy = torch.log(torch.pow(logits.softmax(dim=-1), args.renyi_entropy_alpha).sum(dim=-1)) # entropy: B, L
                # entropy = entropy / (1 - args.renyi_entropy_alpha)
                # entropy = entropy.mean()
                #     entropy_list = entropy.unsqueeze(0)
                # else:
                #     entropy_list = torch.cat((entropy.unsqueeze(0), entropy_list), dim=-1)

                # # NS
                # negative_outputs = logits.clone()
                # negative_mask = torch.where(torch.softmax(negative_outputs, dim=-1) < args.ns_threshold * (10 / negative_outputs.shape[-1]), 1, 0)
                # negative_loss += torch.mean(-torch.log(1 - torch.sum(negative_mask * torch.softmax(negative_outputs / args.temp, dim=-1), dim=-1)))

            # e_loss = entropy_list.mean()
            # loss = args.ns_coef * negative_loss + e_loss
            if step == 0:
                fig0, ax0 = plt.subplots()
                color = 'tab:red'
                ax0.set_xlabel('token')
                ax0.set_ylabel('entropy', color=color)
                ax0.plot([frame.cpu().detach() for frame in entropy_list], color=color)
                ax0.tick_params(axis='y', labelcolor=color)
            if step == 19:
                ax3 = ax0.twinx()  # 共享 x 軸
                color = 'tab:blue'
                ax0.set_xlabel('token')
                ax0.set_ylabel('entropy', color=color)
                ax0.plot([frame.cpu().detach() for frame in entropy_list], color=color, alpha=0.5, linewidth=2)
                ax0.tick_params(axis='y', labelcolor=color)
                plt.title(f'entropy of idx:{count}')
                plt.savefig(f'{exp_name}/entropy_{count}.png')
            loss = entropy_list.mean()
            if step==0:
                step_loss.append(loss)
            step_loss.append(loss)
            loss.backward()
            optimizer.step()
            task.inference.cleanup_caching()

            # output after adaptation
            check_list = [0,2,4,9,12,15,19]
            if step in check_list:
                options = whisper.DecodingOptions(language="en", without_timestamps=True)
                task = DecodingTask(model, options)
                with torch.no_grad():
                    after_text = model.decode(mel, options)[0].text
                after_wer = wer(normalizer(texts[0]), normalizer(after_text))
                f.write(f'step{step}({after_wer}): {normalizer(after_text)}\n')
                wers.append(after_wer)
        del logits
        torch.cuda.empty_cache()

        # plot loss curve and wer
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('step')
        ax1.set_ylabel('loss', color=color)
        ax1.plot([loss.cpu().detach() for loss in step_loss], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # 在右側 y 軸上繪製 data2
        ax2 = ax1.twinx()  # 共享 x 軸
        color = 'tab:blue'
        ax2.set_ylabel('wer', color=color)
        ax2.plot(wers, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title(f'idx:{count}')
        plt.savefig(f'{exp_name}/suta_{count}.png')

        f.write("=======================================\n")
        