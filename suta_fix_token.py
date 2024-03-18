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
from whisper.tokenizer import get_tokenizer
from whisper.audio import (
    log_mel_spectrogram,
    pad_or_trim,
    load_audio,
)
import jiwer
from jiwer import wer

from data import *
from suta import *
from omegaconf import OmegaConf
args = OmegaConf.load("config.yaml")

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from data import load_dataset
dataset = load_dataset(name='noisy', path='./noisy_LibriSpeech', batch_size=1)

from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()
tokenizer = get_tokenizer(True)

original_wers, wers = [], []
for count, batch in tqdm(enumerate(dataset)):
    if count > 50:
        break
    # load model
    model = whisper.load_model(args.asr)
    model.eval()

    # process data
    lens, wavs, texts, files = batch
    if args.asr == 'large' or args.asr == 'large_v2' or args.asr == 'large_v3': # the code is for batch size = 1
        mel = log_mel_spectrogram(pad_or_trim(wavs[0]), n_mels=128).unsqueeze(0).to(DEVICE)
    else:
        mel = log_mel_spectrogram(pad_or_trim(wavs[0])).unsqueeze(0).to(DEVICE)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    task = DecodingTask(model, options)
    audio_features = task._get_audio_features(mel)
    tokens = torch.tensor([task.initial_tokens]).repeat(1, 1).to(device=audio_features.device)
    tokens = tokens.repeat_interleave(task.n_group, dim=0).to(audio_features.device)
    n_batch = tokens.shape[0]
    sum_logprobs = torch.zeros(n_batch, device=audio_features.device)

    # start decoding
    teacher_tokens = []
    sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
    no_speech_probs = [np.nan] * n_batch
    entropy_list = None
    with torch.no_grad():
        try:
            for i in range(task.sample_len):
                logits = task.inference.logits(tokens, audio_features) # (1,2,51864)
                if (
                    i == 0 and task.tokenizer.no_speech is not None
                ):  # save no_speech_probs
                    probs_at_sot = logits[:, task.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, task.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]
                topk_logits = torch.topk(logits, k=30).values

                e_loss = softmax_entropy(topk_logits.unsqueeze(0) / args.temp).mean(0).mean()
                c_loss = mcc_loss(topk_logits.unsqueeze(0) / args.temp, class_num=args.topk)
                word_loss = e_loss * args.em_coef + c_loss * (1 - args.em_coef)
                if entropy_list is None:
                    entropy_list = word_loss.unsqueeze(0)
                else:
                    entropy_list = torch.cat((word_loss.unsqueeze(0), entropy_list), dim=-1)

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in task.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = task.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > task.n_ctx:
                    break
        finally:
            task.inference.cleanup_caching()

    # SUTA

    avg = entropy_list.mean()
    teacher_tokens = tokens
    original_text = normalizer(tokenizer.decode(tokens[0].tolist()))
    normalized_label = normalizer(texts[0])
    ori_wer = wer(normalized_label, original_text)
    print(f'ori_wer:{ori_wer}')
    original_wers.append(ori_wer)

    tokens = torch.tensor([task.initial_tokens]).repeat(1, 1).to(device=audio_features.device)
    tokens = tokens.repeat_interleave(task.n_group, dim=0).to(audio_features.device)

    # start decoding
    n_batch = tokens.shape[0]
    sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
    no_speech_probs = [np.nan] * n_batch
    word_loss = 0
    # set training param
    try:
        for i in range(task.sample_len):
            if i<5 or (i < len(entropy_list) and entropy_list[i] < avg):
                continue
            # start adaptation
            model = whisper.load_model(args.asr)
            params, names = whisper_collect_params(model, args.encoderLN, args.decoderLN, train_feature=args.train_feature)
            if count == 0:
                print(f'training parameter: {names}')
            optimizer, scheduler = setup_optimizer(args, params, args.opt, args.lr, weight_decay=1e-4, scheduler=args.scheduler)
            for step in range(args.steps):
                optimizer.zero_grad()
                task = DecodingTask(model, options)
                tokens = teacher_tokens[0][:i].unsqueeze(0)
                audio_features = task._get_audio_features(mel)
                logits = task.inference.logits(tokens, audio_features) # (1,2,51864)

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]
                topk_logits = torch.topk(logits, k=args.topk).values

                e_loss = softmax_entropy(topk_logits.unsqueeze(0) / args.temp).mean(0).mean()
                c_loss = mcc_loss(topk_logits.unsqueeze(0) / args.temp, class_num=args.topk)
                word_loss = e_loss * args.em_coef + c_loss * (1 - args.em_coef)

                word_loss.backward()
                optimizer.step()
                scheduler.step()

            task = DecodingTask(model, options)
            tokens = teacher_tokens[0][:i].unsqueeze(0)
            audio_features = task._get_audio_features(mel)
            logits = task.inference.logits(tokens, audio_features) # (1,2,51864)
            logits = logits[:, -1]
            
            sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
            # apply the logit filters, e.g. for suppressing or applying penalty to
            for logit_filter in task.logit_filters:
                logit_filter.apply(logits, tokens)

            # expand the tokens tensor with the selected next tokens
            tokens, completed = task.decoder.update(tokens, logits, sum_logprobs)

            if completed or tokens.shape[-1] > task.n_ctx:
                break
    finally:
        task.inference.cleanup_caching()
    del logits
    torch.cuda.empty_cache()
    final_text = tokenizer.decode(tokens[0].tolist())
    after_wer = wer(normalized_label, normalizer(final_text))
    if after_wer > ori_wer:
        print(tokens)
        print(original_text)
        print(final_text)
    print(f'after_wer:{after_wer}')
    wers.append(after_wer)


print(wers)
print(np.array(wers).mean())