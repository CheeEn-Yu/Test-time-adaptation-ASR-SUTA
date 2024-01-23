import os
import argparse
import numpy as np
import pandas as pd

import torch
import torchaudio
import torch.nn.functional as F
from torch import nn
import whisper
from whisper.audio import (
    log_mel_spectrogram,
    pad_or_trim,
)
from tqdm import tqdm

from suta import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('--asr', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--div_coef', type=float, default=0.)
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--dataset_name', type=str, default='librispeech')
    parser.add_argument('--dataset_dir', type=str, default='/home/daniel094144/data/LibriSpeech')
    parser.add_argument('--split', default=['test-other'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--em_coef', type=float, default=1.)
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--bias_only', action='store_true')
    parser.add_argument('--train_feature', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temp', type=float, default=2.5)
    parser.add_argument('--non_blank', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./exps')
    parser.add_argument('--extra_noise', type=float, default=0.)
    parser.add_argument('--scheduler', default=None)
    parser.add_argument('--is_whisper', type=bool, default=False)

    args = parser.parse_args()
    asr = args.asr
    steps = args.steps
    episodic = args.episodic
    opt = args.opt
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    split = args.split
    lr = args.lr
    em_coef = args.em_coef
    reweight = args.reweight
    batch_size = args.batch_size
    temp =  args.temp
    non_blank = args.non_blank
    log_dir = args.log_dir
    extra_noise = args.extra_noise
    scheduler = args.scheduler
    div_coef = args.div_coef
    bias_only = args.bias_only
    train_feature = args.train_feature
    train_all = args.train_all
    is_whisper = args.is_whisper
    skip_short_thd = None
    train_LN = True

    # load datasets
    # dataset = LibriSpeech("test-clean")
    from data import load_dataset
    dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise)
    # load models
    model = whisper.load_model("tiny.en")
    params, names = whisper_collect_params(model)
    model = model.to(DEVICE)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    optimizer, scheduler = setup_optimizer(params, opt, lr, scheduler=scheduler)

    transcriptions_1 = []
    transcriptions_3 = []
    transcriptions_5 = []
    transcriptions_10 = []
    ori_transcriptions = []
    model_state, optimizer_state, scheduler_state = copy_model_and_optimizer(model, optimizer, scheduler)
    for batch in tqdm(dataset):
        lens, wavs, texts, files = batch
        wavs = pad_or_trim(wavs[0])
        mel = log_mel_spectrogram(wavs)
        mel = mel.unsqueeze(-1)
        mel = mel.permute(2,0,1).to(DEVICE)
        outputs = model.decode(mel, options)
        model, optimizer, scheduler = load_model_and_optimizer(model, optimizer, scheduler, model_state, optimizer_state, scheduler_state)
        for i in range(10):
            adapt_output = forward_and_adapt(mel, model, optimizer, em_coef, reweight, temp, non_blank, scheduler, div_coef,is_whisper=is_whisper, options=options)
            if i == 0:
                transcriptions_1.append(adapt_output[0][0].text)
            if i == 2:
                transcriptions_3.append(adapt_output[0][0].text)
            if i == 4:
                transcriptions_5.append(adapt_output[0][0].text)
            

        transcriptions_10.append(adapt_output[0][0].text)
        ori_transcriptions.append(texts[0])
        del outputs, adapt_output
        torch.cuda.empty_cache()
    
    data = pd.DataFrame(dict(step1=transcriptions_1,step3=transcriptions_3,step5=transcriptions_5,step10=transcriptions_10, reference=ori_transcriptions))
    
    import jiwer
    from whisper.normalizers import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer()

    data["step1_clean"] = [normalizer(text) for text in data["step1"]]
    data["step3_clean"] = [normalizer(text) for text in data["step3"]]
    data["step5_clean"] = [normalizer(text) for text in data["step5"]]
    data["step10_clean"] = [normalizer(text) for text in data["step10"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]

    exp_name = dataset_name+'_'+str(temp)+'_noise_'+str(extra_noise)
    data.to_csv(f'{exp_name}.csv')
    wer_list = []
    wer_list.append(jiwer.wer(list(data["reference_clean"]), list(data["step1_clean"])))
    wer_list.append(jiwer.wer(list(data["reference_clean"]), list(data["step3_clean"])))
    wer_list.append(jiwer.wer(list(data["reference_clean"]), list(data["step5_clean"])))
    wer_list.append(jiwer.wer(list(data["reference_clean"]), list(data["step10_clean"])))
    with open(f"wer_{exp_name}.txt", 'w') as f:
        for i in wer_list:
            f.write(f'WER: {i}'+'\n')
        
    