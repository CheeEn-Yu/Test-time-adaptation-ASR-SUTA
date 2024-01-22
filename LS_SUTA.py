import os
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
from main import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
def collect_params(model):
    # collect trainable params
    params = []
    names = []

    for name, param in model.named_parameters():
        param.requires_grad = False

    for nm, m in model.named_modules():
        trainable = ['weight', 'bias']
        # train_LN
        if isinstance(m, nn.LayerNorm) and str(nm).split('.')[0] == 'encoder':
            for np, p in m.named_parameters():
                if np in trainable:  
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")
        # train_feature
        if len(str(nm).split('.')) > 1:
            if str(nm).split('.')[0] == 'encoder' and (str(nm).split('.')[1] == 'conv1' or str(nm).split('.')[1] == 'conv2'):
                for np, p in m.named_parameters():
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names

def forward_and_adapt(x, model, optimizer, em_coef=1.0, reweight=False, temp=1., not_blank=True, scheduler=None, 
                        div_coef=0, repeat_inference=True, skip_short_thd=None):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.

    the index of <pad> in vocab is 0
    """
    # forward
    outputs = model.decode(x, options)
    logits = torch.stack(outputs[1], dim=0)
    logits=logits.permute(1,0,2) # torch.Size([1, 5, 51864])
    # adapt
    loss = 0

    if em_coef > 0: 
        e_loss = softmax_entropy(logits / temp).mean(0).mean() 
        
        loss += e_loss * em_coef

    if 1 - em_coef > 0: 
        c_loss = mcc_loss(logits / temp, reweight)
        loss += c_loss * (1 - em_coef)

    if div_coef > 0: 
        d_loss = div_loss(logits, not_blank) 
        loss += d_loss * div_coef 

    loss.backward()
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()
    model.zero_grad()

    # inference again
    if repeat_inference:
        with torch.no_grad():
            outputs = model.decode(x, options)
    return outputs


if __name__ == '__main__':
    # load datasets
    # dataset = LibriSpeech("test-clean")
    from data import load_dataset
    dataset = load_dataset(split=['test-other'], name='librispeech', path='./LibriSpeech', batch_size=1, extra_noise=0.)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    # load models
    model = whisper.load_model("tiny.en")
    params, names = collect_params(model)
    model = model.to(DEVICE)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    optimizer, scheduler = setup_optimizer(params, 'AdamW', lr=3e-4, scheduler=None)

    transcriptions = []
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
            adapt_output = forward_and_adapt(mel, model, optimizer)
        transcriptions.append(adapt_output[0][0].text)
        ori_transcriptions.append(texts[0])
        del outputs, adapt_output
        torch.cuda.empty_cache()
    
    data = pd.DataFrame(dict(hypothesis=transcriptions, reference=ori_transcriptions))
    
    import jiwer
    from whisper.normalizers import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer()

    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    data.to_csv('step5.csv')
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    print(f"WER: {wer * 100:.2f} %")