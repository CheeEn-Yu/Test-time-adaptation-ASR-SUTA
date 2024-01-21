import torch
import torch.nn.functional as F
from torch import nn

import whisper
from whisper.audio import (
    log_mel_spectrogram,
    pad_or_trim,
    load_audio,
)

import jiwer
from tqdm import tqdm
from main import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def forward_and_adapt(x, model, optimizer, em_coef=0.9, reweight=False, temp=1., not_blank=True, scheduler=None, 
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


model = whisper.load_model("base.en")


# collect trainable params
params = []
names = []

for name, param in model.named_parameters():
    param.requires_grad = False

for nm, m in model.named_modules():
    # print(str(nm).split('.'))
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
# check trainable parameter
# for name, param in model.named_parameters():
#     print("name: ", name)
#     print("requires_grad: ", param.requires_grad)

# load audio
options = whisper.DecodingOptions(language="en", without_timestamps=True)
optimizer, scheduler = setup_optimizer(params, 'AdamW', lr=3e-4, scheduler=None)
audio = load_audio(file='./p232_022.wav')
audio = pad_or_trim(audio)
mel = log_mel_spectrogram(audio)
mel = mel.unsqueeze(-1)
mel = mel.permute(2,0,1)


model = model.to(DEVICE)
mel = mel.to(DEVICE)
test1 = forward_and_adapt(mel, model, optimizer)
print(test1)
