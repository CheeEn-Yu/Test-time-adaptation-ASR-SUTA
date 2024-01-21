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

model = whisper.load_model("base.en")
# collect trainable params
params = []
names = []
trainable = ['weight', 'bias']

for name, param in model.named_parameters():
    param.requires_grad = False

for nm, m in model.named_modules():
    # print(str(nm).split('.'))
    # train_LN
    if isinstance(m, nn.LayerNorm):
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
print(f'trainable layer: {names}')

# check trainable parameter
# for name, param in model.named_parameters():
#     print("name: ", name)
#     print("requires_grad: ", param.requires_grad)

# load audio
model = model.to(DEVICE)
options = whisper.DecodingOptions(language="en", without_timestamps=True)
audio = load_audio(file='./p232_022.wav')
audio = pad_or_trim(audio)
mel = log_mel_spectrogram(audio)
mel = mel.unsqueeze(-1)
mel = mel.permute(2,0,1)

# forward
mel = mel.to(DEVICE)
outputs = model.decode(mel, options)
print(f'before TTA: {outputs}')

steps = 10
optimizer, scheduler = setup_optimizer(params, 'AdamW', lr=0.1, scheduler=None)
model.train()
for i in range(steps):
    outputs = model.decode(mel, options)
    result_tensor = torch.stack(outputs[1], dim=0)
    result_tensor=result_tensor.permute(1,0,2) # torch.Size([1, 5, 51864])

    e_loss = softmax_entropy(result_tensor).mean(0).mean()
    # c_loss = mcc_loss(result_tensor, reweight=False)
    loss = 0
    loss += e_loss * 1000
    loss.requires_grad = True
    loss.backward()
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()
    model.zero_grad()
    print(loss)
    
    # with torch.no_grad():
        # outputs = model.decode(mel, options)
        # print(outputs[1])
