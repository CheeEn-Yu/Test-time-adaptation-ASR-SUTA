import os
import numpy as np
import pandas as pd

import torch
import torchaudio
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

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = pad_or_trim(audio.flatten()).to(self.device)
        mel = log_mel_spectrogram(audio)
        
        return (mel, text)
    
def collect_params(model):
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

    return params, names



if __name__ == '__main__':
    # load datasets
    dataset = LibriSpeech("test-clean")
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    # load models
    model = whisper.load_model("base.en")
    model = model.to(DEVICE)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)


    hypotheses = []
    references = []

    for mels, texts in tqdm(loader):
        outputs = model.decode(mels, options)
        optimizer, scheduler = setup_optimizer(params, 'AdamW', lr=3e-4, scheduler=None)
        result_tensor = torch.stack(outputs[1], dim=0)
        result_tensor=result_tensor.permute(1,0,2) # torch.Size([1, 5, 51864])






        hypotheses.extend([result.text for result in results[0]])
        references.extend(texts)

    