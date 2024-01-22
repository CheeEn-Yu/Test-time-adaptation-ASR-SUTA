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

if __name__ == '__main__':
    # load datasets
    # dataset = LibriSpeech("test-clean")
    from data import load_dataset
    dataset = load_dataset(split=['test-other'], name='librispeech', path='../LibriSpeech', batch_size=1, extra_noise=0.01)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    # load models
    model = whisper.load_model("tiny.en")
    model = model.to(DEVICE)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)

    transcriptions = []
    ori_transcriptions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataset):
            lens, wavs, texts, files = batch
            wavs = pad_or_trim(wavs[0])
            mel = log_mel_spectrogram(wavs)
            mel = mel.unsqueeze(-1)
            mel = mel.permute(2,0,1).to(DEVICE)
            outputs = model.decode(mel, options)
            transcriptions.append(outputs[0][0].text)
            ori_transcriptions.append(texts[0])
            del outputs
            torch.cuda.empty_cache()
    
    data = pd.DataFrame(dict(hypothesis=transcriptions, reference=ori_transcriptions))
    
    import jiwer
    from whisper.normalizers import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer()

    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]

    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    print(f"WER: {wer * 100:.2f} %")