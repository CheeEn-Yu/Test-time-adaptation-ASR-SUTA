from torch.utils.data import Dataset
from datasets import load_dataset
import glob
import numpy as np
import soundfile as sf
import os
import argparse
import configparser as CP
import random
import numpy as np
from tqdm import tqdm
from .audiolib import audioread, audiowrite, snr_mixer
random.seed(42)

class multiLingualLibriDataset(Dataset):
    def __init__(self, split, bucket_size, lang, noise_dir=None, snr=0.):
        # Setup
        split = ['test']
        self.SNR = snr
        self.noisefilenames = None if noise_dir is None else glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
        if self.noisefilenames is not None:
            self.noisefilenames.append("Gaussian")
        self.bucket_size = bucket_size
        ds = load_dataset("facebook/multilingual_librispeech", lang, split="test")
        ds = ds.remove_columns(['original_path', 'begin_time', 'end_time', 'speaker_id', 'chapter_id', 'id'])
        self.ds = ds

    def __getitem__(self, index):
        if self.noisefilenames is None:
            return self.ds[index]['audio_duration'], self.ds[index]['audio']['array'], self.ds[index]['transcript'], self.ds[index]['file']
        else:
            noisefile = random.choice(self.noisefilenames)
            if noisefile == "Gaussian":
                noise = np.random.randn(*self.ds[index]['audio']['array'].shape)
            else:
                noise, fs = audioread(noisefile)

                noiseconcat = noise
                while len(noiseconcat) <= len(self.ds[index]['audio']['array']):
                    noiseconcat = np.append(noiseconcat, noise)
                noise = noiseconcat
                if len(noise)>=len(self.ds[index]['audio']['array']):
                    noise = noise[0:len(self.ds[index]['audio']['array'])]

            clean_snr, noise_snr, noisy_snr = snr_mixer(self.ds[index]['audio']['array'], noise, self.SNR)
            return self.ds[index]['audio_duration'], noisy_snr, self.ds[index]['transcript'], self.ds[index]['file']

    def __len__(self):
        return len(self.ds)