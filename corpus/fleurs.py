from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset
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

class fleursDataset(Dataset):
    def __init__(self, split, bucket_size, path, lang, noise_dir=None, snr=0., task='translation'):
        self.SNR = snr
        self.task = task
        if not noise_dir:
            self.noisefilenames = None
        elif noise_dir == 'Gaussian':
            self.noisefilenames = 'Gaussian'
        else:
            self.noisefilenames = glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
            self.noisefilenames.append("Gaussian")
        self.bucket_size = bucket_size
        if lang == 'en':
            ds = load_dataset("google/fleurs", "en_us", split="test")
        else:
            ds = load_from_disk(f"{path}/{lang}")
        self.ds = ds

    def __getitem__(self, index):
        try:
            label = self.ds[index]['translation'] if self.task == 'translation' else self.ds[index]['transcription']
        except:
            label = self.ds[index]['transcription']

        if self.noisefilenames is None:
            return len(self.ds[index]['audio']['array'])/16000, self.ds[index]['audio']['array'], label, self.ds[index]['id']
        elif self.noisefilenames == 'Gaussian':
            noise = np.random.randn(*self.ds[index]['audio']['array'].shape)
            
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
        return len(self.ds[index]['audio']['array'])/16000, noisy_snr, label, self.ds[index]['id']

    def __len__(self):
        return len(self.ds)