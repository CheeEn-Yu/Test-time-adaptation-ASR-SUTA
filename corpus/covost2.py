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

class covost2Dataset(Dataset):
    def __init__(self, split, bucket_size, path, lang, noise_dir=None, snr=0.):
        # Setup
        split = ['test']
        self.SNR = snr
        if noise_dir is None:
            self.noisefilenames = None
        elif noise_dir == 'Gaussian':
            self.noisefilenames = 'Gaussian'
        else:
            self.noisefilenames = glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
            self.noisefilenames.append("Gaussian")
        self.bucket_size = bucket_size
        ds = load_dataset("covost2", f'{lang}_en', data_dir=path, split='test', trust_remote_code=True)
        ds = ds.remove_columns(['client_id', 'id'])
        self.ds = ds

    def __getitem__(self, index):
        if self.noisefilenames is None:
            return len(self.ds[index]['audio']['array'])/16000, self.ds[index]['audio']['array'], self.ds[index]['translation'], self.ds[index]['file']
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
        return len(self.ds[index]['audio']['array'])/16000, noisy_snr, self.ds[index]['translation'], self.ds[index]['file']

    def __len__(self):
        return len(self.ds)