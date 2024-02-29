from tqdm import tqdm
from pathlib import Path
import os
from joblib import Parallel, delayed
from torch.utils.data import Dataset
import re

def read_text(file):
    '''Get transcription of target wave file, 
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread'''
    texts = {}
    with open(file, 'r') as fp:
        for line in fp:
            texts[line.split(' ')[0]] = ' '.join(line.rstrip().split(' ')[1:])
    return texts

class noisyLibriDataset(Dataset):
    def __init__(self, split, bucket_size, path, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        # List all wave files
        self.file_list = list(Path(path).rglob("*.flac"))
        texts = read_text(f'{path}/label.txt')
        self.text = [texts[sentence.name.split('.')[0]] for sentence in self.file_list]
        self.texts = []
        for text in self.file_list:
            self.texts.append(texts[text.name.split('.')[0]])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)