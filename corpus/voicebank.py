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
    with open(file, 'r') as fp:
        for line in fp:
            return line.rstrip()

class VoicebankDataset(Dataset):
    def __init__(self, split, bucket_size, path, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        # List all wave files
        file_list = list(Path(path).rglob("*.wav"))
        texts = []
        for file in file_list:
            texts.append(read_text(file.parent/ 'testset_txt' / (file.stem + '.txt')))
        self.file_list, self.text = zip(*[(f_name, txt)
                                    for f_name, txt in sorted(zip(file_list, texts), reverse=not ascending, key=lambda x:len(x[1]))])

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