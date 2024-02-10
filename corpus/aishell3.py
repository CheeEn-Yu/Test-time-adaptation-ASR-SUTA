from tqdm import tqdm
from pathlib import Path
import os
from joblib import Parallel, delayed
from torch.utils.data import Dataset
import re


def read_text(file):
    src_file = file+'/content.txt'
    file_list = {}
    with open(src_file, 'r') as fp:
        for line in fp:
            label_string= re.findall(r'[\u4e00-\u9fa5]+', line.split('\t')[1])
            label_string = ''.join(label_string)
            file_list[line.split('\t')[0]] = label_string
    return file_list

class AIShellDataset(Dataset):
    def __init__(self, split, bucket_size, path, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        # List all wave files
        self.file_list = list(Path(path).rglob("*.wav"))

        texts = []
        text_dict = read_text(path)
        for file_path in tqdm(self.file_list, desc='Read text'):
            texts.append(text_dict[os.path.basename(file_path)])
        self.text = texts

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