from torch.utils.data import Dataset
from datasets import load_dataset

class multiLingualLibriDataset(Dataset):
    def __init__(self, split, bucket_size, lang):
        # Setup
        split = ['test']
        self.bucket_size = bucket_size
        ds = load_dataset("facebook/multilingual_librispeech", lang, split="test")
        ds = ds.remove_columns(['original_path', 'begin_time', 'end_time', 'speaker_id', 'chapter_id', 'id'])
        self.ds = ds

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            raise NotImplementedError
        else:
            return self.ds[index]['audio_duration'], self.ds[index]['audio']['array'], self.ds[index]['transcript'], self.ds[index]['file']

    def __len__(self):
        return len(self.ds)