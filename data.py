import torch
torch.manual_seed(0)
import torchaudio
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset

SAMPLE_RATE = 16000

def collect_audio_batch(batch, extra_noise=0., maxLen=600000):
    '''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>) 
       e.g. [(file1,txt1),(file2,txt2),...]
    '''
    def audio_reader(filepath):
        
        wav, sample_rate = torchaudio.load(filepath)
        if sample_rate != SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(wav)
        wav = wav.reshape(-1)
        if wav.shape[-1] >= maxLen:
            print(f'{filepath} has len {wav.shape}, truncate to {maxLen}')
            wav = wav[:maxLen]
            print(wav.shape)
        wav += extra_noise * torch.randn_like(wav)
        
        return wav

    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
    if type(batch[0]) is not tuple:
        batch = batch[0]

    # Read batch
    file, audio_feat, audio_len, text = [], [], [], []
    with torch.no_grad():
        for b in batch:
            feat = audio_reader(str(b[0])).numpy()
            file.append(str(b[0]).split('/')[-1].split('.')[0])
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(b[1])

    # Descending audio length within each batch
    audio_len, file, audio_feat, text = zip(*[(feat_len, f_name, feat, txt)
                                              for feat_len, f_name, feat, txt in sorted(zip(audio_len, file, audio_feat, text), reverse=True, key=lambda x:x[0])])

    return audio_len, audio_feat, text, file


def create_dataset(split, name, path, batch_size=1, **kwargs):
    ''' Interface for creating all kinds of dataset'''
    loader_bs = batch_size

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriDataset as Dataset
    elif name.lower() == "chime":
        from corpus.CHiME import CHiMEDataset as Dataset
    elif name.lower() == "ted":
        from corpus.ted import TedDataset as Dataset
    elif name.lower() == "commonvoice":
        from corpus.commonvoice import CVDataset as Dataset
    elif name.lower() == "valentini":
        from corpus.valentini import ValDataset as Dataset
    elif name.lower() =="l2arctic":
        from corpus.l2arctic import L2ArcticDataset as Dataset
    elif name.lower() == "aishell3":
        from corpus.aishell3 import AIShellDataset as Dataset
    elif name.lower() == "voicebank":
        from corpus.voicebank import VoicebankDataset as Dataset
    elif name.lower() == "noisylibri":
        from corpus.noisylibri import noisyLibriDataset as Dataset
    elif name.lower() == "multilibri":
        from corpus.multilibri import multiLingualLibriDataset as Dataset
        dataset = Dataset(split, batch_size, path, kwargs["noise_dir"], kwargs["snr"])
        print(f'[INFO]    There are {len(dataset)} samples.')
        return dataset, loader_bs
    elif name.lower() == "covost2":
        # ['en_de', 'en_tr', 'en_fa', 'en_sv-SE', 'en_mn', 'en_zh-CN', 'en_cy', 'en_ca', 'en_sl', 'en_et', 'en_id', 'en_ar', 'en_ta', 'en_lv', 'en_ja', 'fr_en', 'de_en', 'es_en', 'ca_en', 'it_en', 'ru_en', 'zh-CN_en', 'pt_en', 'fa_en', 'et_en', 'mn_en', 'nl_en', 'tr_en', 'ar_en', 'sv-SE_en', 'lv_en', 'sl_en', 'ta_en', 'ja_en', 'id_en', 'cy_en']
        from corpus.covost2 import covost2Dataset as Dataset
        dataset = Dataset(split, batch_size, path, kwargs["lang"], kwargs["noise_dir"], kwargs["snr"])
        return dataset, loader_bs
        
    else:
        raise NotImplementedError

    dataset = Dataset(split, batch_size, path)
    print(f'[INFO]    There are {len(dataset)} samples.')

    return dataset, loader_bs


def load_SUTAdataset(split=None, name='librispeech', path=None, batch_size=1, extra_noise=0., num_workers=4, noise_dir=None, snr=0., lang="en"):
    ''' Prepare dataloader for training/validation'''
    dataset, loader_bs = create_dataset(split, name, path, batch_size, noise_dir=noise_dir, snr=snr, lang=lang)
    collate_fn = None
    if name.lower() != 'multilibri' and name.lower() != 'covost2':
        collate_fn = partial(collect_audio_batch, extra_noise=extra_noise)

    dataloader = DataLoader(dataset, batch_size=loader_bs, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)
    return dataloader
