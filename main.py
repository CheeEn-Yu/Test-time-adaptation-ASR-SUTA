import os
import argparse
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
from suta import *

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('--asr', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--task', type=str, default="transcription")
    parser.add_argument('--lang', type=str, default="pt")
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--div_coef', type=float, default=0.)
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--dataset_name', type=str, default='librispeech')
    parser.add_argument('--dataset_dir', type=str, default='/home/daniel094144/data/LibriSpeech')
    parser.add_argument('--split', default=['test-other'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--em_coef', type=float, default=1.)
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--bias_only', action='store_true')
    parser.add_argument('--train_feature', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--is_whisper', action=argparse.BooleanOptionalAction)
    parser.add_argument('--encoderLN', action=argparse.BooleanOptionalAction)
    parser.add_argument('--decoderLN', action=argparse.BooleanOptionalAction)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temp', type=float, default=2.5)
    parser.add_argument('--non_blank', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./exps')
    parser.add_argument('--extra_noise', type=float, default=0.)
    parser.add_argument('--scheduler', default=None)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--beam_size', type=int, default=0)

    args = parser.parse_args()
    em_coef = args.em_coef
    reweight = args.reweight
    batch_size = args.batch_size
    temp =  args.temp
    non_blank = args.non_blank
    extra_noise = args.extra_noise
    scheduler = args.scheduler
    div_coef = args.div_coef
    train_feature = args.train_feature
    is_whisper = args.is_whisper
    skip_short_thd = None

    # load models
    model = whisper.load_model(args.asr)
    params, names = whisper_collect_params(model, args.encoderLN, args.decoderLN, train_feature=train_feature)
    optimizer, scheduler = setup_optimizer(params, args.opt, args.lr, scheduler=scheduler)

    # load datasets and decide decoding option
    print(f'[logger] - loading data...')
    if args.task == 'translation':
        from datasets import load_dataset
        dataset = load_dataset("covost2", f'{args.lang}_en', data_dir=args.dataset_dir,split='test', trust_remote_code=True)
        options = whisper.DecodingOptions(language=args.lang, task='translation', without_timestamps=True)

    else:
        from data import load_dataset
        import jiwer
        from whisper.normalizers import EnglishTextNormalizer
        dataset = load_dataset(args.split, args.dataset_name, args.dataset_dir, batch_size, extra_noise)

        if dataset_name == 'aishell3':
            if args.beam_size == 0:
                options = whisper.DecodingOptions(language="zh", prompt="简体", without_timestamps=True)
            else:
                options = whisper.DecodingOptions(language="zh", beam_size=args.beam_size,prompt="简体", without_timestamps=True)

        elif args.beam_size == 0:
            options = whisper.DecodingOptions(language="en", without_timestamps=True)
        else:
            options = whisper.DecodingOptions(language="en", beam_size=args.beam_size, without_timestamps=True)

    data = pd.DataFrame()
    transcriptions_1 = []
    transcriptions_3 = []
    transcriptions_5 = []
    transcriptions_10 = []
    ori_transcriptions = []
    ori_langs = []
    labels = []
    model_state, optimizer_state, scheduler_state = copy_model_and_optimizer(model, optimizer, scheduler)
    
    print(f'[logger] - start training...')
    try:
        count = 0
        if args.task == 'translation':
            for batch in tqdm(dataset):
                count+=1
                ori_langs.append(batch['sentence'])
                labels.append(batch['translation'])
                if count > 1000:
                    break
                wavs = torch.Tensor(batch['audio']['array'])
                if args.asr == 'large' or args.asr == 'large_v2' or args.asr == 'largev3':
                    mel = log_mel_spectrogram(pad_or_trim(wavs), n_mels=128).unsqueeze(0)
                else:
                    mel = log_mel_spectrogram(pad_or_trim(wavs)).unsqueeze(0)
                model.eval()
                with torch.no_grad():
                    mel = mel.to(DEVICE)
                    outputs = model.decode(mel, options)
                ori_transcriptions.extend([result.text for result in outputs[0]])

                model, optimizer, scheduler = load_model_and_optimizer(model, optimizer, scheduler, model_state, optimizer_state, scheduler_state)
                model = model.to(DEVICE)
                for i in range(args.steps):
                    adapt_output = forward_and_adapt(mel, model, optimizer, em_coef, reweight, temp, non_blank, scheduler, div_coef,topk=args.topk, beam_size=args.beam_size, is_whisper=is_whisper, options=options)
                    if i == 0:
                        transcriptions_1.append(adapt_output[0][0].text)
                    if i == 2:
                        transcriptions_3.append(adapt_output[0][0].text)
                    if i == 4:
                        transcriptions_5.append(adapt_output[0][0].text)
                    

                transcriptions_10.append(adapt_output[0][0].text)
                
        else:
            for batch in tqdm(dataset):
                count+=1
                if count > 1000:
                    break
                lens, wavs, texts, files = batch
                # the code is for batch size = 1
                labels.append(texts[0])
                if args.asr == 'large' or args.asr == 'large_v2' or args.asr == 'largev3':
                    mel = log_mel_spectrogram(pad_or_trim(wavs[0]), n_mels=128).unsqueeze(0).to(DEVICE)
                else:
                    mel = log_mel_spectrogram(pad_or_trim(wavs[0])).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    ori_transcriptions.append(model.decode(mel, options)[0][0].text)
                model, optimizer, scheduler = load_model_and_optimizer(model, optimizer, scheduler, model_state, optimizer_state, scheduler_state)
                model = model.to(DEVICE)
                for i in range(args.steps):
                    adapt_output = forward_and_adapt(mel, model, optimizer, em_coef, reweight, temp, non_blank, scheduler, div_coef,topk=args.topk, beam_size=args.beam_size, is_whisper=is_whisper, options=options)
                    if i == 0:
                        transcriptions_1.append(adapt_output[0][0].text)
                    if i == 2:
                        transcriptions_3.append(adapt_output[0][0].text)
                    if i == 4:
                        transcriptions_5.append(adapt_output[0][0].text)
                    

                transcriptions_10.append(adapt_output[0][0].text)
        del adapt_output
        torch.cuda.empty_cache()
    except Exception as e:
        print("[logger] - error occurred:"+str(e))
        try:
            if args.task == 'translation':
                data = pd.DataFrame(dict(before_adapt=ori_transcriptions,step1=transcriptions_1,step3=transcriptions_3,step5=transcriptions_5,step10=transcriptions_10, reference=labels, ori_lang=ori_langs))
            else:
                data = pd.DataFrame(dict(before_adapt=ori_transcriptions,step1=transcriptions_1,step3=transcriptions_3,step5=transcriptions_5,step10=transcriptions_10, reference=labels))
        except:
            max_length = max(len(ori_transcriptions), len(transcriptions_1), len(transcriptions_3), len(transcriptions_5), len(transcriptions_10), len(labels))
            ori_transcriptions += ['oom_pad'] * (max_length - len(ori_transcriptions))
            transcriptions_1 += ['oom_pad'] * (max_length - len(transcriptions_1))
            transcriptions_3 += ['oom_pad'] * (max_length - len(transcriptions_3))
            transcriptions_5 += ['oom_pad'] * (max_length - len(transcriptions_5))
            transcriptions_10 += ['oom_pad'] * (max_length - len(transcriptions_10))
            labels += ['oom_pad'] * (max_length - len(labels))
            if args.task == 'translation':
                ori_langs += ['oom_pad'] * (max_length - len(ori_transcriptions))
                data = pd.DataFrame(dict(before_adapt=ori_transcriptions,step1=transcriptions_1,step3=transcriptions_3,step5=transcriptions_5,step10=transcriptions_10, reference=labels, ori_lang=ori_langs))
            else:
                data = pd.DataFrame(dict(before_adapt=ori_transcriptions,step1=transcriptions_1,step3=transcriptions_3,step5=transcriptions_5,step10=transcriptions_10, reference=labels))

    if data.empty:
        data = pd.DataFrame(dict(before_adapt=ori_transcriptions,step1=transcriptions_1,step3=transcriptions_3,step5=transcriptions_5,step10=transcriptions_10, reference=labels))
    
    exp_name = args.asr+'_'+args.dataset_name+'_temp_'+str(args.temp)+'_noise_'+str(args.extra_noise)+'_lr_'+str(args.lr)+'_EMcoef_'+str(args.em_coef)+'_trainFeature_'+str(args.train_feature)+'_encoderLN_'+str(args.encoderLN)+'_decoderLN_'+str(args.decoderLN)+'_topk_'+str(args.topk)+'_beam_'+str(args.beam_size)
    if args.task == 'translation':
        exp_name = args.task+'_'+args.lang+'_'+ exp_name

        # calculate BLEU scores
        import nltk
        scores = []
        ori_scores = []
        scores1 = []
        scores3 = []
        scores5 = []
        scores10 = []
        for i in range(len(transcriptions_1)):
            ori_scores.append(nltk.translate.bleu_score.sentence_bleu([word for word in labels[i].split(' ')], [word for word in ori_transcriptions[i].split(' ')]))
            scores1.append(nltk.translate.bleu_score.sentence_bleu([word for word in labels[i].split(' ')], [word for word in transcriptions_1[i].split(' ')]))
            scores3.append(nltk.translate.bleu_score.sentence_bleu([word for word in labels[i].split(' ')], [word for word in transcriptions_3[i].split(' ')]))
            scores5.append(nltk.translate.bleu_score.sentence_bleu([word for word in labels[i].split(' ')], [word for word in transcriptions_5[i].split(' ')]))
            scores10.append(nltk.translate.bleu_score.sentence_bleu([word for word in labels[i].split(' ')], [word for word in transcriptions_10[i].split(' ')]))
        scores.append(np.array(ori_scores).mean())
        scores.append(np.array(scores1).mean())
        scores.append(np.array(scores3).mean())
        scores.append(np.array(scores5).mean())
        scores.append(np.array(scores10).mean())
        with open(f"bleu_{exp_name}.txt", 'w') as f:
            for i in scores:
                f.write(f'BLEU: {i}'+'\n')

    elif args.dataset_name == 'aishell3':
        import cn2an
        from opencc import OpenCC
        cc = OpenCC('t2s')
        data["before_adapt_clean"] = [cn2an.transform(cc.convert(text) ,"an2cn") for text in data["before_adapt"]]
        data["step1_clean"] = [cn2an.transform(cc.convert(text) ,"an2cn") for text in data["step1"]]
        data["step3_clean"] = [cn2an.transform(cc.convert(text) ,"an2cn") for text in data["step3"]]
        data["step5_clean"] = [cn2an.transform(cc.convert(text) ,"an2cn") for text in data["step5"]]
        data["step10_clean"] = [cn2an.transform(cc.convert(text) ,"an2cn") for text in data["step10"]]
        cer_list = []
        cer_list.append(jiwer.cer(list(data["reference_clean"]), list(data["before_adapt_clean"])))
        cer_list.append(jiwer.cer(list(data["reference_clean"]), list(data["step1_clean"])))
        cer_list.append(jiwer.cer(list(data["reference_clean"]), list(data["step3_clean"])))
        cer_list.append(jiwer.cer(list(data["reference_clean"]), list(data["step5_clean"])))
        cer_list.append(jiwer.cer(list(data["reference_clean"]), list(data["step10_clean"])))
        with open(f"wer_{exp_name}.txt", 'w') as f:
            for i in cer_list:
                f.write(f'CER: {i}'+'\n')
    else:
        normalizer = EnglishTextNormalizer()
        data["before_adapt_clean"] = [normalizer(text) for text in data["before_adapt"]]
        data["step1_clean"] = [normalizer(text) for text in data["step1"]]
        data["step3_clean"] = [normalizer(text) for text in data["step3"]]
        data["step5_clean"] = [normalizer(text) for text in data["step5"]]
        data["step10_clean"] = [normalizer(text) for text in data["step10"]]
        data["reference_clean"] = [normalizer(text) for text in data["reference"]]
        wer_list = []
        wer_list.append(jiwer.wer(list(data["reference_clean"]), list(data["before_adapt_clean"])))
        wer_list.append(jiwer.wer(list(data["reference_clean"]), list(data["step1_clean"])))
        wer_list.append(jiwer.wer(list(data["reference_clean"]), list(data["step3_clean"])))
        wer_list.append(jiwer.wer(list(data["reference_clean"]), list(data["step5_clean"])))
        wer_list.append(jiwer.wer(list(data["reference_clean"]), list(data["step10_clean"])))

        with open(f"wer_{exp_name}.txt", 'w') as f:
            for i in wer_list:
                f.write(f'WER: {i}'+'\n')

    data.to_csv(f'{exp_name}.csv')
    print(f'[logger] - finish the exp:\n{exp_name}')
    