import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from jiwer import wer
from data import *
from suta import *
from omegaconf import OmegaConf
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, generation, AutoFeatureExtractor
from whisper.normalizers import EnglishTextNormalizer
from datetime import datetime

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DECODER_STEP = 512


if __name__ == '__main__':
    args = OmegaConf.load("config.yaml")
    normalizer = EnglishTextNormalizer()
    processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
    dataset = load_dataset(name='noisy', path=args.dataset_dir, batch_size=1)
    os.makedirs(args.exp_name, exist_ok=True)
    config_str = OmegaConf.to_yaml(args)
    with open(f'{args.exp_name}/log.txt', 'w') as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'exp_time: {current_time}\n')
        f.write(config_str)

    loss_fn = nn.CrossEntropyLoss()
    with open(f'{args.exp_name}/result.txt', 'a') as f:
        for count, batch in enumerate(dataset):
            step_loss, wers = [], []
            lens, wavs, texts, files = batch
            f.write(f'idx:{count}'+'\n')
            f.write('label:'+normalizer(texts[0])+'\n')
            
            input = feature_extractor(wavs, sampling_rate=16000,return_tensors='pt')
            model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
            model.eval()
            model = model.to(DEVICE)
            names, params = HF_collect_params(args, model)
            input_features = input.input_features.to(DEVICE)
            if count == 0:
                print(names)
                
            optimizer, scheduler = setup_optimizer(args, params, args.opt, args.lr, weight_decay=1e-4, scheduler=args.scheduler)
            # get words before TTA
            with torch.no_grad():
                ori_generated_ids = my_greedy_decode(model, input_features, MAX_DECODER_STEP)
                ori_text = normalizer(processor.batch_decode(ori_generated_ids)[0])
                normalized_label = normalizer(texts[0])
                ori_wer = wer(normalized_label, ori_text)
                wers.append(ori_wer)
                f.write(f'ori({ori_wer}):{ori_text}\n')

            for step in range(args.steps):
                optimizer.zero_grad()
                input_ids = torch.tensor([[1]]) * model.config.decoder_start_token_id
                input_ids = input_ids.to(DEVICE)

                # Teacher forcing
                loss = 0
                record_loss = 0
                for i in range(ori_generated_ids.shape[1]):
                    logits = model(input_features, decoder_input_ids=input_ids).logits
                    next_token_logit = logits[:,-1,:]
                    # create soft label
                    pseudo_logit = torch.full((1,10000), 1e-6).to(DEVICE)
                    teacher_token = 2
                    if i+1 < ori_generated_ids.shape[1]:
                        teacher_token = ori_generated_ids[0][i+1]
                        pseudo_logit[0][teacher_token] = 1
                    
                    loss = loss_fn(next_token_logit, pseudo_logit)
                    record_loss += loss.item()
                    loss.backward()
                    input_ids = torch.cat([input_ids, torch.tensor([[teacher_token]]).to(DEVICE)], dim=-1)
                    
                    # try scheduled sampling
                    # pred_next_tokens = torch.argmax(next_token_logit, dim=-1).to(DEVICE)
                    # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                step_loss.append(record_loss / ori_generated_ids.shape[1])
                optimizer.step()
                if step % 3 == 0 or step == args.steps-1:
                    with torch.no_grad():
                        generated_ids, logits = model.generate(input_features, num_beams=1, do_sample=False)
                        after_text = normalizer(processor.batch_decode(generated_ids)[0])
                        after_wer = wer(normalized_label, after_text)
                    f.write(f'step{step}({after_wer}): {after_text}\n')
                    wers.append(after_wer)

            # fig0, ax0 = plt.subplots(1,1)
            # color = 'tab:red'
            # ax0.set_xlabel('step')
            # ax0.set_ylabel('loss', color=color)
            # ax0.plot([loss for loss in step_loss], color=color)
            # ax0.tick_params(axis='y', labelcolor=color)

            # # plot wers
            # ax2 = ax0.twinx()  # 共享 x 軸
            # color = 'tab:blue'
            # ax2.set_xlabel('step')
            # ax2.set_ylabel('wer', color=color)
            # check_list = [i+1 for i in range(args.steps) if i % 3 == 0 or i == args.steps-1]
            # check_list.insert(0,0) 
            # ax2.plot(check_list, wers, color=color)
            # ax2.tick_params(axis='y', labelcolor=color)
            # plt.title(f'idx:{count}')
            # plt.savefig(f'{exp_name}/suta_{count}.png')
            # plt.close()

            f.write("=======================================\n")