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

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DECODER_STEP = 512

def collect_params(model, train_feature, encoderLN, decoderLN, train_all):
    model.requires_grad_(False)
    params = []
    names = []
    for name, param in model.named_parameters():
        if 'conv' in str(name).split('.'):
            param.requires_grad = True
            params.append(param)
            names.append(f"{name}")
        if 'self_attn_layer_norm' in str(name).split('.'):
            param.requires_grad = True
            params.append(param)
            names.append(f"{name}")

    return names, params

def my_greedy_decode(model, input_features):
    # greedy decode
    # decode for pseudo label
    ori_generated_ids = torch.tensor([[1]]) * model.config.decoder_start_token_id
    ori_generated_ids = ori_generated_ids.to(DEVICE)

    decode_step = 0
    while(decode_step < MAX_DECODER_STEP):
        logits = model(input_features.to(DEVICE), decoder_input_ids=ori_generated_ids).logits
        next_token_logit = logits[:,-1,:]
        next_tokens = torch.argmax(next_token_logit, dim=-1).to(DEVICE)
        ori_generated_ids = torch.cat([ori_generated_ids, next_tokens[:, None]], dim=-1)
        if next_tokens == 2:
            break
    return ori_generated_ids

if __name__ == '__main__':
    args = OmegaConf.load("config.yaml")
    normalizer = EnglishTextNormalizer()
    processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
    dataset = load_dataset(name='noisy', path=args.dataset_dir, batch_size=1)
    exp_name = 'ex_data/suta_s2t_deLN'
    os.makedirs(exp_name, exist_ok=True)

    loss_fn = nn.CrossEntropyLoss()
    with open(f'{exp_name}/log.txt', 'a') as f:
        for count, batch in enumerate(dataset):
            step_loss, wers = [], []
            lens, wavs, texts, files = batch
            f.write(f'idx:{count}'+'\n')
            f.write('label:'+normalizer(texts[0])+'\n')
            
            input = feature_extractor(wavs, sampling_rate=16000,return_tensors='pt')
            model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
            model.eval()
            model = model.to(DEVICE)
            names, params = collect_params(model, args.train_feature, args.encoderLN, args.decoderLN, args.train_all)
            input_features = input.input_features.to(DEVICE)
            if count == 0:
                print(names)
                
            optimizer, scheduler = setup_optimizer(args, params, args.opt, args.lr, weight_decay=1e-4, scheduler=args.scheduler)
            # get words before TTA
            with torch.no_grad():
                ori_generated_ids = my_greedy_decode(model, input_features)
                ori_text = normalizer(processor.batch_decode(ori_generated_ids)[0])
                normalized_label = normalizer(texts[0])
                ori_wer = wer(normalized_label, ori_text)
                wers.append(ori_wer)
                f.write(f'ori({ori_wer}):{ori_text}\n')

            for step in range(args.steps):
                input_ids = torch.tensor([[1]]) * model.config.decoder_start_token_id
                input_ids = input_ids.to(DEVICE)

                # Teacher forcing
                record_loss = 0
                for i in range(ori_generated_ids.shape[1]):
                    logits = model(input_features, decoder_input_ids=input_ids).logits
                    next_token_logit = logits[:,-1,:]
                    # create soft label
                    pseudo_logit = torch.full((1,10000), 0.01).to(DEVICE)
                    if i+1 < ori_generated_ids.shape[1]:
                        pseudo_logit[0][ori_generated_ids[0][i+1]] = 1
                        p_loss = loss_fn(next_token_logit, pseudo_logit)
                    else:
                        # eos
                        pseudo_logit[0][2] = 1
                        p_loss = 5*loss_fn(next_token_logit, pseudo_logit)
                    
                    next_token_logit = torch.topk(next_token_logit, k=30).values
                    e_loss = softmax_entropy(next_token_logit / args.temp, dim=1).mean(0).mean()
                    c_loss = mcc_loss(next_token_logit.unsqueeze(0) / args.temp, class_num=args.topk)

                    loss = e_loss * args.em_coef + c_loss * (1 - args.em_coef) + p_loss/1000
                    record_loss += loss.item()
                    loss.backward()
                    
                    # try scheduled sampling
                    # pred_next_tokens = torch.argmax(next_token_logit, dim=-1).to(DEVICE)
                    # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                step_loss.append(record_loss / ori_generated_ids.shape[1])
                optimizer.step()
                if step % 3 == 0 or step == args.steps-1:
                    generated_ids, logits = model.generate(input_features, num_beams=1, do_sample=False)
                    after_text = normalizer(processor.batch_decode(generated_ids)[0])
                    after_wer = wer(normalized_label, after_text)
                    f.write(f'step{step}({after_wer}): {after_text}\n')
                    wers.append(after_wer)

            fig0, ax0 = plt.subplots(1,1)
            color = 'tab:red'
            ax0.set_xlabel('step')
            ax0.set_ylabel('loss', color=color)
            ax0.plot([loss for loss in step_loss], color=color)
            ax0.tick_params(axis='y', labelcolor=color)

            # plot wers
            ax2 = ax0.twinx()  # 共享 x 軸
            color = 'tab:blue'
            ax2.set_xlabel('step')
            ax2.set_ylabel('wer', color=color)
            check_list = [i+1 for i in range(args.steps) if i % 3 == 0 or i == args.steps-1]
            check_list.insert(0,0) 
            ax2.plot(check_list, wers, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            plt.title(f'idx:{count}')
            plt.savefig(f'{exp_name}/suta_{count}.png')
            plt.close()

            f.write("=======================================\n")

    
