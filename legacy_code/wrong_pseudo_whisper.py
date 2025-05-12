import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch import nn

import whisper
from whisper.audio import (
    log_mel_spectrogram,
    pad_or_trim,
)
from tqdm import tqdm
from suta import *
from utils.data import *

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DECODER_STEP = 128

from jiwer import wer
import hydra
from omegaconf import OmegaConf
from whisper.normalizers import EnglishTextNormalizer
from datetime import datetime
from timeit import default_timer

class MyDecode(whisper.decoding.DecodingTask):
    def _main_loop(self, audio_features, tokens):
        n_batch = tokens.shape[0]
        sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(MAX_DECODER_STEP):
                logits = self.inference.logits(tokens, audio_features) # (1,2,51864)
                if (
                    i == 0 and self.tokenizer.no_speech is not None
                ):  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs
    
    @torch.no_grad()
    def run(self, mel):
        self.decoder.reset()
        tokenizer = self.tokenizer
        n_audio = mel.shape[0]

        audio_features = self._get_audio_features(mel)  # encoder forward pass
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)
        try:
            self.ori_generated_ids
            self.teacher_forcing_step
        except:
            self.ori_generated_ids = tokens
            self.teacher_forcing_step = tokens.shape[1]

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts = [tokenizer.decode(t).strip() for t in tokens]

        return tokens, texts

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def AED_suta(self, audio_features, tokens, optimizer, scheduler, scaler=None,args=None):
        loss_fn = nn.CrossEntropyLoss()
        n_batch = tokens.shape[0]
        sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        optimizer.zero_grad()
        loss = 0
        e_loss_list = []
        c_loss_list = []
        token_count = 0
        try:
            for i in range(self.teacher_forcing_step-3):
                token_count += 1
                logits = self.inference.logits(tokens, audio_features) # (1,2,51864)
                if (
                    i == 0 and self.tokenizer.no_speech is not None
                ):  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]
                next_tokens = torch.argmax(logits, dim=-1)
                # create soft label
                teacher_token = 50257  # init as eos
                p_loss = 0
                if i+4 < self.teacher_forcing_step:
                    teacher_token = self.ori_generated_ids[0][i+4]
                if teacher_token == 50257 or next_tokens == 50257:
                    pseudo_logit = torch.full((1,logits.shape[1]), 1e-6).to(DEVICE)
                    pseudo_logit[0][teacher_token] = 1
                    p_loss = loss_fn(logits, pseudo_logit)
                
                topk_logits = torch.topk(logits, k=30).values
                topk_logits = topk_logits/args.temp
                e_loss = softmax_entropy(topk_logits, dim=1).mean(0).mean()
                c_loss = mcc_loss(topk_logits.unsqueeze(0), class_num=args.topk)
                if next_tokens == 50257:
                    loss += e_loss * args.em_coef + c_loss * (1 - args.em_coef) + args.p_ratio * p_loss
                elif teacher_token == 50257:
                    loss += e_loss * args.em_coef + c_loss * (1 - args.em_coef) + args.p_ratio * p_loss
                    break
                else:
                    loss += e_loss * args.em_coef + c_loss * (1 - args.em_coef)
                e_loss_list.append(e_loss.item())
                c_loss_list.append(c_loss.item())
                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            with torch.no_grad():
                self.inference.cleanup_caching()
        loss /= token_count
        loss.backward()
        optimizer.step()
        # optimizer.step()
        if scheduler is not None:
            scheduler.step()

        return tokens, loss, e_loss_list, c_loss_list,p_loss
    
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def adapt(self, mel, args, optimizer, scheduler=None, scaler=None, generate_text=False):
        self.decoder.reset()
        n_audio = mel.shape[0]
        
        audio_features = self._get_audio_features(mel)  # encoder forward pass
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, loss, e_loss_list, c_loss_list, p_loss = self.AED_suta(audio_features, tokens, optimizer, scheduler, scaler, args)
        # print(tokenizer.decode([50258, 50259, 50359, 50363, 50257]))
        texts = None
        if generate_text:
            _, texts = self.run(mel)
        if not isinstance(p_loss, int):
            p_loss = p_loss.item()
        return texts, loss.item(), p_loss

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args):
    normalizer = EnglishTextNormalizer()
    
    dataset = load_dataset(name='noisy', path=args.dataset_dir, batch_size=1)
    os.makedirs(args.exp_name, exist_ok=True)
    os.makedirs(f'{args.exp_name}/figs', exist_ok=True)
    config_str = OmegaConf.to_yaml(args)
    with open(f'{args.exp_name}/log.txt', 'w') as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'start time: {current_time}\n')
        f.write(config_str)

    step_loss = []
    p_loss_list = []
    with open(f'{args.exp_name}/result.txt', 'a') as f:
        for count, batch in tqdm(enumerate(dataset)):
            model = whisper.load_model(args.asr).to(DEVICE)
            params, names = whisper_collect_params(model, args.encoderLN, args.decoderLN, train_feature=args.train_feature)
            options = whisper.DecodingOptions(language="en", without_timestamps=True)
            p_loss_list, step_loss, wers = [], [], []
            lens, wavs, texts, files = batch
            f.write(f'idx:{count}'+'\n')
            f.write('label:'+normalizer(texts[0])+'\n')
            model.eval()
            if args.asr == 'large' or args.asr == 'large_v2' or args.asr == 'large_v3':
                mel = log_mel_spectrogram(pad_or_trim(wavs[0]), n_mels=128).unsqueeze(0)
            else:
                mel = log_mel_spectrogram(pad_or_trim(wavs[0])).unsqueeze(0)
            mel = mel.to(DEVICE)
            if count == 0:
                import pprint
                with open(f'{args.exp_name}/log.txt', 'a') as logfile:
                    pp = pprint.PrettyPrinter(stream=logfile)
                    pp.pprint(names)
                    total_params = sum(
                        param.numel() for param in model.parameters()
                    )
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    train_ratio = trainable_params / total_params
                    print(f'train_param ratio: {train_ratio}')
                    logfile.write(f'train_param ratio: {train_ratio}\n')
                
            optimizer, scheduler = setup_optimizer(args, params, args.opt, args.lr, weight_decay=1e-4, scheduler=args.scheduler)
            # scaler = GradScaler()
            decode_obj = MyDecode(model, options)
            # get words before TTA
            with torch.no_grad():
                _, ori_text = decode_obj.run(mel)
                ori_text = normalizer(ori_text[0])
                normalized_label = normalizer(texts[0])
                ori_wer = wer(normalized_label, ori_text)
                wers.append(ori_wer)
                f.write(f'ori({ori_wer}):{ori_text}\n')
            for step in range(args.steps):
                if step % 3 == 0 or step == args.steps-1:
                    adapt_text, loss, p_loss = decode_obj.adapt(mel, args, optimizer, scheduler, generate_text=True)
                    adapt_text = normalizer(adapt_text[0])
                    adapt_wer = wer(normalized_label, adapt_text)
                    wers.append(adapt_wer)
                    f.write(f'step{step}({adapt_wer}): {adapt_text}\n')
                else:
                    adapt_text, loss, p_loss = decode_obj.adapt(mel, args, optimizer, scheduler)
                step_loss.append(loss)
                p_loss_list.append(p_loss)

            # 10 figures are enough
            if count < 5 or count >2930:
                fig0, ax0 = plt.subplots(1,1)
                color = 'tab:red'
                ax0.set_xlabel('step')
                ax0.set_ylabel('loss', color=color)
                ax0.plot([loss for loss in step_loss], color=color)
                ax0.tick_params(axis='y', labelcolor=color)

                ax2 = ax0.twinx()  # 共享 x 軸
                color = 'tab:blue'
                ax2.set_xlabel('step')
                ax2.set_ylabel('p_loss', color=color)
                ax2.plot([i for i in p_loss_list], color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                plt.title(f'idx:{count}')
                plt.savefig(f'{args.exp_name}/figs/suta_{count}.png')
                plt.close()

            f.write("=======================================\n")
    with open(f'{args.exp_name}/log.txt', 'a') as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'end time: {current_time}\n')

if __name__ == '__main__':
    main()