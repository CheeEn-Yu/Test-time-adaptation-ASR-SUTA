import numpy as np
import re
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from jiwer import wer
import whisper

class TTADecode(whisper.decoding.DecodingTask):
    '''
    This decoding strategy is fitting in Whisper original codebase.
    '''
    def _main_loop(self, audio_features, tokens, max_decoder_step=128):
        '''
        Original Whisper autoregressive decode
        '''
        n_batch = tokens.shape[0]
        sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        try:
            for i in range(max_decoder_step):
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
    def run(self, mel, max_decoder_step=128):
        '''
        Original Whisper decode. Always run this first before use AED_suta for saving teacher token.
        '''
        self.decoder.reset()
        tokenizer = self.tokenizer
        n_audio = mel.shape[0]
        audio_features = self._get_audio_features(mel)  # encoder forward pass
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)
        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens, max_decoder_step)
        # Save teacher token and the length of original recognized sentence
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
    
    def AED_suta(self, audio_features, tokens, optimizer, scheduler, args=None):
        loss_fn = nn.CrossEntropyLoss()
        n_batch = tokens.shape[0]
        sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        optimizer.zero_grad()
        loss = 0
        c_loss = 0
        p_loss = 0
        e_loss_list = []
        p_loss_list = []
        token_count = 0
        try:
            for i in range(self.teacher_forcing_step-3):
                token_count += 1
                logits = self.inference.logits(tokens, audio_features) # dimension: (1,2,51864)
                if (
                    i == 0 and self.tokenizer.no_speech is not None
                ):  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
                # now we need to consider the logits at the last token only
                logits = logits[:, -1]
                next_tokens = torch.argmax(logits, dim=-1)
                topk_logits = torch.topk(logits, k=args.topk).values
                topk_logits = topk_logits/args.temp
                if "c_loss" in args.objective_f:
                    try:
                        logits_list = torch.cat((logits_list, topk_logits), dim=0)
                    except:
                        logits_list = topk_logits
                teacher_token = 50257  # init as eos
                if i+4 < self.teacher_forcing_step:
                    teacher_token = self.ori_generated_ids[0][i+4]
                e_loss = softmax_entropy(topk_logits, dim=1).mean(0).mean()
                loss += (1/(1+args.alpha*torch.exp(-e_loss))) * e_loss if "weighted" in args.objective_f else e_loss
                if "p_loss" in args.objective_f:
                    if teacher_token == 50257 or next_tokens == 50257:
                        pseudo_logit = torch.full((1,logits.shape[1]), 1e-6).to(args.device) # create soft label
                        pseudo_logit[0][teacher_token] = 1
                        p_loss = loss_fn(logits, pseudo_logit)
                        loss += p_loss
                try:
                    p_loss_list.append(p_loss)
                except:
                    p_loss_list.append(p_loss.item())
                e_loss_list.append(e_loss.item())
                             
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
        if 'c_loss' in args.objective_f:
            c_loss = mcc_loss(logits_list.unsqueeze(0), class_num=args.topk)
            loss = loss * args.em_coef + c_loss * (1 - args.em_coef)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return tokens, loss, e_loss_list, c_loss, p_loss
    
    def adapt(self, mel, args, optimizer, scheduler=None, scaler=None, generate_text=False):
        self.decoder.reset()
        n_audio = mel.shape[0]
        
        audio_features = self._get_audio_features(mel)  # encoder forward pass
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)
        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
        # call the main sampling loop
        tokens, loss, e_loss_list, c_loss, p_loss = self.AED_suta(audio_features, tokens, optimizer, scheduler, args)
        # can check the decoded tokens by: print(tokenizer.decode([50258, 50259, 50359, 50363, 50257]))
        texts = None
        if generate_text:
            _, texts = self.run(mel)
        if not isinstance(p_loss, int):
            p_loss = p_loss.item()
        return texts, loss.item(), c_loss, p_loss

def my_greedy_decode(model, input_features, max_step):
    '''
    self-implement greedy_decode example for HuggingFace codebase.
    '''
    # greedy decode for pseudo label
    ori_generated_ids = torch.tensor([[1]]) * model.config.decoder_start_token_id
    ori_generated_ids = ori_generated_ids.to('cuda')

    decode_step = 0
    while(decode_step < max_step):
        logits = model(input_features.to('cuda'), decoder_input_ids=ori_generated_ids).logits
        next_token_logit = logits[:,-1,:]
        next_tokens = torch.argmax(next_token_logit, dim=-1).to('cuda')
        ori_generated_ids = torch.cat([ori_generated_ids, next_tokens[:, None]], dim=-1)
        if next_tokens == 2:
            break
    return ori_generated_ids

def setup_optimizer(args, params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7, verbose=False):
    opt = getattr(torch.optim, opt_name)
    if verbose:
        print(f'[INFO]    optimizer: {opt}')
        print(f'[INFO]    scheduler: {scheduler}')
    if opt_name == 'Adam':       
        optimizer = opt(params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)

    if scheduler is not None: 
        return optimizer, eval(scheduler)(optimizer, T_max=args.t_max, eta_min=args.lr_min)
    else: 
        return optimizer, None


def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

def mcc_loss(x, reweight=False, dim=2, class_num=32):
    p = x.softmax(dim) # (1, L, D)
    p = p.squeeze(0) # (L, D)
    if reweight: # (1, L,D) * (L, 1) 
        target_entropy_weight = softmax_entropy(x, dim=2).detach().squeeze(0) # instance-wise entropy (1, L, D)
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight) # (1, L)
        target_entropy_weight = x.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = p.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(p)
    else:    
        cov_matrix_t = p.transpose(1, 0).mm(p) # (D, L) * (L, D) -> (D, D)

    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num

    return mcc_loss

def div_loss(x, non_blank=None, L_thd=64):
    # maximize entropy of class prediction for every time-step in a utterance 
    # x (1, L, D)
    loss = 0
    x = x.squeeze(0)
    L = x.shape[0]

    if non_blank is not None: 
        cls_pred = x.mean(0)[1:] # (D, )
    else:
        cls_pred = x.mean(0) # (D, )

    loss = -softmax_entropy(cls_pred, 0)

    return loss


def whisper_collect_params(model, encoderLN, decoderLN, train_feature=False, linear_layer=False, all_encoder=False):
    # collect trainable params
    params = []
    names = []

    for param in model.parameters():
        param.requires_grad = False

    for nm, m in model.named_modules():
        trainable = ['weight', 'bias']
        attr_list = str(nm).split('.')
        # train_LN
        if all_encoder:
            if str(nm).split('.')[0] == 'encoder':
                for np, p in m.named_parameters():
                    if np in trainable:  
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")

        if isinstance(m, nn.LayerNorm):
            if encoderLN and not all_encoder:
                if str(nm).split('.')[0] == 'encoder':
                    for np, p in m.named_parameters():
                        if np in trainable:  
                            p.requires_grad = True
                            params.append(p)
                            names.append(f"{nm}.{np}")
            if decoderLN:
                if str(nm).split('.')[0] == 'decoder':
                    for np, p in m.named_parameters():
                        if np in trainable:  
                            p.requires_grad = True
                            params.append(p)
                            names.append(f"{nm}.{np}")

        # train_feature
        if train_feature:
            if len(attr_list) > 1:
                if attr_list[0] == 'encoder' and (attr_list[1] == 'conv1' or attr_list[1] == 'conv2'):
                    for np, p in m.named_parameters():
                        if np == 'bias':
                            p.requires_grad = True
                            params.append(p)
                            names.append(f"{nm}.{np}")
        # if linear_layer:
        #     if '3' in attr_list and 'mlp' in attr_list and 'encoder' in attr_list:
        #         for np, p in m.named_parameters():
        #             if np in trainable:  
        #                 p.requires_grad = True
        #                 params.append(p)
        #                 names.append(f"{nm}.{np}")

        # cross attention bias
        # if 'cross_attn' in nm.split('.'):
        #     if 'query' in nm.split('.') or 'value' in nm.split('.'):
        #         for np, p in m.named_parameters():
        #             if np in trainable and np == 'bias':
        #                 p.requires_grad = True
        #                 params.append(p)
        #                 names.append(f"{nm}.{np}")

    return params, names

import torch.nn.functional as F
# dropout
def consist_loss(model, input_values, outputs):
    targets = outputs
    # noisy outputs
    model.wav2vec2.encoder.dropout.train()
    noisy_outputs = model(input_values).logits

    import json
    f = open('vocab.json')
    vocab = json.load(f)


    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=False)
    predicted_ids = torch.argmax(outputs, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    target = []
    for s in transcription:
        if s == ' ':
            s = '|'
        target.append(vocab[s])

    logp = noisy_outputs.log_softmax(1).transpose(1, 0) # L,N,D
    input_len = logp.shape[0]
    tgt_len = len(target)
    loss = ctc_loss(logp, torch.tensor(target).int(), torch.tensor([input_len]), torch.tensor([tgt_len]))
    model.eval()
    return loss


from copy import deepcopy
def copy_model_and_optimizer(model, optimizer, scheduler):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    if scheduler is not None:
        scheduler_state = deepcopy(scheduler.state_dict())
        return model_state, optimizer_state, scheduler_state
    else:
        return model_state, optimizer_state, None

def load_model_and_optimizer(model, optimizer, scheduler, model_state, optimizer_state, scheduler_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
        return model, optimizer, scheduler
    else: 
        return model, optimizer, None


def cal_grad(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def configure_model(model):
    """Configure model for use with tent."""
    model.requires_grad_(False)
    return model

def forward_and_adapt(x, model, optimizer, em_coef=0.9, reweight=False, temp=1., not_blank=True, scheduler=None, 
                        div_coef=0, topk=0, beam_size=0, repeat_inference=True, is_whisper=True, options=None, skip_short_thd=None ):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    the index of <pad> in vocab is 0
    """
    # forward
    if is_whisper:
        outputs = model.decode(x, options)
        logits = torch.stack(outputs[1], dim=0)
        if topk != 0:
            logits, idx = torch.topk(logits,k=topk)
    else:
        outputs = model(x).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
    # adapt
    loss = 0

    if em_coef > 0: 
        if not is_whisper:
            if not_blank:      
                e_loss = softmax_entropy(outputs / temp)[non_blank].mean(0).mean()

            else: 
                e_loss = softmax_entropy(outputs / temp).mean(0).mean() 
        else:
            if beam_size == 0:
                logits = torch.permute(logits, (1,0,2))
                e_loss = softmax_entropy(logits / temp).mean(0).mean()
            else:
                logits = logits[:,outputs[2][0], :].unsqueeze(0)
                e_loss = softmax_entropy(logits / temp).mean(0).mean()
        loss += e_loss * em_coef

    if 1 - em_coef > 0: 
        c_loss = mcc_loss(logits / temp, class_num=topk)
        loss += c_loss * (1 - em_coef)

    if div_coef > 0: 
        d_loss = div_loss(outputs, not_blank) 
        loss += d_loss * div_coef 

    loss.backward()
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()
    model.zero_grad()

    # inference again
    if repeat_inference:
        if not is_whisper:
            with torch.no_grad():
                outputs = model(x).logits
        else:
            with torch.no_grad():
                outputs = model.decode(x, options)

    return outputs


class transcriptionProcessor:
    def __init__(self):
        self.data = self._initialize_lists()

    def _initialize_lists(self):
        return {
            'ori_wers': [], 'ori_transcription': [], 'labels': [],
            **{f'wer_{i}': [] for i in [0, 3, 6, 9, 12, 14, 18, 20]},
            **{f'transcriptions_{i}': [] for i in [0, 3, 6, 9, 12, 14, 18, 20]}
        }

    def parse_line(self, line):
        line = line.strip()
        try:
            sentence = line.split(':')[1]
            match = re.search(r'\((.*?)\)', line.split(':')[0])
            value = float(match.group(1))
            return value, sentence
        except:
            return None, None

    def process_line(self, line):
        value, sentence = self.parse_line(line)
        if value is None:
            return

        if line.startswith('ori'):
            self.data['ori_wers'].append(value)
            self.data['ori_transcription'].append(sentence)
        elif line.startswith('label'):
            self.data['labels'].append(sentence)
        else:
            step = re.search(r'step(\d+)', line)
            if step:
                step = int(step.group(1))
                if step in [0, 3, 6, 9, 12, 14, 18, 21]:
                    key = step
                    self.data[f'wer_{key}'].append(value)
                    self.data[f'transcriptions_{key}'].append(sentence)
                elif step in [24, 27, 30, 33, 36, 39]:
                    self.data[f'wer_{step}'].append(value)

    def process_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                self.process_line(line)

    def step_mean_wer(self):
        mean_wer_list = [f'ori_wers: {np.array(self.data["ori_wers"]).mean() * 100:.2f}%']
        for key in [f'wer_{i}' for i in [0, 3, 6, 9, 12, 14, 18, 20]]:
            try:
                mean_wer_list.append(f'{key}: {np.array(self.data[key]).mean() * 100:.2f}%')
            except:
                pass
        return mean_wer_list

    def get_data(self):
        return self.data

from transformers import WhisperForConditionalGeneration
def hf_collect_params(model):
    params, names = [], []
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "layer_norm" in name or "LayerNorm" in name:
            params.append(param)
            names.append(name)
            param.requires_grad = True
        # elif "conv1" in name or "conv2" in name:
        #     if "bias" in name:
        #         params.append(param)
        #         names.append(name)
        #         param.requires_grad = True


    return params, names

class WhisperTTADecoder(WhisperForConditionalGeneration):
    def decode(self, inputs, max_length=64, num_beams=1, do_sample=False, temperature=1.0, forced_decoder_ids=None, **generation_args):
        """
        Decoding function that supports Greedy, Sampling, and Beam Search decoding strategies.
        Arguments:
            inputs: Tensor, the input features for Whisper model.
            max_length: int, the maximum number of tokens to generate.
            num_beams: int, the number of beams for beam search. Set to 1 for greedy or sampling decoding.
            do_sample: bool, whether to use sampling decoding. Set to True to enable sampling.
            temperature: float, used for controlling randomness in sampling decoding.
        """
        if num_beams > 1:
            # Beam Search Decoding
            return self._beam_search_decode(inputs, max_length, num_beams, forced_decoder_ids)
        elif do_sample:
            # Sampling Decoding
            return self._sample_decode(inputs, max_length, temperature, forced_decoder_ids)
        else:
            # Greedy Decoding
            return self._greedy_decode(inputs, max_length, forced_decoder_ids)
    
    def _greedy_decode(self, inputs, max_length, forced_decoder_ids, **generation_args):
        """
        Greedy decoding implementation.
        """
        generated_ids = torch.tensor([[self.config.decoder_start_token_id]], device=inputs.device)
        if forced_decoder_ids:
            forced_token_ids = torch.tensor([token_id for _, token_id in forced_decoder_ids], device=inputs.device).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, forced_token_ids], dim=1)
        for _ in range(max_length):
            outputs = self(input_features=inputs, decoder_input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            if next_token.item() == self.config.eos_token_id:
                break
        return generated_ids

    def _sample_decode(self, inputs, max_length, temperature, forced_decoder_ids):
        """
        Sampling decoding implementation.
        """
        generated_ids = torch.tensor([[self.config.decoder_start_token_id]], device=inputs.device)
        if forced_decoder_ids:
            forced_token_ids = torch.tensor([token_id for _, token_id in forced_decoder_ids], device=inputs.device).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, forced_token_ids], dim=1)
        for _ in range(max_length):
            outputs = self(input_features=inputs, decoder_input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature for sampling
            next_token_logits = next_token_logits / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            if next_token.item() == self.config.eos_token_id:
                break
        return generated_ids

    def _beam_search_decode(self, inputs, max_length, num_beams, forced_decoder_ids):
        """
        Beam search decoding implementation.
        """
        beam_sequences = [torch.tensor([[self.config.decoder_start_token_id]], device=inputs.device) for _ in range(num_beams)]
        beam_scores = torch.zeros(num_beams, device=inputs.device)
        if forced_decoder_ids:
            forced_token_ids = torch.tensor([token_id for _, token_id in forced_decoder_ids], device=inputs.device).unsqueeze(0)
            beam_sequences = [torch.cat([seq, forced_token_ids], dim=1) for seq in beam_sequences]

        for _ in range(max_length):
            all_candidates = []
            for i in range(num_beams):
                outputs = self(input_features=inputs, decoder_input_ids=beam_sequences[i])
                next_token_logits = outputs.logits[:, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)

                top_k_probs, top_k_tokens = torch.topk(next_token_probs, num_beams, dim=-1)
                for k in range(num_beams):
                    candidate = torch.cat([beam_sequences[i], top_k_tokens[:, k].unsqueeze(-1)], dim=1)
                    candidate_score = beam_scores[i] + torch.log(top_k_probs[:, k])
                    all_candidates.append((candidate_score, candidate))

            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            beam_sequences = [seq for _, seq in ordered[:num_beams]]
            beam_scores = torch.tensor([score for score, _ in ordered[:num_beams]], device=inputs.device)

            if all(seq[0, -1] == self.config.eos_token_id for seq in beam_sequences):
                break

        best_sequence = beam_sequences[0]
        return best_sequence
    
    def AED_suta(self, inputs, args, optimizer, scheduler=None, teacher_token_list=None, max_length=100, generate_text=False, forced_decoder_ids=None, **generation_args):
        """
        SUTA algorithm that implement test-time adaptation (TTA) to single utterence.
        """
        optimizer.zero_grad()
        loss = 0
        e_loss = 0
        p_loss = 0
        num_suta_token = 0
        e_loss_list = []
        generated_ids = torch.tensor([[self.config.decoder_start_token_id]], device=inputs.device)
        if forced_decoder_ids:
            forced_token_ids = torch.tensor([token_id for _, token_id in forced_decoder_ids], device=inputs.device).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, forced_token_ids], dim=1)
        
        if teacher_token_list is None:
            with torch.no_grad():
                teacher_token_list = self.decode(inputs)
        for _ in range(len(teacher_token_list[0])-4):
            num_suta_token+=1
            outputs = self(input_features=inputs, decoder_input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            # Append the predicted next token
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

        topk_logits = torch.topk(outputs.logits, k=args.topk).values.squeeze(0)
        # Don't calculate tag token entropy
        topk_logits = topk_logits[4:, :]
        topk_logits = topk_logits/args.temp
        e_loss = softmax_entropy(topk_logits, dim=1)
        if 'weighted' in args.objective_f:
            e_loss = (1/(1+args.alpha*torch.exp(-e_loss))) * e_loss

        loss = e_loss.mean()
        if 'c_loss' in args.objective_f:
            c_loss = mcc_loss(topk_logits, dim=1, class_num=args.topk)
            loss = loss * args.em_coef + c_loss * (1 - args.em_coef)
        if 'p_loss' in args.objective_f and teacher_token_list is not None:
            criterion = nn.CrossEntropyLoss()
            try:
                p_loss = criterion(outputs.logits[0], teacher_token_list[0,1:])
            except:
                print(outputs.logits[0].shape)
                print(teacher_token_list[0,1:].shape)
            loss += p_loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        output = generated_ids
        if generate_text:
            with torch.no_grad():
                output = self.decode(inputs)
        
        return output, loss, e_loss, p_loss
