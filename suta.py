import numpy as np
import re
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from jiwer import wer

def my_greedy_decode(model, input_features, max_step):
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

def HF_collect_params(args, model):
    model.requires_grad_(False)
    params = []
    names = []
    for name, param in model.named_parameters():
        if 'feature' in args.train_params:
            if 'conv' in str(name).split('.'):
                param.requires_grad = True
                params.append(param)
                names.append(f"{name}")
        if 'LN' in args.train_params:
            if 'self_attn_layer_norm' in str(name).split('.'):
                param.requires_grad = True
                params.append(param)
                names.append(f"{name}")

    return names, params

def SB_collect_params(model, bias_only=False, train_feature=False, train_all=False, train_LN=True):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    trainable = []
    if bias_only:
        trainable = ['bias']
    else: 
        trainable = ['weight', 'bias']

    
    for nm, m in model.named_modules():
        if train_LN: 
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in trainable:  
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if train_feature:
            if len(str(nm).split('.')) > 1:
                if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                    for np, p in m.named_parameters():
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        
        if train_all: 
            for np, p in m.named_parameters():
                p.requires_grad = True
                params.append(p)
                names.append(f"{nm}.{np}")
            

    return params, names
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



