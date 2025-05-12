import torch
import torch.nn.functional as F
from jiwer import wer

def softmax_entropy(x, dim=2):
    """
    Calculate the entropy of softmax distribution from logits.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute softmax
        
    Returns:
        torch.Tensor: Entropy values
    """
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

def mcc_loss(x, reweight=False, dim=2, class_num=32):
    """
    Maximum Coding Coherence Loss.
    
    Args:
        x: Input tensor
        reweight (bool): Whether to reweight
        dim (int): Dimension for softmax
        class_num (int): Number of classes to consider
        
    Returns:
        torch.Tensor: MCC loss value
    """
    prob = F.softmax(x, dim=dim)
    mask = torch.zeros_like(prob)
    mask[:, :, :class_num] = 1.
    prob = prob * mask
    prob = prob / (prob.sum(dim=dim, keepdim=True) + 1e-12)
    
    c_prob = prob.mean(dim=1)  # coherence of prediction
    coherence = torch.bmm(prob, c_prob.unsqueeze(-1))
    loss = 1 - coherence.mean()
    
    return loss

class transcriptionProcessor:
    """Process transcription results and calculate WER statistics."""
    
    def __init__(self, task="transcribe"):
        self.ori_wers = []
        self.step_wers = {}
        self.labels = []
        self.task = task
    
    def process_file(self, file_path):
        """Process result file and extract WER values."""
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        current_sample = None
        for line in lines:
            line = line.strip()
            if line.startswith('idx:'):
                # current_sample = int(line.split('idx:')[1].split(' ')[0])
                pass
            elif line.startswith('ori('):
                wer_value = float(line[line.find('(')+1:line.find(')')])
                self.ori_wers.append(wer_value)
            elif line.startswith('step') and '(' in line and ')' in line:
                step = int(line[4:line.find('(')])
                wer_value = float(line[line.find('(')+1:line.find(')')])
                
                if step not in self.step_wers:
                    self.step_wers[step] = []
                self.step_wers[step].append(wer_value)
    
    def step_mean_wer(self):
        """Calculate mean WER for each step."""
        results = []
        ori_mean = sum(self.ori_wers) / len(self.ori_wers) if self.ori_wers else 0
        results.append(f"Original mean {'WER' if self.task == 'transcribe' else 'BLEU'}: {ori_mean:.5f}")
        
        for step, values in sorted(self.step_wers.items()):
            mean_value = sum(values) / len(values) if values else 0
            results.append(f"Step {step} mean {'WER' if self.task == 'transcribe' else 'BLEU'}: {mean_value:.5f}")
            
        return results