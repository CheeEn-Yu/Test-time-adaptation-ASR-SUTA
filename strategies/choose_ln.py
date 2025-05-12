import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu

from .base import BaseAdaptationStrategy

class ChooseLNStrategy(BaseAdaptationStrategy):
    """Strategy that adapts by selecting top-k layer norms to update."""
    
    def __init__(self, k=3, phase_ratio=0.33):
        super().__init__(name="choose_ln")
        self.k = k
        self.phase_ratio = phase_ratio  # First phase ratio of total steps
    
    def find_topk_norm_layers(self, diff_dict, k=3):
        """Find top-k layers with highest norm changes."""
        norms = {key: np.linalg.norm(vec) for key, vec in diff_dict.items()}
        sorted_layers = sorted(norms.items(), key=lambda x: x[1], reverse=True)
        
        topk_layers = sorted_layers[:k]
        topk_layers = {name: norm for name, norm in topk_layers}
        return topk_layers
    
    def plot_losses(self, step_loss, p_loss_list, count, args):
        """Plot loss curves."""
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss', color='tab:red')
        ax.plot(step_loss, color='tab:red', marker='o')
        
        if 'p_loss' in args.objective_f:
            ax2 = ax.twinx()
            ax2.set_ylabel('P Loss', color='tab:blue')
            ax2.plot(p_loss_list, color='tab:blue', marker='o')
        
        plt.title(f'Loss Trajectory - Sample {count} ({self.name})')
        plt.savefig(f'{args.exp_name}/figs/suta_{count}_{self.name}.png')
        plt.close()
    
    def adapt(self, batch, model, processor, normalizer, args, optimizer, scheduler, count, result_logger):
        c_loss_list, p_loss_list, step_loss = [], [], []
        lens, wavs, texts, files = batch
        
        result_logger.info(f'idx:{count} - Strategy: {self.name}')
        label = normalizer(texts[0])
        result_logger.info(f'label:{label}')

        inputs = processor(wavs[0], sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)

        # Original transcription
        teacher_token_list = model.decode(
            input_features, 
            forced_decoder_ids=processor.get_decoder_prompt_ids(
                language=args.asr_lang, 
                task=args.task
            )
        )
        transcription = processor.batch_decode(teacher_token_list, skip_special_tokens=True)[0]
        transcription = normalizer(transcription)
        error_metric = wer(label, transcription) if args.task == "transcribe" else sentence_bleu([label], transcription)
        result_logger.info(f'ori({error_metric:.5f}):{transcription}')

        # Copy original parameters
        pre_adapt_state_dict = copy.deepcopy(model.state_dict())

        if args.tta:
            # Phase 1: Identify important layers
            choose_layers_step = int(args.steps * self.phase_ratio)
            for step in range(choose_layers_step):
                outputs, loss, e_loss, p_loss = model.AED_suta(
                    input_features, args, optimizer,
                    teacher_token_list=teacher_token_list,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language=args.asr_lang,
                        task=args.task
                    ),
                    generate_text=False
                )
            
            # Calculate parameter changes
            post_adapt_state_dict = copy.deepcopy(model.state_dict())
            layer_norm_diff = {}
            relative_changes = {}
            for name in pre_adapt_state_dict:
                if 'layer_norm' in name:
                    diff = post_adapt_state_dict[name] - pre_adapt_state_dict[name]
                    layer_norm_diff[name] = diff.cpu().numpy().flatten()
                    original = pre_adapt_state_dict[name].cpu().numpy().flatten()
                    original_norm = np.linalg.norm(original)
                    diff_norm = np.linalg.norm(layer_norm_diff[name])
                    # Calculate relative change (avoid division by zero)
                    if original_norm > 0:
                        relative_changes[name] = diff_norm / original_norm
                    else:
                        relative_changes[name] = diff_norm  # Fallback to absolute change
                        
            topk_layers = self.find_topk_norm_layers(relative_changes, k=args.topk_layer)
            result_logger.info(f'Topk layers: {topk_layers}')
            
            # Phase 2: Fine-tune selected layers only
            for param in model.parameters():
                param.requires_grad = False
                
            params, names = [], []
            for name, param in model.named_parameters():
                if name in topk_layers.keys():
                    param.requires_grad = True
                    params.append(param)
                    names.append(name)
                    result_logger.info(f'Choose layer {name}')
                    
            optimizer, scheduler = self._setup_optimizer(
                args,
                params,
                args.opt,
                args.lr_scale * args.lr,
                weight_decay=1e-5,
                scheduler=args.scheduler
            )
                
            # Continue adaptation with selected layers
            for step in range(choose_layers_step, args.steps):
                outputs, loss, e_loss, p_loss = model.AED_suta(
                    input_features, args, optimizer,
                    teacher_token_list=teacher_token_list,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language=args.asr_lang,
                        task=args.task
                    ),
                    generate_text=(step % 3 == 0 or step == args.steps-1)
                )
                
                if step % 3 == 0 or step == args.steps-1:
                    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    transcription = normalizer(transcription)
                    adapt_error = wer(label, transcription) if args.task == "transcribe" else sentence_bleu([label], transcription)
                    result_logger.info(f'step{step}({adapt_error:.5f}): {transcription}')
                
                step_loss.append(loss.item())
                if 'p_loss' in args.objective_f:
                    p_loss_list.append(p_loss.item())

            self.plot_losses(step_loss, p_loss_list, count, args)
            result_logger.info("=" * 40)
        
        return {
            "step_loss": step_loss,
            "p_loss_list": p_loss_list
        }
        
    def _setup_optimizer(self, args, params, opt_name='AdamW', lr=1e-4, weight_decay=0., scheduler=None):
        """Set up optimizer and scheduler."""
        opt = getattr(torch.optim, opt_name)
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)

        if scheduler is not None:
            if scheduler == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.t_max, eta_min=args.lr_min
                )
            else:
                scheduler = eval(f"torch.optim.lr_scheduler.{scheduler}")(
                    optimizer, T_max=args.t_max, eta_min=args.lr_min
                )
            return optimizer, scheduler
        else:
            return optimizer, None