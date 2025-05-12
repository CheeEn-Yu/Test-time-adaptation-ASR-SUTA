import os
import logging
import torch
import hydra
from omegaconf import OmegaConf
from whisper.normalizers import EnglishTextNormalizer
from datetime import datetime
from transformers import AutoProcessor
from tqdm import tqdm

from utils.data import load_SUTAdataset
from suta import WhisperTTADecoder, hf_collect_params
from strategies.registry import strategy_registry

logging.getLogger("accelerate.utils.modeling").setLevel(logging.ERROR)

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DECODER_STEP = 512

# Configure logging
def configure_logging(exp_name):
    # Create experiment directory if not exists
    os.makedirs(exp_name, exist_ok=True)
    os.makedirs(os.path.join(exp_name, 'figs'), exist_ok=True)
    
    # Main logger configuration
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    # Results logger configuration
    result_logger = logging.getLogger('results')
    result_logger.setLevel(logging.INFO)
    result_logger.propagate = False  # Prevent propagation to root logger
    result_logger.handlers = []  # Clear any existing handlers

    # File handlers
    log_file = os.path.join(exp_name, 'log.txt')
    result_file = os.path.join(exp_name, 'result.txt')

    # Formatters
    main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    result_formatter = logging.Formatter('%(message)s')

    # Handlers
    main_handler = logging.FileHandler(log_file)
    main_handler.setFormatter(main_formatter)
    
    result_handler = logging.FileHandler(result_file)
    result_handler.setFormatter(result_formatter)

    # Add handlers
    logger.addHandler(main_handler)
    result_logger.addHandler(result_handler)

    return logger, result_logger

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    
def setup_experiment(args, logger):
    config_str = OmegaConf.to_yaml(args)
    start_time = datetime.now()
    logger.info(f'Experiment started at: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('Configuration:\n' + config_str)
    return start_time

def log_model_info(model, logger):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_ratio = trainable_params / total_params
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    logger.info(f'Trainable parameter ratio: {train_ratio:.4f}')

def setup_optimizer(args, params, opt_name='AdamW', lr=1e-4, weight_decay=0., scheduler=None):
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
                current_sample = int(line.split('idx:')[1].split(' ')[0])
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

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args):
    # Initialize logging
    exp_suffix = f"_{args.strategy}" if hasattr(args, "strategy") else ""
    args.exp_name = args.exp_name if args.exp_name else f'ex_data/{args.asr.split("/")[-1]}_{args.task}_{args.lang}{exp_suffix}'
    logger, result_logger = configure_logging(args.exp_name)
    set_seed(args.seed)
    start_time = setup_experiment(args, logger)
    
    # List available strategies
    available_strategies = strategy_registry.list_strategies()
    logger.info(f"Available adaptation strategies: {available_strategies}")
    
    # Get selected strategy
    strategy_name = args.strategy if hasattr(args, "strategy") else "choose_ln"
    try:
        strategy = strategy_registry.get(strategy_name)
        logger.info(f"Using adaptation strategy: {strategy.name}")
    except ValueError as e:
        logger.error(f"Error: {str(e)}")
        logger.info(f"Using default strategy 'choose_ln' instead")
        strategy = strategy_registry.get("choose_ln")
    
    # Load dataset and model
    normalizer = EnglishTextNormalizer()
    dataset = load_SUTAdataset(
        name=args.dataset_name,
        path=args.dataset_dir,
        batch_size=1,
        lang=args.lang,
        noise_dir=args.noise_dir,
        snr=args.snr
    )
    
    # Initial model for parameter logging
    model = WhisperTTADecoder.from_pretrained(args.asr, device_map='auto')
    processor = AutoProcessor.from_pretrained(args.asr)
    
    # Log model information
    logger.info(f"Model: {args.asr}")
    log_model_info(model, logger)
    
    # Process each batch
    for count, batch in tqdm(enumerate(dataset), total=len(dataset)):
        if args.num_data and count >= args.num_data:
            break
            
        # Reload model for each sample to avoid accumulated adaptation
        model = WhisperTTADecoder.from_pretrained(args.asr, device_map='auto')
        
        # Initialize optimizer
        optimizer, scheduler = setup_optimizer(
            args,
            hf_collect_params(model, args.encoderLN, args.decoderLN)[0],
            args.opt,
            args.lr,
            weight_decay=1e-5,
            scheduler=args.scheduler
        )
        
        # Apply selected adaptation strategy
        strategy.adapt(
            batch, model, processor, normalizer,
            args, optimizer, scheduler, count, result_logger
        )

    # Process results
    try:
        wer_processor = transcriptionProcessor(task=args.task)
        wer_processor.process_file(f'{args.exp_name}/result.txt')
        wer_list = wer_processor.step_mean_wer()
        
        logger.info(f"Final Results (Strategy: {strategy.name}):")
        for result in wer_list:
            logger.info(result)
            
    except Exception as e:
        logger.error(f"Failed to process results log: {str(e)}")

    # Finalize experiment
    end_time = datetime.now()
    logger.info(f'Experiment completed at: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Total running time: {end_time - start_time}')
    
    # Close loggers
    for handler in logger.handlers + result_logger.handlers:
        handler.close()
        logger.removeHandler(handler)
        result_logger.removeHandler(handler)

if __name__ == '__main__':
    main()