import hydra
from whisper.normalizers import EnglishTextNormalizer
from transformers import AutoProcessor
from tqdm import tqdm
import logging
from datetime import datetime

# Configure acceleration logging
logging.getLogger("accelerate.utils.modeling").setLevel(logging.ERROR)

# Import custom modules
from suta import WhisperTTADecoder, hf_collect_params
from utils import (
    load_SUTAdataset,
    configure_logging, close_loggers, set_seed, setup_experiment, 
    log_model_info, setup_optimizer, transcriptionProcessor, DEVICE
)
from strategies.registry import strategy_registry

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(args):
    # Initialize experiment
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
    close_loggers(logger, result_logger)

if __name__ == '__main__':
    main()