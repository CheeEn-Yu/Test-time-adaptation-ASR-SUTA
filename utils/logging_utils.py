import os
import logging
from datetime import datetime

def configure_logging(exp_name):
    """
    Configure logging for experiments with separate loggers for main logs and results.
    
    Args:
        exp_name (str): Experiment directory name
        
    Returns:
        tuple: (logger, result_logger) - Main logger and results logger
    """
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

def close_loggers(logger, result_logger):
    """
    Properly close file handlers and clean up loggers.
    
    Args:
        logger: Main experiment logger
        result_logger: Result logger
    """
    for handler in logger.handlers + result_logger.handlers:
        handler.close()
        logger.removeHandler(handler)
        result_logger.removeHandler(handler)