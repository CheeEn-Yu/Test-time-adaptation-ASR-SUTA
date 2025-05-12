import torch
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_experiment(args, logger):
    """
    Set up experiment, log configuration and start time.
    
    Args:
        args: Configuration arguments
        logger: Logger instance
        
    Returns:
        datetime: Experiment start time
    """
    config_str = OmegaConf.to_yaml(args)
    start_time = datetime.now()
    logger.info(f'Experiment started at: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('Configuration:\n' + config_str)
    return start_time

def find_topk_norm_layers(diff_dict, k=3):
    """
    Find top-k layers with highest norm changes.
    
    Args:
        diff_dict (dict): Dictionary of layer name to parameter difference vectors
        k (int): Number of top layers to return
        
    Returns:
        dict: Dictionary of top-k layer names to their norms
    """
    norms = {key: np.linalg.norm(vec) for key, vec in diff_dict.items()}
    sorted_layers = sorted(norms.items(), key=lambda x: x[1], reverse=True)
    
    topk_layers = sorted_layers[:k]
    return {name: norm for name, norm in topk_layers}