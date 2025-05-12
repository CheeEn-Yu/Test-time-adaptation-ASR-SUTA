import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

def log_model_info(model, logger):
    """
    Log model parameter statistics.
    
    Args:
        model: PyTorch model
        logger: Logger instance
    """
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_ratio = trainable_params / total_params
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    logger.info(f'Trainable parameter ratio: {train_ratio:.4f}')

def setup_optimizer(args, params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, verbose=False):
    """
    Set up optimizer and scheduler.
    
    Args:
        args: Configuration arguments
        params: Model parameters to optimize
        opt_name (str): Optimizer name
        lr (float): Learning rate
        beta (float): Beta parameter for Adam
        weight_decay (float): Weight decay factor
        scheduler (str, optional): Scheduler name
        verbose (bool): Whether to print setup information
        
    Returns:
        tuple: (optimizer, scheduler) or (optimizer, None) if no scheduler
    """
    opt = getattr(torch.optim, opt_name)
    if verbose:
        print(f'[INFO]    optimizer: {opt}')
        print(f'[INFO]    scheduler: {scheduler}')
        
    if opt_name == 'Adam':       
        optimizer = opt(params, lr=lr, betas=(beta, 0.999), weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)

    if scheduler is not None:
        if scheduler == 'CosineAnnealingLR':
            return optimizer, CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.lr_min)
        else:
            return optimizer, eval(scheduler)(optimizer, T_max=args.t_max, eta_min=args.lr_min)
    else:
        return optimizer, None