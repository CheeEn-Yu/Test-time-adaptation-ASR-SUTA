import matplotlib.pyplot as plt
import os

def plot_losses(step_loss, p_loss_list, count, args):
    """
    Plot loss curves for training steps.
    
    Args:
        step_loss (list): List of loss values per step
        p_loss_list (list): List of p_loss values per step if applicable
        count (int): Batch/sample counter
        args: Configuration arguments with experiment settings
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss', color='tab:red')
    ax.plot(step_loss, color='tab:red', marker='o')
    
    if p_loss_list and 'p_loss' in args.objective_f:
        ax2 = ax.twinx()
        ax2.set_ylabel('P Loss', color='tab:blue')
        ax2.plot(p_loss_list, color='tab:blue', marker='o')
    
    plt.title(f'Loss Trajectory - Sample {count}')
    plt.savefig(os.path.join(args.exp_name, 'figs', f'suta_{count}.png'))
    plt.close()

def plot_metrics(metrics_dict, steps, count, args, title=None):
    """
    Generic function to plot multiple metrics during adaptation.
    
    Args:
        metrics_dict (dict): Dictionary of metric name to values list
        steps (list): Step indices
        count (int): Sample counter
        args: Configuration arguments
        title (str, optional): Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        color = colors[i % len(colors)]
        if i == 0:
            axis = ax
        else:
            axis = ax.twinx()
            # Offset the right spine for multiple metrics
            axis.spines['right'].set_position(('outward', 60 * (i-1)))
        
        axis.set_ylabel(metric_name, color=color)
        axis.plot(steps, values, color=color, marker='o', label=metric_name)
        axis.tick_params(axis='y', labelcolor=color)
    
    plt.title(title or f'Metrics - Sample {count}')
    plt.savefig(os.path.join(args.exp_name, 'figs', f'metrics_{count}.png'))
    plt.close()