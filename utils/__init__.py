from .logging_utils import configure_logging, close_loggers
from .visualization import plot_losses, plot_metrics
from .experiment import set_seed, setup_experiment, find_topk_norm_layers
from .model_utils import log_model_info, setup_optimizer
from .metrics import softmax_entropy, mcc_loss, transcriptionProcessor
from .data import (
    load_SUTAdataset, collect_audio_batch, multi_collate_fn,
    get_dataset, get_dataloader, get_processor
)

# Constants
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DECODER_STEP = 512