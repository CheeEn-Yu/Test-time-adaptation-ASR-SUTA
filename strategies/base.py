from abc import ABC, abstractmethod
import torch
import copy
import numpy as np
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu

class BaseAdaptationStrategy(ABC):
    """Base class for test-time adaptation strategies."""
    
    def __init__(self, name="base"):
        self.name = name
    
    @abstractmethod
    def adapt(self, batch, model, processor, normalizer, args, optimizer, scheduler, count, result_logger):
        """
        Adapt the model using the specific strategy.
        
        Args:
            batch: Input batch with audio, text, etc.
            model: The model to adapt
            processor: Text processor
            normalizer: Text normalizer
            args: Configuration arguments
            optimizer: Model optimizer
            scheduler: Learning rate scheduler
            count: Batch counter
            result_logger: Logger for results
            
        Returns:
            dict: Metrics and results from the adaptation
        """
        pass
    
    def __str__(self):
        return f"AdaptationStrategy({self.name})"