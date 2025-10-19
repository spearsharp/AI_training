"""
Early stopping callback for training.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to stop training when a metric stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' - whether to minimize or maximize the metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode {mode} is unknown")
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            logger.info(f"Early stopping counter reset. Best score: {self.best_score:.6f}")
        else:
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered! Best score: {self.best_score:.6f}")
                return True
        
        return False
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False