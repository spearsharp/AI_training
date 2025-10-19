"""
Random seed utilities for reproducibility.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch 1.8+ deterministic algorithms
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # Some operations don't support deterministic mode


def get_seed() -> int:
    """Get current random seed."""
    return torch.initial_seed()