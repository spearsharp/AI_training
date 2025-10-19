"""
GPU and hardware utilities.
"""

import torch
from typing import Dict, List, Optional, Any
import logging

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


logger = logging.getLogger(__name__)


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available computing devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cpu_count': None,
        'cpu_memory_gb': None,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': 0,
        'cuda_devices': [],
        'cuda_version': None,
        'cudnn_version': None,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    # CPU info
    if PSUTIL_AVAILABLE:
        info['cpu_count'] = psutil.cpu_count()
        info['cpu_memory_gb'] = psutil.virtual_memory().total / (1024 ** 3)
    
    # CUDA info
    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        
        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),
                'compute_capability': torch.cuda.get_device_capability(i),
            }
            info['cuda_devices'].append(device_info)
    
    return info


def get_optimal_device(device: str = "auto") -> torch.device:
    """
    Get optimal device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "cuda:0", "mps")
        
    Returns:
        torch.device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def print_gpu_utilization():
    """Print current GPU memory utilization."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            logger.info(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            logger.info(f"  Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")
    else:
        logger.info("No CUDA devices available")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU memory cache cleared")


def get_available_gpus() -> List[int]:
    """
    Get list of available GPU IDs.
    
    Returns:
        List of GPU device IDs
    """
    if not torch.cuda.is_available():
        return []
    
    available_gpus = []
    
    if GPUTIL_AVAILABLE:
        # Use GPUtil to get GPUs with low utilization
        gpus = GPUtil.getAvailable(order='memory', limit=10, maxLoad=0.5, maxMemory=0.5)
        available_gpus = gpus
    else:
        # Return all GPUs
        available_gpus = list(range(torch.cuda.device_count()))
    
    return available_gpus


def set_gpu_device(device_id: int):
    """
    Set the active GPU device.
    
    Args:
        device_id: GPU device ID
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        logger.info(f"Set active GPU to device {device_id}")
    else:
        logger.warning("CUDA not available, cannot set GPU device")


def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
    """
    Get memory information for a specific GPU.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        Dictionary with memory information (in GB)
    """
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated': torch.cuda.memory_allocated(device_id) / 1024**3,
        'reserved': torch.cuda.memory_reserved(device_id) / 1024**3,
        'max_allocated': torch.cuda.max_memory_allocated(device_id) / 1024**3,
        'total': torch.cuda.get_device_properties(device_id).total_memory / 1024**3,
    }


def estimate_model_size(num_parameters: int, dtype: str = "float32") -> float:
    """
    Estimate model size in GB.
    
    Args:
        num_parameters: Number of model parameters
        dtype: Data type ("float32", "float16", "int8")
        
    Returns:
        Estimated size in GB
    """
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
    }
    
    bytes_total = num_parameters * bytes_per_param.get(dtype, 4)
    return bytes_total / (1024 ** 3)