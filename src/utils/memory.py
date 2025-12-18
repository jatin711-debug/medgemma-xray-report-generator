"""
Utility functions for system memory monitoring.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def get_system_memory_info() -> str:
    """
    Retrieve system memory information for monitoring and optimization.
    
    Returns detailed memory usage for either GPU (if CUDA available) or CPU.
    This information is crucial for monitoring resource usage and preventing
    out-of-memory errors during model inference.
    
    Returns:
        str: Formatted string containing memory information
        
    Note:
        Requires psutil package for CPU memory monitoring on non-CUDA systems.
    """
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            return f"GPU Memory: {gpu_memory_gb:.1f}GB total, {allocated:.1f}GB allocated"
        else:
            import psutil
            ram = psutil.virtual_memory()
            return f"RAM: {ram.total / (1024**3):.1f}GB total, {ram.available / (1024**3):.1f}GB available"
    except Exception as e:
        return f"Memory info unavailable: {str(e)}"
