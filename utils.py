import os
import time
import logging
import platform
import psutil
import torch
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("stable-audio")

def set_log_level(level):
    """Set the logging level.
    
    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(numeric_level)
    logger.info(f"Log level set to {level}")

def get_system_info():
    """Get system information for debugging."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "ram": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB",
        })
    
    return info

def print_system_info():
    """Print system information in a formatted way."""
    info = get_system_info()
    logger.info("=== System Information ===")
    for key, value in info.items():
        logger.info(f"{key}: {value}")
    logger.info("=========================")

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    memory_usage = {
        "rss": f"{memory_info.rss / (1024**3):.2f} GB",  # Resident Set Size
        "vms": f"{memory_info.vms / (1024**3):.2f} GB",  # Virtual Memory Size
    }
    
    if torch.cuda.is_available():
        memory_usage.update({
            "gpu_allocated": f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB",
            "gpu_reserved": f"{torch.cuda.memory_reserved() / (1024**3):.2f} GB",
        })
    
    return memory_usage

def log_memory_usage(message="Current memory usage"):
    """Log current memory usage with a custom message."""
    memory_usage = get_memory_usage()
    logger.debug(f"{message}:")
    for key, value in memory_usage.items():
        logger.debug(f"  {key}: {value}")

class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.name:
            logger.debug(f"Timer '{self.name}': {elapsed:.4f} seconds")
        return False
    
    @property
    def elapsed(self):
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time

def timeit(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper
