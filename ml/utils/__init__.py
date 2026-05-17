"""Utility functions for ML pipeline."""

import logging
import random
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('ml_pipeline')
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> str:
    """Get available device (cuda/cpu)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: dict, path: Path) -> None:
    """Save data as JSON."""
    import json
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    """Load data from JSON."""
    import json
    
    with open(path, 'r') as f:
        return json.load(f)


def get_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def format_time(seconds: float) -> str:
    """Format seconds to readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.elapsed = None
        
    def __enter__(self):
        self.start = __import__('time').time()
        return self
        
    def __exit__(self, *args):
        self.elapsed = __import__('time').time() - self.start
        logging.getLogger('ml_pipeline').info(f"{self.name} took {format_time(self.elapsed)}")


class DeviceManager:
    """Manage GPU memory and device selection."""
    
    @staticmethod
    def get_device() -> str:
        """Get best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    @staticmethod
    def clear_cache() -> None:
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    @staticmethod
    def get_memory_info() -> dict:
        """Get GPU memory info."""
        if not torch.cuda.is_available():
            return {'available': False}
            
        return {
            'available': True,
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'reserved': torch.cuda.memory_reserved() / 1e9,
            'total': torch.cuda.get_device_properties(0).total_memory / 1e9
        }
    
    @staticmethod
    def set_device(device_id: int = 0) -> None:
        """Set CUDA device."""
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)