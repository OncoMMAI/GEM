"""Utilities for setting random seeds consistently across the codebase."""

import random
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: Optional[int] = 42) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    This function ensures deterministic behavior by setting random seeds
    for all relevant libraries and enabling deterministic algorithms
    in PyTorch. Using this function helps make experiments reproducible.
    
    Args:
        seed: Random seed value. If None, randomization is used.
        
    Note:
        When using CUDA, some operations may still have non-deterministic
        behavior. See PyTorch documentation for details on achieving full
        determinism.
    """
    if seed is None:
        logger.warning("No random seed set - results may not be reproducible")
        return
    
    logger.info(f"Setting random seed to {seed}")
    
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional deterministic settings for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for other libraries
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    logger.info("Random seed set successfully across all libraries")