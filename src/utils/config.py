"""Configuration utilities for loading and parsing YAML config files."""

import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command-line arguments
    
    Example:
        >>> args = parse_args()
        >>> print(args.config)
        configs/finetune_config.yaml
    """
    parser = argparse.ArgumentParser(description="JINA Embeddings for Glioma Classification")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/finetune_config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Output directory for model checkpoints and results (overrides config)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        help="Device to use for computation (e.g., 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (overrides config)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    This function reads a YAML configuration file and returns its contents
    as a dictionary. It performs basic validation to ensure the file exists
    and can be parsed correctly.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the parsed configuration
        
    Raises:
        ConfigurationError: If the configuration file doesn't exist or has invalid syntax
        
    Example:
        >>> config = load_config("configs/finetune_config.yaml")
        >>> print(config["model"]["name"])
        jinaai/jina-embeddings-v3
    """
    config_path = Path(config_path)
    
    try:
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found", str(config_path))
        
        with open(config_path, "r") as f:
            try:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            except yaml.YAMLError as e:
                raise ConfigurationError(f"Error parsing YAML: {str(e)}", str(config_path))
    except Exception as e:
        if not isinstance(e, ConfigurationError):
            e = ConfigurationError(f"Unexpected error reading config: {str(e)}", str(config_path))
        logger.error(str(e))
        raise e


def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update configuration with command-line arguments.
    
    Command-line arguments take precedence over configuration file settings.
    
    Args:
        config: Original configuration dictionary
        args: Command-line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Create a deep copy to avoid modifying the original
    updated_config = config.copy()
    
    # Override output directory if specified
    if args.output_dir:
        if "output" not in updated_config:
            updated_config["output"] = {}
        updated_config["output"]["results_dir"] = args.output_dir
        logger.info(f"Using output directory: {args.output_dir}")
    
    # Override device if specified
    if args.device:
        if "training" not in updated_config:
            updated_config["training"] = {}
        updated_config["training"]["device"] = args.device
        logger.info(f"Using device: {args.device}")
    
    # Override GPU IDs if specified
    if args.gpus:
        if "training" not in updated_config:
            updated_config["training"] = {}
        updated_config["training"]["gpus"] = args.gpus
        logger.info(f"Using GPUs: {args.gpus}")
    
    # Override seed if specified
    if args.seed is not None:
        if "global" not in updated_config:
            updated_config["global"] = {}
        updated_config["global"]["seed"] = args.seed
        logger.info(f"Using random seed: {args.seed}")
    
    return updated_config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    # Check required sections
    required_sections = ["model", "training", "data"]
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required section", section)
    
    # Check model config
    if "name" not in config["model"]:
        raise ConfigurationError("Model name is required", "model.name")
    
    # Check training config
    required_training_params = [
        "num_epochs", "train_batch_size", "eval_batch_size",
        "gradient_accumulation_steps", "learning_rate"
    ]
    for param in required_training_params:
        if param not in config["training"]:
            raise ConfigurationError(f"Missing required training parameter", f"training.{param}")
    
    # Check data paths
    for path_key in ["train_path", "eval_path"]:
        if path_key not in config["data"]:
            raise ConfigurationError(f"Missing required data path", f"data.{path_key}")
        
        file_path = config["data"][path_key]
        if not Path(file_path).exists():
            raise ConfigurationError(f"Data file does not exist: {file_path}", f"data.{path_key}")
    
    logger.info("Configuration validated successfully")


def get_config() -> Dict[str, Any]:
    """Parse arguments and load configuration.
    
    This function combines argument parsing, configuration loading, 
    and validation into a single call.
    
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    try:
        args = parse_args()
        config = load_config(args.config)
        config = update_config_with_args(config, args)
        validate_config(config)
        return config
    except ConfigurationError:
        # Re-raise configuration errors
        raise
    except Exception as e:
        # Wrap other exceptions in ConfigurationError
        logger.error(f"Unexpected error loading configuration: {str(e)}")
        raise ConfigurationError(f"Error loading configuration: {str(e)}")