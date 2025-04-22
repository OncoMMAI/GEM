# -*- coding: utf-8 -*-

"""
Generate embeddings for glioma classification using various models.

This module provides utilities to generate embeddings from clinical text reports
using different embedding models including JINA, NVIDIA NV-Embed, and Gatortron.
These embeddings are used for downstream classification tasks in glioma prognostication.
The implementation is aligned with the approach used in notebook 03_get_embeddings.ipynb.
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModel, AutoConfig, AutoTokenizer

from src.utils.config import get_config
from src.utils.logging import setup_logger, disable_external_loggers

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def jina_embed(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    text_col: str = "report",
    task: Optional[str] = None,
    model_name: str = "jinaai/jina-embeddings-v3",
    checkpoint_path: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 2048,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
) -> np.ndarray:
    """Generate embeddings using Jina model.

    Args:
        df: Input dataframe containing text
        output_path: Path to save embeddings
        text_col: Column name containing text
        task: Optional task type for encoding ('text-matching', etc.)
        model_name: Name or path of the model to use
        checkpoint_path: Optional path to model checkpoint
        batch_size: Batch size for processing
        max_length: Maximum sequence length for encoding
        device: Device to run model on
        trust_remote_code: Whether to trust remote code in the model

    Returns:
        numpy array of embeddings
    """
    logger = logging.getLogger("glioma.embeddings")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading JINA model: {model_name}")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    ).to(device)

    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_file(checkpoint_path)
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    embeddings_list = []

    logger.info(
        f"Generating embeddings for {len(df)} texts using batch size {batch_size}"
    )
    for i in tqdm(range(0, len(df), batch_size), desc="Generating JINA embeddings"):
        batch_texts = df[text_col].iloc[i : i + batch_size].tolist()

        with torch.no_grad():
            encode_kwargs = {
                "max_length": max_length,
                "device": device,
                "show_progress_bar": False,
            }
            if task:
                encode_kwargs["task"] = task

            embeddings = model.encode(batch_texts, **encode_kwargs)
            embeddings_list.append(embeddings)

    embeddings = np.vstack(embeddings_list)

    # Save embeddings
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    logger.info(f"Saved to {output_path}")

    return embeddings


def nvidia_embed(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    text_col: str = "report",
    model_name: str = "nvidia/NV-Embed-v2",
    batch_size: int = 16,
    max_length: int = 32768,
    device: Optional[str] = None,
    device_ids: Optional[List[int]] = None,
    trust_remote_code: bool = True,
    instruction: Optional[str] = None,
) -> np.ndarray:
    """Generate embeddings using NVIDIA NV-Embed model.

    Args:
        df: Input dataframe containing text
        output_path: Path to save embeddings
        text_col: Column name containing text
        model_name: Name or path of the model to use
        batch_size: Batch size for processing
        max_length: Maximum sequence length for encoding
        device: Device to run model on
        device_ids: List of GPU device IDs for multi-GPU support
        trust_remote_code: Whether to trust remote code in the model
        instruction: Optional instruction template ('medical', 'none', or custom)

    Returns:
        numpy array of embeddings
    """
    logger = logging.getLogger("glioma.embeddings")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading NVIDIA model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    if device_ids:
        logger.info(f"Using multiple GPUs: {device_ids}")
        for module_key, module in model._modules.items():
            model._modules[module_key] = DataParallel(module, device_ids=device_ids)
        model.to(f"cuda:{device_ids[0]}")
    else:
        model.to(device)

    # Prepare instruction template if specified
    instruction_prefix = ""
    if instruction:
        instruction_templates = {
            "none": "",
            "medical": """Instruct: Encode medical reports and capture key diagnostic findings, disease characteristics, and prognostic indicators.
Query: """,
        }
        instruction_prefix = instruction_templates.get(instruction, instruction)
        logger.info(f"Using instruction template: '{instruction}'")

    embeddings = []
    original_indices = []

    logger.info(
        f"Generating embeddings for {len(df)} texts using batch size {batch_size}"
    )
    for i in tqdm(range(0, len(df), batch_size), desc="Generating NV-Embed embeddings"):
        batch_indices = list(range(i, min(i + batch_size, len(df))))
        batch = df[text_col].iloc[batch_indices].tolist()
        original_indices.extend(batch_indices)

        with torch.no_grad():
            encode_kwargs = {
                "max_length": max_length,
                "show_progress_bar": False,
            }

            if instruction_prefix:
                encode_kwargs["instruction"] = instruction_prefix

            batch_embeddings = model.encode(batch, **encode_kwargs)

            # Normalize embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings)

    all_embeddings = torch.cat(embeddings, dim=0)

    # Verify integrity
    assert len(all_embeddings) == len(df), (
        "Embeddings length doesn't match dataframe length"
    )
    assert original_indices == list(range(len(df))), "Order mismatch detected"

    # Convert to numpy and save
    numpy_embeddings = all_embeddings.cpu().numpy()

    # Save embeddings
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, numpy_embeddings)

    logger.info(f"Generated embeddings shape: {numpy_embeddings.shape}")
    logger.info(f"Saved to {output_path}")

    return numpy_embeddings


def gatortron_embed(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    text_col: str = "report",
    model_name: str = "UFNLP/gatortron-large",
    checkpoint_path: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
) -> np.ndarray:
    """Generate embeddings using Gatortron model.

    Args:
        df: Input dataframe containing text
        output_path: Path to save embeddings
        text_col: Column name containing text
        model_name: Name or path of the model to use
        checkpoint_path: Optional path to model checkpoint
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run model on
        trust_remote_code: Whether to trust remote code in the model

    Returns:
        numpy array of embeddings
    """
    logger = logging.getLogger("glioma.embeddings")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading Gatortron tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    logger.info(f"Loading Gatortron model configuration")
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    logger.info(f"Loading Gatortron model")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        config=config,
    ).to(device)

    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_file(checkpoint_path)
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    embeddings_list = []

    logger.info(
        f"Generating embeddings for {len(df)} texts using batch size {batch_size}"
    )
    for i in tqdm(
        range(0, len(df), batch_size), desc="Generating Gatortron embeddings"
    ):
        batch_texts = df[text_col].iloc[i : i + batch_size].tolist()

        with torch.no_grad():
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            # Extract embeddings from the [CLS] token
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings_list.append(embeddings)

    embeddings = np.vstack(embeddings_list)

    # Save embeddings
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    logger.info(f"Saved to {output_path}")

    return embeddings


def generate_embeddings(config: Dict) -> None:
    """Generate embeddings for all models and tasks.

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger("glioma.embeddings")

    # Create output directory
    output_dir = Path(config.get("output", {}).get("output_dir", "embeddings"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Embeddings will be saved to: {output_dir}")

    # Load datasets
    data_config = config["data"]
    text_column = data_config.get("text_column", "report")

    datasets = {}
    for task in ["tumor_type", "survival", "mgmt"]:
        path_key = f"{task}_path"
        if path_key in data_config:
            file_path = data_config[path_key]
            logger.info(f"Loading {task} dataset from {file_path}")
            try:
                datasets[task] = pd.read_csv(file_path)
                logger.info(f"Loaded {len(datasets[task])} records for {task}")
            except Exception as e:
                logger.error(f"Failed to load {task} dataset: {e}")

    # Process each task and model combination
    for task_name, df in datasets.items():
        task_dir = output_dir / task_name
        task_dir.mkdir(exist_ok=True)

        # JINA model
        if "jina" in config["model"]:
            jina_config = config["model"]["jina"]
            logger.info(f"Generating JINA embeddings for {task_name} task")

            output_path = task_dir / f"jina_embeddings.npy"
            if jina_config.get("checkpoint_path"):
                output_path = task_dir / f"jina_finetuned.npy"

            jina_embed(
                df=df,
                output_path=output_path,
                text_col=text_column,
                task=task_name,
                model_name=jina_config.get("name", "jinaai/jina-embeddings-v3"),
                checkpoint_path=jina_config.get("checkpoint_path"),
                batch_size=jina_config.get("batch_size", 32),
                max_length=jina_config.get("max_length", 2048),
                trust_remote_code=jina_config.get("trust_remote_code", True),
            )

        # NVIDIA model
        if "nvidia" in config["model"]:
            nvidia_config = config["model"]["nvidia"]
            logger.info(f"Generating NVIDIA embeddings for {task_name} task")

            output_path = task_dir / f"nv_embed.npy"
            if nvidia_config.get("instruction"):
                output_path = (
                    task_dir / f"nv_embed_{nvidia_config.get('instruction')}.npy"
                )

            nvidia_embed(
                df=df,
                output_path=output_path,
                text_col=text_column,
                model_name=nvidia_config.get("name", "nvidia/NV-Embed-v2"),
                batch_size=nvidia_config.get("batch_size", 16),
                max_length=nvidia_config.get("max_length", 32768),
                device_ids=nvidia_config.get("device_ids"),
                instruction=nvidia_config.get("instruction"),
                trust_remote_code=nvidia_config.get("trust_remote_code", True),
            )

        # Gatortron model
        if "gatortron" in config["model"]:
            gatortron_config = config["model"]["gatortron"]
            logger.info(f"Generating Gatortron embeddings for {task_name} task")

            output_path = task_dir / f"gatortron_large.npy"

            gatortron_embed(
                df=df,
                output_path=output_path,
                text_col=text_column,
                model_name=gatortron_config.get("name", "UFNLP/gatortron-large"),
                checkpoint_path=gatortron_config.get("checkpoint_path"),
                batch_size=gatortron_config.get("batch_size", 32),
                max_length=gatortron_config.get("max_length", 512),
                trust_remote_code=gatortron_config.get("trust_remote_code", True),
            )


def main() -> None:
    """Main entry point for embedding generation."""
    # Setup logger
    logger = setup_logger(
        name="glioma.embeddings", level=logging.INFO, log_file="embeddings.log"
    )

    # Disable verbose logging from external libraries
    disable_external_loggers()

    # Load configuration
    config = get_config()

    # Set CUDA devices if specified in config
    if "gpus" in config.get("training", {}):
        os.environ["CUDA_VISIBLE_DEVICES"] = config["training"]["gpus"]

    # Generate embeddings
    generate_embeddings(config)

    logger.info("Embedding generation completed successfully!")


if __name__ == "__main__":
    main()
