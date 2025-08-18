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
    cache_dir: Optional[str] = None,
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
        cache_dir: Directory to cache downloaded models

    Returns:
        numpy array of embeddings
    """
    logger = logging.getLogger("glioma.embeddings")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading JINA model: {model_name}")
    model_kwargs = {"trust_remote_code": trust_remote_code}

    if cache_dir:
        logger.info(f"Using cache directory: {cache_dir}")
        model_kwargs["cache_dir"] = cache_dir

    model = AutoModel.from_pretrained(
        model_name,
        **model_kwargs,
    ).to(device)

    if checkpoint_path:
        # Allow passing either a file path or a directory containing the underlying model weights
        candidate_path = Path(checkpoint_path)
        resolved_ckpt: Optional[Path] = None
        if candidate_path.is_file():
            resolved_ckpt = candidate_path
        elif candidate_path.is_dir():
            # Common SentenceTransformer save layout: <dir>/0_Transformer/model.safetensors
            st_layout = candidate_path / "0_Transformer" / "model.safetensors"
            if st_layout.exists():
                resolved_ckpt = st_layout
            else:
                # Fallback: search up to a couple of levels for a model.safetensors
                for p in candidate_path.rglob("model.safetensors"):
                    resolved_ckpt = p
                    break

        if resolved_ckpt and resolved_ckpt.exists():
            logger.info(f"Loading checkpoint from {resolved_ckpt}")
            checkpoint = load_file(str(resolved_ckpt))
            model.load_state_dict(checkpoint, strict=False)
        else:
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}. Proceeding with base model weights."
            )

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
    cache_dir: Optional[str] = None,
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
        cache_dir: Directory to cache downloaded models

    Returns:
        numpy array of embeddings
    """
    logger = logging.getLogger("glioma.embeddings")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading NVIDIA model: {model_name}")
    model_kwargs = {"trust_remote_code": trust_remote_code}

    if cache_dir:
        logger.info(f"Using cache directory: {cache_dir}")
        model_kwargs["cache_dir"] = cache_dir

    model = AutoModel.from_pretrained(model_name, **model_kwargs)

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
    cache_dir: Optional[str] = None,
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
        cache_dir: Directory to cache downloaded models

    Returns:
        numpy array of embeddings
    """
    logger = logging.getLogger("glioma.embeddings")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare common kwargs for model loading
    model_kwargs = {"trust_remote_code": trust_remote_code}
    if cache_dir:
        logger.info(f"Using cache directory: {cache_dir}")
        model_kwargs["cache_dir"] = cache_dir

    logger.info(f"Loading Gatortron tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        **model_kwargs,
    )

    logger.info(f"Loading Gatortron model configuration")
    config = AutoConfig.from_pretrained(
        model_name,
        **model_kwargs,
    )

    logger.info(f"Loading Gatortron model")
    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        **model_kwargs,
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

    # Get cache directory
    cache_dir = None
    if "global" in config and "cache_dir" in config["global"]:
        cache_dir = config["global"]["cache_dir"]

        # Create cache directory if needed
        if config.get("output", {}).get("create_cache_dir", True):
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using cache directory: {cache_dir}")

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
        if "jina_base" in config["model"]:
            jina_config = config["model"]["jina_base"]
            logger.info(f"Generating JINA embeddings for {task_name} task")

            output_path = task_dir / f"jina_base.npy"

            # Determine device from optional device_ids for convenience
            device = None
            if "device_ids" in jina_config:
                device_ids_cfg = jina_config.get("device_ids")
                if isinstance(device_ids_cfg, int):
                    device = f"cuda:{device_ids_cfg}"
                elif isinstance(device_ids_cfg, list) and len(device_ids_cfg) > 0:
                    device = f"cuda:{device_ids_cfg[0]}"

            jina_embed(
                df=df,
                output_path=output_path,
                text_col=text_column,
                task=jina_config.get("task", "text-matching"),
                model_name=jina_config.get("name", "jinaai/jina-embeddings-v3"),
                checkpoint_path=jina_config.get("checkpoint_path"),
                batch_size=jina_config.get("batch_size", 32),
                max_length=jina_config.get("max_length", 2048),
                device=device,
                trust_remote_code=jina_config.get("trust_remote_code", True),
                cache_dir=cache_dir,
            )

        # JINA model
        if "jina" in config["model"]:
            jina_config = config["model"]["jina"]
            logger.info(f"Generating JINA embeddings for {task_name} task")

            output_path = task_dir / f"jina_embeddings.npy"
            if jina_config.get("checkpoint_path"):
                output_path = task_dir / f"jina_finetuned.npy"

            # Determine device from optional device_ids for convenience
            device = None
            if "device_ids" in jina_config:
                device_ids_cfg = jina_config.get("device_ids")
                if isinstance(device_ids_cfg, int):
                    device = f"cuda:{device_ids_cfg}"
                elif isinstance(device_ids_cfg, list) and len(device_ids_cfg) > 0:
                    device = f"cuda:{device_ids_cfg[0]}"

            jina_embed(
                df=df,
                output_path=output_path,
                text_col=text_column,
                task=jina_config.get("task", "text-matching"),
                model_name=jina_config.get("name", "jinaai/jina-embeddings-v3"),
                checkpoint_path=jina_config.get("checkpoint_path"),
                batch_size=jina_config.get("batch_size", 32),
                max_length=jina_config.get("max_length", 2048),
                device=device,
                trust_remote_code=jina_config.get("trust_remote_code", True),
                cache_dir=cache_dir,
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
                cache_dir=cache_dir,
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
                cache_dir=cache_dir,
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

    # Generate embeddings
    generate_embeddings(config)

    logger.info("Embedding generation completed successfully!")


if __name__ == "__main__":
    main()
