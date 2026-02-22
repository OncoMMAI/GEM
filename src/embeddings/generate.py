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


def huggingface_embed(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    text_col: str = "report",
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    pooling: str = "cls",
    normalize: bool = False,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """Generate embeddings using a standard HuggingFace transformer model.

    Supports CLS-token pooling and mean pooling with optional L2 normalization.
    Works with models like BioClinicalBERT, PubMedBERT, MedEmbed, and ModernBERT.

    Args:
        df: Input dataframe containing text
        output_path: Path to save embeddings
        text_col: Column name containing text
        model_name: HuggingFace model identifier
        pooling: Pooling strategy ('cls' or 'mean')
        normalize: Whether to L2-normalize the output embeddings
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

    model_kwargs = {"trust_remote_code": trust_remote_code}
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    logger.info(f"Loading model: {model_name} (pooling={pooling}, normalize={normalize})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    except Exception:
        logger.warning(f"Fast tokenizer failed for {model_name}, falling back to slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, **model_kwargs)
    model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)

    model.eval()
    embeddings_list = []

    logger.info(
        f"Generating embeddings for {len(df)} texts using batch size {batch_size}"
    )
    for i in tqdm(range(0, len(df), batch_size), desc=f"Generating {model_name} embeddings"):
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

            if pooling == "cls":
                batch_emb = outputs.last_hidden_state[:, 0, :]
            else:
                attention_mask = inputs["attention_mask"]
                hidden = outputs.last_hidden_state
                batch_emb = (hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)

            if normalize:
                batch_emb = F.normalize(batch_emb, p=2, dim=1)

            embeddings_list.append(batch_emb.cpu().numpy())

    embeddings = np.vstack(embeddings_list)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    logger.info(f"Saved to {output_path}")

    return embeddings


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

    # Map internal task keys to output directory names
    _TASK_DIRS = {"tumor_type": "tumor_type", "survival": "os", "mgmt": "mgmt"}

    skip_existing = config.get("output", {}).get("skip_existing", True)

    # Process each task and model combination
    for task_name, df in datasets.items():
        dir_name = _TASK_DIRS.get(task_name, task_name)
        task_dir = output_dir / dir_name
        task_dir.mkdir(exist_ok=True)

        # Helper to check if output already exists
        def _should_skip(path: Path) -> bool:
            if skip_existing and path.exists():
                logger.info(f"Skipping {path.name} (already exists)")
                return True
            return False

        # JINA base model
        if "jina_base" in config["model"]:
            jina_config = config["model"]["jina_base"]
            output_path = task_dir / "jina_base.npy"

            if not _should_skip(output_path):
                logger.info(f"Generating JINA base embeddings for {task_name} task")
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

        # JINA finetuned model (single variant, legacy config key)
        if "jina" in config["model"]:
            jina_config = config["model"]["jina"]
            output_path = task_dir / "jina_embeddings.npy"
            if jina_config.get("checkpoint_path"):
                output_path = task_dir / "jina_finetuned.npy"

            if not _should_skip(output_path):
                logger.info(f"Generating JINA finetuned embeddings for {task_name} task")
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

        # JINA finetuned variants (loss function comparison)
        # Each variant shares the base Jina model but uses a different checkpoint.
        if "jina_finetuned_variants" in config["model"]:
            jina_defaults = config["model"].get("jina_finetuned_defaults", {})
            for variant in config["model"]["jina_finetuned_variants"]:
                output_filename = variant["output_filename"]
                output_path = task_dir / output_filename

                if _should_skip(output_path):
                    continue

                variant_name = variant.get("name", output_filename)
                checkpoint = variant.get("checkpoint_path")
                if not checkpoint or not Path(checkpoint).exists():
                    logger.warning(
                        f"Checkpoint not found for {variant_name}: {checkpoint}, skipping"
                    )
                    continue

                logger.info(
                    f"Generating {variant_name} embeddings for {task_name} task"
                )
                device = None
                device_ids_cfg = variant.get(
                    "device_ids", jina_defaults.get("device_ids")
                )
                if device_ids_cfg is not None:
                    if isinstance(device_ids_cfg, int):
                        device = f"cuda:{device_ids_cfg}"
                    elif isinstance(device_ids_cfg, list) and len(device_ids_cfg) > 0:
                        device = f"cuda:{device_ids_cfg[0]}"

                jina_embed(
                    df=df,
                    output_path=output_path,
                    text_col=text_column,
                    task=variant.get("task", jina_defaults.get("task", "text-matching")),
                    model_name=variant.get(
                        "model_name",
                        jina_defaults.get("model_name", "jinaai/jina-embeddings-v3"),
                    ),
                    checkpoint_path=checkpoint,
                    batch_size=variant.get(
                        "batch_size", jina_defaults.get("batch_size", 32)
                    ),
                    max_length=variant.get(
                        "max_length", jina_defaults.get("max_length", 2048)
                    ),
                    device=device,
                    trust_remote_code=variant.get(
                        "trust_remote_code",
                        jina_defaults.get("trust_remote_code", True),
                    ),
                    cache_dir=cache_dir,
                )

        # NVIDIA model
        if "nvidia" in config["model"]:
            nvidia_config = config["model"]["nvidia"]
            output_path = task_dir / "nv_embed.npy"
            if nvidia_config.get("instruction"):
                output_path = (
                    task_dir / f"nv_embed_{nvidia_config.get('instruction')}.npy"
                )

            if not _should_skip(output_path):
                logger.info(f"Generating NVIDIA embeddings for {task_name} task")
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
            output_path = task_dir / "gatortron_large.npy"

            if not _should_skip(output_path):
                logger.info(f"Generating Gatortron embeddings for {task_name} task")
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

        # Generic HuggingFace models (MedEmbed, BioClinicalBERT, PubMedBERT, ModernBERT)
        _HF_MODELS = {
            "medembed": "medembed_small",
            "bioclinicalbert": "bio_clinicalbert",
            "pubmedbert": "pubmedbert_embeddings",
            "modernbert": "modernbert_embed",
        }
        for model_key, output_name in _HF_MODELS.items():
            if model_key not in config["model"]:
                continue
            hf_config = config["model"][model_key]
            output_path = task_dir / f"{output_name}.npy"

            if _should_skip(output_path):
                continue

            logger.info(f"Generating {model_key} embeddings for {task_name} task")

            device = None
            if "device_ids" in hf_config:
                device_ids_cfg = hf_config.get("device_ids")
                if isinstance(device_ids_cfg, int):
                    device = f"cuda:{device_ids_cfg}"
                elif isinstance(device_ids_cfg, list) and len(device_ids_cfg) > 0:
                    device = f"cuda:{device_ids_cfg[0]}"

            huggingface_embed(
                df=df,
                output_path=output_path,
                text_col=text_column,
                model_name=hf_config["name"],
                pooling=hf_config.get("pooling", "cls"),
                normalize=hf_config.get("normalize", False),
                batch_size=hf_config.get("batch_size", 32),
                max_length=hf_config.get("max_length", 512),
                device=device,
                trust_remote_code=hf_config.get("trust_remote_code", False),
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
