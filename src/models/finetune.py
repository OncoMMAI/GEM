# -*- coding: utf-8 -*-

"""
Fine-tune JINA embeddings for glioma classification using Multiple Negative Ranking Loss (MNRL).

This module implements the training process described in the paper "Fine-tuning JINA Embeddings
for Glioma Classification and Survival Outcome Prediction". It loads paired radiology-pathology
reports and trains a LoRA-adapted version of jina-embeddings-v3 to align these reports.
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Set environment variables before other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

import pandas as pd
from datasets import Dataset
import torch
import torch.distributed as dist
from sentence_transformers import (
    SentenceTransformer,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.training_args import BatchSamplers

from src.utils.config import get_config, load_config
from src.utils.logging import setup_logger, disable_external_loggers
from src.utils.seed import set_seed
from src.utils.exceptions import (
    ConfigurationError,
    DataError,
    ModelError,
    TrainingError,
)


# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*custom_fwd.*")
warnings.filterwarnings("ignore", message=".*custom_bwd.*")
warnings.filterwarnings("ignore", message=".*evaluation_strategy.*")


def load_and_validate_data(file_path: Path) -> Dataset:
    """Load and validate dataset with cancer type information.

    This function loads a CSV file containing paired data for training
    and validates that it contains the required columns.

    Args:
        file_path: Path to the CSV file containing paired data

    Returns:
        Dataset containing sentence pairs for training

    Raises:
        DataError: If the file doesn't exist or required columns are missing
    """
    logger = logging.getLogger("glioma.finetune")
    logger.info(f"Loading data from {file_path}")

    try:
        # Check if file exists
        if not Path(file_path).exists():
            raise DataError(f"Dataset file not found", str(file_path))

        # Load data
        df = pd.read_csv(file_path)

        # Validate data columns
        required_columns = ["anchor", "positive", "cancer_type"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise DataError(
                f"Missing required columns: {missing_columns}", str(file_path)
            )

        # Check for empty dataframe
        if len(df) == 0:
            raise DataError("Dataset is empty", str(file_path))

        # Get unique cancer types
        unique_cancer_types = df["cancer_type"].unique()

        # Log dataset statistics
        logger.info(f"Dataset size: {len(df)}")
        logger.info(f"Number of cancer types: {len(unique_cancer_types)}")
        logger.info(
            f"Cancer types distribution:\n{df['cancer_type'].value_counts(normalize=True)}"
        )

        # Create dataset
        return Dataset.from_dict(
            {
                "sentence1": df["anchor"].tolist(),
                "sentence2": df["positive"].tolist(),
            }
        )

    except Exception as e:
        # If it's not our custom error, wrap it
        if not isinstance(
            e, (ConfigurationError, DataError, ModelError, TrainingError)
        ):
            e = DataError(f"Error loading dataset: {str(e)}", str(file_path))

        logger.error(str(e))
        raise e


def setup_distributed_training() -> Tuple[int, int, bool]:
    """Setup distributed training environment.

    Returns:
        Tuple containing:
            - Local process rank
            - World size (total number of processes)
            - Whether this is the main process (rank 0)
    """
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["LOCAL_RANK"])
    else:
        world_size = torch.cuda.device_count()
        rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)

    is_main_process = rank == 0
    return rank, world_size, is_main_process


def create_model(config: Dict[str, Any]) -> SentenceTransformer:
    """Create and initialize the SentenceTransformer model.

    This function initializes a SentenceTransformer model with the
    specified configuration, including support for LoRA adaptation
    and proper model card metadata.

    Args:
        config: Configuration dictionary containing model parameters

    Returns:
        Initialized SentenceTransformer model

    Raises:
        ModelError: If model initialization fails
    """
    logger = logging.getLogger("glioma.finetune")

    try:
        model_config = config["model"]
        model_name = model_config["name"]

        logger.info(f"Creating model from {model_name}")

        # Initialize model with configuration
        model = SentenceTransformer(
            model_name,
            trust_remote_code=model_config.get("trust_remote_code", True),
            model_kwargs={
                "default_task": model_config.get("default_task", None),
                "lora_main_params_trainable": model_config.get(
                    "lora_main_params_trainable", True
                ),
            },
            model_card_data=SentenceTransformerModelCardData(
                language=["en"],
                license="apache-2.0",
                model_name="Jina-v3-Clinical-Cross-Modal-Alignment",
                model_id="bneo/Jina-v3-Clinical-Cross-Modal-Alignment",
                tags=["clinical", "radiology", "pathology", "cross-modal", "glioma"],
            ),
        )

        # Log model information
        logger.info(
            f"Model created successfully with {model.get_sentence_embedding_dimension()} dimensions"
        )

        return model

    except Exception as e:
        # Wrap exceptions in ModelError
        if not isinstance(
            e, (ConfigurationError, DataError, ModelError, TrainingError)
        ):
            e = ModelError(
                f"Error initializing model: {str(e)}",
                model_config.get("name", "unknown"),
            )

        logger.error(str(e))
        raise e


def create_training_args(
    config: Dict[str, Any],
    output_dir: Path,
    train_dataset: Dataset,
    rank: int,
    world_size: int,
) -> SentenceTransformerTrainingArguments:
    """Create training arguments for the trainer.

    Args:
        config: Configuration dictionary
        output_dir: Directory to save the model
        train_dataset: Training dataset
        rank: Local process rank
        world_size: Total number of processes

    Returns:
        Training arguments for SentenceTransformerTrainer
    """
    training_config = config["training"]

    train_batch_size = training_config["train_batch_size"]
    eval_batch_size = training_config["eval_batch_size"]
    gradient_accumulation_steps = training_config["gradient_accumulation_steps"]

    # Calculate steps per epoch for validation and checkpointing
    total_steps_per_epoch = len(train_dataset) // (
        train_batch_size * gradient_accumulation_steps * world_size
    )

    # Get batch sampler from config or use default
    batch_sampler_str = training_config.get("batch_sampler", "NO_DUPLICATES")
    batch_sampler = (
        BatchSamplers.BATCH_SAMPLER
        if batch_sampler_str == "DEFAULT"
        else BatchSamplers.NO_DUPLICATES
    )

    # Create training arguments
    return SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config["num_epochs"],
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=training_config["learning_rate"],
        warmup_ratio=training_config["warmup_ratio"],
        max_grad_norm=training_config["max_grad_norm"],
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        eval_steps=total_steps_per_epoch,
        save_steps=total_steps_per_epoch,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        logging_dir=str(output_dir / "logs"),
        local_rank=rank,
        ddp_backend="nccl",
        dataloader_drop_last=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        batch_sampler=batch_sampler,
        report_to="none",
    )


def train_model(config: Dict[str, Any], is_main_process: bool) -> None:
    """Train the model using the provided configuration.

    Args:
        config: Configuration dictionary
        is_main_process: Whether this is the main process (rank 0)
    """
    logger = logging.getLogger("glioma.finetune")

    # Define paths and create output directory
    data_config = config["data"]
    output_dir = Path(config.get("output", {}).get("results_dir", "models"))
    model_dir = output_dir / config.get("output", {}).get("model_dir", "mnrl_full_data")

    if is_main_process:
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model will be saved to: {model_dir}")

    # Load datasets
    train_dataset = load_and_validate_data(data_config["train_path"])
    eval_dataset = load_and_validate_data(data_config["eval_path"])

    # Create model
    model = create_model(config)

    # Setup distributed training
    rank, world_size, _ = setup_distributed_training()

    # Create training arguments
    training_args = create_training_args(
        config=config,
        output_dir=model_dir,
        train_dataset=train_dataset,
        rank=rank,
        world_size=world_size,
    )

    # Calculate total steps per epoch for logging
    total_steps_per_epoch = len(train_dataset) // (
        config["training"]["train_batch_size"]
        * config["training"]["gradient_accumulation_steps"]
        * world_size
    )

    if is_main_process:
        logger.info(f"Total steps per epoch: {total_steps_per_epoch}")

    # Initialize loss - Multiple Negative Ranking Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Initialize trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=None,
    )

    # Start training
    if is_main_process:
        logger.info("Starting training...")
    trainer.train()

    # Save final model and results only on main process
    if is_main_process:
        final_output_dir = model_dir / "final"
        model.save(str(final_output_dir))
        logger.info(f"Model saved to {final_output_dir}")

    # Cleanup distributed training resources
    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    """Main entry point for fine-tuning."""
    # Setup logger
    logger = setup_logger(
        name="glioma.finetune", level=logging.INFO, log_file="training.log"
    )

    # Disable verbose logging from external libraries
    disable_external_loggers()

    # Load configuration
    config = get_config()

    # Set random seed for reproducibility
    if "global" in config and "seed" in config["global"]:
        set_seed(config["global"]["seed"])
    else:
        set_seed(42)  # Default seed if not specified

    # Set CUDA devices if specified in config
    if "gpus" in config.get("training", {}):
        os.environ["CUDA_VISIBLE_DEVICES"] = config["training"]["gpus"]

    # Setup distributed training
    _, _, is_main_process = setup_distributed_training()

    # Train model
    train_model(config, is_main_process)

    if is_main_process:
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
