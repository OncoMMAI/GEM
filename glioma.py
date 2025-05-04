# -*- coding: utf-8 -*-

"""
Command-line interface for glioma classification toolkit.

This script provides a simple CLI for running the different modules of the project.
It supports fine-tuning JINA embeddings, generating embeddings, evaluating models,
running inference, and visualizing results.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Glioma Classification Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune JINA embeddings with DDP on 4 GPUs
  glioma finetune --gpus 0,1,2,3 --ddp --num_gpus 4
  
  # Generate embeddings for all models
  glioma generate --gpus 0,1,2,3
  
  # Evaluate model performance
  glioma evaluate
  
  # Run embedding generation on Gatortron model only
  glioma generate --model gatortron --gpus 0
  
  # Visualize embeddings with dimension reduction
  glioma visualize --method umap
  
  # Data preprocessing and preparation
  glioma prepare-data --input path/to/data --output data/processed
""",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command", help="Command to run", required=True
    )

    # Finetune command
    finetune_parser = subparsers.add_parser(
        "finetune", help="Fine-tune JINA embeddings"
    )
    finetune_parser.add_argument(
        "--config",
        default="configs/finetune_config.yaml",
        help="Path to the configuration file",
    )
    finetune_parser.add_argument(
        "--gpus", default=None, help="Comma-separated list of GPU IDs (e.g. '0,1,2,3')"
    )
    finetune_parser.add_argument(
        "--ddp", action="store_true", help="Use PyTorch Distributed Data Parallel (DDP)"
    )
    finetune_parser.add_argument(
        "--num_gpus", type=int, default=None, help="Number of GPUs to use with DDP"
    )
    finetune_parser.add_argument(
        "--port", type=int, default=29500, help="Port for DDP communication"
    )
    finetune_parser.add_argument(
        "--output_dir", help="Override the output directory for model checkpoints"
    )
    finetune_parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility"
    )

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate embeddings")
    generate_parser.add_argument(
        "--config",
        default="configs/generate_config.yaml",
        help="Path to the configuration file",
    )
    generate_parser.add_argument(
        "--gpus", default=None, help="Comma-separated list of GPU IDs (e.g. '0,1,2,3')"
    )
    generate_parser.add_argument(
        "--model",
        choices=["jina", "nvidia", "gatortron", "all"],
        default="all",
        help="Specific model to generate embeddings for",
    )
    generate_parser.add_argument(
        "--task",
        choices=["tumor_type", "mgmt", "survival", "all"],
        default="all",
        help="Specific task to generate embeddings for",
    )
    generate_parser.add_argument(
        "--output_dir", help="Override the output directory for embeddings"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate embeddings")
    evaluate_parser.add_argument(
        "--config",
        default="configs/evaluation_config.yaml",
        help="Path to the configuration file",
    )
    evaluate_parser.add_argument(
        "--task",
        choices=["tumor_type", "mgmt", "survival", "all"],
        default="all",
        help="Specific task to evaluate",
    )
    evaluate_parser.add_argument(
        "--output_dir", help="Override the output directory for results"
    )
    evaluate_parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility"
    )

    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument(
        "--config",
        default="configs/inference_config.yaml",
        help="Path to the configuration file",
    )
    inference_parser.add_argument(
        "--input_text", help="Input text for direct inference"
    )
    inference_parser.add_argument(
        "--input_file", help="Path to file containing texts for batch inference"
    )
    inference_parser.add_argument(
        "--output_file", help="Path to save inference results"
    )
    inference_parser.add_argument(
        "--model",
        choices=["biomistral", "medllama3", "custom"],
        help="Model to use for inference",
    )
    inference_parser.add_argument(
        "--gpus", default=None, help="Comma-separated list of GPU IDs (e.g. '0,1,2,3')"
    )

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize embeddings")
    visualize_parser.add_argument(
        "--config",
        default="configs/visualize_config.yaml",
        help="Path to the configuration file",
    )
    visualize_parser.add_argument(
        "--embeddings_dir", help="Directory containing embeddings to visualize"
    )
    visualize_parser.add_argument(
        "--output_dir", help="Directory to save visualization outputs"
    )
    visualize_parser.add_argument(
        "--task",
        choices=["tumor_type", "mgmt", "survival", "all"],
        default="all",
        help="Specific task to visualize",
    )

    # Data preparation command
    data_parser = subparsers.add_parser(
        "prepare-data", help="Prepare data for training and evaluation"
    )
    data_parser.add_argument(
        "--config",
        default="configs/dataset_config.yaml",
        help="Path to the dataset configuration file",
    )
    data_parser.add_argument(
        "--input", help="Input directory or file with raw data (overrides config)"
    )
    data_parser.add_argument(
        "--output", help="Output directory for processed data (overrides config)"
    )
    data_parser.add_argument(
        "--train_split",
        type=float,
        help="Proportion of data for training (overrides config)",
    )

    return parser.parse_args()


def main():
    """Run the appropriate command based on arguments."""
    args = parse_args()

    cmd = []

    # Always start with UV
    cmd = ["uv", "run", "python"]

    if args.command == "finetune":
        # Check if using DDP
        if args.ddp:
            cmd = ["uv", "run", "torchrun"]

            if args.num_gpus:
                cmd.extend(["--nproc_per_node", str(args.num_gpus)])
            else:
                cmd.extend(["--nproc_per_node", "1"])

            cmd.extend(["--master_port", str(args.port)])
            cmd.extend(["-m", "src.models.finetune", "--config", args.config])

            if args.gpus:
                cmd.extend(["--gpus", args.gpus])

            if args.output_dir:
                cmd.extend(["--output_dir", args.output_dir])

            if args.seed:
                cmd.extend(["--seed", str(args.seed)])

        else:
            cmd.extend(["-m", "src.models.finetune", "--config", args.config])

            if args.gpus:
                cmd.extend(["--gpus", args.gpus])

            if args.output_dir:
                cmd.extend(["--output_dir", args.output_dir])

            if args.seed:
                cmd.extend(["--seed", str(args.seed)])

    elif args.command == "generate":
        cmd.extend(["-m", "src.embeddings.generate", "--config", args.config])

        if args.gpus:
            cmd.extend(["--gpus", args.gpus])

        if args.model != "all":
            cmd.extend(["--model", args.model])

        if args.task != "all":
            cmd.extend(["--task", args.task])

        if args.output_dir:
            cmd.extend(["--output_dir", args.output_dir])

    elif args.command == "evaluate":
        cmd.extend(["-m", "src.models.evaluation", "--config", args.config])

        if args.task != "all":
            cmd.extend(["--task", args.task])

        if args.output_dir:
            cmd.extend(["--output_dir", args.output_dir])

        if args.seed:
            cmd.extend(["--seed", str(args.seed)])

    elif args.command == "inference":
        cmd.extend(["-m", "src.models.inference", "--config", args.config])

        if args.input_text:
            cmd.extend(["--input_text", args.input_text])

        if args.input_file:
            cmd.extend(["--input_file", args.input_file])

        if args.output_file:
            cmd.extend(["--output_file", args.output_file])

        if args.model:
            cmd.extend(["--model", args.model])

        if args.gpus:
            cmd.extend(["--gpus", args.gpus])

    elif args.command == "visualize":
        cmd.extend(["-m", "src.embeddings.visualize", "--config", args.config])

        if args.embeddings_dir:
            cmd.extend(["--embeddings_dir", args.embeddings_dir])

        if args.output_dir:
            cmd.extend(["--output_dir", args.output_dir])

        if args.task != "all":
            cmd.extend(["--task", args.task])

    elif args.command == "prepare-data":
        cmd.extend(["-m", "src.data.data_pipeline", "--config", args.config])

        if args.input:
            cmd.extend(["--input", args.input])

        if args.output:
            cmd.extend(["--output", args.output])

        if args.train_split:
            cmd.extend(["--train_split", str(args.train_split)])

    # Execute the command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
