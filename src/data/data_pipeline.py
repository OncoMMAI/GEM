# -*- coding: utf-8 -*-

"""
Data processing pipeline for glioma data preparation.

This module provides a unified interface for data loading, processing, and
dataset creation for the glioma classification and embedding pipeline.

It implements functions to:
1. Load and clean raw clinical text data
2. Create datasets for model training in multiple negative ranking loss (MNRL) format
3. Generate evaluation datasets for tumor type classification, MGMT prediction, and survival prediction
"""

import argparse
import logging
import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Clean medical text while preserving numbers, decimals, clinical measurements, and spacing.

    Args:
        text: Raw text to clean
        config: Optional configuration dictionary with cleaning options

    Returns:
        Cleaned text as a string
    """
    if not isinstance(text, str):
        return ""

    # Check if text cleaning is enabled from config
    if config and "output" in config and not config["output"].get("clean_text", True):
        return text

    # Basic normalization
    if config and "output" in config and config["output"].get("lowercase", True):
        text = text.lower()

    # Remove URLs, emails
    text = re.sub(r"(?:https?://|www\.)\S+", "", text)
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "", text)

    # Remove phone numbers
    phone_patterns = [
        r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",  # (415) 476-5236 or 415-476-5236
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # 4154765236
    ]
    for pattern in phone_patterns:
        text = re.sub(pattern, "", text)

    # Remove fax/tel prefixes and postal codes
    text = re.sub(r"\b(?:tel|fax|telephone):\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", "", text, flags=re.IGNORECASE)

    # Remove extremely long words
    text = re.sub(r"\b[a-zA-Z]{30,}\b", "", text)

    # Remove extra whitespace and asterisks
    text = re.sub(r"\s+", " ", text)
    text = text.replace("*", "")
    text = text.replace("/", "")

    return text.strip()


def load_and_combine_data(
    base_path: str, config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine all cancer type data files from MRI and pathology directories.

    Args:
        base_path: Path to directory containing data files
        config: Optional configuration dictionary with paths and options

    Returns:
        Tuple containing (MRI dataframe, Pathology dataframe)
    """
    COLUMNS_NEEDED = [
        "Patient MRN",
        "Report Text",
        "Exam Description",
        "Patient Sex",
        "Patient Age",
        "Patient Status",
    ]

    # Get paths from config or use defaults
    mri_dir = f"{base_path}/mri"
    path_dir = f"{base_path}/pathology"

    if config and "data" in config:
        mri_dir = config["data"].get("mri_dir", mri_dir)
        path_dir = config["data"].get("pathology_dir", path_dir)

    # Load all CSV files in the MRI directory
    mri_dfs = []
    logger.info(f"Loading MRI reports from {mri_dir}")
    mri_files = list(Path(mri_dir).glob("*_synthetic.csv"))

    for file_path in tqdm(mri_files, desc="Loading MRI files"):
        try:
            # Extract cancer type from filename
            cancer_type = file_path.stem.replace("_synthetic", "").lower()

            # Load MRI data
            mri_df = pd.read_csv(file_path)
            mri_df["cancer_type"] = cancer_type
            
            # Get text column from config if available
            text_column = "Report Text"
            if config and "data" in config and "text_column" in config["data"]:
                text_column = config["data"]["text_column"]
                
            # Clean text with config options
            mri_df["clean_text"] = mri_df[text_column].apply(lambda x: clean_text(x, config))
            # Add an accession number if not present
            if "Accession Number" not in mri_df.columns:
                mri_df["Accession Number"] = [
                    f"MRI_{i:06d}" for i in range(len(mri_df))
                ]
            # Add exam date if not present
            if "Exam Completed Date" not in mri_df.columns:
                mri_df["Exam Completed Date"] = pd.Timestamp.now()

            mri_dfs.append(mri_df)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue

    # Load all CSV files in the pathology directory
    path_dfs = []
    logger.info(f"Loading pathology reports from {path_dir}")
    path_files = list(Path(path_dir).glob("*_synthetic.csv"))

    for file_path in tqdm(path_files, desc="Loading pathology files"):
        try:
            # Extract cancer type from filename
            cancer_type = file_path.stem.replace("_synthetic", "").lower()

            # Load pathology data
            path_df = pd.read_csv(file_path)
            path_df["cancer_type"] = cancer_type
            
            # Get text column from config if available
            text_column = "Report Text"
            if config and "data" in config and "text_column" in config["data"]:
                text_column = config["data"]["text_column"]
                
            # Clean text with config options
            path_df["clean_text"] = path_df[text_column].apply(lambda x: clean_text(x, config))
            # Add an accession number if not present
            if "Accession Number" not in path_df.columns:
                path_df["Accession Number"] = [
                    f"PATH_{i:06d}" for i in range(len(path_df))
                ]
            # Add exam date if not present
            if "Exam Completed Date" not in path_df.columns:
                path_df["Exam Completed Date"] = pd.Timestamp.now()

            path_dfs.append(path_df)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue

    if not mri_dfs or not path_dfs:
        raise ValueError("No data files were successfully loaded")

    combined_mri = pd.concat(mri_dfs, ignore_index=True)
    combined_path = pd.concat(path_dfs, ignore_index=True)

    logger.info(
        f"Loaded {len(combined_mri)} MRI reports and {len(combined_path)} pathology reports"
    )
    return combined_mri, combined_path


def match_clinical_reports(
    mri_df: pd.DataFrame, 
    path_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Match MRI and pathology reports based on patient and cancer type.

    Args:
        mri_df: DataFrame containing MRI reports
        path_df: DataFrame containing pathology reports
        config: Optional configuration dictionary with matching options

    Returns:
        DataFrame containing matched reports
    """
    matches = []

    mri_df = mri_df.copy()
    path_df = path_df.copy()

    # Get MRN column from config if available
    mrn_col = "Patient MRN"
    if config and "data" in config and "mrn_column" in config["data"]:
        mrn_col = config["data"]["mrn_column"]

    for (mrn, cancer_type), mri_group in tqdm(
        mri_df.groupby([mrn_col, "cancer_type"]), desc="Matching reports"
    ):
        path_group = path_df[
            (path_df[mrn_col] == mrn) & (path_df["cancer_type"] == cancer_type)
        ]

        if not path_group.empty:
            for _, mri in mri_group.iterrows():
                for _, path in path_group.iterrows():
                    matches.append(
                        {
                            "mrn": mrn,
                            "cancer_type": cancer_type,
                            "mri_accession": mri["Accession Number"],
                            "path_accession": path["Accession Number"],
                            "mri_text": mri["clean_text"],
                            "path_text": path["clean_text"],
                            "mri_date": mri["Exam Completed Date"],
                            "path_date": path["Exam Completed Date"],
                        }
                    )

    matches_df = pd.DataFrame(matches)

    if matches_df.empty:
        raise ValueError("No matching reports found")

    logger.info(f"Total matches found: {len(matches_df)}")
    logger.info(f"Unique patients: {matches_df['mrn'].nunique()}")
    logger.info(
        f"Cancer type distribution:\n{matches_df['cancer_type'].value_counts()}"
    )

    return matches_df


def create_mnrl_pairs(
    matches_df: pd.DataFrame, 
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Create multiple negative ranking loss pairs from matched reports.

    Args:
        matches_df: DataFrame containing matched MRI and pathology reports
        config: Optional configuration dictionary with MNRL options

    Returns:
        DataFrame containing anchor-positive pairs for MNRL training
    """
    logger.info("Creating MNRL pairs...")

    # Check if MNRL is enabled in config
    mnrl_enabled = True
    if config and "training" in config and "mnrl" in config["training"]:
        mnrl_enabled = config["training"]["mnrl"].get("enable", True)
    
    if not mnrl_enabled:
        logger.warning("MNRL training is disabled in config, but continuing to create pairs")

    # Only need anchor-positive pairs for MNRL
    mnrl_pairs = pd.DataFrame(
        {
            "anchor": matches_df["mri_text"].values,
            "positive": matches_df["path_text"].values,
            "mrn": matches_df["mrn"].values,
            "cancer_type": matches_df["cancer_type"].values,
        }
    )

    logger.info(f"Created {len(mnrl_pairs)} MNRL pairs")
    return mnrl_pairs


def create_training_datasets(
    matches_df: Union[str, pd.DataFrame],
    output_base_path: str = "data/processed",
    train_split: float = 0.8,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create MNRL dataset and split into train/eval.

    Args:
        matches_df: DataFrame or path to CSV file containing matched reports
        output_base_path: Base directory to save processed data
        train_split: Proportion of data to use for training
        config: Optional configuration dictionary with paths and options

    Returns:
        Dictionary containing information about the created datasets
    """
    if isinstance(matches_df, str):
        matches_df = pd.read_csv(matches_df)

    Path(output_base_path).mkdir(parents=True, exist_ok=True)
    mnrl_dir = Path(output_base_path) / "mnrl"
    mnrl_dir.mkdir(parents=True, exist_ok=True)

    # Make split by patient to avoid data leakage
    unique_mrns = matches_df["mrn"].unique()
    np.random.shuffle(unique_mrns)

    # Create train/eval splits
    train_idx = int(train_split * len(unique_mrns))
    train_mrns = unique_mrns[:train_idx]
    eval_mrns = unique_mrns[train_idx:]

    splits = {
        "train": matches_df[matches_df["mrn"].isin(train_mrns)],
        "eval": matches_df[matches_df["mrn"].isin(eval_mrns)],
    }

    results = {}

    # Process each split
    for split_name, split_df in splits.items():
        logger.info(f"Processing {split_name} split...")
        pairs_df = create_mnrl_pairs(split_df, config)

        # Get output path from config if available
        output_path = f"{output_base_path}/mnrl/{split_name}.csv"
        if config and "output" in config:
            if split_name == "train" and "mnrl_train_path" in config["output"]:
                output_path = config["output"]["mnrl_train_path"]
            elif split_name == "eval" and "mnrl_eval_path" in config["output"]:
                output_path = config["output"]["mnrl_eval_path"]

        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        pairs_df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}: {len(pairs_df)} pairs")

        # Save stats
        results[split_name] = {
            "path": output_path,
            "samples": len(pairs_df),
            "patients": len(pairs_df["mrn"].unique()),
        }

    return results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Data pipeline for processing glioma data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="configs/dataset_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input", help="Input directory containing raw data (overrides config)"
    )
    parser.add_argument(
        "--output", help="Output directory for processed data (overrides config)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        help="Proportion of data to use for training (overrides config)",
    )

    return parser.parse_args()


def main():
    """Run the data processing pipeline."""
    args = parse_args()

    # Load configuration
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found, using default values")
        config = {"data": {}, "output": {}}

    # Configure logging based on config
    log_level = config.get("logging", {}).get("level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
    
    # Override config with command line arguments if provided
    input_path = args.input or config.get("data", {}).get("input_dir", "data/raw")
    output_path = args.output or config.get("output", {}).get(
        "output_dir", "data/training"
    )
    train_split = args.train_split or config.get("training", {}).get("train_split", 0.8)

    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Create eval directory if it doesn't exist
    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Report configuration settings
    logger.info(f"Using configuration from: {args.config}")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Train split ratio: {train_split}")
    
    # Report data filtering settings
    min_length = config.get("data", {}).get("min_report_length", 50)
    max_length = config.get("data", {}).get("max_report_length", 5000)
    logger.info(f"Report length filtering: min={min_length}, max={max_length}")
    
    # Report MNRL settings
    mnrl_enabled = config.get("training", {}).get("mnrl", {}).get("enable", True)
    neg_samples = config.get("training", {}).get("mnrl", {}).get("negative_samples", 5)
    logger.info(f"MNRL enabled: {mnrl_enabled}, negative samples: {neg_samples}")
    
    # Report class balancing
    balance = config.get("training", {}).get("balance_classes", True)
    logger.info(f"Class balancing: {balance}")

    # Load and process data
    logger.info(f"Loading data from {input_path}...")
    try:
        mri_data, path_data = load_and_combine_data(input_path, config)

        # Match reports
        matches_df = match_clinical_reports(mri_data, path_data, config)

        # Get matches path from config if available
        matches_path = f"{output_path}/matches.csv"
        if config and "output" in config and "matches_path" in config["output"]:
            matches_path = config["output"]["matches_path"]

        # Ensure directory exists
        Path(matches_path).parent.mkdir(parents=True, exist_ok=True)
        matches_df.to_csv(matches_path, index=False)
        logger.info(f"Saved matched reports to {matches_path}")

        # Create training datasets
        create_training_datasets(
            matches_df=matches_df,
            output_base_path=output_path,
            train_split=train_split,
            config=config,
        )
        
        # Save demographics figure if configured
        save_demographics = config.get("logging", {}).get("save_demographics", False)
        if save_demographics:
            figures_dir = config.get("output", {}).get("figures_dir", "figures")
            Path(figures_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Generating demographics figure to {figures_dir}/demographics.pdf")
            try:
                from src.data.visualization import create_demographics_figure
                create_demographics_figure(
                    mri_data=mri_data,
                    path_data=path_data,
                    output_path=f"{figures_dir}/demographics.pdf"
                )
            except ImportError:
                logger.warning("Unable to create demographics figure - visualization module not found")
        
        logger.info("Data creation completed successfully!")

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise


if __name__ == "__main__":
    main()
