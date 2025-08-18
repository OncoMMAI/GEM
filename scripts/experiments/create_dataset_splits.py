#!/usr/bin/env python
"""Create dataset splits of different sizes for experiments."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def create_dataset_splits(train_path, eval_path, output_dir, seed=42):
    """Create different sized splits of the MNRL dataset."""
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load full datasets
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"  Full training size: {len(train_df)} samples")
    
    print(f"\nLoading evaluation data from {eval_path}...")
    eval_df = pd.read_csv(eval_path)
    print(f"  Full evaluation size: {len(eval_df)} samples")
    
    # Define split sizes
    splits = {
        "10k": {
            "train_size": 10000,
            "eval_size": 2000,
            "description": "Small dataset (~13% of full)"
        },
        "40k": {
            "train_size": 40000,
            "eval_size": 10000,
            "description": "Medium dataset (~50% of full)"
        }
    }
    
    # Create splits
    for split_name, split_config in splits.items():
        print(f"\n{'='*50}")
        print(f"Creating {split_name} split: {split_config['description']}")
        print(f"{'='*50}")
        
        train_size = split_config["train_size"]
        eval_size = split_config["eval_size"]
        
        # Check if we have enough samples
        if train_size > len(train_df):
            print(f"Warning: Requested train size {train_size} > available {len(train_df)}")
            train_size = len(train_df)
        
        if eval_size > len(eval_df):
            print(f"Warning: Requested eval size {eval_size} > available {len(eval_df)}")
            eval_size = len(eval_df)
        
        # Sample from the full datasets
        train_split = train_df.sample(n=train_size, random_state=seed)
        eval_split = eval_df.sample(n=eval_size, random_state=seed)
        
        # Save splits
        train_output = output_path / f"train_{split_name}.csv"
        eval_output = output_path / f"eval_{split_name}.csv"
        
        train_split.to_csv(train_output, index=False)
        eval_split.to_csv(eval_output, index=False)
        
        print(f"  ✓ Saved training split: {train_output}")
        print(f"    - Size: {len(train_split)} samples")
        if 'cancer_type' in train_split.columns:
            print(f"    - Cancer type distribution:")
            for cancer_type, count in train_split['cancer_type'].value_counts().items():
                print(f"      {cancer_type}: {count} ({100*count/len(train_split):.1f}%)")
        
        print(f"  ✓ Saved evaluation split: {eval_output}")
        print(f"    - Size: {len(eval_split)} samples")
        if 'cancer_type' in eval_split.columns:
            print(f"    - Cancer type distribution:")
            for cancer_type, count in eval_split['cancer_type'].value_counts().items():
                print(f"      {cancer_type}: {count} ({100*count/len(eval_split):.1f}%)")
    
    # Also create symbolic links for full dataset (for consistency)
    print(f"\n{'='*50}")
    print("Creating symbolic links for full dataset")
    print(f"{'='*50}")
    
    # Copy full datasets (not symlinks since we might be on different filesystems)
    train_full_output = output_path / "train_full.csv"
    eval_full_output = output_path / "eval_full.csv"
    
    train_df.to_csv(train_full_output, index=False)
    eval_df.to_csv(eval_full_output, index=False)
    
    print(f"  ✓ Copied full training data: {train_full_output}")
    print(f"  ✓ Copied full evaluation data: {eval_full_output}")
    
    print(f"\n{'='*50}")
    print("Dataset splits created successfully!")
    print(f"{'='*50}")
    print(f"\nAll splits saved to: {output_path}")
    print("\nCreated files:")
    for file in sorted(output_path.glob("*.csv")):
        print(f"  - {file.name}")

def main():
    parser = argparse.ArgumentParser(
        description="Create dataset splits for fine-tuning experiments"
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/training/mnrl/train.csv",
        help="Path to full training dataset"
    )
    parser.add_argument(
        "--eval-path", 
        type=str,
        default="data/training/mnrl/eval.csv",
        help="Path to full evaluation dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training/mnrl/splits",
        help="Directory to save dataset splits"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    create_dataset_splits(
        train_path=args.train_path,
        eval_path=args.eval_path,
        output_dir=args.output_dir,
        seed=args.seed
    )

if __name__ == "__main__":
    import sys
    import os
    # Add the project root to Python path so we can import from src
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    main()