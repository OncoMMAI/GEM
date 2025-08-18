"""
This script creates a supervised fine-tuning dataset with positive and negative pairs
from the matched radiology and pathology reports.
"""

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_supervised_pairs(matches_df_path: str, output_path: str):
    """
    Generates a balanced dataset of positive and negative pairs for supervised training.

    Args:
        matches_df_path (str): Path to the CSV file with matched reports.
        output_path (str): Path to save the final supervised pairs CSV.
    """
    print(f"Loading matched reports from {matches_df_path}...")
    matches_df = pd.read_csv(matches_df_path)

    positive_pairs = []
    negative_pairs = []

    # Create a dictionary for quick lookup of pathology reports by cancer type
    path_reports_by_cancer = {}
    for cancer_type in matches_df['cancer_type'].unique():
        path_reports_by_cancer[cancer_type] = matches_df[matches_df['cancer_type'] == cancer_type][['mrn', 'path_text']].to_dict('records')

    print("Generating positive and negative pairs...")
    for _, row in tqdm(matches_df.iterrows(), total=len(matches_df)):
        # Positive pair (label=1.0)
        positive_pairs.append({
            'sentence1': row['mri_text'],
            'sentence2': row['path_text'],
            'label': 1.0,
            'pair_type': 'positive'
        })

        # Negative pair (label=0.0) - This is the "hard negative"
        anchor_mrn = row['mrn']
        cancer_type = row['cancer_type']
        
        # Get potential negative candidates (same cancer type, different patient)
        candidates = [p for p in path_reports_by_cancer[cancer_type] if p['mrn'] != anchor_mrn]
        
        if candidates:
            # Select a random negative sample
            negative_sample = random.choice(candidates)
            negative_pairs.append({
                'sentence1': row['mri_text'],
                'sentence2': negative_sample['path_text'],
                'label': 0.0,
                'pair_type': 'negative_in_domain'
            })

    # Combine positive and negative pairs into a single DataFrame
    positive_df = pd.DataFrame(positive_pairs)
    negative_df = pd.DataFrame(negative_pairs)
    
    supervised_df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Shuffle the dataset
    supervised_df = supervised_df.sample(frac=1).reset_index(drop=True)

    print("Splitting data into training and evaluation sets (90/10 split)...")
    train_df, eval_df = train_test_split(supervised_df, test_size=0.1, random_state=42, stratify=supervised_df['label'])

    # Ensure output directory exists and save the file
    output_p = Path(output_path)
    output_dir = output_p.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    output_stem = output_p.stem
    train_output_path = output_dir / f"{output_stem}_train.csv"
    eval_output_path = output_dir / f"{output_stem}_eval.csv"

    # Save to separate CSV files
    train_df.to_csv(train_output_path, index=False)
    eval_df.to_csv(eval_output_path, index=False)

    print(f"\nCreated and split {len(supervised_df)} supervised pairs.")
    print(f"Training set: {len(train_df)} pairs, saved to {train_output_path}")
    print(f"Evaluation set: {len(eval_df)} pairs, saved to {eval_output_path}")

if __name__ == "__main__":
    # Assuming 'matches.csv' from your notebook is in 'data/'
    MATCHES_CSV_PATH = "data/matches.csv"
    
    # Define the output path for the supervised pairs dataset
    SUPERVISED_OUTPUT_PATH = "data/finetuning/supervised_pairs.csv"
    
    print("Starting supervised dataset creation...")
    create_supervised_pairs(MATCHES_CSV_PATH, SUPERVISED_OUTPUT_PATH)
    print("Script finished.") 
