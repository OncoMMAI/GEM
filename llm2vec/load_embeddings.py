"""
Helper script to load saved embeddings from the evaluation pipeline.
"""

import json
import numpy as np
from pathlib import Path


def load_embedding_set(embeddings_dir, model_name, task_name):
    """
    Load a complete embedding set including embeddings, labels, texts, and metadata.
    
    Args:
        embeddings_dir: Path to the embeddings directory
        model_name: Name of the model (e.g., 'clinical_mntp_only')
        task_name: Name of the task (e.g., 'tumor_type')
    
    Returns:
        dict: Dictionary containing embeddings, labels, texts, and metadata
    """
    embeddings_dir = Path(embeddings_dir)
    
    # Load embeddings
    embeddings = np.load(embeddings_dir / f"{model_name}_{task_name}_embeddings.npy")
    
    # Load labels
    labels = np.load(embeddings_dir / f"{model_name}_{task_name}_labels.npy")
    
    # Load texts
    with open(embeddings_dir / f"{model_name}_{task_name}_texts.json", 'r') as f:
        texts = json.load(f)
    
    # Load metadata
    with open(embeddings_dir / f"{model_name}_{task_name}_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return {
        'embeddings': embeddings,
        'labels': labels,
        'texts': texts,
        'metadata': metadata
    }


def list_available_embeddings(embeddings_dir):
    """List all available embedding sets in the directory."""
    embeddings_dir = Path(embeddings_dir)
    
    # Find all metadata files
    metadata_files = list(embeddings_dir.glob("*_metadata.json"))
    
    available = []
    for meta_file in metadata_files:
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        available.append({
            'model_name': metadata['model_name'],
            'task_name': metadata['task_name'],
            'num_samples': metadata['num_samples'],
            'embedding_dim': metadata['embedding_dim']
        })
    
    return available


if __name__ == "__main__":
    # Example usage
    embeddings_dir = "./results/embeddings"  # Adjust path as needed
    
    # List available embeddings
    print("Available embeddings:")
    for emb_info in list_available_embeddings(embeddings_dir):
        print(f"  - {emb_info['model_name']} / {emb_info['task_name']}: "
              f"{emb_info['num_samples']} samples, {emb_info['embedding_dim']} dims")
    
    # Load a specific embedding set
    print("\nLoading example embedding set...")
    data = load_embedding_set(embeddings_dir, "clinical_mntp_only", "tumor_type")
    
    print(f"Embeddings shape: {data['embeddings'].shape}")
    print(f"Labels shape: {data['labels'].shape}")
    print(f"Number of texts: {len(data['texts'])}")
    print(f"Model checkpoint: {data['metadata']['checkpoint']}")
    print(f"Embedding dimension: {data['metadata']['embedding_dim']}") 
