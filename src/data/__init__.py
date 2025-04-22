"""
Data processing package for glioma text analysis.

This package provides tools to process, transform, and prepare data
for the glioma classification and embedding pipeline.
"""

from .data_pipeline import (
    load_raw_data,
    create_training_datasets,
    create_evaluation_datasets,
    process_pipeline
)

__all__ = [
    "load_raw_data",
    "create_training_datasets",
    "create_evaluation_datasets",
    "process_pipeline"
]