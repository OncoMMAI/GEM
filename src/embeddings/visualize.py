# -*- coding: utf-8 -*-

"""
Visualization module for glioma embeddings.

This module provides functionality to visualize embeddings generated from different
models. It supports t-SNE dimensionality reduction and coloring by tumor type.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils.config import get_config
from src.utils.logging import setup_logger, disable_external_loggers


def load_embeddings(embedding_path: Path) -> Optional[np.ndarray]:
    """Load embeddings from a numpy file.

    Args:
        embedding_path: Path to the embedding file

    Returns:
        Numpy array of embeddings or None if loading failed
    """
    logger = logging.getLogger("glioma.visualize")

    try:
        embeddings = np.load(embedding_path)
        logger.info(
            f"Loaded embeddings from {embedding_path}, shape: {embeddings.shape}"
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings from {embedding_path}: {e}")
        return None


def apply_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 2500,
    random_state: int = 42,
    metric: str = "cosine",
) -> np.ndarray:
    """Apply t-SNE dimensionality reduction to embeddings.

    Args:
        embeddings: Input embeddings
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed for reproducibility
        metric: Distance metric for t-SNE

    Returns:
        Reduced 2D embeddings
    """
    logger = logging.getLogger("glioma.visualize")

    # Adjust perplexity based on sample size (following notebook approach)
    n_samples = len(embeddings)
    perplexity = min(n_samples // 3, 50) if perplexity > 50 else perplexity

    logger.info(f"Applying t-SNE with perplexity={perplexity}, n_iter={n_iter}")

    # Scale embeddings for more stable results (same as notebook)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Configure t-SNE (matching notebook parameters)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=12,
        learning_rate=200,  # Notebook uses a fixed learning rate instead of "auto"
        init="pca",  # Both use PCA initialization
        random_state=random_state,
        n_iter=n_iter,
        min_grad_norm=1e-7,
        metric=metric,
        n_iter_without_progress=300,
    )

    # Apply t-SNE transformation
    embeddings_2d = tsne.fit_transform(embeddings_scaled)
    logger.info(f"t-SNE completed with KL divergence: {tsne.kl_divergence_:.4f}")

    return embeddings_2d


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    title: str,
    output_path: Optional[str] = None,
    colors: Optional[list] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 600,
) -> plt.Figure:
    """Create t-SNE visualization of embeddings.

    Args:
        embeddings: Array of embeddings to visualize
        labels: Array of class labels (integers)
        class_names: List of class names corresponding to labels
        title: Title for the plot
        output_path: Optional path to save visualization
        colors: Optional list of colors for classes
        figsize: Figure size as (width, height)
        dpi: DPI for saved figure

    Returns:
        Matplotlib figure
    """
    logger = logging.getLogger("glioma.visualize")

    # Set default colors if not provided
    if colors is None:
        colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", alpha=0.2, color="gray", linewidth=0.5)

    # Plot each class
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if not np.any(mask):
            logger.warning(f"No samples found for class {class_name}")
            continue

        # Scatter points
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[colors[class_idx % len(colors)]],
            label=class_name,
            alpha=0.6,
            s=40,
            edgecolor="none",
        )

        # Add centroid
        centroid = np.mean(embeddings[mask], axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            c="black",
            s=150,
            marker="*",
            edgecolor="white",
            linewidth=1,
            zorder=5,
        )

        # Add confidence ellipse
        if np.sum(mask) > 2:
            cov = np.cov(embeddings[mask, 0], embeddings[mask, 1])
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

            ellip = Ellipse(
                xy=centroid,
                width=2 * 2 * np.sqrt(eigenvals[0]),
                height=2 * 2 * np.sqrt(eigenvals[1]),
                angle=angle,
                facecolor="none",
                edgecolor=colors[class_idx % len(colors)],
                alpha=0.3,
                linestyle="--",
                linewidth=0.75,
            )
            ax.add_patch(ellip)

    # Formatting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    ax.set_title(title, fontsize=11, pad=10, weight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10, labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=9, width=0.75, length=4)

    # Add legend (matching notebook's "Brain Tumor tumor_types")
    legend = ax.legend(
        title="Brain Tumor Subtypes",
        title_fontsize=10,
        fontsize=9,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )
    legend.get_frame().set_linewidth(0.75)

    # Add sample size annotation
    ax.text(
        0.02,
        0.98,
        f"n = {len(embeddings)}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")

    return fig


def create_multi_model_visualization(config: Dict[str, Any]) -> None:
    """Create visualization comparison for multiple embedding models.

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger("glioma.visualize")

    # Get configuration
    embedding_config = config["embedding"]
    data_config = config["data"]
    viz_config = config["visualization"]
    output_config = config["output"]

    # Create output directory (figures folder)
    output_dir = Path(output_config["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output visualizations will be saved to {output_dir}")

    # Get visualization parameters (match notebook values)
    tsne_params = viz_config.get("tsne", {})
    perplexity = tsne_params.get("perplexity", 30)
    n_iter = tsne_params.get("n_iter", 2500)  # Notebook uses 2500
    random_state = tsne_params.get("random_state", 42)
    metric = tsne_params.get("metric", "cosine")

    # Log t-SNE parameters
    logger.info(
        f"t-SNE parameters: perplexity={perplexity}, n_iter={n_iter}, metric={metric}"
    )

    # Get colors and output format (matching notebook)
    colors = viz_config.get("colors", ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"])
    output_format = output_config.get("format", "png")
    dpi = output_config.get("dpi", 600)

    # Load data with labels
    try:
        df = pd.read_csv(data_config["type_path"])
        logger.info(f"Loaded data with {len(df)} samples")

        # Extract labels
        label_col = data_config.get("label_column", "class_label")
        if label_col not in df.columns:
            logger.error(f"Label column '{label_col}' not found in data")
            return

        labels = df[label_col].values

        # Get label names (matching lowercase names in notebook)
        class_names = [
            "glioblastoma",
            "astrocytoma",
            "oligodendroglioma",
            "oligoastrocytoma",
        ]

        # Check if we have all labels
        unique_labels = np.unique(labels)
        logger.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Process each model
    models = embedding_config.get("models", [])

    if not models:
        logger.error("No models defined in configuration")
        return

    # Create a figure for comparison
    n_cols = min(2, len(models))
    n_rows = (len(models) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 10, n_rows * 8), squeeze=False
    )
    axes = axes.flatten()

    all_embeddings_2d = {}

    for i, model_config in enumerate(models):
        model_name = model_config["name"]
        model_path = model_config.get("path")
        model_label = model_config.get("label", model_name)

        logger.info(f"Processing model: {model_name}")

        if not model_path:
            logger.warning(f"No path specified for model {model_name}, skipping")
            continue

        # Load embeddings
        embeddings = load_embeddings(Path(model_path))
        if embeddings is None:
            continue

        # Check if embeddings match data size
        if len(embeddings) != len(df):
            logger.warning(
                f"Embedding count ({len(embeddings)}) doesn't match data count ({len(df)}), "
                f"skipping {model_name}"
            )
            continue

        # Apply t-SNE
        embeddings_2d = apply_tsne(
            embeddings,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            metric=metric,
        )

        all_embeddings_2d[model_name] = embeddings_2d

        # Create individual plot
        output_file = output_dir / f"{model_name}.{output_format}"
        visualize_embeddings(
            embeddings=embeddings_2d,
            labels=labels,
            class_names=class_names,
            title=model_label,
            output_path=str(output_file),
            colors=colors,
            dpi=dpi,
        )

        # Add to comparison plot
        if i < len(axes):
            ax = axes[i]
            for class_idx, class_name in enumerate(class_names):
                mask = labels == class_idx
                if not np.any(mask):
                    continue

                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[colors[class_idx % len(colors)]],
                    label=class_name,
                    alpha=0.6,
                    s=40,
                    edgecolor="none",
                )

                # Add centroid with star
                centroid = np.mean(embeddings_2d[mask], axis=0)
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    c="black",
                    s=150,
                    marker="*",
                    edgecolor="white",
                    linewidth=1,
                    zorder=5,
                )

            ax.set_title(model_label, fontsize=11, pad=10, weight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            # Clean up axes
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Add legend only to first plot
            if i == 0:
                ax.legend(title="Brain Tumor types", title_fontsize=10, fontsize=9)

    # Hide unused subplots
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)

    # Save comparison figure
    comparison_file = output_dir / f"model_comparison.{output_format}"
    plt.tight_layout()
    fig.savefig(comparison_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved model comparison to {comparison_file}")


def main() -> None:
    """Main entry point for visualization module."""
    # Setup logger
    logger = setup_logger(
        name="glioma.visualize", level=logging.INFO, log_file="visualization.log"
    )

    # Disable verbose logging from external libraries
    disable_external_loggers()

    # Load configuration
    config = get_config()

    # Ensure output directory is set to figures
    if "output" not in config:
        config["output"] = {}
    if "output_dir" not in config["output"]:
        config["output"]["output_dir"] = "figures"

    logger.info(f"Ensuring output is directed to: {config['output']['output_dir']}")

    # Set plotting style
    plt.style.use("seaborn-whitegrid")
    sns.set_style("whitegrid")

    # Run visualization
    create_multi_model_visualization(config)

    logger.info("Visualization completed successfully!")


if __name__ == "__main__":
    main()
