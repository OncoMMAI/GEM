"""
Visualization script for LLM2Vec embeddings across different training stages.

This script creates t-SNE visualizations for embeddings from:
1. Stage 1: MNTP Only
2. Stage 2: MNTP + SimCSE
3. Stage 3: Full Finetune

The visualizations show tumor type classification to observe progression changes.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import yaml


def setup_logging():
    """Sets up a basic logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str) -> dict:
    """Loads the YAML configuration file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {path}")
        raise
    except Exception as e:
        logging.error(f"Error reading YAML configuration: {e}")
        raise


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
    logger = logging.getLogger(__name__)

    # Check for NaN values
    if np.any(np.isnan(embeddings)):
        logger.warning("Found NaN values in embeddings, removing affected samples")
        valid_mask = ~np.any(np.isnan(embeddings), axis=1)
        embeddings = embeddings[valid_mask]
        logger.info(f"Kept {np.sum(valid_mask)} valid samples out of {len(valid_mask)}")

    # Adjust perplexity based on sample size
    n_samples = len(embeddings)
    perplexity = min(n_samples // 3, 50) if perplexity > 50 else perplexity

    logger.info(f"Applying t-SNE with perplexity={perplexity}, n_iter={n_iter}")

    # Scale embeddings for more stable results
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Configure t-SNE (using max_iter for newer sklearn versions)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=12,
        learning_rate=200,
        init="pca",
        random_state=random_state,
        max_iter=n_iter,  # Changed from n_iter to max_iter
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
        labels: Array of class labels
        class_names: List of class names corresponding to labels
        title: Title for the plot
        output_path: Optional path to save visualization
        colors: Optional list of colors for classes
        figsize: Figure size as (width, height)
        dpi: DPI for saved figure

    Returns:
        Matplotlib figure
    """
    logger = logging.getLogger(__name__)

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
        mask = labels == class_name
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

    # Add legend
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


def main(config_path: str):
    """Main entry point for LLM2Vec embedding visualization."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(config_path)

    # Create output directory
    output_dir = Path("figures/llm2vec_embeddings")
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output visualizations will be saved to {output_dir}")

    # Define the three stages to visualize
    stages = {
        "Stage 1: MNTP Only": "Stage 1: MNTP Only",
        "Stage 2: MNTP + SimCSE": "Stage 2: MNTP + SimCSE",
        "Stage 3: Full Finetune": "Stage 3: Full Finetune",
    }

    # Load tumor type data
    df = pd.read_csv(config["data"]["type_path"])
    labels_all = df["tumor_type"].values

    # --- Filter out the oligoastrocytoma class (class label == "oligoastrocytoma") ---
    filter_mask = labels_all != "oligoastrocytoma"
    labels = labels_all[filter_mask]

    class_names = [
        "glioblastoma",  # 0
        "astrocytoma",  # 1
        "oligodendroglioma",  # 2
    ]

    colors = ["#E64B35", "#4DBBD5", "#00A087"]

    # t-SNE parameters
    tsne_params = {
        "perplexity": 50,
        "n_iter": 2500,
        "random_state": 42,
        "metric": "cosine",
    }

    # Path to embeddings
    embeddings_base_path = Path(config["output"]["results_dir"]) / "embeddings"

    # Create individual plots for each stage
    for stage_name, model_name in stages.items():
        logger.info(f"Processing {stage_name}")

        # Load embeddings
        embedding_path = (
            embeddings_base_path / f"{model_name}_tumor_type_embeddings.npy"
        )

        if not embedding_path.exists():
            logger.warning(
                f"Embeddings not found at {embedding_path}, skipping {stage_name}"
            )
            continue

        # Load embeddings
        embeddings = np.load(embedding_path)

        # Check if embeddings are already filtered (from run_full_evaluation.py)
        if embeddings.shape[0] == len(filter_mask):
            # Embeddings match full dataset size, apply filter
            embeddings = embeddings[filter_mask]
            logger.info(
                f"Applied oligoastrocytoma filter to embeddings for {stage_name}"
            )
        elif embeddings.shape[0] == np.sum(filter_mask):
            # Embeddings are already filtered, no need to apply filter again
            logger.info(f"Embeddings for {stage_name} are already filtered")
        else:
            logger.error(
                f"Embedding size mismatch for {stage_name}: got {embeddings.shape[0]}, expected {len(filter_mask)} or {np.sum(filter_mask)}"
            )
            continue

        logger.info(f"Loaded embeddings for {stage_name}, shape: {embeddings.shape}")

        # Check for NaN values and create valid mask
        valid_mask = ~np.any(np.isnan(embeddings), axis=1)
        if not np.all(valid_mask):
            nan_count = np.sum(~valid_mask)
            logger.warning(
                f"Found {nan_count} samples with NaN values in {stage_name} ({nan_count / len(embeddings) * 100:.1f}%)"
            )
            logger.info(
                f"Filtering out NaN samples, keeping {np.sum(valid_mask)} valid samples"
            )
            embeddings_valid = embeddings[valid_mask]
            labels_valid = labels[valid_mask]
        else:
            embeddings_valid = embeddings
            labels_valid = labels

        # Apply t-SNE
        embeddings_2d = apply_tsne(embeddings_valid, **tsne_params)

        # Create visualization
        # Create a safe filename by replacing special characters
        safe_filename = model_name.replace(":", "").replace(" ", "_")
        output_path = output_dir / f"{safe_filename}_tsne.png"
        visualize_embeddings(
            embeddings=embeddings_2d,
            labels=labels_valid,
            class_names=class_names,
            title=stage_name,
            output_path=str(output_path),
            colors=colors,
            figsize=(10, 8),
            dpi=600,
        )

    # Create comparison plot with all three stages
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    fig.patch.set_facecolor("white")

    for idx, (stage_name, model_name) in enumerate(stages.items()):
        ax = axes[idx]

        # Load embeddings
        embedding_path = (
            embeddings_base_path / f"{model_name}_tumor_type_embeddings.npy"
        )
        if not embedding_path.exists():
            ax.set_visible(False)
            continue

        # Load embeddings and apply tumor-type filter
        embeddings = np.load(embedding_path)

        # Check if embeddings are already filtered (from run_full_evaluation.py)
        if embeddings.shape[0] == len(filter_mask):
            # Embeddings match full dataset size, apply filter
            embeddings = embeddings[filter_mask]
        elif embeddings.shape[0] == np.sum(filter_mask):
            # Embeddings are already filtered, no need to apply filter again
            pass
        else:
            logger.error(
                f"Embedding size mismatch for {stage_name}: got {embeddings.shape[0]}, expected {len(filter_mask)} or {np.sum(filter_mask)}"
            )
            ax.set_visible(False)
            continue

        # Check for NaN values and create valid mask
        valid_mask = ~np.any(np.isnan(embeddings), axis=1)
        if not np.all(valid_mask):
            embeddings_valid = embeddings[valid_mask]
            labels_valid = labels[valid_mask]
        else:
            embeddings_valid = embeddings
            labels_valid = labels

        embeddings_2d = apply_tsne(embeddings_valid, **tsne_params)

        # Plot on subplot
        ax.set_facecolor("white")
        ax.grid(True, linestyle="-", alpha=0.2, color="gray", linewidth=0.5)

        for class_idx, class_name in enumerate(class_names):
            mask = labels_valid == class_name
            if not np.any(mask):
                continue

            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[class_idx]],
                label=class_name,
                alpha=0.6,
                s=40,
                edgecolor="none",
            )

            # Add centroid
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

        ax.set_title(stage_name, fontsize=14, pad=10, weight="bold")
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add legend to first subplot only
        if idx == 0:
            ax.legend(
                title="Brain Tumor Subtypes",
                title_fontsize=12,
                fontsize=11,
                loc="best",
                frameon=True,
                edgecolor="black",
                fancybox=False,
            )

    plt.tight_layout()
    comparison_path = output_dir / "llm2vec_stages_comparison.png"
    plt.savefig(comparison_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved comparison plot to {comparison_path}")
    logger.info("Visualization completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize LLM2Vec embeddings across training stages."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)
