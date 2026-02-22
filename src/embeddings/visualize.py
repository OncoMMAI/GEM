# -*- coding: utf-8 -*-

"""
Visualization module for glioma embeddings.

This module provides functionality to visualize embeddings generated from different
models. It supports t-SNE dimensionality reduction and coloring by tumor type.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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
        learning_rate=200,
        init="pca",
        random_state=random_state,
        max_iter=n_iter,
        min_grad_norm=1e-7,
        metric=metric,
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
    class_values: Optional[list] = None,
    legend_title: str = "Classes",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 600,
) -> plt.Figure:
    """Create t-SNE visualization of embeddings.

    Args:
        embeddings: Array of embeddings to visualize
        labels: Array of class labels (strings or integers)
        class_names: Display names for each class in the legend
        title: Title for the plot
        output_path: Optional path to save visualization
        colors: Optional list of colors for classes
        class_values: Values to match in ``labels`` for each class.  When
            ``None`` the ``class_names`` themselves are used (works when labels
            are strings).  Pass explicit values when the label column contains
            integers (e.g. ``[0, 1]`` for binary survival).
        legend_title: Title for the legend
        figsize: Figure size as (width, height)
        dpi: DPI for saved figure

    Returns:
        Matplotlib figure
    """
    logger = logging.getLogger("glioma.visualize")

    if colors is None:
        colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]

    if class_values is None:
        class_values = class_names

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", alpha=0.2, color="gray", linewidth=0.5)

    for class_idx, (class_val, class_name) in enumerate(
        zip(class_values, class_names)
    ):
        mask = labels == class_val
        if not np.any(mask):
            logger.warning(f"No samples found for class {class_name}")
            continue

        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[colors[class_idx % len(colors)]],
            label=class_name,
            alpha=0.6,
            s=40,
            edgecolor="none",
        )

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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    ax.set_title(title, fontsize=11, pad=10, weight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10, labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=9, width=0.75, length=4)

    legend = ax.legend(
        title=legend_title,
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
    """Create visualization comparison for multiple embedding models across tasks.

    Iterates over all tasks defined in the config (e.g. tumor_type, mgmt,
    survival) and generates per-model individual plots plus a combined
    comparison grid for each task.

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger("glioma.visualize")

    viz_config = config["visualization"]
    output_config = config["output"]
    tasks_config = config["tasks"]

    output_dir = Path(output_config["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output visualizations will be saved to {output_dir}")

    tsne_params = viz_config.get("tsne", {})
    perplexity = tsne_params.get("perplexity", 30)
    n_iter = tsne_params.get("n_iter", 2500)
    random_state = tsne_params.get("random_state", 42)
    metric = tsne_params.get("metric", "cosine")
    logger.info(
        f"t-SNE parameters: perplexity={perplexity}, n_iter={n_iter}, metric={metric}"
    )

    output_format = output_config.get("format", "png")
    dpi = output_config.get("dpi", 600)

    for task_name, task_cfg in tasks_config.items():
        logger.info(f"=== Processing task: {task_name} ===")

        data_path = task_cfg["data_path"]
        label_col = task_cfg["label_column"]
        class_names = task_cfg["class_names"]
        class_values = task_cfg.get("class_values", class_names)
        legend_title = task_cfg.get("legend_title", "Classes")
        colors = task_cfg.get(
            "colors", ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]
        )
        models = task_cfg.get("models", [])

        if not models:
            logger.warning(f"No models defined for task {task_name}, skipping")
            continue

        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} samples from {data_path}")
        except Exception as e:
            logger.error(f"Failed to load data for {task_name}: {e}")
            continue

        if label_col not in df.columns:
            logger.error(
                f"Label column '{label_col}' not found in {data_path}, skipping"
            )
            continue

        labels_raw = df[label_col].values
        valid_mask = pd.notna(df[label_col])
        labels = labels_raw[valid_mask]
        n_total = len(labels_raw)
        n_valid = int(valid_mask.sum())
        if n_valid < n_total:
            logger.info(
                f"Dropped {n_total - n_valid} rows with NaN labels "
                f"({n_valid} valid out of {n_total})"
            )

        unique_labels = np.unique(labels)
        logger.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")

        task_output_dir = output_dir / task_name
        task_output_dir.mkdir(exist_ok=True, parents=True)

        n_cols = min(2, len(models))
        n_rows = (len(models) + n_cols - 1) // n_cols
        fig_cmp, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 10, n_rows * 8), squeeze=False
        )
        axes = axes.flatten()

        plot_idx = 0
        for model_config in models:
            model_name = model_config["name"]
            model_path = model_config.get("path")
            model_label = model_config.get("label", model_name)

            logger.info(f"Processing model: {model_name} for task {task_name}")

            if not model_path:
                logger.warning(f"No path specified for model {model_name}, skipping")
                continue

            embeddings = load_embeddings(Path(model_path))
            if embeddings is None:
                continue

            # Apply the same NaN-label filter to embeddings
            if len(embeddings) == n_total and n_valid < n_total:
                embeddings = embeddings[valid_mask]
            elif len(embeddings) != len(labels):
                logger.warning(
                    f"Embedding count ({len(embeddings)}) doesn't match label count "
                    f"({len(labels)}), skipping {model_name}"
                )
                continue

            embeddings_2d = apply_tsne(
                embeddings,
                perplexity=perplexity,
                n_iter=n_iter,
                random_state=random_state,
                metric=metric,
            )

            output_file = task_output_dir / f"{model_name}.{output_format}"
            visualize_embeddings(
                embeddings=embeddings_2d,
                labels=labels,
                class_names=class_names,
                class_values=class_values,
                title=model_label,
                output_path=str(output_file),
                colors=colors,
                legend_title=legend_title,
                dpi=dpi,
            )

            if plot_idx < len(axes):
                ax = axes[plot_idx]
                for class_idx, (class_val, class_name) in enumerate(
                    zip(class_values, class_names)
                ):
                    mask = labels == class_val
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

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

                if plot_idx == 0:
                    ax.legend(
                        title=legend_title, title_fontsize=10, fontsize=9
                    )

            plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        comparison_file = task_output_dir / f"model_comparison.{output_format}"
        plt.tight_layout()
        fig_cmp.savefig(comparison_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig_cmp)
        logger.info(f"Saved {task_name} model comparison to {comparison_file}")


def main() -> None:
    """Main entry point for visualization module."""
    logger = setup_logger(
        name="glioma.visualize", level=logging.INFO, log_file="visualization.log"
    )
    disable_external_loggers()

    config = get_config()

    if "output" not in config:
        config["output"] = {}
    if "output_dir" not in config["output"]:
        config["output"]["output_dir"] = "figures"

    logger.info(f"Output directory: {config['output']['output_dir']}")

    create_multi_model_visualization(config)

    logger.info("Visualization completed successfully!")


if __name__ == "__main__":
    main()
