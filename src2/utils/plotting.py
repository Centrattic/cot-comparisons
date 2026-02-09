"""Training curve plotting utilities."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(
    history: dict,
    metric_name: str,
    output_path: Path,
    title: str = None,
):
    """Plot train and val/test metric vs epoch and save to file.

    Args:
        history: dict with keys "epoch", "train_{metric}", and
                 "val_{metric}" or "test_{metric}".
        metric_name: base name of the metric (e.g., "r2", "f1", "accuracy").
        output_path: path to save the plot.
        title: optional plot title.
    """
    epochs = history["epoch"]
    train_key = f"train_{metric_name}"

    # Find val or test key
    val_key = None
    for k in [f"val_{metric_name}", f"test_{metric_name}"]:
        if k in history:
            val_key = k
            break

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history[train_key], "b-o", markersize=3, label=f"Train")
    if val_key:
        label = "Val" if "val" in val_key else "Test"
        ax.plot(epochs, history[val_key], "r-o", markersize=3, label=label)

    ax.set_xlabel("Epoch")
    ylabel = metric_name.replace("_", " ")
    if len(metric_name) <= 3:
        ylabel = ylabel.upper()
    else:
        ylabel = ylabel.capitalize()
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved to {output_path}")


def plot_data_scaling(
    sizes: list,
    scores: list,
    metric_name: str,
    output_path: Path,
    title: str = None,
):
    """Plot test metric vs training data size and save to file.

    Args:
        sizes: list of training set sizes.
        scores: list of test metric values.
        metric_name: name of the metric (e.g., "r2", "f1").
        output_path: path to save the plot.
        title: optional plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, scores, "b-o", markersize=5)
    ax.set_xlabel("Training set size")
    ylabel = metric_name.replace("_", " ")
    if len(metric_name) <= 3:
        ylabel = ylabel.upper()
    else:
        ylabel = ylabel.capitalize()
    ax.set_ylabel(f"Test {ylabel}")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Data scaling plot saved to {output_path}")
