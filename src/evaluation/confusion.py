"""
Confusion matrix generation and visualization.

Generates raw and row-normalized confusion matrices with seaborn heatmaps.
"""

import os
import logging
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.
        normalize: None for raw counts, 'true' for row-normalized,
                   'pred' for column-normalized, 'all' for overall.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return cm


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
    label_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    normalize: bool = False,
) -> None:
    """
    Plot confusion matrix as a seaborn heatmap.

    Generates both raw and normalized versions side by side if requested.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.
        label_names: Human-readable label names.
        title: Plot title.
        save_path: Path to save the figure (optional).
        figsize: Figure size.
        normalize: If True, plot row-normalized confusion matrix.
    """
    if label_names is None:
        label_names = [f'Class {i}' for i in range(num_classes)]

    cm_raw = compute_confusion_matrix(y_true, y_pred, num_classes)
    cm_norm = compute_confusion_matrix(y_true, y_pred, num_classes, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # Raw counts
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title(f'{title} (Raw Counts)', fontsize=12)
    axes[0].set_xlabel('Predicted', fontsize=10)
    axes[0].set_ylabel('True', fontsize=10)
    axes[0].tick_params(axis='both', labelsize=8)

    # Row-normalized
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[1], vmin=0, vmax=1, cbar_kws={'shrink': 0.8})
    axes[1].set_title(f'{title} (Row-Normalized)', fontsize=12)
    axes[1].set_xlabel('Predicted', fontsize=10)
    axes[1].set_ylabel('True', fontsize=10)
    axes[1].tick_params(axis='both', labelsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")

    plt.close(fig)


def analyze_adjacent_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
    label_names: Optional[List[str]] = None,
) -> dict:
    """
    Analyze adjacent-class confusion patterns (§13).

    For ordinal sentiment, most errors should be between neighboring classes
    (e.g., 3★↔4★, 4★↔5★). This function quantifies that pattern.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.
        label_names: Human-readable label names.

    Returns:
        Dict with adjacent confusion analysis.
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    total_errors = np.sum(cm) - np.trace(cm)
    if total_errors == 0:
        return {'total_errors': 0, 'adjacent_error_ratio': 1.0}

    # Count adjacent-class errors (off-by-one)
    adjacent_errors = 0
    for i in range(num_classes):
        if i > 0:
            adjacent_errors += cm[i, i-1]  # Predicted one class lower
        if i < num_classes - 1:
            adjacent_errors += cm[i, i+1]  # Predicted one class higher

    # Count far-off errors (off-by-2+)
    far_errors = total_errors - adjacent_errors

    if label_names is None:
        label_names = [f'Class {i}' for i in range(num_classes)]

    # Find worst confusion pairs
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true': label_names[i],
                    'predicted': label_names[j],
                    'count': int(cm[i, j]),
                    'distance': abs(i - j),
                })
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

    return {
        'total_errors': int(total_errors),
        'adjacent_errors': int(adjacent_errors),
        'far_errors': int(far_errors),
        'adjacent_error_ratio': float(adjacent_errors / total_errors) if total_errors > 0 else 0.0,
        'top_confusion_pairs': confusion_pairs[:10],
    }
