"""
Error analysis module.

Implements the error analysis plan from §13:
- Adjacent-class confusion patterns
- Error rate vs. review length
- Category-wise breakdown
- Qualitative misclassification sampling
"""

import os
import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def error_analysis_by_length(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 10,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze error rate as a function of review length (§13).

    Very short reviews carry little signal; very long ones may have
    polarity shifts mid-review.

    Args:
        texts: List of review texts.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        bins: Number of length bins.
        save_path: Path to save the plot.

    Returns:
        Dict with length-based error analysis.
    """
    word_counts = np.array([len(t.split()) for t in texts])
    is_correct = (y_true == y_pred).astype(int)
    is_error = 1 - is_correct

    # Create length bins
    bin_edges = np.percentile(word_counts, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(len(bin_edges)-1)]

    bin_indices = np.digitize(word_counts, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)

    results = {}
    error_rates = []
    for i, label in enumerate(bin_labels):
        mask = bin_indices == i
        n_total = mask.sum()
        n_errors = is_error[mask].sum()
        error_rate = n_errors / n_total if n_total > 0 else 0.0
        results[label] = {
            'n_samples': int(n_total),
            'n_errors': int(n_errors),
            'error_rate': float(error_rate),
        }
        error_rates.append(error_rate)

    # Plot
    if save_path:
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = range(len(bin_labels))
        bars = ax.bar(x_pos, error_rates, color='steelblue', alpha=0.8)
        ax.set_xlabel('Review Length (word count range)', fontsize=11)
        ax.set_ylabel('Error Rate', fontsize=11)
        ax.set_title('Error Rate vs. Review Length', fontsize=13)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add count annotations
        for bar, label in zip(bars, bin_labels):
            n = results[label]['n_samples']
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'n={n}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved length error analysis plot to {save_path}")

    return results


def error_analysis_by_category(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    category_col: str = 'category',
) -> Dict[str, Dict[str, float]]:
    """
    Analyze error rate by product category (§13).

    Args:
        df: DataFrame with category column.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        category_col: Name of the category column.

    Returns:
        Dict mapping categories to error statistics.
    """
    is_correct = (y_true == y_pred)

    results = {}
    for cat in df[category_col].unique():
        mask = (df[category_col] == cat).values
        n_total = mask.sum()
        n_correct = is_correct[mask].sum()
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        results[str(cat)] = {
            'n_samples': int(n_total),
            'accuracy': float(accuracy),
            'error_rate': float(1.0 - accuracy),
        }

    return results


def sample_misclassifications(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int = 50,
    label_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample misclassified examples for qualitative review (§13).

    Randomly samples from misclassified examples, stratified by
    error type (adjacent vs. far-off).

    Args:
        texts: List of review texts.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        n_samples: Number of samples to return.
        label_names: Human-readable label names.
        random_state: Random seed.

    Returns:
        DataFrame with columns: text, true_label, predicted_label, distance, text_length.
    """
    rng = np.random.RandomState(random_state)

    errors_mask = y_true != y_pred
    error_indices = np.where(errors_mask)[0]

    if len(error_indices) == 0:
        return pd.DataFrame(columns=['text', 'true_label', 'predicted_label', 'distance', 'text_length'])

    if label_names is None:
        label_names = [f'Class {i}' for i in range(max(max(y_true), max(y_pred)) + 1)]

    # Sample
    n_take = min(n_samples, len(error_indices))
    sampled_indices = rng.choice(error_indices, size=n_take, replace=False)

    rows = []
    for idx in sampled_indices:
        true_val = int(y_true[idx])
        pred_val = int(y_pred[idx])
        rows.append({
            'text': texts[idx][:500],  # Truncate for readability
            'true_label': label_names[true_val] if true_val < len(label_names) else str(true_val),
            'predicted_label': label_names[pred_val] if pred_val < len(label_names) else str(pred_val),
            'distance': abs(true_val - pred_val),
            'text_length': len(texts[idx].split()),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('distance', ascending=False)
    return df


def full_error_analysis(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
    label_names: Optional[List[str]] = None,
    df: Optional[pd.DataFrame] = None,
    category_col: str = 'category',
    output_dir: Optional[str] = None,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Run the full error analysis pipeline (§13).

    Args:
        texts: List of review texts.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.
        label_names: Human-readable label names.
        df: DataFrame with metadata (for category analysis).
        category_col: Name of the category column.
        output_dir: Directory to save plots and results.
        model_name: Name of the model for file naming.

    Returns:
        Dict with all error analysis results.
    """
    results = {}

    # Length analysis
    length_save_path = os.path.join(output_dir, f'{model_name}_error_by_length.png') if output_dir else None
    results['by_length'] = error_analysis_by_length(
        texts, y_true, y_pred, save_path=length_save_path
    )

    # Category analysis (if available)
    if df is not None and category_col in df.columns:
        results['by_category'] = error_analysis_by_category(df, y_true, y_pred, category_col)

    # Misclassification samples
    misclass_df = sample_misclassifications(
        texts, y_true, y_pred, n_samples=100, label_names=label_names
    )
    results['misclassification_samples'] = misclass_df.to_dict('records')

    if output_dir:
        misclass_path = os.path.join(output_dir, f'{model_name}_misclassifications.csv')
        os.makedirs(output_dir, exist_ok=True)
        misclass_df.to_csv(misclass_path, index=False)
        logger.info(f"Saved misclassifications to {misclass_path}")

    return results
