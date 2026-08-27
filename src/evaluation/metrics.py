"""
Evaluation metrics for sentiment classification.

Implements the full metric suite from §10:
- Macro-F1 (primary metric)
- Weighted-F1
- Accuracy
- Quadratic Weighted Kappa (QWK) — ordinal-aware
- Per-class Precision/Recall/F1
- Macro ROC-AUC (one-vs-rest, for probabilistic models)
"""

import json
import logging
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    roc_auc_score,
    cohen_kappa_score,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 5,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute the full suite of evaluation metrics.

    Args:
        y_true: Ground truth labels (integer array).
        y_pred: Predicted labels (integer array).
        y_prob: Predicted probabilities (shape: [n_samples, num_classes]).
                Required for ROC-AUC. None for hard-prediction-only models.
        num_classes: Number of classes.
        label_names: Human-readable label names.

    Returns:
        Dict with all computed metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics = {}

    # --- Primary: Macro-F1 ---
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    # --- Weighted-F1 ---
    metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # --- Accuracy ---
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

    # --- Quadratic Weighted Kappa (QWK) ---
    # The ordinal-aware metric: penalizes far-off predictions more than adjacent-class confusion
    metrics['qwk'] = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))

    # --- Per-class Precision/Recall/F1 ---
    per_class_precision = precision_score(y_true, y_pred, average=None,
                                          labels=list(range(num_classes)), zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None,
                                    labels=list(range(num_classes)), zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None,
                            labels=list(range(num_classes)), zero_division=0)

    if label_names is None:
        label_names = [f'class_{i}' for i in range(num_classes)]

    metrics['per_class'] = {}
    for i, name in enumerate(label_names):
        metrics['per_class'][name] = {
            'precision': float(per_class_precision[i]) if i < len(per_class_precision) else 0.0,
            'recall': float(per_class_recall[i]) if i < len(per_class_recall) else 0.0,
            'f1': float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
            'support': int(np.sum(y_true == i)),
        }

    # --- Macro ROC-AUC (one-vs-rest) ---
    if y_prob is not None:
        try:
            metrics['macro_roc_auc'] = float(
                roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            )
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['macro_roc_auc'] = None
    else:
        metrics['macro_roc_auc'] = None

    # --- Additional summary stats ---
    metrics['n_samples'] = len(y_true)
    metrics['n_correct'] = int(np.sum(y_true == y_pred))
    metrics['num_classes'] = num_classes

    return metrics


def format_metrics_table(metrics: Dict[str, Any], title: str = "Evaluation Results") -> str:
    """
    Format metrics as a readable table string.

    Args:
        metrics: Output from compute_all_metrics().
        title: Table title.

    Returns:
        Formatted string.
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  {title}")
    lines.append(f"{'='*60}")
    lines.append(f"  {'Metric':<25} {'Value':>10}")
    lines.append(f"  {'-'*40}")
    lines.append(f"  {'Macro-F1 (primary)':<25} {metrics['macro_f1']:>10.4f}")
    lines.append(f"  {'Weighted-F1':<25} {metrics['weighted_f1']:>10.4f}")
    lines.append(f"  {'Accuracy':<25} {metrics['accuracy']:>10.4f}")
    lines.append(f"  {'QWK':<25} {metrics['qwk']:>10.4f}")
    if metrics.get('macro_roc_auc') is not None:
        lines.append(f"  {'Macro ROC-AUC':<25} {metrics['macro_roc_auc']:>10.4f}")
    lines.append(f"  {'-'*40}")
    lines.append(f"  {'Samples':<25} {metrics['n_samples']:>10,}")
    lines.append(f"  {'Correct':<25} {metrics['n_correct']:>10,}")
    lines.append(f"{'='*60}")

    # Per-class breakdown
    if 'per_class' in metrics:
        lines.append(f"\n  Per-Class Breakdown:")
        lines.append(f"  {'Class':<25} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>8}")
        lines.append(f"  {'-'*57}")
        for name, vals in metrics['per_class'].items():
            lines.append(
                f"  {name:<25} {vals['precision']:>7.4f} {vals['recall']:>7.4f} "
                f"{vals['f1']:>7.4f} {vals['support']:>8,}"
            )
        lines.append(f"{'='*60}\n")

    return '\n'.join(lines)


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Save metrics to a JSON file."""
    import os
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {path}")


def load_metrics(path: str) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def aggregate_seed_metrics(seed_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple seeds (mean ± std).

    Args:
        seed_metrics: List of metrics dicts from different seeds.

    Returns:
        Aggregated metrics with mean and std for each scalar metric.
    """
    scalar_keys = ['macro_f1', 'weighted_f1', 'accuracy', 'qwk', 'macro_roc_auc']

    aggregated = {}
    for key in scalar_keys:
        values = [m[key] for m in seed_metrics if m.get(key) is not None]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values,
            }

    # Per-class aggregation
    if 'per_class' in seed_metrics[0]:
        aggregated['per_class'] = {}
        for class_name in seed_metrics[0]['per_class']:
            aggregated['per_class'][class_name] = {}
            for metric_name in ['precision', 'recall', 'f1']:
                values = [m['per_class'][class_name][metric_name] for m in seed_metrics]
                aggregated['per_class'][class_name][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }

    aggregated['n_seeds'] = len(seed_metrics)
    return aggregated


def format_aggregated_metrics(agg: Dict[str, Any], title: str = "Aggregated Results") -> str:
    """Format aggregated (mean ± std) metrics as a readable table."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  {title} (n={agg['n_seeds']} seeds)")
    lines.append(f"{'='*60}")
    lines.append(f"  {'Metric':<25} {'Mean':>10} {'± Std':>10}")
    lines.append(f"  {'-'*47}")

    for key in ['macro_f1', 'weighted_f1', 'accuracy', 'qwk', 'macro_roc_auc']:
        if key in agg:
            lines.append(
                f"  {key:<25} {agg[key]['mean']:>10.4f} ±{agg[key]['std']:>9.4f}"
            )

    lines.append(f"{'='*60}\n")
    return '\n'.join(lines)
