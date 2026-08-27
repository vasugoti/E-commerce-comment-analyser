"""
Tests for the evaluation metrics module (§10).

Validates:
- QWK implementation against a known hand-computed reference calculation
  (QWK is easy to get subtly wrong via the weight matrix)
- Full metrics suite returns expected keys and types
- Edge cases (perfect predictions, completely wrong predictions)
"""

import pytest
import numpy as np
from src.evaluation.metrics import compute_all_metrics
from sklearn.metrics import cohen_kappa_score


def manual_qwk(y_true, y_pred, num_classes=5):
    """
    Hand-computed QWK reference implementation.

    Steps:
    1. Build observed confusion matrix O
    2. Build quadratic weight matrix W: W[i,j] = (i-j)^2 / (K-1)^2
    3. Build expected matrix E from marginal distributions
    4. QWK = 1 - sum(W*O) / sum(W*E)
    """
    # 1. Observed confusion matrix
    O = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1

    # 2. Quadratic weight matrix
    W = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    # 3. Expected matrix from marginals
    hist_true = np.bincount(y_true, minlength=num_classes).astype(float)
    hist_pred = np.bincount(y_pred, minlength=num_classes).astype(float)
    E = np.outer(hist_true, hist_pred) / len(y_true)

    # 4. QWK
    num = np.sum(W * O)
    den = np.sum(W * E)
    return 1.0 - (num / den) if den != 0 else 1.0


class TestQWK:
    """Tests for Quadratic Weighted Kappa correctness."""

    def test_qwk_manual_vs_sklearn(self):
        """Verify our manual QWK formula matches sklearn's cohen_kappa_score."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 3, 4, 1, 1, 1])

        expected_qwk = manual_qwk(y_true, y_pred, num_classes=5)
        sklearn_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

        assert np.isclose(sklearn_qwk, expected_qwk, atol=1e-10), (
            f"Manual QWK {expected_qwk:.6f} != sklearn QWK {sklearn_qwk:.6f}"
        )

    def test_qwk_through_pipeline(self):
        """
        Verify that compute_all_metrics() returns the correct QWK value
        by comparing against the hand-computed reference.
        """
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 1, 2, 1, 2, 3])

        expected_qwk = manual_qwk(y_true, y_pred, num_classes=5)
        metrics = compute_all_metrics(y_true, y_pred, num_classes=5)

        assert np.isclose(metrics['qwk'], expected_qwk, atol=1e-10), (
            f"Pipeline QWK {metrics['qwk']:.6f} != manual QWK {expected_qwk:.6f}"
        )

    def test_qwk_perfect_predictions(self):
        """QWK should be 1.0 for perfect predictions."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2])

        metrics = compute_all_metrics(y_true, y_pred, num_classes=5)
        assert np.isclose(metrics['qwk'], 1.0, atol=1e-10), (
            f"QWK for perfect predictions should be 1.0, got {metrics['qwk']}"
        )

    def test_qwk_penalizes_far_off_more(self):
        """
        QWK should penalize far-off errors more than adjacent errors.
        Predictions off by 1 should yield higher QWK than predictions off by >1.
        """
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

        # Adjacent errors (off by 1)
        y_pred_adjacent = np.array([1, 2, 3, 4, 3, 1, 2, 3, 4, 3])
        qwk_adj = compute_all_metrics(y_true, y_pred_adjacent, num_classes=5)['qwk']

        # Far-off errors (off by 2)
        y_pred_far = np.array([2, 3, 4, 1, 2, 2, 3, 4, 1, 2])
        qwk_far = compute_all_metrics(y_true, y_pred_far, num_classes=5)['qwk']

        assert qwk_adj > qwk_far, (
            f"QWK should penalize far-off errors more: "
            f"adjacent QWK={qwk_adj:.4f} should be > far QWK={qwk_far:.4f}"
        )


class TestComputeAllMetrics:
    """Tests for the full metrics computation suite."""

    def test_returns_expected_keys_and_types(self):
        """Test that the full metrics dictionary returns all expected keys."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])

        # Mock probabilities for 3 classes
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.9, 0.1, 0.0],
            [0.1, 0.2, 0.7],
        ])

        metrics = compute_all_metrics(y_true, y_pred, y_prob=y_prob, num_classes=3)

        # Check all expected keys exist
        assert 'accuracy' in metrics
        assert 'macro_f1' in metrics
        assert 'weighted_f1' in metrics
        assert 'qwk' in metrics
        assert 'macro_roc_auc' in metrics  # Key is 'macro_roc_auc', NOT 'roc_auc'
        assert 'per_class' in metrics
        assert 'n_samples' in metrics
        assert 'n_correct' in metrics
        assert 'num_classes' in metrics

        # Check types
        assert isinstance(metrics['macro_f1'], float)
        assert isinstance(metrics['accuracy'], float)
        assert isinstance(metrics['qwk'], float)
        assert isinstance(metrics['per_class'], dict)

        # Check per-class structure
        assert len(metrics['per_class']) == 3

    def test_no_probabilities_roc_auc_is_none(self):
        """When y_prob is None, macro_roc_auc should be None."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])

        metrics = compute_all_metrics(y_true, y_pred, num_classes=3)
        assert metrics['macro_roc_auc'] is None

    def test_accuracy_value(self):
        """Verify accuracy matches expected value."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])

        metrics = compute_all_metrics(y_true, y_pred, num_classes=3)
        assert np.isclose(metrics['accuracy'], 3 / 5)  # 3 correct out of 5
        assert metrics['n_samples'] == 5
        assert metrics['n_correct'] == 3

    def test_per_class_support(self):
        """Verify per-class support counts are correct."""
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 2])

        label_names = ['neg', 'neu', 'pos']
        metrics = compute_all_metrics(y_true, y_pred, num_classes=3,
                                      label_names=label_names)

        assert metrics['per_class']['neg']['support'] == 3
        assert metrics['per_class']['neu']['support'] == 2
        assert metrics['per_class']['pos']['support'] == 1
