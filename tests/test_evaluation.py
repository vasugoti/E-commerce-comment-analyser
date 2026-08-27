"""
Tests for the evaluation module: confusion matrix generation and error analysis.

Validates:
- Confusion matrix plot generation (raw + normalized)
- Adjacent-class confusion analysis (§13)
- Length-effect binning and error rate computation
"""

import pytest
import numpy as np
import pandas as pd
import os
from src.evaluation.confusion import (
    plot_confusion_matrix,
    compute_confusion_matrix,
    analyze_adjacent_confusion,
)
from src.evaluation.error_analysis import error_analysis_by_length


class TestConfusionMatrix:
    """Tests for confusion matrix generation and visualization."""

    def test_plot_confusion_matrix_creates_file(self, tmp_path):
        """Sanity-check that confusion matrix plot is saved to disk."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 3, 4, 1, 1, 1])

        save_path = tmp_path / "cm.png"
        plot_confusion_matrix(y_true, y_pred, num_classes=5,
                              save_path=str(save_path))

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_compute_confusion_matrix_raw_counts(self):
        """Verify raw confusion matrix counts are correct."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])

        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)

        # Diagonal: correct predictions
        assert cm[0, 0] == 1  # true=0, pred=0
        assert cm[1, 1] == 2  # true=1, pred=1
        assert cm[2, 2] == 1  # true=2, pred=2

        # Off-diagonal: errors
        assert cm[0, 1] == 1  # true=0, pred=1
        assert cm[2, 0] == 1  # true=2, pred=0

        # Total should equal number of samples
        assert cm.sum() == 6

    def test_compute_confusion_matrix_normalized(self):
        """Verify row-normalized confusion matrix sums to 1 per row."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])

        cm_norm = compute_confusion_matrix(y_true, y_pred, num_classes=3,
                                           normalize='true')

        # Each row should sum to 1.0
        for i in range(3):
            assert np.isclose(cm_norm[i].sum(), 1.0, atol=1e-10), (
                f"Row {i} sums to {cm_norm[i].sum()}, expected 1.0"
            )


class TestAdjacentConfusion:
    """Tests for adjacent-class confusion analysis (§13)."""

    def test_adjacent_confusion_basic(self):
        """
        Verify adjacent vs far-off error counting.
        With 5 classes, adjacent means |true - pred| == 1.
        """
        # Designed errors: 2 adjacent (0→1, 2→1) and 1 far (2→0)
        y_true = np.array([0, 1, 2, 3, 4, 0, 2, 2])
        y_pred = np.array([0, 1, 2, 3, 4, 1, 1, 0])
        # Errors: (0→1, dist=1), (2→1, dist=1), (2→0, dist=2)

        result = analyze_adjacent_confusion(y_true, y_pred, num_classes=5)

        assert result['total_errors'] == 3
        assert result['adjacent_errors'] == 2
        assert result['far_errors'] == 1
        assert np.isclose(result['adjacent_error_ratio'], 2 / 3)

    def test_adjacent_confusion_perfect(self):
        """Perfect predictions should have 0 errors and ratio 1.0."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])

        result = analyze_adjacent_confusion(y_true, y_pred, num_classes=5)

        assert result['total_errors'] == 0
        assert result['adjacent_error_ratio'] == 1.0  # Convention for 0 errors

    def test_adjacent_confusion_top_pairs(self):
        """Verify top confusion pairs are sorted by count descending."""
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_pred = np.array([1, 1, 1, 0, 2, 0])
        # Errors: (0→1) x3, (1→0) x1, (1→2) x1, (2→0) x1

        result = analyze_adjacent_confusion(y_true, y_pred, num_classes=3)

        pairs = result['top_confusion_pairs']
        assert len(pairs) > 0
        # First pair should be the most common error
        assert pairs[0]['count'] >= pairs[-1]['count']


class TestLengthEffectAnalysis:
    """Tests for length-effect binning and error rate computation (§13)."""

    def test_length_effect_binning(self):
        """
        Sanity-check that length_effect_analysis bins texts by word count
        and computes error rates per bin.
        """
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        texts = [
            "short",
            "a slightly longer short text here today",
            "this is a medium text " * 10,
            "another medium text " * 15,
            "a very long text that just keeps going " * 20,
            "long text " * 50,
            "super long text " * 100,
            "another super long text " * 100,
        ]

        result = error_analysis_by_length(texts, y_true, y_pred, bins=4)

        # Should return a dict with bin labels as keys
        assert isinstance(result, dict)
        assert len(result) > 0

        # Each bin should have n_samples, n_errors, error_rate
        for bin_label, bin_data in result.items():
            assert 'n_samples' in bin_data
            assert 'n_errors' in bin_data
            assert 'error_rate' in bin_data
            assert 0 <= bin_data['error_rate'] <= 1.0
            assert bin_data['n_samples'] > 0

        # Total samples across all bins should equal input length
        total_samples = sum(v['n_samples'] for v in result.values())
        assert total_samples == len(texts)

    def test_length_effect_all_correct(self):
        """When all predictions are correct, error rates should be 0."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        texts = ["short", "medium text here", "longer text " * 10,
                 "another one", "final text"]

        result = error_analysis_by_length(texts, y_true, y_pred, bins=3)

        for bin_data in result.values():
            assert bin_data['error_rate'] == 0.0
            assert bin_data['n_errors'] == 0
