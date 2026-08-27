"""
Tests for the stratified, group-aware splitting logic (§6).

Verifies BOTH constraints simultaneously:
(a) No product_id (or user_id) appears in more than one split.
(b) The class distribution in each split is within ±3% of the overall distribution.
"""

import pytest
import pandas as pd
import numpy as np
from collections import Counter
from src.data.split import stratified_group_split


def _check_no_group_leakage(train_df, val_df, test_df, group_col):
    """Assert that no group appears in more than one split."""
    train_groups = set(train_df[group_col].unique())
    val_groups = set(val_df[group_col].unique())
    test_groups = set(test_df[group_col].unique())

    assert train_groups.isdisjoint(val_groups), (
        f"Group leakage between train and val on '{group_col}'! "
        f"Overlap: {train_groups & val_groups}"
    )
    assert train_groups.isdisjoint(test_groups), (
        f"Group leakage between train and test on '{group_col}'! "
        f"Overlap: {train_groups & test_groups}"
    )
    assert val_groups.isdisjoint(test_groups), (
        f"Group leakage between val and test on '{group_col}'! "
        f"Overlap: {val_groups & test_groups}"
    )


def _check_stratification(train_df, val_df, test_df, label_col, overall_dist,
                           tolerance=0.03):
    """Assert class distribution in each split is within ±tolerance of overall."""
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        split_size = len(split_df)
        assert split_size > 0, f"{split_name} split is empty!"
        split_counts = Counter(split_df[label_col])
        split_dist = {k: v / split_size for k, v in split_counts.items()}

        for label, overall_prop in overall_dist.items():
            split_prop = split_dist.get(label, 0.0)
            diff = abs(overall_prop - split_prop)
            assert diff <= tolerance, (
                f"{split_name} split has bad stratification for class {label}. "
                f"Overall: {overall_prop:.4f}, Split: {split_prop:.4f}, "
                f"Diff: {diff:.4f} > tolerance {tolerance}"
            )


class TestStratifiedGroupSplit:
    """Test suite for the stratified + group-aware splitting function."""

    def test_3class_product_id_group_and_stratification(self):
        """
        Basic test: 3 classes, 200 product groups, checks BOTH constraints
        with ±3% tolerance.
        """
        np.random.seed(42)
        n_samples = 2000

        groups = np.random.randint(0, 200, n_samples)
        probs = [0.3, 0.6, 0.1]
        labels = np.random.choice([0, 1, 2], p=probs, size=n_samples)

        df = pd.DataFrame({
            'product_id': groups,
            'label_5class': labels,
            'text': ['Review text'] * n_samples,
        })

        overall_dist = {k: v / n_samples for k, v in Counter(labels).items()}

        train_df, val_df, test_df = stratified_group_split(
            df, label_col='label_5class', group_col='product_id',
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=42,
        )

        # Property A: No group leakage
        _check_no_group_leakage(train_df, val_df, test_df, 'product_id')

        # Property B: Stratification within ±3%
        _check_stratification(train_df, val_df, test_df, 'label_5class',
                              overall_dist, tolerance=0.03)

    def test_5class_imbalanced_amazon_like_distribution(self):
        """
        Realistic test: 5-class imbalanced distribution matching Amazon review
        skew (~60-70% are 4-5★). Uses 500 products with ~10 reviews each.
        Checks BOTH constraints with ±3% tolerance.
        """
        np.random.seed(2024)
        n_groups = 500
        reviews_per_group = np.random.randint(5, 20, n_groups)
        n_samples = int(reviews_per_group.sum())

        # Assign groups — each product gets multiple reviews
        groups = np.repeat(np.arange(n_groups), reviews_per_group)

        # Amazon-like skew: 5%/8%/12%/25%/50%
        probs = [0.05, 0.08, 0.12, 0.25, 0.50]
        labels = np.random.choice([0, 1, 2, 3, 4], p=probs, size=n_samples)

        df = pd.DataFrame({
            'product_id': [f'prod_{g}' for g in groups],
            'label_5class': labels,
            'text': ['Review text'] * n_samples,
        })

        overall_dist = {k: v / n_samples for k, v in Counter(labels).items()}

        train_df, val_df, test_df = stratified_group_split(
            df, label_col='label_5class', group_col='product_id',
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=13,
        )

        # Property A: No product_id leakage
        _check_no_group_leakage(train_df, val_df, test_df, 'product_id')

        # Property B: Stratification within ±3%
        _check_stratification(train_df, val_df, test_df, 'label_5class',
                              overall_dist, tolerance=0.03)

        # Sanity: all samples accounted for
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == n_samples, (
            f"Sample count mismatch: {total} vs {n_samples}"
        )

    def test_user_id_group_no_leakage(self):
        """
        Test group-aware splitting on user_id instead of product_id.
        Some datasets have user_id as the leakage-prevention column.
        """
        np.random.seed(13)
        n_users = 300
        reviews_per_user = np.random.randint(3, 15, n_users)
        n_samples = int(reviews_per_user.sum())

        users = np.repeat(np.arange(n_users), reviews_per_user)
        labels = np.random.choice([0, 1, 2, 3, 4], size=n_samples)

        df = pd.DataFrame({
            'user_id': [f'user_{u}' for u in users],
            'label_5class': labels,
            'text': ['Review'] * n_samples,
        })

        overall_dist = {k: v / n_samples for k, v in Counter(labels).items()}

        train_df, val_df, test_df = stratified_group_split(
            df, label_col='label_5class', group_col='user_id',
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=42,
        )

        # Property A: No user_id leakage
        _check_no_group_leakage(train_df, val_df, test_df, 'user_id')

        # Property B: Stratification within ±3%
        _check_stratification(train_df, val_df, test_df, 'label_5class',
                              overall_dist, tolerance=0.03)

    def test_no_groups_falls_back_to_stratified_only(self):
        """
        When no valid group column exists, should fall back to
        StratifiedShuffleSplit and still maintain class proportions.
        """
        np.random.seed(42)
        n_samples = 1000
        labels = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.1, 0.2, 0.3, 0.3],
                                  size=n_samples)

        df = pd.DataFrame({
            'product_id': 'unknown',  # All unknown — triggers fallback
            'label_5class': labels,
            'text': ['Review'] * n_samples,
        })

        overall_dist = {k: v / n_samples for k, v in Counter(labels).items()}

        train_df, val_df, test_df = stratified_group_split(
            df, label_col='label_5class', group_col='product_id',
            random_state=42,
        )

        # Only check stratification — no meaningful groups to test
        _check_stratification(train_df, val_df, test_df, 'label_5class',
                              overall_dist, tolerance=0.03)
