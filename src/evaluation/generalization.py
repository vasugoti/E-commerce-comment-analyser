"""
Cross-Dataset Generalization Experiments (§12).

Implements:
a. Amazon -> Yelp (Zero-shot)
b. Amazon -> Flipkart/Women's Clothing (Zero-shot)
c. Few-shot domain adaptation (fine-tune best Tier 3 model on small target sample)
"""

import os
import time
import json
import logging
import argparse
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from ..evaluation.metrics import compute_all_metrics, format_metrics_table, save_metrics
from ..data.label_map import LABEL5_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _load_generalization_data(dataset_name: str, data_dir: str = "data/processed") -> pd.DataFrame:
    """Load a generalization test set."""
    path = os.path.join(data_dir, dataset_name, 'test.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {dataset_name} ({len(df)} samples)")
    return df


def _get_classical_pipeline(min_df: int = 2):
    """Get the standard Tier 1 pipeline (TF-IDF + SVM)."""
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=min_df)
    clf = LinearSVC(C=1.0, class_weight='balanced', max_iter=5000, random_state=42)
    
    return Pipeline([('tfidf', tfidf), ('clf', clf)])


def evaluate_zero_shot(
    source_df: pd.DataFrame, 
    target_df: pd.DataFrame, 
    dataset_name: str,
    output_dir: str = "experiments/generalization",
) -> Dict[str, Any]:
    """
    Evaluate zero-shot transfer from source domain to target domain.
    For local execution, we use the Tier 1 (SVM) baseline.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Zero-Shot Transfer: Amazon → {dataset_name}")
    logger.info(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    
    # Train on source
    logger.info(f"Training on source domain (Amazon, n={len(source_df)})...")
    min_df = 2 if len(source_df) >= 50 else 1
    pipeline = _get_classical_pipeline(min_df=min_df)
    pipeline.fit(source_df['text'].values, source_df['label_5class'].values)

    # Evaluate on target
    logger.info(f"Evaluating on target domain ({dataset_name}, n={len(target_df)})...")
    start = time.time()
    y_pred = pipeline.predict(target_df['text'].values)
    duration = time.time() - start

    metrics = compute_all_metrics(target_df['label_5class'].values, y_pred, num_classes=5, label_names=LABEL5_NAMES)
    
    logger.info(format_metrics_table(metrics, f"Zero-Shot: Amazon → {dataset_name}"))
    save_metrics(metrics, os.path.join(output_dir, f"zeroshot_{dataset_name}_metrics.json"))

    return metrics


def evaluate_few_shot(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    dataset_name: str,
    n_shots: int = 1000,
    output_dir: str = "experiments/generalization",
) -> Dict[str, Any]:
    """
    Evaluate few-shot domain adaptation.
    Uses classical ML for local execution (fine-tuning is simulated by 
    training on the few-shot target sample).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Few-Shot ({n_shots}) Adaptation: Amazon → {dataset_name}")
    logger.info(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Split target data into few-shot training set and test set
    target_df = target_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if len(target_df) <= n_shots:
        raise ValueError(f"Target dataset size ({len(target_df)}) is too small for {n_shots} shots.")
        
    few_shot_df = target_df.iloc[:n_shots]
    test_df = target_df.iloc[n_shots:]
    
    logger.info(f"Reserved {n_shots} samples for few-shot training, {len(test_df)} for testing.")

    # In a real DL scenario, we'd load the source weights and fine-tune.
    # For classical ML, we train a new model on the combined source + few-shot target data.
    # We could weight the few-shot target data higher, but for simplicity we just combine.
    logger.info("Training on source + few-shot target samples...")
    combined_df = pd.concat([source_df, few_shot_df], ignore_index=True)
    
    pipeline = _get_classical_pipeline()
    pipeline.fit(combined_df['text'].values, combined_df['label_5class'].values)

    # Evaluate on remaining target data
    logger.info(f"Evaluating on remaining target data (n={len(test_df)})...")
    y_pred = pipeline.predict(test_df['text'].values)
    
    metrics = compute_all_metrics(test_df['label_5class'].values, y_pred, num_classes=5, label_names=LABEL5_NAMES)
    
    logger.info(format_metrics_table(metrics, f"Few-Shot ({n_shots}): Amazon → {dataset_name}"))
    save_metrics(metrics, os.path.join(output_dir, f"fewshot_{n_shots}_{dataset_name}_metrics.json"))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Generalization Studies (§12)")
    parser.add_argument('--data-dir', type=str, default="data/processed", help="Path to processed data directory")
    args = parser.parse_args()

    logger.info("Starting Cross-Dataset Generalization Suite (§12)")
    
    try:
        # Load source data (Amazon train set)
        amazon_df = pd.read_parquet(os.path.join(args.data_dir, "amazon_combined", "train.parquet"))
        
        # Datasets to evaluate
        targets = ["yelp", "flipkart", "womens_clothing"]
        
        for target in targets:
            try:
                target_df = _load_generalization_data(target, args.data_dir)
                
                # Zero-shot
                evaluate_zero_shot(amazon_df, target_df, target)
                
                # Few-shot (1k)
                if len(target_df) > 1000:
                    evaluate_few_shot(amazon_df, target_df, target, n_shots=1000)
                else:
                    logger.warning(f"Target {target} is too small for 1000-shot adaptation.")
                    
            except FileNotFoundError:
                logger.warning(f"Skipping {target}: dataset not found.")
                continue
                
    except Exception as e:
        logger.error(f"Generalization suite failed: {e}")

if __name__ == "__main__":
    main()
