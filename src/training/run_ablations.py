"""
Runner for the 7 ablation studies defined in §11 of the research plan.

Each ablation is runnable via:
    python -m src.training.run_ablations --ablation <name>

Where <name> is one of:
    label_granularity, preprocessing, embeddings, loss_function,
    finetuning_depth, sequence_length, data_volume

All results are logged in the same run-log format as the main experiments (§9).
"""

import os
import sys
import copy
import json
import time
import yaml
import logging
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .seed import set_seed, DEFAULT_SEEDS
from .trainer import ExperimentLogger, train_classical_model, train_dl_model, create_data_loaders
from ..evaluation.metrics import compute_all_metrics, format_metrics_table, save_metrics
from ..data.label_map import (
    LABEL5_NAMES, LABEL3_NAMES,
    get_class_weights, label5_to_label2,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

LABEL2_NAMES = ['negative', 'positive']

ABLATION_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'ablations')


# ---------------------------------------------------------------------------
# Utility: load ablation config YAML
# ---------------------------------------------------------------------------

def load_ablation_config(ablation_name: str) -> Dict[str, Any]:
    """Load an ablation config YAML file."""
    path = os.path.join(ABLATION_CONFIGS_DIR, f'{ablation_name}.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ablation config not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _load_split_data(data_dir: str = "data/processed/amazon_combined"):
    """Load the pre-split Amazon combined data."""
    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'val.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    return train_df, val_df, test_df


def _get_labels(df, label_scheme, label_col_5='label_5class', label_col_3='label_3class'):
    """Extract labels for the given scheme, handling 2-class filtering."""
    if label_scheme == '5class':
        return df[label_col_5].values, 5, LABEL5_NAMES
    elif label_scheme == '3class':
        return df[label_col_3].values, 3, LABEL3_NAMES
    elif label_scheme == '2class':
        # Filter out neutral (label_5class == 2) and remap
        mask = df[label_col_5] != 2
        filtered = df[mask].copy()
        labels_2 = filtered[label_col_5].apply(label5_to_label2).values
        return labels_2, 2, LABEL2_NAMES
    else:
        raise ValueError(f"Unknown label scheme: {label_scheme}")


def _filter_for_2class(df):
    """Filter DataFrame to exclude neutral reviews for 2-class ablation."""
    return df[df['label_5class'] != 2].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ablation A: Label Granularity
# ---------------------------------------------------------------------------

def run_label_granularity_ablation(data_dir: str = "data/processed/amazon_combined"):
    """
    Ablation A (§11.1): 5-class vs 3-class vs 2-class comparison.

    Trains SVM baseline with each label scheme and compares macro-F1.
    """
    config = load_ablation_config('label_granularity')
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    exp_logger = ExperimentLogger(output_dir)

    logger.info("=" * 60)
    logger.info("  ABLATION A: Label Granularity (5-class vs 3-class vs 2-class)")
    logger.info("=" * 60)

    train_df, val_df, test_df = _load_split_data(data_dir)

    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer

    for variant in config['variants']:
        scheme = variant['label_scheme']
        num_classes = variant['num_classes']
        variant_name = variant['name']
        label_names = {5: LABEL5_NAMES, 3: LABEL3_NAMES, 2: LABEL2_NAMES}[num_classes]

        logger.info(f"\n--- Variant: {variant_name} ({num_classes} classes) ---")

        # Get labels (may filter for 2-class)
        if scheme == '2class':
            tr_df = _filter_for_2class(train_df)
            va_df = _filter_for_2class(val_df)
            te_df = _filter_for_2class(test_df)
        else:
            tr_df, va_df, te_df = train_df, val_df, test_df

        y_train, _, _ = _get_labels(tr_df, scheme)
        y_val, _, _ = _get_labels(va_df, scheme)
        y_test, _, _ = _get_labels(te_df, scheme)

        # TF-IDF
        tfidf = TfidfVectorizer(
            max_features=config['data']['max_features_tfidf'],
            ngram_range=(1, 2), sublinear_tf=True, min_df=2,
        )
        X_train = tfidf.fit_transform(tr_df['text'].values)
        X_val = tfidf.transform(va_df['text'].values)
        X_test = tfidf.transform(te_df['text'].values)

        for seed in config['training']['seeds']:
            set_seed(seed, deterministic=False)
            start = time.time()

            clf = LinearSVC(C=config['model']['svm_C'],
                            class_weight=config['model']['svm_class_weight'],
                            max_iter=5000, random_state=seed)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            metrics = compute_all_metrics(y_test, y_pred, num_classes=num_classes,
                                          label_names=label_names)
            duration = time.time() - start

            logger.info(format_metrics_table(
                metrics, f"Label Granularity — {variant_name} (Seed {seed})"))

            save_metrics(metrics, os.path.join(
                output_dir, f"{variant_name}_seed{seed}_metrics.json"))

            exp_logger.log_run(
                config={'ablation': 'label_granularity', 'variant': variant_name,
                        'num_classes': num_classes},
                metrics=metrics, seed=seed, duration_seconds=duration,
            )


# ---------------------------------------------------------------------------
# Ablation B: Preprocessing
# ---------------------------------------------------------------------------

def run_preprocessing_ablation(data_dir: str = "data/processed/amazon_combined"):
    """
    Ablation B (§11.2): Preprocessing variations.

    Tests: with/without stopword removal, with/without lemmatization,
    unigram vs uni+bigram TF-IDF.
    """
    config = load_ablation_config('preprocessing')
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    exp_logger = ExperimentLogger(output_dir)

    logger.info("=" * 60)
    logger.info("  ABLATION B: Preprocessing Variations")
    logger.info("=" * 60)

    train_df, val_df, test_df = _load_split_data(data_dir)
    y_train = train_df['label_5class'].values
    y_test = test_df['label_5class'].values

    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer

    for variant in config['variants']:
        variant_name = variant['name']
        ngram_range = tuple(variant['ngram_range'])
        remove_stopwords = variant['remove_stopwords']
        lemmatize = variant['lemmatize']

        logger.info(f"\n--- Variant: {variant_name} ---")
        logger.info(f"    ngram_range={ngram_range}, stopwords={remove_stopwords}, "
                     f"lemmatize={lemmatize}")

        # Preprocess text
        train_texts = train_df['text'].values
        test_texts = test_df['text'].values

        if lemmatize:
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                train_texts = [" ".join([t.lemma_ for t in nlp(text)])
                               for text in train_texts]
                test_texts = [" ".join([t.lemma_ for t in nlp(text)])
                              for text in test_texts]
            except ImportError:
                logger.warning("spacy not available — skipping lemmatization")

        stop_words = 'english' if remove_stopwords else None

        tfidf = TfidfVectorizer(
            max_features=config['data']['max_features_tfidf'],
            ngram_range=ngram_range, sublinear_tf=True, min_df=2,
            stop_words=stop_words,
        )
        X_train = tfidf.fit_transform(train_texts)
        X_test = tfidf.transform(test_texts)

        for seed in config['training']['seeds']:
            set_seed(seed, deterministic=False)
            start = time.time()

            clf = LinearSVC(C=config['model']['svm_C'],
                            class_weight=config['model']['svm_class_weight'],
                            max_iter=5000, random_state=seed)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            metrics = compute_all_metrics(y_test, y_pred, num_classes=5,
                                          label_names=LABEL5_NAMES)
            duration = time.time() - start

            logger.info(format_metrics_table(
                metrics, f"Preprocessing — {variant_name} (Seed {seed})"))

            save_metrics(metrics, os.path.join(
                output_dir, f"{variant_name}_seed{seed}_metrics.json"))

            exp_logger.log_run(
                config={'ablation': 'preprocessing', 'variant': variant_name,
                        'ngram_range': list(ngram_range),
                        'remove_stopwords': remove_stopwords,
                        'lemmatize': lemmatize},
                metrics=metrics, seed=seed, duration_seconds=duration,
            )


# ---------------------------------------------------------------------------
# Ablation C: Embeddings
# ---------------------------------------------------------------------------

def run_embedding_ablation(data_dir: str = "data/processed/amazon_combined"):
    """
    Ablation C (§11.3): Embedding source variations.

    Compares random-init vs GloVe vs FastText on BiLSTM+attention.
    """
    import torch
    config = load_ablation_config('embeddings')
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    exp_logger = ExperimentLogger(output_dir)

    logger.info("=" * 60)
    logger.info("  ABLATION C: Embedding Source (random vs GloVe vs FastText)")
    logger.info("=" * 60)

    train_df, val_df, test_df = _load_split_data(data_dir)

    from ..features.tokenizer import build_vocab, texts_to_sequences
    from ..features.embeddings import load_embedding_matrix
    from ..models.deep_learning.bilstm_attention import BiLSTMAttention

    # Build vocab once
    vocab = build_vocab(
        train_df['text'].values,
        max_vocab_size=config['model']['vocab_size'],
    )
    max_len = config['data']['max_sequence_length']

    X_train = texts_to_sequences(train_df['text'].values, vocab, max_len)
    X_val = texts_to_sequences(val_df['text'].values, vocab, max_len)
    X_test = texts_to_sequences(test_df['text'].values, vocab, max_len)
    y_train = train_df['label_5class'].values
    y_val = val_df['label_5class'].values
    y_test = test_df['label_5class'].values

    class_weights = get_class_weights(y_train, num_classes=5)

    for variant in config['variants']:
        variant_name = variant['name']
        emb_source = variant['embedding_source']
        logger.info(f"\n--- Variant: {variant_name} (source: {emb_source}) ---")

        for seed in config['training']['seeds']:
            set_seed(seed, deterministic=True)

            # Load embedding matrix
            if emb_source == 'random':
                embedding_matrix = None
            else:
                embedding_matrix = load_embedding_matrix(
                    vocab, emb_source,
                    embedding_dim=config['model']['embedding_dim'],
                )

            model = BiLSTMAttention(
                vocab_size=len(vocab),
                embedding_dim=config['model']['embedding_dim'],
                hidden_size=config['model']['hidden_size'],
                num_classes=5,
                dropout=config['model']['dropout'],
                pretrained_embeddings=embedding_matrix,
            )

            train_loader, val_loader, test_loader = create_data_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size=config['training']['batch_size'],
            )

            result = train_dl_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_classes=5,
                label_names=LABEL5_NAMES,
                learning_rate=config['training']['learning_rate'],
                epochs=config['training']['epochs'],
                early_stopping_patience=config['training']['early_stopping_patience'],
                loss_type=config['training']['loss'],
                class_weights=class_weights,
                output_dir=output_dir,
                experiment_name=f"{variant_name}_seed{seed}",
            )

            exp_logger.log_run(
                config={'ablation': 'embeddings', 'variant': variant_name,
                        'embedding_source': emb_source},
                metrics=result['test_metrics'], seed=seed,
                duration_seconds=0,
                model_path=result.get('model_path'),
            )


# ---------------------------------------------------------------------------
# Ablation D: Loss Function
# ---------------------------------------------------------------------------

def run_loss_ablation(data_dir: str = "data/processed/amazon_combined"):
    """
    Ablation D (§11.4): Loss function variations.

    Compares plain CE vs class-weighted CE vs focal loss on BiLSTM.
    """
    import torch
    config = load_ablation_config('loss_function')
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    exp_logger = ExperimentLogger(output_dir)

    logger.info("=" * 60)
    logger.info("  ABLATION D: Loss Function (CE vs weighted CE vs focal)")
    logger.info("=" * 60)

    train_df, val_df, test_df = _load_split_data(data_dir)

    from ..features.tokenizer import build_vocab, texts_to_sequences
    from ..features.embeddings import load_embedding_matrix
    from ..models.deep_learning.bilstm_attention import BiLSTMAttention

    vocab = build_vocab(train_df['text'].values,
                        max_vocab_size=config['model'].get('vocab_size', 30000))
    max_len = config['data']['max_sequence_length']

    X_train = texts_to_sequences(train_df['text'].values, vocab, max_len)
    X_val = texts_to_sequences(val_df['text'].values, vocab, max_len)
    X_test = texts_to_sequences(test_df['text'].values, vocab, max_len)
    y_train = train_df['label_5class'].values
    y_val = val_df['label_5class'].values
    y_test = test_df['label_5class'].values

    class_weights = get_class_weights(y_train, num_classes=5)

    embedding_matrix = load_embedding_matrix(
        vocab, config['model']['embedding_source'],
        embedding_dim=config['model']['embedding_dim'],
    )

    for variant in config['variants']:
        variant_name = variant['name']
        loss_type = variant['loss']
        focal_gamma = variant.get('focal_gamma', 2.0)
        logger.info(f"\n--- Variant: {variant_name} (loss: {loss_type}) ---")

        for seed in config['training']['seeds']:
            set_seed(seed, deterministic=True)

            model = BiLSTMAttention(
                vocab_size=len(vocab),
                embedding_dim=config['model']['embedding_dim'],
                hidden_size=config['model']['hidden_size'],
                num_classes=5,
                dropout=config['model']['dropout'],
                pretrained_embeddings=embedding_matrix,
            )

            train_loader, val_loader, test_loader = create_data_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size=config['training']['batch_size'],
            )

            result = train_dl_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_classes=5,
                label_names=LABEL5_NAMES,
                learning_rate=config['training']['learning_rate'],
                epochs=config['training']['epochs'],
                early_stopping_patience=config['training']['early_stopping_patience'],
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                class_weights=class_weights,
                output_dir=output_dir,
                experiment_name=f"{variant_name}_seed{seed}",
            )

            exp_logger.log_run(
                config={'ablation': 'loss_function', 'variant': variant_name,
                        'loss_type': loss_type, 'focal_gamma': focal_gamma},
                metrics=result['test_metrics'], seed=seed,
                duration_seconds=0,
                model_path=result.get('model_path'),
            )


# ---------------------------------------------------------------------------
# Ablation E: Fine-tuning Depth
# ---------------------------------------------------------------------------

def run_finetuning_depth_ablation(data_dir: str = "data/processed/amazon_combined"):
    """
    Ablation E (§11.5): Fine-tuning depth.

    For Tier 2 (local): frozen vs unfrozen embeddings in BiLSTM.
    Tier 3 configs (frozen backbone vs full fine-tune) are in the YAML
    and should be run on Colab.
    """
    import torch
    config = load_ablation_config('finetuning_depth')
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    exp_logger = ExperimentLogger(output_dir)

    logger.info("=" * 60)
    logger.info("  ABLATION E: Fine-tuning Depth (frozen vs unfrozen)")
    logger.info("=" * 60)

    train_df, val_df, test_df = _load_split_data(data_dir)

    from ..features.tokenizer import build_vocab, texts_to_sequences
    from ..features.embeddings import load_embedding_matrix
    from ..models.deep_learning.bilstm_attention import BiLSTMAttention

    vocab = build_vocab(train_df['text'].values, max_vocab_size=30000)
    max_len = 128

    X_train = texts_to_sequences(train_df['text'].values, vocab, max_len)
    X_val = texts_to_sequences(val_df['text'].values, vocab, max_len)
    X_test = texts_to_sequences(test_df['text'].values, vocab, max_len)
    y_train = train_df['label_5class'].values
    y_val = val_df['label_5class'].values
    y_test = test_df['label_5class'].values

    class_weights = get_class_weights(y_train, num_classes=5)

    embedding_matrix = load_embedding_matrix(
        vocab, config['model']['embedding_source'],
        embedding_dim=config['model']['embedding_dim'],
    )

    for variant in config['tier2_variants']:
        variant_name = variant['name']
        freeze = variant['freeze_embeddings']
        logger.info(f"\n--- Variant: {variant_name} (freeze={freeze}) ---")

        # If frozen, set unfreeze_after to a very large epoch (effectively never)
        unfreeze_after = 999 if freeze else 1

        for seed in config['training']['seeds']:
            set_seed(seed, deterministic=True)

            model = BiLSTMAttention(
                vocab_size=len(vocab),
                embedding_dim=config['model']['embedding_dim'],
                hidden_size=config['model']['hidden_size'],
                num_classes=5,
                dropout=config['model']['dropout'],
                pretrained_embeddings=embedding_matrix,
            )

            train_loader, val_loader, test_loader = create_data_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size=config['training']['batch_size'],
            )

            result = train_dl_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_classes=5,
                label_names=LABEL5_NAMES,
                learning_rate=config['training']['learning_rate'],
                epochs=config['training']['epochs'],
                early_stopping_patience=config['training']['early_stopping_patience'],
                loss_type=config['training']['loss'],
                class_weights=class_weights,
                unfreeze_embeddings_after_epoch=unfreeze_after,
                output_dir=output_dir,
                experiment_name=f"{variant_name}_seed{seed}",
            )

            exp_logger.log_run(
                config={'ablation': 'finetuning_depth', 'variant': variant_name,
                        'freeze_embeddings': freeze},
                metrics=result['test_metrics'], seed=seed,
                duration_seconds=0,
                model_path=result.get('model_path'),
            )

    logger.info("\n  NOTE: Tier 3 variants (frozen DistilBERT vs full fine-tune)")
    logger.info("  are defined in configs/ablations/finetuning_depth.yaml")
    logger.info("  and should be run on Colab with a T4 GPU.")


# ---------------------------------------------------------------------------
# Ablation F: Sequence Length
# ---------------------------------------------------------------------------

def run_sequence_length_ablation(data_dir: str = "data/processed/amazon_combined"):
    """
    Ablation F (§11.6): Sequence length variations (64 vs 128 vs 256 tokens).
    """
    import torch
    config = load_ablation_config('sequence_length')
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    exp_logger = ExperimentLogger(output_dir)

    logger.info("=" * 60)
    logger.info("  ABLATION F: Sequence Length (64 vs 128 vs 256)")
    logger.info("=" * 60)

    train_df, val_df, test_df = _load_split_data(data_dir)

    from ..features.tokenizer import build_vocab, texts_to_sequences
    from ..features.embeddings import load_embedding_matrix
    from ..models.deep_learning.bilstm_attention import BiLSTMAttention

    vocab = build_vocab(train_df['text'].values, max_vocab_size=30000)
    y_train = train_df['label_5class'].values
    y_val = val_df['label_5class'].values
    y_test = test_df['label_5class'].values
    class_weights = get_class_weights(y_train, num_classes=5)

    embedding_matrix = load_embedding_matrix(
        vocab, config['model']['embedding_source'],
        embedding_dim=config['model']['embedding_dim'],
    )

    for variant in config['variants']:
        variant_name = variant['name']
        max_len = variant['max_sequence_length']
        logger.info(f"\n--- Variant: {variant_name} (max_len={max_len}) ---")

        X_train = texts_to_sequences(train_df['text'].values, vocab, max_len)
        X_val = texts_to_sequences(val_df['text'].values, vocab, max_len)
        X_test = texts_to_sequences(test_df['text'].values, vocab, max_len)

        for seed in config['training']['seeds']:
            set_seed(seed, deterministic=True)

            model = BiLSTMAttention(
                vocab_size=len(vocab),
                embedding_dim=config['model']['embedding_dim'],
                hidden_size=config['model']['hidden_size'],
                num_classes=5,
                dropout=config['model']['dropout'],
                pretrained_embeddings=embedding_matrix,
            )

            train_loader, val_loader, test_loader = create_data_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size=config['training']['batch_size'],
            )

            result = train_dl_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_classes=5,
                label_names=LABEL5_NAMES,
                learning_rate=config['training']['learning_rate'],
                epochs=config['training']['epochs'],
                early_stopping_patience=config['training']['early_stopping_patience'],
                loss_type=config['training']['loss'],
                class_weights=class_weights,
                output_dir=output_dir,
                experiment_name=f"{variant_name}_seed{seed}",
            )

            exp_logger.log_run(
                config={'ablation': 'sequence_length', 'variant': variant_name,
                        'max_sequence_length': max_len},
                metrics=result['test_metrics'], seed=seed,
                duration_seconds=0,
                model_path=result.get('model_path'),
            )


# ---------------------------------------------------------------------------
# Ablation G: Data Volume
# ---------------------------------------------------------------------------

def run_data_volume_ablation(data_dir: str = "data/processed/amazon_combined"):
    """
    Ablation G (§11.7): Data volume learning curve.

    Trains SVM on 10%/25%/50%/100% of the training data.
    Subsamples from the already-split training set to prevent leakage.
    """
    config = load_ablation_config('data_volume')
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    exp_logger = ExperimentLogger(output_dir)

    logger.info("=" * 60)
    logger.info("  ABLATION G: Data Volume (10% / 25% / 50% / 100%)")
    logger.info("=" * 60)

    train_df, val_df, test_df = _load_split_data(data_dir)
    y_test = test_df['label_5class'].values

    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer

    for variant in config['variants']:
        variant_name = variant['name']
        frac = variant['train_fraction']
        logger.info(f"\n--- Variant: {variant_name} ({frac*100:.0f}% training data) ---")

        for seed in config['training']['seeds']:
            set_seed(seed, deterministic=False)
            start = time.time()

            # Subsample training data
            if frac < 1.0:
                sub_train = train_df.sample(frac=frac, random_state=seed,
                                            replace=False)
            else:
                sub_train = train_df

            y_train_sub = sub_train['label_5class'].values
            logger.info(f"    Training on {len(sub_train)} samples "
                         f"(seed {seed})")

            tfidf = TfidfVectorizer(
                max_features=config['data']['max_features_tfidf'],
                ngram_range=(1, 2), sublinear_tf=True, min_df=2,
            )
            X_train = tfidf.fit_transform(sub_train['text'].values)
            X_test = tfidf.transform(test_df['text'].values)

            clf = LinearSVC(C=config['model']['svm_C'],
                            class_weight=config['model']['svm_class_weight'],
                            max_iter=5000, random_state=seed)
            clf.fit(X_train, y_train_sub)

            y_pred = clf.predict(X_test)
            metrics = compute_all_metrics(y_test, y_pred, num_classes=5,
                                          label_names=LABEL5_NAMES)
            duration = time.time() - start

            logger.info(format_metrics_table(
                metrics, f"Data Volume — {variant_name} (Seed {seed})"))

            save_metrics(metrics, os.path.join(
                output_dir, f"{variant_name}_seed{seed}_metrics.json"))

            exp_logger.log_run(
                config={'ablation': 'data_volume', 'variant': variant_name,
                        'train_fraction': frac,
                        'n_train_samples': len(sub_train)},
                metrics=metrics, seed=seed, duration_seconds=duration,
            )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ABLATION_REGISTRY = {
    'label_granularity': run_label_granularity_ablation,
    'preprocessing': run_preprocessing_ablation,
    'embeddings': run_embedding_ablation,
    'loss_function': run_loss_ablation,
    'finetuning_depth': run_finetuning_depth_ablation,
    'sequence_length': run_sequence_length_ablation,
    'data_volume': run_data_volume_ablation,
}


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies (§11)")
    parser.add_argument('--ablation', type=str, required=False, default=None,
                        choices=list(ABLATION_REGISTRY.keys()),
                        help="Which ablation to run (or omit for all)")
    parser.add_argument('--data-dir', type=str,
                        default="data/processed/amazon_combined",
                        help="Path to pre-split data directory")
    args = parser.parse_args()

    if args.ablation:
        logger.info(f"Running single ablation: {args.ablation}")
        ABLATION_REGISTRY[args.ablation](data_dir=args.data_dir)
    else:
        logger.info("Running ALL ablation studies (§11)")
        for name, func in ABLATION_REGISTRY.items():
            logger.info(f"\n{'#' * 70}")
            logger.info(f"  Starting ablation: {name}")
            logger.info(f"{'#' * 70}")
            try:
                func(data_dir=args.data_dir)
            except Exception as e:
                logger.error(f"Ablation '{name}' failed: {e}")
                continue


if __name__ == "__main__":
    main()
