"""
Tier 3: DistilBERT fine-tuning script for Colab T4 GPU.

Implements §8/§20 for DistilBERT:
- distilbert-base-uncased
- max_len=128, batch_size=32, lr=2e-5, weight_decay=0.01, epochs=3-4
- fp16 enabled, Colab disconnect resilience (auto-resume).
"""

import os
import logging
import argparse
from typing import Dict, Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Assume script is run from project root, so we can import src
from src.evaluation.metrics import compute_all_metrics, save_metrics, format_metrics_table
from src.data.label_map import LABEL5_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics_hf(eval_pred):
    """Adapter for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Softmax for probabilities
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    metrics = compute_all_metrics(labels, predictions, y_prob=probs, num_classes=5)
    
    # HF Trainer expects flat dictionary
    return {
        'macro_f1': metrics['macro_f1'],
        'accuracy': metrics['accuracy'],
        'qwk': metrics['qwk'],
        'weighted_f1': metrics['weighted_f1']
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Colab.")
    parser.add_argument("--data_dir", type=str, default="data/processed/amazon_combined", help="Path to processed data splits.")
    parser.add_argument("--output_dir", type=str, default="experiments/tier3_distilbert", help="Output directory.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples (for HP search).")
    args = parser.parse_args()

    logger.info(f"Starting DistilBERT fine-tuning. Data dir: {args.data_dir}")

    # 1. Load Data
    train_path = os.path.join(args.data_dir, "train.parquet")
    val_path = os.path.join(args.data_dir, "val.parquet")
    test_path = os.path.join(args.data_dir, "test.parquet")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(f"Missing data splits in {args.data_dir}")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    if args.max_samples:
        df_train = df_train.sample(n=min(args.max_samples, len(df_train)), random_state=42)
        logger.info(f"Subsampled training data to {len(df_train)} rows.")

    # 2. Tokenize
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    logger.info("Tokenizing datasets...")
    # Using text_minimal or text if minimal is missing
    text_col = 'text_minimal' if 'text_minimal' in df_train.columns else 'text'
    train_dataset = ReviewDataset(df_train[text_col], df_train['label_5class'], tokenizer)
    val_dataset = ReviewDataset(df_val[text_col], df_val['label_5class'], tokenizer)
    test_dataset = ReviewDataset(df_test[text_col], df_test['label_5class'], tokenizer)

    # 3. Model
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)

    # 4. Training Arguments
    # Colab T4 specifics: fp16=True, save_steps for resilience
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_ratio=0.06,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=True, # Crucial for T4 speedup
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_hf,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 5. Train
    # Automatically resume from latest checkpoint if it exists
    latest_checkpoint = None
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            latest_checkpoint = True
            logger.info("Found existing checkpoint. Resuming training...")

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    # 6. Evaluate
    logger.info("Evaluating on Test Set...")
    test_results = trainer.predict(test_dataset)
    
    # Save full metrics
    predictions = np.argmax(test_results.predictions, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(test_results.predictions), dim=-1).numpy()
    final_metrics = compute_all_metrics(df_test['label_5class'].values, predictions, y_prob=probs, num_classes=5)
    
    print(format_metrics_table(final_metrics, "DistilBERT Final Results"))
    save_metrics(final_metrics, os.path.join(args.output_dir, "distilbert_final_metrics.json"))
    
    # We could also generate the confusion matrix here if matplotlib is available.

if __name__ == "__main__":
    main()
