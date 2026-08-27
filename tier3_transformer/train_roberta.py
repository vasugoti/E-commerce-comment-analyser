"""
Tier 3: RoBERTa fine-tuning script for Colab T4 GPU (Opportunistic).

Similar to DistilBERT script but uses roberta-base.
Requires slightly more memory, so batch size is reduced to 16 with gradient accumulation.
"""

import os
import logging
import argparse

import pandas as pd
import numpy as np
import torch
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Reusing the dataset and metrics logic from train_distilbert
from train_distilbert import ReviewDataset, compute_metrics_hf
from src.evaluation.metrics import compute_all_metrics, save_metrics, format_metrics_table

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on Colab.")
    parser.add_argument("--data_dir", type=str, default="data/processed/amazon_combined")
    parser.add_argument("--output_dir", type=str, default="experiments/tier3_roberta")
    args = parser.parse_args()

    logger.info(f"Starting RoBERTa fine-tuning. Data dir: {args.data_dir}")

    # 1. Load Data
    train_path = os.path.join(args.data_dir, "train.parquet")
    val_path = os.path.join(args.data_dir, "val.parquet")
    test_path = os.path.join(args.data_dir, "test.parquet")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    # 2. Tokenize
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    text_col = 'text_minimal' if 'text_minimal' in df_train.columns else 'text'
    train_dataset = ReviewDataset(df_train[text_col], df_train['label_5class'], tokenizer)
    val_dataset = ReviewDataset(df_val[text_col], df_val['label_5class'], tokenizer)
    test_dataset = ReviewDataset(df_test[text_col], df_test['label_5class'], tokenizer)

    # 3. Model
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=5)

    # 4. Training Arguments
    # RoBERTa is larger, use batch_size=16 and grad_accum=2 to match effective bs=32
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=32,
        warmup_ratio=0.06,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=True,
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
    latest_checkpoint = None
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            latest_checkpoint = True

    trainer.train(resume_from_checkpoint=latest_checkpoint)

    # 6. Evaluate
    logger.info("Evaluating on Test Set...")
    test_results = trainer.predict(test_dataset)
    
    predictions = np.argmax(test_results.predictions, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(test_results.predictions), dim=-1).numpy()
    final_metrics = compute_all_metrics(df_test['label_5class'].values, predictions, y_prob=probs, num_classes=5)
    
    print(format_metrics_table(final_metrics, "RoBERTa Final Results"))
    save_metrics(final_metrics, os.path.join(args.output_dir, "roberta_final_metrics.json"))

if __name__ == "__main__":
    main()
