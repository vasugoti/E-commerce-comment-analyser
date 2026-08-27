"""
Training script for the Aspect-Based Sentiment Analysis module.

Fine-tunes a BERT-base model on the SemEval-2014 Laptop dataset.
"""

import os
import time
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from .dataset import ABSADataset
from .model import ABSABertClassifier
from ..data.ingest import load_semeval_2014
from ..training.seed import set_seed, get_device
from ..evaluation.metrics import compute_all_metrics, format_metrics_table, save_metrics
from ..data.schema import ABSA_POLARITY_NAMES

logger = logging.getLogger(__name__)


def train_absa(
    data_dir: str = "data",
    output_dir: str = "experiments/absa",
    pretrained_model_name: str = "bert-base-uncased",
    max_length: int = 96,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    epochs: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train the ABSA model."""
    
    set_seed(seed)
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting ABSA Training on {device}")
    
    # 1. Load Data
    train_instances, test_instances = load_semeval_2014(os.path.join(data_dir, "raw"))
    if not train_instances or not test_instances:
        logger.error("Could not load SemEval data. Cannot proceed with ABSA training.")
        return {}

    # 2. Setup Tokenizer and Datasets
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    
    train_dataset = ABSADataset(train_instances, tokenizer, max_length)
    test_dataset = ABSADataset(test_instances, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Setup Model, Optimizer, Loss
    model = ABSABertClassifier(pretrained_model_name, num_classes=3).to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Evaluation
        logger.info("Evaluating...")
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                     token_type_ids = token_type_ids.to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask, token_type_ids)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        metrics = compute_all_metrics(all_labels, all_preds, num_classes=3, label_names=ABSA_POLARITY_NAMES)
        history['val_acc'].append(metrics['accuracy'])
        
        logger.info(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f}, Test Acc: {metrics['accuracy']:.4f}")
        
        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            model_path = os.path.join(output_dir, "best_absa_model.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved new best model with acc: {best_acc:.4f}")
            
    # Final Results
    print(format_metrics_table(metrics, "ABSA Final Results"))
    metrics_path = os.path.join(output_dir, "absa_metrics.json")
    save_metrics(metrics, metrics_path)
    
    return {'metrics': metrics, 'history': history}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    train_absa()
