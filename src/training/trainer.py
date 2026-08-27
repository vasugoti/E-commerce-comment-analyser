"""
Unified training loop for all model tiers.

Handles:
- Tier 1: Classical ML training (sklearn fit/predict)
- Tier 2: DL training loop with early stopping, checkpointing
- Experiment logging (config → metrics) per §9
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    nn_Module = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    DataLoader = Any
    TensorDataset = Any
    nn_Module = object

from .seed import set_seed, get_device, DEFAULT_SEEDS
from .config import ExperimentConfig
from ..evaluation.metrics import compute_all_metrics, format_metrics_table, save_metrics
from ..evaluation.confusion import plot_confusion_matrix
from ..data.label_map import get_class_weights, LABEL5_NAMES, LABEL3_NAMES

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Logs experiment runs (config → metrics) to a CSV/JSON file."""

    def __init__(self, log_dir: str = "experiments"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "run_log.json")
        self.runs = []
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                self.runs = json.load(f)

    def log_run(self, config: Dict, metrics: Dict, seed: int,
                duration_seconds: float, model_path: Optional[str] = None):
        """Log a single experiment run."""
        run = {
            'timestamp': datetime.now().isoformat(),
            'seed': seed,
            'duration_seconds': round(duration_seconds, 1),
            'model_path': model_path,
            'config': config,
            'metrics': {k: v for k, v in metrics.items()
                        if k not in ('per_class',)},  # Keep summary metrics
        }
        self.runs.append(run)
        with open(self.log_path, 'w') as f:
            json.dump(self.runs, f, indent=2, default=str)

    def get_runs_df(self) -> pd.DataFrame:
        """Get all runs as a DataFrame."""
        rows = []
        for run in self.runs:
            row = {
                'timestamp': run['timestamp'],
                'seed': run['seed'],
                'duration_s': run['duration_seconds'],
            }
            row.update(run.get('config', {}))
            row.update({f'metric_{k}': v for k, v in run.get('metrics', {}).items()
                        if isinstance(v, (int, float))})
            rows.append(row)
        return pd.DataFrame(rows)


class FocalLoss(nn_Module):
    """Focal Loss for class-imbalanced classification (§11 ablation)."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train_classical_model(
    model,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    num_classes: int = 5,
    label_names: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    output_dir: str = "experiments",
    experiment_name: str = "classical",
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate a classical ML model across multiple seeds.

    Args:
        model: Callable that creates a new model instance.
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels.
        X_test, y_test: Test features and labels.
        num_classes: Number of classes.
        label_names: Label names for reporting.
        config: Model configuration dict.
        output_dir: Output directory.
        experiment_name: Name for this experiment.
        seeds: List of seeds (default: [13, 42, 2024]).

    Returns:
        Dict with aggregated results across seeds.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if label_names is None:
        label_names = LABEL5_NAMES if num_classes == 5 else LABEL3_NAMES

    exp_logger = ExperimentLogger(output_dir)
    all_seed_metrics = []

    for seed in seeds:
        set_seed(seed, deterministic=False)
        start_time = time.time()

        logger.info(f"\n{'='*50}")
        logger.info(f"  {experiment_name} — Seed {seed}")
        logger.info(f"{'='*50}")

        # Create and train model
        clf = model()
        clf.fit(X_train, y_train)

        # Predict
        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)

        # Get probabilities if available
        y_prob_test = None
        if hasattr(clf, 'predict_proba'):
            try:
                y_prob_test = clf.predict_proba(X_test)
            except Exception:
                pass

        # Compute metrics
        val_metrics = compute_all_metrics(y_val, y_pred_val, num_classes=num_classes,
                                          label_names=label_names)
        test_metrics = compute_all_metrics(y_test, y_pred_test, y_prob=y_prob_test,
                                           num_classes=num_classes, label_names=label_names)

        duration = time.time() - start_time

        # Log
        print(format_metrics_table(val_metrics, f"{experiment_name} — Val (Seed {seed})"))
        print(format_metrics_table(test_metrics, f"{experiment_name} — Test (Seed {seed})"))

        # Confusion matrix
        cm_path = os.path.join(output_dir, f"{experiment_name}_seed{seed}_confusion.png")
        plot_confusion_matrix(y_test, y_pred_test, num_classes=num_classes,
                              label_names=label_names,
                              title=f"{experiment_name} (Seed {seed})",
                              save_path=cm_path)

        # Save metrics
        metrics_path = os.path.join(output_dir, f"{experiment_name}_seed{seed}_metrics.json")
        save_metrics({'val': val_metrics, 'test': test_metrics}, metrics_path)

        exp_logger.log_run(
            config=config or {},
            metrics=test_metrics,
            seed=seed,
            duration_seconds=duration,
        )

        all_seed_metrics.append(test_metrics)

    return {
        'seed_metrics': all_seed_metrics,
        'experiment_name': experiment_name,
    }


def train_dl_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int = 5,
    label_names: Optional[List[str]] = None,
    learning_rate: float = 1e-3,
    epochs: int = 15,
    early_stopping_patience: int = 3,
    loss_type: str = 'cross_entropy_weighted',
    focal_gamma: float = 2.0,
    class_weights: Optional[np.ndarray] = None,
    unfreeze_embeddings_after_epoch: int = 1,
    config: Optional[Dict] = None,
    output_dir: str = "experiments",
    experiment_name: str = "dl_model",
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Train a deep learning model with early stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        num_classes: Number of classes.
        label_names: Label names.
        learning_rate: Learning rate.
        epochs: Max epochs.
        early_stopping_patience: Patience for early stopping.
        loss_type: Loss function type.
        focal_gamma: Gamma for focal loss.
        class_weights: Class weights array.
        unfreeze_embeddings_after_epoch: Epoch after which to unfreeze embeddings.
        config: Configuration dict.
        output_dir: Output directory.
        experiment_name: Experiment name.
        device: 'cpu' or 'cuda'.

    Returns:
        Dict with training history and final metrics.
    """
    if label_names is None:
        label_names = LABEL5_NAMES if num_classes == 5 else LABEL3_NAMES

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.FloatTensor(class_weights).to(device)

    if loss_type == 'focal':
        criterion = FocalLoss(gamma=focal_gamma, weight=weight_tensor)
    elif loss_type == 'cross_entropy_weighted':
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}
    best_model_state = None

    for epoch in range(epochs):
        # Unfreeze embeddings after specified epoch
        if epoch == unfreeze_embeddings_after_epoch and hasattr(model, 'unfreeze_embeddings'):
            model.unfreeze_embeddings()
            # Re-create optimizer with all parameters
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate
            )

        # --- Train ---
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)
        history['train_loss'].append(avg_train_loss)

        # --- Validate ---
        val_metrics, val_loss = _evaluate_dl(model, val_loader, criterion, num_classes,
                                              label_names, device)
        history['val_loss'].append(val_loss)
        history['val_macro_f1'].append(val_metrics['macro_f1'])

        logger.info(
            f"  Epoch {epoch+1}/{epochs}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        # Early stopping on val macro-F1
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch+1} "
                             f"(best val macro-F1: {best_val_f1:.4f})")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    # --- Final evaluation on test set ---
    test_metrics, _ = _evaluate_dl(model, test_loader, criterion, num_classes,
                                    label_names, device)

    print(format_metrics_table(test_metrics, f"{experiment_name} — Test"))

    # Save
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{experiment_name}_best.pt")
    torch.save(best_model_state, model_path)

    metrics_path = os.path.join(output_dir, f"{experiment_name}_metrics.json")
    save_metrics({'test': test_metrics, 'history': history}, metrics_path)

    return {
        'test_metrics': test_metrics,
        'history': history,
        'best_val_f1': best_val_f1,
        'model_path': model_path,
    }


def _evaluate_dl(model: nn.Module, data_loader: DataLoader,
                 criterion: nn.Module, num_classes: int,
                 label_names: List[str], device: str) -> Tuple[Dict, float]:
    """Evaluate a DL model on a data loader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    avg_loss = total_loss / max(n_batches, 1)

    metrics = compute_all_metrics(y_true, y_pred, y_prob=y_prob,
                                  num_classes=num_classes, label_names=label_names)

    return metrics, avg_loss


def create_data_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    batch_size: int = 64,
) -> Tuple[Any, Any, Any]:
    """Create PyTorch DataLoaders from numpy arrays."""
    if torch is None:
        raise RuntimeError("PyTorch is required to create DataLoaders.")
    train_dataset = TensorDataset(
        torch.LongTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.LongTensor(X_val), torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.LongTensor(X_test), torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


