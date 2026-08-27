"""
YAML config loader and validator.

Implements config-driven experiments per §9: every run (model, hyperparameters,
dataset slice, seed) is defined in a versioned YAML file, not hardcoded.
Supports merging defaults with per-experiment overrides.
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""
    dataset: str = "amazon"
    categories: List[str] = field(default_factory=lambda: ["Electronics", "Home_and_Kitchen"])
    max_samples_per_category: int = 125000
    max_features_tfidf: int = 50000
    max_sequence_length: int = 128
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_classes_5: int = 5
    num_classes_3: int = 3


@dataclass
class ModelConfig:
    """Model-related configuration."""
    tier: str = "tier1"                    # tier1, tier2, tier3, absa
    model_type: str = "svm"                # svm, nb, lr, bilstm, cnn_bilstm, distilbert, roberta, absa_bert
    # Classical ML params
    svm_C: float = 1.0
    svm_class_weight: str = "balanced"
    nb_alpha: float = 1.0
    lr_C: float = 1.0
    lr_solver: str = "lbfgs"
    # DL params
    embedding_dim: int = 300
    embedding_source: str = "glove"        # glove, fasttext, random
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.3
    cnn_filter_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    cnn_num_filters: int = 100
    vocab_size: int = 30000
    # Transformer params
    pretrained_model: str = "distilbert-base-uncased"
    freeze_backbone: bool = False
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    seeds: List[int] = field(default_factory=lambda: [13, 42, 2024])
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 15
    early_stopping_patience: int = 3
    early_stopping_metric: str = "macro_f1"
    optimizer: str = "adam"                 # adam, adamw, sgd
    loss: str = "cross_entropy_weighted"   # cross_entropy, cross_entropy_weighted, focal
    focal_gamma: float = 2.0
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    checkpoint_every_n_steps: Optional[int] = None
    log_every_n_steps: int = 100
    num_workers: int = 0


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    experiment_name: str = "default"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "experiments"
    label_scheme: str = "5class"           # 5class, 3class, both

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load config from a YAML file, merging with defaults."""
        with open(path, 'r') as f:
            raw = yaml.safe_load(f) or {}

        config = cls()

        # Update data config
        if 'data' in raw:
            for k, v in raw['data'].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)

        # Update model config
        if 'model' in raw:
            for k, v in raw['model'].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)

        # Update training config
        if 'training' in raw:
            for k, v in raw['training'].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        # Update top-level fields
        for k in ['experiment_name', 'description', 'output_dir', 'label_scheme']:
            if k in raw:
                setattr(config, k, raw[k])

        return config

    def to_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'label_scheme': self.label_scheme,
            'output_dir': self.output_dir,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to a flat dictionary for logging."""
        result = {
            'experiment_name': self.experiment_name,
            'label_scheme': self.label_scheme,
        }
        for prefix, section in [('data', self.data), ('model', self.model), ('training', self.training)]:
            for k, v in section.__dict__.items():
                result[f'{prefix}.{k}'] = v
        return result


def load_config(path: str) -> ExperimentConfig:
    """
    Load an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        ExperimentConfig instance with defaults merged with file values.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    return ExperimentConfig.from_yaml(path)
