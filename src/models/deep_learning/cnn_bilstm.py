"""
Tier 2: CNN-BiLSTM Hybrid for sentiment classification.

Implements §8/§20:
- CNN branch: filter sizes [3,4,5], 100 filters each
- BiLSTM on CNN features
- Same training protocol as BiLSTM+Attention

This hybrid architecture is explicitly noted in the review paper
(Minaee et al., Zuheros et al., Meng et al.) as a recurring, effective pattern.
"""

import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class CNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM Hybrid for text classification.

    Architecture:
    1. Embedding layer (pretrained or random)
    2. Parallel CNN branches with different filter sizes
    3. BiLSTM on concatenated CNN features
    4. Classification head
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        cnn_filter_sizes: List[int] = None,
        cnn_num_filters: int = 100,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 5,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
        pad_idx: int = 0,
    ):
        super().__init__()

        if cnn_filter_sizes is None:
            cnn_filter_sizes = [3, 4, 5]

        self.hidden_size = hidden_size
        self.cnn_num_filters = cnn_num_filters
        self.cnn_filter_sizes = cnn_filter_sizes
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout(dropout)

        # CNN branches (parallel convolutions with different filter sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=cnn_num_filters,
                kernel_size=fs,
                padding=fs // 2,  # Same-ish padding to preserve sequence length
            )
            for fs in cnn_filter_sizes
        ])

        cnn_output_size = cnn_num_filters * len(cnn_filter_sizes)

        # BiLSTM on CNN features
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_size = hidden_size * 2  # bidirectional

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token indices, shape (batch, seq_len).

        Returns:
            Logits, shape (batch, num_classes).
        """
        # Embed
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)

        # CNN expects (batch, channels, seq_len)
        embedded_t = embedded.transpose(1, 2)  # (batch, embedding_dim, seq_len)

        # Apply parallel convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded_t))  # (batch, num_filters, seq_len')
            conv_outputs.append(conv_out)

        # Concatenate along filter dimension
        cnn_out = torch.cat(conv_outputs, dim=1)  # (batch, total_filters, seq_len')

        # Back to (batch, seq_len', total_filters) for LSTM
        cnn_out = cnn_out.transpose(1, 2)

        # BiLSTM
        lstm_out, (hidden, _) = self.lstm(cnn_out)

        # Use final hidden state (concatenate forward and backward)
        # hidden shape: (num_layers * 2, batch, hidden_size)
        forward_hidden = hidden[-2]  # Last forward layer
        backward_hidden = hidden[-1]  # Last backward layer
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_size*2)

        # Classify
        logits = self.classifier(combined)  # (batch, num_classes)

        return logits

    def unfreeze_embeddings(self):
        """Unfreeze embedding weights."""
        self.embedding.weight.requires_grad = True
        logger.info("  Embeddings unfrozen")

    def get_config(self) -> dict:
        return {
            'model_type': 'cnn_bilstm',
            'cnn_filter_sizes': self.cnn_filter_sizes,
            'cnn_num_filters': self.cnn_num_filters,
            'hidden_size': self.hidden_size,
            'embedding_dim': self.embedding.embedding_dim,
            'vocab_size': self.embedding.num_embeddings,
        }
