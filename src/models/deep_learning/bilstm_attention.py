"""
Tier 2: BiLSTM with Attention for sentiment classification.

Implements §8/§20:
- BiLSTM: hidden_size=128, num_layers=1, bidirectional=True, dropout=0.3
- Single-head additive attention over BiLSTM outputs
- GloVe 300d embeddings, frozen epoch 1, unfrozen after
- Adam(lr=1e-3), up to 15 epochs, early stopping patience=3
- Class-weighted cross-entropy loss
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class AdditiveAttention(nn.Module):
    """
    Single-head additive (Bahdanau-style) attention.

    Computes attention weights over BiLSTM hidden states and produces
    a fixed-size context vector.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: BiLSTM output, shape (batch, seq_len, hidden_size).
            mask: Boolean mask, shape (batch, seq_len). True for valid tokens.

        Returns:
            Tuple of (context_vector, attention_weights).
            context_vector: shape (batch, hidden_size).
            attention_weights: shape (batch, seq_len).
        """
        # Compute attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)

        # Apply mask (set padding positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax to get attention weights
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum of LSTM outputs
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_size)

        return context, weights


class BiLSTMAttention(nn.Module):
    """
    BiLSTM + Additive Attention for text classification.

    Architecture:
    1. Embedding layer (pretrained GloVe/FastText or random)
    2. Bidirectional LSTM
    3. Additive attention
    4. Classification head (Linear → ReLU → Dropout → Linear)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 5,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
        pad_idx: int = 0,
    ):
        """
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of word embeddings.
            hidden_size: LSTM hidden size (per direction).
            num_layers: Number of LSTM layers.
            num_classes: Number of output classes.
            dropout: Dropout probability.
            pretrained_embeddings: Pretrained embedding matrix.
            freeze_embeddings: Whether to freeze embeddings initially.
            pad_idx: Padding token index.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout(dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention
        lstm_output_size = hidden_size * 2  # bidirectional
        self.attention = AdditiveAttention(lstm_output_size)

        # Classification head
        self.classifier = nn.Sequential(
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
        # Create mask (True for non-padding tokens)
        mask = (x != self.pad_idx)  # (batch, seq_len)

        # Embed
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)

        # BiLSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_size*2)

        # Attention
        context, attn_weights = self.attention(lstm_out, mask)  # (batch, hidden_size*2)

        # Classify
        logits = self.classifier(context)  # (batch, num_classes)

        return logits

    def unfreeze_embeddings(self):
        """Unfreeze embedding weights (call after epoch 1 per §20)."""
        self.embedding.weight.requires_grad = True
        logger.info("  Embeddings unfrozen")

    def get_config(self) -> dict:
        return {
            'model_type': 'bilstm_attention',
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'embedding_dim': self.embedding.embedding_dim,
            'vocab_size': self.embedding.num_embeddings,
        }
