"""
Word-level tokenizer for Tier 2 deep learning models.

Builds a vocabulary from training text and converts text to integer sequences
with padding/truncation to max_len.
"""

import os
import json
import logging
from typing import List, Optional, Tuple
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


class WordTokenizer:
    """
    Word-level tokenizer for DL models.

    Builds vocabulary from training data and converts text to integer sequences.
    """

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_vocab_size: int = 30000, max_length: int = 128):
        """
        Args:
            max_vocab_size: Maximum vocabulary size (including special tokens).
            max_length: Maximum sequence length (pad/truncate to this).
        """
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}
        self.word_counts = Counter()
        self.is_fitted = False

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self.word2idx)

    def fit(self, texts: List[str]) -> 'WordTokenizer':
        """
        Build vocabulary from training texts.

        Args:
            texts: List of text strings.

        Returns:
            Self.
        """
        logger.info(f"Building vocabulary (max_size={self.max_vocab_size})...")

        # Count word frequencies
        self.word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            self.word_counts.update(words)

        # Take top-k most frequent words
        most_common = self.word_counts.most_common(self.max_vocab_size - 2)  # Reserve for PAD, UNK

        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        for i, (word, count) in enumerate(most_common):
            self.word2idx[word] = i + 2

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.is_fitted = True

        n_total_words = len(self.word_counts)
        coverage = sum(c for w, c in most_common) / sum(self.word_counts.values()) * 100
        logger.info(f"  Vocabulary: {self.vocab_size} tokens "
                     f"({n_total_words} unique words, {coverage:.1f}% coverage)")

        return self

    def encode(self, text: str) -> List[int]:
        """Convert text to a list of token indices."""
        if not self.is_fitted:
            raise RuntimeError("Tokenizer is not fitted. Call fit() first.")
        words = text.lower().split()
        return [self.word2idx.get(w, self.UNK_IDX) for w in words]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert a batch of texts to padded/truncated integer sequences.

        Args:
            texts: List of text strings.

        Returns:
            numpy array of shape (len(texts), max_length).
        """
        sequences = []
        for text in texts:
            seq = self.encode(text)

            # Truncate
            if len(seq) > self.max_length:
                seq = seq[:self.max_length]

            # Pad
            if len(seq) < self.max_length:
                seq = seq + [self.PAD_IDX] * (self.max_length - len(seq))

            sequences.append(seq)

        return np.array(sequences, dtype=np.int64)

    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices
                 if idx != self.PAD_IDX]
        return ' '.join(words)

    def save(self, path: str) -> None:
        """Save tokenizer state to a JSON file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        state = {
            'max_vocab_size': self.max_vocab_size,
            'max_length': self.max_length,
            'word2idx': self.word2idx,
        }
        with open(path, 'w') as f:
            json.dump(state, f)
        logger.info(f"Saved tokenizer to {path}")

    def load(self, path: str) -> 'WordTokenizer':
        """Load tokenizer state from a JSON file."""
        with open(path, 'r') as f:
            state = json.load(f)
        self.max_vocab_size = state['max_vocab_size']
        self.max_length = state['max_length']
        self.word2idx = state['word2idx']
        self.idx2word = {int(idx): word for word, idx in self.word2idx.items()}
        self.is_fitted = True
        logger.info(f"Loaded tokenizer from {path} (vocab_size={self.vocab_size})")
        return self
