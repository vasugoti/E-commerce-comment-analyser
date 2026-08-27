"""
Pretrained embedding loaders (GloVe, FastText).

Loads pretrained word vectors and builds an embedding matrix aligned
to the vocabulary from our WordTokenizer. Supports random initialization
as a baseline for the embedding-source ablation (§11).
"""

import os
import logging
from typing import Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)


def load_glove_vectors(glove_path: str, embedding_dim: int = 300) -> Dict[str, np.ndarray]:
    """
    Load GloVe vectors from a text file.

    Args:
        glove_path: Path to GloVe file (e.g., glove.6B.300d.txt).
        embedding_dim: Expected embedding dimension.

    Returns:
        Dict mapping words to numpy vectors.
    """
    logger.info(f"Loading GloVe vectors from {glove_path}...")
    vectors = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            parts = line.rstrip().split(' ')
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == embedding_dim:
                    vectors[word] = vec
            except ValueError:
                continue  # Skip malformed lines

            if (line_num + 1) % 100000 == 0:
                logger.info(f"  Loaded {line_num + 1} vectors...")

    logger.info(f"  Total GloVe vectors: {len(vectors)}")
    return vectors


def load_fasttext_vectors(fasttext_path: str,
                          embedding_dim: int = 300,
                          max_vectors: int = 500000) -> Dict[str, np.ndarray]:
    """
    Load FastText vectors from a text file.

    Args:
        fasttext_path: Path to FastText file.
        embedding_dim: Expected embedding dimension.
        max_vectors: Maximum number of vectors to load.

    Returns:
        Dict mapping words to numpy vectors.
    """
    logger.info(f"Loading FastText vectors from {fasttext_path}...")
    vectors = {}
    with open(fasttext_path, 'r', encoding='utf-8') as f:
        # First line of FastText format is often header: num_words dim
        first_line = f.readline().strip().split()
        if len(first_line) == 2:
            try:
                int(first_line[0])
                # It's a header line, skip it
            except ValueError:
                # Not a header, treat as a vector
                f.seek(0)

        for line_num, line in enumerate(f):
            if line_num >= max_vectors:
                break
            parts = line.rstrip().split(' ')
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == embedding_dim:
                    vectors[word] = vec
            except ValueError:
                continue

    logger.info(f"  Total FastText vectors: {len(vectors)}")
    return vectors


def build_embedding_matrix(
    word2idx: Dict[str, int],
    pretrained_vectors: Optional[Dict[str, np.ndarray]] = None,
    embedding_dim: int = 300,
    init_method: str = 'uniform',
    seed: int = 42,
) -> np.ndarray:
    """
    Build an embedding matrix aligned to the tokenizer vocabulary.

    Args:
        word2idx: Dictionary mapping words to indices.
        pretrained_vectors: Dict of pretrained word vectors. None for random init.
        embedding_dim: Embedding dimension.
        init_method: Initialization for OOV words ('uniform', 'normal', 'zero').
        seed: Random seed.

    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim).
    """
    vocab_size = len(word2idx)
    rng = np.random.RandomState(seed)

    # Initialize
    if init_method == 'uniform':
        scale = np.sqrt(3.0 / embedding_dim)  # Xavier-like
        matrix = rng.uniform(-scale, scale, (vocab_size, embedding_dim)).astype(np.float32)
    elif init_method == 'normal':
        matrix = rng.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01
    else:
        matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    # Zero out PAD token
    matrix[0] = 0.0  # PAD_IDX = 0

    if pretrained_vectors is not None:
        found = 0
        for word, idx in word2idx.items():
            if word in pretrained_vectors:
                matrix[idx] = pretrained_vectors[word]
                found += 1

        coverage = found / vocab_size * 100
        logger.info(f"  Embedding coverage: {found}/{vocab_size} ({coverage:.1f}%)")
    else:
        logger.info(f"  Using random initialization ({init_method}, dim={embedding_dim})")

    return matrix


def get_embedding_matrix(
    word2idx: Dict[str, int],
    source: str = 'glove',
    embedding_dim: int = 300,
    embeddings_dir: str = 'data/embeddings',
    seed: int = 42,
) -> np.ndarray:
    """
    High-level function to get an embedding matrix for a given source.

    Args:
        word2idx: Dictionary mapping words to indices.
        source: 'glove', 'fasttext', or 'random'.
        embedding_dim: Embedding dimension.
        embeddings_dir: Directory containing pretrained embedding files.
        seed: Random seed.

    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim).
    """
    if source == 'random':
        logger.info("Using random embeddings (ablation baseline)")
        return build_embedding_matrix(word2idx, None, embedding_dim, seed=seed)

    elif source == 'glove':
        glove_path = os.path.join(embeddings_dir, f'glove.6B.{embedding_dim}d.txt')
        if not os.path.exists(glove_path):
            logger.warning(f"GloVe file not found at {glove_path}. "
                           f"Download from https://nlp.stanford.edu/data/glove.6B.zip")
            logger.info("Falling back to random initialization")
            return build_embedding_matrix(word2idx, None, embedding_dim, seed=seed)
        vectors = load_glove_vectors(glove_path, embedding_dim)
        return build_embedding_matrix(word2idx, vectors, embedding_dim, seed=seed)

    elif source == 'fasttext':
        fasttext_path = os.path.join(embeddings_dir, f'wiki-news-300d-1M.vec')
        if not os.path.exists(fasttext_path):
            logger.warning(f"FastText file not found at {fasttext_path}. "
                           f"Download from https://fasttext.cc/docs/en/english-vectors.html")
            logger.info("Falling back to random initialization")
            return build_embedding_matrix(word2idx, None, embedding_dim, seed=seed)
        vectors = load_fasttext_vectors(fasttext_path, embedding_dim)
        return build_embedding_matrix(word2idx, vectors, embedding_dim, seed=seed)

    else:
        raise ValueError(f"Unknown embedding source: {source}")
