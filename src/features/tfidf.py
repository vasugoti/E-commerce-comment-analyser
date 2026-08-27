"""
TF-IDF feature extraction for Tier 1 classical ML models.

Implements §8/§20 defaults:
- ngram_range=(1,2), max_features=50000, min_df=2, sublinear_tf=True
- Ablation variants: uni-gram only, with/without stopwords, with/without lemmatization
"""

import os
import logging
import pickle
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class TfidfFeatureExtractor:
    """
    TF-IDF feature extraction with ablation support.

    Supports the preprocessing ablation in §11:
    - uni-gram vs uni+bi-gram
    - with/without stopwords
    - different max_features
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        use_stopwords: bool = False,
    ):
        """
        Args:
            max_features: Maximum vocabulary size.
            ngram_range: N-gram range (e.g., (1,1) for unigrams, (1,2) for uni+bigrams).
            min_df: Minimum document frequency.
            max_df: Maximum document frequency ratio.
            sublinear_tf: Apply sublinear TF scaling (1 + log(tf)).
            use_stopwords: Whether to use English stopwords.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.use_stopwords = use_stopwords

        stop_words = 'english' if use_stopwords else None

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            stop_words=stop_words,
            strip_accents='unicode',
            dtype=np.float32,
        )
        self.is_fitted = False

    def fit(self, texts: list) -> 'TfidfFeatureExtractor':
        """Fit the TF-IDF vectorizer on training texts."""
        logger.info(f"Fitting TF-IDF: max_features={self.max_features}, "
                     f"ngram_range={self.ngram_range}, sublinear_tf={self.sublinear_tf}")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"  Vocabulary size: {vocab_size}")
        return self

    def transform(self, texts: list) -> csr_matrix:
        """Transform texts to TF-IDF features."""
        if not self.is_fitted:
            raise RuntimeError("TF-IDF vectorizer is not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list) -> csr_matrix:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str) -> None:
        """Save the fitted vectorizer."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"Saved TF-IDF vectorizer to {path}")

    def load(self, path: str) -> 'TfidfFeatureExtractor':
        """Load a fitted vectorizer."""
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Loaded TF-IDF vectorizer from {path}")
        return self

    def get_config(self) -> dict:
        """Return configuration as a dictionary."""
        return {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'sublinear_tf': self.sublinear_tf,
            'use_stopwords': self.use_stopwords,
        }


# Pre-defined ablation variants (§11)
TFIDF_ABLATION_CONFIGS = {
    'default': {
        'max_features': 50000,
        'ngram_range': (1, 2),
        'sublinear_tf': True,
        'use_stopwords': False,
    },
    'unigram_only': {
        'max_features': 50000,
        'ngram_range': (1, 1),
        'sublinear_tf': True,
        'use_stopwords': False,
    },
    'with_stopwords': {
        'max_features': 50000,
        'ngram_range': (1, 2),
        'sublinear_tf': True,
        'use_stopwords': True,
    },
    'small_vocab': {
        'max_features': 20000,
        'ngram_range': (1, 2),
        'sublinear_tf': True,
        'use_stopwords': False,
    },
    'large_vocab': {
        'max_features': 100000,
        'ngram_range': (1, 2),
        'sublinear_tf': True,
        'use_stopwords': False,
    },
}


def create_tfidf_extractor(variant: str = 'default') -> TfidfFeatureExtractor:
    """Create a TF-IDF extractor with a predefined ablation configuration."""
    config = TFIDF_ABLATION_CONFIGS.get(variant, TFIDF_ABLATION_CONFIGS['default'])
    return TfidfFeatureExtractor(**config)
