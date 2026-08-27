"""
Tier 1: Multinomial Naive Bayes classifier.

Implements §8/§20:
- MultinomialNB(alpha=1.0)
- Grid search over alpha ∈ {0.1, 0.5, 1.0}
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes for sentiment classification.

    Second most-used technique in the reviewed literature (16/54 papers).
    Good minority-class behavior in some studies.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Additive (Laplace) smoothing parameter.
        """
        self.alpha = alpha
        self.model = MultinomialNB(alpha=alpha)
        self.is_fitted = False

    def fit(self, X, y) -> 'NaiveBayesClassifier':
        """Train the NB classifier."""
        logger.info(f"Training MultinomialNB (alpha={self.alpha})...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("  Training complete.")
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(X)

    def get_config(self) -> Dict[str, Any]:
        return {
            'model_type': 'multinomial_nb',
            'alpha': self.alpha,
        }


def grid_search_nb(
    X_train, y_train,
    param_grid: Optional[Dict] = None,
    cv: int = 3,
    scoring: str = 'f1_macro',
) -> Dict[str, Any]:
    """
    Grid search for NB hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        param_grid: Parameter grid. Defaults to alpha ∈ {0.1, 0.5, 1.0}.
        cv: Number of CV folds.
        scoring: Scoring metric.

    Returns:
        Dict with best params, scores, and fitted model.
    """
    if param_grid is None:
        param_grid = {'alpha': [0.1, 0.5, 1.0]}

    logger.info(f"Running NB grid search: {param_grid}")

    gs = GridSearchCV(
        MultinomialNB(),
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)

    logger.info(f"  Best params: {gs.best_params_}")
    logger.info(f"  Best CV score ({scoring}): {gs.best_score_:.4f}")

    return {
        'best_params': gs.best_params_,
        'best_score': gs.best_score_,
        'cv_results': {
            'params': [str(p) for p in gs.cv_results_['params']],
            'mean_scores': gs.cv_results_['mean_test_score'].tolist(),
            'std_scores': gs.cv_results_['std_test_score'].tolist(),
        },
        'best_model': gs.best_estimator_,
    }
