"""
Tier 1: Linear SVM classifier for sentiment classification.

Implements §8/§20:
- LinearSVC(C=1.0, class_weight='balanced', max_iter=5000)
- Grid search over C ∈ {0.1, 1, 10}
"""

import logging
from typing import Optional, Dict, Any, List

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class SVMClassifier:
    """
    Linear SVM for sentiment classification.

    Wraps sklearn LinearSVC with optional calibration for probability outputs
    and hyperparameter grid search.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = 'balanced',
        max_iter: int = 5000,
        calibrate: bool = True,
        random_state: int = 42,
    ):
        """
        Args:
            C: Regularization parameter.
            class_weight: 'balanced' for inverse-frequency weighting.
            max_iter: Maximum iterations for convergence.
            calibrate: Whether to wrap with CalibratedClassifierCV for probability outputs.
            random_state: Random seed.
        """
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.random_state = random_state

        self.model = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            dual='auto',
        )

        if calibrate:
            self.calibrated_model = CalibratedClassifierCV(
                self.model, cv=3, method='sigmoid'
            )
        else:
            self.calibrated_model = None

        self.is_fitted = False

    def fit(self, X, y) -> 'SVMClassifier':
        """
        Train the SVM classifier.

        Args:
            X: Feature matrix (TF-IDF sparse matrix or dense array).
            y: Label array.

        Returns:
            Self.
        """
        logger.info(f"Training LinearSVC (C={self.C}, class_weight={self.class_weight})...")

        if self.calibrate:
            self.calibrated_model.fit(X, y)
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        logger.info("  Training complete.")
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self.calibrate:
            return self.calibrated_model.predict(X)
        return self.model.predict(X)

    def predict_proba(self, X) -> Optional[np.ndarray]:
        """
        Predict class probabilities (only if calibrated).

        Returns:
            Probability array of shape (n_samples, n_classes) or None.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self.calibrate:
            return self.calibrated_model.predict_proba(X)
        return None

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'model_type': 'linear_svm',
            'C': self.C,
            'class_weight': self.class_weight,
            'max_iter': self.max_iter,
            'calibrate': self.calibrate,
        }


def grid_search_svm(
    X_train, y_train,
    X_val=None, y_val=None,
    param_grid: Optional[Dict] = None,
    cv: int = 3,
    scoring: str = 'f1_macro',
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run grid search for SVM hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (unused; CV is used internally).
        y_val: Validation labels (unused).
        param_grid: Parameter grid. Defaults to C ∈ {0.1, 1, 10}.
        cv: Number of CV folds.
        scoring: Scoring metric.
        random_state: Random seed.

    Returns:
        Dict with best params, scores, and fitted model.
    """
    if param_grid is None:
        param_grid = {'C': [0.1, 1.0, 10.0]}

    logger.info(f"Running SVM grid search: {param_grid}")

    base_model = LinearSVC(
        class_weight='balanced',
        max_iter=5000,
        random_state=random_state,
        dual='auto',
    )

    gs = GridSearchCV(
        base_model,
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
