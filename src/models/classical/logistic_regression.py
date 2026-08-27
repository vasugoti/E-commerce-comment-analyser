"""
Tier 1: Logistic Regression classifier.

Implements §8/§20:
- LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000,
                     solver='lbfgs', multi_class='multinomial')
- Provides calibrated probabilities for ROC-AUC
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


class LogisticRegressionClassifier:
    """
    Logistic Regression for sentiment classification.

    Provides calibrated probabilities, making it suitable for ROC-AUC
    computation and a fairer "linear model" comparison than NB.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = 'balanced',
        max_iter: int = 1000,
        solver: str = 'lbfgs',
        random_state: int = 42,
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state

        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            solver=solver,
            multi_class='multinomial',
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X, y) -> 'LogisticRegressionClassifier':
        """Train the LR classifier."""
        logger.info(f"Training LogisticRegression (C={self.C}, solver={self.solver})...")
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
        """Predict class probabilities (natively calibrated)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(X)

    def get_config(self) -> Dict[str, Any]:
        return {
            'model_type': 'logistic_regression',
            'C': self.C,
            'class_weight': self.class_weight,
            'max_iter': self.max_iter,
            'solver': self.solver,
        }


def grid_search_lr(
    X_train, y_train,
    param_grid: Optional[Dict] = None,
    cv: int = 3,
    scoring: str = 'f1_macro',
    random_state: int = 42,
) -> Dict[str, Any]:
    """Grid search for LR hyperparameters."""
    if param_grid is None:
        param_grid = {'C': [0.1, 1.0, 10.0]}

    logger.info(f"Running LR grid search: {param_grid}")

    gs = GridSearchCV(
        LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=random_state,
            n_jobs=-1,
        ),
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
