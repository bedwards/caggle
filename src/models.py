"""Model training utilities."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def train_with_cv(
    model_class: Any,
    model_params: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Train model with cross-validation and return OOF and test predictions.

    Returns:
        oof_preds: Out-of-fold predictions for training set
        test_preds: Averaged predictions for test set
        scores: List of validation scores per fold
    """
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_class(**model_params)

        # Handle different model types
        model_name = model_class.__name__

        if model_name == "CatBoostClassifier":
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=categorical_features,
                verbose=100,
            )
        elif model_name == "LGBMClassifier":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[],
            )
        elif model_name == "XGBClassifier":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100,
            )
        else:
            model.fit(X_train, y_train)

        # Get predictions
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

        # Calculate score
        from sklearn.metrics import roc_auc_score
        fold_score = roc_auc_score(y_val, val_preds)
        scores.append(fold_score)
        print(f"  Fold {fold + 1} AUC: {fold_score:.5f}")

    print(f"\nMean AUC: {np.mean(scores):.5f} (+/- {np.std(scores):.5f})")

    return oof_preds, test_preds, scores
