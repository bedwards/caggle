#!/usr/bin/env python3
"""
Playground Series S5E12 - Diabetes Prediction Challenge
Baseline model using LightGBM with 5-fold CV
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_competition_data, save_submission

COMPETITION = "playground-series-s5e12"
SEED = 42
N_SPLITS = 5


def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Preprocess data - encode categoricals."""
    # Identify columns
    target_col = "diagnosed_diabetes"
    id_col = "id"

    # Separate features
    feature_cols = [c for c in train.columns if c not in [target_col, id_col]]

    # Identify categorical columns
    cat_cols = train[feature_cols].select_dtypes(include=["object"]).columns.tolist()

    # Label encode categoricals
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on combined train + test to handle unseen categories
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        encoders[col] = le

    X = train[feature_cols]
    y = train[target_col]
    X_test = test[feature_cols]
    test_ids = test[id_col]

    return X, y, X_test, test_ids, cat_cols


def train_lgb_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cat_cols: list,
) -> tuple:
    """Train LightGBM with cross-validation."""

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_estimators": 1000,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{N_SPLITS}")
        print(f"{'='*50}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100),
            ],
        )

        # Predictions
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

        # Score
        fold_score = roc_auc_score(y_val, val_preds)
        scores.append(fold_score)
        print(f"Fold {fold + 1} AUC: {fold_score:.5f}")

    print(f"\n{'='*50}")
    print(f"CV Results")
    print(f"{'='*50}")
    print(f"Mean AUC: {np.mean(scores):.5f} (+/- {np.std(scores):.5f})")
    print(f"OOF AUC:  {roc_auc_score(y, oof_preds):.5f}")

    return oof_preds, test_preds, scores


def main():
    print("Loading data...")
    train, test, sample_sub = load_competition_data(COMPETITION)

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    print("\nPreprocessing...")
    X, y, X_test, test_ids, cat_cols = preprocess(train, test)
    print(f"Features: {X.shape[1]}")
    print(f"Categorical: {cat_cols}")

    print("\nTraining LightGBM...")
    oof_preds, test_preds, scores = train_lgb_cv(X, y, X_test, cat_cols)

    # Create submission
    submission = pd.DataFrame({
        "id": test_ids,
        "diagnosed_diabetes": test_preds,
    })

    mean_score = np.mean(scores)
    sub_name = f"lgb_baseline_{mean_score:.5f}"
    save_submission(submission, COMPETITION, sub_name)

    print(f"\nDone! Submission saved as {sub_name}.csv")
    print(f"To submit: kaggle competitions submit -c {COMPETITION} -f submissions/{COMPETITION}/{sub_name}.csv -m 'LightGBM baseline'")


if __name__ == "__main__":
    main()
