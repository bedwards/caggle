#!/usr/bin/env python3
"""
Playground Series S5E12 - Diabetes Prediction Challenge
Ensemble model with proper validation strategy

Key insight from competition discussions:
- The first ~678K training samples have a different distribution than the test set
- The last ~22K training samples match the test distribution
- Using the last 22K as validation gives more reliable CV-LB correlation
"""
import sys
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_competition_data, save_submission

COMPETITION = "playground-series-s5e12"
SEED = 42
N_SPLITS = 5
VAL_SIZE = 22000  # Last 22K samples match test distribution


def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Encode categorical features as integers."""
    target_col = "diagnosed_diabetes"
    id_col = "id"

    feature_cols = [c for c in train.columns if c not in [target_col, id_col]]
    cat_cols = train[feature_cols].select_dtypes(include=["object"]).columns.tolist()

    # Label encode categoricals
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    X = train[feature_cols]
    y = train[target_col]
    X_test = test[feature_cols]
    test_ids = test[id_col]

    return X, y, X_test, test_ids, cat_cols, feature_cols


def train_lgb(X_train, y_train, X_val, y_val, X_test, cat_cols):
    """Train LightGBM model."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_estimators": 2000,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    return val_pred, test_pred, model


def train_xgb(X_train, y_train, X_val, y_val, X_test, cat_cols):
    """Train XGBoost model."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_estimators": 2000,
        "random_state": SEED,
        "n_jobs": -1,
        "enable_categorical": True,
        "tree_method": "hist",
    }

    # Convert categoricals for XGBoost
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = X_test.copy()

    for col in cat_cols:
        X_train_xgb[col] = X_train_xgb[col].astype("category")
        X_val_xgb[col] = X_val_xgb[col].astype("category")
        X_test_xgb[col] = X_test_xgb[col].astype("category")

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train_xgb, y_train,
        eval_set=[(X_val_xgb, y_val)],
        verbose=200,
    )

    val_pred = model.predict_proba(X_val_xgb)[:, 1]
    test_pred = model.predict_proba(X_test_xgb)[:, 1]

    return val_pred, test_pred, model


def train_catboost(X_train, y_train, X_val, y_val, X_test, cat_cols):
    """Train CatBoost model."""
    params = {
        "objective": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 3,
        "min_data_in_leaf": 50,
        "iterations": 2000,
        "random_seed": SEED,
        "thread_count": -1,
        "verbose": 200,
        "early_stopping_rounds": 100,
    }

    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    model = cb.CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_indices,
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    return val_pred, test_pred, model


def optimize_weights(val_preds: list, y_val: pd.Series) -> np.ndarray:
    """Find optimal ensemble weights using Ridge regression."""
    # Stack predictions as features
    X = np.column_stack(val_preds)

    # Use Ridge to find weights (regularization prevents overfitting)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y_val)

    # Normalize weights to sum to 1
    weights = ridge.coef_
    weights = np.maximum(weights, 0)  # Non-negative
    weights = weights / weights.sum()

    return weights


def main():
    print("=" * 60)
    print("Diabetes Prediction - Ensemble with Proper Validation")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train, test, sample_sub = load_competition_data(COMPETITION)
    print(f"Train: {train.shape}, Test: {test.shape}")

    # Preprocess
    print("\nPreprocessing...")
    X, y, X_test, test_ids, cat_cols, feature_cols = preprocess(train, test)

    # Split: Use last 22K as validation (matches test distribution)
    print(f"\nSplitting: last {VAL_SIZE} samples for validation...")
    X_train_full = X.iloc[:-VAL_SIZE]
    y_train_full = y.iloc[:-VAL_SIZE]
    X_val = X.iloc[-VAL_SIZE:]
    y_val = y.iloc[-VAL_SIZE:]

    print(f"Train: {len(X_train_full)}, Val: {len(X_val)}")
    print(f"Train target rate: {y_train_full.mean():.4f}")
    print(f"Val target rate: {y_val.mean():.4f}")

    # Train individual models
    print("\n" + "=" * 60)
    print("Training LightGBM...")
    print("=" * 60)
    lgb_val, lgb_test, lgb_model = train_lgb(
        X_train_full, y_train_full, X_val, y_val, X_test, cat_cols
    )
    lgb_score = roc_auc_score(y_val, lgb_val)
    print(f"LightGBM Val AUC: {lgb_score:.5f}")

    print("\n" + "=" * 60)
    print("Training XGBoost...")
    print("=" * 60)
    xgb_val, xgb_test, xgb_model = train_xgb(
        X_train_full, y_train_full, X_val, y_val, X_test, cat_cols
    )
    xgb_score = roc_auc_score(y_val, xgb_val)
    print(f"XGBoost Val AUC: {xgb_score:.5f}")

    print("\n" + "=" * 60)
    print("Training CatBoost...")
    print("=" * 60)
    cb_val, cb_test, cb_model = train_catboost(
        X_train_full, y_train_full, X_val, y_val, X_test, cat_cols
    )
    cb_score = roc_auc_score(y_val, cb_val)
    print(f"CatBoost Val AUC: {cb_score:.5f}")

    # Optimize ensemble weights
    print("\n" + "=" * 60)
    print("Optimizing ensemble weights...")
    print("=" * 60)

    val_preds = [lgb_val, xgb_val, cb_val]
    test_preds = [lgb_test, xgb_test, cb_test]
    model_names = ["LightGBM", "XGBoost", "CatBoost"]

    weights = optimize_weights(val_preds, y_val)

    print("\nOptimal weights:")
    for name, w in zip(model_names, weights):
        print(f"  {name}: {w:.4f}")

    # Create ensemble predictions
    ensemble_val = np.zeros(len(X_val))
    ensemble_test = np.zeros(len(X_test))

    for w, v_pred, t_pred in zip(weights, val_preds, test_preds):
        ensemble_val += w * v_pred
        ensemble_test += w * t_pred

    ensemble_score = roc_auc_score(y_val, ensemble_val)

    # Also try simple average for comparison
    simple_avg_val = np.mean(val_preds, axis=0)
    simple_avg_test = np.mean(test_preds, axis=0)
    simple_score = roc_auc_score(y_val, simple_avg_val)

    print("\n" + "=" * 60)
    print("RESULTS (Validation on last 22K samples)")
    print("=" * 60)
    print(f"\nIndividual models:")
    print(f"  LightGBM:  {lgb_score:.5f}")
    print(f"  XGBoost:   {xgb_score:.5f}")
    print(f"  CatBoost:  {cb_score:.5f}")
    print(f"\nEnsembles:")
    print(f"  Simple avg:     {simple_score:.5f}")
    print(f"  Weighted:       {ensemble_score:.5f}")

    # Choose best ensemble
    if ensemble_score >= simple_score:
        final_test = ensemble_test
        final_score = ensemble_score
        method = "weighted"
    else:
        final_test = simple_avg_test
        final_score = simple_score
        method = "simple_avg"

    print(f"\nUsing {method} ensemble (Val AUC: {final_score:.5f})")

    # Create submission
    submission = pd.DataFrame({
        "id": test_ids,
        "diagnosed_diabetes": final_test,
    })

    sub_name = f"ensemble_{method}_{final_score:.5f}"
    save_submission(submission, COMPETITION, sub_name)

    print(f"\nSubmission saved!")
    print(f"\nTo submit:")
    print(f"kaggle competitions submit -c {COMPETITION} \\")
    print(f"  -f submissions/{COMPETITION}/{sub_name}.csv \\")
    print(f"  -m 'LGB+XGB+CB ensemble, {method}, val22k={final_score:.5f}'")


if __name__ == "__main__":
    main()
