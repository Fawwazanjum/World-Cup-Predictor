"""Train and evaluate the match outcome classifier (logistic regression / XGBoost)."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config
from src.features import FEATURE_COLUMNS


# Matches before this date are training data; from this date onward are test data.
# Choosing 2022 keeps ~4 years of modern football as the held-out test set while
# still training on the full 150-year history of the sport.
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature DataFrame into train (before cutoff) and test (from cutoff)."""
    train = df[df["date"] < TRAIN_CUTOFF].copy()
    test  = df[df["date"] >= TRAIN_CUTOFF].copy()
    return train, test


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[LogisticRegression, StandardScaler]:
    """Fit a multinomial logistic regression with L2 regularisation.

    Logistic regression needs features on a similar scale (it treats a rank_gap
    of 50 as 50× more important than 1 otherwise), so we standardise first:
    subtract the mean and divide by the standard deviation of each feature.
    The scaler is fitted ONLY on training data, then applied to test/prediction
    data, so no test information leaks into the scaling step.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Fit an XGBoost gradient-boosted tree classifier.

    XGBoost doesn't need feature scaling (trees split on thresholds, not
    distances) and handles the class imbalance in our dataset better than
    logistic regression out of the box. num_class=3 tells it this is a
    3-class problem (0=away win, 1=draw, 2=home win).
    """
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=config.RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler | None = None,
) -> dict:
    """Print accuracy, log-loss, and classification report; return metrics dict."""
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc  = accuracy_score(y_test, y_pred)
    ll   = log_loss(y_test, y_prob)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Log-loss : {ll:.4f}  (lower = better calibrated probabilities)")
    print()
    print(classification_report(y_test, y_pred, target_names=["Away win","Draw","Home win"]))
    return {"name": name, "accuracy": acc, "log_loss": ll}


def save_model(obj: object, filename: str) -> Path:
    path = config.MODELS_DIR / filename
    config.MODELS_DIR.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


if __name__ == "__main__":
    from src import data_loader, features

    print("Loading and building features...")
    df = data_loader.load_results()
    played, _ = data_loader.split_played_and_pending(df)
    played     = data_loader.filter_competitive(played)
    rankings   = data_loader.load_rankings()
    fs         = features.build_feature_set(played, rankings)

    train_df, test_df = time_split(fs)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["result"]
    X_test  = test_df[FEATURE_COLUMNS]
    y_test  = test_df["result"]

    print(f"Training rows : {len(train_df):,}  (before {TRAIN_CUTOFF.date()})")
    print(f"Test rows     : {len(test_df):,}   (from {TRAIN_CUTOFF.date()})")

    print("\nTraining logistic regression...")
    lr, scaler = train_logistic_regression(X_train, y_train)

    print("Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)

    lr_metrics  = evaluate("Logistic Regression", lr,  X_test, y_test, scaler)
    xgb_metrics = evaluate("XGBoost",             xgb, X_test, y_test)

    # Save both models (and the LR scaler, needed at inference time)
    save_model({"model": lr, "scaler": scaler}, "logistic_regression.pkl")
    save_model(xgb, "xgboost.pkl")
    print("\nModels saved to models/")

    # Report which model had better log-loss (the metric that matters most for
    # a probability-based simulation — we care about calibration, not just accuracy)
    best = min([lr_metrics, xgb_metrics], key=lambda m: m["log_loss"])
    print(f"\nBest model by log-loss: {best['name']}  (log-loss {best['log_loss']:.4f})")
