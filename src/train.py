"""Train and evaluate the match outcome classifier (logistic regression / XGBoost)."""

import pandas as pd

from src import config


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    raise NotImplementedError


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    raise NotImplementedError


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return accuracy, log loss, and a classification report."""
    raise NotImplementedError


def save_model(model, name: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    pass
