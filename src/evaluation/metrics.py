"""
Model Evaluation Utilities
Provides metrics, cross-validation, and visualization for all models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classifier(y_true, y_pred, y_prob=None, model_name="Model"):
    """Evaluate a binary classifier."""
    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_prob)
        results["log_loss"] = log_loss(y_true, y_prob)
        results["brier_score"] = brier_score_loss(y_true, y_prob)

    print(f"\n{'='*50}")
    print(f"  {model_name} - Classification Results")
    print(f"{'='*50}")
    for k, v in results.items():
        if k != "model":
            print(f"  {k:>15}: {v:.4f}")

    return results


def evaluate_regressor(y_true, y_pred, model_name="Model"):
    """Evaluate a regression model."""
    results = {
        "model": model_name,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} - Regression Results")
    print(f"{'='*50}")
    for k, v in results.items():
        if k != "model":
            print(f"  {k:>15}: {v:.4f}")

    return results


def temporal_train_test_split(df: pd.DataFrame, season_col: str = "season",
                               train_end: int = 2022, val_end: int = 2025):
    """
    Temporal train/val/test split.
    Train: up to train_end
    Val: train_end+1 to val_end
    Test: val_end+1 onwards
    """
    # Handle season formats like "2007/08", "2020/21"
    def season_to_int(s):
        try:
            return int(s)
        except (ValueError, TypeError):
            if "/" in str(s):
                return int(str(s).split("/")[0])
            return 0

    df = df.copy()
    df["_season_int"] = df[season_col].apply(season_to_int)

    train = df[df["_season_int"] <= train_end].drop(columns=["_season_int"])
    val = df[(df["_season_int"] > train_end) & (df["_season_int"] <= val_end)].drop(columns=["_season_int"])
    test = df[df["_season_int"] > val_end].drop(columns=["_season_int"])

    print(f"Train: {len(train)} samples (up to {train_end})")
    print(f"Val:   {len(val)} samples ({train_end+1}-{val_end})")
    print(f"Test:  {len(test)} samples (after {val_end})")

    return train, val, test


def get_feature_columns(df: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    """Get numeric feature columns, excluding specified columns."""
    if exclude is None:
        exclude = []

    # Standard columns to always exclude
    always_exclude = [
        "match_id", "season", "venue", "team1", "team2", "batting_team",
        "bowling_team", "player", "opposition", "target", "target_won",
        "final_score", "actual_runs", "actual_sr", "actual_wickets",
        "actual_economy", "winner", "date", "city", "phase"
    ]
    exclude = list(set(exclude + always_exclude))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels or ["Loss", "Win"],
                yticklabels=labels or ["Loss", "Win"])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    return plt


def plot_calibration(y_true, y_prob, n_bins=10, title="Calibration Curve"):
    """Plot calibration curve for probability predictions."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """Plot feature importance from tree-based model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        print("Model does not have feature_importances_ or coef_")
        return None

    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)

    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(feat_imp['feature'], feat_imp['importance'])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    return plt
