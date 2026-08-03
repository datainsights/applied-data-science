#!/usr/bin/env python3
"""Generate the model-improvement results and figures for ADS Chapter 10."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform
from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "model-improvement"
FIGURES_DIR = ROOT / "results" / "figures"


def build_baseline() -> Pipeline:
    """Return the reproducible baseline pipeline."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
            ),
        ]
    )


def build_search(cv: StratifiedKFold) -> RandomizedSearchCV:
    """Return a focused hyperparameter search around the baseline."""
    return RandomizedSearchCV(
        estimator=build_baseline(),
        param_distributions={
            "model__C": loguniform(1e-3, 1e2),
            "model__class_weight": [None, "balanced"],
        },
        n_iter=24,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        return_train_score=True,
    )


def save_learning_curve(model, X, y, cv) -> None:
    """Plot training and validation performance as sample size increases."""
    sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring="average_precision",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)
    validation_sd = validation_scores.std(axis=1, ddof=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_mean, marker="o", label="Training")
    ax.plot(sizes, validation_mean, marker="o", label="Validation")
    ax.fill_between(
        sizes,
        validation_mean - validation_sd,
        validation_mean + validation_sd,
        alpha=0.2,
        label="Validation ±1 SD",
    )
    ax.set(
        xlabel="Training observations",
        ylabel="Average precision (PR AUC)",
        title="Learning curve: does more data still help?",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10-learning-curve.png", dpi=180)
    plt.close(fig)


def save_validation_curve(model, X, y, cv) -> None:
    """Plot the effect of logistic-regression regularization."""
    values = np.logspace(-3, 2, 8)
    train_scores, validation_scores = validation_curve(
        estimator=model,
        X=X,
        y=y,
        param_name="model__C",
        param_range=values,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(values, train_scores.mean(axis=1), marker="o", label="Training")
    ax.semilogx(
        values,
        validation_scores.mean(axis=1),
        marker="o",
        label="Validation",
    )
    ax.set(
        xlabel="Inverse regularization strength (C)",
        ylabel="Average precision (PR AUC)",
        title="Validation curve: tune complexity deliberately",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10-validation-curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run the complete demonstration without using the test set for tuning."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    X, y = make_classification(
        n_samples=2_000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        weights=[0.82, 0.18],
        flip_y=0.025,
        class_sep=1.0,
        random_state=RANDOM_STATE,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    inner_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=2026
    )
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

    baseline_scores = cross_validate(
        build_baseline(), X_train, y_train, cv=outer_cv, scoring=scoring, n_jobs=-1
    )
    nested_scores = cross_validate(
        build_search(inner_cv),
        X_train,
        y_train,
        cv=outer_cv,
        scoring=scoring,
        n_jobs=1,
    )

    comparison = pd.DataFrame(
        {
            "fold": np.arange(1, outer_cv.get_n_splits() + 1),
            "baseline_roc_auc": baseline_scores["test_roc_auc"],
            "tuned_roc_auc": nested_scores["test_roc_auc"],
            "baseline_pr_auc": baseline_scores["test_pr_auc"],
            "tuned_pr_auc": nested_scores["test_pr_auc"],
        }
    )
    comparison["pr_auc_difference"] = (
        comparison["tuned_pr_auc"] - comparison["baseline_pr_auc"]
    )
    comparison.to_csv(RESULTS_DIR / "10-model-comparison.csv", index=False)

    search = build_search(inner_cv)
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_

    oof_probability = cross_val_predict(
        tuned_model,
        X_train,
        y_train,
        cv=inner_cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        y_train, oof_probability
    )
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    selected_threshold = float(thresholds[np.argmax(f1)])

    tuned_model.fit(X_train, y_train)
    test_probability = tuned_model.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= selected_threshold).astype(int)

    summary = pd.DataFrame(
        [
            {
                "selected_threshold": selected_threshold,
                "selection_rule": "maximum out-of-fold F1 on training data",
                "oof_roc_auc": roc_auc_score(y_train, oof_probability),
                "oof_pr_auc": average_precision_score(y_train, oof_probability),
                "test_roc_auc": roc_auc_score(y_test, test_probability),
                "test_pr_auc": average_precision_score(y_test, test_probability),
                "test_precision": precision_score(y_test, test_prediction),
                "test_recall": recall_score(y_test, test_prediction),
                "test_f1": f1_score(y_test, test_prediction),
                "best_C": search.best_params_["model__C"],
                "best_class_weight": search.best_params_["model__class_weight"],
            }
        ]
    )
    summary.to_csv(RESULTS_DIR / "10-threshold-summary.csv", index=False)

    save_learning_curve(tuned_model, X_train, y_train, inner_cv)
    save_validation_curve(build_baseline(), X_train, y_train, inner_cv)

    fig, ax = plt.subplots(figsize=(6, 6))
    CalibrationDisplay.from_predictions(
        y_train,
        oof_probability,
        n_bins=10,
        strategy="quantile",
        name="Tuned model",
        ax=ax,
    )
    ax.set_title("Out-of-fold probability calibration")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10-calibration-curve.png", dpi=180)
    plt.close(fig)

    print(f"Best parameters: {search.best_params_}")
    print(f"Selected threshold: {selected_threshold:.3f}")
    print(f"Test ROC AUC: {summary.loc[0, 'test_roc_auc']:.3f}")
    print(f"Test average precision: {summary.loc[0, 'test_pr_auc']:.3f}")
    print("Figures written to results/figures")


if __name__ == "__main__":
    main()
