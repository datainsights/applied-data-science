#!/usr/bin/env python3
"""Generate the pipeline and cross-validation figures used in ADS Chapter 12."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
OUTPUT_DIR = Path("results/figures")


def build_demo_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create an imbalanced classification dataset with missing values."""
    features, target = make_classification(
        n_samples=1_000,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        weights=[0.82, 0.18],
        class_sep=1.0,
        flip_y=0.03,
        random_state=RANDOM_STATE,
    )

    rng = np.random.default_rng(RANDOM_STATE)
    missing = rng.random(features.shape) < 0.04
    features[missing] = np.nan

    columns = [f"feature_{index:02d}" for index in range(1, 13)]
    return pd.DataFrame(features, columns=columns), pd.Series(target, name="target")


def build_pipeline():
    """Return the complete leakage-safe modelling procedure."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2_000),
    )


def calculate_fold_results(pipeline, features, target, cv) -> pd.DataFrame:
    """Fit the pipeline within each fold and return validation metrics."""
    records = []
    for fold, (train_index, validation_index) in enumerate(cv.split(features, target), 1):
        pipeline.fit(features.iloc[train_index], target.iloc[train_index])
        probability = pipeline.predict_proba(features.iloc[validation_index])[:, 1]
        validation_target = target.iloc[validation_index]
        records.extend(
            [
                {"fold": fold, "metric": "ROC AUC", "score": roc_auc_score(validation_target, probability)},
                {
                    "fold": fold,
                    "metric": "Average precision",
                    "score": average_precision_score(validation_target, probability),
                },
            ]
        )
    return pd.DataFrame(records)


def plot_fold_scores(results: pd.DataFrame) -> None:
    """Plot validation metrics for every cross-validation fold."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=results,
        x="fold",
        y="score",
        hue="metric",
        marker="o",
        linewidth=2.2,
        markersize=8,
        ax=ax,
    )
    ax.set(title="Validation performance varies across folds", xlabel="Validation fold", ylabel="Score")
    ax.set_xticks(sorted(results["fold"].unique()))
    ax.set_ylim(0.5, 1.0)
    ax.legend(title=None, frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12-cross-validation-fold-scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(pipeline, features, target, cv) -> None:
    """Plot calibration using probabilities predicted out of fold."""
    probability = cross_val_predict(
        pipeline,
        features,
        target,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    CalibrationDisplay.from_predictions(
        target,
        probability,
        n_bins=10,
        strategy="quantile",
        name="Pipeline",
        ax=ax,
    )
    ax.set_title("Out-of-fold probability calibration")
    sns.despine()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12-out-of-fold-calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate and save all Chapter 12 figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    features, target = build_demo_data()
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_results = calculate_fold_results(pipeline, features, target, cv)
    plot_fold_scores(fold_results)
    plot_calibration(pipeline, features, target, cv)

    summary = fold_results.groupby("metric")["score"].agg(["mean", "std"])
    print(summary.round(3))
    print(f"Figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
