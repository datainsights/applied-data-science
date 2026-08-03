#!/usr/bin/env python3
"""Generate the model-evaluation figures used in ADS Chapter 09."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
OUTPUT_DIR = Path("results/figures")


def configure_style() -> None:
    """Apply a consistent, accessible plotting style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def create_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a reproducible imbalanced binary-classification example."""
    features, target = make_classification(
        n_samples=2_000,
        n_features=14,
        n_informative=7,
        n_redundant=3,
        weights=[0.82, 0.18],
        class_sep=1.05,
        flip_y=0.025,
        random_state=RANDOM_STATE,
    )
    return train_test_split(
        features,
        target,
        test_size=0.25,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def build_model():
    """Return a leakage-safe scaling and classification pipeline."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
    )


def plot_cross_validation(model, x_train, y_train) -> None:
    """Plot fold-level performance for complementary metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    result = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring={
            "ROC AUC": "roc_auc",
            "Average precision": "average_precision",
            "1 − log loss": "neg_log_loss",
        },
    )

    rows = []
    for metric in ("ROC AUC", "Average precision", "1 − log loss"):
        values = result[f"test_{metric}"]
        if metric == "1 − log loss":
            values = 1 + values
        rows.extend(
            {"Metric": metric, "Fold": fold, "Score": score}
            for fold, score in enumerate(values, start=1)
        )
    scores = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.stripplot(
        data=scores,
        x="Score",
        y="Metric",
        hue="Metric",
        palette="colorblind",
        size=8,
        jitter=0.08,
        legend=False,
        ax=ax,
    )
    for position, (_, group) in enumerate(scores.groupby("Metric", sort=False)):
        mean = group["Score"].mean()
        std = group["Score"].std(ddof=1)
        ax.errorbar(mean, position, xerr=std, fmt="D", color="#222222", capsize=5)
    ax.set(
        title="Cross-validation reveals performance variability",
        xlabel="Score",
        ylabel=None,
    )
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09-cross-validation-performance.png", bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(y_test, probability) -> None:
    """Plot discrimination and calibration diagnostics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    roc_display = RocCurveDisplay.from_predictions(
        y_test,
        probability,
        ax=axes[0],
    )
    roc_display.line_.set_color("#0072B2")

    pr_display = PrecisionRecallDisplay.from_predictions(
        y_test,
        probability,
        ax=axes[1],
    )
    pr_display.line_.set_color("#D55E00")

    calibration_display = CalibrationDisplay.from_predictions(
        y_test,
        probability,
        n_bins=8,
        strategy="quantile",
        ax=axes[2],
    )
    calibration_display.line_.set_color("#009E73")

    axes[0].set_title("Discrimination: ROC")
    axes[1].set_title("Positive-case retrieval")
    axes[2].set_title("Probability calibration")
    fig.suptitle("Held-out classification diagnostics", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09-classification-diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(
    y_validation,
    validation_probability,
) -> float:
    """Plot validation-set precision and recall and return a threshold."""
    precision, recall, thresholds = precision_recall_curve(
        y_validation,
        validation_probability,
    )
    eligible = np.where(recall[:-1] >= 0.80)[0]
    selected = thresholds[eligible[np.argmax(precision[:-1][eligible])]]

    curve = pd.DataFrame(
        {
            "Threshold": thresholds,
            "Precision": precision[:-1],
            "Recall": recall[:-1],
        }
    ).melt("Threshold", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.lineplot(
        data=curve,
        x="Threshold",
        y="Score",
        hue="Metric",
        palette={"Precision": "#D55E00", "Recall": "#0072B2"},
        linewidth=2.3,
        ax=ax,
    )
    ax.axvline(selected, color="#222222", linestyle="--", linewidth=1.5)
    ax.text(
        selected + 0.02,
        0.08,
        f"Selected = {selected:.2f}",
        rotation=90,
        va="bottom",
    )
    ax.set(
        title="Classification thresholds encode a decision trade-off",
        xlabel="Classification threshold",
        ylabel="Score",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09-threshold-tradeoff.png", bbox_inches="tight")
    plt.close(fig)
    return float(selected)


def main() -> None:
    """Fit the example model, generate figures, and print key metrics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    x_train, x_test, y_train, y_test = create_data()
    model = build_model()

    plot_cross_validation(model, x_train, y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    validation_probability = cross_val_predict(
        model,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    selected_threshold = plot_threshold_tradeoff(
        y_train,
        validation_probability,
    )

    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    plot_diagnostics(y_test, probability)

    print(f"ROC AUC: {roc_auc_score(y_test, probability):.3f}")
    print(f"Average precision: {average_precision_score(y_test, probability):.3f}")
    print(f"Log loss: {log_loss(y_test, probability):.3f}")
    print(f"Demonstration threshold: {selected_threshold:.3f}")
    print(f"Figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
