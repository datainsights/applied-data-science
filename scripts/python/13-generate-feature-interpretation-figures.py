#!/usr/bin/env python3
"""Generate the model-interpretation figures used in ADS Chapter 13."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
OUTPUT_DIR = Path("results/figures")
TOP_N = 10


def load_split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the demonstration data and create a stratified holdout split."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def fit_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Fit the demonstration random forest."""
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return model.fit(X_train, y_train)


def plot_importance_comparison(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Compare fitted-tree impurity importance with held-out permutation importance."""
    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "impurity": model.feature_importances_,
            "permutation": permutation.importances_mean,
            "permutation_sd": permutation.importances_std,
        }
    )

    impurity_top = importance.nlargest(TOP_N, "impurity").sort_values("impurity")
    permutation_top = importance.nlargest(TOP_N, "permutation").sort_values(
        "permutation"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    axes[0].barh(
        impurity_top["feature"],
        impurity_top["impurity"],
        color=sns.color_palette("crest", TOP_N),
    )
    axes[0].set_title("Impurity importance\n(training-time split reductions)")
    axes[0].set_xlabel("Mean decrease in impurity")

    axes[1].barh(
        permutation_top["feature"],
        permutation_top["permutation"],
        xerr=permutation_top["permutation_sd"],
        color=sns.color_palette("flare", TOP_N),
        error_kw={"elinewidth": 1, "capsize": 2},
    )
    axes[1].axvline(0, color="0.35", linewidth=1)
    axes[1].set_title("Permutation importance\n(held-out ROC AUC decrease)")
    axes[1].set_xlabel("Mean score decrease across 30 permutations")

    fig.suptitle("Different importance methods answer different questions", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "13-feature-importance-comparison.png", dpi=300)
    plt.close(fig)


def plot_partial_dependence_ice(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
) -> None:
    """Plot average partial dependence together with individual ICE curves."""
    feature = "worst radius"
    fig, ax = plt.subplots(figsize=(9, 6.5))
    PartialDependenceDisplay.from_estimator(
        model,
        X_test,
        features=[feature],
        kind="both",
        subsample=min(80, len(X_test)),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        ice_lines_kw={"color": "#7A8B99", "alpha": 0.18, "linewidth": 0.8},
        pd_line_kw={"color": "#B23A48", "linewidth": 3, "label": "Average PDP"},
        ax=ax,
    )
    ax.set_title("Average response and individual variation")
    ax.set_ylabel("Predicted probability of the benign class")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "13-partial-dependence-ice.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Fit the example and write all Chapter 13 figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split_data()
    model = fit_model(X_train, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    plot_importance_comparison(model, X_test, y_test)
    plot_partial_dependence_ice(model, X_test)

    print(f"Held-out ROC AUC: {test_auc:.3f}")
    print(f"Figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
