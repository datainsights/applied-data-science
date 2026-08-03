"""Generate the illustrations used in ADS Chapter 03.

Run from the project root:
    python scripts/python/03-generate_feature_engineering_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RANDOM_SEED = 42
OUTPUT_DIR = Path("results/figures")


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save one figure using consistent output settings."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_log_transformation(rng: np.random.Generator) -> None:
    """Compare a right-skewed feature with its log transformation."""
    annual_income = rng.lognormal(mean=10.7, sigma=0.75, size=1_200)
    plot_data = pd.DataFrame(
        {
            "annual_income": annual_income,
            "log_annual_income": np.log1p(annual_income),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.histplot(
        data=plot_data,
        x="annual_income",
        bins=35,
        color="#036281",
        edgecolor="white",
        ax=axes[0],
    )
    axes[0].set(
        title="Original feature",
        xlabel="Annual income",
        ylabel="Observations",
    )
    axes[0].ticklabel_format(axis="x", style="plain")

    sns.histplot(
        data=plot_data,
        x="log_annual_income",
        bins=35,
        color="#d97706",
        edgecolor="white",
        ax=axes[1],
    )
    axes[1].set(
        title="After log1p transformation",
        xlabel="log(1 + annual income)",
        ylabel="Observations",
    )
    fig.suptitle("A log transformation reduces strong right skew", y=1.02)
    fig.tight_layout()
    save_figure(fig, "03-log-transformation.png")


def plot_age_binning(rng: np.random.Generator) -> None:
    """Show the thresholds used to convert continuous age into groups."""
    ages = np.clip(rng.normal(loc=43, scale=18, size=1_000), 0, 89)
    age_data = pd.DataFrame({"age": ages})
    boundaries = [18, 35, 50, 65]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    sns.histplot(
        data=age_data,
        x="age",
        binwidth=3,
        color="#5aa9a2",
        edgecolor="white",
        ax=ax,
    )
    for boundary in boundaries:
        ax.axvline(boundary, color="#b42318", linestyle="--", linewidth=1.4)

    group_centres = [9, 26.5, 42.5, 57.5, 77]
    group_labels = ["0–17", "18–34", "35–49", "50–64", "65+"]
    label_height = ax.get_ylim()[1] * 0.93
    for centre, label in zip(group_centres, group_labels, strict=True):
        ax.text(centre, label_height, label, ha="center", va="top", weight="bold")

    ax.set(
        title="Binning adds interpretable thresholds but removes detail",
        xlabel="Age (years)",
        ylabel="Observations",
        xlim=(0, 90),
    )
    fig.tight_layout()
    save_figure(fig, "03-age-binning.png")


def plot_feature_set_validation() -> None:
    """Compare fold-level ROC AUC for three candidate feature sets."""
    validation_scores = pd.DataFrame(
        {
            "feature_set": np.repeat(
                [
                    "Baseline",
                    "Baseline + ratios",
                    "Baseline + ratios\nand interactions",
                ],
                5,
            ),
            "roc_auc": [
                0.731,
                0.748,
                0.739,
                0.755,
                0.742,
                0.758,
                0.769,
                0.761,
                0.777,
                0.765,
                0.753,
                0.781,
                0.759,
                0.774,
                0.768,
            ],
        }
    )
    order = [
        "Baseline",
        "Baseline + ratios",
        "Baseline + ratios\nand interactions",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=validation_scores,
        x="feature_set",
        y="roc_auc",
        order=order,
        color="#b8ded9",
        width=0.55,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=validation_scores,
        x="feature_set",
        y="roc_auc",
        order=order,
        color="#024d66",
        size=7,
        jitter=0.08,
        ax=ax,
    )
    ax.set(
        title="Compare engineered features across the same validation folds",
        xlabel="Feature set",
        ylabel="Validation ROC AUC",
        ylim=(0.71, 0.80),
    )
    fig.tight_layout()
    save_figure(fig, "03-feature-set-validation.png")


def main() -> None:
    """Generate all Chapter 03 figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    rng = np.random.default_rng(RANDOM_SEED)

    plot_log_transformation(rng)
    plot_age_binning(rng)
    plot_feature_set_validation()


if __name__ == "__main__":
    main()
