#!/usr/bin/env python3
"""Generate the statistical-inference figure used in ADS Chapter 05."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


SEED = 20260801
OUTPUT_PATH = Path("results/figures/05-statistical-inference-overview.png")


def simulate_sample_means(
    rng: np.random.Generator,
    sample_size: int,
    repetitions: int = 5_000,
) -> np.ndarray:
    """Simulate means from a right-skewed population."""
    samples = rng.lognormal(
        mean=2.2,
        sigma=0.55,
        size=(repetitions, sample_size),
    )
    return samples.mean(axis=1)


def welch_interval(
    group_1: np.ndarray,
    group_0: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Return a mean difference and its Welch confidence interval."""
    difference = group_1.mean() - group_0.mean()
    variance_1 = group_1.var(ddof=1)
    variance_0 = group_0.var(ddof=1)
    term_1 = variance_1 / len(group_1)
    term_0 = variance_0 / len(group_0)
    standard_error = np.sqrt(term_1 + term_0)
    degrees_of_freedom = (term_1 + term_0) ** 2 / (
        term_1**2 / (len(group_1) - 1)
        + term_0**2 / (len(group_0) - 1)
    )
    critical_value = stats.t.ppf(
        1 - (1 - confidence_level) / 2,
        df=degrees_of_freedom,
    )
    margin = critical_value * standard_error
    return difference, difference - margin, difference + margin


def main() -> None:
    """Create and save the two-panel chapter figure."""
    rng = np.random.default_rng(SEED)
    means_n10 = simulate_sample_means(rng, sample_size=10)
    means_n50 = simulate_sample_means(rng, sample_size=50)

    control = np.array([71, 68, 75, 73, 69, 74, 72, 70, 76, 67, 73, 71])
    treatment = np.array([66, 64, 70, 68, 63, 69, 65, 67, 71, 64, 66, 68])
    estimate, lower, upper = welch_interval(treatment, control)

    sns.set_theme(style="whitegrid", context="talk")
    colors = {"n = 10": "#D55E00", "n = 50": "#0072B2"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    sns.kdeplot(
        x=means_n10,
        fill=True,
        alpha=0.28,
        linewidth=2,
        color=colors["n = 10"],
        label="n = 10",
        ax=axes[0],
    )
    sns.kdeplot(
        x=means_n50,
        fill=True,
        alpha=0.28,
        linewidth=2,
        color=colors["n = 50"],
        label="n = 50",
        ax=axes[0],
    )
    axes[0].set(
        title="Sampling distributions of the mean",
        xlabel="Sample mean",
        ylabel="Density",
    )
    axes[0].legend(title="Sample size", frameon=True)

    axes[1].axvline(0, color="#59636E", linestyle="--", linewidth=1.5)
    axes[1].errorbar(
        estimate,
        0,
        xerr=[[estimate - lower], [upper - estimate]],
        fmt="o",
        markersize=10,
        capsize=7,
        linewidth=2.5,
        color="#009E73",
    )
    axes[1].set(
        title="Estimated mean difference",
        xlabel="Treatment − control (95% CI)",
        yticks=[],
        ylim=(-0.7, 0.7),
    )
    axes[1].annotate(
        f"{estimate:.1f} [{lower:.1f}, {upper:.1f}]",
        xy=(estimate, 0),
        xytext=(0, 34),
        textcoords="offset points",
        ha="center",
        fontsize=12,
    )
    sns.despine(ax=axes[0])
    sns.despine(ax=axes[1], left=True)

    figure.suptitle(
        "Uncertainty depends on sampling variability and sample size",
        fontsize=18,
        fontweight="bold",
    )
    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
