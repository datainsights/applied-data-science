"""Generate the illustrative figures used in ADS Chapter 06.

Run from the project root:
    python scripts/python/06-generate_hypothesis_testing_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


SEED = 20260801
OUTPUT_DIR = Path("results/figures")
COLORS = {"A": "#036281", "B": "#e07a5f"}


def save_figure(filename: str) -> None:
    """Save the current figure using consistent publication settings."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def make_example_data() -> pd.DataFrame:
    """Create deterministic teaching data for two independent groups."""
    rng = np.random.default_rng(SEED)
    group_a = rng.normal(loc=104.0, scale=8.0, size=42)
    group_b = rng.normal(loc=99.0, scale=10.5, size=38)
    return pd.DataFrame(
        {
            "group": np.repeat(["A", "B"], [len(group_a), len(group_b)]),
            "outcome": np.concatenate([group_a, group_b]),
        }
    )


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Return the sample mean and its 95% t confidence interval."""
    mean = float(np.mean(values))
    sem = stats.sem(values)
    low, high = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
    return mean, float(low), float(high)


def plot_group_estimates(data: pd.DataFrame) -> None:
    """Show raw observations together with group means and 95% intervals."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    sns.stripplot(
        data=data,
        x="group",
        y="outcome",
        hue="group",
        palette=COLORS,
        jitter=0.22,
        alpha=0.55,
        size=5,
        legend=False,
        ax=ax,
    )

    for position, group in enumerate(["A", "B"]):
        values = data.loc[data["group"] == group, "outcome"].to_numpy()
        mean, low, high = mean_ci(values)
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            color="#172b3a",
            markerfacecolor="white",
            markersize=8,
            capsize=6,
            linewidth=2.2,
            zorder=5,
        )

    ax.set(title="Observed outcomes with mean and 95% confidence interval")
    ax.set_xlabel("Group")
    ax.set_ylabel("Outcome")
    sns.despine()
    fig.tight_layout()
    save_figure("06-group-estimates.png")


def plot_permutation_distribution(data: pd.DataFrame) -> None:
    """Display the permutation null distribution and observed difference."""
    rng = np.random.default_rng(SEED)
    a = data.loc[data["group"] == "A", "outcome"].to_numpy()
    b = data.loc[data["group"] == "B", "outcome"].to_numpy()
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    differences = np.empty(10_000)

    for index in range(len(differences)):
        shuffled = rng.permutation(pooled)
        differences[index] = shuffled[: len(a)].mean() - shuffled[len(a) :].mean()

    p_value = (np.count_nonzero(np.abs(differences) >= abs(observed)) + 1) / (
        len(differences) + 1
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.histplot(differences, bins=45, color="#86b8c6", edgecolor="white", ax=ax)
    ax.axvline(observed, color="#c43c39", linewidth=2.5, label="Observed difference")
    ax.axvline(-observed, color="#c43c39", linewidth=1.6, linestyle="--")
    ax.set(
        title="Permutation distribution under the null hypothesis",
        xlabel="Difference in means: A − B",
        ylabel="Permutation count",
    )
    ax.text(
        0.98,
        0.94,
        f"Two-sided p = {p_value:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#172b3a",
    )
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    save_figure("06-permutation-null-distribution.png")


def plot_interval_interpretation() -> None:
    """Contrast precise, imprecise, and negligible interval estimates."""
    labels = ["Clear positive effect", "Imprecise estimate", "Small precise effect"]
    estimates = np.array([4.2, 2.0, 0.7])
    lower = np.array([1.1, -3.8, 0.2])
    upper = np.array([7.3, 7.8, 1.2])
    positions = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axvline(0, color="#71808a", linestyle="--", linewidth=1.4)
    ax.errorbar(
        estimates,
        positions,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o",
        color="#036281",
        ecolor="#036281",
        capsize=5,
        markersize=7,
        linewidth=2.2,
    )
    ax.set_yticks(positions, labels)
    ax.set(
        title="Confidence intervals reveal magnitude and precision",
        xlabel="Estimated difference with 95% confidence interval",
        ylabel="",
    )
    sns.despine(left=True)
    fig.tight_layout()
    save_figure("06-confidence-interval-interpretation.png")


def plot_multiple_testing() -> None:
    """Compare raw p-values with Holm and Benjamini–Hochberg adjustments."""
    raw = np.array([0.001, 0.004, 0.011, 0.018, 0.031, 0.046, 0.072, 0.13, 0.28, 0.61])
    order = np.argsort(raw)
    ordered = raw[order]
    count = len(raw)

    holm_ordered = np.maximum.accumulate((count - np.arange(count)) * ordered)
    bh_ordered = np.minimum.accumulate(
        (ordered * count / np.arange(1, count + 1))[::-1]
    )[::-1]

    holm = np.empty_like(raw)
    bh = np.empty_like(raw)
    holm[order] = np.clip(holm_ordered, 0, 1)
    bh[order] = np.clip(bh_ordered, 0, 1)
    table = pd.DataFrame(
        {
            "Hypothesis": np.arange(1, len(raw) + 1),
            "Raw p-value": raw,
            "Holm adjusted": holm,
            "BH adjusted": bh,
        }
    ).melt(id_vars="Hypothesis", var_name="Method", value_name="Value")

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    sns.lineplot(
        data=table,
        x="Hypothesis",
        y="Value",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        palette=["#71808a", "#c43c39", "#036281"],
        ax=ax,
    )
    ax.axhline(0.05, color="#172b3a", linestyle="--", linewidth=1.2, label="0.05 threshold")
    ax.set(
        title="Multiplicity adjustment changes the evidence threshold",
        xlabel="Hypothesis ordered by raw p-value",
        ylabel="p-value or adjusted p-value",
        xticks=np.arange(1, 11),
        ylim=(-0.01, 0.68),
    )
    ax.legend(frameon=False, title="")
    sns.despine()
    fig.tight_layout()
    save_figure("06-multiple-testing-adjustment.png")


def main() -> None:
    """Generate all Chapter 06 figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
        }
    )
    data = make_example_data()
    plot_group_estimates(data)
    plot_permutation_distribution(data)
    plot_interval_interpretation()
    plot_multiple_testing()
    print(f"Created four Chapter 06 figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
