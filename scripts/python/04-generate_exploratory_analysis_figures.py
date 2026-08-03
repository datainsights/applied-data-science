"""Generate the reproducible figures for ADS Chapter 04."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
DATA_PATH = Path("data/processed/04-exploratory_data.csv")
FIGURE_DIR = Path("results/figures")


def build_example_data(n_participants: int = 360) -> pd.DataFrame:
    """Create a synthetic dataset with realistic exploratory structure."""
    rng = np.random.default_rng(RANDOM_SEED)

    site = rng.choice(["North", "Central", "South"], n_participants, p=[0.32, 0.40, 0.28])
    treatment = rng.choice(["Control", "Intervention"], n_participants)
    age = np.clip(rng.normal(49, 14, n_participants), 18, 85)
    site_effect = pd.Series(site).map({"North": 0.7, "Central": 0.0, "South": -0.5}).to_numpy()
    treatment_effect = (treatment == "Intervention").astype(float)

    biomarker_a = rng.lognormal(1.25 + 0.10 * treatment_effect + 0.08 * site_effect, 0.42)
    biomarker_b = 0.72 * biomarker_a + 0.025 * age + rng.normal(0, 1.0, n_participants)
    biomarker_c = rng.normal(5.2 + 0.35 * site_effect, 1.15, n_participants)
    response = (
        42
        + 2.4 * biomarker_a
        - 0.20 * age
        + 3.2 * treatment_effect
        + 2.0 * site_effect
        + rng.normal(0, 5.5, n_participants)
    )

    data = pd.DataFrame(
        {
            "participant_id": [f"P{i:04d}" for i in range(1, n_participants + 1)],
            "site": site,
            "treatment_group": treatment,
            "age": age,
            "biomarker_a": biomarker_a,
            "biomarker_b": biomarker_b,
            "biomarker_c": biomarker_c,
            "response_score": response,
        }
    )

    missing_a = rng.random(n_participants) < np.where(site == "South", 0.18, 0.05)
    missing_b = rng.random(n_participants) < np.where(site == "North", 0.14, 0.04)
    missing_response = rng.random(n_participants) < 0.035
    data.loc[missing_a, "biomarker_a"] = np.nan
    data.loc[missing_b, "biomarker_b"] = np.nan
    data.loc[missing_response, "response_score"] = np.nan

    return data


def save_figure(filename: str) -> None:
    """Apply shared layout settings and save the current figure."""
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_missingness(data: pd.DataFrame) -> None:
    """Plot overall and site-specific missingness."""
    measured = ["age", "biomarker_a", "biomarker_b", "biomarker_c", "response_score"]
    overall = data[measured].isna().mean().mul(100).sort_values(ascending=False)
    by_site = data.groupby("site", observed=True)[measured].apply(lambda x: x.isna().mean() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [0.9, 1.4]})
    sns.barplot(x=overall.values, y=overall.index, color="#2A9D8F", ax=axes[0])
    axes[0].set(title="Overall missingness", xlabel="Missing values (%)", ylabel="")
    sns.heatmap(by_site, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "% missing"}, ax=axes[1])
    axes[1].set(title="Missingness varies across sites", xlabel="Variable", ylabel="Site")
    axes[1].tick_params(axis="x", rotation=35)
    save_figure("04-missingness-patterns.png")


def plot_distributions(data: pd.DataFrame) -> None:
    """Compare histogram and ECDF views of a biomarker."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.histplot(data=data, x="biomarker_a", hue="treatment_group", element="step", stat="density", common_norm=False, ax=axes[0])
    axes[0].set(title="Distribution shape", xlabel="Biomarker A", ylabel="Density")
    sns.ecdfplot(data=data, x="biomarker_a", hue="treatment_group", ax=axes[1])
    axes[1].set(title="Full-distribution comparison", xlabel="Biomarker A", ylabel="Cumulative proportion")
    save_figure("04-group-distributions.png")


def plot_relationships(data: pd.DataFrame) -> None:
    """Combine a grouped relationship plot with rank correlations."""
    features = ["age", "biomarker_a", "biomarker_b", "biomarker_c", "response_score"]
    correlations = data[features].corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
    sns.scatterplot(data=data, x="biomarker_a", y="response_score", hue="site", style="treatment_group", alpha=0.75, ax=axes[0])
    axes[0].set(title="Relationship in group context", xlabel="Biomarker A", ylabel="Response score")
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, square=True, ax=axes[1])
    axes[1].set_title("Spearman rank correlations")
    axes[1].tick_params(axis="x", rotation=35)
    save_figure("04-feature-relationships.png")


def plot_pca(data: pd.DataFrame) -> None:
    """Plot the first two principal components of numeric features."""
    features = ["age", "biomarker_a", "biomarker_b", "biomarker_c", "response_score"]
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=2))
    scores = pipeline.fit_transform(data[features])
    pca = pipeline.named_steps["pca"]
    plot_data = data[["site", "treatment_group"]].copy()
    plot_data["PC1"] = scores[:, 0]
    plot_data["PC2"] = scores[:, 1]

    plt.figure(figsize=(8.5, 6.2))
    sns.scatterplot(data=plot_data, x="PC1", y="PC2", hue="site", style="treatment_group", alpha=0.78, s=55)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("Multivariate sample structure")
    plt.axhline(0, color="0.75", linewidth=0.8)
    plt.axvline(0, color="0.75", linewidth=0.8)
    save_figure("04-pca-structure.png")


def main() -> None:
    """Create data and all Chapter 04 figures."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")

    data = build_example_data()
    data.to_csv(DATA_PATH, index=False)
    plot_missingness(data)
    plot_distributions(data)
    plot_relationships(data)
    plot_pca(data)

    print(f"Saved illustrative data to {DATA_PATH}")
    print(f"Saved four figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()

