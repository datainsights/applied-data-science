#!/usr/bin/env python3
"""Generate the unsupervised-learning figures used in ADS Chapter 11."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
OUTPUT_DIR = Path("results/figures")


def save_figure(filename: str) -> None:
    """Save the current figure using consistent output settings."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def generate_blob_data() -> tuple[np.ndarray, np.ndarray]:
    """Return standardized multivariate data with known teaching groups."""
    X, hidden_groups = make_blobs(
        n_samples=240,
        n_features=6,
        centers=3,
        cluster_std=(1.0, 1.35, 0.85),
        random_state=RANDOM_STATE,
    )
    rng = np.random.default_rng(RANDOM_STATE)
    scales = np.array([1.0, 4.0, 0.6, 2.5, 8.0, 1.8])
    X = X * scales + rng.normal(0, 0.15, size=X.shape)
    return StandardScaler().fit_transform(X), hidden_groups


def plot_pca_overview(X: np.ndarray, hidden_groups: np.ndarray) -> None:
    """Plot PCA scores and cumulative explained variance."""
    pca = PCA().fit(X)
    scores = pca.transform(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sns.scatterplot(
        x=scores[:, 0],
        y=scores[:, 1],
        hue=hidden_groups.astype(str),
        palette="colorblind",
        s=45,
        alpha=0.8,
        ax=axes[0],
    )
    axes[0].set(
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        title="PCA score plot",
    )
    axes[0].legend(title="Simulation group", frameon=False)

    component_numbers = np.arange(1, len(cumulative) + 1)
    sns.lineplot(x=component_numbers, y=cumulative, marker="o", ax=axes[1])
    axes[1].axhline(0.90, color="0.4", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Number of components",
        ylabel="Cumulative explained variance",
        title="Variance retained",
        ylim=(0, 1.03),
        xticks=component_numbers,
    )
    fig.tight_layout()
    save_figure("11-pca-overview.png")


def plot_kmeans_diagnostics(X: np.ndarray) -> None:
    """Plot inertia and silhouette coefficient over candidate cluster counts."""
    candidates = range(2, 9)
    inertias: list[float] = []
    silhouettes: list[float] = []

    for number_of_clusters in candidates:
        model = KMeans(
            n_clusters=number_of_clusters,
            n_init=20,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.lineplot(x=list(candidates), y=inertias, marker="o", ax=axes[0])
    axes[0].set(xlabel="Number of clusters (K)", ylabel="Inertia", title="Within-cluster variation")
    sns.lineplot(x=list(candidates), y=silhouettes, marker="o", ax=axes[1])
    axes[1].set(
        xlabel="Number of clusters (K)",
        ylabel="Mean silhouette coefficient",
        title="Compactness and separation",
    )
    fig.tight_layout()
    save_figure("11-kmeans-diagnostics.png")


def plot_hierarchical_dendrogram(X: np.ndarray) -> None:
    """Plot a readable truncated Ward-linkage dendrogram."""
    linkage_matrix = linkage(X, method="ward", metric="euclidean")
    plt.figure(figsize=(11, 5.2))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=45,
        leaf_font_size=8,
        show_contracted=True,
    )
    plt.axhline(18, color="0.4", linestyle="--", linewidth=1, label="Illustrative cut")
    plt.xlabel("Merged observations or subclusters")
    plt.ylabel("Ward distance")
    plt.title("Hierarchical clustering structure")
    plt.legend(frameon=False)
    plt.tight_layout()
    save_figure("11-hierarchical-dendrogram.png")


def plot_method_comparison() -> None:
    """Compare K-means and DBSCAN on non-spherical data."""
    X, _ = make_moons(n_samples=350, noise=0.075, random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)
    kmeans_labels = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
    dbscan_labels = DBSCAN(eps=0.28, min_samples=8).fit_predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, labels, title in zip(
        axes,
        (kmeans_labels, dbscan_labels),
        ("K-means: centroid geometry", "DBSCAN: density connectivity"),
        strict=True,
    ):
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=labels.astype(str),
            palette="colorblind",
            s=34,
            alpha=0.85,
            legend=False,
            ax=ax,
        )
        ax.set(xlabel="Standardized feature 1", ylabel="Standardized feature 2", title=title)
    fig.tight_layout()
    save_figure("11-method-comparison.png")


def main() -> None:
    """Generate every Chapter 11 figure."""
    sns.set_theme(style="whitegrid", context="notebook")
    X, hidden_groups = generate_blob_data()
    plot_pca_overview(X, hidden_groups)
    plot_kmeans_diagnostics(X)
    plot_hierarchical_dendrogram(X)
    plot_method_comparison()
    print(f"Figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
