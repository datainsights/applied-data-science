#!/usr/bin/env python3
"""Generate the decision-making figure used in ADS Chapter 16."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
OUTPUT_DIR = Path("results/figures")
FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 5
CAPACITY_SHARE = 0.20


def build_validation_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Fit an illustrative model and return validation outcomes and risks."""
    features, outcome = make_classification(
        n_samples=3000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.78, 0.22],
        class_sep=1.0,
        flip_y=0.03,
        random_state=RANDOM_STATE,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        outcome,
        test_size=0.40,
        stratify=outcome,
        random_state=RANDOM_STATE,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(x_train, y_train)
    predicted_risk = model.predict_proba(x_valid)[:, 1]
    return y_valid, predicted_risk


def evaluate_thresholds(
    outcome: np.ndarray, predicted_risk: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate thresholds, average cost, and selection share."""
    thresholds = np.linspace(0.01, 0.80, 160)
    average_cost = []
    selected_share = []

    for threshold in thresholds:
        selected = predicted_risk >= threshold
        false_positives = np.sum(selected & (outcome == 0))
        false_negatives = np.sum(~selected & (outcome == 1))
        cost = (
            FALSE_POSITIVE_COST * false_positives
            + FALSE_NEGATIVE_COST * false_negatives
        ) / len(outcome)
        average_cost.append(cost)
        selected_share.append(np.mean(selected))

    return thresholds, np.asarray(average_cost), np.asarray(selected_share)


def create_figure() -> None:
    """Create and save the threshold trade-off figure."""
    outcome, predicted_risk = build_validation_predictions()
    thresholds, average_cost, selected_share = evaluate_thresholds(
        outcome, predicted_risk
    )

    cost_index = int(np.argmin(average_cost))
    capacity_index = int(np.argmin(np.abs(selected_share - CAPACITY_SHARE)))
    cost_threshold = thresholds[cost_index]
    capacity_threshold = thresholds[capacity_index]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].plot(thresholds, average_cost, color="#176B87", linewidth=2.5)
    axes[0].scatter(
        cost_threshold,
        average_cost[cost_index],
        color="#C8553D",
        s=60,
        zorder=3,
        label=f"Lowest cost: {cost_threshold:.2f}",
    )
    axes[0].set(
        title="Specified validation cost",
        xlabel="Decision threshold",
        ylabel="Average cost per participant",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(thresholds, selected_share, color="#176B87", linewidth=2.5)
    axes[1].axhline(
        CAPACITY_SHARE,
        color="#6C757D",
        linestyle="--",
        linewidth=1.5,
        label=f"Capacity: {CAPACITY_SHARE:.0%}",
    )
    axes[1].scatter(
        capacity_threshold,
        selected_share[capacity_index],
        color="#C8553D",
        s=60,
        zorder=3,
        label=f"Capacity threshold: {capacity_threshold:.2f}",
    )
    axes[1].set(
        title="Operational selection rate",
        xlabel="Decision threshold",
        ylabel="Share selected for action",
        ylim=(0, 1),
    )
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Decision thresholds reflect consequences and constraints",
        fontsize=14,
        fontweight="bold",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "16-decision-threshold-trade-offs.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Cost-minimising threshold: {cost_threshold:.3f}")
    print(f"Capacity-based threshold: {capacity_threshold:.3f}")
    print(f"Figure written to {output_path}")


if __name__ == "__main__":
    create_figure()
