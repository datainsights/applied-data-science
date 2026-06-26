#!/usr/bin/env python3
"""Run pipeline-based cross-validation for the diabetes example.

Inputs
------
data/diabetes.csv

Outputs
-------
reports/diabetes-pipeline-cross-validation-folds.csv
reports/diabetes-pipeline-cross-validation-summary.csv
reports/figures/diabetes-cross-validation-mae.png
reports/figures/diabetes-cross-validation-r2.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/diabetes.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {DATA_PATH}. "
            "Run scripts/python/09a_save_diabetes_dataset.py first."
        )

    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if "disease_progression" not in df.columns:
        raise ValueError("Expected target column 'disease_progression' in data/diabetes.csv")

    X = df.drop(columns=["disease_progression"])
    y = df["disease_progression"]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    mae_scores = -cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error",
    )

    r2_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="r2",
    )

    results_df = pd.DataFrame(
        {
            "fold": range(1, len(mae_scores) + 1),
            "mae": mae_scores,
            "r2": r2_scores,
        }
    )

    summary_df = pd.DataFrame(
        {
            "metric": ["mae", "r2"],
            "mean": [np.mean(mae_scores), np.mean(r2_scores)],
            "std": [np.std(mae_scores), np.std(r2_scores)],
        }
    )

    results_path = REPORTS_DIR / "diabetes-pipeline-cross-validation-folds.csv"
    summary_path = REPORTS_DIR / "diabetes-pipeline-cross-validation-summary.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    folds = results_df["fold"]

    plt.figure(figsize=(7, 5))
    plt.plot(folds, results_df["mae"], marker="o")
    plt.xlabel("Fold")
    plt.ylabel("MAE")
    plt.title("MAE Across Cross-Validation Folds")
    plt.grid(alpha=0.2)
    plt.savefig(FIGURES_DIR / "diabetes-cross-validation-mae.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(folds, results_df["r2"], marker="o")
    plt.xlabel("Fold")
    plt.ylabel("R²")
    plt.title("R² Across Cross-Validation Folds")
    plt.grid(alpha=0.2)
    plt.savefig(FIGURES_DIR / "diabetes-cross-validation-r2.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Cross-validation complete.")
    print(f"Fold results: {results_path}")
    print(f"Summary:      {summary_path}")
    print(f"Figures:      {FIGURES_DIR}")
    print("\nSummary:")
    print(summary_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
