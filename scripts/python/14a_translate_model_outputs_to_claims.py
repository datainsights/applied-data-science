#!/usr/bin/env python3
"""
Lesson 14: Translate model outputs into defensible real-world claims.

This script fits a baseline linear regression model, saves an observed-versus-predicted
figure, builds a claim review table, and writes a short plain-language summary.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/diabetes.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset not found: {DATA_PATH}. "
            "Run scripts/python/09a_save_diabetes_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["disease_progression"])
    y = df["disease_progression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    error = np.abs(y_test - y_pred)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        y_test,
        y_pred,
        c=error,
        cmap="viridis",
        edgecolors="black",
        linewidth=0.4,
        alpha=0.9,
    )
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)
    plt.xlabel("Observed disease progression")
    plt.ylabel("Predicted disease progression")
    plt.title("Observed vs Predicted Values")
    plt.colorbar(scatter, label="Absolute error")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "diabetes-observed-vs-predicted-error-highlighted.png",
        dpi=300,
    )
    plt.close()

    claim_review_df = pd.DataFrame(
        {
            "claim": [
                "BMI is included as a feature in the model.",
                "BMI contributes to predictions in this fitted model.",
                "Higher BMI is associated with higher predicted disease progression in this model.",
                "BMI causes disease progression.",
                "Reducing BMI will reduce disease progression for a patient.",
            ],
            "claim_strength": [
                "descriptive",
                "model_based",
                "model_based_association",
                "causal",
                "intervention_effect",
            ],
            "supported_by_this_workflow": [
                "yes",
                "yes",
                "yes_with_scope_limits",
                "no",
                "no",
            ],
            "reason": [
                "The feature table contains BMI.",
                "The fitted model uses BMI when forming predictions.",
                "The statement is limited to model behavior and association.",
                "The workflow is predictive, not causal.",
                "The workflow does not estimate intervention effects.",
            ],
        }
    )

    claim_review_df.to_csv(
        REPORTS_DIR / "diabetes-claim-review-table.csv",
        index=False,
    )

    summary_text = """
The fitted linear regression model shows moderate predictive ability for disease progression
using the diabetes dataset. Observed-versus-predicted values indicate that some predictions
are close to the observed outcome, while others show meaningful error. Feature-level patterns
should be interpreted as model-based associations, not causal mechanisms. Claims from this
workflow should therefore focus on prediction, model behavior, and association within the
current dataset and modeling setup.
""".strip()

    (REPORTS_DIR / "diabetes-model-claim-summary.txt").write_text(summary_text)

    print("Saved claim review outputs:")
    print(f"- {REPORTS_DIR / 'diabetes-claim-review-table.csv'}")
    print(f"- {REPORTS_DIR / 'diabetes-model-claim-summary.txt'}")
    print(f"- {FIGURES_DIR / 'diabetes-observed-vs-predicted-error-highlighted.png'}")


if __name__ == "__main__":
    main()
