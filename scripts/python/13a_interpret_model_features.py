"""Interpret feature influence for the diabetes modeling example.

This script fits two simple models and saves interpretation artifacts:

- Linear regression coefficients
- Decision tree feature importance
- Coefficient plot
- Feature importance plot

Run from the project root:

    python scripts/python/13a_interpret_model_features.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


DATA_PATH = Path("data/diabetes.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

COEFFICIENTS_PATH = REPORTS_DIR / "diabetes-linear-model-coefficients.csv"
IMPORTANCE_PATH = REPORTS_DIR / "diabetes-tree-feature-importance.csv"
COEFFICIENTS_FIGURE_PATH = FIGURES_DIR / "diabetes-linear-model-coefficients.png"
IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "diabetes-tree-feature-importance.png"


def main() -> None:
    """Fit interpretation models and save feature influence outputs."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run the earlier diabetes data script first."
        )

    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["disease_progression"])
    y = df["disease_progression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    coefficients_df = pd.DataFrame(
        {
            "feature": X.columns,
            "coefficient": linear_model.coef_,
        }
    )
    coefficients_df["absolute_coefficient"] = coefficients_df["coefficient"].abs()
    coefficients_df = coefficients_df.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    coefficients_df.to_csv(COEFFICIENTS_PATH, index=False)

    coefficient_plot_df = coefficients_df.sort_values("coefficient")

    plt.figure(figsize=(8, 5))
    plt.barh(coefficient_plot_df["feature"], coefficient_plot_df["coefficient"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Coefficient value")
    plt.ylabel("Feature")
    plt.title("Linear Model Coefficients")
    plt.tight_layout()
    plt.savefig(COEFFICIENTS_FIGURE_PATH, dpi=300)
    plt.close()

    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": tree_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    importance_plot_df = importance_df.sort_values("importance")

    plt.figure(figsize=(8, 5))
    plt.barh(importance_plot_df["feature"], importance_plot_df["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Decision Tree Feature Importance")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_FIGURE_PATH, dpi=300)
    plt.close()

    print("Feature interpretation complete.")
    print(f"Saved: {COEFFICIENTS_PATH}")
    print(f"Saved: {IMPORTANCE_PATH}")
    print(f"Saved: {COEFFICIENTS_FIGURE_PATH}")
    print(f"Saved: {IMPORTANCE_FIGURE_PATH}")


if __name__ == "__main__":
    main()
