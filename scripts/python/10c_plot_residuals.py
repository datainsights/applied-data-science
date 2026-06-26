from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    input_path = Path("data/diabetes.csv")
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            "Missing input file: data/diabetes.csv. "
            "Run scripts/python/09a_save_diabetes_dataset.py first."
        )

    df = pd.read_csv(input_path)

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
    residuals = y_test.to_numpy() - y_pred

    residuals_df = pd.DataFrame(
        {
            "observed": y_test.to_numpy(),
            "predicted": y_pred,
            "residual": residuals,
            "absolute_residual": np.abs(residuals),
        }
    )

    residuals_path = reports_dir / "diabetes-residuals.csv"
    residuals_df.to_csv(residuals_path, index=False)

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(
        residuals_df["predicted"],
        residuals_df["residual"],
        c=residuals_df["absolute_residual"],
        cmap="viridis",
        edgecolors="black",
        linewidth=0.4,
        alpha=0.9,
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.colorbar(scatter, label="Absolute Error")
    plt.grid(alpha=0.2)

    figure_path = figures_dir / "diabetes-residual-plot.png"
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"Residual values written to: {residuals_path}")
    print(f"Residual plot written to: {figure_path}")


if __name__ == "__main__":
    main()
