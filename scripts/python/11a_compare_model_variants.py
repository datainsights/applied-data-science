#!/usr/bin/env python3
"""Compare controlled model variants for the diabetes regression example."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    """Fit a model and return basic evaluation metrics."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": name,
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def main():
    data_path = Path("data/diabetes.csv")
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    x = df.drop(columns=["disease_progression"])
    y = df["disease_progression"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model_variants = [
        (
            "linear_regression_baseline",
            LinearRegression(),
        ),
        (
            "scaled_linear_regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]),
        ),
        (
            "ridge_regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]),
        ),
        (
            "decision_tree",
            DecisionTreeRegressor(random_state=42),
        ),
    ]

    results = [
        evaluate_model(name, model, x_train, x_test, y_train, y_test)
        for name, model in model_variants
    ]

    comparison_df = (
        pd.DataFrame(results)
        .sort_values("mae")
        .reset_index(drop=True)
    )

    output_path = output_dir / "diabetes-model-comparison.csv"
    comparison_df.to_csv(output_path, index=False)

    print("Saved model comparison:", output_path)
    print(comparison_df.round(3))


if __name__ == "__main__":
    main()
