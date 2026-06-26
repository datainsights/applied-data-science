from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def evaluate_once(X: pd.DataFrame, y: pd.Series, random_state: int) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "random_state": random_state,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
    }


def main() -> None:
    input_path = Path("data/diabetes.csv")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            "Missing input file: data/diabetes.csv. "
            "Run scripts/python/09a_save_diabetes_dataset.py first."
        )

    df = pd.read_csv(input_path)

    X = df.drop(columns=["disease_progression"])
    y = df["disease_progression"]

    results = [evaluate_once(X, y, random_state) for random_state in range(20)]

    results_df = pd.DataFrame(results)
    summary_df = results_df[["mae", "r2"]].describe().round(3)

    results_path = reports_dir / "diabetes-repeated-split-metrics.csv"
    summary_path = reports_dir / "diabetes-repeated-split-summary.csv"

    results_df.round(3).to_csv(results_path, index=False)
    summary_df.to_csv(summary_path)

    print(f"Repeated split metrics written to: {results_path}")
    print(f"Repeated split summary written to: {summary_path}")
    print()
    print(summary_df)


if __name__ == "__main__":
    main()
