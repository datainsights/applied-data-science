from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def main() -> None:
    input_path = Path("data/diabetes.csv")
    models_dir = Path("models")
    reports_dir = Path("reports")

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

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

    metrics = pd.DataFrame(
        {
            "metric": ["MAE", "R-squared"],
            "value": [
                mean_absolute_error(y_test, y_pred),
                r2_score(y_test, y_pred),
            ],
        }
    ).round(3)

    predictions = pd.DataFrame(
        {
            "observed": y_test.to_numpy(),
            "predicted": y_pred,
        }
    )
    predictions["absolute_error"] = (
        predictions["observed"] - predictions["predicted"]
    ).abs()

    coefficients = pd.DataFrame(
        {
            "feature": X_train.columns,
            "coefficient": model.coef_,
        }
    ).sort_values("coefficient", ascending=False)

    model_path = models_dir / "diabetes-linear-regression.joblib"
    metrics_path = reports_dir / "diabetes-model-metrics.csv"
    predictions_path = reports_dir / "diabetes-observed-vs-predicted.csv"
    coefficients_path = reports_dir / "diabetes-model-coefficients.csv"

    joblib.dump(model, model_path)
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)

    print(f"Model written to: {model_path}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Predictions written to: {predictions_path}")
    print(f"Coefficients written to: {coefficients_path}")
    print()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
