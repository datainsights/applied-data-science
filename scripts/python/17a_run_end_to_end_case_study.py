#!/usr/bin/env python3
"""Run the end-to-end Applied Data Science case study.

This script reproduces the core workflow from Lesson 17:
- load the diabetes dataset
- build a preprocessing + modeling pipeline
- evaluate with cross-validation
- fit one train/test model for interpretation
- save metrics, coefficients, figures, and a case summary
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/diabetes.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_directories() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {DATA_PATH}. Run Lesson 09 setup first."
        )

    df = pd.read_csv(DATA_PATH)

    if "disease_progression" not in df.columns:
        raise ValueError("Expected target column 'disease_progression' was not found.")

    X = df.drop(columns=["disease_progression"])
    y = df["disease_progression"]
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def save_cross_validation_outputs(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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

    folds_df = pd.DataFrame({
        "fold": range(1, len(mae_scores) + 1),
        "mae": mae_scores,
        "r2": r2_scores,
    })

    summary_df = pd.DataFrame({
        "metric": ["mae", "r2"],
        "mean": [float(np.mean(mae_scores)), float(np.mean(r2_scores))],
        "std": [float(np.std(mae_scores)), float(np.std(r2_scores))],
        "min": [float(np.min(mae_scores)), float(np.min(r2_scores))],
        "max": [float(np.max(mae_scores)), float(np.max(r2_scores))],
    })

    folds_df.to_csv(
        REPORTS_DIR / "diabetes-end-to-end-cross-validation-folds.csv",
        index=False,
    )
    summary_df.to_csv(
        REPORTS_DIR / "diabetes-end-to-end-cross-validation-summary.csv",
        index=False,
    )

    return mae_scores, r2_scores, summary_df


def plot_cross_validation_scores(mae_scores: np.ndarray, r2_scores: np.ndarray) -> None:
    folds = range(1, len(mae_scores) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(folds, mae_scores, marker="o", linewidth=1)
    plt.xlabel("Fold")
    plt.ylabel("MAE")
    plt.title("End-to-End Case Study: MAE Across Folds")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "diabetes-end-to-end-cv-mae.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(folds, r2_scores, marker="o", linewidth=1)
    plt.xlabel("Fold")
    plt.ylabel("R²")
    plt.title("End-to-End Case Study: R² Across Folds")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "diabetes-end-to-end-cv-r2.png", dpi=300)
    plt.close()


def fit_interpretation_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[Pipeline, pd.Series, pd.Series, np.ndarray, pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    test_metrics_df = pd.DataFrame({
        "metric": ["mae", "r2"],
        "value": [
            mean_absolute_error(y_test, y_pred),
            r2_score(y_test, y_pred),
        ],
    })

    test_metrics_df.to_csv(
        REPORTS_DIR / "diabetes-end-to-end-test-metrics.csv",
        index=False,
    )

    return pipeline, y_test, pd.Series(y_pred, index=y_test.index), y_pred, test_metrics_df


def save_prediction_plot(y_test: pd.Series, y_pred: pd.Series) -> None:
    error = np.abs(y_test - y_pred)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        y_test,
        y_pred,
        c=error,
        edgecolors="black",
        linewidth=0.4,
        alpha=0.9,
    )

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)

    plt.xlabel("Observed progression")
    plt.ylabel("Predicted progression")
    plt.title("End-to-End Case Study: Observed vs Predicted")
    plt.colorbar(scatter, label="Absolute Error")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "diabetes-end-to-end-observed-vs-predicted.png", dpi=300)
    plt.close()


def save_coefficients(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": pipeline.named_steps["model"].coef_,
    }).sort_values("coefficient")

    coefficients.to_csv(
        REPORTS_DIR / "diabetes-end-to-end-coefficients.csv",
        index=False,
    )

    plt.figure(figsize=(7, 5))
    plt.barh(coefficients["feature"], coefficients["coefficient"])
    plt.xlabel("Coefficient")
    plt.title("End-to-End Case Study: Linear Model Coefficients")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "diabetes-end-to-end-coefficients.png", dpi=300)
    plt.close()

    return coefficients


def save_case_summary(
    cv_summary: pd.DataFrame,
    test_metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    cv_mae = cv_summary.loc[cv_summary["metric"] == "mae", "mean"].iloc[0]
    cv_r2 = cv_summary.loc[cv_summary["metric"] == "r2", "mean"].iloc[0]
    test_mae = test_metrics.loc[test_metrics["metric"] == "mae", "value"].iloc[0]
    test_r2 = test_metrics.loc[test_metrics["metric"] == "r2", "value"].iloc[0]

    strongest_positive = coefficients.sort_values("coefficient", ascending=False).iloc[0]
    strongest_negative = coefficients.sort_values("coefficient", ascending=True).iloc[0]

    summary = f"""# Diabetes End-to-End Case Study Summary

## Objective

Assess whether baseline clinical measurements can support useful prediction of disease progression.

## Workflow

The workflow used a reproducible pipeline with standard scaling and linear regression.

## Cross-validation performance

- Mean MAE: {cv_mae:.3f}
- Mean R²: {cv_r2:.3f}

## Test-set performance

- Test MAE: {test_mae:.3f}
- Test R²: {test_r2:.3f}

## Model interpretation

The strongest positive coefficient was `{strongest_positive['feature']}`.
The strongest negative coefficient was `{strongest_negative['feature']}`.

These coefficients describe model behavior, not causation.

## Defensible claim

The model shows moderate predictive ability, and several features contribute meaningfully to predicted disease progression in this dataset.

## Decision context

The model may support monitoring or prioritization, but it should not be used as the sole basis for treatment or intervention decisions.
"""

    (REPORTS_DIR / "diabetes-end-to-end-case-summary.md").write_text(summary)


def main() -> None:
    ensure_directories()
    X, y = load_data()

    pipeline = build_pipeline()
    mae_scores, r2_scores, cv_summary = save_cross_validation_outputs(pipeline, X, y)
    plot_cross_validation_scores(mae_scores, r2_scores)

    fitted_pipeline, y_test, y_pred_series, _, test_metrics = fit_interpretation_model(
        build_pipeline(),
        X,
        y,
    )
    save_prediction_plot(y_test, y_pred_series)
    coefficients = save_coefficients(fitted_pipeline, list(X.columns))
    save_case_summary(cv_summary, test_metrics, coefficients)

    print("End-to-end case study complete.")
    print(f"Outputs saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
