"""Build communication-ready summaries for the diabetes model.

This script supports Lesson 15 of the Applied Data Science System.
It translates saved model outputs into reusable communication artifacts.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

METRICS_PATH = REPORTS_DIR / "diabetes-model-metrics.csv"
COEFFICIENTS_PATH = REPORTS_DIR / "diabetes-model-coefficients.csv"
COMMUNICATION_TABLE_PATH = REPORTS_DIR / "diabetes-results-communication-table.csv"
COMMUNICATION_SUMMARY_PATH = REPORTS_DIR / "diabetes-communication-summary.md"


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV file if it exists."""
    if path.exists():
        return pd.read_csv(path)
    return None


def build_communication_table() -> pd.DataFrame:
    """Create a table translating technical outputs into careful language."""
    rows = [
        {
            "technical_output": "R2 score",
            "plain_language": "How much outcome variation the model explains",
            "careful_statement": (
                "The model explains part of the variation in disease progression, "
                "but not all of it."
            ),
        },
        {
            "technical_output": "Mean absolute error",
            "plain_language": "Average size of prediction error",
            "careful_statement": (
                "Predictions differ from observed values by a measurable average amount."
            ),
        },
        {
            "technical_output": "Model coefficient",
            "plain_language": "How the model changes prediction when a feature changes",
            "careful_statement": (
                "A larger coefficient means the model relies more strongly on that feature, "
                "not that the feature causes the outcome."
            ),
        },
        {
            "technical_output": "Cross-validation variability",
            "plain_language": "How much performance changes across data splits",
            "careful_statement": (
                "Stable performance across folds increases confidence, but does not prove causation."
            ),
        },
        {
            "technical_output": "Feature importance",
            "plain_language": "How strongly the model uses a feature for prediction",
            "careful_statement": (
                "A feature can be useful for prediction without being a causal driver."
            ),
        },
    ]

    return pd.DataFrame(rows)


def extract_metric(metrics_df: pd.DataFrame | None, metric_name: str) -> float | None:
    """Extract a metric value from a small metrics table when possible."""
    if metrics_df is None or metrics_df.empty:
        return None

    lower_columns = {column.lower(): column for column in metrics_df.columns}

    if "metric" in lower_columns and "value" in lower_columns:
        metric_col = lower_columns["metric"]
        value_col = lower_columns["value"]
        matched = metrics_df[metrics_df[metric_col].astype(str).str.lower() == metric_name.lower()]
        if not matched.empty:
            return float(matched.iloc[0][value_col])

    if metric_name in metrics_df.columns:
        return float(metrics_df.iloc[0][metric_name])

    return None


def build_summary(metrics_df: pd.DataFrame | None, coefficients_df: pd.DataFrame | None) -> str:
    """Build a reusable markdown communication summary."""
    mae = extract_metric(metrics_df, "mae")
    r2 = extract_metric(metrics_df, "r2")

    metric_sentence = (
        "The model shows moderate predictive ability. It captures some signal in the data, "
        "but prediction errors remain large enough that results should be interpreted cautiously."
    )

    if mae is not None and r2 is not None:
        metric_sentence = (
            f"The model explains about {r2:.0%} of the variation in disease progression. "
            f"Its predictions differ from observed values by about {mae:.1f} units on average."
        )

    feature_sentence = (
        "Some clinical features are important predictors in this model. "
        "This means the model uses those features to estimate disease progression."
    )

    if coefficients_df is not None and not coefficients_df.empty:
        coef_df = coefficients_df.copy()
        coef_cols = {column.lower(): column for column in coef_df.columns}
        feature_col = coef_cols.get("feature") or coef_cols.get("variable")
        coefficient_col = coef_cols.get("coefficient") or coef_cols.get("coef")

        if feature_col and coefficient_col:
            coef_df["absolute_coefficient"] = coef_df[coefficient_col].abs()
            top_features = (
                coef_df.sort_values("absolute_coefficient", ascending=False)
                .head(3)[feature_col]
                .astype(str)
                .tolist()
            )
            if top_features:
                feature_sentence = (
                    "The strongest predictors used by the model include "
                    + ", ".join(top_features)
                    + ". These should be interpreted as predictive features, not causal explanations."
                )

    return f"""# Diabetes Model Communication Summary

## Objective
Predict disease progression using the available clinical features in the diabetes dataset.

## Model Summary
A linear regression model was used as a baseline predictive model.

## Key Result
{metric_sentence}

## Interpretation
{feature_sentence}

## Limitation
These results describe predictive associations in this dataset. They do not establish causation, clinical effectiveness, or general truth beyond the data and modeling setup.
"""


def main() -> None:
    """Create communication artifacts."""
    REPORTS_DIR.mkdir(exist_ok=True)

    metrics_df = read_optional_csv(METRICS_PATH)
    coefficients_df = read_optional_csv(COEFFICIENTS_PATH)

    communication_df = build_communication_table()
    communication_df.to_csv(COMMUNICATION_TABLE_PATH, index=False)

    summary_text = build_summary(metrics_df, coefficients_df)
    COMMUNICATION_SUMMARY_PATH.write_text(summary_text)

    print(f"Saved communication table: {COMMUNICATION_TABLE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved communication summary: {COMMUNICATION_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
