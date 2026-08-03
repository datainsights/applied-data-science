"""Create leakage-safe, modelling-ready train and test datasets.

Run this script from the repository root after generating the input data:

    python scripts/python/02-prepare-advanced-data.py
    python scripts/python/02-advanced-data-preparation.py

Expected input
--------------
data/processed/advanced_modeling_data.csv

The input must contain one row per customer, a binary ``churned`` target, and
the feature columns declared below. Run this script from the project root.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = Path("data/processed/advanced_modeling_data.csv")
MODELING_DIR = Path("data/modeling")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/tables")

TARGET = "churned"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "usage_hours",
]

CATEGORICAL_FEATURES = [
    "region",
    "plan_type",
    "contract_type",
    "payment_method",
]


def load_and_validate_data(path: Path) -> pd.DataFrame:
    """Load the modelling table and enforce its minimum schema."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "Run scripts/python/02-prepare-advanced-data.py first."
        )

    data = pd.read_csv(path)
    required = {TARGET, ID_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    if data.empty:
        raise ValueError("The modelling table contains no rows.")
    if data[ID_COLUMN].isna().any():
        raise ValueError(f"{ID_COLUMN} contains missing values.")
    if data[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be unique for this customer-level task.")
    if data[TARGET].isna().any():
        raise ValueError(f"{TARGET} contains missing values.")

    observed_targets = set(data[TARGET].unique())
    if not observed_targets.issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only binary values 0 and 1.")
    if data[TARGET].nunique() != 2:
        raise ValueError(f"{TARGET} must contain both outcome classes.")

    return data


def build_preprocessor() -> ColumnTransformer:
    """Define transformations that will be fitted on training data only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


def validate_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Fail early when transformed outputs are incomplete or misaligned."""
    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Training and test feature schemas do not match.")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing values remain after preprocessing.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")
    if TARGET in X_train.columns or ID_COLUMN in X_train.columns:
        raise ValueError("The target or identifier leaked into the feature matrix.")


def save_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    source_rows: int,
) -> None:
    """Write prepared data, the fitted transformer, and audit metadata."""
    for directory in (MODELING_DIR, MODEL_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(MODELING_DIR / "X_train_prepared.csv", index=False)
    X_test.to_csv(MODELING_DIR / "X_test_prepared.csv", index=False)
    y_train.rename(TARGET).to_csv(MODELING_DIR / "y_train.csv", index=False)
    y_test.rename(TARGET).to_csv(MODELING_DIR / "y_test.csv", index=False)
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    summary = {
        "source_file": str(DATA_PATH),
        "source_rows": source_rows,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "target": TARGET,
        "excluded_identifier": ID_COLUMN,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "prepared_feature_count": X_train.shape[1],
        "prepared_features": X_train.columns.tolist(),
        "training_target_rate": float(y_train.mean()),
        "test_target_rate": float(y_test.mean()),
    }

    summary_path = RESULTS_DIR / "data_preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run the complete advanced data-preparation workflow."""
    data = load_and_validate_data(DATA_PATH)

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = data[TARGET].astype("int8")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    validate_outputs(X_train_ready, X_test_ready, y_train, y_test)
    save_outputs(
        X_train_ready,
        X_test_ready,
        y_train,
        y_test,
        preprocessor,
        source_rows=len(data),
    )

    print("Advanced data preparation completed.")
    print(f"Training matrix: {X_train_ready.shape}")
    print(f"Test matrix: {X_test_ready.shape}")
    print(f"Fitted preprocessor: {MODEL_DIR / 'preprocessor.joblib'}")


if __name__ == "__main__":
    main()
