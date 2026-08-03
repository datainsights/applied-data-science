"""Generate the synthetic customer modelling table used in ADS Chapter 02.

Run this script from the repository root:

    python scripts/python/02-prepare-advanced-data.py

Output
------
data/processed/advanced_modeling_data.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_CUSTOMERS = 1_500
OUTPUT_PATH = Path("data/processed/advanced_modeling_data.csv")


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert log-odds to probabilities."""
    return 1.0 / (1.0 + np.exp(-values))


def generate_customer_data(
    n_customers: int = N_CUSTOMERS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create a reproducible customer-level table with controlled missingness."""
    rng = np.random.default_rng(random_state)

    age = np.clip(np.rint(rng.normal(41, 13, n_customers)), 18, 82).astype(float)
    tenure_months = np.clip(
        np.rint(rng.gamma(shape=2.2, scale=14.0, size=n_customers)), 1, 96
    ).astype(int)

    region = rng.choice(
        ["north", "central", "coastal", "southern"],
        size=n_customers,
        p=[0.23, 0.31, 0.27, 0.19],
    )
    plan_type = rng.choice(
        ["basic", "standard", "premium"],
        size=n_customers,
        p=[0.39, 0.43, 0.18],
    )
    contract_type = rng.choice(
        ["monthly", "annual", "two_year"],
        size=n_customers,
        p=[0.55, 0.34, 0.11],
    )
    payment_method = rng.choice(
        ["bank_transfer", "card", "mobile_money", "invoice"],
        size=n_customers,
        p=[0.25, 0.38, 0.27, 0.10],
    ).astype(object)

    plan_base = pd.Series(plan_type).map(
        {"basic": 29.0, "standard": 52.0, "premium": 84.0}
    ).to_numpy()
    monthly_spend = np.clip(
        plan_base + rng.normal(0, 8, n_customers) + 0.08 * tenure_months,
        12,
        140,
    )

    support_rate = (
        0.9
        + 0.8 * (contract_type == "monthly")
        + 0.5 * (plan_type == "basic")
    )
    support_tickets = rng.poisson(support_rate).astype(float)

    usage_hours = np.clip(
        rng.normal(31, 9, n_customers)
        + 5 * (plan_type == "premium")
        - 3 * (plan_type == "basic"),
        1,
        75,
    )

    # The target is generated from information available before renewal.
    log_odds = (
        -1.15
        + 1.00 * (contract_type == "monthly")
        + 0.38 * (plan_type == "basic")
        + 0.20 * (payment_method == "invoice")
        + 0.23 * support_tickets
        - 0.018 * tenure_months
        - 0.025 * usage_hours
    )
    churn_probability = sigmoid(log_odds)
    churned = rng.binomial(1, churn_probability).astype("int8")

    data = pd.DataFrame(
        {
            "customer_id": [f"CUST-{number:05d}" for number in range(1, n_customers + 1)],
            "age": age,
            "tenure_months": tenure_months,
            "monthly_spend": np.round(monthly_spend, 2),
            "support_tickets": support_tickets,
            "usage_hours": np.round(usage_hours, 1),
            "region": region,
            "plan_type": plan_type,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "churned": churned,
        }
    )

    # Add realistic missing values for Chapter 02's fitted imputers to handle.
    missing_rates = {
        "age": 0.025,
        "monthly_spend": 0.035,
        "support_tickets": 0.020,
        "usage_hours": 0.045,
        "payment_method": 0.030,
    }
    for column, rate in missing_rates.items():
        missing_rows = rng.random(n_customers) < rate
        data.loc[missing_rows, column] = np.nan

    return data


def validate_data(data: pd.DataFrame) -> None:
    """Validate the schema and invariants required by Chapter 02."""
    expected_columns = [
        "customer_id",
        "age",
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "usage_hours",
        "region",
        "plan_type",
        "contract_type",
        "payment_method",
        "churned",
    ]

    if data.columns.tolist() != expected_columns:
        raise ValueError("Generated columns do not match the Chapter 02 schema.")
    if data.empty:
        raise ValueError("The generated dataset is empty.")
    if data["customer_id"].isna().any() or data["customer_id"].duplicated().any():
        raise ValueError("customer_id must be complete and unique.")
    if data["churned"].isna().any():
        raise ValueError("churned must not contain missing values.")
    if set(data["churned"].unique()) != {0, 1}:
        raise ValueError("churned must contain both binary outcome classes.")


def main() -> None:
    """Generate, validate, and save the Chapter 02 input dataset."""
    data = generate_customer_data()
    validate_data(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print("Advanced modelling dataset created.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {data.shape[1]}")
    print(f"Churn rate: {data['churned'].mean():.1%}")
    print(f"Missing values: {int(data.isna().sum().sum()):,}")


if __name__ == "__main__":
    main()
