from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


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

    model = LinearRegression()

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2",
    )

    cv_df = pd.DataFrame(
        {
            "cv_fold": range(1, len(cv_scores) + 1),
            "r2": cv_scores,
        }
    ).round(3)

    summary_df = pd.DataFrame(
        {
            "metric": ["mean_r2", "std_r2", "min_r2", "max_r2"],
            "value": [
                cv_scores.mean(),
                cv_scores.std(),
                cv_scores.min(),
                cv_scores.max(),
            ],
        }
    ).round(3)

    cv_path = reports_dir / "diabetes-cross-validation-r2.csv"
    summary_path = reports_dir / "diabetes-cross-validation-summary.csv"

    cv_df.to_csv(cv_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Cross-validation fold scores written to: {cv_path}")
    print(f"Cross-validation summary written to: {summary_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
