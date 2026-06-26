from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def main() -> None:
    input_path = Path("data/raw/people-example.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            "Missing input file: data/raw/people-example.csv. "
            "Run scripts/python/08a_create_example_people_dataset.py first."
        )

    people = pd.read_csv(input_path)

    target = people["purchased"]
    identifiers = people[["person_id"]]

    predictors = people.drop(columns=["purchased", "person_id"])

    numeric_columns = ["age", "income"]
    categorical_columns = ["city"]

    scaler = StandardScaler()

    scaled_numeric = pd.DataFrame(
        scaler.fit_transform(predictors[numeric_columns]),
        columns=[f"{column}_scaled" for column in numeric_columns],
        index=predictors.index,
    )

    encoded_categories = pd.get_dummies(
        predictors[categorical_columns],
        prefix=categorical_columns,
        dtype=int,
    )

    derived_features = pd.DataFrame(index=predictors.index)
    derived_features["income_per_age"] = people["income"] / people["age"]

    model_ready = pd.concat(
        [
            identifiers,
            scaled_numeric,
            encoded_categories,
            derived_features,
            target.rename("target_purchased"),
        ],
        axis=1,
    )

    output_path = output_dir / "model-ready-example.csv"
    model_ready.to_csv(output_path, index=False)

    print(f"Model-ready feature table written to: {output_path}")
    print(f"Rows: {model_ready.shape[0]}")
    print(f"Columns: {model_ready.shape[1]}")
    print()
    print("Columns:")
    for column in model_ready.columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()
