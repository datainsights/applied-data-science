from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes


def main() -> None:
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    diabetes = load_diabetes()

    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["disease_progression"] = diabetes.target

    output_path = output_dir / "diabetes.csv"
    df.to_csv(output_path, index=False)

    print(f"Diabetes dataset written to: {output_path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print()
    print("Columns:")
    for column in df.columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()
