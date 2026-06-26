from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    people = pd.DataFrame(
        {
            "person_id": ["P001", "P002", "P003", "P004", "P005", "P006"],
            "age": [22, 35, 47, 51, 29, 43],
            "income": [25000, 48000, 52000, 61000, 39000, 57000],
            "city": ["A", "B", "A", "C", "B", "C"],
            "purchased": [0, 1, 1, 1, 0, 1],
        }
    )

    output_path = output_dir / "people-example.csv"
    people.to_csv(output_path, index=False)

    print(f"Example dataset written to: {output_path}")
    print(f"Rows: {people.shape[0]}")
    print(f"Columns: {people.shape[1]}")


if __name__ == "__main__":
    main()
