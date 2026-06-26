from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    input_path = Path("reports/diabetes-observed-vs-predicted.csv")
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            "Missing input file: reports/diabetes-observed-vs-predicted.csv. "
            "Run scripts/python/09b_train_baseline_linear_model.py first."
        )

    results_df = pd.read_csv(input_path)

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(
        results_df["observed"],
        results_df["predicted"],
        c=results_df["absolute_error"],
        cmap="viridis",
        edgecolors="black",
        linewidth=0.4,
        alpha=0.9,
    )

    min_val = min(results_df["observed"].min(), results_df["predicted"].min())
    max_val = max(results_df["observed"].max(), results_df["predicted"].max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Observed progression")
    plt.ylabel("Predicted progression")
    plt.title("Observed vs Predicted")
    plt.colorbar(scatter, label="Absolute Error")
    plt.grid(alpha=0.2)

    output_path = figures_dir / "diabetes-observed-vs-predicted.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Figure written to: {output_path}")


if __name__ == "__main__":
    main()
