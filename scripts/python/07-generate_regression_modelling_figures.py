"""Generate the teaching figures for ADS Chapter 07: Regression Modelling."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
FIGURE_DIRECTORY = Path("results/figures")


def create_regression_data() -> pd.DataFrame:
    """Return the reproducible synthetic dataset used throughout Chapter 07."""
    rng = np.random.default_rng(RANDOM_SEED)
    sample_size = 320
    data = pd.DataFrame(
        {
            "exposure": rng.uniform(0, 10, sample_size),
            "age": rng.normal(45, 12, sample_size).clip(18, 80),
            "group": rng.choice(
                ["A", "B", "C"], sample_size, p=[0.45, 0.35, 0.20]
            ),
        }
    )
    data["biomarker"] = 0.55 * data["age"] + rng.normal(0, 6, sample_size)
    group_effect = data["group"].map({"A": 0.0, "B": 4.0, "C": -3.0})
    data["response"] = (
        18
        + 2.4 * data["exposure"]
        - 0.14 * data["exposure"] ** 2
        + 0.28 * data["age"]
        + group_effect
        + rng.normal(0, 4.5, sample_size)
    )
    return data


def save_fit_figure(train_data: pd.DataFrame) -> None:
    """Compare linear and quadratic fits against the observed data."""
    figure, axis = plt.subplots(figsize=(9, 5.6))
    sns.scatterplot(
        data=train_data,
        x="exposure",
        y="response",
        hue="group",
        alpha=0.68,
        palette="colorblind",
        ax=axis,
    )
    sns.regplot(
        data=train_data,
        x="exposure",
        y="response",
        scatter=False,
        ci=None,
        color="#C44E52",
        line_kws={"label": "Linear fit", "linewidth": 2.3},
        ax=axis,
    )
    sns.regplot(
        data=train_data,
        x="exposure",
        y="response",
        order=2,
        scatter=False,
        ci=None,
        color="#2A9D8F",
        line_kws={"label": "Quadratic fit", "linewidth": 2.3},
        ax=axis,
    )
    axis.set(title="A straight line can hide a nonlinear relationship")
    axis.legend(title="Group and model", frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "07-regression-fit.png", dpi=300)
    plt.close(figure)


def create_design_matrix(data: pd.DataFrame) -> np.ndarray:
    """Create the design matrix used by the documented multiple model."""
    return np.column_stack(
        [
            data["exposure"],
            data["exposure"] ** 2,
            data["age"],
            (data["group"] == "B").astype(float),
            (data["group"] == "C").astype(float),
        ]
    )


def save_diagnostic_figure(
    model: LinearRegression, train_data: pd.DataFrame, design: np.ndarray
) -> None:
    """Plot residual-versus-fitted and normal Q-Q diagnostics."""
    fitted = model.predict(design)
    residuals = train_data["response"].to_numpy() - fitted
    residual_scale = residuals.std(ddof=design.shape[1] + 1)
    studentized_residuals = residuals / residual_scale
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    sns.scatterplot(x=fitted, y=studentized_residuals, alpha=0.7, ax=axes[0])
    residual_trend = pd.DataFrame(
        {"fitted": fitted, "residual": studentized_residuals}
    ).sort_values("fitted")
    residual_trend["smooth"] = residual_trend["residual"].rolling(
        window=35, center=True, min_periods=12
    ).mean()
    axes[0].plot(
        residual_trend["fitted"],
        residual_trend["smooth"],
        color="#C44E52",
        linewidth=2,
    )
    axes[0].axhline(0, color="0.35", linestyle="--", linewidth=1)
    axes[0].set(
        title="Residuals versus fitted values",
        xlabel="Fitted response",
        ylabel="Studentized residual",
    )

    theoretical, ordered = stats.probplot(studentized_residuals, dist="norm")[0]
    axes[1].scatter(theoretical, ordered, alpha=0.7)
    slope, intercept = np.polyfit(theoretical, ordered, 1)
    axes[1].plot(
        theoretical,
        intercept + slope * theoretical,
        color="#C44E52",
        linewidth=2,
    )
    axes[1].set(title="Normal Q–Q plot", xlabel="Theoretical quantile", ylabel="Observed quantile")
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "07-regression-diagnostics.png", dpi=300)
    plt.close(figure)


def save_uncertainty_figure(
    model: LinearRegression, train_data: pd.DataFrame, design: np.ndarray
) -> None:
    """Show confidence and prediction intervals across exposure values."""
    prediction_grid = pd.DataFrame(
        {
            "exposure": np.linspace(0, 10, 150),
            "age": train_data["age"].mean(),
            "group": "A",
        }
    )
    prediction_design = create_design_matrix(prediction_grid)
    predicted_mean = model.predict(prediction_design)

    design_with_intercept = np.column_stack([np.ones(len(design)), design])
    prediction_with_intercept = np.column_stack(
        [np.ones(len(prediction_design)), prediction_design]
    )
    residuals = train_data["response"].to_numpy() - model.predict(design)
    degrees_of_freedom = len(train_data) - design_with_intercept.shape[1]
    residual_variance = np.sum(residuals**2) / degrees_of_freedom
    covariance_factor = np.linalg.pinv(
        design_with_intercept.T @ design_with_intercept
    )
    mean_variance = np.einsum(
        "ij,jk,ik->i",
        prediction_with_intercept,
        covariance_factor,
        prediction_with_intercept,
    ) * residual_variance
    critical_value = stats.t.ppf(0.975, degrees_of_freedom)
    mean_margin = critical_value * np.sqrt(mean_variance)
    prediction_margin = critical_value * np.sqrt(mean_variance + residual_variance)

    x_values = prediction_grid["exposure"].to_numpy()
    figure, axis = plt.subplots(figsize=(9, 5.6))
    axis.fill_between(
        x_values,
        predicted_mean - prediction_margin,
        predicted_mean + prediction_margin,
        color="#4C72B0",
        alpha=0.16,
        label="95% prediction interval",
    )
    axis.fill_between(
        x_values,
        predicted_mean - mean_margin,
        predicted_mean + mean_margin,
        color="#2A9D8F",
        alpha=0.35,
        label="95% confidence interval",
    )
    axis.plot(
        x_values,
        predicted_mean,
        color="#173F5F",
        linewidth=2.5,
        label="Estimated mean response",
    )
    axis.set(
        title="Uncertainty depends on what is being estimated",
        xlabel="Exposure",
        ylabel="Predicted response",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "07-regression-uncertainty.png", dpi=300)
    plt.close(figure)


def main() -> None:
    """Fit the chapter models and write all figures."""
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    data = create_regression_data()
    train_data, _ = train_test_split(
        data, test_size=0.20, random_state=RANDOM_SEED
    )
    design = create_design_matrix(train_data)
    multiple_model = LinearRegression().fit(design, train_data["response"])

    save_fit_figure(train_data)
    save_diagnostic_figure(multiple_model, train_data, design)
    save_uncertainty_figure(multiple_model, train_data, design)
    print(f"Saved Chapter 07 figures to {FIGURE_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
