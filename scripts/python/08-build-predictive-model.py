#!/usr/bin/env python3
"""Build and evaluate the Chapter 08 predictive modelling workflow."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "08-property-modelling-data.csv"
FIGURE_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
MODEL_PATH = ROOT / "models" / "08-property-value-pipeline.joblib"


def create_property_data(n_rows: int = 900) -> pd.DataFrame:
    """Create reproducible mixed-type regression data with modest missingness."""
    rng = np.random.default_rng(RANDOM_SEED)
    neighbourhood = rng.choice(["central", "north", "south", "west"], n_rows)
    area_m2 = np.clip(rng.normal(145, 48, n_rows), 45, 330)
    age_years = np.clip(rng.gamma(2.2, 9, n_rows), 0, 80)
    bedrooms = np.clip(np.rint(area_m2 / 48 + rng.normal(0, 0.75, n_rows)), 1, 7)
    distance_km = np.clip(rng.gamma(2.0, 3.0, n_rows), 0.2, 25)
    energy_rating = rng.choice(["A", "B", "C", "D"], n_rows, p=[0.12, 0.33, 0.38, 0.17])

    neighbourhood_effect = pd.Series(neighbourhood).map(
        {"central": 95_000, "north": 42_000, "south": 12_000, "west": 28_000}
    ).to_numpy()
    energy_effect = pd.Series(energy_rating).map(
        {"A": 35_000, "B": 20_000, "C": 8_000, "D": -5_000}
    ).to_numpy()
    value = (
        85_000
        + 2_350 * area_m2
        - 1_050 * age_years
        - 5_000 * distance_km
        + 13_000 * bedrooms
        + neighbourhood_effect
        + energy_effect
        + 24_000 * np.sin(area_m2 / 58)
        + rng.normal(0, 28_000, n_rows)
    )

    data = pd.DataFrame(
        {
            "area_m2": area_m2.round(1),
            "age_years": age_years.round(1),
            "bedrooms": bedrooms.astype(int),
            "distance_km": distance_km.round(1),
            "neighbourhood": neighbourhood,
            "energy_rating": energy_rating,
            "property_value": value.round(0),
        }
    )
    for column in ["area_m2", "age_years", "energy_rating"]:
        missing_rows = rng.choice(data.index, size=int(0.03 * n_rows), replace=False)
        data.loc[missing_rows, column] = np.nan
    return data


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric_features),
         ("categorical", categorical_pipeline, categorical_features)]
    )


def main() -> None:
    for path in [DATA_PATH.parent, FIGURE_DIR, TABLE_DIR, MODEL_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    data = create_property_data()
    data.to_csv(DATA_PATH, index=False)

    target = "property_value"
    numeric_features = ["area_m2", "age_years", "bedrooms", "distance_km"]
    categorical_features = ["neighbourhood", "energy_rating"]
    features = numeric_features + categorical_features
    X, y = data[features], data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    estimators = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=10.0),
        "Random forest": RandomForestRegressor(
            n_estimators=350, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }
    cv_rows = []
    for model_name, estimator in estimators.items():
        pipeline = Pipeline(
            [("preprocessor", make_preprocessor(numeric_features, categorical_features)),
             ("model", estimator)]
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error",
                     "rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=-1,
        )
        for fold in range(cv.get_n_splits()):
            cv_rows.append(
                {"model": model_name, "fold": fold + 1,
                 "mae": -scores["test_mae"][fold],
                 "rmse": -scores["test_rmse"][fold], "r2": scores["test_r2"][fold]}
            )
    cv_results = pd.DataFrame(cv_rows)
    cv_results.to_csv(TABLE_DIR / "08-cross-validation-results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.boxplot(data=cv_results, x="model", y="rmse", hue="model", palette="viridis", ax=ax)
    sns.stripplot(data=cv_results, x="model", y="rmse", color="black", size=5, ax=ax)
    ax.set(xlabel=None, ylabel="Cross-validated RMSE", title="Validation error across five folds")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "08-model-comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    forest_pipeline = Pipeline(
        [("preprocessor", make_preprocessor(numeric_features, categorical_features)),
         ("model", RandomForestRegressor(n_estimators=350, random_state=RANDOM_SEED, n_jobs=-1))]
    )
    search = GridSearchCV(
        forest_pipeline,
        {"model__max_depth": [None, 8, 14], "model__min_samples_leaf": [1, 3, 6],
         "model__max_features": ["sqrt", 0.8]},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    predictions = search.best_estimator_.predict(X_test)
    metrics = pd.DataFrame(
        [{"mae": mean_absolute_error(y_test, predictions),
          "rmse": mean_squared_error(y_test, predictions) ** 0.5,
          "r2": r2_score(y_test, predictions),
          "best_cv_rmse": -search.best_score_, "best_parameters": str(search.best_params_)}]
    )
    metrics.to_csv(TABLE_DIR / "08-test-metrics.csv", index=False)
    prediction_table = pd.DataFrame(
        {"observed": y_test.to_numpy(), "predicted": predictions,
         "residual": y_test.to_numpy() - predictions}
    )
    prediction_table.to_csv(TABLE_DIR / "08-test-predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    sns.scatterplot(data=prediction_table, x="observed", y="predicted", alpha=0.72, ax=ax)
    limits = [prediction_table[["observed", "predicted"]].min().min(),
              prediction_table[["observed", "predicted"]].max().max()]
    ax.plot(limits, limits, linestyle="--", color="#B4442A", linewidth=1.8, label="Perfect prediction")
    ax.set(xlim=limits, ylim=limits, title="Observed versus predicted property values")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "08-observed-vs-predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    importance_result = permutation_importance(
        search.best_estimator_, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=20, random_state=RANDOM_SEED, n_jobs=-1
    )
    importance = pd.DataFrame(
        {"feature": features, "importance_mean": importance_result.importances_mean,
         "importance_sd": importance_result.importances_std}
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(TABLE_DIR / "08-permutation-importance.csv", index=False)

    plot_data = importance.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.errorbar(plot_data["importance_mean"], plot_data["feature"],
                xerr=plot_data["importance_sd"], fmt="o", color="#2A6F62",
                ecolor="#7AA89F", capsize=3)
    ax.axvline(0, color="0.35", linewidth=1, linestyle="--")
    ax.set(xlabel="Increase in RMSE after permutation", ylabel=None,
           title="Test-set permutation importance")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "08-permutation-importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    joblib.dump(search.best_estimator_, MODEL_PATH)
    print(f"Saved data: {DATA_PATH.relative_to(ROOT)}")
    print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")
    print(metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

