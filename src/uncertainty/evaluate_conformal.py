
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error


# ==================================================
# CONFIGURATION
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "raw" / "CMAPSSData"

ARTIFACT_DIR = ROOT_DIR / "artifacts" / "uncertainty"
BASELINE_DIR = ROOT_DIR / "artifacts" / "baseline"

OBSERVATION_MODEL_PATH = (
    ARTIFACT_DIR / "conformal_calibration_model.joblib"
)

ENGINE_MODEL_PATH = (
    ARTIFACT_DIR / "engine_level_calibration_model.joblib"
)

OBSERVATION_METRICS_PATH = (
    ARTIFACT_DIR / "conformal_metrics.json"
)

ENGINE_METRICS_PATH = (
    ARTIFACT_DIR / "engine_level_conformal_metrics.json"
)

BASELINE_METADATA_PATH = (
    BASELINE_DIR / "metrics.json"
)

OUTPUT_DIR = ARTIFACT_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUL_CAP = 125

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ==================================================
# METRICS FUNCTIONS
# ==================================================

def calculate_coverage(actual, lower, upper):
    covered = (
        (actual >= lower)
        & (actual <= upper)
    )

    return float(np.mean(covered))


def calculate_interval_width(lower, upper):
    return float(np.mean(upper - lower))


def evaluate_intervals(actual, predicted, q_hat):
    lower = np.clip(
        predicted - q_hat,
        0,
        RUL_CAP
    )

    upper = np.clip(
        predicted + q_hat,
        0,
        RUL_CAP
    )

    coverage = calculate_coverage(
        actual,
        lower,
        upper
    )

    width = calculate_interval_width(
        lower,
        upper
    )

    return lower, upper, coverage, width


# ==================================================
# LOAD CALIBRATION ARTIFACTS
# ==================================================

print("Loading calibration artifacts...")

observation_model = joblib.load(
    OBSERVATION_MODEL_PATH
)

engine_model = joblib.load(
    ENGINE_MODEL_PATH
)

with open(OBSERVATION_METRICS_PATH, "r") as file:
    observation_metadata = json.load(file)

with open(ENGINE_METRICS_PATH, "r") as file:
    engine_metadata = json.load(file)

with open(BASELINE_METADATA_PATH, "r") as file:
    baseline_metadata = json.load(file)

features = baseline_metadata["features"]

q_observation = observation_metadata["conformal_quantile"]
q_engine = engine_metadata["conformal_quantile"]

print("Calibration artifacts loaded.")
print(f"Observation-level quantile: {q_observation:.4f}")
print(f"Engine-level quantile: {q_engine:.4f}")


# ==================================================
# LOAD OFFICIAL TEST DATA
# ==================================================

print("\nLoading official FD001 test data...")

test_data = pd.read_csv(
    DATA_DIR / "test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=COLUMNS
)

official_rul = pd.read_csv(
    DATA_DIR / "RUL_FD001.txt",
    sep=r"\s+",
    header=None
).iloc[:, 0].to_numpy()

engine_ids = sorted(
    test_data["engine_id"].unique()
)

if len(engine_ids) != len(official_rul):
    raise ValueError(
        "Test engine count and official RUL count do not match."
    )

print(f"Test observations: {len(test_data)}")
print(f"Test engines: {len(engine_ids)}")


# ==================================================
# RECONSTRUCT TEST RUL
# ==================================================

final_cycles = (
    test_data.groupby("engine_id")["cycle"].max()
)

rul_mapping = dict(
    zip(engine_ids, official_rul)
)

test_data["final_observed_cycle"] = (
    test_data["engine_id"].map(final_cycles)
)

test_data["official_final_RUL"] = (
    test_data["engine_id"].map(rul_mapping)
)

test_data["actual_RUL"] = (
    test_data["official_final_RUL"]
    + test_data["final_observed_cycle"]
    - test_data["cycle"]
)

test_data["actual_RUL_capped"] = (
    test_data["actual_RUL"].clip(upper=RUL_CAP)
)

if (test_data["actual_RUL"] < 0).any():
    raise ValueError("Negative reconstructed RUL detected.")

print("\nRUL reconstruction completed.")


# ==================================================
# PREPARE FEATURES
# ==================================================

missing_features = [
    feature for feature in features
    if feature not in test_data.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

X_test = test_data[features]

y_true = test_data["actual_RUL_capped"].to_numpy()


# ==================================================
# GENERATE PREDICTIONS
# ==================================================

print("\nGenerating predictions...")

test_data["observation_prediction"] = (
    observation_model.predict(X_test)
)

test_data["engine_prediction"] = (
    engine_model.predict(X_test)
)

test_data["observation_prediction"] = (
    test_data["observation_prediction"].clip(
        lower=0,
        upper=RUL_CAP
    )
)

test_data["engine_prediction"] = (
    test_data["engine_prediction"].clip(
        lower=0,
        upper=RUL_CAP
    )
)


# ==================================================
# OBSERVATION-LEVEL INTERVALS
# ==================================================

(
    test_data["observation_lower"],
    test_data["observation_upper"],
    observation_coverage,
    observation_width
) = evaluate_intervals(
    y_true,
    test_data["observation_prediction"].to_numpy(),
    q_observation
)


# ==================================================
# ENGINE-LEVEL INTERVALS
# ==================================================

(
    test_data["engine_lower"],
    test_data["engine_upper"],
    engine_observation_coverage,
    engine_width
) = evaluate_intervals(
    y_true,
    test_data["engine_prediction"].to_numpy(),
    q_engine
)


# ==================================================
# ENGINE TRAJECTORY COVERAGE
# ==================================================

test_data["observation_covered"] = (
    (test_data["actual_RUL_capped"] >= test_data["observation_lower"])
    &
    (test_data["actual_RUL_capped"] <= test_data["observation_upper"])
)

test_data["engine_covered"] = (
    (test_data["actual_RUL_capped"] >= test_data["engine_lower"])
    &
    (test_data["actual_RUL_capped"] <= test_data["engine_upper"])
)

trajectory_results = (
    test_data.groupby("engine_id")
    .agg(
        observation_trajectory_covered=(
            "observation_covered",
            "all"
        ),
        engine_trajectory_covered=(
            "engine_covered",
            "all"
        ),
        observation_coverage=(
            "observation_covered",
            "mean"
        ),
        engine_observation_coverage=(
            "engine_covered",
            "mean"
        ),
        observations=("cycle", "count")
    )
    .reset_index()
)

observation_trajectory_coverage = float(
    trajectory_results[
        "observation_trajectory_covered"
    ].mean()
)

engine_trajectory_coverage = float(
    trajectory_results[
        "engine_trajectory_covered"
    ].mean()
)


# ==================================================
# FINAL-CYCLE EVALUATION
# ==================================================

final_data = (
    test_data.sort_values(["engine_id", "cycle"])
    .groupby("engine_id", as_index=False)
    .tail(1)
    .copy()
)

final_coverage_observation = calculate_coverage(
    final_data["actual_RUL_capped"].to_numpy(),
    final_data["observation_lower"].to_numpy(),
    final_data["observation_upper"].to_numpy()
)

final_coverage_engine = calculate_coverage(
    final_data["actual_RUL_capped"].to_numpy(),
    final_data["engine_lower"].to_numpy(),
    final_data["engine_upper"].to_numpy()
)

final_width_observation = calculate_interval_width(
    final_data["observation_lower"].to_numpy(),
    final_data["observation_upper"].to_numpy()
)

final_width_engine = calculate_interval_width(
    final_data["engine_lower"].to_numpy(),
    final_data["engine_upper"].to_numpy()
)


# ==================================================
# COVERAGE BY RUL RANGE
# ==================================================

def assign_rul_range(rul):
    if rul <= 30:
        return "0-30 cycles"
    elif rul <= 60:
        return "31-60 cycles"
    elif rul <= 90:
        return "61-90 cycles"
    elif rul <= 125:
        return "91-125 cycles"

    return "Above 125"


test_data["rul_range"] = (
    test_data["actual_RUL"].apply(assign_rul_range)
)

range_results = (
    test_data.groupby("rul_range", observed=True)
    .agg(
        observations=("engine_id", "count"),
        observation_coverage=("observation_covered", "mean"),
        engine_interval_coverage=("engine_covered", "mean"),
        observation_width=("observation_lower", lambda x: np.nan),
    )
    .reset_index()
)

# Calculate average widths separately.
range_widths = (
    test_data.groupby("rul_range", observed=True)
    .agg(
        observation_width=(
            "observation_upper",
            lambda x: np.nan
        ),
        engine_width=(
            "engine_upper",
            lambda x: np.nan
        )
    )
    .reset_index()
)

range_widths["observation_width"] = (
    test_data.assign(
        observation_interval_width=(
            test_data["observation_upper"]
            - test_data["observation_lower"]
        )
    )
    .groupby("rul_range", observed=True)[
        "observation_interval_width"
    ].mean().values
)

range_widths["engine_width"] = (
    test_data.assign(
        engine_interval_width=(
            test_data["engine_upper"]
            - test_data["engine_lower"]
        )
    )
    .groupby("rul_range", observed=True)[
        "engine_interval_width"
    ].mean().values
)

range_results = range_results.drop(
    columns=["observation_width"]
).merge(
    range_widths,
    on="rul_range"
)


# ==================================================
# MAE COMPARISON
# ==================================================

observation_mae = mean_absolute_error(
    y_true,
    test_data["observation_prediction"]
)

engine_mae = mean_absolute_error(
    y_true,
    test_data["engine_prediction"]
)


# ==================================================
# SAVE PREDICTIONS AND RESULTS
# ==================================================

test_data.to_csv(
    OUTPUT_DIR / "conformal_test_predictions.csv",
    index=False
)

final_data.to_csv(
    OUTPUT_DIR / "conformal_test_final_predictions.csv",
    index=False
)

trajectory_results.to_csv(
    OUTPUT_DIR / "conformal_test_engine_coverage.csv",
    index=False
)

range_results.to_csv(
    OUTPUT_DIR / "conformal_test_rul_range_coverage.csv",
    index=False
)


# ==================================================
# SUMMARY JSON
# ==================================================

results = {
    "dataset": "NASA C-MAPSS FD001 Official Test",
    "test_engines": int(len(engine_ids)),
    "test_observations": int(len(test_data)),
    "target": "RUL_capped",
    "rul_cap": RUL_CAP,

    "observation_level": {
        "quantile": float(q_observation),
        "MAE": float(observation_mae),
        "all_observation_coverage": observation_coverage,
        "all_observation_average_width": observation_width,
        "trajectory_coverage": observation_trajectory_coverage,
        "final_cycle_coverage": final_coverage_observation,
        "final_cycle_average_width": final_width_observation
    },

    "engine_level": {
        "quantile": float(q_engine),
        "MAE": float(engine_mae),
        "all_observation_coverage": engine_observation_coverage,
        "all_observation_average_width": engine_width,
        "trajectory_coverage": engine_trajectory_coverage,
        "final_cycle_coverage": final_coverage_engine,
        "final_cycle_average_width": final_width_engine
    },

    "interpretation_note": (
        "Independent evaluation on official test engines. "
        "Coverage is empirical test coverage, not a guarantee. "
        "The calibration quantiles were estimated using training "
        "engines only. Metrics use capped RUL targets."
    )
}

with open(
    OUTPUT_DIR / "conformal_test_metrics.json",
    "w"
) as file:
    json.dump(results, file, indent=4)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n========== OBSERVATION-LEVEL CONFORMAL ==========")

print(f"MAE: {observation_mae:.4f}")
print(f"All-observation coverage: {observation_coverage * 100:.2f}%")
print(f"Trajectory coverage: {observation_trajectory_coverage * 100:.2f}%")
print(f"Average interval width: {observation_width:.2f} cycles")
print(f"Final-cycle coverage: {final_coverage_observation * 100:.2f}%")
print(f"Final-cycle width: {final_width_observation:.2f} cycles")


print("\n========== ENGINE-LEVEL CONFORMAL ==========")

print(f"MAE: {engine_mae:.4f}")
print(f"All-observation coverage: {engine_observation_coverage * 100:.2f}%")
print(f"Trajectory coverage: {engine_trajectory_coverage * 100:.2f}%")
print(f"Average interval width: {engine_width:.2f} cycles")
print(f"Final-cycle coverage: {final_coverage_engine * 100:.2f}%")
print(f"Final-cycle width: {final_width_engine:.2f} cycles")


print("\n========== COVERAGE BY RUL RANGE ==========")
print(range_results.round(4).to_string(index=False))


print("\n========== GENERATED FILES ==========")
print("conformal_test_predictions.csv")
print("conformal_test_final_predictions.csv")
print("conformal_test_engine_coverage.csv")
print("conformal_test_rul_range_coverage.csv")
print("conformal_test_metrics.json")

print("\nIndependent conformal evaluation completed.")
