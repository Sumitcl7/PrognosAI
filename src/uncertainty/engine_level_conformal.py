
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error


# ==================================================
# CONFIGURATION
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "FD001_train_with_RUL.csv"
)

METRICS_PATH = (
    ROOT_DIR
    / "artifacts"
    / "baseline"
    / "metrics.json"
)

OUTPUT_DIR = ROOT_DIR / "artifacts" / "uncertainty"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
RUL_CAP = 125
RANDOM_STATE = 42


# ==================================================
# CONFORMAL QUANTILE
# ==================================================

def calculate_conformal_quantile(scores, alpha=0.10):
    """
    Calculate the finite-sample split conformal quantile.

    Scores must contain one score per calibration engine.
    """

    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        raise ValueError("No calibration scores available.")

    if not np.isfinite(scores).all():
        raise ValueError("Calibration scores contain invalid values.")

    scores = np.sort(scores)

    n = len(scores)

    rank = int(np.ceil((n + 1) * (1 - alpha)))

    # If the required rank exceeds the sample size,
    # the valid finite-sample quantile is infinity.
    if rank > n:
        return float("inf")

    return float(scores[rank - 1])


# ==================================================
# LOAD DATA
# ==================================================

print("Loading FD001 training data...")

data = pd.read_csv(DATA_PATH)

with open(METRICS_PATH, "r") as file:
    metadata = json.load(file)

features = metadata["features"]

required_columns = ["engine_id", "cycle", "RUL_capped"]

missing_columns = [
    column for column in required_columns + features
    if column not in data.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

data = data.sort_values(
    ["engine_id", "cycle"]
).reset_index(drop=True)

print(f"Observations: {len(data)}")
print(f"Engines: {data['engine_id'].nunique()}")
print(f"Features: {len(features)}")


# ==================================================
# ENGINE-LEVEL TRAIN/CALIBRATION SPLIT
# ==================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_indices, calibration_indices = next(
    splitter.split(
        data,
        groups=data["engine_id"]
    )
)

train_data = data.iloc[train_indices].copy()
calibration_data = data.iloc[calibration_indices].copy()

train_engines = set(train_data["engine_id"].unique())
calibration_engines = set(calibration_data["engine_id"].unique())

if train_engines.intersection(calibration_engines):
    raise RuntimeError("Engine overlap detected!")

print("\n========== DATA SPLIT ==========")
print(f"Training engines: {len(train_engines)}")
print(f"Calibration engines: {len(calibration_engines)}")


# ==================================================
# TRAIN CALIBRATION MODEL
# ==================================================

# Separate model: the calibration engines must not
# have been used to fit this model.
# The original baseline remains unchanged.

print("\nTraining Random Forest calibration model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(
    train_data[features],
    train_data["RUL_capped"]
)


# ==================================================
# CALIBRATION PREDICTIONS
# ==================================================

print("Predicting calibration engines...")

calibration_data["predicted_RUL"] = model.predict(
    calibration_data[features]
)

calibration_data["predicted_RUL"] = (
    calibration_data["predicted_RUL"].clip(
        lower=0,
        upper=RUL_CAP
    )
)

calibration_data["absolute_error"] = (
    calibration_data["RUL_capped"]
    - calibration_data["predicted_RUL"]
).abs()


# ==================================================
# ONE SCORE PER ENGINE
# ==================================================

# The maximum absolute residual across all observed
# cycles of an engine is used as its calibration score.

engine_scores = (
    calibration_data.groupby("engine_id")
    .agg(
        max_absolute_error=("absolute_error", "max"),
        mean_absolute_error=("absolute_error", "mean"),
        observations=("absolute_error", "count")
    )
    .reset_index()
)

if len(engine_scores) != len(calibration_engines):
    raise RuntimeError("Unexpected number of engine scores.")

print("\n========== ENGINE CALIBRATION SCORES ==========")
print(engine_scores.round(3).to_string(index=False))


# ==================================================
# ENGINE-LEVEL CONFORMAL QUANTILE
# ==================================================

q_hat = calculate_conformal_quantile(
    engine_scores["max_absolute_error"].to_numpy(),
    alpha=ALPHA
)

print("\n========== ENGINE-LEVEL CONFORMAL ==========")
print(f"Calibration engines: {len(engine_scores)}")
print(f"Target coverage: {(1 - ALPHA) * 100:.1f}%")
print(f"Conformal quantile: {q_hat:.4f}")


# ==================================================
# CALIBRATION INTERVALS
# ==================================================

# These are observation-level intervals using the
# engine-level maximum-error quantile.

calibration_data["lower_bound"] = (
    calibration_data["predicted_RUL"] - q_hat
).clip(lower=0, upper=RUL_CAP)

calibration_data["upper_bound"] = (
    calibration_data["predicted_RUL"] + q_hat
).clip(lower=0, upper=RUL_CAP)

calibration_data["covered"] = (
    (
        calibration_data["RUL_capped"]
        >= calibration_data["lower_bound"]
    )
    &
    (
        calibration_data["RUL_capped"]
        <= calibration_data["upper_bound"]
    )
)

# Evaluate whether every observation of an engine
# is covered by its corresponding interval.

engine_coverage = (
    calibration_data.groupby("engine_id")
    .agg(
        all_observations_covered=("covered", "all"),
        observation_coverage=("covered", "mean"),
        average_interval_width=(
            "upper_bound",
            lambda upper: np.nan
        )
    )
    .reset_index()
)

# Calculate interval width separately.
calibration_data["interval_width"] = (
    calibration_data["upper_bound"]
    - calibration_data["lower_bound"]
)

engine_width = (
    calibration_data.groupby("engine_id")["interval_width"]
    .mean()
    .rename("average_interval_width")
    .reset_index()
)

engine_coverage = engine_coverage.drop(
    columns=["average_interval_width"]
).merge(
    engine_width,
    on="engine_id"
)

trajectory_coverage = float(
    engine_coverage["all_observations_covered"].mean()
)

observation_coverage = float(
    calibration_data["covered"].mean()
)

average_width = float(
    calibration_data["interval_width"].mean()
)

calibration_mae = float(
    mean_absolute_error(
        calibration_data["RUL_capped"],
        calibration_data["predicted_RUL"]
    )
)


# ==================================================
# SAVE RESULTS
# ==================================================

joblib.dump(
    model,
    OUTPUT_DIR / "engine_level_calibration_model.joblib"
)

engine_scores.to_csv(
    OUTPUT_DIR / "engine_level_calibration_scores.csv",
    index=False
)

calibration_data.to_csv(
    OUTPUT_DIR / "engine_level_calibration_predictions.csv",
    index=False
)

engine_coverage.to_csv(
    OUTPUT_DIR / "engine_level_coverage.csv",
    index=False
)

results = {
    "method": "Engine-Level Split Conformal Prediction",
    "dataset": "NASA C-MAPSS FD001",
    "alpha": ALPHA,
    "target_coverage": 1 - ALPHA,
    "calibration_engines": int(len(engine_scores)),
    "training_engines": int(len(train_engines)),
    "calibration_observations": int(len(calibration_data)),
    "conformal_quantile": q_hat,
    "calibration_MAE": calibration_mae,
    "observation_coverage": observation_coverage,
    "trajectory_coverage": trajectory_coverage,
    "average_interval_width": average_width,
    "rul_cap": RUL_CAP,
    "score_definition": (
        "Maximum absolute residual across observed cycles "
        "for each calibration engine"
    ),
    "coverage_note": (
        "The engine-level finite-sample guarantee relies on "
        "exchangeability of engine trajectories and a fixed "
        "prediction model. Calibration coverage is descriptive; "
        "it is not an independent test evaluation."
    )
}

with open(
    OUTPUT_DIR / "engine_level_conformal_metrics.json",
    "w"
) as file:
    json.dump(results, file, indent=4)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n========== CALIBRATION PERFORMANCE ==========")

print(f"Observation coverage: {observation_coverage * 100:.2f}%")
print(f"Trajectory coverage: {trajectory_coverage * 100:.2f}%")
print(f"Average interval width: {average_width:.2f} cycles")
print(f"Calibration MAE: {calibration_mae:.4f} cycles")

print("\n========== FILES GENERATED ==========")
print("engine_level_calibration_model.joblib")
print("engine_level_calibration_scores.csv")
print("engine_level_calibration_predictions.csv")
print("engine_level_coverage.csv")
print("engine_level_conformal_metrics.json")

print("\nEngine-level conformal experiment completed.")
