
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

DATA_PATH = ROOT_DIR / "data" / "processed" / "FD001_train_with_RUL.csv"

MODEL_PATH = ROOT_DIR / "artifacts" / "baseline" / "random_forest_fd001.joblib"
METRICS_PATH = ROOT_DIR / "artifacts" / "baseline" / "metrics.json"

OUTPUT_DIR = ROOT_DIR / "artifacts" / "uncertainty"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
RANDOM_STATE = 42


# ==================================================
# CONFORMAL QUANTILE
# ==================================================

def calculate_conformal_quantile(residuals, alpha=0.10):
    """
    Calculate the finite-sample split conformal quantile.

    residuals: absolute prediction errors on calibration data.
    alpha: miscoverage rate (0.10 means 90% nominal coverage).
    """

    residuals = np.asarray(residuals)
    residuals = np.sort(residuals)

    n = len(residuals)

    if n == 0:
        raise ValueError("Calibration residuals are empty.")

    rank = int(np.ceil((n + 1) * (1 - alpha)))

    # Finite-sample correction.
    if rank > n:
        return float("inf")

    return float(residuals[rank - 1])


# ==================================================
# LOAD DATA AND BASELINE
# ==================================================

print("Loading FD001 training data...")

data = pd.read_csv(DATA_PATH)

with open(METRICS_PATH, "r") as file:
    metadata = json.load(file)

features = metadata["features"]

if "RUL_capped" not in data.columns:
    raise ValueError("RUL_capped column is missing.")

if "engine_id" not in data.columns:
    raise ValueError("engine_id column is missing.")

missing_features = [
    feature for feature in features
    if feature not in data.columns
]

if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

model = joblib.load(MODEL_PATH)

print(f"Total observations: {len(data)}")
print(f"Total engines: {data['engine_id'].nunique()}")
print(f"Features: {len(features)}")


# ==================================================
# ENGINE-LEVEL TRAIN/CALIBRATION SPLIT
# ==================================================

# Important:
# We split by engine, not by individual observations,
# to avoid observations from the same engine appearing
# in both training and calibration datasets.

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
print(f"Training observations: {len(train_data)}")
print(f"Calibration observations: {len(calibration_data)}")


# ==================================================
# IMPORTANT: FIT A SEPARATE MODEL FOR CALIBRATION
# ==================================================

# Split conformal requires the calibration observations
# to be independent of the data used to fit the model.
#
# We therefore train a separate model on the training
# subset and calibrate it on held-out engines.
#
# The original frozen baseline remains unchanged.

print("\nTraining calibration model...")

calibration_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

calibration_model.fit(
    train_data[features],
    train_data["RUL_capped"]
)


# ==================================================
# CALIBRATION PREDICTIONS
# ==================================================

print("Calculating calibration residuals...")

X_calibration = calibration_data[features]
y_calibration = calibration_data["RUL_capped"].to_numpy()

calibration_predictions = calibration_model.predict(
    X_calibration
)

calibration_predictions = np.clip(
    calibration_predictions,
    0,
    125
)

absolute_residuals = np.abs(
    y_calibration - calibration_predictions
)


# ==================================================
# CALCULATE CONFORMAL QUANTILE
# ==================================================

q_hat = calculate_conformal_quantile(
    absolute_residuals,
    alpha=ALPHA
)

print("\n========== CONFORMAL CALIBRATION ==========")
print(f"Target coverage: {(1 - ALPHA) * 100:.1f}%")
print(f"Calibration residual count: {len(absolute_residuals)}")
print(f"Conformal quantile: {q_hat:.4f}")


# ==================================================
# CALIBRATION INTERVALS
# ==================================================

lower = np.clip(
    calibration_predictions - q_hat,
    0,
    125
)

upper = np.clip(
    calibration_predictions + q_hat,
    0,
    125
)

covered = (
    (y_calibration >= lower)
    & (y_calibration <= upper)
)

coverage = float(np.mean(covered))
average_width = float(np.mean(upper - lower))

print("\n========== CALIBRATION PERFORMANCE ==========")
print(f"Empirical coverage: {coverage * 100:.2f}%")
print(f"Average interval width: {average_width:.2f} cycles")
print(f"Calibration MAE: {mean_absolute_error(y_calibration, calibration_predictions):.4f}")


# ==================================================
# SAVE CALIBRATION MODEL AND ARTIFACTS
# ==================================================

joblib.dump(
    calibration_model,
    OUTPUT_DIR / "conformal_calibration_model.joblib"
)

calibration_results = calibration_data[
    ["engine_id", "cycle", "RUL_capped"]
].copy()

calibration_results["predicted_RUL"] = calibration_predictions
calibration_results["lower_bound"] = lower
calibration_results["upper_bound"] = upper
calibration_results["absolute_error"] = absolute_residuals
calibration_results["covered"] = covered

calibration_results.to_csv(
    OUTPUT_DIR / "calibration_predictions.csv",
    index=False
)

results = {
    "method": "Split Conformal Prediction",
    "dataset": "NASA C-MAPSS FD001",
    "alpha": ALPHA,
    "target_coverage": 1 - ALPHA,
    "conformal_quantile": q_hat,
    "calibration_engines": len(calibration_engines),
    "calibration_observations": len(calibration_data),
    "calibration_MAE": float(
        mean_absolute_error(y_calibration, calibration_predictions)
    ),
    "calibration_empirical_coverage": coverage,
    "calibration_average_interval_width": average_width,
    "rul_cap": 125
}

with open(
    OUTPUT_DIR / "conformal_metrics.json",
    "w"
) as file:
    json.dump(results, file, indent=4)


# ==================================================
# COMPLETION
# ==================================================

print("\n========== FILES GENERATED ==========")
print("conformal_calibration_model.joblib")
print("calibration_predictions.csv")
print("conformal_metrics.json")

print("\nSplit conformal calibration completed.")
