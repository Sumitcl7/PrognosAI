
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "raw" / "CMAPSSData"
MODEL_PATH = ROOT_DIR / "artifacts" / "baseline" / "random_forest_fd001.joblib"
METRICS_PATH = ROOT_DIR / "artifacts" / "baseline" / "metrics.json"

OUTPUT_DIR = ROOT_DIR / "artifacts" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUL_CAP = 125

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# --------------------------------------------------
# NASA ASYMMETRIC SCORE
# --------------------------------------------------

def nasa_score(actual, predicted):
    errors = predicted - actual
    scores = np.zeros(len(errors))

    early = errors < 0
    late = errors >= 0

    scores[early] = np.exp(-errors[early] / 13.0) - 1
    scores[late] = np.exp(errors[late] / 10.0) - 1

    return float(np.sum(scores))


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "NASA_score": nasa_score(actual, predicted),
    }


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading trained baseline model...")

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, "r") as file:
    metadata = json.load(file)

features = metadata["features"]

print("Model loaded successfully.")
print(f"Features: {len(features)}")


# --------------------------------------------------
# LOAD OFFICIAL TEST DATA
# --------------------------------------------------

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

engine_ids = sorted(test_data["engine_id"].unique())

if len(engine_ids) != len(official_rul):
    raise ValueError("Test engine count and official RUL count do not match.")

print(f"Test observations: {len(test_data)}")
print(f"Test engines: {len(engine_ids)}")


# --------------------------------------------------
# RECONSTRUCT ACTUAL RUL
# --------------------------------------------------

final_cycles = test_data.groupby("engine_id")["cycle"].max()

rul_mapping = dict(zip(engine_ids, official_rul))

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

print("\nRUL reconstruction completed.")
print(f"Maximum actual test RUL: {test_data['actual_RUL'].max()}")
print(
    "Final-cycle labels above cap:",
    int((official_rul > RUL_CAP).sum())
)


# --------------------------------------------------
# MODEL PREDICTIONS
# --------------------------------------------------

missing_features = [
    feature for feature in features
    if feature not in test_data.columns
]

if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

X_test = test_data[features]

# Preserve original model output.
test_data["predicted_RUL_raw"] = model.predict(X_test)

# Lower bound only for raw evaluation.
test_data["predicted_RUL_raw"] = (
    test_data["predicted_RUL_raw"].clip(lower=0)
)

# Capped prediction for capped-target evaluation.
test_data["predicted_RUL_capped"] = (
    test_data["predicted_RUL_raw"].clip(lower=0, upper=RUL_CAP)
)


# --------------------------------------------------
# FINAL OBSERVATION PER ENGINE
# --------------------------------------------------

final_predictions = (
    test_data.sort_values(["engine_id", "cycle"])
    .groupby("engine_id", as_index=False)
    .tail(1)
    .copy()
)

# Track A: Capped evaluation.
capped_metrics = calculate_metrics(
    final_predictions["actual_RUL_capped"].to_numpy(),
    final_predictions["predicted_RUL_capped"].to_numpy()
)

# Track B: Uncapped evaluation.
# Note: model was trained on capped targets.
uncapped_metrics = calculate_metrics(
    final_predictions["actual_RUL"].to_numpy(),
    final_predictions["predicted_RUL_raw"].to_numpy()
)


# --------------------------------------------------
# ADDITIONAL ALL-CYCLE CAPPED EVALUATION
# --------------------------------------------------

all_cycle_metrics = calculate_metrics(
    test_data["actual_RUL_capped"].to_numpy(),
    test_data["predicted_RUL_capped"].to_numpy()
)


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

test_data.to_csv(
    OUTPUT_DIR / "official_test_predictions.csv",
    index=False
)

final_predictions.to_csv(
    OUTPUT_DIR / "official_test_final_predictions.csv",
    index=False
)


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

results = {
    "dataset": "NASA C-MAPSS FD001",
    "number_of_test_engines": int(len(final_predictions)),
    "number_of_test_observations": int(len(test_data)),
    "training_target": "RUL_capped",
    "rul_cap": RUL_CAP,

    "capped_final_cycle_evaluation": capped_metrics,

    "uncapped_final_cycle_diagnostic": {
        **uncapped_metrics,
        "warning": (
            "Model was trained on capped RUL targets. "
            "Uncapped evaluation is diagnostic and is not "
            "directly comparable to a model trained on uncapped RUL."
        )
    },

    "additional_all_cycle_capped_evaluation": all_cycle_metrics
}

with open(
    OUTPUT_DIR / "official_test_metrics.json",
    "w"
) as file:
    json.dump(results, file, indent=4)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n========== TRACK A: CAPPED FINAL-CYCLE ==========")

for metric, value in capped_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n========== TRACK B: UNCAPPED FINAL-CYCLE ==========")

for metric, value in uncapped_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nNOTE: Track B is diagnostic because the model")
print("was trained using RUL capped at 125 cycles.")

print("\n========== ADDITIONAL ALL-CYCLE CAPPED RESULTS ==========")

for metric, value in all_cycle_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n========== FILES SAVED ==========")
print("official_test_predictions.csv")
print("official_test_final_predictions.csv")
print("official_test_metrics.json")

print("\nEvaluation completed successfully.")
