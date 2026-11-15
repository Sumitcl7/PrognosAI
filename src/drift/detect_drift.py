
from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.model_selection import GroupShuffleSplit


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

OUTPUT_DIR = ROOT_DIR / "artifacts" / "drift"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FDR_THRESHOLD = 0.05

# Simulated gradual shifts measured in standard deviations.
SHIFT_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5]

# Number of sensors to modify in the simulation.
TOP_SENSORS_TO_SHIFT = 5


# ==================================================
# BENJAMINI-HOCHBERG CORRECTION
# ==================================================

def benjamini_hochberg(p_values, alpha=0.05):
    """
    Benjamini-Hochberg false discovery rate correction.

    Returns:
        adjusted_p_values
        rejected: boolean array
    """

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)

    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    if not np.isfinite(p_values).all():
        raise ValueError("Invalid p-values encountered.")

    order = np.argsort(p_values)

    sorted_p = p_values[order]

    ranks = np.arange(1, n + 1)

    adjusted_sorted = sorted_p * n / ranks

    # Enforce monotonicity.
    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1
    )

    adjusted = np.empty(n)

    adjusted[order] = adjusted_sorted

    rejected = adjusted <= alpha

    return adjusted, rejected


# ==================================================
# LOAD DATA
# ==================================================

print("Loading FD001 training data...")

data = pd.read_csv(DATA_PATH)

with open(METRICS_PATH, "r") as file:
    metadata = json.load(file)

features = metadata["features"]

sensor_features = [
    feature
    for feature in features
    if feature.startswith("sensor_")
]

if not sensor_features:
    raise ValueError("No sensor features found.")

required_columns = ["engine_id"] + sensor_features

missing = [
    column for column in required_columns
    if column not in data.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

print(f"Observations: {len(data)}")
print(f"Engines: {data['engine_id'].nunique()}")
print(f"Sensor features: {len(sensor_features)}")


# ==================================================
# ENGINE-LEVEL SENSOR SUMMARIES
# ==================================================

# Calculate one mean sensor value per engine.
#
# This prototype detects changes in cross-engine
# sensor distributions, not within-engine temporal
# degradation trends.

engine_data = (
    data.groupby("engine_id")[sensor_features]
    .mean()
    .reset_index()
)

print("\nEngine-level summaries created.")
print(f"Engine summaries: {len(engine_data)}")


# ==================================================
# REFERENCE / CURRENT ENGINE SPLIT
# ==================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

reference_indices, current_indices = next(
    splitter.split(
        engine_data,
        groups=engine_data["engine_id"]
    )
)

reference_data = (
    engine_data.iloc[reference_indices]
    .copy()
)

current_data = (
    engine_data.iloc[current_indices]
    .copy()
)

reference_ids = set(reference_data["engine_id"])
current_ids = set(current_data["engine_id"])

if reference_ids.intersection(current_ids):
    raise RuntimeError("Engine overlap detected!")

reference_values = reference_data[sensor_features]
current_values = current_data[sensor_features]

print("\n========== ENGINE SPLIT ==========")
print(f"Reference engines: {len(reference_data)}")
print(f"Current engines: {len(current_data)}")
print("Engine overlap: 0")


# ==================================================
# SENSOR DRIFT DETECTOR
# ==================================================

def detect_drift(reference, current, alpha=0.05):
    """
    Compare reference and current sensor distributions.

    KS test is applied independently to each sensor.
    Benjamini-Hochberg corrects for multiple testing.
    """

    records = []

    for sensor in sensor_features:

        reference_sensor = reference[sensor].dropna()
        current_sensor = current[sensor].dropna()

        if len(reference_sensor) < 2 or len(current_sensor) < 2:
            continue

        statistic, p_value = ks_2samp(
            reference_sensor,
            current_sensor,
            alternative="two-sided",
            method="auto"
        )

        reference_mean = float(reference_sensor.mean())
        current_mean = float(current_sensor.mean())

        reference_std = float(reference_sensor.std())

        if reference_std > 0:
            standardized_shift = (
                current_mean - reference_mean
            ) / reference_std
        else:
            standardized_shift = 0.0

        records.append({
            "sensor": sensor,
            "reference_mean": reference_mean,
            "current_mean": current_mean,
            "reference_std": reference_std,
            "standardized_mean_shift": standardized_shift,
            "ks_statistic": float(statistic),
            "p_value": float(p_value)
        })

    results = pd.DataFrame(records)

    if results.empty:
        raise ValueError("No valid sensor comparisons.")

    adjusted_p, rejected = benjamini_hochberg(
        results["p_value"].to_numpy(),
        alpha=alpha
    )

    results["adjusted_p_value"] = adjusted_p
    results["drift_detected"] = rejected

    results = results.sort_values(
        "adjusted_p_value"
    ).reset_index(drop=True)

    return results


# ==================================================
# BASELINE DRIFT CHECK
# ==================================================

print("\n========== BASELINE DRIFT CHECK ==========")

baseline_results = detect_drift(
    reference_values,
    current_values,
    alpha=FDR_THRESHOLD
)

baseline_drift_count = int(
    baseline_results["drift_detected"].sum()
)

baseline_status = (
    "DRIFT_DETECTED"
    if baseline_drift_count > 0
    else "STABLE"
)

print(f"Status: {baseline_status}")
print(f"Drifted sensors: {baseline_drift_count}")
print(f"Total sensors: {len(baseline_results)}")

print("\nSensor-level results:")
print(
    baseline_results.round(4).to_string(index=False)
)

baseline_results.to_csv(
    OUTPUT_DIR / "baseline_drift_report.csv",
    index=False
)


# ==================================================
# SIMULATED GRADUAL SENSOR DRIFT
# ==================================================

print("\n========== SIMULATED DRIFT EXPERIMENT ==========")

# Select sensors with the highest variability
# across reference engines.

reference_std = (
    reference_values.std()
    .sort_values(ascending=False)
)

selected_sensors = (
    reference_std.head(TOP_SENSORS_TO_SHIFT)
    .index.tolist()
)

print("Sensors selected for simulation:")
print(selected_sensors)

simulation_results = []

rng = np.random.default_rng(RANDOM_STATE)

for shift_level in SHIFT_LEVELS:

    simulated_current = current_values.copy()

    # Apply a controlled positive mean shift to
    # selected sensors.
    for sensor in selected_sensors:

        sensor_std = reference_values[sensor].std()

        shift_amount = shift_level * sensor_std

        simulated_current[sensor] = (
            simulated_current[sensor] + shift_amount
        )

    results = detect_drift(
        reference_values,
        simulated_current,
        alpha=FDR_THRESHOLD
    )

    detected_sensors = results.loc[
        results["drift_detected"],
        "sensor"
    ].tolist()

    selected_detected = [
        sensor
        for sensor in selected_sensors
        if sensor in detected_sensors
    ]

    simulation_results.append({
        "shift_level_std": shift_level,
        "drifted_sensor_count": len(detected_sensors),
        "total_sensor_count": len(sensor_features),
        "injected_sensor_count": len(selected_sensors),
        "injected_sensors_detected": len(selected_detected),
        "injected_sensors": ", ".join(selected_sensors),
        "detected_sensors": ", ".join(detected_sensors),
        "status": (
            "DRIFT_DETECTED"
            if len(detected_sensors) > 0
            else "STABLE"
        )
    })

    print(
        f"\nShift: {shift_level:.2f} standard deviations"
    )

    print(
        f"Detected sensors: {len(detected_sensors)}"
    )

    print(
        f"Injected sensors detected: "
        f"{len(selected_detected)}/{len(selected_sensors)}"
    )

    print(
        f"Status: {simulation_results[-1]['status']}"
    )


# ==================================================
# SAVE SIMULATION RESULTS
# ==================================================

simulation_df = pd.DataFrame(simulation_results)

simulation_df.to_csv(
    OUTPUT_DIR / "simulated_drift_results.csv",
    index=False
)


# ==================================================
# SAVE METRICS
# ==================================================

summary = {
    "experiment": "Sensor Distribution Drift Detection",
    "dataset": "NASA C-MAPSS FD001",
    "method": "Kolmogorov-Smirnov + Benjamini-Hochberg",
    "fdr_threshold": FDR_THRESHOLD,
    "reference_engines": int(len(reference_data)),
    "current_engines": int(len(current_data)),
    "number_of_sensors": len(sensor_features),
    "baseline_status": baseline_status,
    "baseline_drifted_sensors": baseline_drift_count,
    "simulation_shift_levels": SHIFT_LEVELS,
    "simulated_sensors": selected_sensors,
    "notes": (
        "Uses per-engine mean sensor summaries. "
        "Simulated drift is an artificial distribution shift. "
        "Results are a prototype, not evidence of real industrial drift."
    )
}

with open(
    OUTPUT_DIR / "drift_metrics.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)


# ==================================================
# COMPLETION
# ==================================================

print("\n========== FILES GENERATED ==========")

print("baseline_drift_report.csv")
print("simulated_drift_results.csv")
print("drift_metrics.json")

print("\n========== EXPERIMENT COMPLETED ==========")

print(f"Baseline status: {baseline_status}")
print(f"Baseline drifted sensors: {baseline_drift_count}")
print(f"Simulation scenarios: {len(simulation_results)}")

print("\nDrift detection prototype completed successfully.")
