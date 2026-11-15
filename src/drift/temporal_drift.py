
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.stats import ks_2samp


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

REFERENCE_CYCLES = 30
WINDOW_SIZE = 20
WINDOW_STEP = 10

FDR_THRESHOLD = 0.05

INJECTION_START_CYCLE = 80

SHIFT_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5]

TOP_SENSORS_TO_SHIFT = 5


# ==================================================
# BENJAMINI-HOCHBERG CORRECTION
# ==================================================

def benjamini_hochberg(p_values, alpha=0.05):

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)

    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    order = np.argsort(p_values)

    sorted_p = p_values[order]

    ranks = np.arange(1, n + 1)

    adjusted_sorted = sorted_p * n / ranks

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

required_columns = (
    ["engine_id", "cycle"] + sensor_features
)

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# IMPORTANT FIX:
# Convert all sensor values to float before
# introducing fractional synthetic shifts.

data[sensor_features] = (
    data[sensor_features].astype(float)
)

data = data.sort_values(
    ["engine_id", "cycle"]
).reset_index(drop=True)

print(f"Observations: {len(data)}")
print(f"Engines: {data['engine_id'].nunique()}")
print(f"Sensors monitored: {len(sensor_features)}")


# ==================================================
# REFERENCE DISTRIBUTION
# ==================================================

reference_data = data[
    data["cycle"] <= REFERENCE_CYCLES
].copy()

reference_std = reference_data[
    sensor_features
].std()

print("\n========== REFERENCE WINDOW ==========")
print(f"Reference cycles: 1-{REFERENCE_CYCLES}")
print(f"Reference observations: {len(reference_data)}")


# ==================================================
# TEMPORAL DRIFT DETECTOR
# ==================================================

def detect_temporal_drift(
    input_data,
    scenario_name="baseline",
    minimum_window_start=None
):

    sensor_results = []

    for engine_id, engine_data in input_data.groupby("engine_id"):

        engine_data = engine_data.sort_values(
            "cycle"
        ).reset_index(drop=True)

        reference = engine_data[
            engine_data["cycle"] <= REFERENCE_CYCLES
        ]

        if len(reference) < 10:
            continue

        monitoring_data = engine_data[
            engine_data["cycle"] > REFERENCE_CYCLES
        ]

        if monitoring_data.empty:
            continue

        min_cycle = int(monitoring_data["cycle"].min())
        max_cycle = int(monitoring_data["cycle"].max())

        window_starts = range(
            min_cycle,
            max_cycle - WINDOW_SIZE + 2,
            WINDOW_STEP
        )

        for window_start in window_starts:

            window_end = window_start + WINDOW_SIZE - 1

            if (
                minimum_window_start is not None
                and window_start < minimum_window_start
            ):
                continue

            current_window = engine_data[
                (engine_data["cycle"] >= window_start)
                &
                (engine_data["cycle"] <= window_end)
            ]

            if len(current_window) < WINDOW_SIZE:
                continue

            window_records = []

            for sensor in sensor_features:

                ref_values = (
                    reference[sensor]
                    .dropna()
                    .to_numpy()
                )

                current_values = (
                    current_window[sensor]
                    .dropna()
                    .to_numpy()
                )

                if len(ref_values) < 2 or len(current_values) < 2:
                    continue

                statistic, p_value = ks_2samp(
                    ref_values,
                    current_values,
                    alternative="two-sided",
                    method="auto"
                )

                ref_std = reference_std[sensor]

                if ref_std > 0:
                    standardized_shift = (
                        current_values.mean()
                        - ref_values.mean()
                    ) / ref_std
                else:
                    standardized_shift = 0.0

                window_records.append({
                    "scenario": scenario_name,
                    "engine_id": int(engine_id),
                    "window_start": int(window_start),
                    "window_end": int(window_end),
                    "sensor": sensor,
                    "ks_statistic": float(statistic),
                    "p_value": float(p_value),
                    "standardized_mean_shift": float(
                        standardized_shift
                    ),
                    "reference_observations": len(ref_values),
                    "current_observations": len(current_values)
                })

            if not window_records:
                continue

            adjusted_p, rejected = benjamini_hochberg(
                [
                    record["p_value"]
                    for record in window_records
                ],
                alpha=FDR_THRESHOLD
            )

            for index, record in enumerate(window_records):

                record["adjusted_p_value"] = float(
                    adjusted_p[index]
                )

                record["drift_detected"] = bool(
                    rejected[index]
                )

                sensor_results.append(record)

    sensor_df = pd.DataFrame(sensor_results)

    if sensor_df.empty:
        return sensor_df, pd.DataFrame()

    window_df = (
        sensor_df.groupby(
            [
                "scenario",
                "engine_id",
                "window_start",
                "window_end"
            ]
        )
        .agg(
            drifted_sensor_count=("drift_detected", "sum"),
            total_sensors=("sensor", "count")
        )
        .reset_index()
    )

    window_df["drift_status"] = np.where(
        window_df["drifted_sensor_count"] > 0,
        "DRIFT_DETECTED",
        "STABLE"
    )

    return sensor_df, window_df


# ==================================================
# BASELINE TEMPORAL MONITORING
# ==================================================

print("\n========== REAL TEMPORAL MONITORING ==========")

baseline_sensor_results, baseline_window_results = (
    detect_temporal_drift(
        data,
        scenario_name="baseline"
    )
)

if baseline_sensor_results.empty:
    raise ValueError("No baseline monitoring results generated.")

baseline_sensor_results.to_csv(
    OUTPUT_DIR / "temporal_baseline_sensor_results.csv",
    index=False
)

baseline_window_results.to_csv(
    OUTPUT_DIR / "temporal_baseline_window_results.csv",
    index=False
)

baseline_drift_windows = int(
    (
        baseline_window_results["drift_status"]
        == "DRIFT_DETECTED"
    ).sum()
)

total_windows = len(baseline_window_results)

print(f"Monitoring windows: {total_windows}")
print(f"Windows with changes: {baseline_drift_windows}")

print(
    f"Window change rate: "
    f"{baseline_drift_windows / total_windows * 100:.2f}%"
)

print("\nMost frequently changing sensors:")

sensor_frequency = (
    baseline_sensor_results.groupby("sensor")[
        "drift_detected"
    ]
    .sum()
    .sort_values(ascending=False)
)

print(sensor_frequency.to_string())


# ==================================================
# SELECT SIMULATION SENSORS
# ==================================================

selected_sensors = (
    reference_std
    .sort_values(ascending=False)
    .head(TOP_SENSORS_TO_SHIFT)
    .index.tolist()
)

print("\n========== SIMULATION CONFIGURATION ==========")
print(f"Selected sensors: {selected_sensors}")
print(f"Injection cycle: {INJECTION_START_CYCLE}")


# ==================================================
# SIMULATE SENSOR DRIFT
# ==================================================

simulation_results = []

for shift_level in SHIFT_LEVELS:

    print(
        f"\nRunning simulation: "
        f"{shift_level:.2f} standard deviations"
    )

    simulated_data = data.copy()

    for sensor in selected_sensors:

        shift_amount = (
            shift_level * reference_std[sensor]
        )

        mask = (
            simulated_data["cycle"]
            >= INJECTION_START_CYCLE
        )

        simulated_data.loc[mask, sensor] = (
            simulated_data.loc[mask, sensor]
            + shift_amount
        )

    sensor_results, window_results = (
        detect_temporal_drift(
            simulated_data,
            scenario_name=f"shift_{shift_level}",
            minimum_window_start=INJECTION_START_CYCLE
        )
    )

    if sensor_results.empty:
        continue

    sensor_results["is_injected_sensor"] = (
        sensor_results["sensor"].isin(selected_sensors)
    )

    injected = sensor_results[
        sensor_results["is_injected_sensor"]
    ]

    non_injected = sensor_results[
        ~sensor_results["is_injected_sensor"]
    ]

    injected_detection_rate = float(
        injected["drift_detected"].mean()
    )

    non_injected_detection_rate = float(
        non_injected["drift_detected"].mean()
    )

    injected_window_results = (
        injected.groupby(
            ["engine_id", "window_start", "window_end"]
        )["drift_detected"]
        .any()
    )

    injected_window_detection_rate = float(
        injected_window_results.mean()
    )

    simulation_results.append({
        "shift_level_std": shift_level,
        "injected_sensor_detection_rate": injected_detection_rate,
        "non_injected_sensor_detection_rate": non_injected_detection_rate,
        "injected_window_detection_rate": injected_window_detection_rate,
        "monitoring_windows": int(len(window_results))
    })

    print(
        f"Injected sensor detection rate: "
        f"{injected_detection_rate * 100:.2f}%"
    )

    print(
        f"Non-injected sensor detection rate: "
        f"{non_injected_detection_rate * 100:.2f}%"
    )

    print(
        f"Windows with injected drift detected: "
        f"{injected_window_detection_rate * 100:.2f}%"
    )


# ==================================================
# SAVE SIMULATION RESULTS
# ==================================================

simulation_df = pd.DataFrame(simulation_results)

simulation_df.to_csv(
    OUTPUT_DIR / "temporal_simulation_results.csv",
    index=False
)


# ==================================================
# VISUALIZATION 1: TEMPORAL DRIFT
# ==================================================

temporal_plot_data = (
    baseline_window_results.groupby("window_start")[
        "drifted_sensor_count"
    ]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))

plt.plot(
    temporal_plot_data["window_start"],
    temporal_plot_data["drifted_sensor_count"],
    marker="o"
)

plt.xlabel("Monitoring window start cycle")
plt.ylabel("Average number of changed sensors")
plt.title("Temporal Sensor Distribution Changes - FD001")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "temporal_drift_over_time.png",
    dpi=300
)

plt.close()


# ==================================================
# VISUALIZATION 2: SIMULATION SENSITIVITY
# ==================================================

if not simulation_df.empty:

    plt.figure(figsize=(9, 6))

    plt.plot(
        simulation_df["shift_level_std"],
        simulation_df["injected_sensor_detection_rate"] * 100,
        marker="o",
        label="Injected sensors"
    )

    plt.plot(
        simulation_df["shift_level_std"],
        simulation_df["non_injected_sensor_detection_rate"] * 100,
        marker="s",
        label="Non-injected sensors"
    )

    plt.xlabel("Injected shift (standard deviations)")
    plt.ylabel("Detection rate (%)")
    plt.title("Temporal Drift Detection Sensitivity")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "temporal_drift_sensitivity.png",
        dpi=300
    )

    plt.close()


# ==================================================
# SAVE METRICS
# ==================================================

summary = {
    "experiment": "Temporal Sensor Drift Detection",
    "dataset": "NASA C-MAPSS FD001",
    "method": "Rolling KS Test + Benjamini-Hochberg",
    "reference_cycles": REFERENCE_CYCLES,
    "window_size": WINDOW_SIZE,
    "window_step": WINDOW_STEP,
    "fdr_threshold": FDR_THRESHOLD,
    "monitoring_windows": int(total_windows),
    "windows_with_changes": baseline_drift_windows,
    "baseline_change_percentage": float(
        baseline_drift_windows / total_windows * 100
    ),
    "selected_sensors": selected_sensors,
    "injection_start_cycle": INJECTION_START_CYCLE,
    "simulation_shift_levels": SHIFT_LEVELS,
    "interpretation": (
        "Detects statistical changes relative to early-cycle "
        "sensor distributions. Changes may reflect normal "
        "degradation, operating variation, or injected drift. "
        "They are not automatically faults."
    )
}

with open(
    OUTPUT_DIR / "temporal_drift_metrics.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)


# ==================================================
# COMPLETION
# ==================================================

print("\n========== FILES GENERATED ==========")

print("temporal_baseline_sensor_results.csv")
print("temporal_baseline_window_results.csv")
print("temporal_simulation_results.csv")
print("temporal_drift_over_time.png")
print("temporal_drift_sensitivity.png")
print("temporal_drift_metrics.json")

print("\nTemporal drift experiment completed successfully.")

print(
    f"Baseline change windows: "
    f"{baseline_drift_windows}/{total_windows}"
)
