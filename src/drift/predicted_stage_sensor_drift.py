
"""
PrognosAI - Predicted-Stage-Conditioned Sensor Drift

Research objective:
Distinguish unexpected sensor distribution changes from
normal degradation by comparing observations against
sensor profiles conditioned on predicted RUL stages.

Important:
- Uses the frozen baseline Random Forest.
- Uses predicted RUL for stage assignment.
- Never uses actual RUL to assign monitoring stages.
- Training engines build reference profiles.
- Held-out validation engines evaluate detection.
- Synthetic sensor shifts are controlled experiments,
  not evidence of real industrial faults.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed"
    / "FD001_train_with_RUL.csv"
)

MODEL_FILE = (
    ROOT / "artifacts" / "baseline"
    / "random_forest_fd001.joblib"
)

METRICS_FILE = (
    ROOT / "artifacts" / "baseline"
    / "metrics.json"
)

OUTPUT_DIR = (
    ROOT / "artifacts" / "drift"
    / "predicted_stage_sensor"
)

RANDOM_STATE = 42

ALPHA = 0.05

WINDOW_SIZE = 20
WINDOW_STEP = 10

MIN_REFERENCE_SAMPLES = 30

STAGE_BINS = [0, 30, 60, 90, 125]

STAGE_LABELS = [
    "Critical",
    "Late-middle",
    "Middle",
    "Early",
]

SHIFT_LEVELS = [
    0.0, 0.25, 0.50, 0.75, 1.0, 1.5
]

TOP_SENSORS_TO_SHIFT = 5

INJECTION_START_CYCLE = 80


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
        adjusted_sorted, 0, 1
    )

    adjusted = np.empty(n)

    adjusted[order] = adjusted_sorted

    return adjusted, adjusted <= alpha


# ==================================================
# STAGE ASSIGNMENT
# ==================================================

def assign_predicted_stage(df):

    result = df.copy()

    result["predicted_RUL_clipped"] = (
        result["predicted_RUL"].clip(0, 125)
    )

    result["predicted_stage"] = pd.cut(
        result["predicted_RUL_clipped"],
        bins=STAGE_BINS,
        labels=STAGE_LABELS,
        include_lowest=True,
    )

    result = result.dropna(
        subset=["predicted_stage"]
    ).copy()

    result["predicted_stage"] = (
        result["predicted_stage"].astype(str)
    )

    return result


# ==================================================
# LOAD DATA AND FROZEN MODEL
# ==================================================

def load_experiment_data():

    print("Loading FD001 dataset...")

    data = pd.read_csv(DATA_FILE)

    with open(METRICS_FILE, "r") as file:
        metadata = json.load(file)

    features = metadata["features"]

    sensor_features = [
        feature
        for feature in features
        if feature.startswith("sensor_")
    ]

    required = (
        ["engine_id", "cycle", "RUL", "RUL_capped"]
        + features
    )

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    model = joblib.load(MODEL_FILE)

    if len(features) != 18:
        raise ValueError(
            "Expected exactly 18 baseline features."
        )

    data = data.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    data[sensor_features] = (
        data[sensor_features].astype(float)
    )

    print("Dataset records:", len(data))
    print("Engines:", data["engine_id"].nunique())
    print("Features:", len(features))
    print("Sensors:", len(sensor_features))

    return data, model, features, sensor_features, metadata


# ==================================================
# ENGINE SPLIT
# ==================================================

def split_engines(data, metadata):

    training_count = metadata["training_engines"]
    validation_count = metadata["validation_engines"]

    all_engines = sorted(
        data["engine_id"].unique()
    )

    if len(all_engines) != (
        training_count + validation_count
    ):
        raise ValueError(
            "Dataset engine count does not match baseline metadata."
        )

    rng = np.random.default_rng(RANDOM_STATE)

    shuffled = rng.permutation(all_engines)

    training_engines = set(
        int(x) for x in shuffled[:training_count]
    )

    validation_engines = set(
        int(x) for x in shuffled[training_count:]
    )

    # Important: the original baseline split is not reconstructed
    # from a new random split. Use the existing validation file's
    # engine IDs to preserve the original holdout.

    validation_file = (
        ROOT / "artifacts" / "baseline"
        / "validation_predictions.csv"
    )

    original_validation = pd.read_csv(
        validation_file
    )

    validation_engines = set(
        original_validation["engine_id"].astype(int)
    )

    training_engines = (
        set(int(x) for x in all_engines)
        - validation_engines
    )

    if len(training_engines) != training_count:
        raise ValueError(
            "Original training engine count mismatch."
        )

    if len(validation_engines) != validation_count:
        raise ValueError(
            "Original validation engine count mismatch."
        )

    print("\nEngine split:")
    print("Reference engines:", len(training_engines))
    print("Validation engines:", len(validation_engines))

    return training_engines, validation_engines


# ==================================================
# PREDICTIONS
# ==================================================

def generate_predictions(
    data,
    model,
    features,
    training_engines,
    validation_engines,
):

    working = data.copy()

    working["predicted_RUL"] = model.predict(
        working[features]
    )

    working = assign_predicted_stage(working)

    working["split"] = np.where(
        working["engine_id"].isin(training_engines),
        "reference",
        "validation",
    )

    reference = working[
        working["engine_id"].isin(training_engines)
    ].copy()

    validation = working[
        working["engine_id"].isin(validation_engines)
    ].copy()

    # Actual RUL is used only for offline evaluation metrics.
    # It is not used for prediction or stage assignment.

    mae = mean_absolute_error(
        validation["RUL_capped"],
        validation["predicted_RUL"],
    )

    rmse = np.sqrt(
        mean_squared_error(
            validation["RUL_capped"],
            validation["predicted_RUL"],
        )
    )

    r2 = r2_score(
        validation["RUL_capped"],
        validation["predicted_RUL"],
    )

    print("\nFrozen model validation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    return reference, validation


# ==================================================
# BUILD STAGE-CONDITIONED SENSOR PROFILES
# ==================================================

def build_reference_profiles(
    reference,
    sensor_features,
):

    profiles = []

    print("\nBuilding predicted-stage profiles...")

    for stage in STAGE_LABELS:

        stage_data = reference[
            reference["predicted_stage"] == stage
        ]

        for sensor in sensor_features:

            values = stage_data[sensor].dropna()

            if len(values) < MIN_REFERENCE_SAMPLES:
                print(
                    f"WARNING: Insufficient reference data "
                    f"for {stage}, {sensor}: {len(values)}"
                )

                continue

            profiles.append({
                "predicted_stage": stage,
                "sensor": sensor,
                "reference_count": len(values),
                "reference_mean": float(values.mean()),
                "reference_std": float(values.std()),
                "reference_median": float(values.median()),
                "reference_q05": float(values.quantile(0.05)),
                "reference_q95": float(values.quantile(0.95)),
            })

    profile_df = pd.DataFrame(profiles)

    if profile_df.empty:
        raise ValueError(
            "No valid stage-conditioned reference profiles."
        )

    expected_profiles = (
        len(STAGE_LABELS) * len(sensor_features)
    )

    print("Profiles created:", len(profile_df))
    print("Expected profiles:", expected_profiles)

    return profile_df


# ==================================================
# SENSOR DRIFT DETECTION
# ==================================================

def detect_sensor_drift(
    validation,
    reference,
    profiles,
    sensor_features,
    scenario,
):

    results = []

    for engine_id, engine_data in validation.groupby(
        "engine_id"
    ):

        engine_data = engine_data.sort_values(
            "cycle"
        ).reset_index(drop=True)

        min_cycle = int(engine_data["cycle"].min())
        max_cycle = int(engine_data["cycle"].max())

        starts = range(
            min_cycle,
            max_cycle - WINDOW_SIZE + 2,
            WINDOW_STEP,
        )

        for start in starts:

            end = start + WINDOW_SIZE - 1

            window = engine_data[
                (engine_data["cycle"] >= start)
                & (engine_data["cycle"] <= end)
            ]

            if len(window) < WINDOW_SIZE:
                continue

            # Use the dominant predicted stage in the window.
            # The actual RUL is not consulted.

            stage = (
                window["predicted_stage"]
                .mode()
                .iloc[0]
            )

            stage_reference = reference[
                reference["predicted_stage"] == stage
            ]

            sensor_records = []

            for sensor in sensor_features:

                profile = profiles[
                    (profiles["predicted_stage"] == stage)
                    & (profiles["sensor"] == sensor)
                ]

                if profile.empty:
                    continue

                ref_values = (
                    stage_reference[sensor]
                    .dropna()
                    .to_numpy()
                )

                current_values = (
                    window[sensor]
                    .dropna()
                    .to_numpy()
                )

                if (
                    len(ref_values) < MIN_REFERENCE_SAMPLES
                    or len(current_values) < 2
                ):
                    continue

                statistic, p_value = ks_2samp(
                    ref_values,
                    current_values,
                    alternative="two-sided",
                    method="auto",
                )

                profile_row = profile.iloc[0]

                reference_std = profile_row[
                    "reference_std"
                ]

                if reference_std > 0:
                    standardized_shift = (
                        current_values.mean()
                        - profile_row["reference_mean"]
                    ) / reference_std
                else:
                    standardized_shift = 0.0

                sensor_records.append({
                    "scenario": scenario,
                    "engine_id": int(engine_id),
                    "window_start": int(start),
                    "window_end": int(end),
                    "predicted_stage": stage,
                    "sensor": sensor,
                    "ks_statistic": float(statistic),
                    "p_value": float(p_value),
                    "standardized_shift": float(
                        standardized_shift
                    ),
                })

            if not sensor_records:
                continue

            adjusted, rejected = benjamini_hochberg(
                [
                    record["p_value"]
                    for record in sensor_records
                ],
                ALPHA,
            )

            for i, record in enumerate(sensor_records):

                record["adjusted_p_value"] = float(
                    adjusted[i]
                )

                record["drift_detected"] = bool(
                    rejected[i]
                )

                results.append(record)

    sensor_df = pd.DataFrame(results)

    if sensor_df.empty:
        raise ValueError(
            "No sensor monitoring results were generated."
        )

    window_df = (
        sensor_df.groupby(
            [
                "scenario",
                "engine_id",
                "window_start",
                "window_end",
                "predicted_stage",
            ]
        )
        .agg(
            drifted_sensor_count=("drift_detected", "sum"),
            total_sensors=("sensor", "count"),
        )
        .reset_index()
    )

    window_df["drift_status"] = np.where(
        window_df["drifted_sensor_count"] > 0,
        "DRIFT_DETECTED",
        "STABLE",
    )

    return sensor_df, window_df


# ==================================================
# SYNTHETIC DRIFT SIMULATION
# ==================================================

def simulate_sensor_shifts(
    validation,
    reference,
    profiles,
    sensor_features,
    selected_sensors,
):

    simulation_rows = []

    for shift_level in SHIFT_LEVELS:

        print(
            f"\nSynthetic shift: {shift_level:.2f} SD"
        )

        simulated = validation.copy()

        for sensor in selected_sensors:

            reference_std = (
                reference[sensor].std()
            )

            shift_amount = (
                shift_level * reference_std
            )

            mask = (
                simulated["cycle"]
                >= INJECTION_START_CYCLE
            )

            simulated.loc[
                mask, sensor
            ] += shift_amount

        sensor_results, window_results = (
            detect_sensor_drift(
                simulated,
                reference,
                profiles,
                sensor_features,
                scenario=f"shift_{shift_level}",
            )
        )

        sensor_results["is_injected_sensor"] = (
            sensor_results["sensor"].isin(
                selected_sensors
            )
        )

        injected = sensor_results[
            sensor_results["is_injected_sensor"]
        ]

        non_injected = sensor_results[
            ~sensor_results["is_injected_sensor"]
        ]

        simulation_rows.append({
            "shift_level_std": shift_level,
            "injected_sensor_detection_percent": (
                float(injected["drift_detected"].mean() * 100)
            ),
            "non_injected_sensor_alarm_percent": (
                float(non_injected["drift_detected"].mean() * 100)
            ),
            "detected_windows_percent": (
                float(
                    (
                        window_results["drift_status"]
                        == "DRIFT_DETECTED"
                    ).mean() * 100
                )
            ),
            "monitoring_windows": len(window_results),
        })

    return pd.DataFrame(simulation_rows)


# ==================================================
# MAIN
# ==================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("PROGNOSAI - PREDICTED-STAGE SENSOR DRIFT")
    print("=" * 65)

    (
        data,
        model,
        features,
        sensor_features,
        metadata,
    ) = load_experiment_data()

    training_engines, validation_engines = (
        split_engines(data, metadata)
    )

    reference, validation = generate_predictions(
        data,
        model,
        features,
        training_engines,
        validation_engines,
    )

    reference.to_csv(
        OUTPUT_DIR / "reference_predictions.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    profiles = build_reference_profiles(
        reference,
        sensor_features,
    )

    profiles.to_csv(
        OUTPUT_DIR / "predicted_stage_sensor_profiles.csv",
        index=False,
    )

    print("\nReference profiles by stage:")

    print(
        profiles.groupby("predicted_stage")[
            "sensor"
        ].count().to_string()
    )

    print("\n========== NORMAL VALIDATION MONITORING ==========")

    baseline_sensor, baseline_windows = (
        detect_sensor_drift(
            validation,
            reference,
            profiles,
            sensor_features,
            scenario="normal_validation",
        )
    )

    baseline_sensor.to_csv(
        OUTPUT_DIR / "normal_sensor_results.csv",
        index=False,
    )

    baseline_windows.to_csv(
        OUTPUT_DIR / "normal_window_results.csv",
        index=False,
    )

    normal_alarm_rate = float(
        (
            baseline_windows["drift_status"]
            == "DRIFT_DETECTED"
        ).mean() * 100
    )

    print("Monitoring windows:", len(baseline_windows))
    print(
        "Normal validation drift windows:",
        int(
            (
                baseline_windows["drift_status"]
                == "DRIFT_DETECTED"
            ).sum()
        ),
    )
    print(
        f"Normal validation alert rate: "
        f"{normal_alarm_rate:.2f}%"
    )

    # Select sensors using reference data only.
    selected_sensors = (
        reference[sensor_features]
        .std()
        .sort_values(ascending=False)
        .head(TOP_SENSORS_TO_SHIFT)
        .index.tolist()
    )

    print("\nSensors selected for injection:")
    print(selected_sensors)

    simulation = simulate_sensor_shifts(
        validation,
        reference,
        profiles,
        sensor_features,
        selected_sensors,
    )

    simulation.to_csv(
        OUTPUT_DIR / "simulation_results.csv",
        index=False,
    )

    summary = {
        "experiment": "Predicted-stage-conditioned sensor drift",
        "dataset": "NASA C-MAPSS FD001",
        "model": "Frozen baseline RandomForestRegressor",
        "stage_assignment": "Predicted RUL only",
        "reference_engines": len(training_engines),
        "validation_engines": len(validation_engines),
        "features": features,
        "sensors": sensor_features,
        "stages": STAGE_LABELS,
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "alpha": ALPHA,
        "normal_validation_alert_rate_percent": normal_alarm_rate,
        "selected_sensors": selected_sensors,
        "injection_start_cycle": INJECTION_START_CYCLE,
        "simulation_results": simulation.to_dict(
            orient="records"
        ),
        "limitations": [
            "Predicted RUL stages may be misclassified.",
            "Reference profiles combine observations from multiple engines.",
            "KS tests do not establish that a change is a fault.",
            "Synthetic shifts do not represent all real industrial drift.",
            "The same validation cohort is used across simulation levels.",
            "This is a research prototype, not a production-validated detector.",
        ],
    }

    with open(
        OUTPUT_DIR / "predicted_stage_drift_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n========== RESULTS ==========")

    print(
        simulation.round(2).to_string(index=False)
    )

    print("\nArtifacts saved to:")
    print(OUTPUT_DIR)

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
