"""
PrognosAI - Calibrated Sensor Drift Detection

Research design:
    64 engines: reference profile construction
    16 engines: threshold calibration
    20 engines: held-out validation

Stage assignment:
    Frozen baseline predicted RUL only.

Scoring:
    Stage-conditioned robust sensor deviations.
    Median/MAD reference scaling.
    Mean of the five largest standardized sensor deviations per window.
    Engine-level calibration using each calibration engine's maximum window score.

Synthetic experiment:
    Persistent mean shifts are injected after a selected cycle into 10 of
    the 20 validation engines. The other 10 engines remain controls.
    Detection on affected engines and false alarms on controls are reported
    separately.

Important:
    Synthetic drift is not real industrial fault evidence.
    This experiment is separate from previous drift experiments.
    The 16-engine calibration cohort is small; thresholds are exploratory,
    not production guarantees.
"""

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

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

VALIDATION_FILE = (
    ROOT / "artifacts" / "baseline"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "artifacts" / "drift"
    / "calibrated_sensor"
)

RANDOM_STATE = 42
ALPHA = 0.10

WINDOW_SIZE = 20
WINDOW_STEP = 10

STAGE_BINS = [0, 30, 60, 90, 125]
STAGE_LABELS = [
    "Critical",
    "Late-middle",
    "Middle",
    "Early",
]

VARIABILITY_FLOORS = [0.01, 0.05, 0.10]
SHIFT_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5]

TOP_SENSORS_TO_SHIFT = 5
TOP_K_SCORE = 5

INJECTION_START_CYCLE = 80
AFFECTED_ENGINE_COUNT = 10
MIN_PROFILE_ROWS = 30


# ==================================================
# UTILITIES
# ==================================================

def json_safe(value):
    """Convert NumPy/Pandas scalar values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]

    if isinstance(value, np.generic):
        return value.item()

    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set)) else False:
        return None

    return value


def engine_flag_table(window_scores, threshold):
    """Convert window scores into one alarm flag per engine."""
    scored = window_scores.copy()
    scored["drift_detected"] = (
        scored["aggregate_score"] > threshold
    )

    flags = (
        scored.groupby("engine_id")
        .agg(
            drift_detected=("drift_detected", "any"),
            maximum_score=("aggregate_score", "max"),
            windows=("aggregate_score", "size"),
        )
        .reset_index()
    )

    flags["engine_id"] = flags["engine_id"].astype(int)
    return scored, flags


def percentage_flagged(flags, engine_ids):
    """Return percentage of the supplied engine IDs with an alarm."""
    engine_ids = sorted(int(x) for x in engine_ids)
    subset = flags[flags["engine_id"].isin(engine_ids)]

    if len(engine_ids) == 0:
        return None

    # Missing engine scores are treated as not scored, not silently as stable.
    if subset["engine_id"].nunique() != len(engine_ids):
        missing = sorted(set(engine_ids) - set(subset["engine_id"].astype(int)))
        raise ValueError(f"Some engines have no scored windows: {missing}")

    return float(subset["drift_detected"].mean() * 100.0)


# ==================================================
# ENGINE SPLIT
# ==================================================

def split_engines(data):
    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation predictions not found: {VALIDATION_FILE}"
        )

    original_validation = pd.read_csv(VALIDATION_FILE)

    if "engine_id" not in original_validation.columns:
        raise ValueError(
            "validation_predictions.csv must contain engine_id."
        )

    validation_engines = sorted(
        original_validation["engine_id"].astype(int).unique().tolist()
    )

    all_engines = sorted(data["engine_id"].astype(int).unique().tolist())
    reference_engines = sorted(set(all_engines) - set(validation_engines))

    if len(reference_engines) != 80:
        raise ValueError(
            f"Expected 80 reference engines, found {len(reference_engines)}."
        )

    if len(validation_engines) != 20:
        raise ValueError(
            f"Expected 20 validation engines, found {len(validation_engines)}."
        )

    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = rng.permutation(reference_engines)

    profile_engines = sorted(shuffled[:64].astype(int).tolist())
    calibration_engines = sorted(shuffled[64:].astype(int).tolist())

    return (
        set(profile_engines),
        set(calibration_engines),
        set(validation_engines),
    )


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

    result = result.dropna(subset=["predicted_stage"]).copy()
    result["predicted_stage"] = result["predicted_stage"].astype(str)

    return result


# ==================================================
# LOAD AND PREDICT
# ==================================================

def load_and_predict():
    print("Loading dataset and frozen baseline...")

    for path in [DATA_FILE, MODEL_FILE, METRICS_FILE, VALIDATION_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required project file not found: {path}")

    data = pd.read_csv(DATA_FILE)

    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    features = metadata.get("features")
    if not features:
        raise ValueError("Baseline metrics.json does not contain 'features'.")

    sensor_features = [
        feature for feature in features
        if feature.startswith("sensor_")
    ]

    if not sensor_features:
        raise ValueError("No sensor features were found in baseline metadata.")

    required = ["engine_id", "cycle", "RUL_capped"] + features
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    model = joblib.load(MODEL_FILE)

    predictions = data[
        ["engine_id", "cycle", "RUL_capped"] + sensor_features
    ].copy()

    # Ensure decimal synthetic shifts can be applied safely.
    predictions[sensor_features] = (
        predictions[sensor_features].astype(float)
    )

    predictions["predicted_RUL"] = model.predict(data[features])
    predictions = assign_predicted_stage(predictions)

    profile_engines, calibration_engines, validation_engines = (
        split_engines(data)
    )

    predictions["split"] = np.select(
        [
            predictions["engine_id"].isin(profile_engines),
            predictions["engine_id"].isin(calibration_engines),
        ],
        ["profile", "calibration"],
        default="validation",
    )

    validation = predictions[
        predictions["split"] == "validation"
    ]

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

    print("\nENGINE SPLIT")
    print("Profile engines:", len(profile_engines))
    print("Calibration engines:", len(calibration_engines))
    print("Validation engines:", len(validation_engines))

    print("\nFROZEN MODEL VALIDATION")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    return (
        predictions,
        sensor_features,
        profile_engines,
        calibration_engines,
        validation_engines,
    )


# ==================================================
# BUILD ROBUST STAGE-CONDITIONED PROFILES
# ==================================================

def build_profiles(
    predictions,
    sensor_features,
    profile_engines,
    variability_floor,
):
    reference = predictions[
        predictions["engine_id"].isin(profile_engines)
    ].copy()

    profiles = []

    for stage in STAGE_LABELS:
        stage_data = reference[
            reference["predicted_stage"] == stage
        ]

        if stage_data.empty:
            raise ValueError(
                f"No reference observations found for stage '{stage}'."
            )

        for sensor in sensor_features:
            values = stage_data[sensor].dropna()

            if len(values) < MIN_PROFILE_ROWS:
                raise ValueError(
                    f"Insufficient profile observations for "
                    f"{stage}/{sensor}: {len(values)}"
                )

            median = float(values.median())
            mad = float(
                np.median(np.abs(values.to_numpy(dtype=float) - median))
            )
            std = float(values.std())

            robust_scale = 1.4826 * mad
            effective_scale = max(robust_scale, variability_floor)

            profiles.append({
                "predicted_stage": stage,
                "sensor": sensor,
                "reference_count": int(len(values)),
                "reference_engines": int(stage_data["engine_id"].nunique()),
                "reference_median": median,
                "reference_mad": mad,
                "reference_std": std,
                "robust_scale": robust_scale,
                "effective_scale": effective_scale,
                "variability_floor": float(variability_floor),
            })

    profile_df = pd.DataFrame(profiles)

    expected = len(STAGE_LABELS) * len(sensor_features)
    if len(profile_df) != expected:
        raise ValueError(
            f"Expected {expected} stage/sensor profiles, "
            f"created {len(profile_df)}."
        )

    return profile_df


# ==================================================
# WINDOW-LEVEL SENSOR DEVIATION SCORING
# ==================================================

def score_windows(
    data,
    profiles,
    sensor_features,
    scenario,
):
    results = []

    profile_lookup = {
        (row.predicted_stage, row.sensor): row
        for row in profiles.itertuples(index=False)
    }

    for engine_id, engine_data in data.groupby("engine_id"):
        engine_data = engine_data.sort_values("cycle").reset_index(drop=True)

        min_cycle = int(engine_data["cycle"].min())
        max_cycle = int(engine_data["cycle"].max())

        for start in range(
            min_cycle,
            max_cycle - WINDOW_SIZE + 2,
            WINDOW_STEP,
        ):
            end = start + WINDOW_SIZE - 1

            window = engine_data[
                (engine_data["cycle"] >= start)
                & (engine_data["cycle"] <= end)
            ]

            if len(window) < WINDOW_SIZE:
                continue

            stage_mode = window["predicted_stage"].mode()
            if stage_mode.empty:
                continue

            stage = str(stage_mode.iloc[0])
            sensor_scores = []

            for sensor in sensor_features:
                profile = profile_lookup.get((stage, sensor))
                if profile is None:
                    continue

                current_median = float(window[sensor].median())
                deviation = current_median - float(profile.reference_median)
                scale = max(float(profile.effective_scale), 1e-12)
                standardized = deviation / scale

                sensor_scores.append({
                    "sensor": sensor,
                    "score": abs(standardized),
                    "standardized_deviation": standardized,
                })

            if not sensor_scores:
                continue

            scores = np.array(
                [item["score"] for item in sensor_scores],
                dtype=float,
            )

            top_k = min(TOP_K_SCORE, len(scores))
            top_scores = np.sort(scores)[-top_k:]

            # Sparse-shift-sensitive aggregate:
            # average the five largest sensor deviations, not the median
            # across all 15 sensors.
            aggregate_score = float(np.mean(top_scores))

            results.append({
                "scenario": scenario,
                "engine_id": int(engine_id),
                "window_start": int(start),
                "window_end": int(end),
                "predicted_stage": stage,
                "aggregate_score": aggregate_score,
                "top_k_mean_score": aggregate_score,
                "max_sensor_score": float(np.max(scores)),
                "mean_sensor_score": float(np.mean(scores)),
                "median_sensor_score": float(np.median(scores)),
                "sensors_scored": int(len(scores)),
            })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        raise ValueError(
            f"No monitoring windows were scored for scenario '{scenario}'."
        )

    return result_df


# ==================================================
# ENGINE-LEVEL CALIBRATION
# ==================================================

def calibrate_threshold(calibration_scores, alpha=0.10):
    """
    Calibrate using one maximum window score per calibration engine.

    With 16 calibration engines and alpha=0.10, the finite-sample rank
    is ceil((16 + 1) * 0.90) = 16, i.e. the maximum calibration score.
    """
    engine_scores = (
        calibration_scores.groupby("engine_id")
        .agg(
            calibration_score=("aggregate_score", "max"),
            windows=("aggregate_score", "size"),
        )
        .reset_index()
    )

    values = engine_scores["calibration_score"].to_numpy(dtype=float)
    n = len(values)

    if n == 0:
        raise ValueError("No calibration engine scores were generated.")

    rank = math.ceil((n + 1) * (1 - alpha))

    if rank > n:
        threshold = float("inf")
    else:
        threshold = float(np.sort(values)[rank - 1])

    return threshold, engine_scores, rank


# ==================================================
# SYNTHETIC DRIFT INJECTION
# ==================================================

def inject_drift(
    validation_data,
    sensor_features,
    selected_sensors,
    shift_level,
    profile_data,
    affected_engine_ids,
):
    simulated = validation_data.copy()

    simulated[sensor_features] = (
        simulated[sensor_features].astype(float)
    )

    affected_engine_ids = set(int(x) for x in affected_engine_ids)

    injection_mask = (
        simulated["engine_id"].isin(affected_engine_ids)
        & (simulated["cycle"] >= INJECTION_START_CYCLE)
    )

    for sensor in selected_sensors:
        sensor_std = float(profile_data[sensor].std())
        shift_amount = float(shift_level * sensor_std)

        simulated.loc[injection_mask, sensor] = (
            simulated.loc[injection_mask, sensor] + shift_amount
        )

    return simulated


# ==================================================
# MAIN EXPERIMENT
# ==================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PROGNOSAI - CALIBRATED SENSOR DRIFT")
    print("=" * 65)

    (
        predictions,
        sensor_features,
        profile_engines,
        calibration_engines,
        validation_engines,
    ) = load_and_predict()

    profile_data = predictions[
        predictions["engine_id"].isin(profile_engines)
    ].copy()

    calibration_data = predictions[
        predictions["engine_id"].isin(calibration_engines)
    ].copy()

    validation_data = predictions[
        predictions["engine_id"].isin(validation_engines)
    ].copy()

    # Choose the five highest-variance sensors from reference engines.
    selected_sensors = (
        profile_data[sensor_features]
        .std()
        .sort_values(ascending=False)
        .head(TOP_SENSORS_TO_SHIFT)
        .index.tolist()
    )

    # Fixed affected/control assignment, reused across all floors and shifts.
    validation_engine_list = sorted(int(x) for x in validation_engines)
    rng = np.random.default_rng(RANDOM_STATE)
    affected_engines = sorted(
        rng.choice(
            validation_engine_list,
            size=AFFECTED_ENGINE_COUNT,
            replace=False,
        ).astype(int).tolist()
    )
    control_engines = sorted(
        set(validation_engine_list) - set(affected_engines)
    )

    print("\nSYNTHETIC EXPERIMENT GROUPS")
    print("Affected engines:", affected_engines)
    print("Control engines:", control_engines)
    print("Selected sensors:", selected_sensors)
    print(
        f"Injection starts at cycle {INJECTION_START_CYCLE}; "
        f"{TOP_K_SCORE}-largest sensor scores are aggregated."
    )

    all_results = []
    threshold_sensitivity = []

    for floor in VARIABILITY_FLOORS:
        print("\n" + "=" * 60)
        print(f"VARIABILITY FLOOR: {floor}")
        print("=" * 60)

        profiles = build_profiles(
            predictions,
            sensor_features,
            profile_engines,
            floor,
        )

        profiles.to_csv(
            OUTPUT_DIR / f"sensor_profiles_floor_{floor}.csv",
            index=False,
        )

        calibration_scores = score_windows(
            calibration_data,
            profiles,
            sensor_features,
            scenario=f"calibration_floor_{floor}",
        )

        validation_scores = score_windows(
            validation_data,
            profiles,
            sensor_features,
            scenario=f"normal_validation_floor_{floor}",
        )

        threshold, engine_scores, rank = calibrate_threshold(
            calibration_scores,
            ALPHA,
        )

        engine_scores.to_csv(
            OUTPUT_DIR / f"calibration_engine_scores_floor_{floor}.csv",
            index=False,
        )

        normal_scored, normal_flags = engine_flag_table(
            validation_scores,
            threshold,
        )

        normal_alarm_rate = float(
            normal_flags["drift_detected"].mean() * 100.0
        )

        normal_scored.to_csv(
            OUTPUT_DIR / f"normal_validation_scores_floor_{floor}.csv",
            index=False,
        )
        normal_flags.to_csv(
            OUTPUT_DIR / f"normal_validation_flags_floor_{floor}.csv",
            index=False,
        )

        print("Profile sensors:", len(sensor_features))
        print("Calibration engines:", len(engine_scores))
        print("Calibration rank:", rank)
        print("Calibrated threshold:", threshold)
        print(
            "Normal validation engine alarm rate:",
            f"{normal_alarm_rate:.2f}%",
        )

        threshold_sensitivity.append({
            "variability_floor": float(floor),
            "calibration_engines": int(len(engine_scores)),
            "calibration_rank": int(rank),
            "threshold": (
                float(threshold) if np.isfinite(threshold) else None
            ),
            "normal_validation_engine_alarm_percent": normal_alarm_rate,
        })

        for shift_level in SHIFT_LEVELS:
            simulated = inject_drift(
                validation_data=validation_data,
                sensor_features=sensor_features,
                selected_sensors=selected_sensors,
                shift_level=shift_level,
                profile_data=profile_data,
                affected_engine_ids=affected_engines,
            )

            simulated_scores = score_windows(
                simulated,
                profiles,
                sensor_features,
                scenario=f"floor_{floor}_shift_{shift_level}",
            )

            simulated_scored, simulated_flags = engine_flag_table(
                simulated_scores,
                threshold,
            )

            affected_detection = percentage_flagged(
                simulated_flags,
                affected_engines,
            )
            control_alarm = percentage_flagged(
                simulated_flags,
                control_engines,
            )
            overall_alarm = float(
                simulated_flags["drift_detected"].mean() * 100.0
            )

            result_row = {
                "variability_floor": float(floor),
                "shift_level_std": float(shift_level),
                "affected_engine_detection_percent": affected_detection,
                "control_engine_false_alarm_percent": control_alarm,
                "overall_engine_alarm_percent": overall_alarm,
                "normal_validation_engine_alarm_percent": normal_alarm_rate,
                "threshold": (
                    float(threshold) if np.isfinite(threshold) else None
                ),
                "monitoring_windows": int(len(simulated_scores)),
                "affected_engine_count": int(len(affected_engines)),
                "control_engine_count": int(len(control_engines)),
            }
            all_results.append(result_row)

            simulated_scored.to_csv(
                OUTPUT_DIR
                / f"simulation_floor_{floor}_shift_{shift_level}.csv",
                index=False,
            )
            simulated_flags.to_csv(
                OUTPUT_DIR
                / f"simulation_engine_flags_floor_{floor}_shift_{shift_level}.csv",
                index=False,
            )

            print(
                f"Shift {shift_level:.2f} SD | "
                f"Affected detection: {affected_detection:.2f}% | "
                f"Control alarms: {control_alarm:.2f}% | "
                f"Overall: {overall_alarm:.2f}%"
            )

    results_df = pd.DataFrame(all_results)
    sensitivity_df = pd.DataFrame(threshold_sensitivity)

    results_df.to_csv(
        OUTPUT_DIR / "simulation_results.csv",
        index=False,
    )
    sensitivity_df.to_csv(
        OUTPUT_DIR / "threshold_sensitivity.csv",
        index=False,
    )

    summary = {
        "experiment": "Calibrated robust sensor drift detection",
        "dataset": "NASA C-MAPSS FD001",
        "stage_assignment": "Frozen baseline predicted RUL",
        "profile_engines": sorted(int(x) for x in profile_engines),
        "calibration_engines": sorted(int(x) for x in calibration_engines),
        "validation_engines": sorted(int(x) for x in validation_engines),
        "affected_engines": affected_engines,
        "control_engines": control_engines,
        "sensor_features": sensor_features,
        "selected_injection_sensors": selected_sensors,
        "variability_floors": VARIABILITY_FLOORS,
        "alpha": ALPHA,
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "injection_start_cycle": INJECTION_START_CYCLE,
        "shift_levels": SHIFT_LEVELS,
        "score_aggregation": (
            f"Mean of the {TOP_K_SCORE} largest absolute "
            "standardized sensor deviations per window"
        ),
        "threshold_sensitivity": sensitivity_df.to_dict(orient="records"),
        "simulation_results": results_df.to_dict(orient="records"),
        "limitations": [
            "Only 16 engines are available for threshold calibration.",
            "The finite-sample calibration threshold is the maximum of 16 engine scores at alpha=0.10.",
            "The 20 validation engines form a small cohort.",
            "Sensor profiles pool observations from different reference engines.",
            "Synthetic shifts do not represent all real industrial faults.",
            "The injected shift affects only the selected five sensors and only after the configured cycle.",
            "The experiment does not establish production false-alarm or coverage guarantees.",
            "The top-five aggregation is a design choice and should be compared with alternative scoring rules.",
        ],
    }

    report_path = OUTPUT_DIR / "calibrated_sensor_drift_metrics.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(summary),
            file,
            indent=4,
            allow_nan=False,
        )

    print("\n" + "=" * 65)
    print("FINAL THRESHOLD COMPARISON")
    print("=" * 65)
    print(sensitivity_df.round(3).to_string(index=False))

    print("\nSYNTHETIC DRIFT RESULTS")
    print(
        results_df[
            [
                "variability_floor",
                "shift_level_std",
                "affected_engine_detection_percent",
                "control_engine_false_alarm_percent",
                "overall_engine_alarm_percent",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nArtifacts saved to:")
    print(OUTPUT_DIR)
    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
