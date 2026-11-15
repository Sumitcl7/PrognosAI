"""
PrognosAI - Repeated-Split Robustness for Calibrated Sensor Drift Detectors

Compares:
1) Calibrated abrupt sensor mean-shift detector
2) Stage-homogeneous sensor variance-increase detector

Research design:
- Frozen baseline Random Forest; never retrained in this experiment.
- Fixed 20-engine validation cohort from baseline/validation_predictions.csv.
- For each of five seeds, repartition the other 80 engines into:
    64 profile engines, 16 calibration engines.
- Same affected/control validation engines across all seeds.
- Recalibrate each detector independently for each split.
- This is split-sensitivity analysis, NOT five independent external validations.

No existing files or artifacts are overwritten. Output is written to:
artifacts/drift/sensor_drift_repeated_split/
"""

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "processed" / "FD001_train_with_RUL.csv"
MODEL_FILE = ROOT / "artifacts" / "baseline" / "random_forest_fd001.joblib"
METRICS_FILE = ROOT / "artifacts" / "baseline" / "metrics.json"
VALIDATION_FILE = ROOT / "artifacts" / "baseline" / "validation_predictions.csv"

OUTPUT_DIR = ROOT / "artifacts" / "drift" / "sensor_drift_repeated_split"

SPLIT_SEEDS = [42, 7, 21, 100, 2026]
AFFECTED_SELECTION_SEED = 42
AFFECTED_ENGINE_COUNT = 10

ALPHA = 0.10
WINDOW_SIZE = 20
WINDOW_STEP = 10
INJECTION_START_CYCLE = 80

STAGE_BINS = [0, 30, 60, 90, 125]
STAGE_LABELS = ["Critical", "Late-middle", "Middle", "Early"]

# Mean detector settings, matching calibrated_sensor_drift.py.
MEAN_VARIABILITY_FLOOR = 0.05
TOP_K_MEAN_SCORE = 5
MEAN_SHIFT_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5]

# Variance detector settings, matching sensor_variance_drift.py.
VARIANCE_MULTIPLIERS = [1.25, 1.50, 2.00, 3.00]
EPSILON = 1e-8
MIN_STAGE_REFERENCE_ENGINES = 10
TOP_VARIANCE_SENSORS = 5


# ---------------------------------------------------------------------
# COMMON UTILITIES
# ---------------------------------------------------------------------

def assign_predicted_stage(frame):
    result = frame.copy()
    result["predicted_RUL_clipped"] = result["predicted_RUL"].clip(0, 125)
    result["predicted_stage"] = pd.cut(
        result["predicted_RUL_clipped"],
        bins=STAGE_BINS,
        labels=STAGE_LABELS,
        include_lowest=True,
    )
    result = result.dropna(subset=["predicted_stage"]).copy()
    result["predicted_stage"] = result["predicted_stage"].astype(str)
    return result


def finite_sample_threshold(window_scores, alpha=ALPHA):
    """Calibrate on one maximum window score per calibration engine."""
    if window_scores.empty:
        raise ValueError("No calibration windows were scored.")

    engine_scores = (
        window_scores.groupby("engine_id")
        .agg(
            calibration_score=("aggregate_score", "max"),
            calibration_windows=("aggregate_score", "size"),
        )
        .reset_index()
    )

    values = engine_scores["calibration_score"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)

    if n == 0:
        raise ValueError("No finite calibration engine scores.")

    rank = math.ceil((n + 1) * (1 - alpha))
    threshold = float("inf") if rank > n else float(np.sort(values)[rank - 1])
    return threshold, engine_scores, rank


def flag_engines(window_scores, threshold, expected_engine_ids):
    if window_scores.empty:
        raise ValueError("No monitoring windows were scored.")

    scored = window_scores.copy()
    scored["window_alarm"] = scored["aggregate_score"] > threshold

    flags = (
        scored.groupby("engine_id")
        .agg(
            drift_detected=("window_alarm", "any"),
            maximum_score=("aggregate_score", "max"),
            windows_scored=("aggregate_score", "size"),
        )
        .reset_index()
    )

    expected = set(int(x) for x in expected_engine_ids)
    observed = set(flags["engine_id"].astype(int))
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Engines have no scored windows: {missing}")

    return scored, flags


def flag_rate(flags, engine_ids):
    ids = set(int(x) for x in engine_ids)
    subset = flags[flags["engine_id"].astype(int).isin(ids)]
    if len(subset) != len(ids):
        missing = sorted(ids - set(subset["engine_id"].astype(int)))
        raise ValueError(f"Missing engine flags for: {missing}")
    return float(subset["drift_detected"].mean() * 100.0)


def save_json(path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return [clean(v) for v in value.tolist()]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(clean(payload), handle, indent=4, allow_nan=False)


# ---------------------------------------------------------------------
# DATA LOADING: FROZEN BASELINE ONLY
# ---------------------------------------------------------------------

def load_project_data():
    required_paths = [DATA_FILE, MODEL_FILE, METRICS_FILE, VALIDATION_FILE]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required project file not found: {path}")

    data = pd.read_csv(DATA_FILE)
    with open(METRICS_FILE, "r", encoding="utf-8") as handle:
        baseline_metadata = json.load(handle)

    features = baseline_metadata.get("features")
    if not features:
        raise ValueError("Baseline metrics.json does not contain 'features'.")

    sensor_features = [name for name in features if name.startswith("sensor_")]
    if not sensor_features:
        raise ValueError("No sensor features found in baseline metadata.")

    required_columns = ["engine_id", "cycle", "RUL_capped"] + features
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {missing}")

    original_validation = pd.read_csv(VALIDATION_FILE)
    if "engine_id" not in original_validation.columns:
        raise ValueError("validation_predictions.csv must contain engine_id.")

    validation_ids = sorted(
        original_validation["engine_id"].astype(int).unique().tolist()
    )
    all_ids = sorted(data["engine_id"].astype(int).unique().tolist())
    profile_calibration_pool = sorted(set(all_ids) - set(validation_ids))

    if len(validation_ids) != 20 or len(profile_calibration_pool) != 80:
        raise ValueError(
            "Expected 20 frozen validation engines and 80 profile/calibration "
            f"pool engines; found {len(validation_ids)} and "
            f"{len(profile_calibration_pool)}."
        )

    data = data.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    model = joblib.load(MODEL_FILE)

    predictions = data[["engine_id", "cycle", "RUL_capped"] + sensor_features].copy()
    predictions[sensor_features] = predictions[sensor_features].astype(float)
    predictions["predicted_RUL"] = model.predict(data[features])
    predictions = assign_predicted_stage(predictions)

    validation = predictions[
        predictions["engine_id"].astype(int).isin(validation_ids)
    ].copy()

    mae = mean_absolute_error(validation["RUL_capped"], validation["predicted_RUL"])
    rmse = np.sqrt(
        mean_squared_error(validation["RUL_capped"], validation["predicted_RUL"])
    )
    r2 = r2_score(validation["RUL_capped"], validation["predicted_RUL"])

    rng = np.random.default_rng(AFFECTED_SELECTION_SEED)
    affected_ids = sorted(
        rng.choice(validation_ids, size=AFFECTED_ENGINE_COUNT, replace=False)
        .astype(int).tolist()
    )
    control_ids = sorted(set(validation_ids) - set(affected_ids))

    print("\nFROZEN BASELINE ON FIXED VALIDATION COHORT")
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
    print(f"Validation engines: {len(validation_ids)}")
    print(f"Affected engines (fixed across splits): {affected_ids}")
    print(f"Control engines (fixed across splits): {control_ids}")

    return (
        predictions,
        features,
        sensor_features,
        profile_calibration_pool,
        validation_ids,
        affected_ids,
        control_ids,
        {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)},
    )


# ---------------------------------------------------------------------
# MEAN-SHIFT DETECTOR: CALIBRATED SENSOR DRIFT METHOD
# ---------------------------------------------------------------------

def build_mean_profiles(profile_data, sensor_features):
    rows = []
    for stage in STAGE_LABELS:
        stage_data = profile_data[profile_data["predicted_stage"] == stage]
        if stage_data.empty:
            raise ValueError(f"No profile observations for stage {stage}.")

        for sensor in sensor_features:
            values = stage_data[sensor].dropna().to_numpy(dtype=float)
            if len(values) < 30:
                raise ValueError(
                    f"Insufficient profile observations for {stage}/{sensor}."
                )
            median = float(np.median(values))
            mad_value = float(np.median(np.abs(values - median)))
            robust_scale = 1.4826 * mad_value
            effective_scale = max(robust_scale, MEAN_VARIABILITY_FLOOR)
            rows.append({
                "predicted_stage": stage,
                "sensor": sensor,
                "reference_count": int(len(values)),
                "reference_engines": int(stage_data["engine_id"].nunique()),
                "reference_median": median,
                "reference_mad": mad_value,
                "effective_scale": float(effective_scale),
            })
    return pd.DataFrame(rows)


def score_mean_windows(frame, profiles, sensor_features, scenario):
    lookup = {
        (row.predicted_stage, row.sensor): row
        for row in profiles.itertuples(index=False)
    }
    results = []

    for engine_id, engine_data in frame.groupby("engine_id"):
        engine_data = engine_data.sort_values("cycle").reset_index(drop=True)
        min_cycle = int(engine_data["cycle"].min())
        max_cycle = int(engine_data["cycle"].max())

        for start in range(
            min_cycle, max_cycle - WINDOW_SIZE + 2, WINDOW_STEP
        ):
            end = start + WINDOW_SIZE - 1
            window = engine_data[
                (engine_data["cycle"] >= start) & (engine_data["cycle"] <= end)
            ]
            if len(window) < WINDOW_SIZE:
                continue

            stage_mode = window["predicted_stage"].mode()
            if stage_mode.empty:
                continue
            stage = str(stage_mode.iloc[0])

            sensor_scores = []
            for sensor in sensor_features:
                profile = lookup.get((stage, sensor))
                if profile is None:
                    continue
                current_median = float(window[sensor].median())
                deviation = current_median - float(profile.reference_median)
                standardized = deviation / max(float(profile.effective_scale), 1e-12)
                sensor_scores.append(abs(standardized))

            if not sensor_scores:
                continue
            top_k = min(TOP_K_MEAN_SCORE, len(sensor_scores))
            aggregate = float(np.mean(np.sort(sensor_scores)[-top_k:]))

            results.append({
                "scenario": scenario,
                "engine_id": int(engine_id),
                "window_start": int(start),
                "window_end": int(end),
                "predicted_stage": stage,
                "aggregate_score": aggregate,
            })

    result = pd.DataFrame(results)
    if result.empty:
        raise ValueError(f"No mean-shift windows scored for {scenario}.")
    return result


def inject_mean_shift(validation_data, selected_sensors, profile_data,
                      affected_ids, shift_level):
    simulated = validation_data.copy()
    simulated[selected_sensors] = simulated[selected_sensors].astype(float)
    mask = (
        simulated["engine_id"].astype(int).isin(set(affected_ids))
        & (simulated["cycle"] >= INJECTION_START_CYCLE)
    )

    for sensor in selected_sensors:
        sensor_std = float(profile_data[sensor].std())
        shift_amount = float(shift_level * sensor_std)
        simulated.loc[mask, sensor] = (
            simulated.loc[mask, sensor] + shift_amount
        )
    return simulated


# ---------------------------------------------------------------------
# VARIANCE DETECTOR: STAGE-HOMOGENEOUS MAD METHOD
# ---------------------------------------------------------------------

def mad(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def stage_windows(engine_data):
    ordered = engine_data.sort_values("cycle")
    for stage, stage_data in ordered.groupby("predicted_stage", sort=False):
        stage_data = stage_data.sort_values("cycle").reset_index(drop=True)
        if len(stage_data) < WINDOW_SIZE:
            continue
        for start_idx in range(0, len(stage_data) - WINDOW_SIZE + 1, WINDOW_STEP):
            window = stage_data.iloc[start_idx:start_idx + WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                yield str(stage), window


def select_variance_sensors(profile_data, sensor_features):
    pooled_std = profile_data[sensor_features].astype(float).std()
    eligible = pooled_std[pooled_std > EPSILON].sort_values(ascending=False)
    selected = eligible.head(TOP_VARIANCE_SENSORS).index.tolist()
    if len(selected) != TOP_VARIANCE_SENSORS:
        raise ValueError("Could not select five non-constant variance sensors.")
    return selected


def build_variance_profiles(profile_data, selected_sensors):
    # First aggregate within each engine, then take the median across engines.
    engine_values = {}
    for engine_id, engine_data in profile_data.groupby("engine_id"):
        stage_sensor_values = {}
        for stage, window in stage_windows(engine_data):
            for sensor in selected_sensors:
                value = mad(window[sensor].to_numpy(dtype=float))
                if np.isfinite(value):
                    stage_sensor_values.setdefault((stage, sensor), []).append(value)

        for key, values in stage_sensor_values.items():
            if values:
                engine_values.setdefault(key, []).append(float(np.median(values)))

    rows = []
    for stage in STAGE_LABELS:
        for sensor in selected_sensors:
            values = engine_values.get((stage, sensor), [])
            if len(values) < MIN_STAGE_REFERENCE_ENGINES:
                raise ValueError(
                    f"Insufficient reference engines for {stage}/{sensor}: "
                    f"{len(values)}"
                )
            reference_mad = float(np.median(values))
            if reference_mad <= EPSILON:
                raise ValueError(
                    f"Reference MAD is zero/near-zero for {stage}/{sensor}."
                )
            rows.append({
                "predicted_stage": stage,
                "sensor": sensor,
                "reference_engines": len(values),
                "reference_mad": reference_mad,
            })
    return pd.DataFrame(rows)


def score_variance_windows(frame, profiles, selected_sensors, scenario):
    lookup = {
        (str(row.predicted_stage), row.sensor): float(row.reference_mad)
        for row in profiles.itertuples(index=False)
    }
    results = []

    for engine_id, engine_data in frame.groupby("engine_id"):
        for stage, window in stage_windows(engine_data):
            scores = []
            for sensor in selected_sensors:
                reference_mad = lookup[(stage, sensor)]
                current_mad = mad(window[sensor].to_numpy(dtype=float))
                score = max(
                    0.0,
                    float(np.log(
                        max(current_mad, EPSILON) /
                        max(reference_mad, EPSILON)
                    )),
                )
                scores.append(score)

            if scores:
                results.append({
                    "scenario": scenario,
                    "engine_id": int(engine_id),
                    "window_start": int(window["cycle"].min()),
                    "window_end": int(window["cycle"].max()),
                    "predicted_stage": stage,
                    "aggregate_score": float(np.mean(scores)),
                })

    result = pd.DataFrame(results)
    if result.empty:
        raise ValueError(f"No variance windows scored for {scenario}.")
    return result


def inject_variance_shift(validation_data, selected_sensors, affected_ids,
                          multiplier):
    simulated = validation_data.copy()
    simulated[selected_sensors] = simulated[selected_sensors].astype(float)
    affected = set(int(x) for x in affected_ids)
    injection_mask = (
        simulated["engine_id"].astype(int).isin(affected)
        & (simulated["cycle"] >= INJECTION_START_CYCLE)
    )

    for (engine_id, stage), index in (
        simulated.loc[injection_mask]
        .groupby(["engine_id", "predicted_stage"])
        .groups.items()
    ):
        for sensor in selected_sensors:
            values = simulated.loc[index, sensor].to_numpy(dtype=float)
            center = float(np.median(values))
            simulated.loc[index, sensor] = (
                center + (values - center) * float(multiplier)
            )
    return simulated


# ---------------------------------------------------------------------
# RUN ONE SPLIT
# ---------------------------------------------------------------------

def run_one_split(seed, predictions, pool_ids, validation_ids,
                  affected_ids, control_ids, sensor_features):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(pool_ids)
    profile_ids = sorted(shuffled[:64].astype(int).tolist())
    calibration_ids = sorted(shuffled[64:].astype(int).tolist())

    profile_data = predictions[
        predictions["engine_id"].astype(int).isin(profile_ids)
    ].copy()
    calibration_data = predictions[
        predictions["engine_id"].astype(int).isin(calibration_ids)
    ].copy()
    validation_data = predictions[
        predictions["engine_id"].astype(int).isin(validation_ids)
    ].copy()

    print("\n" + "=" * 76)
    print(f"SPLIT SEED {seed}: profile={len(profile_ids)}, "
          f"calibration={len(calibration_ids)}, validation={len(validation_ids)}")
    print("=" * 76)

    split_rows = []

    # ----- Mean-shift detector -----
    mean_profiles = build_mean_profiles(profile_data, sensor_features)
    mean_selected = (
        profile_data[sensor_features].astype(float).std()
        .sort_values(ascending=False)
        .head(5).index.tolist()
    )
    mean_cal_scores = score_mean_windows(
        calibration_data, mean_profiles, sensor_features, "calibration"
    )
    mean_threshold, mean_cal_engine_scores, mean_rank = finite_sample_threshold(
        mean_cal_scores
    )
    mean_normal_scores = score_mean_windows(
        validation_data, mean_profiles, sensor_features, "normal_validation"
    )
    _, mean_normal_flags = flag_engines(
        mean_normal_scores, mean_threshold, validation_ids
    )
    mean_normal_rate = flag_rate(mean_normal_flags, validation_ids)

    print(f"MEAN detector | threshold={mean_threshold:.6f} "
          f"| normal alarms={mean_normal_rate:.1f}% "
          f"| sensors={mean_selected}")

    for level in MEAN_SHIFT_LEVELS:
        simulated = inject_mean_shift(
            validation_data, mean_selected, profile_data, affected_ids, level
        )
        scores = score_mean_windows(
            simulated, mean_profiles, sensor_features, f"mean_shift_{level}"
        )
        _, flags = flag_engines(scores, mean_threshold, validation_ids)
        row = {
            "seed": seed,
            "detector": "calibrated_mean_shift",
            "scenario_level": float(level),
            "scenario_unit": "standard_deviations",
            "threshold": mean_threshold,
            "calibration_rank": mean_rank,
            "calibration_engine_count": len(mean_cal_engine_scores),
            "normal_validation_alarm_percent": mean_normal_rate,
            "affected_detection_percent": flag_rate(flags, affected_ids),
            "control_false_alarm_percent": flag_rate(flags, control_ids),
            "affected_engines_flagged": int(
                flags[flags["engine_id"].isin(affected_ids)]["drift_detected"].sum()
            ),
            "control_engines_flagged": int(
                flags[flags["engine_id"].isin(control_ids)]["drift_detected"].sum()
            ),
            "profile_engines": len(profile_ids),
            "selected_sensors": ";".join(mean_selected),
        }
        split_rows.append(row)
        print(f"  Mean shift {level:>4.2f} SD | "
              f"affected={row['affected_detection_percent']:>5.1f}% | "
              f"controls={row['control_false_alarm_percent']:>5.1f}%")

    # ----- Variance detector -----
    variance_selected = select_variance_sensors(profile_data, sensor_features)
    variance_profiles = build_variance_profiles(profile_data, variance_selected)
    variance_cal_scores = score_variance_windows(
        calibration_data, variance_profiles, variance_selected, "calibration"
    )
    variance_threshold, variance_cal_engine_scores, variance_rank = (
        finite_sample_threshold(variance_cal_scores)
    )
    variance_normal_scores = score_variance_windows(
        validation_data, variance_profiles, variance_selected, "normal_validation"
    )
    _, variance_normal_flags = flag_engines(
        variance_normal_scores, variance_threshold, validation_ids
    )
    variance_normal_rate = flag_rate(variance_normal_flags, validation_ids)

    print(f"VARIANCE detector | threshold={variance_threshold:.6f} "
          f"| normal alarms={variance_normal_rate:.1f}% "
          f"| sensors={variance_selected}")

    for multiplier in VARIANCE_MULTIPLIERS:
        simulated = inject_variance_shift(
            validation_data, variance_selected, affected_ids, multiplier
        )
        scores = score_variance_windows(
            simulated, variance_profiles, variance_selected,
            f"variance_multiplier_{multiplier}"
        )
        _, flags = flag_engines(scores, variance_threshold, validation_ids)
        row = {
            "seed": seed,
            "detector": "stage_homogeneous_variance",
            "scenario_level": float(multiplier),
            "scenario_unit": "variance_multiplier",
            "threshold": variance_threshold,
            "calibration_rank": variance_rank,
            "calibration_engine_count": len(variance_cal_engine_scores),
            "normal_validation_alarm_percent": variance_normal_rate,
            "affected_detection_percent": flag_rate(flags, affected_ids),
            "control_false_alarm_percent": flag_rate(flags, control_ids),
            "affected_engines_flagged": int(
                flags[flags["engine_id"].isin(affected_ids)]["drift_detected"].sum()
            ),
            "control_engines_flagged": int(
                flags[flags["engine_id"].isin(control_ids)]["drift_detected"].sum()
            ),
            "profile_engines": len(profile_ids),
            "selected_sensors": ";".join(variance_selected),
        }
        split_rows.append(row)
        print(f"  Variance {multiplier:>4.2f}x | "
              f"affected={row['affected_detection_percent']:>5.1f}% | "
              f"controls={row['control_false_alarm_percent']:>5.1f}%")

    split_metadata = {
        "seed": seed,
        "profile_engine_ids": profile_ids,
        "calibration_engine_ids": calibration_ids,
        "validation_engine_ids": sorted(int(x) for x in validation_ids),
        "affected_engine_ids": sorted(int(x) for x in affected_ids),
        "control_engine_ids": sorted(int(x) for x in control_ids),
        "mean_selected_sensors": mean_selected,
        "variance_selected_sensors": variance_selected,
        "mean_threshold": mean_threshold,
        "mean_calibration_rank": mean_rank,
        "variance_threshold": variance_threshold,
        "variance_calibration_rank": variance_rank,
    }
    return split_rows, split_metadata


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("PROGNOSAI - REPEATED-SPLIT SENSOR DRIFT ROBUSTNESS")
    print("=" * 76)
    print("Frozen baseline; no model retraining; existing artifacts untouched.")

    (
        predictions,
        features,
        sensor_features,
        pool_ids,
        validation_ids,
        affected_ids,
        control_ids,
        baseline_metrics,
    ) = load_project_data()

    all_rows = []
    split_metadata = []

    for seed in SPLIT_SEEDS:
        rows, metadata = run_one_split(
            seed,
            predictions,
            pool_ids,
            validation_ids,
            affected_ids,
            control_ids,
            sensor_features,
        )
        all_rows.extend(rows)
        split_metadata.append(metadata)

    results = pd.DataFrame(all_rows)
    results.to_csv(OUTPUT_DIR / "repeated_split_detector_results.csv", index=False)

    summary = (
        results.groupby(["detector", "scenario_unit", "scenario_level"])
        .agg(
            mean_detection_percent=("affected_detection_percent", "mean"),
            std_detection_percent=("affected_detection_percent", "std"),
            min_detection_percent=("affected_detection_percent", "min"),
            max_detection_percent=("affected_detection_percent", "max"),
            mean_control_false_alarm_percent=("control_false_alarm_percent", "mean"),
            max_control_false_alarm_percent=("control_false_alarm_percent", "max"),
            mean_normal_alarm_percent=("normal_validation_alarm_percent", "mean"),
            max_normal_alarm_percent=("normal_validation_alarm_percent", "max"),
            mean_threshold=("threshold", "mean"),
            std_threshold=("threshold", "std"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "repeated_split_detector_summary.csv", index=False)

    split_table = pd.DataFrame([
        {
            "seed": item["seed"],
            "mean_threshold": item["mean_threshold"],
            "mean_calibration_rank": item["mean_calibration_rank"],
            "variance_threshold": item["variance_threshold"],
            "variance_calibration_rank": item["variance_calibration_rank"],
            "mean_selected_sensors": ";".join(item["mean_selected_sensors"]),
            "variance_selected_sensors": ";".join(item["variance_selected_sensors"]),
        }
        for item in split_metadata
    ])
    split_table.to_csv(OUTPUT_DIR / "split_calibration_details.csv", index=False)

    save_json(
        OUTPUT_DIR / "repeated_split_detector_metrics.json",
        {
            "experiment": "Repeated-split robustness of calibrated sensor mean and variance drift detectors",
            "dataset": "NASA C-MAPSS FD001",
            "split_seeds": SPLIT_SEEDS,
            "profile_engines_per_split": 64,
            "calibration_engines_per_split": 16,
            "fixed_validation_engines": sorted(int(x) for x in validation_ids),
            "fixed_affected_engines": sorted(int(x) for x in affected_ids),
            "fixed_control_engines": sorted(int(x) for x in control_ids),
            "frozen_baseline_validation_metrics": baseline_metrics,
            "baseline_model_retrained": False,
            "mean_detector": {
                "profile": "Stage-conditioned pooled median and MAD across profile-engine observations",
                "score": "Mean of the five largest absolute standardized sensor-median deviations per 20-cycle window",
                "injection": "Persistent additive mean shift after cycle 80, scaled by profile sensor standard deviation",
                "shift_levels_sd": MEAN_SHIFT_LEVELS,
                "variability_floor": MEAN_VARIABILITY_FLOOR,
            },
            "variance_detector": {
                "profile": "Stage-homogeneous 20-observation windows; per-engine median window MAD, then median across profile engines",
                "score": "Mean across five selected sensors of max(0, log(current window MAD / reference stage-sensor MAD))",
                "injection": "Post-cycle-80 dispersion multiplier around each affected engine's own median within predicted stage",
                "variance_multipliers": VARIANCE_MULTIPLIERS,
            },
            "calibration": {
                "alpha": ALPHA,
                "method": "Finite-sample quantile of per-calibration-engine maximum window score",
                "expected_calibration_engines": 16,
                "expected_rank": 16,
            },
            "split_details": split_metadata,
            "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
            "limitations": [
                "Five partitions reuse the same 20 held-out validation engines; this is split-sensitivity analysis, not five independent external validations.",
                "The affected and control engine groups are fixed across splits to isolate profile/calibration partition sensitivity.",
                "Only 16 calibration engines are available per split; thresholds are exploratory and finite-sample quantiles are coarse.",
                "Synthetic shifts and variance multipliers do not establish real industrial fault performance.",
                "The frozen baseline predicted RUL and predicted-stage assignments are not recomputed after synthetic sensor perturbations.",
                "Mean and variance detectors use different window construction and scoring definitions; their sensitivity figures should be compared within their respective injected-shift scenarios, not as directly equivalent effect sizes.",
                "Zero observed control alarms in a small cohort is not a guarantee of zero population false-alarm probability.",
            ],
        },
    )

    print("\n" + "=" * 76)
    print("REPEATED-SPLIT SUMMARY")
    print("=" * 76)
    print(summary.round(3).to_string(index=False))
    print("\nArtifacts saved to:")
    print(OUTPUT_DIR)
    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
