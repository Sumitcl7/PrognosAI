"""
PrognosAI - Stage-Homogeneous Sensor Variance Drift Experiment

Independent experiment. Does not overwrite:
- calibrated_sensor_drift.py
- sensor_drift_robustness.py
- frozen baseline model or prior artifacts

Design:
- Reuses the frozen baseline predictions and existing 64/16/20 engine split.
- Builds stage-homogeneous 20-observation reference windows from 64 profile engines.
- Uses the five highest-variability sensors selected from profile engines.
- Calibrates an engine-level threshold on 16 calibration engines.
- Evaluates the same 20 held-out validation engines: 10 affected, 10 controls.
- Injects variance changes around each affected engine's own post-injection,
  predicted-stage median, preserving the local median by construction.
- Uses a one-sided log-MAD ratio: decreases in dispersion do not count as
  evidence of increased variance.

This is a synthetic sensitivity experiment, not real-world fault validation.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.drift import calibrated_sensor_drift as base


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

OUTPUT_DIR = (
    base.ROOT / "artifacts" / "drift" / "sensor_variance_drift"
)

WINDOW_SIZE = int(base.WINDOW_SIZE)
WINDOW_STEP = int(base.WINDOW_STEP)
INJECTION_START_CYCLE = int(base.INJECTION_START_CYCLE)

VARIANCE_MULTIPLIERS = [1.25, 1.50, 2.00, 3.00]
AFFECTED_ENGINE_COUNT = 10
ALPHA = float(base.ALPHA)
RANDOM_STATE = int(base.RANDOM_STATE)
EPSILON = 1e-8
MIN_STAGE_REFERENCE_ENGINES = 10


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def mad(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def stage_windows(engine_data):
    """Yield non-overlapping-stage-mixed rolling windows."""
    ordered = engine_data.sort_values("cycle")

    for stage, stage_data in ordered.groupby("predicted_stage", sort=False):
        stage_data = stage_data.sort_values("cycle").reset_index(drop=True)

        if len(stage_data) < WINDOW_SIZE:
            continue

        for start_idx in range(
            0, len(stage_data) - WINDOW_SIZE + 1, WINDOW_STEP
        ):
            window = stage_data.iloc[start_idx:start_idx + WINDOW_SIZE]

            if len(window) != WINDOW_SIZE:
                continue

            yield str(stage), window


def build_variance_profiles(profile_data, selected_sensors):
    """
    Estimate each stage/sensor reference MAD from engine-balanced
    stage-homogeneous windows.

    Each reference engine contributes one median window-MAD per stage/sensor;
    the final reference is the median across engines.
    """
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
    for stage in base.STAGE_LABELS:
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
                    f"Reference MAD is zero/near-zero for {stage}/{sensor}. "
                    "This sensor-stage profile cannot support a ratio score."
                )

            rows.append({
                "predicted_stage": stage,
                "sensor": sensor,
                "reference_engines": len(values),
                "reference_mad": reference_mad,
                "reference_mad_min": float(np.min(values)),
                "reference_mad_max": float(np.max(values)),
            })

    return pd.DataFrame(rows)


def score_windows(data, profiles, selected_sensors, scenario):
    """Score same-stage windows with a one-sided log-MAD ratio."""
    lookup = {
        (str(row.predicted_stage), row.sensor): float(row.reference_mad)
        for row in profiles.itertuples(index=False)
    }

    window_rows = []
    sensor_rows = []

    for engine_id, engine_data in data.groupby("engine_id"):
        for stage, window in stage_windows(engine_data):
            scores = []

            for sensor in selected_sensors:
                reference_mad = lookup[(stage, sensor)]
                current_mad = mad(window[sensor].to_numpy(dtype=float))

                # Directional score: only increased dispersion is drift evidence.
                score = max(
                    0.0,
                    float(np.log(max(current_mad, EPSILON) /
                                 max(reference_mad, EPSILON)))
                )

                scores.append(score)
                sensor_rows.append({
                    "scenario": scenario,
                    "engine_id": int(engine_id),
                    "window_start": int(window["cycle"].min()),
                    "window_end": int(window["cycle"].max()),
                    "predicted_stage": stage,
                    "sensor": sensor,
                    "reference_mad": reference_mad,
                    "current_mad": current_mad,
                    "mad_ratio": current_mad / reference_mad,
                    "sensor_score": score,
                })

            if not scores:
                continue

            window_rows.append({
                "scenario": scenario,
                "engine_id": int(engine_id),
                "window_start": int(window["cycle"].min()),
                "window_end": int(window["cycle"].max()),
                "predicted_stage": stage,
                "aggregate_score": float(np.mean(scores)),
                "max_sensor_score": float(np.max(scores)),
                "mean_sensor_score": float(np.mean(scores)),
                "sensors_scored": len(scores),
            })

    windows_df = pd.DataFrame(window_rows)
    sensors_df = pd.DataFrame(sensor_rows)

    if windows_df.empty:
        raise ValueError(f"No windows scored for scenario: {scenario}")

    return windows_df, sensors_df


def inject_variance_drift(
    validation_data,
    selected_sensors,
    affected_engine_ids,
    multiplier,
):
    """
    Scale deviations around each affected engine's own median within each
    predicted-stage/post-injection subset. This isolates dispersion changes
    more cleanly than centering on a pooled reference median.
    """
    simulated = validation_data.copy()
    simulated[selected_sensors] = simulated[selected_sensors].astype(float)

    affected = set(int(x) for x in affected_engine_ids)
    injection_mask = (
        simulated["engine_id"].isin(affected)
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


def engine_flags(scores, threshold):
    """Flag an engine if any of its stage-homogeneous windows exceeds threshold."""
    scored = scores.copy()
    scored["drift_detected"] = scored["aggregate_score"] > threshold

    flags = (
        scored.groupby("engine_id")
        .agg(
            drift_detected=("drift_detected", "any"),
            maximum_score=("aggregate_score", "max"),
            windows=("aggregate_score", "size"),
        )
        .reset_index()
    )
    return scored, flags


def flagged_percent(flags, engine_ids):
    ids = set(int(x) for x in engine_ids)
    subset = flags[flags["engine_id"].isin(ids)]
    if subset.empty:
        return 0.0
    return float(subset["drift_detected"].mean() * 100.0)


def save_json(path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return [clean(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with open(path, "w", encoding="utf-8") as file:
        json.dump(clean(payload), file, indent=4, allow_nan=False)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PROGNOSAI - STAGE-HOMOGENEOUS SENSOR VARIANCE DRIFT")
    print("=" * 72)

    (
        predictions,
        sensor_features,
        profile_engines,
        calibration_engines,
        validation_engines,
    ) = base.load_and_predict()

    profile_data = predictions[
        predictions["engine_id"].isin(profile_engines)
    ].copy()

    calibration_data = predictions[
        predictions["engine_id"].isin(calibration_engines)
    ].copy()

    validation_data = predictions[
        predictions["engine_id"].isin(validation_engines)
    ].copy()

    # Exclude constant sensors by requiring non-trivial pooled reference spread.
    sensor_std = profile_data[sensor_features].astype(float).std()
    eligible_sensors = sensor_std[sensor_std > EPSILON].sort_values(
        ascending=False
    )

    selected_sensors = eligible_sensors.head(
        int(base.TOP_SENSORS_TO_SHIFT)
    ).index.tolist()

    if len(selected_sensors) != int(base.TOP_SENSORS_TO_SHIFT):
        raise ValueError("Could not select the expected number of variable sensors.")

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

    print("\nENGINE GROUPS")
    print("Profile engines:", len(profile_engines))
    print("Calibration engines:", len(calibration_engines))
    print("Validation engines:", len(validation_engines))
    print("Affected engines:", affected_engines)
    print("Control engines:", control_engines)
    print("Selected sensors:", selected_sensors)
    print("Window size/step:", WINDOW_SIZE, "/", WINDOW_STEP)

    profiles = build_variance_profiles(profile_data, selected_sensors)
    profiles.to_csv(OUTPUT_DIR / "variance_reference_profiles.csv", index=False)

    print("\nREFERENCE MAD PROFILES")
    print(
        profiles.pivot(
            index="predicted_stage",
            columns="sensor",
            values="reference_mad",
        ).round(5).to_string()
    )

    # Calibrate on the untouched 16-engine calibration cohort.
    calibration_scores, calibration_sensor_scores = score_windows(
        calibration_data,
        profiles,
        selected_sensors,
        "normal_calibration",
    )

    threshold, calibration_engine_scores, rank = base.calibrate_threshold(
        calibration_scores,
        ALPHA,
    )

    calibration_scores.to_csv(
        OUTPUT_DIR / "calibration_window_scores.csv", index=False
    )
    calibration_sensor_scores.to_csv(
        OUTPUT_DIR / "calibration_sensor_scores.csv", index=False
    )
    calibration_engine_scores.to_csv(
        OUTPUT_DIR / "calibration_engine_scores.csv", index=False
    )

    # Evaluate untouched normal validation before synthetic injections.
    normal_scores, normal_sensor_scores = score_windows(
        validation_data,
        profiles,
        selected_sensors,
        "normal_validation",
    )
    normal_scored, normal_flags = engine_flags(normal_scores, threshold)

    normal_scored.to_csv(
        OUTPUT_DIR / "normal_validation_window_scores.csv", index=False
    )
    normal_sensor_scores.to_csv(
        OUTPUT_DIR / "normal_validation_sensor_scores.csv", index=False
    )
    normal_flags.to_csv(
        OUTPUT_DIR / "normal_validation_engine_flags.csv", index=False
    )

    normal_alarm = float(normal_flags["drift_detected"].mean() * 100.0)

    print("\nCALIBRATION")
    print("Calibration engines:", len(calibration_engine_scores))
    print("Finite-sample rank:", rank)
    print(f"Threshold: {threshold:.6f}")
    print(f"Normal validation engine alarm rate: {normal_alarm:.2f}%")

    results = []
    scenario_sensor_scores = []

    for multiplier in VARIANCE_MULTIPLIERS:
        scenario = f"variance_multiplier_{multiplier:.2f}x"
        print(f"\nRunning {scenario}...")

        simulated = inject_variance_drift(
            validation_data,
            selected_sensors,
            affected_engines,
            multiplier,
        )

        scores, sensor_scores = score_windows(
            simulated,
            profiles,
            selected_sensors,
            scenario,
        )
        scored, flags = engine_flags(scores, threshold)

        scored.to_csv(
            OUTPUT_DIR / f"{scenario}_window_scores.csv", index=False
        )
        sensor_scores.to_csv(
            OUTPUT_DIR / f"{scenario}_sensor_scores.csv", index=False
        )
        flags.to_csv(
            OUTPUT_DIR / f"{scenario}_engine_flags.csv", index=False
        )
        scenario_sensor_scores.append(sensor_scores)

        results.append({
            "experiment": "within_engine_stage_conditioned_variance_increase",
            "multiplier": float(multiplier),
            "threshold": float(threshold) if np.isfinite(threshold) else None,
            "affected_engine_detection_percent": flagged_percent(
                flags, affected_engines
            ),
            "control_engine_false_alarm_percent": flagged_percent(
                flags, control_engines
            ),
            "overall_engine_alarm_percent": float(
                flags["drift_detected"].mean() * 100.0
            ),
            "normal_validation_engine_alarm_percent": normal_alarm,
            "affected_engine_count": len(affected_engines),
            "control_engine_count": len(control_engines),
            "monitoring_windows": len(scores),
        })

        print(
            f"Multiplier {multiplier:.2f}x | "
            f"affected detection="
            f"{results[-1]['affected_engine_detection_percent']:.2f}% | "
            f"control alarms="
            f"{results[-1]['control_engine_false_alarm_percent']:.2f}%"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "variance_drift_results.csv", index=False
    )

    if scenario_sensor_scores:
        pd.concat(scenario_sensor_scores, ignore_index=True).to_csv(
            OUTPUT_DIR / "all_scenario_sensor_scores.csv", index=False
        )

    summary = {
        "experiment": "Stage-homogeneous within-engine variance drift",
        "dataset": "NASA C-MAPSS FD001",
        "stage_assignment": "Frozen baseline predicted RUL; not recomputed after injection",
        "profile_engine_count": len(profile_engines),
        "calibration_engine_count": len(calibration_engines),
        "validation_engine_count": len(validation_engines),
        "profile_engines": sorted(int(x) for x in profile_engines),
        "calibration_engines": sorted(int(x) for x in calibration_engines),
        "validation_engines": validation_engine_list,
        "affected_engines": affected_engines,
        "control_engines": control_engines,
        "selected_sensors": selected_sensors,
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "injection_start_cycle": INJECTION_START_CYCLE,
        "variance_multipliers": VARIANCE_MULTIPLIERS,
        "score": "max(0, log(current_window_MAD / stage_sensor_reference_MAD))",
        "aggregation": "Mean one-sided score across the five selected sensors per stage-homogeneous window",
        "calibration_alpha": ALPHA,
        "calibration_rank": int(rank),
        "threshold": float(threshold) if np.isfinite(threshold) else None,
        "normal_validation_engine_alarm_percent": normal_alarm,
        "results": results_df.to_dict(orient="records"),
        "limitations": [
            "Synthetic variance perturbations are not real industrial fault evidence.",
            "Only 16 engines calibrate the threshold and 20 engines validate it.",
            "The same 20 validation engines are reused across multiplier scenarios.",
            "The five selected sensors are selected by pooled reference variability.",
            "Reference profiles use stage-homogeneous windows and engine-balanced medians.",
            "Frozen baseline predicted RUL and predicted stages are not recomputed after injection.",
            "The calibration threshold is based on the maximum window score per calibration engine.",
            "Observed control alarms in this small cohort do not establish a population false-alarm guarantee.",
            "Results do not establish production performance or real maintenance savings."
        ],
    }

    save_json(
        OUTPUT_DIR / "sensor_variance_drift_metrics.json",
        summary,
    )

    print("\n" + "=" * 72)
    print("FINAL VARIANCE DRIFT RESULTS")
    print("=" * 72)
    print(
        results_df[
            [
                "multiplier",
                "affected_engine_detection_percent",
                "control_engine_false_alarm_percent",
                "normal_validation_engine_alarm_percent",
            ]
        ].round(2).to_string(index=False)
    )
    print("\nArtifacts saved to:", OUTPUT_DIR)
    print("Variance drift experiment completed.")


if __name__ == "__main__":
    main()
