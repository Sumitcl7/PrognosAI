"""
PrognosAI - Sensor Drift Pattern Robustness Experiments

Creates a separate robustness experiment without modifying the existing
calibrated_sensor_drift.py script or frozen baseline artifacts.

Experiments:
1. Gradual sensor mean drift:
   A linear ramp begins at cycle 80 and reaches the configured final shift
   (in reference standard deviations) at the engine's final observed cycle.
2. Sensor variance drift:
   After cycle 80, deviations around the stage-conditioned reference median
   are multiplied by a configured factor. The median/location is preserved
   approximately while dispersion increases.

Design:
- Reuses the existing 64/16/20 engine split and frozen baseline predictions.
- Builds stage-conditioned profiles from 64 reference engines.
- Calibrates separate location-drift and variance-drift thresholds using
  the 16 calibration engines.
- Evaluates 10 affected and 10 unaffected control engines.
- Reports affected-engine detection and control-engine alarm rates separately.

Limitations:
- Synthetic experiments are not evidence of real industrial fault detection.
- Only 16 engines are used for threshold calibration and 20 for validation.
- The same 20 validation engines are reused across all synthetic scenarios.
- The frozen model's predicted RUL/stage is not recomputed after injection.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.drift import calibrated_sensor_drift as base


# ==================================================
# CONFIGURATION
# ==================================================

OUTPUT_DIR = (
    base.ROOT / "artifacts" / "drift"
    / "sensor_drift_robustness"
)

GRADUAL_SHIFT_LEVELS = [0.25, 0.50, 0.75, 1.00, 1.50]
VARIANCE_MULTIPLIERS = [1.25, 1.50, 2.00, 3.00]

AFFECTED_ENGINE_COUNT = 10
TOP_K_SCORE = 5
EPSILON = 1e-8


# ==================================================
# SCORING
# ==================================================

def score_windows(data, profiles, sensor_features, score_type, scenario):
    """
    score_type:
      - 'location': standardized absolute window-median deviation
      - 'variance': absolute log ratio of window MAD to reference MAD

    Each window score is the mean of the five largest sensor scores.
    """
    if score_type not in {"location", "variance"}:
        raise ValueError("score_type must be 'location' or 'variance'.")

    lookup = {
        (row.predicted_stage, row.sensor): row
        for row in profiles.itertuples(index=False)
    }

    results = []

    for engine_id, engine_data in data.groupby("engine_id"):
        engine_data = (
            engine_data.sort_values("cycle")
            .reset_index(drop=True)
        )

        min_cycle = int(engine_data["cycle"].min())
        max_cycle = int(engine_data["cycle"].max())

        for start in range(
            min_cycle,
            max_cycle - base.WINDOW_SIZE + 2,
            base.WINDOW_STEP,
        ):
            end = start + base.WINDOW_SIZE - 1

            window = engine_data[
                (engine_data["cycle"] >= start)
                & (engine_data["cycle"] <= end)
            ]

            if len(window) < base.WINDOW_SIZE:
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

                values = window[sensor].to_numpy(dtype=float)
                current_median = float(np.median(values))
                current_mad = float(
                    np.median(np.abs(values - current_median))
                )

                reference_median = float(profile.reference_median)
                reference_mad = float(profile.reference_mad)
                effective_scale = max(
                    float(profile.effective_scale), EPSILON
                )

                if score_type == "location":
                    sensor_score = (
                        abs(current_median - reference_median)
                        / effective_scale
                    )
                else:
                    # Dimensionless dispersion change. The small epsilon
                    # avoids division/log errors for near-constant sensors.
                    ref_dispersion = max(reference_mad, EPSILON)
                    current_dispersion = max(current_mad, EPSILON)
                    sensor_score = abs(
                        np.log(current_dispersion / ref_dispersion)
                    )

                sensor_scores.append(float(sensor_score))

            if not sensor_scores:
                continue

            ordered = np.sort(np.asarray(sensor_scores, dtype=float))
            top_k = ordered[-min(TOP_K_SCORE, len(ordered)):]
            aggregate_score = float(np.mean(top_k))

            results.append({
                "scenario": scenario,
                "score_type": score_type,
                "engine_id": int(engine_id),
                "window_start": int(start),
                "window_end": int(end),
                "predicted_stage": stage,
                "aggregate_score": aggregate_score,
                "max_sensor_score": float(np.max(ordered)),
                "mean_sensor_score": float(np.mean(ordered)),
                "median_sensor_score": float(np.median(ordered)),
                "sensors_scored": int(len(ordered)),
            })

    result = pd.DataFrame(results)

    if result.empty:
        raise ValueError(f"No windows scored for scenario: {scenario}")

    return result


def calibrate_and_flag(calibration_scores, evaluation_scores, alpha):
    threshold, engine_calibration, rank = base.calibrate_threshold(
        calibration_scores,
        alpha,
    )

    scored, flags = base.engine_flag_table(
        evaluation_scores,
        threshold,
    )

    return threshold, engine_calibration, rank, scored, flags


def group_alarm_percent(flags, engine_ids):
    return base.percentage_flagged(flags, engine_ids)


# ==================================================
# SYNTHETIC INJECTION
# ==================================================

def inject_gradual_location_drift(
    validation_data,
    sensor_features,
    selected_sensors,
    profile_data,
    affected_engine_ids,
    shift_level,
):
    """Inject a linear mean-shift ramp after cycle 80."""
    simulated = validation_data.copy()
    simulated[sensor_features] = (
        simulated[sensor_features].astype(float)
    )

    affected = set(int(x) for x in affected_engine_ids)

    for engine_id in affected:
        engine_mask = simulated["engine_id"] == engine_id
        engine_rows = simulated.loc[engine_mask]

        max_cycle = float(engine_rows["cycle"].max())
        ramp_denominator = max(max_cycle - base.INJECTION_START_CYCLE, 1.0)

        post_start = (
            engine_mask
            & (simulated["cycle"] >= base.INJECTION_START_CYCLE)
        )

        ramp = (
            simulated.loc[post_start, "cycle"].astype(float)
            - base.INJECTION_START_CYCLE
        ) / ramp_denominator

        ramp = ramp.clip(0.0, 1.0).to_numpy(dtype=float)

        for sensor in selected_sensors:
            reference_std = float(profile_data[sensor].std())
            final_shift = float(shift_level * reference_std)

            simulated.loc[post_start, sensor] = (
                simulated.loc[post_start, sensor].to_numpy(dtype=float)
                + final_shift * ramp
            )

    return simulated


def inject_variance_drift(
    validation_data,
    sensor_features,
    selected_sensors,
    profiles,
    affected_engine_ids,
    variance_multiplier,
):
    """
    Increase sensor dispersion after cycle 80 by scaling deviations
    around the predicted-stage reference median.
    """
    simulated = validation_data.copy()
    simulated[sensor_features] = (
        simulated[sensor_features].astype(float)
    )

    affected = set(int(x) for x in affected_engine_ids)

    reference_medians = {
        (row.predicted_stage, row.sensor): float(row.reference_median)
        for row in profiles.itertuples(index=False)
    }

    for engine_id in affected:
        engine_mask = (
            (simulated["engine_id"] == engine_id)
            & (simulated["cycle"] >= base.INJECTION_START_CYCLE)
        )

        for sensor in selected_sensors:
            for stage in base.STAGE_LABELS:
                mask = (
                    engine_mask
                    & (simulated["predicted_stage"] == stage)
                )

                if not mask.any():
                    continue

                reference_median = reference_medians[(stage, sensor)]
                values = simulated.loc[mask, sensor].to_numpy(dtype=float)

                simulated.loc[mask, sensor] = (
                    reference_median
                    + (values - reference_median) * variance_multiplier
                )

    return simulated


# ==================================================
# RESULT HELPERS
# ==================================================

def make_result_row(
    experiment,
    strength,
    threshold,
    flags,
    affected_engines,
    control_engines,
    normal_alarm,
    windows,
):
    return {
        "experiment": experiment,
        "strength": float(strength),
        "threshold": float(threshold) if np.isfinite(threshold) else None,
        "affected_engine_detection_percent": group_alarm_percent(
            flags, affected_engines
        ),
        "control_engine_false_alarm_percent": group_alarm_percent(
            flags, control_engines
        ),
        "overall_engine_alarm_percent": float(
            flags["drift_detected"].mean() * 100.0
        ),
        "normal_validation_engine_alarm_percent": float(normal_alarm),
        "affected_engine_count": int(len(affected_engines)),
        "control_engine_count": int(len(control_engines)),
        "monitoring_windows": int(windows),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ==================================================
# MAIN
# ==================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROGNOSAI - SENSOR DRIFT PATTERN ROBUSTNESS")
    print("=" * 70)

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

    validation_engine_list = sorted(int(x) for x in validation_engines)

    rng = np.random.default_rng(base.RANDOM_STATE)
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

    selected_sensors = (
        profile_data[sensor_features]
        .std()
        .sort_values(ascending=False)
        .head(base.TOP_SENSORS_TO_SHIFT)
        .index.tolist()
    )

    print("\nEXPERIMENT GROUPS")
    print("Affected engines:", affected_engines)
    print("Control engines:", control_engines)
    print("Sensors:", selected_sensors)
    print("Gradual shift begins at cycle:", base.INJECTION_START_CYCLE)

    all_results = []
    calibration_summary = []

    # Run each variability floor independently.
    for floor in base.VARIABILITY_FLOORS:
        print("\n" + "=" * 65)
        print(f"VARIABILITY FLOOR: {floor}")
        print("=" * 65)

        profiles = base.build_profiles(
            predictions,
            sensor_features,
            profile_engines,
            floor,
        )

        profiles.to_csv(
            OUTPUT_DIR / f"profiles_floor_{floor}.csv",
            index=False,
        )

        # Calibrate separate thresholds for location and dispersion scores.
        for score_type in ["location", "variance"]:
            calibration_scores = score_windows(
                calibration_data,
                profiles,
                sensor_features,
                score_type,
                f"calibration_{score_type}_floor_{floor}",
            )

            normal_validation_scores = score_windows(
                validation_data,
                profiles,
                sensor_features,
                score_type,
                f"normal_validation_{score_type}_floor_{floor}",
            )

            (
                threshold,
                calibration_engine_scores,
                rank,
                normal_scored,
                normal_flags,
            ) = calibrate_and_flag(
                calibration_scores,
                normal_validation_scores,
                base.ALPHA,
            )

            normal_alarm = float(
                normal_flags["drift_detected"].mean() * 100.0
            )

            calibration_engine_scores.to_csv(
                OUTPUT_DIR
                / f"calibration_{score_type}_floor_{floor}.csv",
                index=False,
            )
            normal_scored.to_csv(
                OUTPUT_DIR
                / f"normal_validation_{score_type}_floor_{floor}.csv",
                index=False,
            )
            normal_flags.to_csv(
                OUTPUT_DIR
                / f"normal_validation_flags_{score_type}_floor_{floor}.csv",
                index=False,
            )

            calibration_summary.append({
                "variability_floor": float(floor),
                "score_type": score_type,
                "calibration_engine_count": int(len(calibration_engine_scores)),
                "calibration_rank": int(rank),
                "threshold": (
                    float(threshold) if np.isfinite(threshold) else None
                ),
                "normal_validation_engine_alarm_percent": normal_alarm,
            })

            print(
                f"\n{score_type.upper()} DETECTOR | "
                f"threshold={threshold:.6f} | "
                f"normal alarms={normal_alarm:.2f}%"
            )

            # --------------------------------------------------
            # GRADUAL MEAN DRIFT: location detector
            # --------------------------------------------------
            if score_type == "location":
                for shift_level in GRADUAL_SHIFT_LEVELS:
                    simulated = inject_gradual_location_drift(
                        validation_data,
                        sensor_features,
                        selected_sensors,
                        profile_data,
                        affected_engines,
                        shift_level,
                    )

                    scores = score_windows(
                        simulated,
                        profiles,
                        sensor_features,
                        "location",
                        f"gradual_mean_floor_{floor}_shift_{shift_level}",
                    )

                    scored, flags = base.engine_flag_table(
                        scores,
                        threshold,
                    )

                    row = make_result_row(
                        "gradual_mean_drift",
                        shift_level,
                        threshold,
                        flags,
                        affected_engines,
                        control_engines,
                        normal_alarm,
                        len(scores),
                    )
                    row["variability_floor"] = float(floor)
                    row["score_type"] = score_type
                    all_results.append(row)

                    scored.to_csv(
                        OUTPUT_DIR
                        / f"gradual_mean_floor_{floor}_shift_{shift_level}.csv",
                        index=False,
                    )
                    flags.to_csv(
                        OUTPUT_DIR
                        / f"gradual_mean_flags_floor_{floor}_shift_{shift_level}.csv",
                        index=False,
                    )

                    print(
                        f"Gradual final shift {shift_level:.2f} SD | "
                        f"affected={row['affected_engine_detection_percent']:.2f}% | "
                        f"controls={row['control_engine_false_alarm_percent']:.2f}%"
                    )

            # --------------------------------------------------
            # VARIANCE DRIFT: dispersion detector
            # --------------------------------------------------
            else:
                for multiplier in VARIANCE_MULTIPLIERS:
                    simulated = inject_variance_drift(
                        validation_data,
                        sensor_features,
                        selected_sensors,
                        profiles,
                        affected_engines,
                        multiplier,
                    )

                    scores = score_windows(
                        simulated,
                        profiles,
                        sensor_features,
                        "variance",
                        f"variance_floor_{floor}_multiplier_{multiplier}",
                    )

                    scored, flags = base.engine_flag_table(
                        scores,
                        threshold,
                    )

                    row = make_result_row(
                        "variance_drift",
                        multiplier,
                        threshold,
                        flags,
                        affected_engines,
                        control_engines,
                        normal_alarm,
                        len(scores),
                    )
                    row["variability_floor"] = float(floor)
                    row["score_type"] = score_type
                    all_results.append(row)

                    scored.to_csv(
                        OUTPUT_DIR
                        / f"variance_floor_{floor}_multiplier_{multiplier}.csv",
                        index=False,
                    )
                    flags.to_csv(
                        OUTPUT_DIR
                        / f"variance_flags_floor_{floor}_multiplier_{multiplier}.csv",
                        index=False,
                    )

                    print(
                        f"Variance multiplier {multiplier:.2f}x | "
                        f"affected={row['affected_engine_detection_percent']:.2f}% | "
                        f"controls={row['control_engine_false_alarm_percent']:.2f}%"
                    )

    results_df = pd.DataFrame(all_results)
    calibration_df = pd.DataFrame(calibration_summary)

    results_df.to_csv(
        OUTPUT_DIR / "robustness_simulation_results.csv",
        index=False,
    )
    calibration_df.to_csv(
        OUTPUT_DIR / "robustness_calibration_summary.csv",
        index=False,
    )

    summary = {
        "experiment": "Gradual mean drift and variance drift robustness",
        "dataset": "NASA C-MAPSS FD001",
        "stage_assignment": "Frozen baseline predicted RUL",
        "profile_engines": sorted(int(x) for x in profile_engines),
        "calibration_engines": sorted(int(x) for x in calibration_engines),
        "validation_engines": sorted(int(x) for x in validation_engines),
        "affected_engines": affected_engines,
        "control_engines": control_engines,
        "selected_sensors": selected_sensors,
        "variability_floors": list(base.VARIABILITY_FLOORS),
        "gradual_final_shift_levels_sd": GRADUAL_SHIFT_LEVELS,
        "variance_multipliers": VARIANCE_MULTIPLIERS,
        "window_size": int(base.WINDOW_SIZE),
        "window_step": int(base.WINDOW_STEP),
        "injection_start_cycle": int(base.INJECTION_START_CYCLE),
        "score_aggregation": (
            f"Mean of top {TOP_K_SCORE} sensor scores per window"
        ),
        "calibration_summary": calibration_df.to_dict(orient="records"),
        "simulation_results": results_df.to_dict(orient="records"),
        "limitations": [
            "Synthetic data perturbations are not real industrial fault evidence.",
            "Only 16 calibration engines and 20 validation engines are available.",
            "The same validation engines are reused across all scenarios.",
            "The frozen model predictions and predicted stages are not recomputed after sensor injection.",
            "Gradual drift is a linear ramp beginning at the configured injection cycle.",
            "Variance drift scales deviations around stage-conditioned reference medians.",
            "Separate location and dispersion thresholds are calibrated from normal calibration engines.",
            "Observed zero control alarms in this small cohort do not prove a zero population false-alarm rate.",
            "Results do not establish production guarantees.",
        ],
    }

    with open(
        OUTPUT_DIR / "sensor_drift_robustness_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(summary),
            file,
            indent=4,
            allow_nan=False,
        )

    print("\n" + "=" * 70)
    print("FINAL ROBUSTNESS RESULTS")
    print("=" * 70)

    display_columns = [
        "experiment",
        "strength",
        "variability_floor",
        "affected_engine_detection_percent",
        "control_engine_false_alarm_percent",
        "normal_validation_engine_alarm_percent",
    ]

    print(
        results_df[display_columns]
        .round(2)
        .to_string(index=False)
    )

    print("\nArtifacts saved to:")
    print(OUTPUT_DIR)
    print("\nRobustness experiment completed.")


if __name__ == "__main__":
    main()
