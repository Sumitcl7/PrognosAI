
"""
PrognosAI - Calibrated Residual Drift Detection

Research design:
    64 engines: research model training
    16 engines: residual calibration
    20 engines: independent validation

The original frozen baseline is not modified.
The official FD001 test set is not used.

Residual:
    actual_RUL - predicted_RUL

Positive residual:
    Model underpredicts RUL.

Negative residual:
    Model overpredicts RUL.

Calibration:
    Engine-level residual statistics, separately for each
    actual-RUL stage.

Synthetic experiment:
    Additional prediction bias applied to half of validation
    engines in middle, late-middle and critical stages.

IMPORTANT:
    Actual RUL is used for offline stage assignment.
    This is not yet a deployable online drift detector.
"""

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed"
    / "FD001_train_with_RUL.csv"
)

BASELINE_METRICS_FILE = (
    ROOT / "artifacts" / "baseline" / "metrics.json"
)

VALIDATION_PREDICTIONS_FILE = (
    ROOT / "artifacts" / "baseline"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "artifacts" / "drift"
    / "calibrated_residual"
)

RANDOM_STATE = 42

ALPHA = 0.10

BIAS_LEVELS = [0, 3, 5, 10, 15, 20]

STAGE_BINS = [0, 30, 60, 90, 125]

STAGE_LABELS = [
    "Critical",
    "Late-middle",
    "Middle",
    "Early",
]

# Match the existing baseline model configuration.
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------

def assign_stage(df):

    result = df.copy()

    result["stage"] = pd.cut(
        result["actual_RUL"],
        bins=STAGE_BINS,
        labels=STAGE_LABELS,
        include_lowest=True,
    )

    result = result.dropna(
        subset=["stage"]
    ).copy()

    result["stage"] = result["stage"].astype(str)

    return result


def engine_stage_metrics(df):

    """
    One row per engine per RUL stage.

    Observations are aggregated within each engine so that
    individual cycles are not treated as independent engines.
    """

    working = df.copy()

    working["residual"] = (
        working["actual_RUL"]
        - working["predicted_RUL"]
    )

    working["absolute_residual"] = (
        working["residual"].abs()
    )

    result = (
        working.groupby(
            ["engine_id", "stage"],
            observed=True,
        )
        .agg(
            observations=("residual", "size"),
            mean_residual=("residual", "mean"),
            mae=("absolute_residual", "mean"),
            residual_std=("residual", "std"),
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------

def finite_sample_threshold(scores, alpha):

    """
    Split-conformal finite-sample order statistic.

    For n=16 and alpha=0.10:
        rank = ceil((16+1)*0.90) = 16

    Therefore the threshold is the maximum calibration score.
    """

    scores = np.asarray(
        scores,
        dtype=float,
    )

    scores = scores[np.isfinite(scores)]

    n = len(scores)

    if n == 0:
        raise ValueError(
            "No valid calibration scores."
        )

    rank = math.ceil(
        (n + 1) * (1 - alpha)
    )

    if rank > n:
        threshold = np.inf
    else:
        threshold = np.sort(scores)[rank - 1]

    return float(threshold), int(rank), int(n)


def calibrate_thresholds(calibration_metrics):

    thresholds = []

    for stage in STAGE_LABELS:

        stage_df = calibration_metrics[
            calibration_metrics["stage"] == stage
        ].copy()

        if stage_df["engine_id"].nunique() < 10:
            raise ValueError(
                f"Insufficient calibration engines for {stage}."
            )

        # Robust centers estimated from calibration engines.
        center_mean = float(
            stage_df["mean_residual"].median()
        )

        center_mae = float(
            stage_df["mae"].median()
        )

        mean_scores = (
            stage_df["mean_residual"] - center_mean
        ).abs()

        mae_scores = (
            stage_df["mae"] - center_mae
        ).abs()

        mean_threshold, mean_rank, mean_n = (
            finite_sample_threshold(
                mean_scores,
                ALPHA,
            )
        )

        mae_threshold, mae_rank, mae_n = (
            finite_sample_threshold(
                mae_scores,
                ALPHA,
            )
        )

        thresholds.append({
            "stage": stage,
            "calibration_engines": int(
                stage_df["engine_id"].nunique()
            ),
            "center_mean_residual": center_mean,
            "center_mae": center_mae,
            "mean_residual_threshold": mean_threshold,
            "mean_residual_rank": mean_rank,
            "mean_residual_calibration_n": mean_n,
            "mae_threshold": mae_threshold,
            "mae_rank": mae_rank,
            "mae_calibration_n": mae_n,
        })

    return pd.DataFrame(thresholds)


# ---------------------------------------------------------
# DRIFT EVALUATION
# ---------------------------------------------------------

def evaluate_engine_flags(
    engine_metrics,
    threshold_table,
):

    """
    An engine is flagged if either its signed residual
    deviation or MAE deviation exceeds a stage threshold.

    An engine is considered drifted if at least one stage
    produces an alarm.
    """

    merged = engine_metrics.merge(
        threshold_table,
        on="stage",
        how="left",
        validate="many_to_one",
    )

    merged["mean_residual_score"] = (
        merged["mean_residual"]
        - merged["center_mean_residual"]
    ).abs()

    merged["mae_score"] = (
        merged["mae"]
        - merged["center_mae"]
    ).abs()

    merged["mean_residual_alarm"] = (
        merged["mean_residual_score"]
        > merged["mean_residual_threshold"]
    )

    merged["mae_alarm"] = (
        merged["mae_score"]
        > merged["mae_threshold"]
    )

    merged["stage_alarm"] = (
        merged["mean_residual_alarm"]
        | merged["mae_alarm"]
    )

    engine_flags = (
        merged.groupby("engine_id")
        .agg(
            drift_detected=("stage_alarm", "any"),
            stages_flagged=("stage_alarm", "sum"),
            stages_evaluated=("stage_alarm", "size"),
        )
        .reset_index()
    )

    return merged, engine_flags


def calculate_detection_rates(
    engine_flags,
    affected_engines,
):

    affected_engines = set(affected_engines)

    affected = engine_flags[
        engine_flags["engine_id"].isin(
            affected_engines
        )
    ]

    controls = engine_flags[
        ~engine_flags["engine_id"].isin(
            affected_engines
        )
    ]

    return {
        "affected_detection_rate_percent": (
            float(
                affected["drift_detected"].mean() * 100
            )
            if len(affected) else 0.0
        ),
        "control_alarm_rate_percent": (
            float(
                controls["drift_detected"].mean() * 100
            )
            if len(controls) else 0.0
        ),
        "affected_engines_evaluated": int(len(affected)),
        "control_engines_evaluated": int(len(controls)),
        "affected_engines_flagged": int(
            affected["drift_detected"].sum()
        ),
        "control_engines_flagged": int(
            controls["drift_detected"].sum()
        ),
    }


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():

    print("Loading FD001 data...")

    data = pd.read_csv(DATA_FILE)

    with open(BASELINE_METRICS_FILE, "r") as f:
        baseline_metrics = json.load(f)

    original_validation = pd.read_csv(
        VALIDATION_PREDICTIONS_FILE
    )

    features = baseline_metrics["features"]

    required_columns = (
        ["engine_id", "cycle", "RUL_capped"]
        + features
    )

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    validation_engines = set(
        original_validation["engine_id"].astype(int)
    )

    all_engines = set(
        data["engine_id"].astype(int).unique()
    )

    original_training_engines = sorted(
        all_engines - validation_engines
    )

    if len(original_training_engines) != 80:
        raise ValueError(
            "Expected 80 original training engines."
        )

    if len(validation_engines) != 20:
        raise ValueError(
            "Expected 20 held-out validation engines."
        )

    # Deterministic split of the original 80 training engines.
    rng = np.random.default_rng(RANDOM_STATE)

    shuffled_engines = rng.permutation(
        original_training_engines
    )

    research_train_engines = sorted(
        shuffled_engines[:64].tolist()
    )

    calibration_engines = sorted(
        shuffled_engines[64:].tolist()
    )

    print("\n========== ENGINE SPLIT ==========")

    print(
        "Research model training engines:",
        len(research_train_engines),
    )

    print(
        "Calibration engines:",
        len(calibration_engines),
    )

    print(
        "Independent validation engines:",
        len(validation_engines),
    )

    # -----------------------------------------------------
    # TRAIN RESEARCH MODEL
    # -----------------------------------------------------

    train_data = data[
        data["engine_id"].isin(
            research_train_engines
        )
    ].copy()

    calibration_data = data[
        data["engine_id"].isin(
            calibration_engines
        )
    ].copy()

    validation_data = data[
        data["engine_id"].isin(
            validation_engines
        )
    ].copy()

    train_data = train_data.sort_values(
        ["engine_id", "cycle"]
    )

    calibration_data = calibration_data.sort_values(
        ["engine_id", "cycle"]
    )

    validation_data = validation_data.sort_values(
        ["engine_id", "cycle"]
    )

    print("\n========== MODEL TRAINING ==========")

    model = RandomForestRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        train_data[features],
        train_data["RUL_capped"],
    )

    print("Research model trained.")
    print("Training records:", len(train_data))

    joblib.dump(
        model,
        OUTPUT_DIR / "research_random_forest.joblib",
    )

    # -----------------------------------------------------
    # PREDICT CALIBRATION ENGINES
    # -----------------------------------------------------

    print("\n========== CALIBRATION PREDICTIONS ==========")

    calibration_predictions = calibration_data[
        ["engine_id", "cycle", "RUL_capped"]
    ].copy()

    calibration_predictions = (
        calibration_predictions.rename(
            columns={"RUL_capped": "actual_RUL"}
        )
    )

    calibration_predictions["predicted_RUL"] = (
        model.predict(
            calibration_data[features]
        )
    )

    calibration_predictions = assign_stage(
        calibration_predictions
    )

    calibration_metrics = engine_stage_metrics(
        calibration_predictions
    )

    calibration_metrics.to_csv(
        OUTPUT_DIR / "calibration_engine_stage_metrics.csv",
        index=False,
    )

    print(
        "Calibration engines:",
        calibration_metrics["engine_id"].nunique(),
    )

    print(
        "Calibration stage records:",
        len(calibration_metrics),
    )

    # -----------------------------------------------------
    # CALIBRATE THRESHOLDS
    # -----------------------------------------------------

    print("\n========== CALIBRATING THRESHOLDS ==========")

    threshold_table = calibrate_thresholds(
        calibration_metrics
    )

    threshold_table.to_csv(
        OUTPUT_DIR / "residual_thresholds.csv",
        index=False,
    )

    print(
        threshold_table.round(4).to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # PREDICT INDEPENDENT VALIDATION ENGINES
    # -----------------------------------------------------

    print("\n========== VALIDATION PREDICTIONS ==========")

    validation_predictions = validation_data[
        ["engine_id", "cycle", "RUL_capped"]
    ].copy()

    validation_predictions = (
        validation_predictions.rename(
            columns={"RUL_capped": "actual_RUL"}
        )
    )

    validation_predictions["predicted_RUL"] = (
        model.predict(
            validation_data[features]
        )
    )

    validation_predictions = assign_stage(
        validation_predictions
    )

    validation_predictions.to_csv(
        OUTPUT_DIR / "research_validation_predictions.csv",
        index=False,
    )

    baseline_mae = mean_absolute_error(
        validation_predictions["actual_RUL"],
        validation_predictions["predicted_RUL"],
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            validation_predictions["actual_RUL"],
            validation_predictions["predicted_RUL"],
        )
    )

    baseline_r2 = r2_score(
        validation_predictions["actual_RUL"],
        validation_predictions["predicted_RUL"],
    )

    print("Validation observations:", len(validation_predictions))
    print("Validation MAE:", round(baseline_mae, 4))
    print("Validation RMSE:", round(baseline_rmse, 4))
    print("Validation R2:", round(baseline_r2, 4))

    # -----------------------------------------------------
    # BASELINE VALIDATION ALARMS
    # -----------------------------------------------------

    print("\n========== NORMAL VALIDATION ALARMS ==========")

    validation_engine_metrics = engine_stage_metrics(
        validation_predictions
    )

    baseline_stage_flags, baseline_engine_flags = (
        evaluate_engine_flags(
            validation_engine_metrics,
            threshold_table,
        )
    )

    baseline_engine_flags.to_csv(
        OUTPUT_DIR / "normal_validation_engine_flags.csv",
        index=False,
    )

    baseline_stage_flags.to_csv(
        OUTPUT_DIR / "normal_validation_stage_flags.csv",
        index=False,
    )

    baseline_rates = calculate_detection_rates(
        baseline_engine_flags,
        affected_engines=[],
    )

    print(
        "Validation engines flagged:",
        int(
            baseline_engine_flags["drift_detected"].sum()
        ),
        "/",
        len(baseline_engine_flags),
    )

    print(
        "Normal validation engine alarm rate:",
        f"{baseline_engine_flags['drift_detected'].mean() * 100:.2f}%",
    )

    # -----------------------------------------------------
    # SYNTHETIC DRIFT EXPERIMENT
    # -----------------------------------------------------

    print("\n========== SYNTHETIC DRIFT SIMULATION ==========")

    validation_engine_ids = sorted(
        validation_predictions["engine_id"].unique()
    )

    rng = np.random.default_rng(RANDOM_STATE)

    shuffled_validation = rng.permutation(
        validation_engine_ids
    )

    affected_engines = set(
        shuffled_validation[
            :len(shuffled_validation) // 2
        ]
    )

    control_engines = (
        set(validation_engine_ids) - affected_engines
    )

    print(
        "Affected engines:",
        sorted(affected_engines),
    )

    print(
        "Control engines:",
        sorted(control_engines),
    )

    simulation_rows = []

    for bias in BIAS_LEVELS:

        simulated = validation_predictions.copy()

        injection_mask = (
            simulated["engine_id"].isin(
                affected_engines
            )
            & simulated["stage"].isin(
                ["Critical", "Late-middle", "Middle"]
            )
        )

        # Increasing predicted RUL creates an optimistic
        # prediction bias (negative signed residual).
        simulated.loc[
            injection_mask,
            "predicted_RUL",
        ] += bias

        simulated_metrics = engine_stage_metrics(
            simulated
        )

        stage_flags, engine_flags = (
            evaluate_engine_flags(
                simulated_metrics,
                threshold_table,
            )
        )

        rates = calculate_detection_rates(
            engine_flags,
            affected_engines,
        )

        rates["injected_bias_cycles"] = int(bias)

        simulation_rows.append(rates)

        stage_flags.to_csv(
            OUTPUT_DIR
            / f"simulation_stage_flags_bias_{bias}.csv",
            index=False,
        )

        engine_flags.to_csv(
            OUTPUT_DIR
            / f"simulation_engine_flags_bias_{bias}.csv",
            index=False,
        )

        print(f"\nInjected prediction bias: {bias} cycles")

        print(
            "Affected engine detection:",
            f"{rates['affected_detection_rate_percent']:.2f}%",
        )

        print(
            "Control engine alarm rate:",
            f"{rates['control_alarm_rate_percent']:.2f}%",
        )

    simulation_results = pd.DataFrame(
        simulation_rows
    )

    simulation_results.to_csv(
        OUTPUT_DIR / "simulation_results.csv",
        index=False,
    )

    # -----------------------------------------------------
    # STAGE-WISE VALIDATION PERFORMANCE
    # -----------------------------------------------------

    validation_stage_metrics = (
        validation_predictions.assign(
            residual=lambda x: (
                x["actual_RUL"] - x["predicted_RUL"]
            )
        )
        .assign(
            absolute_residual=lambda x: (
                x["residual"].abs()
            )
        )
        .groupby("stage", observed=True)
        .agg(
            observations=("residual", "size"),
            engines=("engine_id", "nunique"),
            mean_residual=("residual", "mean"),
            MAE=("absolute_residual", "mean"),
            residual_std=("residual", "std"),
        )
        .reset_index()
    )

    validation_stage_metrics.to_csv(
        OUTPUT_DIR / "validation_stage_metrics.csv",
        index=False,
    )

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        validation_stage_metrics["stage"],
        validation_stage_metrics["MAE"],
    )

    plt.title(
        "Research Model Validation MAE by RUL Stage"
    )

    plt.xlabel("RUL Stage")
    plt.ylabel("MAE (cycles)")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "validation_stage_mae.png",
        dpi=150,
    )

    plt.close()

    plt.figure(figsize=(9, 5))

    plt.plot(
        simulation_results["injected_bias_cycles"],
        simulation_results[
            "affected_detection_rate_percent"
        ],
        marker="o",
        label="Affected-engine detection",
    )

    plt.plot(
        simulation_results["injected_bias_cycles"],
        simulation_results[
            "control_alarm_rate_percent"
        ],
        marker="s",
        label="Control-engine alarm rate",
    )

    plt.xlabel("Injected prediction bias (cycles)")
    plt.ylabel("Engine-level rate (%)")

    plt.title(
        "Calibrated Residual Drift Detection Sensitivity"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "calibrated_drift_sensitivity.png",
        dpi=150,
    )

    plt.close()

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    metrics = {
        "experiment": "calibrated_residual_drift",
        "dataset": "NASA C-MAPSS FD001",
        "model": "RandomForestRegressor",
        "model_parameters": MODEL_PARAMS,
        "features": features,
        "split": {
            "research_training_engines": research_train_engines,
            "calibration_engines": calibration_engines,
            "validation_engines": validation_engine_ids,
        },
        "alpha": ALPHA,
        "calibration_method": (
            "Engine-level absolute deviation from "
            "calibration median, using finite-sample "
            "order statistic."
        ),
        "research_model_validation": {
            "observations": int(
                len(validation_predictions)
            ),
            "MAE": float(baseline_mae),
            "RMSE": float(baseline_rmse),
            "R2": float(baseline_r2),
        },
        "normal_validation": {
            "engines_flagged": int(
                baseline_engine_flags[
                    "drift_detected"
                ].sum()
            ),
            "engines_evaluated": int(
                len(baseline_engine_flags)
            ),
            "alarm_rate_percent": float(
                baseline_engine_flags[
                    "drift_detected"
                ].mean() * 100
            ),
        },
        "affected_engines": sorted(
            int(x) for x in affected_engines
        ),
        "control_engines": sorted(
            int(x) for x in control_engines
        ),
        "simulation_results": simulation_rows,
        "limitations": [
            "Only 16 engines are used for calibration.",
            "The 90% finite-sample threshold is the maximum calibration score.",
            "Actual RUL is used for offline degradation-stage assignment.",
            "Synthetic prediction bias is not real industrial drift.",
            "The validation group contains only 20 engines.",
            "Threshold performance must be assessed on additional independent splits.",
            "This experiment does not establish a production false-alarm guarantee.",
        ],
    }

    with open(
        OUTPUT_DIR / "calibrated_residual_metrics.json",
        "w",
    ) as f:
        json.dump(
    metrics,
    f,
    indent=4,
    default=lambda obj: (
        obj.item() if isinstance(obj, np.generic)
        else obj.tolist() if isinstance(obj, np.ndarray)
        else str(obj)
    ),
)

    print("\n========== GENERATED FILES ==========")

    for file in sorted(OUTPUT_DIR.iterdir()):
        print(file.name)

    print("\nCalibrated residual drift experiment completed.")


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    main()
