
"""
PrognosAI - Repeated Engine Split Robustness Experiment

Five different splits of the original 80 training engines:
    64 model-training engines
    16 calibration engines

The same 20 previously held-out validation engines are
used for each repetition.

This tests sensitivity to partition choices, not five
independent external validation cohorts.

The original baseline and official test set are untouched.
"""

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/processed/FD001_train_with_RUL.csv"
BASELINE_METRICS = ROOT / "artifacts/baseline/metrics.json"
VALIDATION_FILE = ROOT / "artifacts/baseline/validation_predictions.csv"

OUTPUT_DIR = ROOT / "artifacts/drift/repeated_split"

SPLIT_SEEDS = [42, 7, 21, 100, 2026]
BIAS_LEVELS = [0, 5, 10, 20]
ALPHA = 0.10

STAGE_BINS = [0, 30, 60, 90, 125]
STAGE_LABELS = ["Critical", "Late-middle", "Middle", "Early"]

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


def assign_stage(df):
    result = df.copy()
    result["stage"] = pd.cut(
        result["actual_RUL"],
        bins=STAGE_BINS,
        labels=STAGE_LABELS,
        include_lowest=True,
    )
    return result.dropna(subset=["stage"]).assign(
        stage=lambda x: x["stage"].astype(str)
    )


def engine_stage_metrics(df):
    work = df.copy()
    work["residual"] = work["actual_RUL"] - work["predicted_RUL"]
    work["absolute_residual"] = work["residual"].abs()

    return (
        work.groupby(["engine_id", "stage"], observed=True)
        .agg(
            mean_residual=("residual", "mean"),
            mae=("absolute_residual", "mean"),
            observations=("residual", "size"),
        )
        .reset_index()
    )


def conformal_threshold(scores):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    n = len(scores)
    rank = math.ceil((n + 1) * (1 - ALPHA))

    if n == 0:
        raise ValueError("No calibration scores.")

    if rank > n:
        return float("inf")

    return float(np.sort(scores)[rank - 1])


def build_thresholds(calibration_metrics):
    rows = []

    for stage in STAGE_LABELS:
        part = calibration_metrics[
            calibration_metrics["stage"] == stage
        ]

        if part["engine_id"].nunique() < 10:
            raise ValueError(
                f"Insufficient calibration engines for {stage}"
            )

        center_residual = float(part["mean_residual"].median())
        center_mae = float(part["mae"].median())

        residual_threshold = conformal_threshold(
            np.abs(part["mean_residual"] - center_residual)
        )

        mae_threshold = conformal_threshold(
            np.abs(part["mae"] - center_mae)
        )

        rows.append({
            "stage": stage,
            "center_residual": center_residual,
            "center_mae": center_mae,
            "residual_threshold": residual_threshold,
            "mae_threshold": mae_threshold,
            "calibration_engines": int(
                part["engine_id"].nunique()
            ),
        })

    return pd.DataFrame(rows)


def flag_engines(validation_metrics, thresholds):
    merged = validation_metrics.merge(
        thresholds,
        on="stage",
        how="left",
        validate="many_to_one",
    )

    merged["residual_alarm"] = (
        (merged["mean_residual"] - merged["center_residual"]).abs()
        > merged["residual_threshold"]
    )

    merged["mae_alarm"] = (
        (merged["mae"] - merged["center_mae"]).abs()
        > merged["mae_threshold"]
    )

    merged["stage_alarm"] = (
        merged["residual_alarm"] | merged["mae_alarm"]
    )

    engine_flags = (
        merged.groupby("engine_id")
        .agg(
            drift_detected=("stage_alarm", "any"),
            stages_flagged=("stage_alarm", "sum"),
        )
        .reset_index()
    )

    return merged, engine_flags


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FD001 data...")

    data = pd.read_csv(DATA_FILE)

    with open(BASELINE_METRICS, "r") as f:
        baseline_config = json.load(f)

    original_validation = pd.read_csv(VALIDATION_FILE)

    features = baseline_config["features"]

    validation_ids = sorted(
        original_validation["engine_id"].astype(int).unique()
    )

    all_ids = sorted(data["engine_id"].astype(int).unique())

    train_pool = sorted(set(all_ids) - set(validation_ids))

    if len(train_pool) != 80 or len(validation_ids) != 20:
        raise ValueError("Unexpected engine split.")

    validation_data = data[
        data["engine_id"].isin(validation_ids)
    ].copy()

    validation_data = validation_data.sort_values(
        ["engine_id", "cycle"]
    )

    validation_base = validation_data[
        ["engine_id", "cycle", "RUL_capped"]
    ].rename(columns={"RUL_capped": "actual_RUL"}).copy()

    results = []

    for seed in SPLIT_SEEDS:

        print("\n" + "=" * 55)
        print(f"EXPERIMENT SEED: {seed}")
        print("=" * 55)

        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(train_pool)

        model_ids = sorted(shuffled[:64].tolist())
        calibration_ids = sorted(shuffled[64:].tolist())

        model_data = data[
            data["engine_id"].isin(model_ids)
        ]

        calibration_data = data[
            data["engine_id"].isin(calibration_ids)
        ].sort_values(["engine_id", "cycle"])

        model = RandomForestRegressor(**MODEL_PARAMS)

        model.fit(
            model_data[features],
            model_data["RUL_capped"],
        )

        # Calibration predictions
        cal = calibration_data[
            ["engine_id", "cycle", "RUL_capped"]
        ].rename(columns={"RUL_capped": "actual_RUL"}).copy()

        cal["predicted_RUL"] = model.predict(
            calibration_data[features]
        )

        cal = assign_stage(cal)

        cal_metrics = engine_stage_metrics(cal)
        thresholds = build_thresholds(cal_metrics)

        thresholds.to_csv(
            OUTPUT_DIR / f"thresholds_seed_{seed}.csv",
            index=False,
        )

        # Independent validation predictions
        val = validation_base.copy()

        val["predicted_RUL"] = model.predict(
            validation_data[features]
        )

        val = assign_stage(val)

        mae = mean_absolute_error(
            val["actual_RUL"], val["predicted_RUL"]
        )

        rmse = np.sqrt(mean_squared_error(
            val["actual_RUL"], val["predicted_RUL"]
        ))

        r2 = r2_score(
            val["actual_RUL"], val["predicted_RUL"]
        )

        # Randomly select half the validation engines
        val_ids = np.sort(val["engine_id"].unique())
        affected = set(
            rng.permutation(val_ids)[:len(val_ids) // 2]
        )

        controls = set(val_ids) - affected

        # Normal validation alarm rate
        normal_metrics = engine_stage_metrics(val)
        _, normal_flags = flag_engines(
            normal_metrics, thresholds
        )

        normal_alarm_rate = (
            normal_flags["drift_detected"].mean() * 100
        )

        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation R2: {r2:.4f}")
        print(
            "Normal alarms:",
            f"{normal_alarm_rate:.2f}%",
        )

        for bias in BIAS_LEVELS:

            simulated = val.copy()

            injection_mask = (
                simulated["engine_id"].isin(affected)
                & simulated["stage"].isin(
                    ["Critical", "Late-middle", "Middle"]
                )
            )

            simulated.loc[
                injection_mask,
                "predicted_RUL",
            ] += bias

            sim_metrics = engine_stage_metrics(simulated)

            stage_flags, engine_flags = flag_engines(
                sim_metrics,
                thresholds,
            )

            affected_flags = engine_flags[
                engine_flags["engine_id"].isin(affected)
            ]

            control_flags = engine_flags[
                engine_flags["engine_id"].isin(controls)
            ]

            affected_rate = (
                affected_flags["drift_detected"].mean() * 100
            )

            control_rate = (
                control_flags["drift_detected"].mean() * 100
            )

            row = {
                "seed": seed,
                "bias_cycles": bias,
                "validation_MAE": mae,
                "validation_RMSE": rmse,
                "validation_R2": r2,
                "normal_alarm_rate_percent": normal_alarm_rate,
                "affected_detection_percent": affected_rate,
                "control_alarm_percent": control_rate,
                "affected_engines_flagged": int(
                    affected_flags["drift_detected"].sum()
                ),
                "control_engines_flagged": int(
                    control_flags["drift_detected"].sum()
                ),
            }

            results.append(row)

            print(
                f"Bias {bias:>2} | "
                f"Affected: {affected_rate:>5.1f}% | "
                f"Controls: {control_rate:>5.1f}%"
            )

        joblib.dump(
            model,
            OUTPUT_DIR / f"research_model_seed_{seed}.joblib",
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "repeated_split_results.csv",
        index=False,
    )

    # Aggregate across repeated partition choices.
    summary = (
        results_df.groupby("bias_cycles")
        .agg(
            mean_detection=("affected_detection_percent", "mean"),
            std_detection=("affected_detection_percent", "std"),
            mean_control_alarm=("control_alarm_percent", "mean"),
            std_control_alarm=("control_alarm_percent", "std"),
            mean_normal_alarm=("normal_alarm_rate_percent", "mean"),
            mean_MAE=("validation_MAE", "mean"),
            std_MAE=("validation_MAE", "std"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "repeated_split_summary.csv",
        index=False,
    )

    metrics = {
        "experiment": "repeated_engine_split_robustness",
        "seeds": SPLIT_SEEDS,
        "training_engines_per_split": 64,
        "calibration_engines_per_split": 16,
        "fixed_validation_engines": validation_ids,
        "note": (
            "Repeated training/calibration partitions share the "
            "same 20 validation engines. Results measure split "
            "sensitivity, not independent external replications."
        ),
        "summary": summary.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
    }

    with open(
        OUTPUT_DIR / "repeated_split_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\n========== REPEATED SPLIT SUMMARY ==========")
    print(summary.round(3).to_string(index=False))

    print("\nGenerated files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)

    print("\nRepeated split experiment completed.")


if __name__ == "__main__":
    main()

"""
PrognosAI - Repeated Engine Split Robustness Experiment

Five different splits of the original 80 training engines:
    64 model-training engines
    16 calibration engines

The same 20 previously held-out validation engines are
used for each repetition.

This tests sensitivity to partition choices, not five
independent external validation cohorts.

The original baseline and official test set are untouched.
"""

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/processed/FD001_train_with_RUL.csv"
BASELINE_METRICS = ROOT / "artifacts/baseline/metrics.json"
VALIDATION_FILE = ROOT / "artifacts/baseline/validation_predictions.csv"

OUTPUT_DIR = ROOT / "artifacts/drift/repeated_split"

SPLIT_SEEDS = [42, 7, 21, 100, 2026]
BIAS_LEVELS = [0, 5, 10, 20]
ALPHA = 0.10

STAGE_BINS = [0, 30, 60, 90, 125]
STAGE_LABELS = ["Critical", "Late-middle", "Middle", "Early"]

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


def assign_stage(df):
    result = df.copy()
    result["stage"] = pd.cut(
        result["actual_RUL"],
        bins=STAGE_BINS,
        labels=STAGE_LABELS,
        include_lowest=True,
    )
    return result.dropna(subset=["stage"]).assign(
        stage=lambda x: x["stage"].astype(str)
    )


def engine_stage_metrics(df):
    work = df.copy()
    work["residual"] = work["actual_RUL"] - work["predicted_RUL"]
    work["absolute_residual"] = work["residual"].abs()

    return (
        work.groupby(["engine_id", "stage"], observed=True)
        .agg(
            mean_residual=("residual", "mean"),
            mae=("absolute_residual", "mean"),
            observations=("residual", "size"),
        )
        .reset_index()
    )


def conformal_threshold(scores):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    n = len(scores)
    rank = math.ceil((n + 1) * (1 - ALPHA))

    if n == 0:
        raise ValueError("No calibration scores.")

    if rank > n:
        return float("inf")

    return float(np.sort(scores)[rank - 1])


def build_thresholds(calibration_metrics):
    rows = []

    for stage in STAGE_LABELS:
        part = calibration_metrics[
            calibration_metrics["stage"] == stage
        ]

        if part["engine_id"].nunique() < 10:
            raise ValueError(
                f"Insufficient calibration engines for {stage}"
            )

        center_residual = float(part["mean_residual"].median())
        center_mae = float(part["mae"].median())

        residual_threshold = conformal_threshold(
            np.abs(part["mean_residual"] - center_residual)
        )

        mae_threshold = conformal_threshold(
            np.abs(part["mae"] - center_mae)
        )

        rows.append({
            "stage": stage,
            "center_residual": center_residual,
            "center_mae": center_mae,
            "residual_threshold": residual_threshold,
            "mae_threshold": mae_threshold,
            "calibration_engines": int(
                part["engine_id"].nunique()
            ),
        })

    return pd.DataFrame(rows)


def flag_engines(validation_metrics, thresholds):
    merged = validation_metrics.merge(
        thresholds,
        on="stage",
        how="left",
        validate="many_to_one",
    )

    merged["residual_alarm"] = (
        (merged["mean_residual"] - merged["center_residual"]).abs()
        > merged["residual_threshold"]
    )

    merged["mae_alarm"] = (
        (merged["mae"] - merged["center_mae"]).abs()
        > merged["mae_threshold"]
    )

    merged["stage_alarm"] = (
        merged["residual_alarm"] | merged["mae_alarm"]
    )

    engine_flags = (
        merged.groupby("engine_id")
        .agg(
            drift_detected=("stage_alarm", "any"),
            stages_flagged=("stage_alarm", "sum"),
        )
        .reset_index()
    )

    return merged, engine_flags


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FD001 data...")

    data = pd.read_csv(DATA_FILE)

    with open(BASELINE_METRICS, "r") as f:
        baseline_config = json.load(f)

    original_validation = pd.read_csv(VALIDATION_FILE)

    features = baseline_config["features"]

    validation_ids = sorted(
        original_validation["engine_id"].astype(int).unique()
    )

    all_ids = sorted(data["engine_id"].astype(int).unique())

    train_pool = sorted(set(all_ids) - set(validation_ids))

    if len(train_pool) != 80 or len(validation_ids) != 20:
        raise ValueError("Unexpected engine split.")

    validation_data = data[
        data["engine_id"].isin(validation_ids)
    ].copy()

    validation_data = validation_data.sort_values(
        ["engine_id", "cycle"]
    )

    validation_base = validation_data[
        ["engine_id", "cycle", "RUL_capped"]
    ].rename(columns={"RUL_capped": "actual_RUL"}).copy()

    results = []

    for seed in SPLIT_SEEDS:

        print("\n" + "=" * 55)
        print(f"EXPERIMENT SEED: {seed}")
        print("=" * 55)

        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(train_pool)

        model_ids = sorted(shuffled[:64].tolist())
        calibration_ids = sorted(shuffled[64:].tolist())

        model_data = data[
            data["engine_id"].isin(model_ids)
        ]

        calibration_data = data[
            data["engine_id"].isin(calibration_ids)
        ].sort_values(["engine_id", "cycle"])

        model = RandomForestRegressor(**MODEL_PARAMS)

        model.fit(
            model_data[features],
            model_data["RUL_capped"],
        )

        # Calibration predictions
        cal = calibration_data[
            ["engine_id", "cycle", "RUL_capped"]
        ].rename(columns={"RUL_capped": "actual_RUL"}).copy()

        cal["predicted_RUL"] = model.predict(
            calibration_data[features]
        )

        cal = assign_stage(cal)

        cal_metrics = engine_stage_metrics(cal)
        thresholds = build_thresholds(cal_metrics)

        thresholds.to_csv(
            OUTPUT_DIR / f"thresholds_seed_{seed}.csv",
            index=False,
        )

        # Independent validation predictions
        val = validation_base.copy()

        val["predicted_RUL"] = model.predict(
            validation_data[features]
        )

        val = assign_stage(val)

        mae = mean_absolute_error(
            val["actual_RUL"], val["predicted_RUL"]
        )

        rmse = np.sqrt(mean_squared_error(
            val["actual_RUL"], val["predicted_RUL"]
        ))

        r2 = r2_score(
            val["actual_RUL"], val["predicted_RUL"]
        )

        # Randomly select half the validation engines
        val_ids = np.sort(val["engine_id"].unique())
        affected = set(
            rng.permutation(val_ids)[:len(val_ids) // 2]
        )

        controls = set(val_ids) - affected

        # Normal validation alarm rate
        normal_metrics = engine_stage_metrics(val)
        _, normal_flags = flag_engines(
            normal_metrics, thresholds
        )

        normal_alarm_rate = (
            normal_flags["drift_detected"].mean() * 100
        )

        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation R2: {r2:.4f}")
        print(
            "Normal alarms:",
            f"{normal_alarm_rate:.2f}%",
        )

        for bias in BIAS_LEVELS:

            simulated = val.copy()

            injection_mask = (
                simulated["engine_id"].isin(affected)
                & simulated["stage"].isin(
                    ["Critical", "Late-middle", "Middle"]
                )
            )

            simulated.loc[
                injection_mask,
                "predicted_RUL",
            ] += bias

            sim_metrics = engine_stage_metrics(simulated)

            stage_flags, engine_flags = flag_engines(
                sim_metrics,
                thresholds,
            )

            affected_flags = engine_flags[
                engine_flags["engine_id"].isin(affected)
            ]

            control_flags = engine_flags[
                engine_flags["engine_id"].isin(controls)
            ]

            affected_rate = (
                affected_flags["drift_detected"].mean() * 100
            )

            control_rate = (
                control_flags["drift_detected"].mean() * 100
            )

            row = {
                "seed": seed,
                "bias_cycles": bias,
                "validation_MAE": mae,
                "validation_RMSE": rmse,
                "validation_R2": r2,
                "normal_alarm_rate_percent": normal_alarm_rate,
                "affected_detection_percent": affected_rate,
                "control_alarm_percent": control_rate,
                "affected_engines_flagged": int(
                    affected_flags["drift_detected"].sum()
                ),
                "control_engines_flagged": int(
                    control_flags["drift_detected"].sum()
                ),
            }

            results.append(row)

            print(
                f"Bias {bias:>2} | "
                f"Affected: {affected_rate:>5.1f}% | "
                f"Controls: {control_rate:>5.1f}%"
            )

        joblib.dump(
            model,
            OUTPUT_DIR / f"research_model_seed_{seed}.joblib",
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "repeated_split_results.csv",
        index=False,
    )

    # Aggregate across repeated partition choices.
    summary = (
        results_df.groupby("bias_cycles")
        .agg(
            mean_detection=("affected_detection_percent", "mean"),
            std_detection=("affected_detection_percent", "std"),
            mean_control_alarm=("control_alarm_percent", "mean"),
            std_control_alarm=("control_alarm_percent", "std"),
            mean_normal_alarm=("normal_alarm_rate_percent", "mean"),
            mean_MAE=("validation_MAE", "mean"),
            std_MAE=("validation_MAE", "std"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "repeated_split_summary.csv",
        index=False,
    )

    metrics = {
        "experiment": "repeated_engine_split_robustness",
        "seeds": SPLIT_SEEDS,
        "training_engines_per_split": 64,
        "calibration_engines_per_split": 16,
        "fixed_validation_engines": validation_ids,
        "note": (
            "Repeated training/calibration partitions share the "
            "same 20 validation engines. Results measure split "
            "sensitivity, not independent external replications."
        ),
        "summary": summary.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
    }

    with open(
        OUTPUT_DIR / "repeated_split_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\n========== REPEATED SPLIT SUMMARY ==========")
    print(summary.round(3).to_string(index=False))

    print("\nGenerated files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)

    print("\nRepeated split experiment completed.")


if __name__ == "__main__":
    main()
