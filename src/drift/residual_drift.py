
"""
PrognosAI - Degradation-Aware Residual Drift Detection

Uses held-out validation predictions from the frozen Random Forest model.
No model retraining or official test-set tuning is performed.

Residual = actual_RUL - predicted_RUL

Positive residual: model underpredicts RUL (conservative).
Negative residual: model overpredicts RUL (optimistic).

Synthetic experiment:
- Inject an additional prediction bias after cycle 80.
- Only half of validation engines are affected.
- Remaining engines act as controls.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA = (
    ROOT / "data" / "processed" / "FD001_train_with_RUL.csv"
)

VALIDATION_PREDICTIONS = (
    ROOT / "artifacts" / "baseline" / "validation_predictions.csv"
)

OUTPUT_DIR = ROOT / "artifacts" / "drift" / "residual"

REFERENCE_CYCLES = 30
WINDOW_SIZE = 20
STEP_SIZE = 10

INJECTION_CYCLE = 80

# Additional bias in predicted RUL, measured in cycles.
BIAS_LEVELS = [0, 5, 10, 15, 20]

ALPHA = 0.05

RANDOM_STATE = 42


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def benjamini_hochberg(p_values, alpha=0.05):
    """Return BH-adjusted significance decisions."""

    p_values = np.asarray(p_values, dtype=float)

    if len(p_values) == 0:
        return np.array([], dtype=bool)

    valid = np.isfinite(p_values)
    decisions = np.zeros(len(p_values), dtype=bool)

    if not valid.any():
        return decisions

    valid_indices = np.where(valid)[0]
    valid_p = p_values[valid]

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    m = len(sorted_p)
    thresholds = alpha * np.arange(1, m + 1) / m

    passing = np.where(sorted_p <= thresholds)[0]

    if len(passing) > 0:
        cutoff = sorted_p[passing[-1]]
        decisions[valid_indices] = valid_p <= cutoff

    return decisions


def calculate_windows(engine_df):
    """Generate rolling monitoring windows for one engine."""

    windows = []

    max_cycle = int(engine_df["cycle"].max())

    start_cycle = REFERENCE_CYCLES + 1

    while start_cycle + WINDOW_SIZE - 1 <= max_cycle:

        end_cycle = start_cycle + WINDOW_SIZE - 1

        window = engine_df[
            (engine_df["cycle"] >= start_cycle)
            & (engine_df["cycle"] <= end_cycle)
        ]

        if len(window) >= WINDOW_SIZE:
            windows.append(
                {
                    "start_cycle": start_cycle,
                    "end_cycle": end_cycle,
                    "data": window.copy(),
                }
            )

        start_cycle += STEP_SIZE

    return windows


def detect_engine_windows(engine_df, residual_column):
    """
    Compare each rolling residual window with the engine's
    own first 30 cycles using a two-sample KS test.
    """

    reference = engine_df[
        engine_df["cycle"] <= REFERENCE_CYCLES
    ][residual_column].dropna().values

    if len(reference) < 10:
        return []

    results = []

    for window_info in calculate_windows(engine_df):

        current = window_info["data"][
            residual_column
        ].dropna().values

        if len(current) < 10:
            continue

        test = ks_2samp(
            reference,
            current,
            alternative="two-sided",
            method="auto",
        )

        results.append(
            {
                "engine_id": int(engine_df["engine_id"].iloc[0]),
                "start_cycle": window_info["start_cycle"],
                "end_cycle": window_info["end_cycle"],
                "reference_size": len(reference),
                "window_size": len(current),
                "reference_mean": float(np.mean(reference)),
                "window_mean": float(np.mean(current)),
                "reference_mae": float(np.mean(np.abs(reference))),
                "window_mae": float(np.mean(np.abs(current))),
                "ks_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )

    if not results:
        return []

    # Correct multiple rolling-window tests within each engine.
    decisions = benjamini_hochberg(
        [item["p_value"] for item in results],
        alpha=ALPHA,
    )

    for item, detected in zip(results, decisions):
        item["drift_detected"] = bool(detected)

    return results


def monitor_all_engines(data, residual_column):
    all_results = []

    for engine_id, engine_df in data.groupby("engine_id"):

        engine_df = engine_df.sort_values("cycle")

        results = detect_engine_windows(
            engine_df,
            residual_column,
        )

        all_results.extend(results)

    return pd.DataFrame(all_results)


def detection_rate(df):
    if df.empty:
        return 0.0

    return float(df["drift_detected"].mean() * 100)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def main():

    print("Loading validation predictions...")
    predictions = pd.read_csv(VALIDATION_PREDICTIONS)

    print("Loading processed FD001 data...")
    processed = pd.read_csv(PROCESSED_DATA)

    required_prediction_columns = {
        "actual_RUL",
        "predicted_RUL",
        "engine_id",
    }

    required_processed_columns = {
        "engine_id",
        "cycle",
        "RUL_capped",
    }

    if not required_prediction_columns.issubset(
        predictions.columns
    ):
        raise ValueError(
            "Validation predictions are missing required columns."
        )

    if not required_processed_columns.issubset(
        processed.columns
    ):
        raise ValueError(
            "Processed dataset is missing engine_id, cycle, "
            "or RUL_capped."
        )

    predictions["engine_id"] = predictions[
        "engine_id"
    ].astype(int)

    processed["engine_id"] = processed[
        "engine_id"
    ].astype(int)

    # Preserve the existing row order within each engine.
    predictions["row_in_engine"] = (
        predictions.groupby("engine_id").cumcount()
    )

    processed["row_in_engine"] = (
        processed.groupby("engine_id").cumcount()
    )

    # Keep only the held-out validation engines.
    validation_engines = predictions[
        "engine_id"
    ].unique()

    processed_validation = processed[
        processed["engine_id"].isin(validation_engines)
    ].copy()

    # Align each prediction with its original engine cycle.
    data = predictions.merge(
        processed_validation[
            [
                "engine_id",
                "row_in_engine",
                "cycle",
                "RUL_capped",
            ]
        ],
        on=["engine_id", "row_in_engine"],
        how="left",
        validate="one_to_one",
    )

    if data["cycle"].isna().any():
        raise ValueError(
            "Could not align all validation predictions "
            "with original engine cycles."
        )

    # Strict label verification prevents silent row misalignment.
    if not np.allclose(
        data["actual_RUL"].values,
        data["RUL_capped"].values,
    ):
        raise ValueError(
            "Actual RUL values do not match the processed "
            "RUL_capped sequence. Stop: row alignment needs investigation."
        )

    data = data.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    data["residual"] = (
        data["actual_RUL"] - data["predicted_RUL"]
    )

    data["absolute_residual"] = data[
        "residual"
    ].abs()

    print("\n========== VALIDATION DATA ==========")
    print("Observations:", len(data))
    print("Validation engines:", data["engine_id"].nunique())
    print("Cycle range:", data["cycle"].min(), "-", data["cycle"].max())

    print("\nResidual summary:")
    print(data["residual"].describe().round(3))

    print(
        "\nOverall validation MAE:",
        round(data["absolute_residual"].mean(), 4),
    )

    # -----------------------------------------------------
    # BASELINE TEMPORAL RESIDUAL MONITORING
    # -----------------------------------------------------

    print("\n========== BASELINE RESIDUAL MONITORING ==========")

    baseline_results = monitor_all_engines(
        data,
        "residual",
    )

    if baseline_results.empty:
        raise ValueError(
            "No baseline monitoring windows were generated."
        )

    baseline_results.to_csv(
        OUTPUT_DIR / "residual_baseline_windows.csv",
        index=False,
    )

    baseline_late = baseline_results[
        baseline_results["start_cycle"] >= INJECTION_CYCLE
    ]

    print("Monitoring windows:", len(baseline_results))
    print(
        "All-window detection rate:",
        f"{detection_rate(baseline_results):.2f}%",
    )

    print(
        "Detection rate after cycle 80:",
        f"{detection_rate(baseline_late):.2f}%",
    )

    # -----------------------------------------------------
    # SYNTHETIC BIAS EXPERIMENT
    # -----------------------------------------------------

    print("\n========== SYNTHETIC RESIDUAL DRIFT ==========")

    rng = np.random.default_rng(RANDOM_STATE)

    engine_ids = np.sort(data["engine_id"].unique())

    shuffled = rng.permutation(engine_ids)

    affected_engines = set(
        shuffled[: max(1, len(shuffled) // 2)]
    )

    print("Affected engines:", sorted(affected_engines))
    print("Control engines:", sorted(set(engine_ids) - affected_engines))
    print("Injection cycle:", INJECTION_CYCLE)

    simulation_rows = []
    simulation_window_results = {}

    for bias in BIAS_LEVELS:

        print(f"\nTesting additional prediction bias: {bias} cycles")

        simulated = data.copy()

        affected_mask = (
            simulated["engine_id"].isin(affected_engines)
            & (simulated["cycle"] >= INJECTION_CYCLE)
        )

        # Increase predicted RUL by the injected bias.
        # This creates an optimistic prediction error.
        simulated.loc[
            affected_mask,
            "predicted_RUL",
        ] += bias

        simulated["simulated_residual"] = (
            simulated["actual_RUL"]
            - simulated["predicted_RUL"]
        )

        results = monitor_all_engines(
            simulated,
            "simulated_residual",
        )

        if results.empty:
            continue

        late_results = results[
            results["start_cycle"] >= INJECTION_CYCLE
        ].copy()

        affected_late = late_results[
            late_results["engine_id"].isin(affected_engines)
        ]

        control_late = late_results[
            ~late_results["engine_id"].isin(affected_engines)
        ]

        affected_rate = detection_rate(affected_late)
        control_rate = detection_rate(control_late)

        affected_window_mae = (
            affected_late["window_mae"].mean()
            if not affected_late.empty else np.nan
        )

        simulation_rows.append(
            {
                "injected_bias_cycles": bias,
                "affected_engine_detection_rate": affected_rate,
                "control_engine_alarm_rate": control_rate,
                "affected_windows": len(affected_late),
                "control_windows": len(control_late),
                "affected_mean_window_mae": affected_window_mae,
            }
        )

        simulation_window_results[bias] = results

        print(
            "Affected-engine detection:",
            f"{affected_rate:.2f}%",
        )

        print(
            "Control-engine alarm rate:",
            f"{control_rate:.2f}%",
        )

    simulation_metrics = pd.DataFrame(simulation_rows)

    simulation_metrics.to_csv(
        OUTPUT_DIR / "residual_simulation_results.csv",
        index=False,
    )

    # -----------------------------------------------------
    # SAVE ENGINE-LEVEL RESIDUAL SUMMARY
    # -----------------------------------------------------

    engine_summary = (
        data.groupby("engine_id")
        .agg(
            observations=("residual", "size"),
            mean_residual=("residual", "mean"),
            mae=("absolute_residual", "mean"),
            residual_std=("residual", "std"),
            first_cycle=("cycle", "min"),
            last_cycle=("cycle", "max"),
        )
        .reset_index()
    )

    engine_summary.to_csv(
        OUTPUT_DIR / "residual_engine_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.hist(
        data["residual"],
        bins=40,
        edgecolor="black",
    )

    plt.axvline(0, linestyle="--")

    plt.title("Validation Residual Distribution")
    plt.xlabel("Actual RUL - Predicted RUL")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "residual_distribution.png",
        dpi=150,
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(
        simulation_metrics["injected_bias_cycles"],
        simulation_metrics[
            "affected_engine_detection_rate"
        ],
        marker="o",
        label="Affected engine detection",
    )

    plt.plot(
        simulation_metrics["injected_bias_cycles"],
        simulation_metrics[
            "control_engine_alarm_rate"
        ],
        marker="s",
        label="Control engine alarm rate",
    )

    plt.xlabel("Injected prediction bias (RUL cycles)")
    plt.ylabel("Detection / alarm rate (%)")
    plt.title("Residual Drift Detection Sensitivity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "residual_drift_sensitivity.png",
        dpi=150,
    )

    plt.close()

    # -----------------------------------------------------
    # METRICS JSON
    # -----------------------------------------------------

    metrics = {
        "experiment": "degradation_aware_residual_drift",
        "dataset": "NASA C-MAPSS FD001",
        "model": "Frozen RandomForestRegressor",
        "data_source": "Held-out validation predictions",
        "validation_engines": int(data["engine_id"].nunique()),
        "validation_observations": int(len(data)),
        "reference_cycles": REFERENCE_CYCLES,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "injection_cycle": INJECTION_CYCLE,
        "alpha": ALPHA,
        "baseline_mae": float(
            data["absolute_residual"].mean()
        ),
        "baseline_window_count": int(len(baseline_results)),
        "baseline_detection_rate_percent": detection_rate(
            baseline_results
        ),
        "baseline_late_detection_rate_percent": detection_rate(
            baseline_late
        ),
        "affected_engine_ids": sorted(
            int(x) for x in affected_engines
        ),
        "simulation_results": simulation_rows,
        "limitations": [
            "Residuals are evaluated on held-out validation engines.",
            "Synthetic bias is an experimental perturbation, not observed industrial drift.",
            "The KS detector compares early-life residuals with later windows.",
            "Normal degradation may change residual distributions.",
            "Overlapping windows are statistically dependent.",
            "Results do not establish real-world false alarm rates.",
        ],
    }

    with open(
        OUTPUT_DIR / "residual_drift_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\n========== FILES GENERATED ==========")

    for file in sorted(OUTPUT_DIR.iterdir()):
        print(file.name)

    print("\nResidual drift experiment completed successfully.")


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    main()
