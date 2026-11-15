
"""
PrognosAI - Stage-Aware Residual Drift Experiment

Offline research experiment using held-out validation predictions.

RUL stages:
    Early:       91-125
    Middle:      61-90
    Late-middle: 31-60
    Critical:    0-30

Residual:
    actual_RUL - predicted_RUL

Synthetic bias is applied to selected engines only.
Control engines remain unchanged.

Important:
    Actual RUL is used for stage assignment. This is an
    offline evaluation, not a deployable online detector.
"""

from pathlib import Path
import json

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

INPUT_FILE = (
    ROOT / "artifacts" / "baseline"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "artifacts" / "drift"
    / "stage_aware_residual"
)

RANDOM_STATE = 42
ALPHA = 0.05

# Artificial increase in predicted RUL.
BIAS_LEVELS = [0, 3, 5, 10, 15, 20]

# Minimum observations required for a stage-level comparison.
MIN_STAGE_OBSERVATIONS = 10

STAGE_BINS = [0, 30, 60, 90, 125]

STAGE_LABELS = [
    "Critical",
    "Late-middle",
    "Middle",
    "Early",
]


# ---------------------------------------------------------
# STATISTICAL HELPERS
# ---------------------------------------------------------

def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg multiple testing correction."""

    p_values = np.asarray(p_values, dtype=float)

    decisions = np.zeros(len(p_values), dtype=bool)

    valid = np.isfinite(p_values)

    if not valid.any():
        return decisions

    indices = np.where(valid)[0]
    values = p_values[valid]

    order = np.argsort(values)
    sorted_values = values[order]

    m = len(sorted_values)

    thresholds = (
        alpha * np.arange(1, m + 1) / m
    )

    passing = np.where(
        sorted_values <= thresholds
    )[0]

    if len(passing):
        cutoff = sorted_values[passing[-1]]
        decisions[indices] = values <= cutoff

    return decisions


def get_stage_metrics(df):
    """Calculate error metrics by RUL stage."""

    rows = []

    for stage in STAGE_LABELS:

        subset = df[df["stage"] == stage]

        if subset.empty:
            continue

        residual = subset["residual"]

        rows.append({
            "stage": stage,
            "observations": len(subset),
            "engines": subset["engine_id"].nunique(),
            "mean_residual": residual.mean(),
            "residual_std": residual.std(),
            "MAE": residual.abs().mean(),
            "RMSE": np.sqrt(np.mean(residual ** 2)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# STAGE-AWARE DETECTION
# ---------------------------------------------------------

def detect_stage_drift(df):
    """
    Compare affected and control residual distributions
    within matching RUL stages.

    Each engine contributes observations within a stage.
    One KS test is conducted per engine and stage.
    """

    results = []

    for engine_id, engine_df in df.groupby("engine_id"):

        for stage in STAGE_LABELS:

            stage_df = engine_df[
                engine_df["stage"] == stage
            ]

            # Compare early and late portions within the
            # same engine and the same RUL stage.
            stage_df = stage_df.sort_values(
                "actual_RUL",
                ascending=False,
            )

            n = len(stage_df)

            if n < MIN_STAGE_OBSERVATIONS * 2:
                continue

            midpoint = n // 2

            reference = stage_df.iloc[:midpoint][
                "residual"
            ].dropna().values

            monitoring = stage_df.iloc[midpoint:][
                "residual"
            ].dropna().values

            if (
                len(reference) < MIN_STAGE_OBSERVATIONS
                or len(monitoring) < MIN_STAGE_OBSERVATIONS
            ):
                continue

            test = ks_2samp(
                reference,
                monitoring,
                alternative="two-sided",
                method="auto",
            )

            results.append({
                "engine_id": int(engine_id),
                "stage": stage,
                "reference_size": len(reference),
                "monitoring_size": len(monitoring),
                "reference_mean": float(
                    np.mean(reference)
                ),
                "monitoring_mean": float(
                    np.mean(monitoring)
                ),
                "reference_mae": float(
                    np.mean(np.abs(reference))
                ),
                "monitoring_mae": float(
                    np.mean(np.abs(monitoring))
                ),
                "ks_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
            })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    # Correct tests across stages and engines.
    results_df["detected"] = benjamini_hochberg(
        results_df["p_value"].values,
        alpha=ALPHA,
    )

    return results_df


def engine_detection_summary(results, affected_engines):

    if results.empty:
        return {
            "affected_detection_rate": 0.0,
            "control_alarm_rate": 0.0,
        }

    affected = results[
        results["engine_id"].isin(affected_engines)
    ]

    control = results[
        ~results["engine_id"].isin(affected_engines)
    ]

    # Engine-level aggregation:
    # an engine is flagged if any stage is flagged.
    affected_flags = (
        affected.groupby("engine_id")["detected"]
        .any()
    )

    control_flags = (
        control.groupby("engine_id")["detected"]
        .any()
    )

    return {
        "affected_detection_rate": (
            float(affected_flags.mean() * 100)
            if len(affected_flags) else 0.0
        ),
        "control_alarm_rate": (
            float(control_flags.mean() * 100)
            if len(control_flags) else 0.0
        ),
        "affected_engines_evaluated": len(
            affected_flags
        ),
        "control_engines_evaluated": len(
            control_flags
        ),
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("Loading validation predictions...")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "actual_RUL",
        "predicted_RUL",
        "engine_id",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    if df[required].isnull().any().any():
        raise ValueError(
            "Missing values found in validation predictions."
        )

    # Keep only capped target range.
    df = df[
        df["actual_RUL"].between(0, 125)
    ].copy()

    df["engine_id"] = df["engine_id"].astype(int)

    # Stage assignment is based on actual RUL.
    df["stage"] = pd.cut(
        df["actual_RUL"],
        bins=STAGE_BINS,
        labels=STAGE_LABELS,
        include_lowest=True,
    )

    df = df.dropna(subset=["stage"]).copy()

    df["stage"] = df["stage"].astype(str)

    df["residual"] = (
        df["actual_RUL"] - df["predicted_RUL"]
    )

    print("\n========== DATA OVERVIEW ==========")
    print("Observations:", len(df))
    print("Engines:", df["engine_id"].nunique())

    print("\n========== BASELINE STAGE METRICS ==========")

    stage_metrics = get_stage_metrics(df)

    print(
        stage_metrics.round(3).to_string(
            index=False
        )
    )

    stage_metrics.to_csv(
        OUTPUT_DIR / "stage_baseline_metrics.csv",
        index=False,
    )

    # -----------------------------------------------------
    # ENGINE SPLIT
    # -----------------------------------------------------

    rng = np.random.default_rng(RANDOM_STATE)

    engines = np.sort(df["engine_id"].unique())

    shuffled = rng.permutation(engines)

    affected_engines = set(
        shuffled[:len(shuffled) // 2]
    )

    control_engines = set(engines) - affected_engines

    print("\nAffected engines:", sorted(affected_engines))
    print("Control engines:", sorted(control_engines))

    # -----------------------------------------------------
    # SIMULATION
    # -----------------------------------------------------

    simulation_rows = []

    for bias in BIAS_LEVELS:

        simulated = df.copy()

        affected_mask = (
            simulated["engine_id"].isin(
                affected_engines
            )
        )

        # Apply synthetic bias only to middle and later
        # degradation stages, not the early stage.
        injection_mask = (
            affected_mask
            & simulated["stage"].isin(
                ["Middle", "Late-middle", "Critical"]
            )
        )

        simulated.loc[
            injection_mask,
            "predicted_RUL",
        ] += bias

        simulated["residual"] = (
            simulated["actual_RUL"]
            - simulated["predicted_RUL"]
        )

        results = detect_stage_drift(simulated)

        if results.empty:
            print(
                f"No valid comparisons for bias {bias}"
            )
            continue

        summary = engine_detection_summary(
            results,
            affected_engines,
        )

        summary["injected_bias_cycles"] = bias

        simulation_rows.append(summary)

        results.to_csv(
            OUTPUT_DIR
            / f"stage_detection_bias_{bias}.csv",
            index=False,
        )

        print(f"\nInjected bias: {bias} cycles")

        print(
            "Affected-engine detection:",
            f"{summary['affected_detection_rate']:.2f}%",
        )

        print(
            "Control-engine alarm rate:",
            f"{summary['control_alarm_rate']:.2f}%",
        )

    simulation_metrics = pd.DataFrame(
        simulation_rows
    )

    simulation_metrics.to_csv(
        OUTPUT_DIR / "stage_aware_simulation_results.csv",
        index=False,
    )

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        stage_metrics["stage"],
        stage_metrics["MAE"],
    )

    plt.title("RUL Prediction MAE by Degradation Stage")
    plt.xlabel("Degradation Stage")
    plt.ylabel("MAE (cycles)")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "stage_mae.png",
        dpi=150,
    )

    plt.close()

    if not simulation_metrics.empty:

        plt.figure(figsize=(9, 5))

        plt.plot(
            simulation_metrics[
                "injected_bias_cycles"
            ],
            simulation_metrics[
                "affected_detection_rate"
            ],
            marker="o",
            label="Affected engine detection",
        )

        plt.plot(
            simulation_metrics[
                "injected_bias_cycles"
            ],
            simulation_metrics[
                "control_alarm_rate"
            ],
            marker="s",
            label="Control engine alarm rate",
        )

        plt.xlabel("Injected prediction bias (cycles)")
        plt.ylabel("Engine-level rate (%)")
        plt.title(
            "Stage-Aware Residual Drift Sensitivity"
        )

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "stage_aware_sensitivity.png",
            dpi=150,
        )

        plt.close()

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    metrics = {
        "experiment": "stage_aware_residual_drift",
        "dataset": "NASA C-MAPSS FD001",
        "data_source": "Held-out validation predictions",
        "validation_engines": int(
            df["engine_id"].nunique()
        ),
        "validation_observations": int(len(df)),
        "stage_definitions": {
            "Critical": "0-30",
            "Late-middle": "31-60",
            "Middle": "61-90",
            "Early": "91-125",
        },
        "affected_engines": sorted(
            int(x) for x in affected_engines
        ),
        "control_engines": sorted(
            int(x) for x in control_engines
        ),
        "alpha": ALPHA,
        "simulation_results": simulation_rows,
        "limitations": [
            "Actual RUL is used to assign degradation stages.",
            "The experiment is offline, not a deployable online detector.",
            "Only 20 validation engines are available.",
            "Within-engine comparisons have limited sample sizes.",
            "Synthetic prediction bias is not real industrial drift.",
            "Results require independent validation.",
        ],
    }

    with open(
        OUTPUT_DIR / "stage_aware_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\n========== GENERATED FILES ==========")

    for file in sorted(OUTPUT_DIR.iterdir()):
        print(file.name)

    print("\nStage-aware residual experiment completed.")


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    main()
