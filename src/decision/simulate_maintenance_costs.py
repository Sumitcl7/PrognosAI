
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

COMPARISON_FILE = (
    ROOT / "artifacts" / "decision" /
    "v1_v2_comparison.csv"
)

ACTUAL_FILE = (
    ROOT / "artifacts" / "uncertainty" /
    "evaluation" / "conformal_test_final_predictions.csv"
)

OUTPUT_DIR = ROOT / "artifacts" / "decision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# All costs are hypothetical INR scenario assumptions.
BASE_COSTS = {
    "plan": 8000,
    "schedule_soon": 12000,
    "urgent": 18000,
    "inspection": 1000,
    "missed_maintenance": 100000,
}

MAINTENANCE_NEED_RUL = 30

# Sensitivity scenarios.
MISSED_PENALTIES = [50000, 100000, 200000]
INSPECTION_COSTS = [500, 1000, 2000]


def rul_only_action(predicted_rul):
    """RUL-only baseline using the V2 point-RUL thresholds."""

    if predicted_rul <= 15:
        return "URGENT"

    if predicted_rul <= 30:
        return "SCHEDULE_SOON"

    if predicted_rul <= 60:
        return "PLAN"

    return "MONITOR"


def normalize_v1_action(priority):
    mapping = {
        "MONITOR": "MONITOR",
        "PLAN": "PLAN",
        "SCHEDULE_SOON": "SCHEDULE_SOON",
        "URGENT": "URGENT",
    }

    if priority not in mapping:
        raise ValueError(f"Unknown V1 priority: {priority}")

    return mapping[priority]


def normalize_v2_action(action):
    mapping = {
        "CONTINUE_MONITORING": "MONITOR",
        "PLAN_MAINTENANCE": "PLAN",
        "SCHEDULE_MAINTENANCE_SOON": "SCHEDULE_SOON",
        "URGENT_MAINTENANCE_REVIEW": "URGENT",
    }

    if action not in mapping:
        raise ValueError(f"Unknown V2 action: {action}")

    return mapping[action]


def calculate_engine_cost(
    action,
    actual_rul,
    model_review_required,
    missed_penalty,
    inspection_cost,
):
    """Calculate simulated costs for one engine."""

    intervention = action != "MONITOR"
    maintenance_needed = actual_rul <= MAINTENANCE_NEED_RUL

    maintenance_costs = {
        "MONITOR": 0,
        "PLAN": BASE_COSTS["plan"],
        "SCHEDULE_SOON": BASE_COSTS["schedule_soon"],
        "URGENT": BASE_COSTS["urgent"],
    }

    maintenance_cost = maintenance_costs[action]

    inspection_expense = (
        inspection_cost if model_review_required else 0
    )

    missed = maintenance_needed and not intervention

    missed_cost = missed_penalty if missed else 0

    return {
        "action": action,
        "maintenance_needed_reference": maintenance_needed,
        "maintenance_intervention": intervention,
        "model_review_required": model_review_required,
        "maintenance_cost": maintenance_cost,
        "inspection_cost": inspection_expense,
        "missed_maintenance": missed,
        "missed_maintenance_cost": missed_cost,
        "intervention_outside_reference_need_band": (
            intervention and not maintenance_needed
        ),
        "total_simulated_cost": (
            maintenance_cost + inspection_expense + missed_cost
        ),
    }


def build_policy_records(df, missed_penalty, inspection_cost):
    records = []

    for _, row in df.iterrows():
        actual_rul = float(row["actual_RUL"])
        predicted_rul = float(row["predicted_rul"])

        policies = {
            "RUL_ONLY": (
                rul_only_action(predicted_rul),
                False,
            ),
            "V1": (
                normalize_v1_action(row["v1_priority"]),
                False,
            ),
            "V2": (
                normalize_v2_action(row["v2_action"]),
                bool(row["v2_model_review_required"]),
            ),
        }

        for policy, (action, review_required) in policies.items():
            cost = calculate_engine_cost(
                action=action,
                actual_rul=actual_rul,
                model_review_required=review_required,
                missed_penalty=missed_penalty,
                inspection_cost=inspection_cost,
            )

            records.append({
                "policy": policy,
                "engine_id": int(row["engine_id"]),
                "predicted_rul": predicted_rul,
                "actual_RUL_evaluation_only": actual_rul,
                "interval_width": float(row["interval_width"]),
                "missed_penalty_assumption": missed_penalty,
                "inspection_cost_assumption": inspection_cost,
                **cost,
            })

    return pd.DataFrame(records)


def summarize_costs(records):
    return (
        records.groupby("policy")
        .agg(
            engines=("engine_id", "count"),
            total_cost=("total_simulated_cost", "sum"),
            average_cost_per_engine=("total_simulated_cost", "mean"),
            maintenance_interventions=(
                "maintenance_intervention", "sum"
            ),
            model_reviews=("model_review_required", "sum"),
            missed_maintenance=("missed_maintenance", "sum"),
            interventions_outside_reference_need_band=(
                "intervention_outside_reference_need_band", "sum"
            ),
        )
        .reset_index()
    )


def main():
    if not COMPARISON_FILE.exists():
        raise FileNotFoundError(COMPARISON_FILE)

    if not ACTUAL_FILE.exists():
        raise FileNotFoundError(ACTUAL_FILE)

    comparison = pd.read_csv(COMPARISON_FILE)

    actual = pd.read_csv(ACTUAL_FILE)[
        ["engine_id", "actual_RUL"]
    ]

    if comparison["engine_id"].duplicated().any():
        raise ValueError("Duplicate engine IDs in comparison file.")

    if actual["engine_id"].duplicated().any():
        raise ValueError("Duplicate engine IDs in actual test file.")

    df = comparison.merge(
        actual,
        on="engine_id",
        how="left",
        validate="one_to_one",
    )

    if df["actual_RUL"].isna().any():
        raise ValueError("Some engines have no matching actual RUL.")

    # -------------------------------------------------
    # Base scenario
    # -------------------------------------------------

    base_records = build_policy_records(
        df,
        missed_penalty=BASE_COSTS["missed_maintenance"],
        inspection_cost=BASE_COSTS["inspection"],
    )

    base_records.to_csv(
        OUTPUT_DIR / "maintenance_cost_by_engine.csv",
        index=False,
    )

    base_summary = summarize_costs(base_records)

    base_summary.to_csv(
        OUTPUT_DIR / "maintenance_cost_summary.csv",
        index=False,
    )

    # -------------------------------------------------
    # Sensitivity analysis
    # -------------------------------------------------

    sensitivity_rows = []

    for missed_penalty in MISSED_PENALTIES:
        for inspection_cost in INSPECTION_COSTS:

            scenario_records = build_policy_records(
                df,
                missed_penalty=missed_penalty,
                inspection_cost=inspection_cost,
            )

            scenario_summary = summarize_costs(
                scenario_records
            )

            for _, row in scenario_summary.iterrows():
                sensitivity_rows.append({
                    "missed_penalty": missed_penalty,
                    "inspection_cost": inspection_cost,
                    "policy": row["policy"],
                    "total_cost": int(row["total_cost"]),
                    "average_cost_per_engine": float(
                        row["average_cost_per_engine"]
                    ),
                    "missed_maintenance": int(
                        row["missed_maintenance"]
                    ),
                    "maintenance_interventions": int(
                        row["maintenance_interventions"]
                    ),
                    "model_reviews": int(row["model_reviews"]),
                })

    sensitivity = pd.DataFrame(sensitivity_rows)

    sensitivity.to_csv(
        OUTPUT_DIR / "maintenance_cost_sensitivity.csv",
        index=False,
    )

    # -------------------------------------------------
    # JSON report
    # -------------------------------------------------

    summary = {
        "dataset": "NASA C-MAPSS FD001 official test",
        "engines": int(len(df)),
        "currency": "INR",
        "scenario_type": "hypothetical simulation",
        "maintenance_need_reference": (
            f"Actual RUL <= {MAINTENANCE_NEED_RUL} cycles"
        ),
        "cost_assumptions": BASE_COSTS,
        "base_results": base_summary.to_dict(orient="records"),
        "limitations": [
            "Costs are hypothetical, not actual industrial costs.",
            "Actual RUL is used only for offline evaluation.",
            "Actual RUL <= 30 is an experimental reference rule.",
            "The simulation is not a validated maintenance schedule.",
            "V2 inspection review is not itself a maintenance intervention.",
            "Test-set results do not establish future operational savings.",
        ],
    }

    with open(
        OUTPUT_DIR / "maintenance_cost_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4)

    # -------------------------------------------------
    # Terminal report
    # -------------------------------------------------

    print("\nPROGNOSAI MAINTENANCE COST SIMULATION")
    print("=" * 65)

    print(f"Engines evaluated: {len(df)}")
    print(f"Reference maintenance threshold: {MAINTENANCE_NEED_RUL} cycles")

    print("\nCOST ASSUMPTIONS")
    for name, value in BASE_COSTS.items():
        print(f"{name}: Rs. {value:,}")

    print("\nBASE SCENARIO RESULTS")
    print(
        base_summary.round(2).to_string(index=False)
    )

    print("\nSENSITIVITY ANALYSIS")
    print(
        sensitivity.round(2).to_string(index=False)
    )

    print("\nSaved reports:")
    print(OUTPUT_DIR / "maintenance_cost_by_engine.csv")
    print(OUTPUT_DIR / "maintenance_cost_summary.csv")
    print(OUTPUT_DIR / "maintenance_cost_summary.json")
    print(OUTPUT_DIR / "maintenance_cost_sensitivity.csv")


if __name__ == "__main__":
    main()
