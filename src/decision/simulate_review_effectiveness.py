
from pathlib import Path
import json

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "decision"
    / "maintenance_cost_by_engine.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "decision"
    / "review_effectiveness"
)

# Match the assumptions used in the original simulation
MISSED_MAINTENANCE_PENALTY = 100_000
INSPECTION_COST = 1_000

# Assumed cost when a review results in maintenance.
# This is a scenario assumption, not an observed cost.
REVIEW_TRIGGERED_MAINTENANCE_COST = 12_000

EFFECTIVENESS_LEVELS = [0, 25, 50, 75, 100]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "policy",
        "engine_id",
        "predicted_rul",
        "actual_RUL_evaluation_only",
        "model_review_required",
        "missed_maintenance",
        "total_simulated_cost",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# --------------------------------------------------
# REVIEW EFFECTIVENESS SIMULATION
# --------------------------------------------------

def simulate_review_effectiveness(df):

    v2 = df[df["policy"] == "V2"].copy()

    if v2.empty:
        raise ValueError("No V2 records found.")

    v2["model_review_required"] = (
        v2["model_review_required"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    if v2["model_review_required"].isna().any():
        raise ValueError(
            "Unexpected values in model_review_required."
        )

    v2["missed_maintenance"] = (
        v2["missed_maintenance"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    if v2["missed_maintenance"].isna().any():
        raise ValueError(
            "Unexpected values in missed_maintenance."
        )

    # Identify missed cases that received a model review flag.
    missed = v2[
        v2["missed_maintenance"]
        & v2["model_review_required"]
    ].copy()

    review_count = int(
        v2["model_review_required"].sum()
    )

    missed_review_count = len(missed)

    baseline_total_cost = float(
        v2["total_simulated_cost"].sum()
    )

    baseline_missed_count = int(
        v2["missed_maintenance"].sum()
    )

    print("\nV2 REVIEW ANALYSIS")
    print("-" * 55)

    print(f"Total engines: {len(v2)}")
    print(f"Engines reviewed: {review_count}")
    print(f"Total missed maintenance cases: {baseline_missed_count}")
    print(
        "Missed cases flagged for review: "
        f"{missed_review_count}"
    )

    if not missed.empty:
        print("\nMissed cases eligible for review:")
        print(
            missed[
                [
                    "engine_id",
                    "predicted_rul",
                    "actual_RUL_evaluation_only",
                    "model_review_required",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------
    # Calculate expected costs
    # --------------------------------------------------

    results = []

    for effectiveness in EFFECTIVENESS_LEVELS:

        probability = effectiveness / 100.0

        # Expected number of missed cases recovered.
        expected_recovered = (
            missed_review_count * probability
        )

        expected_remaining_missed = (
            baseline_missed_count
            - expected_recovered
        )

        # All review-required engines incur inspection cost.
        total_inspection_cost = (
            review_count * INSPECTION_COST
        )

        # Expected maintenance cost for recovered cases.
        expected_followup_maintenance_cost = (
            expected_recovered
            * REVIEW_TRIGGERED_MAINTENANCE_COST
        )

        # Expected remaining missed-maintenance penalties.
        expected_missed_penalty = (
            expected_remaining_missed
            * MISSED_MAINTENANCE_PENALTY
        )

        # Other V2 costs remain unchanged.
        original_v2_cost = baseline_total_cost

        # Remove the original missed penalties and inspection
        # costs before adding scenario-specific expected costs.
        adjusted_cost = (
            original_v2_cost
            - baseline_missed_count
            * MISSED_MAINTENANCE_PENALTY
            - total_inspection_cost
            + total_inspection_cost
            + expected_missed_penalty
            + expected_followup_maintenance_cost
        )

        # Net benefit compared with V2 without review benefits.
        net_benefit = (
            baseline_total_cost - adjusted_cost
        )

        results.append({
            "review_effectiveness_percent": effectiveness,
            "reviewed_engines": review_count,
            "eligible_missed_cases": missed_review_count,
            "expected_recovered_cases": round(
                expected_recovered, 2
            ),
            "expected_remaining_missed_cases": round(
                expected_remaining_missed, 2
            ),
            "inspection_cost_inr": round(
                total_inspection_cost, 2
            ),
            "expected_followup_maintenance_cost_inr": round(
                expected_followup_maintenance_cost, 2
            ),
            "expected_missed_penalty_inr": round(
                expected_missed_penalty, 2
            ),
            "expected_total_cost_inr": round(
                adjusted_cost, 2
            ),
            "net_benefit_vs_v2_inr": round(
                net_benefit, 2
            ),
        })

    return pd.DataFrame(results), missed


# --------------------------------------------------
# SAVE REPORTS
# --------------------------------------------------

def save_results(results, missed):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        OUTPUT_DIR
        / "review_effectiveness_results.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "review_effectiveness_summary.json"
    )

    missed_path = (
        OUTPUT_DIR
        / "missed_cases_flagged_for_review.csv"
    )

    results.to_csv(
        csv_path,
        index=False
    )

    missed.to_csv(
        missed_path,
        index=False
    )

    summary = {
        "experiment": "Hypothetical review effectiveness simulation",
        "baseline_policy": "V2",
        "review_effectiveness_levels_percent": EFFECTIVENESS_LEVELS,
        "assumptions": {
            "missed_maintenance_penalty_inr":
                MISSED_MAINTENANCE_PENALTY,
            "inspection_cost_inr":
                INSPECTION_COST,
            "review_triggered_maintenance_cost_inr":
                REVIEW_TRIGGERED_MAINTENANCE_COST,
        },
        "limitations": [
            "Review effectiveness levels are hypothetical assumptions.",
            "No real inspection effectiveness was measured.",
            "Actual RUL is used only for offline evaluation.",
            "The experiment does not prove real-world maintenance savings.",
            "Only two missed cases were flagged for review in this dataset.",
            "Expected recovered cases are probabilistic averages, not observed engine outcomes.",
        ],
        "results": results.to_dict(orient="records"),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nREPORTS SAVED")
    print("-" * 55)
    print(csv_path)
    print(json_path)
    print(missed_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("PROGNOSAI — REVIEW EFFECTIVENESS SIMULATION")
    print("=" * 65)

    df = load_data()

    results, missed = simulate_review_effectiveness(df)

    print("\nEXPECTED COST SCENARIOS")
    print("-" * 65)

    display_columns = [
        "review_effectiveness_percent",
        "expected_recovered_cases",
        "expected_remaining_missed_cases",
        "expected_total_cost_inr",
        "net_benefit_vs_v2_inr",
    ]

    print(
        results[display_columns].to_string(
            index=False
        )
    )

    save_results(results, missed)

    print("\nEXPERIMENT COMPLETED.")


if __name__ == "__main__":
    main()
