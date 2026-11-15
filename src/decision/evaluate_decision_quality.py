
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    ROOT / "artifacts" / "decision" /
    "policy_sensitivity_results.csv"
)

PREDICTIONS_FILE = (
    ROOT / "artifacts" / "uncertainty" /
    "evaluation" / "conformal_test_final_predictions.csv"
)

OUTPUT_DIR = ROOT / "artifacts" / "decision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = {
    "MONITOR": 0,
    "PLAN": 1,
    "SCHEDULE_SOON": 2,
    "URGENT": 3,
}


def reference_category(actual_rul):
    if actual_rul <= 15:
        return "URGENT"
    elif actual_rul <= 30:
        return "SCHEDULE_SOON"
    elif actual_rul <= 60:
        return "PLAN"
    return "MONITOR"


def main():
    decisions = pd.read_csv(RESULTS_FILE)

    actual = pd.read_csv(PREDICTIONS_FILE)[
        ["engine_id", "actual_RUL"]
    ]

    if actual["engine_id"].duplicated().any():
        raise ValueError("Duplicate engine IDs in evaluation labels.")

    data = decisions.merge(
        actual,
        on="engine_id",
        how="left",
        validate="many_to_one",
    )

    if data["actual_RUL"].isna().any():
        raise ValueError("Some decisions have no matching actual RUL.")

    data["reference_priority"] = data["actual_RUL"].apply(
        reference_category
    )

    data["priority_difference"] = (
        data["priority"].map(LEVELS)
        - data["reference_priority"].map(LEVELS)
    )

    data["agreement"] = data["priority_difference"] == 0
    data["under_prioritized"] = data["priority_difference"] < 0
    data["over_prioritized"] = data["priority_difference"] > 0

    data.to_csv(
        OUTPUT_DIR / "decision_quality_by_engine.csv",
        index=False,
    )

    summary = (
        data.groupby("policy")
        .agg(
            engines=("engine_id", "count"),
            exact_agreement=("agreement", "mean"),
            under_prioritized=("under_prioritized", "mean"),
            over_prioritized=("over_prioritized", "mean"),
            mean_absolute_priority_difference=(
                "priority_difference",
                lambda x: x.abs().mean(),
            ),
        )
        .reset_index()
    )

    percentage_columns = [
        "exact_agreement",
        "under_prioritized",
        "over_prioritized",
    ]

    for column in percentage_columns:
        summary[column] *= 100

    summary.to_csv(
        OUTPUT_DIR / "decision_quality_summary.csv",
        index=False,
    )

    for policy in data["policy"].unique():
        matrix = pd.crosstab(
            data.loc[data["policy"] == policy, "reference_priority"],
            data.loc[data["policy"] == policy, "priority"],
        )

        matrix = matrix.reindex(
            index=list(LEVELS.keys()),
            columns=list(LEVELS.keys()),
            fill_value=0,
        )

        matrix.to_csv(
            OUTPUT_DIR / f"confusion_matrix_{policy}.csv"
        )

        print(f"\nCONFUSION MATRIX: {policy.upper()}")
        print(matrix.to_string())

    print("\nDECISION QUALITY SUMMARY")
    print("=" * 60)
    print(summary.round(2).to_string(index=False))

    print("\nSaved reports in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
