
import json
from pathlib import Path

import pandas as pd

from src.decision.maintenance_engine import evaluate_maintenance


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "artifacts"
    / "uncertainty"
    / "evaluation"
    / "conformal_test_final_predictions.csv"
)

OUTPUT_DIR = ROOT / "artifacts" / "decision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_assessment():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "engine_id",
        "engine_prediction",
        "engine_lower",
        "engine_upper",
        "engine_covered",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["engine_id"].duplicated().any():
        raise ValueError("Duplicate engine IDs found in final predictions.")

    results = []

    for _, row in df.iterrows():

        decision = evaluate_maintenance(
            predicted_rul=float(row["engine_prediction"]),
            lower_bound=float(row["engine_lower"]),
            upper_bound=float(row["engine_upper"]),
            drift_status="stable",
        )

        result = decision.to_dict()

        result["engine_id"] = int(row["engine_id"])

        # Actual RUL is included only for offline evaluation.
        # It is never passed into the decision function.
        if "actual_RUL" in df.columns:
            result["actual_RUL_evaluation_only"] = float(
                row["actual_RUL"]
            )

        result["interval_covered_evaluation_only"] = bool(
            row["engine_covered"]
        )

        result["drift_evaluation_status"] = "not_evaluated"

        results.append(result)

    results_df = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "maintenance_assessment.csv"
    json_path = OUTPUT_DIR / "maintenance_assessment.json"
    metrics_path = OUTPUT_DIR / "maintenance_summary.json"

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    summary = {
        "dataset": "NASA C-MAPSS FD001 official test",
        "total_engines": int(len(results_df)),
        "priority_distribution": {
            str(key): int(value)
            for key, value in (
                results_df["priority"].value_counts().items()
            )
        },
        "mean_predicted_rul": float(
            results_df["predicted_rul"].mean()
        ),
        "mean_interval_width": float(
            results_df["interval_width"].mean()
        ),
        "engine_interval_coverage": float(
            results_df["interval_covered_evaluation_only"].mean()
        ),
        "drift_status": "not_evaluated",
        "decision_policy": {
            "critical_rul": 15,
            "urgent_rul": 30,
            "planning_rul": 60,
            "wide_interval": 40,
        },
        "warning": (
            "Research decision policy only. "
            "Thresholds are configurable assumptions, "
            "not certified industrial maintenance limits. "
            "Drift was not evaluated for these test engines."
        ),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nPROGNOSAI MAINTENANCE ASSESSMENT")
    print("=" * 50)
    print(f"Total engines: {len(results_df)}")

    print("\nPriority distribution:")
    print(results_df["priority"].value_counts().to_string())

    print("\nAverage predicted RUL:")
    print(round(results_df["predicted_rul"].mean(), 2))

    print("\nAverage uncertainty interval width:")
    print(round(results_df["interval_width"].mean(), 2))

    print("\nEngine interval coverage:")
    print(
        f"{summary['engine_interval_coverage'] * 100:.2f}%"
    )

    print("\nDrift evaluation: NOT EVALUATED")

    print("\nSaved reports:")
    print(csv_path)
    print(json_path)
    print(metrics_path)


if __name__ == "__main__":
    run_assessment()
