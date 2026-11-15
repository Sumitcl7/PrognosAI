
import json
from pathlib import Path

import pandas as pd

from src.decision.maintenance_engine import evaluate_maintenance
from src.decision.maintenance_engine_v2 import evaluate_maintenance_v2


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


def main():
    df = pd.read_csv(INPUT_FILE)

    required = {
        "engine_id",
        "engine_prediction",
        "engine_lower",
        "engine_upper",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["engine_id"].duplicated().any():
        raise ValueError("Duplicate engine IDs found.")

    results = []

    for _, row in df.iterrows():
        prediction = float(row["engine_prediction"])
        lower = float(row["engine_lower"])
        upper = float(row["engine_upper"])

        # V1: conservative lower-bound decision policy.
        v1 = evaluate_maintenance(
            predicted_rul=prediction,
            lower_bound=lower,
            upper_bound=upper,
            drift_status="stable",
        )

        # V2: separate predicted risk and model confidence.
        v2 = evaluate_maintenance_v2(
            predicted_rul=prediction,
            lower_bound=lower,
            upper_bound=upper,
            drift_status="not_evaluated",
        )

        results.append({
            "engine_id": int(row["engine_id"]),
            "predicted_rul": prediction,
            "lower_bound": lower,
            "upper_bound": upper,
            "interval_width": upper - lower,

            "v1_priority": v1.priority,
            "v1_action": v1.action,

            "v2_risk_level": v2.risk_level,
            "v2_confidence_level": v2.confidence_level,
            "v2_action": v2.maintenance_action,
            "v2_model_review_required": v2.model_review_required,

            "drift_status": "not_evaluated",
        })

    comparison = pd.DataFrame(results)

    comparison_path = OUTPUT_DIR / "v1_v2_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    v1_distribution = {
        str(k): int(v)
        for k, v in comparison["v1_priority"].value_counts().items()
    }

    v2_risk_distribution = {
        str(k): int(v)
        for k, v in comparison["v2_risk_level"].value_counts().items()
    }

    v2_confidence_distribution = {
        str(k): int(v)
        for k, v in comparison["v2_confidence_level"].value_counts().items()
    }

    v2_action_distribution = {
        str(k): int(v)
        for k, v in comparison["v2_action"].value_counts().items()
    }

    review_count = int(
        comparison["v2_model_review_required"].sum()
    )

    summary = {
        "dataset": "NASA C-MAPSS FD001 official test",
        "total_engines": int(len(comparison)),
        "v1_priority_distribution": v1_distribution,
        "v2_risk_distribution": v2_risk_distribution,
        "v2_confidence_distribution": v2_confidence_distribution,
        "v2_action_distribution": v2_action_distribution,
        "v2_model_review_required": review_count,
        "v2_model_review_percentage": (
            round(review_count / len(comparison) * 100, 2)
            if len(comparison) else 0
        ),
        "drift_status": "not_evaluated",
        "note": (
            "V1 and V2 are experimental decision policies. "
            "This comparison describes recommendation distributions; "
            "it does not establish that either policy is safer or better."
        ),
    }

    summary_path = OUTPUT_DIR / "v1_v2_comparison_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nPROGNOSAI: V1 vs V2 COMPARISON")
    print("=" * 55)
    print(f"Total engines: {len(comparison)}")

    print("\nV1 priority distribution:")
    print(pd.Series(v1_distribution).to_string())

    print("\nV2 risk distribution:")
    print(pd.Series(v2_risk_distribution).to_string())

    print("\nV2 confidence distribution:")
    print(pd.Series(v2_confidence_distribution).to_string())

    print("\nV2 maintenance action distribution:")
    print(pd.Series(v2_action_distribution).to_string())

    print(
        f"\nV2 model reviews required: "
        f"{review_count} ({summary['v2_model_review_percentage']}%)"
    )

    print("\nDrift: NOT EVALUATED")

    print("\nSaved:")
    print(comparison_path)
    print(summary_path)


if __name__ == "__main__":
    main()
