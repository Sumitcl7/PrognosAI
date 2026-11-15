
import json
from pathlib import Path

import pandas as pd

from src.decision.maintenance_engine import (
    MaintenanceConfig,
    evaluate_maintenance,
)

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "artifacts" / "uncertainty" /
    "evaluation" / "conformal_test_final_predictions.csv"
)

OUTPUT = ROOT / "artifacts" / "decision"
OUTPUT.mkdir(parents=True, exist_ok=True)


POLICIES = {
    "conservative": MaintenanceConfig(
        critical_rul=15,
        urgent_rul=30,
        planning_rul=60,
        wide_interval=40,
    ),
    "balanced": MaintenanceConfig(
        critical_rul=10,
        urgent_rul=20,
        planning_rul=40,
        wide_interval=60,
    ),
    "less_aggressive": MaintenanceConfig(
        critical_rul=5,
        urgent_rul=15,
        planning_rul=30,
        wide_interval=80,
    ),
}


def main():
    df = pd.read_csv(INPUT)

    rows = []

    for policy_name, config in POLICIES.items():

        for _, row in df.iterrows():

            decision = evaluate_maintenance(
                predicted_rul=float(row["engine_prediction"]),
                lower_bound=float(row["engine_lower"]),
                upper_bound=float(row["engine_upper"]),
                drift_status="stable",
                config=config,
            )

            rows.append({
                "policy": policy_name,
                "engine_id": int(row["engine_id"]),
                "predicted_rul": float(row["engine_prediction"]),
                "lower_bound": float(row["engine_lower"]),
                "upper_bound": float(row["engine_upper"]),
                "interval_width": decision.interval_width,
                "priority": decision.priority,
            })

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT / "policy_sensitivity_results.csv",
        index=False,
    )

    summary = (
        results.groupby(["policy", "priority"])
        .size()
        .unstack(fill_value=0)
    )

    summary = summary.reindex(
        columns=["MONITOR", "PLAN", "SCHEDULE_SOON", "URGENT"],
        fill_value=0,
    )

    summary.to_csv(
        OUTPUT / "policy_sensitivity_summary.csv"
    )

    print("\nPOLICY SENSITIVITY RESULTS")
    print("=" * 60)
    print(summary.to_string())

    print("\nSaved:")
    print(OUTPUT / "policy_sensitivity_results.csv")
    print(OUTPUT / "policy_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
