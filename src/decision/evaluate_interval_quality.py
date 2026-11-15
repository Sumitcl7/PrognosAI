
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "artifacts" / "uncertainty" /
    "evaluation" / "conformal_test_final_predictions.csv"
)

OUTPUT = ROOT / "artifacts" / "decision"


def main():
    df = pd.read_csv(INPUT)

    df["interval_width"] = (
        df["engine_upper"] - df["engine_lower"]
    )

    df["normalized_width"] = (
        df["interval_width"]
        / df["actual_RUL_capped"].clip(lower=1)
    )

    df["actual_rul_band"] = pd.cut(
        df["actual_RUL"],
        bins=[-1, 15, 30, 60, float("inf")],
        labels=["CRITICAL", "HIGH", "MODERATE", "LOW"],
    )

    print("\nOVERALL FINAL-CYCLE INTERVAL QUALITY")
    print("=" * 60)

    print("Engines:", len(df))
    print(
        "Engine interval coverage:",
        round(df["engine_covered"].mean() * 100, 2), "%"
    )
    print(
        "Mean interval width:",
        round(df["interval_width"].mean(), 2)
    )
    print(
        "Median interval width:",
        round(df["interval_width"].median(), 2)
    )
    print(
        "Mean normalized width:",
        round(df["normalized_width"].mean(), 4)
    )

    print("\nQUALITY BY ACTUAL RUL CATEGORY")

    summary = (
        df.groupby("actual_rul_band", observed=False)
        .agg(
            engines=("engine_id", "count"),
            coverage=("engine_covered", "mean"),
            mean_width=("interval_width", "mean"),
            median_width=("interval_width", "median"),
            mean_normalized_width=("normalized_width", "mean"),
        )
    )

    summary["coverage"] *= 100

    print(summary.round(3).to_string())

    summary.to_csv(
        OUTPUT / "interval_quality_by_rul.csv"
    )

    print("\nSaved interval_quality_by_rul.csv")


if __name__ == "__main__":
    main()
