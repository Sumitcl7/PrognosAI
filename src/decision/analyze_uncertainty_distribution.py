
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "artifacts" / "decision" /
    "v1_v2_comparison.csv"
)

OUTPUT = ROOT / "artifacts" / "decision"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(INPUT)

    print("\nINTERVAL WIDTH STATISTICS")
    print("=" * 55)

    print(
        df["interval_width"].describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
        ).round(2).to_string()
    )

    bins = [-1, 40, 80, float("inf")]
    labels = ["HIGH", "MEDIUM", "LOW"]

    df["confidence_band"] = pd.cut(
        df["interval_width"],
        bins=bins,
        labels=labels,
    )

    print("\nCONFIDENCE DISTRIBUTION")
    print(df["confidence_band"].value_counts().to_string())

    df["predicted_rul_band"] = pd.cut(
        df["predicted_rul"],
        bins=[-1, 15, 30, 60, float("inf")],
        labels=["CRITICAL", "HIGH", "MODERATE", "LOW"],
    )

    print("\nRISK VS CONFIDENCE")
    cross = pd.crosstab(
        df["predicted_rul_band"],
        df["confidence_band"],
        margins=True,
    )
    print(cross.to_string())

    if "actual_RUL" in df.columns:
        df["actual_rul_band"] = pd.cut(
            df["actual_RUL"],
            bins=[-1, 15, 30, 60, float("inf")],
            labels=["CRITICAL", "HIGH", "MODERATE", "LOW"],
        )

        print("\nACTUAL RUL VS CONFIDENCE (OFFLINE ONLY)")
        print(
            pd.crosstab(
                df["actual_rul_band"],
                df["confidence_band"],
                margins=True,
            ).to_string()
        )

    correlation = df[
        ["predicted_rul", "interval_width"]
    ].corr().iloc[0, 1]

    print("\nPrediction / interval-width correlation:")
    print(round(correlation, 4))

    df.to_csv(
        OUTPUT / "uncertainty_distribution_by_engine.csv",
        index=False,
    )

    print("\nSaved uncertainty_distribution_by_engine.csv")


if __name__ == "__main__":
    main()
