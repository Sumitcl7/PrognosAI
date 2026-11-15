
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "artifacts" / "decision" /
    "v1_v2_comparison.csv"
)

OUTPUT = ROOT / "artifacts" / "decision"


def main():
    df = pd.read_csv(INPUT)

    df["relative_interval_width"] = (
        df["interval_width"]
        / df["predicted_rul"].clip(lower=1.0)
    )

    df["relative_width_percent"] = (
        df["relative_interval_width"] * 100
    )

    print("\nRELATIVE UNCERTAINTY STATISTICS")
    print("=" * 55)

    print(
        df["relative_width_percent"]
        .describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
        )
        .round(2)
        .to_string()
    )

    df["relative_confidence_band"] = pd.cut(
        df["relative_width_percent"],
        bins=[-1, 50, 100, float("inf")],
        labels=["HIGH", "MEDIUM", "LOW"],
    )

    print("\nRELATIVE CONFIDENCE DISTRIBUTION")
    print(
        df["relative_confidence_band"]
        .value_counts()
        .to_string()
    )

    print("\nABSOLUTE VS RELATIVE CONFIDENCE")

    df["absolute_confidence_band"] = pd.cut(
        df["interval_width"],
        bins=[-1, 40, 80, float("inf")],
        labels=["HIGH", "MEDIUM", "LOW"],
    )

    print(
        pd.crosstab(
            df["absolute_confidence_band"],
            df["relative_confidence_band"],
        ).to_string()
    )

    print("\nRELATIVE WIDTH VS PREDICTED RUL CORRELATION")
    print(
        round(
            df[
                ["predicted_rul", "relative_width_percent"]
            ].corr().iloc[0, 1],
            4,
        )
    )

    df.to_csv(
        OUTPUT / "relative_uncertainty_analysis.csv",
        index=False,
    )

    print("\nSaved relative_uncertainty_analysis.csv")


if __name__ == "__main__":
    main()
