
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot

import matplotlib.pyplot as plt
import pandas as pd
import joblib


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "artifacts" / "baseline" / "random_forest_fd001.joblib"
PREDICTIONS_PATH = ROOT_DIR / "artifacts" / "baseline" / "validation_predictions.csv"
METRICS_PATH = ROOT_DIR / "artifacts" / "baseline" / "metrics.json"

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

print("Loading baseline model and validation results...")

model = joblib.load(MODEL_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

print("\nModel loaded successfully.")
print(f"Validation records: {len(predictions)}")


# --------------------------------------------------
# RESIDUAL ANALYSIS
# --------------------------------------------------

# Residual = Actual RUL - Predicted RUL
predictions["residual"] = (
    predictions["actual_RUL"] - predictions["predicted_RUL"]
)

predictions["absolute_error"] = predictions["residual"].abs()

print("\n========== RESIDUAL STATISTICS ==========")

print(
    predictions["residual"].describe().round(2)
)

print("\n========== PREDICTION SUMMARY ==========")

print(
    predictions[
        ["actual_RUL", "predicted_RUL"]
    ].describe().round(2)
)


# --------------------------------------------------
# PER-ENGINE PERFORMANCE
# --------------------------------------------------

engine_metrics = (
    predictions.groupby("engine_id")
    .agg(
        MAE=("absolute_error", "mean"),
        RMSE=("residual", lambda x: (x.pow(2).mean()) ** 0.5),
        observations=("residual", "count"),
    )
    .reset_index()
    .sort_values("MAE", ascending=False)
)

engine_metrics.to_csv(
    OUTPUT_DIR / "baseline_per_engine_metrics.csv",
    index=False
)

print("\n========== PER-ENGINE MAE ==========")
print(engine_metrics.round(2).to_string(index=False))


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

feature_names = metrics.get("features")

if feature_names is None:
    raise ValueError(
        "The metrics.json file does not contain a 'features' field. "
        "Check the metadata saved by train_baseline.py."
    )

if len(feature_names) != len(model.feature_importances_):
    raise ValueError(
        "Feature count mismatch between metrics.json and trained model."
    )

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

feature_importance.to_csv(
    OUTPUT_DIR / "baseline_feature_importance.csv",
    index=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance.round(4).to_string(index=False))


# --------------------------------------------------
# PLOT 1: ACTUAL VS PREDICTED RUL
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    predictions["actual_RUL"],
    predictions["predicted_RUL"],
    alpha=0.35
)

min_value = min(
    predictions["actual_RUL"].min(),
    predictions["predicted_RUL"].min()
)

max_value = max(
    predictions["actual_RUL"].max(),
    predictions["predicted_RUL"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL - FD001 Baseline")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# PLOT 2: ERROR DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    predictions["residual"],
    bins=40,
    edgecolor="black"
)

plt.axvline(0, linestyle="--")

plt.xlabel("Residual (Actual RUL - Predicted RUL)")
plt.ylabel("Frequency")
plt.title("Baseline Prediction Error Distribution")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_error_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# PLOT 3: FEATURE IMPORTANCE
# --------------------------------------------------

plot_data = feature_importance.sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(9, 7))

plt.barh(
    plot_data["feature"],
    plot_data["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance - FD001")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_feature_importance.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

plt.close("all")

print("\n========== ANALYSIS COMPLETED ==========")

print(f"Output directory: {OUTPUT_DIR}")

print("\nGenerated files:")
print("- baseline_per_engine_metrics.csv")
print("- baseline_feature_importance.csv")
print("- baseline_actual_vs_predicted.png")
print("- baseline_error_distribution.png")
print("- baseline_feature_importance.png")

print("\nNo interactive plots were opened.")
print("Matplotlib backend:", matplotlib.get_backend())
