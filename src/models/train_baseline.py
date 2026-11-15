
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

DATA_PATH = Path("data/processed/FD001_train_with_RUL.csv")
ARTIFACT_DIR = Path("artifacts/baseline")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\n========== BASELINE MODEL TRAINING ==========")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. Select features
# --------------------------------------------------

sensor_features = [
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21"
]

setting_features = [
    "setting_1", "setting_2", "setting_3"
]

feature_columns = setting_features + sensor_features

target_column = "RUL_capped"

X = df[feature_columns]
y = df[target_column]
groups = df["engine_id"]

print("Number of features:", len(feature_columns))
print("Target:", target_column)

# --------------------------------------------------
# 4. Split by engine ID (prevent data leakage)
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

train_engines = set(groups.iloc[train_idx])
val_engines = set(groups.iloc[val_idx])

assert train_engines.isdisjoint(val_engines)

print("\n========== DATA SPLIT ==========")
print("Training engines:", len(train_engines))
print("Validation engines:", len(val_engines))
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Engine overlap:", len(train_engines & val_engines))

# --------------------------------------------------
# 5. Train Random Forest
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------------------
# 6. Generate validation predictions
# --------------------------------------------------

predictions = model.predict(X_val)

# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------

mae = mean_absolute_error(y_val, predictions)

rmse = np.sqrt(
    mean_squared_error(y_val, predictions)
)

r2 = r2_score(y_val, predictions)

# Simple mean-prediction baseline
mean_prediction = np.full(
    shape=len(y_val),
    fill_value=y_train.mean()
)

mean_mae = mean_absolute_error(y_val, mean_prediction)

print("\n========== VALIDATION RESULTS ==========")
print(f"Random Forest MAE: {mae:.4f}")
print(f"Random Forest RMSE: {rmse:.4f}")
print(f"Random Forest R2: {r2:.4f}")
print(f"Mean baseline MAE: {mean_mae:.4f}")

# --------------------------------------------------
# 8. Save model
# --------------------------------------------------

model_path = ARTIFACT_DIR / "random_forest_fd001.joblib"

joblib.dump(model, model_path)

# Save validation predictions
results = pd.DataFrame({
    "actual_RUL": y_val.to_numpy(),
    "predicted_RUL": predictions,
    "engine_id": groups.iloc[val_idx].to_numpy()
})

results.to_csv(
    ARTIFACT_DIR / "validation_predictions.csv",
    index=False
)

# Save metrics and metadata
metrics = {
    "model": "RandomForestRegressor",
    "dataset": "NASA C-MAPSS FD001",
    "target": target_column,
    "random_state": RANDOM_STATE,
    "training_engines": len(train_engines),
    "validation_engines": len(val_engines),
    "training_records": len(X_train),
    "validation_records": len(X_val),
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "mean_baseline_MAE": float(mean_mae),
    "features": feature_columns
}

with open(ARTIFACT_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved:", model_path)
print("Metrics saved:", ARTIFACT_DIR / "metrics.json")
print("\nBaseline experiment completed successfully!")
