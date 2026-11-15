
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
DATA_DIR = Path("data/raw/CMAPSSData")
OUTPUT_DIR = Path("data/processed/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column definitions
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load training data
df = pd.read_csv(
    DATA_DIR / "train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

# Calculate sensor statistics
stats = df[sensor_columns].agg(["mean", "std", "min", "max"]).T
stats["unique_values"] = df[sensor_columns].nunique()

# Identify constant and variable sensors
constant_sensors = stats[
    stats["std"] < 1e-10
].index.tolist()

variable_sensors = [
    sensor for sensor in sensor_columns
    if sensor not in constant_sensors
]

print("\n========== SENSOR ANALYSIS ==========")

print("\nConstant sensors:")
print(constant_sensors)

print("\nVariable sensors:")
print(variable_sensors)

print("\nSensor statistics:")
print(stats.round(5))

# Save statistics
stats.to_csv(OUTPUT_DIR / "sensor_statistics.csv")

# Plot sensor standard deviations
plt.figure(figsize=(12, 7))

stats["std"].sort_values().plot(kind="barh")

plt.title("Sensor Variability - FD001")
plt.xlabel("Standard Deviation")
plt.ylabel("Sensor")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "sensor_variability.png", dpi=150)
plt.close()

# Plot selected variable sensors for Engine 1
engine_id = 1

engine_data = df[df["engine_id"] == engine_id]

# Select top 4 sensors by standard deviation
top_sensors = stats.loc[variable_sensors, "std"] \
    .sort_values(ascending=False).head(4).index.tolist()

plt.figure(figsize=(12, 7))

for sensor in top_sensors:
    plt.plot(
        engine_data["cycle"],
        engine_data[sensor],
        label=sensor
    )

plt.title(f"Sensor Trends - Engine {engine_id}")
plt.xlabel("Operating Cycle")
plt.ylabel("Sensor Reading")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "engine_1_sensor_trends.png", dpi=150)
plt.close()

print("\nTop 4 sensors by standard deviation:")
print(top_sensors)

print("\nAnalysis completed!")
print("Reports saved in:", OUTPUT_DIR.resolve())
