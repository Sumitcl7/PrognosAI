
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/processed/FD001_train_with_RUL.csv")
OUTPUT_DIR = Path("data/processed/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

# Calculate sensor-RUL correlation
correlations = df[sensor_columns + ["RUL_capped"]].corr()["RUL_capped"]
correlations = correlations.drop("RUL_capped")

correlations = correlations.sort_values(
    key=abs,
    ascending=False
)

print("\n========== SENSOR-RUL CORRELATION ==========")
print(correlations.round(4))

correlations.to_csv(
    OUTPUT_DIR / "sensor_rul_correlations.csv"
)

# Plot correlations
plt.figure(figsize=(11, 7))

correlations.sort_values().plot(kind="barh")

plt.title("Sensor Correlation with Capped RUL - FD001")
plt.xlabel("Pearson Correlation")
plt.ylabel("Sensor")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "sensor_rul_correlation.png",
    dpi=150
)
plt.close()

# Visualize top 4 sensors against RUL
top_sensors = correlations.abs().nlargest(4).index.tolist()

sample = df.sample(
    n=min(3000, len(df)),
    random_state=42
)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, sensor in zip(axes.flatten(), top_sensors):
    ax.scatter(
        sample["RUL_capped"],
        sample[sensor],
        alpha=0.25,
        s=8
    )
    ax.set_title(f"{sensor} vs RUL")
    ax.set_xlabel("Capped RUL")
    ax.set_ylabel("Sensor Reading")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_sensor_rul_relationships.png",
    dpi=150
)
plt.close()

print("\nTop 4 sensors:", top_sensors)
print("\nAnalysis completed!")
