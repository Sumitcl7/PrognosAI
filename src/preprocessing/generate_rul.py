
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path("data/raw/CMAPSSData")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FD001 column names
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

# Sort to ensure correct cycle order
df = df.sort_values(
    ["engine_id", "cycle"]
).reset_index(drop=True)

# Calculate maximum cycle for each engine
max_cycles = df.groupby("engine_id")["cycle"].transform("max")

# Original uncapped RUL
df["RUL"] = max_cycles - df["cycle"]

# Common capped RUL variant
df["RUL_capped"] = df["RUL"].clip(upper=125)

# Validation checks
assert df["RUL"].min() == 0
assert df.groupby("engine_id")["RUL"].last().eq(0).all()
assert df["RUL_capped"].max() <= 125
assert df["RUL"].ge(0).all()

# Save processed dataset
output_file = OUTPUT_DIR / "FD001_train_with_RUL.csv"
df.to_csv(output_file, index=False)

# Display results
print("\n========== RUL GENERATION ==========")
print("Dataset shape:", df.shape)
print("Number of engines:", df["engine_id"].nunique())

print("\nFirst 10 records:")
print(df[["engine_id", "cycle", "RUL", "RUL_capped"]].head(10))

print("\nRUL statistics:")
print(df["RUL"].describe())

print("\nCapped RUL statistics:")
print(df["RUL_capped"].describe())

# Plot RUL distribution
plt.figure(figsize=(10, 6))
plt.hist(df["RUL"], bins=40)
plt.title("RUL Distribution - FD001 Training Data")
plt.xlabel("Remaining Useful Life (Cycles)")
plt.ylabel("Number of Observations")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "FD001_RUL_distribution.png",
    dpi=150
)
plt.close()

print("\nRUL generation completed successfully!")
print("Saved:", output_file.resolve())
