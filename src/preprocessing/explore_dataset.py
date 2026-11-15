
import pandas as pd
from pathlib import Path

# Dataset location
DATA_DIR = Path("data/raw/CMAPSSData")

# C-MAPSS column names
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load FD001 training data
train_path = DATA_DIR / "train_FD001.txt"

train_df = pd.read_csv(
    train_path,
    sep=r"\s+",
    header=None,
    names=columns
)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", train_df.shape)
print("Number of engines:", train_df["engine_id"].nunique())
print("Total records:", len(train_df))

print("\n========== FIRST 5 ROWS ==========")
print(train_df.head())

print("\n========== DATA TYPES ==========")
print(train_df.dtypes)

print("\n========== MISSING VALUES ==========")
print(train_df.isnull().sum().sum())

print("\n========== ENGINE CYCLE INFORMATION ==========")
engine_cycles = train_df.groupby("engine_id")["cycle"].max()

print(engine_cycles.describe())

print("\n========== SENSOR STATISTICS ==========")
print(train_df.iloc[:, 5:].describe().round(3))

print("\n========== RUL LABEL FILE ==========")
rul_path = DATA_DIR / "RUL_FD001.txt"
rul_df = pd.read_csv(rul_path, header=None, names=["RUL"])

print(rul_df.head())
print("Number of test RUL labels:", len(rul_df))

print("\nDataset exploration completed!")
