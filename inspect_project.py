
from pathlib import Path
import joblib
import pandas as pd

# Inspect dataset columns
for file in Path("data").glob("*.csv"):
    print(f"\n--- {file.name} ---")
    df = pd.read_csv(file, nrows=3)
    print("Columns:", df.columns.tolist())
    print(df.head(2).to_string(index=False))

# Inspect saved preprocessing information
bundle = joblib.load("models/multimodal_preprocessors.joblib")

print("\n--- SAVED PREPROCESSING INFORMATION ---")
for key, value in bundle.items():
    if isinstance(value, (list, tuple)):
        print(f"{key}: {list(value)}")
    else:
        print(f"{key}: {type(value).__name__}")
