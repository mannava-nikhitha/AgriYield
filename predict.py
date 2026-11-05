
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# 1. File paths
DATA_DIR = Path("data")
MODEL_PATH = Path("models/improved_multimodal_model.keras")
BUNDLE_PATH = Path("models/multimodal_preprocessors.joblib")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load model and fitted preprocessors
model = keras.models.load_model(MODEL_PATH, compile=False)
bundle = joblib.load(BUNDLE_PATH)

# 3. Load CSV datasets
crop_df = pd.read_csv(DATA_DIR / "crop_yield.csv")
weather_df = pd.read_csv(
    DATA_DIR / "state_weather_data_1997_2020.csv"
)
soil_df = pd.read_csv(DATA_DIR / "state_soil_data.csv")

# 4. Merge datasets
combined_df = pd.merge(
    crop_df,
    weather_df,
    on=["state", "year"],
    how="inner"
)

combined_df = pd.merge(
    combined_df,
    soil_df,
    on="state",
    how="left"
)

combined_df = combined_df.reset_index(drop=True)

if combined_df.empty:
    raise ValueError("No rows remain after merging the datasets.")

print("Rows after merging:", len(combined_df))

# 5. Select the saved input columns
X_crop = combined_df[bundle["crop_columns"]].copy()
X_climate = combined_df[bundle["climate_columns"]].copy()
X_soil = combined_df[bundle["soil_columns"]].copy()

# 6. Check for missing or invalid input values
for name, frame in [
    ("crop", X_crop),
    ("climate", X_climate),
    ("soil", X_soil)
]:
    if frame.isnull().any().any():
        missing_columns = frame.columns[
            frame.isnull().any()
        ].tolist()
        raise ValueError(
            f"{name} data contains missing values in "
            f"{missing_columns}. Check Colab preprocessing."
        )

    if name != "crop":
        numeric = frame.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(
                f"{name} data contains non-finite values."
            )

# 7. Apply the saved preprocessors
X_crop_processed = bundle["crop_preprocessor"].transform(X_crop)
X_climate_processed = bundle["climate_preprocessor"].transform(X_climate)
X_soil_processed = bundle["soil_preprocessor"].transform(X_soil)

print("Crop input shape:", X_crop_processed.shape)
print("Climate input shape:", X_climate_processed.shape)
print("Soil input shape:", X_soil_processed.shape)

# 8. Validate input dimensions
if X_crop_processed.shape[1] != 65:
    raise ValueError("Model expects 65 processed crop features.")

if X_climate_processed.shape[1] != 3:
    raise ValueError("Model expects 3 climate features.")

if X_soil_processed.shape[1] != 4:
    raise ValueError("Model expects 4 soil features.")

# 9. Predict using named model inputs
predicted_log = model.predict(
    {
        "crop_input": X_crop_processed,
        "climate_input": X_climate_processed,
        "soil_input": X_soil_processed
    },
    verbose=0
).reshape(-1)

# The Colab training code uses log1p(y)
predicted_yield = np.maximum(np.expm1(predicted_log), 0)

# 10. Save predictions
results = combined_df[
    ["crop", "year", "season", "state"]
].copy()

# Include actual values when the source dataset has them
if "yield" in combined_df.columns:
    results["Actual Yield"] = combined_df["yield"]

results["Predicted Yield"] = predicted_yield

output_path = OUTPUT_DIR / "improved_multimodal_predictions.csv"
results.to_csv(output_path, index=False)

print("\nPredictions generated successfully.")
print("Rows:", len(results))
print("Output:", output_path.resolve())
print("\nSample predictions:")
print(results.head(10).to_string(index=False))
