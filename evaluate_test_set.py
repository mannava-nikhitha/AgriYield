
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load the saved model and fitted preprocessors
model = keras.models.load_model(
    "models/improved_multimodal_model.keras",
    compile=False
)
bundle = joblib.load("models/multimodal_preprocessors.joblib")

# 2. Load the datasets
crop_df = pd.read_csv("data/crop_yield.csv")
weather_df = pd.read_csv("data/state_weather_data_1997_2020.csv")
soil_df = pd.read_csv("data/state_soil_data.csv")

# 3. Recreate the data merge
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

print("Combined dataset shape:", combined_df.shape)

# 4. Recreate the original 80/20 split
indices = combined_df.index.to_numpy()

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42
)

test_df = combined_df.iloc[test_idx].copy()

# 5. Select features in the saved order
X_crop = test_df[bundle["crop_columns"]].copy()
X_climate = test_df[bundle["climate_columns"]].copy()
X_soil = test_df[bundle["soil_columns"]].copy()

# Do not silently invent replacement values.
# Missing values should be handled the same way as in Colab.
for name, frame in [
    ("crop", X_crop),
    ("climate", X_climate),
    ("soil", X_soil)
]:
    if frame.isnull().any().any():
        missing = frame.columns[frame.isnull().any()].tolist()
        raise ValueError(
            f"Missing values in {name} features: {missing}. "
            "Check the original Colab preprocessing."
        )

# 6. Apply the saved fitted preprocessors
X_crop_processed = bundle["crop_preprocessor"].transform(X_crop)
X_climate_processed = bundle["climate_preprocessor"].transform(X_climate)
X_soil_processed = bundle["soil_preprocessor"].transform(X_soil)

print("Processed input shapes:")
print("Crop:", X_crop_processed.shape)
print("Climate:", X_climate_processed.shape)
print("Soil:", X_soil_processed.shape)

# 7. Pass inputs by their model input names
predicted_log = model.predict(
    {
        "crop_input": X_crop_processed,
        "climate_input": X_climate_processed,
        "soil_input": X_soil_processed
    },
    verbose=0
).reshape(-1)

# 8. Convert log predictions back to the yield scale
predicted_yield = np.maximum(np.expm1(predicted_log), 0)
actual_yield = test_df["yield"].to_numpy(dtype=float)

if not np.isfinite(actual_yield).all():
    raise ValueError("Actual yield contains non-finite values.")

if not np.isfinite(predicted_yield).all():
    raise ValueError("Predictions contain non-finite values.")

# 9. Evaluate on the original yield scale
mae = mean_absolute_error(actual_yield, predicted_yield)
rmse = np.sqrt(mean_squared_error(actual_yield, predicted_yield))
r2 = r2_score(actual_yield, predicted_yield)

print("\nTEST-SET EVALUATION")
print("-------------------")
print("Test rows:", len(actual_yield))
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Also evaluate on log1p scale to diagnose the model's training objective
actual_log = np.log1p(np.maximum(actual_yield, 0))

print("\nLOG-SCALE DIAGNOSTIC")
print("--------------------")
print(
    f"Log MAE:  "
    f"{mean_absolute_error(actual_log, predicted_log):.4f}"
)
print(
    f"Log RMSE: "
    f"{np.sqrt(mean_squared_error(actual_log, predicted_log)):.4f}"
)
print(f"Log R²:   {r2_score(actual_log, predicted_log):.4f}")

# 10. Save test predictions
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

results = test_df[
    ["crop", "year", "season", "state", "yield"]
].copy()

results["Predicted Yield"] = predicted_yield

output_path = output_dir / "test_set_predictions.csv"
results.to_csv(output_path, index=False)

print("\nSaved test predictions to:", output_path.resolve())
print("\nSample results:")
print(results.head(10).to_string(index=False))
