
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

file_path = Path("outputs/improved_multimodal_predictions.csv")
df = pd.read_csv(file_path)

actual = df["Actual Yield"].to_numpy()
predicted = df["Predicted Yield"].to_numpy()

valid = np.isfinite(actual) & np.isfinite(predicted)
actual = actual[valid]
predicted = predicted[valid]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("PREDICTION EVALUATION")
print("---------------------")
print("Valid rows:", len(actual))
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

print("\nFirst 10 predictions:")
print(
    df.loc[valid, ["crop", "state", "Actual Yield", "Predicted Yield"]]
    .head(10)
    .to_string(index=False)
)
