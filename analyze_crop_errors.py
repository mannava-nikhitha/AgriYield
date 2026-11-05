
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("outputs/test_set_predictions.csv")

# Calculate absolute and percentage errors
df["Absolute Error"] = (
    df["yield"] - df["Predicted Yield"]
).abs()

df["Absolute Percentage Error"] = (
    df["Absolute Error"]
    / df["yield"].abs().replace(0, np.nan)
) * 100

# Summarize errors for each crop
summary = df.groupby("crop").apply(
    lambda group: pd.Series({
        "Samples": len(group),
        "Actual Mean": group["yield"].mean(),
        "Predicted Mean": group["Predicted Yield"].mean(),
        "MAE": mean_absolute_error(
            group["yield"], group["Predicted Yield"]
        ),
        "R2": (
            r2_score(group["yield"], group["Predicted Yield"])
            if len(group) >= 2
            and group["yield"].nunique() > 1
            else np.nan
        )
    }),
    include_groups=False
).sort_values("MAE", ascending=False)

print("\nCROPS WITH THE LARGEST MAE")
print(summary.head(15).round(4).to_string())

print("\nCROPS WITH THE SMALLEST MAE")
print(summary.tail(10).round(4).to_string())

summary.to_csv("outputs/crop_error_summary.csv")
print("\nSaved: outputs/crop_error_summary.csv")
