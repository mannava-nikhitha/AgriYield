
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" / "residual_mlp_crop_yield_model.keras"
)

PREPROCESSOR_PATH = (
    BASE_DIR / "models" / "crop_yield_preprocessor.joblib"
)

SOIL_DATA_PATH = BASE_DIR / "data" / "state_soil_data.csv"

WEATHER_DATA_PATH = (
    BASE_DIR / "data" / "state_weather_data_1997_2020.csv"
)


# ==================================================
# 2. V1 MODEL FEATURE COLUMNS
# ==================================================

FEATURE_COLUMNS = [
    "crop",
    "season",
    "state",
    "year",
    "area",
    "fertilizer",
    "pesticide",
    "avg_temp_c",
    "total_rainfall_mm",
    "avg_humidity_percent",
    "N",
    "P",
    "K",
    "pH",
]


# ==================================================
# 3. CHECK REQUIRED FILES
# ==================================================

REQUIRED_FILES = [
    MODEL_PATH,
    PREPROCESSOR_PATH,
    SOIL_DATA_PATH,
    WEATHER_DATA_PATH,
]

for file_path in REQUIRED_FILES:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ==================================================
# 4. LOAD V1 RESIDUAL MLP MODEL
# ==================================================

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    preprocessor = joblib.load(PREPROCESSOR_PATH)

except Exception as exc:
    raise RuntimeError(
        f"Could not load V1 model or preprocessor: {exc}"
    ) from exc


# ==================================================
# 5. LOAD WEATHER AND SOIL DATA
# ==================================================

try:
    soil_data = pd.read_csv(SOIL_DATA_PATH)

    weather_data = pd.read_csv(WEATHER_DATA_PATH)

    # Normalize state names for matching.
    soil_data["state_key"] = (
        soil_data["state"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    weather_data["state_key"] = (
        weather_data["state"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    weather_data["year"] = pd.to_numeric(
        weather_data["year"],
        errors="coerce"
    )

except Exception as exc:
    raise RuntimeError(
        f"Could not load weather or soil data: {exc}"
    ) from exc


# ==================================================
# 6. CREATE FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Crop Yield Prediction API",
    description=(
        "Crop yield prediction using the V1 Residual MLP "
        "with crop, climate, and soil features."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# 7. DEFINE REQUEST INPUTS
# ==================================================

class CropRequest(BaseModel):
    crop: str = Field(min_length=1)
    year: int = Field(ge=1997, le=2020)
    season: str = Field(min_length=1)
    state: str = Field(min_length=1)

    area: float = Field(gt=0)
    fertilizer: float = Field(ge=0)
    pesticide: float = Field(ge=0)


# ==================================================
# 8. HOME ENDPOINT
# ==================================================

@app.get("/")
def home():
    return {
        "message": "Crop Yield Prediction API is running",
        "model": "V1 Residual MLP",
        "version": "2.0.0",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


# ==================================================
# 9. HEALTH CHECK
# ==================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_name": "V1 Residual MLP",
    }


# ==================================================
# 10. PREDICT CROP YIELD
# ==================================================

@app.post("/predict")
def predict_crop_yield(request: CropRequest):

    try:

        # ------------------------------------------
        # A. NORMALIZE STATE NAME
        # ------------------------------------------

        state_key = request.state.strip().casefold()

        # ------------------------------------------
        # B. FIND SOIL INFORMATION
        # ------------------------------------------

        soil_match = soil_data[
            soil_data["state_key"] == state_key
        ]

        if soil_match.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Soil data not found for '{request.state}'."
                )
            )

        # ------------------------------------------
        # C. FIND WEATHER INFORMATION
        # ------------------------------------------

        weather_match = weather_data[
            (weather_data["state_key"] == state_key)
            & (weather_data["year"] == request.year)
        ]

        if weather_match.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Weather data not found for "
                    f"{request.state} in {request.year}."
                )
            )

        soil_row = soil_match.iloc[0]
        weather_row = weather_match.iloc[0]

        # ------------------------------------------
        # D. BUILD MODEL INPUT
        # ------------------------------------------

        input_data = pd.DataFrame([{
            "crop": request.crop.strip(),
            "season": request.season.strip(),
            "state": request.state.strip(),
            "year": request.year,
            "area": request.area,
            "fertilizer": request.fertilizer,
            "pesticide": request.pesticide,
            "avg_temp_c": weather_row["avg_temp_c"],
            "total_rainfall_mm": weather_row[
                "total_rainfall_mm"
            ],
            "avg_humidity_percent": weather_row[
                "avg_humidity_percent"
            ],
            "N": soil_row["N"],
            "P": soil_row["P"],
            "K": soil_row["K"],
            "pH": soil_row["pH"],
        }])

        # ------------------------------------------
        # E. VALIDATE INPUT VALUES
        # ------------------------------------------

        numeric_columns = [
            "year",
            "area",
            "fertilizer",
            "pesticide",
            "avg_temp_c",
            "total_rainfall_mm",
            "avg_humidity_percent",
            "N",
            "P",
            "K",
            "pH",
        ]

        input_data[numeric_columns] = input_data[
            numeric_columns
        ].apply(pd.to_numeric, errors="coerce")

        if input_data[numeric_columns].isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Weather or soil data contains missing "
                    "or invalid numerical values."
                )
            )

        if not np.isfinite(
            input_data[numeric_columns].to_numpy(dtype=float)
        ).all():
            raise HTTPException(
                status_code=400,
                detail="Input contains non-finite numerical values."
            )

        # ------------------------------------------
        # F. PRESERVE TRAINING FEATURE ORDER
        # ------------------------------------------

        input_data = input_data[FEATURE_COLUMNS]

        # ------------------------------------------
        # G. APPLY V1 PREPROCESSOR
        # ------------------------------------------

        X = preprocessor.transform(input_data)

        # ------------------------------------------
        # H. PREDICT LOG1P(YIELD)
        # ------------------------------------------

        predicted_log = model.predict(
            X,
            verbose=0
        ).reshape(-1)[0]

        # ------------------------------------------
        # I. CONVERT TO ORIGINAL YIELD SCALE
        # ------------------------------------------

        predicted_yield = float(
            np.expm1(predicted_log)
        )

        if not np.isfinite(predicted_yield):
            raise ValueError(
                "The model returned a non-finite prediction."
            )

        predicted_yield = max(predicted_yield, 0.0)

        # ------------------------------------------
        # J. RETURN PREDICTION
        # ------------------------------------------

        return {
            "crop": request.crop.strip(),
            "state": request.state.strip(),
            "year": request.year,
            "season": request.season.strip(),
            "predicted_yield": round(predicted_yield, 4),
            "note": (
                "Prediction from the V1 Residual MLP model. "
                "Accuracy is not guaranteed."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc
