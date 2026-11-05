
from pathlib import Path
import joblib
from sklearn.utils.validation import check_is_fitted

bundle_path = Path("models/multimodal_preprocessors.joblib")
bundle = joblib.load(bundle_path)

for name in [
    "crop_preprocessor",
    "climate_preprocessor",
    "soil_preprocessor"
]:
    preprocessor = bundle[name]

    try:
        check_is_fitted(preprocessor)
        print(f"{name}: FITTED")
    except Exception as error:
        print(f"{name}: NOT FITTED")
        print("Reason:", error)
