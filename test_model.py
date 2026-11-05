
from pathlib import Path
from tensorflow import keras

model_path = Path("models/improved_multimodal_model.keras")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = keras.models.load_model(model_path, compile=False)

print("Model loaded successfully!")
print("Input shapes:", model.input_shape)
print("Output shape:", model.output_shape)
