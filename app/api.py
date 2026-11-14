from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"

# Order must match the training columns of the California housing dataset
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

app = FastAPI(title="House Price Prediction API")
model = joblib.load(MODEL_PATH)


@app.get("/")
def health():
    return {"status": "ok", "expected_features": FEATURES}


@app.post("/predict")
def predict(features: dict):
    # Order values by feature name so the JSON key order doesn't matter
    x = np.array([[features[name] for name in FEATURES]], dtype=float)
    prediction = model.predict(x)[0]
    return {"predicted_price": float(prediction * 100000)}
