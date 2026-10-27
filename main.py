import os
import joblib
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = hf_hub_download(
    repo_id="namrathabuilds/house_price_prediction",
    filename="house_price_model.pkl",
    repo_type="model"
)

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts median house values using a trained ML model.",
    version="1.0.0"
)

class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, le=20)
    HouseAge: float = Field(gt=0, le=100)
    AveRooms: float = Field(gt=0, le=100)
    AveBedrms: float = Field(gt=0, le=25)
    Population: float = Field(gt=0, le=50000)
    AveOccup: float = Field(gt=0, le=100)
    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)

@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_price(house: HouseFeatures):
    features = pd.DataFrame(
        [house.model_dump()],
        columns=model.feature_names_in_
    )
    prediction = float(model.predict(features)[0])

    return {
        "predicted_price_100k_usd": round(prediction, 4),
        "predicted_price_usd": round(prediction * 100000, 2),
        "currency": "USD",
        "note": "Estimated median value from the California Housing dataset."
    }
from huggingface_hub import hf_hub_download
