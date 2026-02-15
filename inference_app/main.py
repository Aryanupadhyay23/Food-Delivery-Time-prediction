import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import mlflow

from inference_app.utility_files.inference_transformer import (
    FoodDeliveryFeatureEngine
)

# Configuration

REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
PRODUCTION_ALIAS = "production"

# Fixed values inside script
DAGSHUB_USERNAME = "Aryanupadhyay23"
MLFLOW_TRACKING_URI = (
    "https://dagshub.com/Aryanupadhyay23/Food-Delivery-Time-prediction.mlflow"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Food Delivery Time Prediction API")

model_pipeline = None
feature_engine = FoodDeliveryFeatureEngine()

# MLflow Setup

def configure_mlflow():

    dagshub_token = os.environ.get("DAGSHUB_TOKEN")

    if not dagshub_token:
        raise RuntimeError("DAGSHUB_TOKEN environment variable not set.")

    # Set MLflow auth variables internally
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info("MLflow configured successfully using DAGSHUB_TOKEN.")

# Load Production Model

def load_production_model():
    global model_pipeline

    try:
        logger.info("Loading production model from MLflow Registry...")

        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
        model_pipeline = mlflow.pyfunc.load_model(model_uri)

        logger.info("Production model loaded successfully.")

    except Exception as e:
        logger.exception("CRITICAL ERROR: Could not load production model.")
        raise e

# Startup Event

@app.on_event("startup")
def startup_event():
    configure_mlflow()
    load_production_model()

# Health Endpoint

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None
    }

# Model Info Endpoint

@app.get("/model-info")
def model_info():
    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": PRODUCTION_ALIAS,
        "model_loaded": model_pipeline is not None
    }

# Input Schema
class DeliveryInput(BaseModel):

    ID: str = Field(..., min_length=1)
    Delivery_person_ID: str = Field(..., min_length=1)

    Delivery_person_Age: float = Field(..., gt=18.0)
    Delivery_person_Ratings: float = Field(..., ge=1.0, le=5.0)

    Restaurant_latitude: float = Field(..., ge=6.0, le=38.0)
    Restaurant_longitude: float = Field(..., ge=68.0, le=98.0)
    Delivery_location_latitude: float = Field(..., ge=6.0, le=38.0)
    Delivery_location_longitude: float = Field(..., ge=68.0, le=98.0)

    Order_Date: str = Field(..., pattern=r"\d{2}-\d{2}-\d{4}")
    Time_Orderd: str = Field(..., min_length=1)
    Weather_conditions: str = Field(..., min_length=1)
    Road_traffic_density: str = Field(..., min_length=1)
    Vehicle_condition: int = Field(..., ge=0)
    Type_of_order: str = Field(..., min_length=1)
    Type_of_vehicle: str = Field(..., min_length=1)
    multiple_deliveries: float = Field(..., ge=0.0)
    Festival: str = Field(..., min_length=1)
    City: str = Field(..., min_length=1)

    @field_validator("Delivery_person_Age")
    @classmethod
    def validate_age_strict(cls, v: float) -> float:
        if v <= 18:
            raise ValueError("Rider must be older than 18.")
        return v

# Prediction Endpoint

@app.post("/predict")
def predict_delivery_time(data: DeliveryInput):

    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw_df = pd.DataFrame([data.model_dump()])
        clean_df = feature_engine.transform(raw_df)
        prediction = model_pipeline.predict(clean_df)

        return {
            "estimated_delivery_minutes": float(prediction[0])
        }

    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=400, detail=str(e))
