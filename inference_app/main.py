import os
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import pandas as pd

import mlflow
import dagshub
from dotenv import load_dotenv

from inference_app.utility_files.inference_transformer import FoodDeliveryFeatureEngine


# CONFIGURATION

REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
PRODUCTION_ALIAS = "production"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Food Delivery Time Prediction API")

model_pipeline = None
feature_engine = FoodDeliveryFeatureEngine()



# MLFLOW + DAGSHUB SETUP

def configure_mlflow():

    # Explicit experiment name
    experiment_name = "FoodDeliveryTimePipeline"

    # Initialize DagsHub MLflow tracking
    dagshub.init(
        repo_owner="Aryanupadhyay23",
        repo_name="Food-Delivery-Time-prediction",
        mlflow=True
    )

    # Set experiment directly
    mlflow.set_experiment(experiment_name)

    logger.info("MLflow connected via DagsHub using explicit experiment name.")



# LOAD PRODUCTION MODEL FROM REGISTRY

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



# STARTUP EVENT

@app.on_event("startup")
def startup_event():
    configure_mlflow()
    load_production_model()



# PYDANTIC SCHEMA

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



# PREDICTION ENDPOINT

@app.post("/predict")
def predict_delivery_time(data: DeliveryInput):

    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Convert validated input to DataFrame
        raw_df = pd.DataFrame([data.model_dump()])

        # 2. Feature Engineering
        clean_df = feature_engine.transform(raw_df)

        # 3. Predict from MLflow model
        prediction = model_pipeline.predict(clean_df)

        return {
            "estimated_delivery_minutes": float(prediction[0])
        }

    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=400, detail=str(e))
