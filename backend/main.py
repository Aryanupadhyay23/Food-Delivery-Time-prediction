import os
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import mlflow

try:
    from dotenv import load_dotenv
    if os.path.exists(".env.fd"):
        load_dotenv(".env.fd")
    elif os.path.exists(".env"):
        load_dotenv(".env")
except ImportError:
    pass

try:
    from backend.utility_files.inference_transformer import prepare_inference_input
except ImportError:
    from utility_files.inference_transformer import prepare_inference_input


# Configuration
REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
PRODUCTION_ALIAS = "production"
DAGSHUB_USERNAME = "Aryanupadhyay23"
MLFLOW_TRACKING_URI = (
    "https://dagshub.com/Aryanupadhyay23/Food-Delivery-Time-prediction.mlflow"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_pipeline = None


def configure_mlflow():
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise RuntimeError("DAGSHUB_TOKEN environment variable not set.")

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("MLflow configured successfully using DAGSHUB_TOKEN.")


def load_production_model():
    global model_pipeline
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise RuntimeError("DAGSHUB_TOKEN environment variable not set. Cannot access MLflow.")

    logger.info(f"Loading production model from MLflow Registry ('models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}')...")
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
    model_pipeline = mlflow.pyfunc.load_model(model_uri)
    logger.info("Production model loaded successfully from MLflow Registry.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Asynchronous lifespan manager handling startup and shutdown non-blockingly."""
    await asyncio.to_thread(configure_mlflow)
    await asyncio.to_thread(load_production_model)
    yield


app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="Asynchronous decoupled backend API for predicting food delivery duration with Stacking Regressor (LightGBM + Random Forest)",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for decoupled frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_model_ready():
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or initializing. Please try again shortly."
        )


def _predict(records: list[dict]) -> list[float]:
    """Unified CPU-bound inference routine executed in worker threadpool."""
    raw_df = pd.DataFrame(records)
    clean_df = prepare_inference_input(raw_df)
    preds = model_pipeline.predict(clean_df)
    return [round(float(p), 2) for p in preds]


# ── Health & Metadata Endpoints ───────────────────────────────────────────────

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "healthy",
        "service": "backend",
        "async": True,
        "model_loaded": model_pipeline is not None
    }


@app.get("/model-info", status_code=status.HTTP_200_OK)
async def model_info():
    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": PRODUCTION_ALIAS,
        "model_loaded": model_pipeline is not None
    }


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class DeliveryInput(BaseModel):
    # Route & Geolocation (default: Mumbai Route)
    Restaurant_latitude: float = Field(19.0760, ge=6.0, le=38.0, description="Restaurant latitude")
    Restaurant_longitude: float = Field(72.8777, ge=68.0, le=98.0, description="Restaurant longitude")
    Delivery_location_latitude: float = Field(19.1200, ge=6.0, le=38.0, description="Customer delivery latitude")
    Delivery_location_longitude: float = Field(72.9000, ge=68.0, le=98.0, description="Customer delivery longitude")

    # Delivery Partner & Fleet Profile (ge=18.0 enforces rider age >= 18)
    Delivery_person_Age: float = Field(29.0, ge=18.0, le=70.0, description="Rider age (>= 18)")
    Delivery_person_Ratings: float = Field(4.8, ge=1.0, le=5.0, description="Rider rating between 1.0 and 5.0")
    Delivery_person_ID: str = Field("MUMRES12345", min_length=1, description="Delivery partner ID")
    Type_of_vehicle: str = Field("motorcycle", min_length=1, description="Vehicle type: motorcycle, scooter, electric_scooter, bicycle")
    Vehicle_condition: int = Field(2, ge=0, le=3, description="Vehicle condition: 0 (poor) to 3 (excellent)")
    multiple_deliveries: float = Field(0.0, ge=0.0, le=3.0, description="Active parallel deliveries")

    # Environmental & Traffic Conditions
    Weather_conditions: str = Field("Sunny", min_length=1, description="e.g. Sunny, Cloudy, Fog, Windy, Stormy, Sandstorms")
    Road_traffic_density: str = Field("Medium", min_length=1, description="e.g. Low, Medium, High, Jam")
    Festival: str = Field("No", min_length=1, description="Festival surge: Yes or No")
    City: str = Field("Metropolitian", min_length=1, description="City type: Metropolitian, Urban, Semi-Urban")

    # Order Details & Timing
    ID: str = Field("ORD_001", min_length=1, description="Order ID")
    Type_of_order: str = Field("Meal", min_length=1, description="Order category: Meal, Snack, Drinks, Buffet")
    Order_Date: str = Field(
        default_factory=lambda: datetime.now().strftime("%d-%m-%Y"),
        pattern=r"\d{2}-\d{2}-\d{4}",
        description="Date in DD-MM-YYYY format"
    )
    Time_Orderd: str = Field("19:30", min_length=1, description="Order time in HH:MM format")


class BatchDeliveryInput(BaseModel):
    orders: list[DeliveryInput] = Field(..., description="List of delivery orders for batch prediction")


# ── Prediction Endpoints ──────────────────────────────────────────────────────

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_delivery_time(data: DeliveryInput):
    """Asynchronously predict food delivery time for a single order."""
    _check_model_ready()
    try:
        predictions = await asyncio.to_thread(_predict, [data.model_dump()])
        return {"estimated_delivery_minutes": predictions[0]}
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/predict/batch", status_code=status.HTTP_200_OK)
async def predict_delivery_time_batch(batch_data: BatchDeliveryInput):
    """Asynchronously predict food delivery times for multiple orders."""
    _check_model_ready()
    try:
        dicts = [order.model_dump() for order in batch_data.orders]
        predictions = await asyncio.to_thread(_predict, dicts)
        return {
            "predictions": [
                {
                    "order_id": dicts[i].get("ID", f"order_{i}"),
                    "estimated_delivery_minutes": predictions[i]
                }
                for i in range(len(predictions))
            ],
            "total_orders": len(predictions)
        }
    except Exception as e:
        logger.exception("Batch prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch inference error: {str(e)}"
        )
