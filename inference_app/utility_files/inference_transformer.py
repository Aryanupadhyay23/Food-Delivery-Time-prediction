import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ── Time Utilities ────────────────────────────────────────────────────────────

def decimal_to_hhmm(x) -> str | None:
    if pd.isna(x) or str(x).strip() == "":
        return np.nan
    x = str(x).strip()
    if ":" in x:
        try:
            h, m = int(x.split(":")[0]), int(x.split(":")[1])
            return f"{h % 24:02d}:{m:02d}"
        except:
            return np.nan
    try:
        total = round(float(x) * 24 * 60)
        return f"{(total // 60) % 24:02d}:{total % 60:02d}"
    except:
        return np.nan


def time_of_day(time_str) -> str:
    if pd.isna(time_str):
        return "unknown"
    t = pd.to_datetime(time_str, format="%H:%M", errors="coerce")
    if pd.isna(t):
        return "unknown"
    h = t.hour
    if 5 <= h < 8:    return "early_morning"
    elif 8 <= h < 11: return "breakfast"
    elif 11 <= h < 14: return "lunch_peak"
    elif 14 <= h < 17: return "afternoon"
    elif 17 <= h < 20: return "evening_snacks"
    elif 20 <= h < 23: return "dinner_peak"
    else:              return "late_night"


# ── Feature Engineering ───────────────────────────────────────────────────────

def extract_city_features(df: pd.DataFrame) -> pd.DataFrame:
    city_mapping = {
        'DEH': 'dehradun', 'KOC': 'kochi',    'PUNE': 'pune',
        'LUDH': 'ludhiana', 'KNP': 'kanpur',  'MUM': 'mumbai',
        'MYS': 'mysore',   'HYD': 'hyderabad','KOL': 'kolkata',
        'RANCHI': 'ranchi', 'COIMB': 'coimbatore', 'CHEN': 'chennai',
        'JAP': 'jaipur',   'SUR': 'surat',    'BANG': 'bangalore',
        'GOA': 'goa',      'AURG': 'aurangabad', 'AGR': 'agra',
        'VAD': 'vadodara', 'ALH': 'prayagraj', 'BHP': 'bhopal', 'INDO': 'indore'
    }
    if "Delivery_person_ID" in df.columns:
        df["city_name"] = (
            df["Delivery_person_ID"]
            .astype(str)
            .str.split("RES")
            .str[0]
            .replace(city_mapping)
        )
    if "City" in df.columns:
        df["city_type"] = df["City"].astype(str).str.strip().str.lower()
    return df


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Order_Date" not in df.columns:
        return df
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
    df["day_name"]  = df["Order_Date"].dt.day_name().str.lower()
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Time_Orderd" not in df.columns:
        return df
    order_time = df["Time_Orderd"].apply(decimal_to_hhmm)
    df["time_of_day"] = order_time.apply(time_of_day)
    return df


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Weather_conditions", "Road_traffic_density", "Type_of_order", "Festival"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def compute_haversine(df: pd.DataFrame) -> pd.DataFrame:
    loc_cols = [
        "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude"
    ]
    if not all(c in df.columns for c in loc_cols):
        return df

    R = 6371.0
    lat1 = np.radians(df["Restaurant_latitude"])
    lon1 = np.radians(df["Restaurant_longitude"])
    lat2 = np.radians(df["Delivery_location_latitude"])
    lon2 = np.radians(df["Delivery_location_longitude"])

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    df["distance"] = R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "Delivery_person_Age":     "rider_age",
        "Delivery_person_Ratings": "rider_ratings",
        "Weather_conditions":      "weather",
        "Road_traffic_density":    "traffic_density",
        "Vehicle_condition":       "vehicle_condition",
        "Type_of_order":           "order_type",
        "Type_of_vehicle":         "vehicle_type",
        "Festival":                "festival",
        "multiple_deliveries":     "multiple_deliveries",
    })


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ID", "Delivery_person_ID", "rider_id",
        "Time_Orderd", "Time_Order_picked",
        "City", "city_name",
        "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude",
        "Order_Date", "order_date",
        "order_time", "order_hour", "order_day", "order_month",
        "is_weekend", "distance_bin", "prep_time_minutes",
    ]
    return df.drop(columns=cols, errors="ignore")


# ── Public Entry Point ────────────────────────────────────────────────────────

def prepare_inference_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a raw input DataFrame (from the API request) into
    the feature format expected by the production model.

    Mirrors the training-time data_preparation pipeline but skips
    steps that are training-only (null drops, age/rating filters,
    distance bins, target column handling).
    """
    return (
        df
        .pipe(extract_city_features)
        .pipe(extract_date_features)
        .pipe(extract_time_features)
        .pipe(standardize_categoricals)
        .pipe(compute_haversine)
        .pipe(rename_columns)
        .pipe(drop_unused_columns)
    )