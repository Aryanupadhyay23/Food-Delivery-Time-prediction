import numpy as np
import pandas as pd
from pathlib import Path
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded with shape {df.shape}")
    return df


def city_features(df):
    logger.info("Creating city-related features")

    df["city_name"] = df["Delivery_person_ID"].str.split("RES").str[0]

    city_mapping = {
        'DEH': 'dehradun', 'KOC': 'kochi', 'PUNE': 'pune',
        'LUDH': 'ludhiana', 'KNP': 'kanpur', 'MUM': 'mumbai',
        'MYS': 'mysore', 'HYD': 'hyderabad', 'KOL': 'kolkata',
        'RANCHI': 'ranchi', 'COIMB': 'coimbatore', 'CHEN': 'chennai',
        'JAP': 'jaipur', 'SUR': 'surat', 'BANG': 'bangalore',
        'GOA': 'goa', 'AURG': 'aurangabad', 'AGR': 'agra',
        'VAD': 'vadodara', 'ALH': 'prayagraj',
        'BHP': 'bhopal', 'INDO': 'indore'
    }

    df["city_name"] = df["city_name"].replace(city_mapping)
    df["city_type"] = df["City"].str.lower()

    return df


def age_feature(df):
    logger.info("Filtering delivery persons with age >= 18")

    df["Delivery_person_Age"] = pd.to_numeric(
        df["Delivery_person_Age"], errors="coerce"
    )

    return df[df["Delivery_person_Age"] >= 18]



def rating_feature_cleaning(df):
    logger.info("Filtering delivery persons with rating <= 5")
    return df[df["Delivery_person_Ratings"] <= 5]


def decimal_to_hhmm(x):
    if pd.isna(x) or str(x).strip() == "":
        return np.nan

    x = str(x).strip()

    if ":" in x:
        try:
            hour, minute = int(x.split(":")[0]), int(x.split(":")[1])
            hour = hour % 24
            return f"{hour:02d}:{minute:02d}"
        except:
            return np.nan

    try:
        frac = float(x)
        total_minutes = round(frac * 24 * 60)
        hours = (total_minutes // 60) % 24
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"
    except:
        return np.nan


def time_of_day(time):
    if pd.isna(time):
        return np.nan

    time = str(time)

    if "05:00" <= time < "08:00":
        return "early_morning"
    elif "08:00" <= time < "11:00":
        return "breakfast"
    elif "11:00" <= time < "14:00":
        return "lunch_peak"
    elif "14:00" <= time < "17:00":
        return "afternoon"
    elif "17:00" <= time < "20:00":
        return "evening_snacks"
    elif "20:00" <= time < "23:00":
        return "dinner_peak"
    else:
        return "late_night"


def order_date_features(df):
    logger.info("Creating order date features")
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        dayfirst=True,
        errors="coerce"
    )
    df["order_day"] = df["Order_Date"].dt.day
    df["order_month"] = df["Order_Date"].dt.month
    df["day_name"] = df["Order_Date"].dt.day_name().str.lower()
    df["is_weekend"] = df["Order_Date"].dt.dayofweek.isin([5, 6]).astype(int)
    return df

def cleaning_time_features(df):
    logger.info("Creating time-based features")

    # Convert raw time columns once
    order_time = df["Time_Orderd"].apply(decimal_to_hhmm)
    pickup_time = df["Time_Order_picked"].apply(decimal_to_hhmm)

    # Assign engineered columns
    df["order_time"] = order_time
    df["order_pickup_time"] = pickup_time
    df["time_of_day"] = order_time.apply(time_of_day)

    # Convert to datetime for computation
    order_dt = pd.to_datetime(order_time, format="%H:%M", errors="coerce")
    pickup_dt = pd.to_datetime(pickup_time, format="%H:%M", errors="coerce")

    # order hour (0–23)
    df["order_hour"] = order_dt.dt.hour

    # Compute preparation time (handle midnight crossing)
    prep_minutes = (pickup_dt - order_dt).dt.total_seconds() / 60
    prep_minutes = prep_minutes.where(prep_minutes >= 0, prep_minutes + 1440)

    df["prep_time_minutes"] = prep_minutes

    return df



def to_lower(df):
    logger.info("Converting categorical columns to lowercase")

    cols = [
        "Weather_conditions",
        "Road_traffic_density",
        "Type_of_order",
        "Festival"
    ]

    for col in cols:
        df[col] = df[col].str.lower()

    return df


def clean_location_features(df, threshold=1.0):
    logger.info("Cleaning latitude and longitude features")

    location_cols = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude"
    ]

    for col in location_cols:
        df[col] = df[col].abs()
        df.loc[df[col] < threshold, col] = np.nan

    return df


def add_haversine_distance(df):
    logger.info("Adding haversine distance feature")

    # Earth radius in kilometers
    R = 6371.0

    lat1 = np.radians(df["Restaurant_latitude"])
    lon1 = np.radians(df["Restaurant_longitude"])
    lat2 = np.radians(df["Delivery_location_latitude"])
    lon2 = np.radians(df["Delivery_location_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    df["haversine_distance_km"] = R * c

    return df


def drop_unused_features(df):
    logger.info("Dropping unused columns")

    cols_to_drop = [
        "ID",
        "rider_id"
        "Time_Orderd",
        "Time_Order_picked",
        "City"
    ]

    return df.drop(columns=cols_to_drop, errors="ignore")


def rename_features(df):
    logger.info("Renaming columns")

    rename_map = {
        "Delivery_person_Age": "rider_age",
        "Delivery_person_Ratings": "rider_ratings",
        "Restaurant_latitude": "restaurant_lat",
        "Restaurant_longitude": "restaurant_long",
        "Delivery_location_latitude": "location_lat",
        "Delivery_location_longitude": "location_long",
        "Weather_conditions": "weather",
        "Road_traffic_density": "traffic_density",
        "Vehicle_condition": "vehicle_condition",
        "Type_of_order": "order_type",
        "Type_of_vehicle": "vehicle_type",
        "multiple_deliveries": "multiple_deliveries",
        "Festival": "festival",
        "Time_taken (min)": "time_taken",
        "Delivery_person_ID":"rider_id",
        "haversine_distance_km":"distance",
        "Order_Date":"order_date"
    }

    return df.rename(columns=rename_map)



def cleaned_data(data: pd.DataFrame, saved_data_path: Path):
    logger.info("Starting full data cleaning pipeline")

    cleaned_df = (
        data
        .pipe(city_features)
        .pipe(age_feature)
        .pipe(rating_feature_cleaning)
        .pipe(order_date_features)
        .pipe(cleaning_time_features)
        .pipe(to_lower)
        .pipe(clean_location_features)
        .pipe(add_haversine_distance)
        .pipe(drop_unused_features)
        .pipe(rename_features)
    )

    cleaned_df.to_csv(saved_data_path, index=False)
    logger.info(f"Cleaned data saved at {saved_data_path}")


if __name__ == "__main__":
    logger.info("Data cleaning script started")

    root_path = Path(__file__).parent.parent.parent

    cleaned_data_save_dir = root_path / "data" / "cleaned"
    cleaned_data_save_dir.mkdir(exist_ok=True, parents=True)

    cleaned_data_filename = "zomato_cleaned.csv"
    cleaned_data_save_path = cleaned_data_save_dir / cleaned_data_filename

    data_load_path = root_path / "data" / "raw" / "Zomato-Dataset.csv"

    df = load_data(data_load_path)
    cleaned_data(data=df, saved_data_path=cleaned_data_save_path)

    logger.info("Data cleaning pipeline completed successfully")

