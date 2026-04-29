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

    logger.info(f"Loading dataset from {data_path}")

    df = pd.read_csv(data_path)

    logger.info(f"Dataset loaded successfully with shape {df.shape}")

    return df


def city_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Creating city based features")

    df["city_name"] = df["Delivery_person_ID"].str.split("RES").str[0]

    city_mapping = {
        "DEH": "dehradun",
        "KOC": "kochi",
        "PUNE": "pune",
        "LUDH": "ludhiana",
        "KNP": "kanpur",
        "MUM": "mumbai",
        "MYS": "mysore",
        "HYD": "hyderabad",
        "KOL": "kolkata",
        "RANCHI": "ranchi",
        "COIMB": "coimbatore",
        "CHEN": "chennai",
        "JAP": "jaipur",
        "SUR": "surat",
        "BANG": "bangalore",
        "GOA": "goa",
        "AURG": "aurangabad",
        "AGR": "agra",
        "VAD": "vadodara",
        "ALH": "prayagraj",
        "BHP": "bhopal",
        "INDO": "indore"
    }

    df["city_name"] = df["city_name"].replace(city_mapping)
    df["city_type"] = df["City"].str.lower()

    logger.debug("City features created successfully")

    return df


def age_feature(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Validating rider age values")

    initial_rows = len(df)

    df["Delivery_person_Age"] = pd.to_numeric(
        df["Delivery_person_Age"],
        errors="coerce"
    )

    df = df[df["Delivery_person_Age"] >= 18]

    removed_rows = initial_rows - len(df)

    logger.info(f"Removed {removed_rows} rows with invalid age")

    return df


def rating_feature(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Validating rider ratings")

    initial_rows = len(df)

    df = df[df["Delivery_person_Ratings"] <= 5]

    removed_rows = initial_rows - len(df)

    logger.info(f"Removed {removed_rows} rows with invalid ratings")

    return df


def decimal_to_hhmm(value):

    if pd.isna(value) or str(value).strip() == "":
        return np.nan

    value = str(value).strip()

    if ":" in value:
        try:
            hour, minute = map(int, value.split(":"))
            return f"{hour % 24:02d}:{minute:02d}"
        except ValueError:
            return np.nan

    try:
        total_minutes = round(float(value) * 24 * 60)
        hour = (total_minutes // 60) % 24
        minute = total_minutes % 60

        return f"{hour:02d}:{minute:02d}"

    except ValueError:
        return np.nan


def time_of_day(value):

    if pd.isna(value):
        return np.nan

    time_obj = pd.to_datetime(
        value,
        format="%H:%M",
        errors="coerce"
    )

    if pd.isna(time_obj):
        return np.nan

    hour = time_obj.hour

    if 5 <= hour < 8:
        return "early_morning"
    if 8 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 14:
        return "lunch_peak"
    if 14 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 20:
        return "evening_snacks"
    if 20 <= hour < 23:
        return "dinner_peak"

    return "late_night"


def order_date_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Creating date related features")

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["order_day"] = df["Order_Date"].dt.day
    df["order_month"] = df["Order_Date"].dt.month
    df["day_name"] = df["Order_Date"].dt.day_name().str.lower()
    df["is_weekend"] = df["Order_Date"].dt.dayofweek.isin([5, 6]).astype(int)

    logger.debug("Date features created")

    return df


def cleaning_time_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Creating time related features")

    order_time = df["Time_Orderd"].apply(decimal_to_hhmm)
    pickup_time = df["Time_Order_picked"].apply(decimal_to_hhmm)

    df["order_time"] = order_time
    df["order_pickup_time"] = pickup_time
    df["time_of_day"] = order_time.apply(time_of_day)

    order_dt = pd.to_datetime(order_time, format="%H:%M", errors="coerce")
    pickup_dt = pd.to_datetime(pickup_time, format="%H:%M", errors="coerce")

    df["order_hour"] = order_dt.dt.hour

    prep_minutes = (pickup_dt - order_dt).dt.total_seconds() / 60
    prep_minutes = prep_minutes.where(prep_minutes >= 0, prep_minutes + 1440)

    df["prep_time_minutes"] = prep_minutes

    logger.debug("Time features created")

    return df


def lowercase_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Standardizing text columns")

    columns = [
        "Weather_conditions",
        "Road_traffic_density",
        "Type_of_order",
        "Festival"
    ]

    for column in columns:
        df[column] = df[column].str.lower()

    return df


def clean_location_features(
    df: pd.DataFrame,
    threshold: float = 1.0
) -> pd.DataFrame:

    logger.info("Cleaning coordinate columns")

    columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude"
    ]

    for column in columns:
        df[column] = df[column].abs()
        df.loc[df[column] < threshold, column] = np.nan

    logger.debug("Coordinate cleaning completed")

    return df


def add_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Calculating delivery distance")

    radius = 6371.0

    lat1 = np.radians(df["Restaurant_latitude"])
    lon1 = np.radians(df["Restaurant_longitude"])
    lat2 = np.radians(df["Delivery_location_latitude"])
    lon2 = np.radians(df["Delivery_location_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    df["haversine_distance_km"] = radius * c

    logger.debug("Distance feature created")

    return df


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Dropping rows with missing values")

    initial_shape = df.shape

    df = df.dropna()

    logger.info(f"Dataset shape changed from {initial_shape} to {df.shape}")

    return df


def rename_features(df: pd.DataFrame) -> pd.DataFrame:

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
        "Delivery_person_ID": "rider_id",
        "haversine_distance_km": "distance",
        "Order_Date": "order_date"
    }

    return df.rename(columns=rename_map)


def drop_unused_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Dropping unused columns")

    columns = [
        "ID",
        "rider_id",
        "restaurant_lat",
        "restaurant_long",
        "location_lat",
        "location_long",
        "Time_Orderd",
        "Time_Order_picked",
        "City",
        "order_time",
        "order_pickup_time",
        "prep_time_minutes",
        "order_date",
        "order_hour",
        "order_day",
        "order_month",
        "distance_bin",
        "city_name",
        "is_weekend"
    ]

    df = df.drop(columns=columns, errors="ignore")

    logger.debug("Unused columns removed")

    return df


def clean_data(data: pd.DataFrame, save_path: Path):

    logger.info("Starting data cleaning pipeline")

    try:
        final_df = (
            data
            .pipe(city_features)
            .pipe(age_feature)
            .pipe(rating_feature)
            .pipe(order_date_features)
            .pipe(cleaning_time_features)
            .pipe(lowercase_features)
            .pipe(clean_location_features)
            .pipe(add_haversine_distance)
            .pipe(drop_missing_values)
            .pipe(rename_features)
            .pipe(drop_unused_features)
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        final_df.to_csv(save_path, index=False)

        logger.info(f"Cleaned dataset saved to {save_path}")
        logger.info(f"Final dataset shape: {final_df.shape}")

    except Exception:
        logger.exception("Data cleaning pipeline failed")
        raise


if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent

    raw_path = root_path / "data" / "raw" / "Zomato-Dataset.csv"

    save_path = (
        root_path
        / "data"
        / "processed"
        / "food_delivery_cleaned.csv"
    )

    df = load_data(raw_path)

    clean_data(df, save_path)

    logger.info("Data cleaning stage completed successfully")