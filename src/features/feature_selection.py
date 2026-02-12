import numpy as np
import pandas as pd
from pathlib import Path
import logging



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values."""
    initial_shape = df.shape
    df = df.dropna()
    logger.info(f"Dropped missing values: {initial_shape} → {df.shape}")
    return df



def drop_unused_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove irrelevant or leakage-prone features."""
    columns_to_drop = [
        "rider_id",
        "restaurant_lat",
        "restaurant_long",
        "location_lat",
        "location_long",
        "order_date",
        "order_hour",
        "order_day",
        "distance_bin",
        "city_name",
        "is_weekend",
        "order_month"
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")
    logger.info(f"Dropped unused columns: {columns_to_drop}")
    return df



def load_data(data_path: Path) -> pd.DataFrame:
    """Load dataset from disk."""
    logger.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)



def preprocess_data(data: pd.DataFrame, saved_data_path: Path) -> None:
    """Run preprocessing pipeline and save output."""

    logger.info("Starting preprocessing pipeline")

    cleaned_df = (
        data
        .pipe(drop_missing_values)
        .pipe(drop_unused_features)
    )

    cleaned_df.to_csv(saved_data_path, index=False)
    logger.info(f"Preprocessed data saved to {saved_data_path}")
    logger.info(f"Final dataset shape: {cleaned_df.shape}")





if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent

    preprocessed_data_save_dir = root_path / "data" / "interim"
    preprocessed_data_save_dir.mkdir(exist_ok=True, parents=True)

    preprocessed_data_filename = "food_delivery_interim.csv"
    preprocessed_data_save_path = preprocessed_data_save_dir / preprocessed_data_filename

    data_load_path = root_path / "data" / "cleaned" / "zomato_cleaned.csv"

    df = load_data(data_load_path)
    preprocess_data(data=df, saved_data_path=preprocessed_data_save_path)

