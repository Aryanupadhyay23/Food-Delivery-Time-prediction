import pandas as pd
from pathlib import Path
import logging
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn import set_config


set_config(transform_output="pandas")

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_COLUMN = "time_taken"

# Load Data
def load_data(data_path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset shape: {df.shape}")
    return df

# Build Preprocessor 
def build_preprocessor() -> ColumnTransformer:

    num_cols = ["rider_age", "rider_ratings", "distance"]

    nominal_cat_cols = [
        "weather",
        "order_type",
        "vehicle_type",
        "festival",
        "city_type",
        "day_name",
        "time_of_day"
    ]

    ordinal_cat_cols = ["traffic_density"]

    traffic_order = ["low", "medium", "high", "jam"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                num_cols
            ),
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False
                ),
                nominal_cat_cols
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[traffic_order],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                ordinal_cat_cols
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
        n_jobs=-1
    )

    return preprocessor

# Transformation Stage
def fit_and_save_preprocessor(
    train_df: pd.DataFrame,
    artifact_dir: Path
) -> None:

    logger.info("Starting transformation stage")

    # Separate features and target
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    logger.info(f"Training feature shape: {X_train.shape}")

    # Build preprocessor
    preprocessor = build_preprocessor()

    # Fit ONLY on training data (leakage-safe)
    preprocessor.fit(X_train)

    logger.info("Preprocessor fitted successfully")

    # Save artifact
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = artifact_dir / "preprocessor.pkl"

    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Preprocessor saved at {preprocessor_path}")



# Main 
if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent

    train_path = root_path / "data" / "processed" / "train.csv"
    artifact_dir = root_path / "artifacts"

    train_df = load_data(train_path)

    fit_and_save_preprocessor(train_df, artifact_dir)


    logger.info("Data transformation stage completed successfully")
