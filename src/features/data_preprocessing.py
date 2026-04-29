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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

TARGET_COLUMN = "time_taken"


def load_data(data_path: Path) -> pd.DataFrame:

    logger.info(f"Loading training dataset from {data_path}")

    try:
        df = pd.read_csv(data_path)

        logger.info(f"Dataset loaded successfully with shape {df.shape}")

        return df

    except Exception:
        logger.exception("Failed to load training dataset")
        raise


def build_preprocessor() -> ColumnTransformer:

    logger.info("Building preprocessing pipeline")

    numeric_features = [
        "rider_age",
        "rider_ratings",
        "distance"
    ]

    nominal_features = [
        "weather",
        "order_type",
        "vehicle_type",
        "festival",
        "city_type",
        "day_name",
        "time_of_day"
    ]

    ordinal_features = [
        "traffic_density"
    ]

    traffic_order = [
        "low",
        "medium",
        "high",
        "jam"
    ]

    logger.debug(f"Numeric features: {numeric_features}")
    logger.debug(f"Nominal features: {nominal_features}")
    logger.debug(f"Ordinal features: {ordinal_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False
                ),
                nominal_features
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[traffic_order],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                ordinal_features
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
        n_jobs=-1
    )

    logger.info("Preprocessing pipeline created successfully")

    return preprocessor


def fit_and_save_preprocessor(
    train_df: pd.DataFrame,
    artifact_dir: Path
) -> None:

    logger.info("Starting data preprocessing stage")

    try:
        X_train = train_df.drop(columns=[TARGET_COLUMN])

        logger.info(f"Training feature matrix shape: {X_train.shape}")

        preprocessor = build_preprocessor()

        logger.info("Fitting preprocessor on training data")

        preprocessor.fit(X_train)

        logger.info("Preprocessor fitted successfully")

        artifact_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        save_path = artifact_dir / "preprocessor.pkl"

        joblib.dump(preprocessor, save_path)

        logger.info(f"Preprocessor saved at {save_path}")

    except KeyError:
        logger.exception(
            f"Target column '{TARGET_COLUMN}' not found in dataset"
        )
        raise

    except Exception:
        logger.exception("Data preprocessing stage failed")
        raise


if __name__ == "__main__":

    try:
        root_path = Path(__file__).parent.parent.parent

        train_path = (
            root_path
            / "data"
            / "processed"
            / "train.csv"
        )

        artifact_dir = root_path / "artifacts"

        train_df = load_data(train_path)

        fit_and_save_preprocessor(
            train_df=train_df,
            artifact_dir=artifact_dir
        )

        logger.info("Data preprocessing stage completed successfully")

    except Exception:
        logger.exception("Pipeline execution failed")
        raise