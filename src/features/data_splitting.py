import pandas as pd
from pathlib import Path
import logging
import yaml
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_params(param_path: Path) -> dict:

    logger.info(f"Loading parameters from {param_path}")

    try:
        with open(param_path, "r") as file:
            params = yaml.safe_load(file)

        logger.debug("Parameters loaded successfully")

        return params["data_splitting"]

    except Exception:
        logger.exception("Failed to load parameters")
        raise


def load_data(data_path: Path) -> pd.DataFrame:

    logger.info(f"Loading dataset from {data_path}")

    try:
        df = pd.read_csv(data_path)

        logger.info(f"Dataset loaded successfully with shape {df.shape}")

        return df

    except Exception:
        logger.exception("Failed to load dataset")
        raise


def split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int
):

    logger.info("Starting train test split")

    logger.debug(
        f"Split configuration: "
        f"test_size={test_size}, "
        f"random_state={random_state}"
    )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    logger.info(
        f"Split completed. "
        f"Train shape: {train_df.shape}, "
        f"Test shape: {test_df.shape}"
    )

    return train_df, test_df


def save_split_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: Path
):

    logger.info(f"Saving split datasets to {save_dir}")

    try:
        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        train_path = save_dir / "train.csv"
        test_path = save_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Train dataset saved at {train_path}")
        logger.info(f"Test dataset saved at {test_path}")

    except Exception:
        logger.exception("Failed to save split datasets")
        raise


if __name__ == "__main__":

    try:
        root_path = Path(__file__).parent.parent.parent

        data_path = (
            root_path
            / "data"
            / "processed"
            / "food_delivery_cleaned.csv"
        )

        save_dir = (
            root_path
            / "data"
            / "processed"
        )

        param_path = root_path / "params.yaml"

        params = load_params(param_path)

        df = load_data(data_path)

        train_df, test_df = split_data(
            df=df,
            test_size=params["test_size"],
            random_state=params["random_state"]
        )

        save_split_data(
            train_df=train_df,
            test_df=test_df,
            save_dir=save_dir
        )

        logger.info("Data splitting stage completed successfully")

    except Exception:
        logger.exception("Data splitting pipeline failed")
        raise