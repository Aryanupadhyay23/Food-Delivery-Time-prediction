import pandas as pd
from pathlib import Path
import logging
import yaml
from sklearn.model_selection import train_test_split



# ---------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



# ---------------- Load Params ----------------
def load_params(param_path: Path) -> dict:
    logger.info(f"Loading parameters from {param_path}")
    with open(param_path, "r") as file:
        params = yaml.safe_load(file)
    return params["data_splitting"]

# ---------------- Load Data ----------------
def load_data(data_path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset shape: {df.shape}")
    return df

# ---------------- Split Data ----------------
def split_data(df: pd.DataFrame, test_size: float, random_state: int):

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    logger.info(
        f"Split complete | "
        f"Train shape: {train_df.shape} | "
        f"Test shape: {test_df.shape}"
    )

    return train_df, test_df

# ---------------- Save Data ----------------
def save_split_data(train_df, test_df, save_dir: Path):

    save_dir.mkdir(parents=True, exist_ok=True)

    train_path = save_dir / "train.csv"
    test_path = save_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train data saved at: {train_path}")
    logger.info(f"Test data saved at: {test_path}")


# ---------------- Main ----------------
if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent

    data_path = root_path / "data" / "interim" / "food_delivery_interim.csv"
    save_dir = root_path / "data" / "processed"
    param_path = root_path / "params.yaml"

    params = load_params(param_path)

    df = load_data(data_path)

    train_df, test_df = split_data(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"]
    )


    save_split_data(train_df, test_df, save_dir)

    logger.info("Data splitting stage completed successfully")
