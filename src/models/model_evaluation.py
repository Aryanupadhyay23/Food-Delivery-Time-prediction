import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error
)
from scipy.stats import skew
import dagshub


# ======================================================
# Configuration
# ======================================================

TARGET = "time_taken"
REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
CANDIDATE_ALIAS = "candidate"
EXPERIMENT_NAME = "FoodDeliveryTimePipeline"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ======================================================
# MLflow Setup
# ======================================================

def configure_mlflow():
    dagshub.init(
        repo_owner="Aryanupadhyay23",
        repo_name="Food-Delivery-Time-prediction",
        mlflow=True
    )
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("DagsHub and MLflow configured successfully.")


# ======================================================
# Utilities
# ======================================================

def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path, engine="pyarrow")


def split_xy(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    return df.drop(columns=[target]), df[target]


def compute_metrics(y_true, y_pred, prefix=""):

    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        f"{prefix}MAE": round(mean_absolute_error(y_true, y_pred), 4),
        f"{prefix}RMSE": round(rmse, 4),
        f"{prefix}R2": round(r2_score(y_true, y_pred), 4),
        f"{prefix}MedianAE": round(median_absolute_error(y_true, y_pred), 4),
        f"{prefix}MAPE": round(mean_absolute_percentage_error(y_true, y_pred), 4),
        f"{prefix}Max_Error": round(np.max(abs_residuals), 4),
        f"{prefix}Error_Mean": round(np.mean(residuals), 4),
        f"{prefix}Error_Std": round(np.std(residuals), 4),
        f"{prefix}Error_Skewness": round(skew(residuals), 4),
        f"{prefix}P90_Error": round(np.percentile(abs_residuals, 90), 4),
        f"{prefix}P95_Error": round(np.percentile(abs_residuals, 95), 4),
    }


def compute_generalization_gap(train_metrics, test_metrics):

    gap = {}
    for key in train_metrics:
        metric_name = key.replace("train_", "")
        test_key = f"test_{metric_name}"
        if test_key in test_metrics:
            gap[f"gap_{metric_name}"] = round(
                test_metrics[test_key] - train_metrics[key],
                4
            )
    return gap


# ======================================================
# Main
# ======================================================

def main():

    try:
        configure_mlflow()
        client = MlflowClient()

        root_path = Path(__file__).parent.parent.parent

        train_path = root_path / "data" / "processed" / "train.csv"
        test_path = root_path / "data" / "processed" / "test.csv"

        reports_dir = root_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = reports_dir / "metrics.json"

        # ======================================================
        # Load Candidate Model
        # ======================================================

        try:
            model_version_obj = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                CANDIDATE_ALIAS
            )
        except Exception:
            raise RuntimeError("No candidate model found in registry.")

        version_number = model_version_obj.version
        logger.info(f"Evaluating candidate version {version_number}")

        model = mlflow.pyfunc.load_model(
            f"models:/{REGISTERED_MODEL_NAME}@{CANDIDATE_ALIAS}"
        )

        # ======================================================
        # Load Data
        # ======================================================

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        X_train, y_train = split_xy(train_df, TARGET)
        X_test, y_test = split_xy(test_df, TARGET)

        # ======================================================
        # Predictions
        # ======================================================

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # ======================================================
        # Metrics
        # ======================================================

        train_metrics = compute_metrics(y_train, y_train_pred, "train_")
        test_metrics = compute_metrics(y_test, y_test_pred, "test_")
        gap_metrics = compute_generalization_gap(train_metrics, test_metrics)

        final_metrics = {
            "model_version": version_number,
            **train_metrics,
            **test_metrics,
            **gap_metrics
        }

        print(json.dumps(final_metrics, indent=2))

        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=4)

        logger.info(f"Metrics saved at {metrics_path}")

        # ======================================================
        # Log Evaluation Run
        # ======================================================

        with mlflow.start_run(run_name=f"evaluation_v{version_number}"):

            numeric_metrics = {
                k: v for k, v in final_metrics.items()
                if isinstance(v, (int, float))
            }

            mlflow.log_metrics(numeric_metrics)
            mlflow.set_tag("evaluated_model_version", version_number)
            mlflow.set_tag("evaluated_alias", CANDIDATE_ALIAS)

        logger.info("Evaluation stage completed successfully.")

    except Exception as e:
        logger.exception(f"Evaluation pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
