import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import dagshub

from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error
)

from scipy.stats import skew


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


TARGET = "time_taken"

REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
CANDIDATE_ALIAS = "candidate"
EXPERIMENT_NAME = "FoodDeliveryTimePipeline"


def configure_environment():

    try:
        dagshub.init(
            repo_owner="Aryanupadhyay23",
            repo_name="Food-Delivery-Time-prediction",
            mlflow=True
        )

        mlflow.set_experiment(EXPERIMENT_NAME)

        logger.info("MLflow environment configured successfully")

    except Exception:
        logger.exception("Failed to configure MLflow environment")
        raise


def load_data(path: Path) -> pd.DataFrame:

    logger.info(f"Loading dataset from {path}")

    if not path.exists():
        logger.error(f"Dataset not found at {path}")
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path, engine="pyarrow")

        logger.info(f"Dataset loaded successfully with shape {df.shape}")

        return df

    except Exception:
        logger.exception("Failed to load dataset")
        raise


def split_xy(
    df: pd.DataFrame,
    target: str
):

    if target not in df.columns:
        logger.error(f"Target column '{target}' not found")
        raise ValueError(
            f"Target column '{target}' not found"
        )

    X = df.drop(columns=[target])
    y = df[target]

    logger.debug(
        f"Features shape: {X.shape}, Target shape: {y.shape}"
    )

    return X, y


def compute_metrics(
    y_true,
    y_pred,
    prefix=""
):

    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    metrics = {
        f"{prefix}MAE":
            round(
                mean_absolute_error(
                    y_true,
                    y_pred
                ), 4
            ),

        f"{prefix}RMSE":
            round(rmse, 4),

        f"{prefix}R2":
            round(
                r2_score(
                    y_true,
                    y_pred
                ), 4
            ),

        f"{prefix}MedianAE":
            round(
                median_absolute_error(
                    y_true,
                    y_pred
                ), 4
            ),

        f"{prefix}MAPE":
            round(
                mean_absolute_percentage_error(
                    y_true,
                    y_pred
                ), 4
            ),

        f"{prefix}Max_Error":
            round(
                np.max(abs_residuals), 4
            ),

        f"{prefix}Error_Mean":
            round(
                np.mean(residuals), 4
            ),

        f"{prefix}Error_Std":
            round(
                np.std(residuals), 4
            ),

        f"{prefix}Error_Skewness":
            round(
                skew(residuals), 4
            ),

        f"{prefix}P90_Error":
            round(
                np.percentile(
                    abs_residuals,
                    90
                ), 4
            ),

        f"{prefix}P95_Error":
            round(
                np.percentile(
                    abs_residuals,
                    95
                ), 4
            )
    }

    logger.debug(f"{prefix}metrics calculated")

    return metrics


def compute_generalization_gap(
    train_metrics: dict,
    test_metrics: dict
):

    gap = {}

    for key in train_metrics:

        metric_name = key.replace(
            "train_",
            ""
        )

        test_key = f"test_{metric_name}"

        if test_key in test_metrics:

            gap[
                f"gap_{metric_name}"
            ] = round(
                test_metrics[test_key]
                - train_metrics[key],
                4
            )

    logger.debug("Generalization gap metrics calculated")

    return gap


def main():

    try:
        configure_environment()

        client = MlflowClient()

        root_path = Path(__file__).parent.parent.parent

        train_path = (
            root_path
            / "data"
            / "processed"
            / "train.csv"
        )

        test_path = (
            root_path
            / "data"
            / "processed"
            / "test.csv"
        )

        reports_dir = root_path / "reports"

        reports_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        metrics_path = (
            reports_dir
            / "metrics.json"
        )

        logger.info("Loading candidate model from registry")

        try:
            model_version_obj = (
                client.get_model_version_by_alias(
                    REGISTERED_MODEL_NAME,
                    CANDIDATE_ALIAS
                )
            )

        except Exception:
            logger.exception("Candidate model not found")
            raise RuntimeError(
                "No candidate model found"
            )

        version_number = model_version_obj.version

        logger.info(
            f"Evaluating model version {version_number}"
        )

        model = mlflow.pyfunc.load_model(
            f"models:/"
            f"{REGISTERED_MODEL_NAME}"
            f"@{CANDIDATE_ALIAS}"
        )

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        X_train, y_train = split_xy(
            train_df,
            TARGET
        )

        X_test, y_test = split_xy(
            test_df,
            TARGET
        )

        logger.info("Generating train predictions")

        y_train_pred = model.predict(X_train)

        logger.info("Generating test predictions")

        y_test_pred = model.predict(X_test)

        train_metrics = compute_metrics(
            y_train,
            y_train_pred,
            "train_"
        )

        test_metrics = compute_metrics(
            y_test,
            y_test_pred,
            "test_"
        )

        gap_metrics = compute_generalization_gap(
            train_metrics,
            test_metrics
        )

        final_metrics = {
            "model_version": version_number,
            **train_metrics,
            **test_metrics,
            **gap_metrics
        }

        with open(metrics_path, "w") as file:
            json.dump(
                final_metrics,
                file,
                indent=4
            )

        logger.info(
            f"Metrics saved at {metrics_path}"
        )

        with mlflow.start_run(
            run_name=f"evaluation_v{version_number}"
        ):

            logger.info(
                "Logging evaluation metrics to MLflow"
            )

            numeric_metrics = {
                k: v
                for k, v in final_metrics.items()
                if isinstance(v, (int, float))
            }

            mlflow.log_metrics(
                numeric_metrics
            )

            mlflow.set_tag(
                "evaluated_model_version",
                version_number
            )

            mlflow.set_tag(
                "evaluated_alias",
                CANDIDATE_ALIAS
            )

        logger.info(
            "Model evaluation completed successfully"
        )

    except Exception:
        logger.exception(
            "Model evaluation pipeline failed"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()