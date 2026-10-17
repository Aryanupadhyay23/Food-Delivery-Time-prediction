import json
import time
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

try:
    from backend.src.models.mlflow_utils import (
        configure_mlflow,
        start_pipeline_stage_run
    )
except ImportError:
    from mlflow_utils import (
        configure_mlflow,
        start_pipeline_stage_run
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


TARGET = "time_taken"
REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
CANDIDATE_ALIAS = "candidate"


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


def split_xy(df: pd.DataFrame, target: str):
    if target not in df.columns:
        logger.error(f"Target column '{target}' not found")
        raise ValueError(f"Target column '{target}' not found")

    X = df.drop(columns=[target])
    y = df[target]
    logger.debug(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def compute_metrics(y_true, y_pred, prefix=""):
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        f"{prefix}MAE": round(mean_absolute_error(y_true, y_pred), 4),
        f"{prefix}RMSE": round(rmse, 4),
        f"{prefix}R2": round(r2_score(y_true, y_pred), 4),
        f"{prefix}MedianAE": round(median_absolute_error(y_true, y_pred), 4),
        f"{prefix}MAPE": round(mean_absolute_percentage_error(y_true, y_pred), 4),
        f"{prefix}Max_Error": round(float(np.max(abs_residuals)), 4),
        f"{prefix}Error_Mean": round(float(np.mean(residuals)), 4),
        f"{prefix}Error_Std": round(float(np.std(residuals)), 4),
        f"{prefix}Error_Skewness": round(float(skew(residuals)), 4),
        f"{prefix}P90_Error": round(float(np.percentile(abs_residuals, 90)), 4),
        f"{prefix}P95_Error": round(float(np.percentile(abs_residuals, 95)), 4)
    }

    logger.debug(f"{prefix}metrics calculated successfully")
    return metrics


def compute_generalization_gap(train_metrics: dict, test_metrics: dict):
    gap = {}
    for key in train_metrics:
        metric_name = key.replace("train_", "")
        test_key = f"test_{metric_name}"
        if test_key in test_metrics:
            gap[f"gap_{metric_name}"] = round(
                test_metrics[test_key] - train_metrics[key],
                4
            )
    logger.debug("Generalization gap metrics calculated")
    return gap


def main():
    try:
        configure_mlflow()
        client = MlflowClient()

        root_path = Path(__file__).resolve().parents[2]
        train_path = root_path / "data" / "processed" / "train.csv"
        test_path = root_path / "data" / "processed" / "test.csv"
        reports_dir = root_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = reports_dir / "metrics.json"

        logger.info("Loading candidate model from registry")
        try:
            model_version_obj = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                CANDIDATE_ALIAS
            )
        except Exception:
            logger.exception("Candidate model not found")
            raise RuntimeError("No candidate model found in MLflow registry")

        version_number = str(model_version_obj.version)
        logger.info(f"Evaluating candidate model version {version_number}")

        local_model_path = root_path / "models" / "stacking_lgb_rf_pipeline.joblib"
        if not local_model_path.exists():
            local_model_path = root_path / "models" / "stacking_cat_rf_pipeline.joblib"

        try:
            model = mlflow.pyfunc.load_model(
                f"models:/{REGISTERED_MODEL_NAME}@{CANDIDATE_ALIAS}"
            )
        except Exception as err:
            if local_model_path.exists():
                import joblib
                logger.warning(
                    f"MLflow model load fallback to local artifact {local_model_path}: {err}"
                )
                model = joblib.load(local_model_path)
            else:
                raise

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        X_train, y_train = split_xy(train_df, TARGET)
        X_test, y_test = split_xy(test_df, TARGET)

        logger.info("Generating predictions and computing metrics")
        eval_start = time.time()
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        eval_duration = time.time() - eval_start

        train_metrics = compute_metrics(y_train, y_train_pred, "train_")
        test_metrics = compute_metrics(y_test, y_test_pred, "test_")
        gap_metrics = compute_generalization_gap(train_metrics, test_metrics)

        final_metrics = {
            "model_version": version_number,
            "evaluation_duration_seconds": round(eval_duration, 4),
            **train_metrics,
            **test_metrics,
            **gap_metrics
        }

        with open(metrics_path, "w") as file:
            json.dump(final_metrics, file, indent=4)
        logger.info(f"Metrics saved locally at {metrics_path}")

        # Start nested Stage 02 evaluation run under parent pipeline
        with start_pipeline_stage_run(
            stage_name="Model_Evaluation",
            stage_number="02",
            tags={
                "evaluated_model_version": version_number,
                "evaluated_alias": CANDIDATE_ALIAS,
                "model_name": REGISTERED_MODEL_NAME
            }
        ) as eval_run:
            logger.info(f"Logging evaluation metrics to Stage 02 run (ID: {eval_run.info.run_id})")

            numeric_metrics = {
                k: v for k, v in final_metrics.items()
                if isinstance(v, (int, float))
            }
            mlflow.log_metrics(numeric_metrics)

            mlflow.log_param("train_samples", len(train_df))
            mlflow.log_param("test_samples", len(test_df))
            mlflow.log_param("evaluated_version", version_number)

            # Log metrics.json as a run artifact
            mlflow.log_artifact(str(metrics_path), artifact_path="evaluation_reports")

            mlflow.set_tag("evaluation_status", "COMPLETED")

        logger.info("Stage 02: Model Evaluation completed successfully")

    except Exception:
        logger.exception("Model evaluation pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()