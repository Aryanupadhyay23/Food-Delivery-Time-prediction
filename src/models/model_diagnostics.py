import os
import time
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration

TARGET = "time_taken"
REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"

VALIDATED_ALIAS = "validated"
PRODUCTION_ALIAS = "production"

MAX_ALLOWED_LATENCY_MS = 50
MAX_ALLOWED_EXTREME_ERRORS = 2
EXTREME_ERROR_THRESHOLD = 15


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# MLflow Configuration

def configure_mlflow():
    root_path = Path(__file__).parent.parent.parent
    dotenv_path = root_path / ".env"

    if not dotenv_path.exists():
        raise FileNotFoundError(".env file not found in project root.")

    load_dotenv(dotenv_path)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not tracking_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI missing.")
    if not experiment_name:
        raise EnvironmentError("MLFLOW_EXPERIMENT_NAME missing.")
    if not username or not password:
        raise EnvironmentError("MLFLOW credentials missing.")

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("MLflow configured successfully.")


# Utility 

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, engine="pyarrow")


def plot_residuals(y_true, y_pred, save_dir):
    residuals = y_true - y_pred

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--')
    plt.title("Actual vs Predicted")

    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title("Residual Distribution")

    save_path = save_dir / "residual_analysis.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def test_latency(model, sample_input, iterations=500):
    model.predict(sample_input)  

    start = time.time()
    for _ in range(iterations):
        model.predict(sample_input)
    end = time.time()

    return ((end - start) / iterations) * 1000


# Main

def main():
    try:
        configure_mlflow()
        client = MlflowClient()

        root_path = Path(__file__).parent.parent.parent
        test_path = root_path / "data" / "processed" / "test.csv"
        reports_dir = root_path / "reports" / "diagnostics"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Load validated model

        try:
            model_version = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                VALIDATED_ALIAS
            )
        except Exception:
            raise RuntimeError("No validated model found.")

        version_number = model_version.version

        logger.info(f"Running diagnostics on validated version {version_number}")

        model = mlflow.pyfunc.load_model(
            f"models:/{REGISTERED_MODEL_NAME}@{VALIDATED_ALIAS}"
        )

        # Load Data

        df_test = load_data(test_path)
        X = df_test.drop(columns=[TARGET])
        y = df_test[TARGET]

        y_pred = model.predict(X)

        # Diagnostics 

        residual_plot_path = plot_residuals(y, y_pred, reports_dir)

        sample_row = X.iloc[[0]]
        avg_latency = test_latency(model, sample_row)

        residuals = np.abs(y - y_pred)
        num_extreme_errors = int(np.sum(residuals > EXTREME_ERROR_THRESHOLD))

        logger.info(f"Latency: {avg_latency:.2f} ms")
        logger.info(f"Extreme errors: {num_extreme_errors}")

        # Log new diagnostics run 

        with mlflow.start_run(run_name=f"diagnostics_v{version_number}"):

            mlflow.log_metric("diagnostics_avg_latency_ms", avg_latency)
            mlflow.log_metric("diagnostics_extreme_error_count", num_extreme_errors)
            mlflow.log_artifact(residual_plot_path)

            mlflow.set_tag("diagnosed_model_version", version_number)

        # Promotion Logic 

        passed = (
            avg_latency <= MAX_ALLOWED_LATENCY_MS and
            num_extreme_errors <= MAX_ALLOWED_EXTREME_ERRORS
        )

        if passed:
            logger.info("Diagnostics PASSED → Promoting to production")

            try:
                client.delete_registered_model_alias(
                    REGISTERED_MODEL_NAME,
                    PRODUCTION_ALIAS
                )
            except Exception:
                pass

            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME,
                PRODUCTION_ALIAS,
                version=str(version_number)
            )

            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                VALIDATED_ALIAS
            )

            client.set_model_version_tag(
                REGISTERED_MODEL_NAME,
                str(version_number),
                "lifecycle_stage",
                "production"
            )

            logger.info("Model promoted to production.")

        else:
            logger.warning("Diagnostics FAILED.")

            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                VALIDATED_ALIAS
            )

            client.set_model_version_tag(
                REGISTERED_MODEL_NAME,
                str(version_number),
                "lifecycle_stage",
                "rejected_diagnostics"
            )

            logger.warning("Model marked as rejected_diagnostics.")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Diagnostics pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
