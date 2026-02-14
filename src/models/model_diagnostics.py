import json
import time
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

import matplotlib.pyplot as plt
import seaborn as sns
import dagshub


# ======================================================
# Configuration
# ======================================================

TARGET = "time_taken"
REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
CANDIDATE_ALIAS = "candidate"
EXPERIMENT_NAME = "FoodDeliveryTimePipeline"

# Purely measurement threshold (NOT governance)
EXTREME_ERROR_THRESHOLD = 15


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


def plot_residuals(y_true, y_pred, save_dir: Path):

    residuals = y_true - y_pred

    plt.figure(figsize=(14, 6))

    # Actual vs Predicted
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--"
    )
    plt.title("Actual vs Predicted")

    # Residual Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, bins=30)
    plt.axvline(x=0, color="r", linestyle="--")
    plt.title("Residual Distribution")

    save_path = save_dir / "residual_analysis.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def test_latency(model, sample_input, iterations=500):

    # Warm-up call
    model.predict(sample_input)

    start = time.time()
    for _ in range(iterations):
        model.predict(sample_input)
    end = time.time()

    return ((end - start) / iterations) * 1000


# ======================================================
# Main
# ======================================================

def main():

    try:
        configure_mlflow()
        client = MlflowClient()

        root_path = Path(__file__).parent.parent.parent
        test_path = root_path / "data" / "processed" / "test.csv"

        reports_dir = root_path / "reports"
        diagnostics_dir = reports_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        diagnostics_metrics_path = reports_dir / "diagnostics_metrics.json"

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
        logger.info(f"Running diagnostics on candidate version {version_number}")

        model = mlflow.pyfunc.load_model(
            f"models:/{REGISTERED_MODEL_NAME}@{CANDIDATE_ALIAS}"
        )

        # ======================================================
        # Load Data
        # ======================================================

        df_test = load_data(test_path)

        X = df_test.drop(columns=[TARGET])
        y = df_test[TARGET]

        y_pred = model.predict(X)

        # ======================================================
        # Compute Diagnostics
        # ======================================================

        residual_plot_path = plot_residuals(y, y_pred, diagnostics_dir)

        sample_row = X.iloc[[0]]
        avg_latency = test_latency(model, sample_row)

        residuals = np.abs(y - y_pred)
        num_extreme_errors = int(
            np.sum(residuals > EXTREME_ERROR_THRESHOLD)
        )

        diagnostics_metrics = {
            "model_version": version_number,
            "avg_latency_ms": round(avg_latency, 4),
            "extreme_error_count": num_extreme_errors,
            "extreme_error_threshold": EXTREME_ERROR_THRESHOLD
        }

        with open(diagnostics_metrics_path, "w") as f:
            json.dump(diagnostics_metrics, f, indent=4)

        logger.info(f"Diagnostics metrics saved at {diagnostics_metrics_path}")

        # ======================================================
        # Log to MLflow
        # ======================================================

        with mlflow.start_run(run_name=f"diagnostics_v{version_number}"):

            mlflow.log_metric("diagnostics_avg_latency_ms", avg_latency)
            mlflow.log_metric("diagnostics_extreme_error_count", num_extreme_errors)
            mlflow.log_metric("diagnostics_extreme_error_threshold", EXTREME_ERROR_THRESHOLD)

            mlflow.log_artifact(residual_plot_path)

            mlflow.set_tag("diagnosed_model_version", version_number)
            mlflow.set_tag("diagnosed_alias", CANDIDATE_ALIAS)

        logger.info("Diagnostics stage completed successfully.")

    except Exception as e:
        logger.exception(f"Diagnostics pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
