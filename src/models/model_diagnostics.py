import json
import time
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import mlflow
import dagshub

from mlflow.tracking import MlflowClient

import matplotlib.pyplot as plt
import seaborn as sns


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


def load_threshold(params_path: Path) -> float:

    logger.info(f"Loading threshold values from {params_path}")

    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        threshold = (
            params["promotion_policy"]
            ["staging_thresholds"]
            ["diagnostics"]
            ["extreme_error_threshold"]
        )

        logger.info(
            f"Extreme error threshold loaded: {threshold}"
        )

        return threshold

    except Exception:
        logger.exception("Failed to load threshold values")
        raise


def plot_residuals(
    y_true,
    y_pred,
    save_dir: Path
):

    logger.info("Generating residual diagnostic plot")

    try:
        residuals = y_true - y_pred

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)

        sns.scatterplot(
            x=y_true,
            y=y_pred,
            alpha=0.35
        )

        plt.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--"
        )

        plt.title("Actual vs Predicted")

        plt.subplot(1, 2, 2)

        sns.histplot(
            residuals,
            kde=True,
            bins=30
        )

        plt.axvline(
            x=0,
            color="r",
            linestyle="--"
        )

        plt.title("Residual Distribution")

        save_path = (
            save_dir /
            "residual_analysis.png"
        )

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Residual plot saved at {save_path}")

        return save_path

    except Exception:
        logger.exception("Failed to generate residual plot")
        raise


def test_latency(
    model,
    sample_input,
    iterations=500
):

    logger.info(
        f"Running latency test for {iterations} iterations"
    )

    try:
        model.predict(sample_input)

        start = time.time()

        for _ in range(iterations):
            model.predict(sample_input)

        end = time.time()

        avg_ms = (
            (end - start) / iterations
        ) * 1000

        logger.info(
            f"Average prediction latency: {avg_ms:.4f} ms"
        )

        return avg_ms

    except Exception:
        logger.exception("Latency test failed")
        raise


def main():

    try:
        configure_environment()

        client = MlflowClient()

        root_path = Path(__file__).parent.parent.parent

        test_path = (
            root_path
            / "data"
            / "processed"
            / "test.csv"
        )

        params_path = root_path / "params.yaml"

        reports_dir = root_path / "reports"

        diagnostics_dir = (
            reports_dir
            / "diagnostics"
        )

        diagnostics_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        metrics_path = (
            reports_dir
            / "diagnostics_metrics.json"
        )

        threshold = load_threshold(params_path)

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
            f"Running diagnostics for model version {version_number}"
        )

        model = mlflow.pyfunc.load_model(
            f"models:/"
            f"{REGISTERED_MODEL_NAME}"
            f"@{CANDIDATE_ALIAS}"
        )

        test_df = load_data(test_path)

        X_test = test_df.drop(columns=[TARGET])
        y_test = test_df[TARGET]

        logger.info("Generating predictions")

        y_pred = model.predict(X_test)

        residual_plot_path = plot_residuals(
            y_test,
            y_pred,
            diagnostics_dir
        )

        sample_row = X_test.iloc[[0]]

        avg_latency = test_latency(
            model,
            sample_row
        )

        residual_errors = np.abs(
            y_test - y_pred
        )

        extreme_error_count = int(
            np.sum(
                residual_errors > threshold
            )
        )

        logger.info(
            f"Extreme error count: {extreme_error_count}"
        )

        diagnostics_metrics = {
            "model_version": version_number,
            "avg_latency_ms": round(avg_latency, 4),
            "extreme_error_count": extreme_error_count,
            "extreme_error_threshold": threshold
        }

        with open(metrics_path, "w") as file:
            json.dump(
                diagnostics_metrics,
                file,
                indent=4
            )

        logger.info(
            f"Diagnostics metrics saved at {metrics_path}"
        )

        with mlflow.start_run(
            run_name=f"diagnostics_v{version_number}"
        ):

            logger.info(
                "Logging diagnostics metrics to MLflow"
            )

            mlflow.log_metric(
                "diagnostics_avg_latency_ms",
                avg_latency
            )

            mlflow.log_metric(
                "diagnostics_extreme_error_count",
                extreme_error_count
            )

            mlflow.log_metric(
                "diagnostics_extreme_error_threshold",
                threshold
            )

            mlflow.log_artifact(
                residual_plot_path
            )

            mlflow.set_tag(
                "diagnosed_model_version",
                version_number
            )

            mlflow.set_tag(
                "diagnosed_alias",
                CANDIDATE_ALIAS
            )

        logger.info(
            "Model diagnostics completed successfully"
        )

    except Exception:
        logger.exception(
            "Model diagnostics pipeline failed"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()