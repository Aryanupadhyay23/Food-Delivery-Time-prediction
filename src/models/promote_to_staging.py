import os
import json
import sys
import logging
from pathlib import Path

import yaml
import mlflow

from mlflow.tracking import MlflowClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"

CANDIDATE_ALIAS = "candidate"
STAGING_ALIAS = "staging"

DAGSHUB_USERNAME = "aryanupadhyay23"

TRACKING_URI = (
    "https://dagshub.com/"
    "aryanupadhyay23/"
    "Food-Delivery-Time-prediction.mlflow"
)


def configure_mlflow():

    logger.info("Configuring MLflow connection")

    token = os.environ.get("DAGSHUB_TOKEN")

    if not token:
        logger.error("DAGSHUB_TOKEN not found in environment")
        raise RuntimeError(
            "DAGSHUB_TOKEN environment variable not set"
        )

    try:
        os.environ[
            "MLFLOW_TRACKING_USERNAME"
        ] = DAGSHUB_USERNAME

        os.environ[
            "MLFLOW_TRACKING_PASSWORD"
        ] = token

        mlflow.set_tracking_uri(
            TRACKING_URI
        )

        logger.info(
            "Connected to DagsHub MLflow successfully"
        )

    except Exception:
        logger.exception(
            "Failed to configure MLflow"
        )
        raise


def load_json(path: Path):

    logger.info(f"Loading JSON file from {path}")

    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(path)

    try:
        with open(path, "r") as file:
            data = json.load(file)

        logger.debug("JSON file loaded successfully")

        return data

    except Exception:
        logger.exception("Failed to load JSON file")
        raise


def load_yaml(path: Path):

    logger.info(f"Loading YAML file from {path}")

    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(path)

    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        logger.debug("YAML file loaded successfully")

        return data

    except Exception:
        logger.exception("Failed to load YAML file")
        raise


def main():

    try:
        configure_mlflow()

        client = MlflowClient()

        root_path = Path(
            __file__
        ).resolve().parents[2]

        params = load_yaml(
            root_path / "params.yaml"
        )

        metrics = load_json(
            root_path
            / "reports"
            / "metrics.json"
        )

        diagnostics = load_json(
            root_path
            / "reports"
            / "diagnostics_metrics.json"
        )

        logger.info("Applying staging governance checks")

        policy = (
            params["promotion_policy"]
            ["staging_thresholds"]
        )

        perf = policy["performance"]
        diag = policy["diagnostics"]

        version = str(
            metrics["model_version"]
        )

        if metrics["test_R2"] < perf["min_test_r2"]:
            logger.error(
                f"R2 threshold failed. "
                f"Actual: {metrics['test_R2']}, "
                f"Required: {perf['min_test_r2']}"
            )
            sys.exit(1)

        if metrics["test_MAE"] > perf["max_test_mae"]:
            logger.error(
                f"MAE threshold failed. "
                f"Actual: {metrics['test_MAE']}, "
                f"Allowed: {perf['max_test_mae']}"
            )
            sys.exit(1)

        if (
            diagnostics["avg_latency_ms"]
            > diag["max_avg_latency_ms"]
        ):
            logger.error(
                f"Latency threshold failed. "
                f"Actual: {diagnostics['avg_latency_ms']}, "
                f"Allowed: {diag['max_avg_latency_ms']}"
            )
            sys.exit(1)

        if (
            diagnostics["extreme_error_count"]
            > diag["max_extreme_errors"]
        ):
            logger.error(
                f"Extreme error threshold failed. "
                f"Actual: {diagnostics['extreme_error_count']}, "
                f"Allowed: {diag['max_extreme_errors']}"
            )
            sys.exit(1)

        logger.info(
            "All staging thresholds passed successfully"
        )

        logger.info(
            "Validating candidate alias version"
        )

        candidate_obj = (
            client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                CANDIDATE_ALIAS
            )
        )

        candidate_version = str(
            candidate_obj.version
        )

        if candidate_version != version:
            logger.error(
                f"Candidate alias mismatch. "
                f"Alias version: {candidate_version}, "
                f"Metrics version: {version}"
            )
            sys.exit(1)

        logger.info(
            f"Candidate version {version} verified"
        )

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                STAGING_ALIAS
            )

            logger.debug(
                "Previous staging alias removed"
            )

        except Exception:
            logger.debug(
                "No previous staging alias found"
            )

        logger.info(
            f"Promoting version {version} to staging"
        )

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            STAGING_ALIAS,
            version=version
        )

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                CANDIDATE_ALIAS
            )

            logger.debug(
                "Candidate alias removed"
            )

        except Exception:
            logger.debug(
                "No candidate alias to remove"
            )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "lifecycle_stage",
            "staging"
        )

        logger.info(
            f"Version {version} promoted to STAGING successfully"
        )

    except Exception:
        logger.exception(
            "Staging promotion failed"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()