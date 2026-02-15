import os
import json
import sys
import logging
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
CANDIDATE_ALIAS = "candidate"
STAGING_ALIAS = "staging"

DAGSHUB_USERNAME = "aryanupadhyay23"
TRACKING_URI = "https://dagshub.com/aryanupadhyay23/Food-Delivery-Time-prediction.mlflow"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow Configuration 

def configure_mlflow():
    token = os.environ.get("DAGSHUB_TOKEN")

    if not token:
        raise RuntimeError("DAGSHUB_TOKEN environment variable not set.")

    # Set MLflow credentials
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(TRACKING_URI)

    logger.info("Connected to DagsHub MLflow using token authentication.")

# Utilities

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "r") as f:
        return json.load(f)


def load_params(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Main Logic

def main():

    try:
        configure_mlflow()
        client = MlflowClient()

        root_path = Path(__file__).parent.parent

        params = load_params(root_path / "params.yaml")
        metrics = load_json(root_path / "reports" / "metrics.json")
        diagnostics = load_json(root_path / "reports" / "diagnostics_metrics.json")

        policy = params["promotion_policy"]["staging_thresholds"]

        perf = policy["performance"]
        diag = policy["diagnostics"]

        version = str(metrics["model_version"])

        # Apply Governance Thresholds
       
        if metrics["test_R2"] < perf["min_test_r2"]:
            logger.error("R2 threshold failed.")
            sys.exit(1)

        if metrics["test_MAE"] > perf["max_test_mae"]:
            logger.error("MAE threshold failed.")
            sys.exit(1)

        if diagnostics["avg_latency_ms"] > diag["max_avg_latency_ms"]:
            logger.error("Latency threshold failed.")
            sys.exit(1)

        if diagnostics["extreme_error_count"] > diag["max_extreme_errors"]:
            logger.error("Extreme error threshold failed.")
            sys.exit(1)

        logger.info("All staging thresholds passed.")

        # Verify Candidate Alias

        candidate_obj = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            CANDIDATE_ALIAS
        )

        if str(candidate_obj.version) != version:
            logger.error("Candidate alias mismatch.")
            sys.exit(1)

        # Promote to Staging        

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                STAGING_ALIAS
            )
        except Exception:
            pass

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            STAGING_ALIAS,
            version=version
        )

        # Remove candidate alias (clean transition)
        client.delete_registered_model_alias(
            REGISTERED_MODEL_NAME,
            CANDIDATE_ALIAS
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "lifecycle_stage",
            "staging"
        )

        logger.info(f"Version {version} promoted to STAGING cleanly.")

    except Exception as e:
        logger.exception(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
