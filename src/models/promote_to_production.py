import os
import sys
import logging
import mlflow
from mlflow.tracking import MlflowClient


# Hardcoded configuration values
REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"
EXPERIMENT_NAME = "FoodDeliveryTimePipeline"

# DagsHub MLflow configuration
DAGSHUB_USERNAME = "aryanupadhyay23"
TRACKING_URI = "https://dagshub.com/aryanupadhyay23/Food-Delivery-Time-prediction.mlflow"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_mlflow():
    # Read token from environment
    token = os.environ.get("DAGSHUB_TOKEN")

    # Fail if token missing
    if not token:
        raise RuntimeError("DAGSHUB_TOKEN environment variable not set.")

    # Set MLflow authentication credentials
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # Set MLflow tracking server
    mlflow.set_tracking_uri(TRACKING_URI)

    # Explicitly set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Connected to DagsHub MLflow using token authentication.")


def main():
    try:
        # Configure MLflow connection
        configure_mlflow()

        # Create MLflow client
        client = MlflowClient()

        # Fetch staging model version
        staging_obj = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            STAGING_ALIAS
        )

        version = str(staging_obj.version)
        logger.info(f"Staging version found: {version}")

        # Archive existing production model if present
        try:
            current_prod = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                PRODUCTION_ALIAS
            )

            old_version = str(current_prod.version)

            client.set_model_version_tag(
                REGISTERED_MODEL_NAME,
                old_version,
                "lifecycle_stage",
                "archived"
            )

            logger.info(f"Archived previous production version {old_version}")

        except Exception:
            logger.info("No existing production version found.")

        # Remove current production alias
        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                PRODUCTION_ALIAS
            )
        except Exception:
            pass

        # Assign production alias to staging version
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            PRODUCTION_ALIAS,
            version=version
        )

        # Remove staging alias after promotion
        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                STAGING_ALIAS
            )
        except Exception:
            pass

        # Update lifecycle tag to production
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "lifecycle_stage",
            "production"
        )

        logger.info(f"Model version {version} promoted to PRODUCTION cleanly.")

    except Exception as e:
        logger.exception(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
