import os
import sys
import logging
import mlflow

from mlflow.tracking import MlflowClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"

STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"

EXPERIMENT_NAME = "FoodDeliveryTimePipeline"

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

        mlflow.set_experiment(
            EXPERIMENT_NAME
        )

        logger.info(
            "Connected to DagsHub MLflow successfully"
        )

    except Exception:
        logger.exception(
            "Failed to configure MLflow"
        )
        raise


def main():

    try:
        configure_mlflow()

        client = MlflowClient()

        logger.info(
            "Fetching staging model alias"
        )

        staging_obj = (
            client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                STAGING_ALIAS
            )
        )

        version = str(
            staging_obj.version
        )

        logger.info(
            f"Staging version detected: {version}"
        )

        try:
            logger.info(
                "Checking existing production model"
            )

            prod_obj = (
                client.get_model_version_by_alias(
                    REGISTERED_MODEL_NAME,
                    PRODUCTION_ALIAS
                )
            )

            old_version = str(
                prod_obj.version
            )

            client.set_model_version_tag(
                REGISTERED_MODEL_NAME,
                old_version,
                "lifecycle_stage",
                "archived"
            )

            logger.info(
                f"Archived production version {old_version}"
            )

        except Exception:
            logger.warning(
                "No previous production version found"
            )

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                PRODUCTION_ALIAS
            )

            logger.debug(
                "Old production alias removed"
            )

        except Exception:
            logger.debug(
                "No production alias to remove"
            )

        logger.info(
            f"Promoting version {version} to production"
        )

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            PRODUCTION_ALIAS,
            version=version
        )

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                STAGING_ALIAS
            )

            logger.debug(
                "Staging alias removed"
            )

        except Exception:
            logger.debug(
                "No staging alias to remove"
            )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "lifecycle_stage",
            "production"
        )

        logger.info(
            f"Version {version} promoted to PRODUCTION successfully"
        )

    except Exception:
        logger.exception(
            "Production promotion failed"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()