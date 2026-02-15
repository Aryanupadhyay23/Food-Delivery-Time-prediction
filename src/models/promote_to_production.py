import os
import sys
import logging
import mlflow
from mlflow.tracking import MlflowClient


# ======================================================
# Configuration (Hardcoded Safe Values)
# ======================================================

REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"

DAGSHUB_USERNAME = "aryanupadhyay23"
TRACKING_URI = "https://dagshub.com/aryanupadhyay23/Food-Delivery-Time-prediction.mlflow"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================
# MLflow Configuration (Token-Based)
# ======================================================

def configure_mlflow():
    token = os.environ.get("DAGSHUB_TOKEN")

    if not token:
        raise RuntimeError("DAGSHUB_TOKEN environment variable not set.")

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(TRACKING_URI)

    logger.info("Connected to DagsHub MLflow using token authentication.")


# ======================================================
# Main Promotion Logic
# ======================================================

def main():

    try:
        configure_mlflow()
        client = MlflowClient()

        # ======================================================
        # Verify Staging Alias Exists
        # ======================================================

        staging_obj = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            STAGING_ALIAS
        )

        version = str(staging_obj.version)
        logger.info(f"Staging version found: {version}")

        # ======================================================
        # Archive Current Production (If Exists)
        # ======================================================

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

        # ======================================================
        # Remove Existing Production Alias
        # ======================================================

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                PRODUCTION_ALIAS
            )
        except Exception:
            pass

        # ======================================================
        # Promote Staging → Production
        # ======================================================

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            PRODUCTION_ALIAS,
            version=version
        )

        # Remove staging alias (clean transition)
        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                STAGING_ALIAS
            )
        except Exception:
            pass

        # Update lifecycle tag
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
