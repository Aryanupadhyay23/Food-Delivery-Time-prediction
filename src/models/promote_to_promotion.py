import sys
import logging
import dagshub
import mlflow
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"
EXPERIMENT_NAME = "FoodDeliveryTimePipeline"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_mlflow():
    dagshub.init(
        repo_owner="Aryanupadhyay23",
        repo_name="Food-Delivery-Time-prediction",
        mlflow=True
    )
    mlflow.set_experiment(EXPERIMENT_NAME)


def main():
    try:
        configure_mlflow()
        client = MlflowClient()

        # ======================================================
        # Verify staging alias exists
        # ======================================================

        staging_obj = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            STAGING_ALIAS
        )

        version = str(staging_obj.version)

        logger.info(f"Staging version found: {version}")

        # ======================================================
        # Archive current production (if exists)
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
        # Remove existing production alias
        # ======================================================

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                PRODUCTION_ALIAS
            )
        except Exception:
            pass

        # ======================================================
        # Promote staging → production
        # ======================================================

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            PRODUCTION_ALIAS,
            version=version
        )

        # Remove staging alias (clean transition)
        client.delete_registered_model_alias(
            REGISTERED_MODEL_NAME,
            STAGING_ALIAS
        )

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
