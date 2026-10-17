import sys
import logging
import mlflow
from mlflow.tracking import MlflowClient

try:
    from backend.src.models.mlflow_utils import (
        configure_mlflow,
        start_pipeline_stage_run,
        end_pipeline_run
    )
except ImportError:
    from mlflow_utils import (
        configure_mlflow,
        start_pipeline_stage_run,
        end_pipeline_run
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"


def main():
    try:
        configure_mlflow()
        client = MlflowClient()

        logger.info("Fetching staging model alias from MLflow registry")
        staging_obj = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            STAGING_ALIAS
        )
        version = str(staging_obj.version)
        logger.info(f"Staging version detected: {version}")

        old_version = None
        try:
            logger.info("Checking existing production model version")
            prod_obj = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                PRODUCTION_ALIAS
            )
            old_version = str(prod_obj.version)
            client.set_model_version_tag(
                REGISTERED_MODEL_NAME,
                old_version,
                "lifecycle_stage",
                "archived"
            )
            logger.info(f"Archived previous production version {old_version}")
        except Exception:
            logger.warning("No previous production version found in registry")

        # Start Stage 05 Production Governance nested run
        with start_pipeline_stage_run(
            stage_name="Model_Governance_Production",
            stage_number="05",
            tags={
                "promoted_version": version,
                "archived_version": old_version or "none",
                "target_stage": "production",
                "governance_decision": "DEPLOYED_TO_PRODUCTION"
            }
        ) as prod_run:
            logger.info(f"Logging Production Governance Gate run (ID: {prod_run.info.run_id})")

            mlflow.log_param("promoted_version", version)
            mlflow.log_param("archived_previous_version", old_version or "none")
            mlflow.log_metric("production_deployment_gate", 1.0)
            mlflow.set_tag("deployment_status", "SUCCESS")

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                PRODUCTION_ALIAS
            )
            logger.debug("Previous production alias cleared")
        except Exception:
            pass

        logger.info(f"Promoting version {version} to PRODUCTION")
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
            logger.debug("Staging alias cleared")
        except Exception:
            pass

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "lifecycle_stage",
            "production"
        )

        logger.info(f"Version {version} promoted to PRODUCTION successfully")

        # Conclude the top-level parent pipeline run
        end_pipeline_run(status="FINISHED")

    except Exception:
        logger.exception("Production promotion failed")
        end_pipeline_run(status="FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()