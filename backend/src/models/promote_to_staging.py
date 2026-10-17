import json
import sys
import logging
from pathlib import Path

import yaml
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
CANDIDATE_ALIAS = "candidate"
STAGING_ALIAS = "staging"


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

        root_path = Path(__file__).resolve().parents[2]
        params = load_yaml(root_path / "params.yaml")
        metrics = load_json(root_path / "reports" / "metrics.json")
        diagnostics = load_json(root_path / "reports" / "diagnostics_metrics.json")

        logger.info("Applying staging governance gate policies")

        policy = params["promotion_policy"]["staging_thresholds"]
        perf = policy["performance"]
        diag = policy["diagnostics"]

        try:
            candidate_obj = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                CANDIDATE_ALIAS
            )
            version = str(candidate_obj.version)
            logger.info(f"Active candidate version from MLflow Registry: {version}")
        except Exception:
            version = str(metrics.get("model_version", "unknown"))
            logger.warning(
                f"Candidate alias not yet found in registry, using metrics version: {version}"
            )

        # Track failure reasons if any
        reasons = []
        if metrics["test_R2"] < perf["min_test_r2"]:
            reasons.append(
                f"R2 failed: actual {metrics['test_R2']} < min {perf['min_test_r2']}"
            )
        if metrics["test_MAE"] > perf["max_test_mae"]:
            reasons.append(
                f"MAE failed: actual {metrics['test_MAE']} > max {perf['max_test_mae']}"
            )
        if diagnostics["avg_latency_ms"] > diag["max_avg_latency_ms"]:
            reasons.append(
                f"Latency failed: actual {diagnostics['avg_latency_ms']}ms > max {diag['max_avg_latency_ms']}ms"
            )
        if diagnostics["extreme_error_count"] > diag["max_extreme_errors"]:
            reasons.append(
                f"Extreme errors failed: actual {diagnostics['extreme_error_count']} > max {diag['max_extreme_errors']}"
            )

        passed = len(reasons) == 0

        # Execute Stage 04 Governance run under parent pipeline
        with start_pipeline_stage_run(
            stage_name="Model_Governance_Staging",
            stage_number="04",
            tags={
                "evaluated_version": version,
                "target_stage": "staging",
                "governance_decision": "PASSED" if passed else "REJECTED"
            }
        ) as gov_run:
            logger.info(f"Logging Staging Governance Gate run (ID: {gov_run.info.run_id})")

            mlflow.log_params({
                "policy_min_test_r2": perf["min_test_r2"],
                "policy_max_test_mae": perf["max_test_mae"],
                "policy_max_avg_latency_ms": diag["max_avg_latency_ms"],
                "policy_max_extreme_errors": diag["max_extreme_errors"],
                "candidate_version": version
            })

            mlflow.log_metrics({
                "actual_test_r2": metrics["test_R2"],
                "actual_test_mae": metrics["test_MAE"],
                "actual_avg_latency_ms": diagnostics["avg_latency_ms"],
                "actual_extreme_error_count": float(diagnostics["extreme_error_count"]),
                "staging_governance_passed": 1.0 if passed else 0.0
            })

            if not passed:
                failure_msg = "; ".join(reasons)
                logger.error(f"Staging governance failed: {failure_msg}")
                mlflow.set_tag("rejection_reasons", failure_msg)
                end_pipeline_run("FAILED")
                sys.exit(1)

            mlflow.set_tag("governance_status", "PROMOTED_TO_STAGING")

        logger.info("All staging thresholds passed successfully")
        logger.info(f"Validating candidate alias in MLflow registry (version: {version})")

        candidate_obj = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            CANDIDATE_ALIAS
        )
        version = str(candidate_obj.version)

        logger.info(f"Promoting candidate version {version} to STAGING")
        try:
            client.delete_registered_model_alias(REGISTERED_MODEL_NAME, STAGING_ALIAS)
            logger.debug("Previous staging alias removed")
        except Exception:
            logger.debug("No previous staging alias found")

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            STAGING_ALIAS,
            version=version
        )

        try:
            client.delete_registered_model_alias(REGISTERED_MODEL_NAME, CANDIDATE_ALIAS)
            logger.debug("Candidate alias cleared")
        except Exception:
            pass

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "lifecycle_stage",
            "staging"
        )

        logger.info(f"Version {version} promoted to STAGING successfully")

    except Exception:
        logger.exception("Staging promotion pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()