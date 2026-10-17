import os
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
try:
    from dotenv import load_dotenv
    _repo_root = Path(__file__).resolve().parents[3]
    if (_repo_root / ".env.fd").exists():
        load_dotenv(_repo_root / ".env.fd")
    elif (_repo_root / ".env").exists():
        load_dotenv(_repo_root / ".env")
except ImportError:
    pass

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

DAGSHUB_REPO_OWNER = "Aryanupadhyay23"
DAGSHUB_REPO_NAME = "Food-Delivery-Time-prediction"
EXPERIMENT_NAME = "FoodDeliveryTimePipeline"

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
PIPELINE_RUN_FILE = REPORTS_DIR / "active_pipeline_run.json"


def configure_mlflow():
    """
    Configures centralized MLflow tracking purely using tokens (no browser authentication).
    Uses DAGSHUB_TOKEN / DAGSHUB_USER_TOKEN from .env/.env.fd locally or GitHub Secrets in CI.
    """
    token = os.environ.get("DAGSHUB_TOKEN") or os.environ.get("DAGSHUB_USER_TOKEN")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME", DAGSHUB_REPO_OWNER)

    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking configured purely via token: '{tracking_uri}'")


def get_git_commit() -> str:
    """Retrieves current git commit hash if available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def clean_and_prefix(params: dict, prefix: str) -> dict:
    """Sanitizes parameter dictionary and prefixes keys for clean MLflow tracking."""
    cleaned = {}
    for key, value in params.items():
        if value is None:
            cleaned[f"{prefix}_{key}"] = "None"
        elif isinstance(value, (int, float, str, bool)):
            cleaned[f"{prefix}_{key}"] = value
        else:
            cleaned[f"{prefix}_{key}"] = str(value)
    return cleaned


def get_or_create_pipeline_run(run_name: str = None, tags: dict = None) -> str:
    """
    Retrieves the active root pipeline run ID, or starts a new top-level
    parent pipeline run in MLflow if one is not currently running.
    Persists the parent run ID to disk to coordinate downstream DVC stages.
    """
    configure_mlflow()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    if PIPELINE_RUN_FILE.exists():
        try:
            with open(PIPELINE_RUN_FILE, "r") as f:
                data = json.load(f)
            parent_id = data.get("pipeline_run_id")
            if parent_id:
                run = client.get_run(parent_id)
                if run.info.status == "RUNNING":
                    logger.info(f"Reusing active parent pipeline run: {parent_id}")
                    return parent_id
        except Exception as e:
            logger.debug(f"Could not reuse existing parent run ({e}). Initializing new parent run.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_name = run_name or f"Pipeline_Execution_{timestamp}"

    pipeline_tags = {
        "pipeline.name": "Food-Delivery-Time-Prediction-MLOps",
        "pipeline.type": "End-to-End-Pipeline",
        "pipeline.architecture": "StackingRegressor(LightGBM + RandomForest)",
        "pipeline.status": "RUNNING",
        "git.commit": get_git_commit(),
        "created_at": datetime.now().isoformat()
    }
    if tags:
        pipeline_tags.update(tags)

    parent_run = client.create_run(
        experiment_id=experiment_id,
        tags=pipeline_tags,
        run_name=parent_name
    )
    parent_id = parent_run.info.run_id

    with open(PIPELINE_RUN_FILE, "w") as f:
        json.dump({
            "pipeline_run_id": parent_id,
            "pipeline_name": parent_name,
            "created_at": datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"Initialized top-level parent pipeline run '{parent_name}' (ID: {parent_id})")
    return parent_id


def start_pipeline_stage_run(stage_name: str, stage_number: str = "", tags: dict = None):
    """
    Starts an MLflow run representing a pipeline stage, automatically linked to
    the active parent pipeline run via 'mlflow.parentRunId'.
    """
    parent_id = get_or_create_pipeline_run()
    run_tags = {
        "mlflow.parentRunId": parent_id,
        "pipeline.stage": stage_name,
        "pipeline_run_id": parent_id,
        "git.commit": get_git_commit()
    }
    if stage_number:
        run_tags["pipeline.stage_order"] = stage_number
    if tags:
        run_tags.update(tags)

    full_run_name = f"{stage_number}_{stage_name}" if stage_number else stage_name
    return mlflow.start_run(run_name=full_run_name, tags=run_tags)


def get_pipeline_run_id() -> str | None:
    """Retrieves active parent pipeline run ID from file without creating a new one."""
    if PIPELINE_RUN_FILE.exists():
        try:
            with open(PIPELINE_RUN_FILE, "r") as f:
                data = json.load(f)
            return data.get("pipeline_run_id")
        except Exception:
            return None
    return None


def end_pipeline_run(status: str = "FINISHED"):
    """Terminates the active root pipeline run with given completion status."""
    parent_id = get_pipeline_run_id()
    if parent_id:
        try:
            client = MlflowClient()
            client.set_terminated(parent_id, status=status)
            client.set_tag(parent_id, "pipeline.status", status)
            client.set_tag(parent_id, "completed_at", datetime.now().isoformat())
            logger.info(f"Top-level pipeline run {parent_id} finished with status '{status}'")
            if PIPELINE_RUN_FILE.exists():
                PIPELINE_RUN_FILE.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not mark parent pipeline run as finished: {e}")

