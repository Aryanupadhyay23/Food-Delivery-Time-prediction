import time
import logging
import sys
from pathlib import Path

import pandas as pd
import joblib
import yaml
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

try:
    from backend.src.models.mlflow_utils import (
        configure_mlflow,
        start_pipeline_stage_run,
        clean_and_prefix,
        get_git_commit,
        get_pipeline_run_id
    )
except ImportError:
    from mlflow_utils import (
        configure_mlflow,
        start_pipeline_stage_run,
        clean_and_prefix,
        get_git_commit,
        get_pipeline_run_id
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


TARGET = "time_taken"
REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
CANDIDATE_ALIAS = "candidate"


def load_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading dataset from {path}")
    try:
        df = pd.read_csv(path, engine="pyarrow")
        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        return df
    except Exception:
        logger.exception("Failed to load dataset")
        raise


def load_params(path: Path) -> dict:
    logger.info(f"Loading parameters from {path}")
    try:
        with open(path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded successfully")
        return params
    except Exception:
        logger.exception("Failed to load parameters")
        raise


def load_preprocessor(path: Path):
    logger.info(f"Loading preprocessor from {path}")
    try:
        preprocessor = joblib.load(path)
        logger.info("Preprocessor loaded successfully")
        return preprocessor
    except Exception:
        logger.exception("Failed to load preprocessor")
        raise


def build_model(params: dict, preprocessor):
    logger.info("Building model pipeline")
    try:
        lgb_params = params["model_training"]["LightGBM_Regressor"]
        rf_params = params["model_training"]["RandomForest_Regressor"]
        stack_params = params["model_training"]["Stacking_Regressor"]
        meta_params = params["model_training"]["Meta_Model"]

        lgb_model = LGBMRegressor(**lgb_params)
        rf_model = RandomForestRegressor(**rf_params)
        meta_model = DecisionTreeRegressor(**meta_params)

        stacking_model = StackingRegressor(
            estimators=[
                ("lightgbm", lgb_model),
                ("random_forest", rf_model)
            ],
            final_estimator=meta_model,
            cv=stack_params["cv"],
            n_jobs=stack_params["n_jobs"],
            passthrough=stack_params["passthrough"]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", stacking_model)
            ]
        )

        final_model = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=PowerTransformer(),
            check_inverse=False
        )

        logger.info("Model pipeline built successfully")
        return (
            final_model,
            lgb_params,
            rf_params,
            stack_params,
            meta_params
        )
    except Exception:
        logger.exception("Failed to build model pipeline")
        raise


def main():
    try:
        configure_mlflow()

        root_path = Path(__file__).resolve().parents[2]
        train_path = root_path / "data" / "processed" / "train.csv"
        preprocessor_path = root_path / "artifacts" / "preprocessor.pkl"
        params_path = root_path / "params.yaml"
        model_dir = root_path / "models"

        params = load_params(params_path)
        train_df = load_data(train_path)

        if TARGET not in train_df.columns:
            logger.error(f"Target column '{TARGET}' not found")
            raise ValueError(f"{TARGET} column not found")

        X_train = train_df.drop(columns=[TARGET])
        y_train = train_df[TARGET]

        logger.info(f"Training data prepared with shape {X_train.shape}")
        preprocessor = load_preprocessor(preprocessor_path)

        (
            final_model,
            lgb_params,
            rf_params,
            stack_params,
            meta_params
        ) = build_model(params, preprocessor)

        stage_run_id = None
        model_version = None

        # Execute Stage 01 under hierarchical parent pipeline run
        with start_pipeline_stage_run(
            stage_name="Model_Training",
            stage_number="01",
            tags={
                "model.type": "StackingRegressor",
                "model.architecture": "LightGBM + RandomForest -> DecisionTree",
                "data.target": TARGET
            }
        ) as train_run:
            stage_run_id = train_run.info.run_id
            logger.info(f"Started Stage 01 run (Run ID: {stage_run_id})")

            # ------------------------------------------------------------------
            # Decomposed Nested Sub-runs for Base & Meta Estimators
            # ------------------------------------------------------------------
            logger.info("Logging nested sub-run for Base Estimator: LightGBM")
            with mlflow.start_run(run_name="01a_Base_Estimator_LightGBM", nested=True):
                mlflow.set_tag("estimator_role", "base_estimator")
                mlflow.set_tag("estimator_name", "LightGBM")
                mlflow.set_tag("library", "lightgbm")
                mlflow.log_params(clean_and_prefix(lgb_params, "lgb"))

            logger.info("Logging nested sub-run for Base Estimator: RandomForest")
            with mlflow.start_run(run_name="01b_Base_Estimator_RandomForest", nested=True):
                mlflow.set_tag("estimator_role", "base_estimator")
                mlflow.set_tag("estimator_name", "RandomForest")
                mlflow.set_tag("library", "scikit-learn")
                mlflow.log_params(clean_and_prefix(rf_params, "rf"))

            logger.info("Logging nested sub-run for Meta Estimator: DecisionTree")
            with mlflow.start_run(run_name="01c_Meta_Estimator_DecisionTree", nested=True):
                mlflow.set_tag("estimator_role", "meta_estimator")
                mlflow.set_tag("estimator_name", "DecisionTree")
                mlflow.set_tag("library", "scikit-learn")
                mlflow.log_params(clean_and_prefix(meta_params, "meta"))

            # ------------------------------------------------------------------
            # Stage 01 Overall Parameters & Metadata
            # ------------------------------------------------------------------
            mlflow.log_params(clean_and_prefix(stack_params, "stack"))
            mlflow.log_param("dataset_rows", train_df.shape[0])
            mlflow.log_param("dataset_columns", train_df.shape[1])
            mlflow.log_param("features_count", X_train.shape[1])
            mlflow.log_param("target_name", TARGET)
            mlflow.log_param("git_commit", get_git_commit())

            # ------------------------------------------------------------------
            # Model Fitting with Latency Profiling
            # ------------------------------------------------------------------
            logger.info("Fitting StackingRegressor pipeline...")
            start_time = time.time()
            final_model.fit(X_train, y_train)
            duration_sec = time.time() - start_time
            logger.info(f"Model fitting completed in {duration_sec:.2f} seconds")
            mlflow.log_metric("training_duration_seconds", round(duration_sec, 4))

            # ------------------------------------------------------------------
            # Local Model Persistence
            # ------------------------------------------------------------------
            model_dir.mkdir(parents=True, exist_ok=True)
            local_model_path = model_dir / "stacking_lgb_rf_pipeline.joblib"
            joblib.dump(final_model, local_model_path)
            logger.info(f"Model saved locally at {local_model_path}")

            # ------------------------------------------------------------------
            # Model Signature & MLflow Registry Registration
            # ------------------------------------------------------------------
            sample_input = X_train.head(5)
            sample_pred = final_model.predict(sample_input)
            signature = infer_signature(sample_input, sample_pred)

            logger.info("Logging model to MLflow with input schema signature and example")
            model_info = mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                input_example=sample_input,
                registered_model_name=REGISTERED_MODEL_NAME
            )

            model_version = str(model_info.registered_model_version)
            mlflow.log_param("registered_model_version", model_version)
            mlflow.set_tag("lifecycle_stage", "candidate")

        # ----------------------------------------------------------------------
        # Model Registry Aliasing & Lineage Tagging
        # ----------------------------------------------------------------------
        client = MlflowClient()
        try:
            client.delete_registered_model_alias(REGISTERED_MODEL_NAME, CANDIDATE_ALIAS)
            logger.debug("Existing candidate alias removed")
        except Exception:
            logger.debug("No previous candidate alias found")

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            CANDIDATE_ALIAS,
            version=model_version
        )
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            model_version,
            "training_run_id",
            stage_run_id
        )

        parent_id = get_pipeline_run_id()
        if parent_id:
            client.set_model_version_tag(
                REGISTERED_MODEL_NAME,
                model_version,
                "pipeline_run_id",
                parent_id
            )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            model_version,
            "model_architecture",
            "StackingRegressor(LightGBM + RandomForest)"
        )

        logger.info(
            f"Model version {model_version} registered and assigned as candidate (Run: {stage_run_id})"
        )
        logger.info("Stage 01: Model Training completed successfully")

    except MlflowException:
        logger.exception("MLflow operation failed")
        sys.exit(1)
    except Exception:
        logger.exception("Model training pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()