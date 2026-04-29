import logging
import sys
import subprocess
from pathlib import Path

import dagshub
import pandas as pd
import joblib
import yaml
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


TARGET = "time_taken"

REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
CANDIDATE_ALIAS = "candidate"
EXPERIMENT_NAME = "FoodDeliveryTimePipeline"


def configure_environment():

    try:
        dagshub.init(
            repo_owner="Aryanupadhyay23",
            repo_name="Food-Delivery-Time-prediction",
            mlflow=True
        )

        mlflow.set_experiment(EXPERIMENT_NAME)

        logger.info("MLflow environment configured successfully")

    except Exception:
        logger.exception("Failed to configure MLflow environment")
        sys.exit(1)


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


def clean_and_prefix(params_dict: dict, prefix: str) -> dict:

    return {
        f"{prefix}_{k}": (
            str(v) if v is None else v
        )
        for k, v in params_dict.items()
    }


def get_git_commit():

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()

        logger.debug(f"Git commit detected: {commit}")

        return commit

    except Exception:
        logger.warning("Unable to fetch git commit hash")
        return "unknown"


def build_model(params: dict, preprocessor):

    logger.info("Building model pipeline")

    try:
        cat_params = params["model_training"]["CatBoost_Regressor"]
        rf_params = params["model_training"]["RandomForest_Regressor"]
        stack_params = params["model_training"]["Stacking_Regressor"]
        meta_params = params["model_training"]["Meta_Model"]

        cat_model = CatBoostRegressor(**cat_params)

        rf_model = RandomForestRegressor(**rf_params)

        meta_model = DecisionTreeRegressor(**meta_params)

        stacking_model = StackingRegressor(
            estimators=[
                ("catboost", cat_model),
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
            cat_params,
            rf_params,
            stack_params,
            meta_params
        )

    except Exception:
        logger.exception("Failed to build model pipeline")
        raise


def main():

    try:
        configure_environment()

        root_path = Path(__file__).parent.parent.parent

        train_path = (
            root_path
            / "data"
            / "processed"
            / "train.csv"
        )

        preprocessor_path = (
            root_path
            / "artifacts"
            / "preprocessor.pkl"
        )

        params_path = root_path / "params.yaml"

        model_dir = root_path / "models"

        params = load_params(params_path)

        train_df = load_data(train_path)

        if TARGET not in train_df.columns:
            logger.error(
                f"Target column '{TARGET}' not found"
            )
            raise ValueError(
                f"{TARGET} column not found"
            )

        X_train = train_df.drop(columns=[TARGET])
        y_train = train_df[TARGET]

        logger.info(
            f"Training data prepared with shape {X_train.shape}"
        )

        preprocessor = load_preprocessor(preprocessor_path)

        (
            final_model,
            cat_params,
            rf_params,
            stack_params,
            meta_params
        ) = build_model(params, preprocessor)

        client = MlflowClient()

        with mlflow.start_run(
            run_name="model_training"
        ) as run:

            logger.info("Logging training parameters")

            mlflow.log_params(
                clean_and_prefix(cat_params, "cat")
            )

            mlflow.log_params(
                clean_and_prefix(rf_params, "rf")
            )

            mlflow.log_params(
                clean_and_prefix(stack_params, "stack")
            )

            mlflow.log_params(
                clean_and_prefix(meta_params, "meta")
            )

            mlflow.log_param(
                "dataset_rows",
                train_df.shape[0]
            )

            mlflow.log_param(
                "dataset_columns",
                train_df.shape[1]
            )

            mlflow.log_param(
                "git_commit",
                get_git_commit()
            )

            logger.info("Starting model training")

            final_model.fit(X_train, y_train)

            logger.info("Model training completed")

            model_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            local_model_path = (
                model_dir
                / "stacking_cat_rf_pipeline.joblib"
            )

            joblib.dump(
                final_model,
                local_model_path
            )

            logger.info(
                f"Model saved locally at {local_model_path}"
            )

            logger.info("Registering model in MLflow")

            model_info = mlflow.sklearn.log_model(
                sk_model=final_model,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME
            )

            model_version = (
                model_info.registered_model_version
            )

            run_id = run.info.run_id

            mlflow.log_param(
                "registered_model_version",
                model_version
            )

            mlflow.set_tag(
                "lifecycle_stage",
                "candidate"
            )

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                CANDIDATE_ALIAS
            )

            logger.debug(
                "Existing candidate alias removed"
            )

        except Exception:
            logger.debug(
                "No previous candidate alias found"
            )

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            CANDIDATE_ALIAS,
            version=str(model_version)
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            str(model_version),
            "training_run_id",
            run_id
        )

        logger.info(
            f"Model version {model_version} assigned as candidate"
        )

        logger.info(
            "Model training stage completed successfully"
        )

    except MlflowException:
        logger.exception("MLflow operation failed")
        sys.exit(1)

    except Exception:
        logger.exception("Model training pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()