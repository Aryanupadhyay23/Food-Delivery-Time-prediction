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
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline



# Configuration


TARGET = "time_taken"
REGISTERED_MODEL_NAME = "FoodDeliveryTimeModel"
CANDIDATE_ALIAS = "candidate"
EXPERIMENT_NAME = "FoodDeliveryTimePipeline"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)



# Environment Setup


def setup_environment():
    try:
        dagshub.init(
            repo_owner="Aryanupadhyay23",
            repo_name="Food-Delivery-Time-prediction",
            mlflow=True
        )
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info("DagsHub MLflow configured successfully.")
    except Exception as e:
        logger.error(f"DagsHub init failed: {e}")
        sys.exit(1)



# Utilities


def load_data(path: Path):
    try:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        logger.exception("Failed to load dataset.")
        sys.exit(1)


def load_params(path: Path):
    try:
        logger.info(f"Loading parameters from {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        logger.exception("Failed to load params.yaml.")
        sys.exit(1)


def clean_and_prefix(params_dict, prefix):
    return {
        f"{prefix}_{k}": (str(v) if v is None else v)
        for k, v in params_dict.items()
    }


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception:
        return "unknown"



# Model Builder


def build_model(params, preprocessor):

    cat_params = params["model_training"]["CatBoost_Regressor"]
    rf_params = params["model_training"]["RandomForest_Regressor"]
    stack_params = params["model_training"]["Stacking_Regressor"]
    meta_params = params["model_training"]["Meta_Model"]

    cat_model = CatBoostRegressor(**cat_params)
    rf_model = RandomForestRegressor(**rf_params)
    meta_model = DecisionTreeRegressor(**meta_params)

    stacking_regressor = StackingRegressor(
        estimators=[
            ("catboost", cat_model),
            ("random_forest", rf_model)
        ],
        final_estimator=meta_model,
        cv=stack_params["cv"],
        n_jobs=stack_params["n_jobs"],
        passthrough=stack_params["passthrough"]
    )

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", stacking_regressor)
    ])

    final_model = TransformedTargetRegressor(
        regressor=model_pipeline,
        transformer=PowerTransformer(),
        check_inverse=False
    )

    return final_model, cat_params, rf_params, stack_params, meta_params



# Main


def main():

    try:
        setup_environment()

        root_path = Path(__file__).parent.parent.parent

        train_path = root_path / "data" / "processed" / "train.csv"
        preprocessor_path = root_path / "artifacts" / "preprocessor.pkl"
        param_path = root_path / "params.yaml"
        model_save_dir = root_path / "models"

        params = load_params(param_path)
        train_df = load_data(train_path)

        if TARGET not in train_df.columns:
            raise ValueError(f"{TARGET} column not found in dataset")

        X = train_df.drop(columns=[TARGET])
        y = train_df[TARGET]

        logger.info("Loading preprocessor...")
        preprocessor = joblib.load(preprocessor_path)

        final_model, cat_params, rf_params, stack_params, meta_params = \
            build_model(params, preprocessor)

        client = MlflowClient()

       
        # Training
      

        with mlflow.start_run(run_name="model_training") as run:

            logger.info("Logging hyperparameters...")

            mlflow.log_params(clean_and_prefix(cat_params, "cat"))
            mlflow.log_params(clean_and_prefix(rf_params, "rf"))
            mlflow.log_params(clean_and_prefix(stack_params, "stack"))
            mlflow.log_params(clean_and_prefix(meta_params, "meta"))

            mlflow.log_param("dataset_rows", train_df.shape[0])
            mlflow.log_param("dataset_columns", train_df.shape[1])
            mlflow.log_param("git_commit", get_git_commit())

            logger.info("Training model...")
            final_model.fit(X, y)
            logger.info("Training completed.")

            model_save_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                final_model,
                model_save_dir / "stacking_cat_rf_pipeline.joblib"
            )

            logger.info("Registering model in MLflow...")

            model_info = mlflow.sklearn.log_model(
                sk_model=final_model,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME
            )

            model_version = model_info.registered_model_version
            run_id = run.info.run_id

            mlflow.log_param("registered_model_version", model_version)
            mlflow.set_tag("lifecycle_stage", "candidate")

     
        # Candidate Alias Assignment
   

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                CANDIDATE_ALIAS
            )
        except Exception:
            pass

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

        logger.info(f"Model version {model_version} marked as candidate.")
        logger.info("Training pipeline completed successfully.")

    except MlflowException:
        logger.exception("MLflow operation failed.")
        sys.exit(1)

    except Exception:
        logger.exception("Unexpected failure in training pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
