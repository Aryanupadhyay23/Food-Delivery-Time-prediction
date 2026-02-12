import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator

# ---------------- Configuration ----------------
TARGET = "time_taken"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------------- Helper Functions ----------------

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from {data_path}...")
        return pd.read_csv(data_path, engine="pyarrow")
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def get_models() -> Tuple[List[Tuple[str, BaseEstimator]], BaseEstimator]:
    """
    Returns CatBoost + Random Forest as Base, Decision Tree as Meta.
    """
    # 1. CatBoost (The Precision Specialist)
    cat_model = CatBoostRegressor(
        iterations=1466,
        depth=10,
        learning_rate=0.038,
        l2_leaf_reg=26.02,
        random_strength=9.56,
        bagging_temperature=0.73,
        random_seed=42,
        verbose=0,
        allow_writing_files=False
    )

    # 2. Random Forest (The Safety Net)
    rf_model = RandomForestRegressor(
        n_estimators=290,
        max_depth=14,
        min_samples_split=7,
        min_samples_leaf=1,
        max_features=None,
        bootstrap=True,
        random_state=42,
        n_jobs=None # StackingRegressor controls threading
    )

    # Meta Model: Decision Tree
    meta_model = DecisionTreeRegressor(
        max_depth=6,
        min_samples_split=7,
        min_samples_leaf=8,
        random_state=42
    )
    
    estimators = [
        ('catboost', cat_model),
        ('random_forest', rf_model)
    ]
    
    return estimators, meta_model

def main(args):
    # ---------------- Setup Paths ----------------
    root_path = Path(args.root_dir)
    train_path = root_path / "data" / "processed" / "train.csv"
    preprocessor_path = root_path / "artifacts" / "preprocessor.pkl"
    model_save_dir = root_path / "models"
    
    # ---------------- Load & Split ----------------
    train_df = load_data(train_path)
    
    if TARGET not in train_df.columns:
        logger.error(f"Target column '{TARGET}' not found in dataset.")
        sys.exit(1)

    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]

    # ---------------- Preprocessing ----------------
    logger.info("Loading preprocessor...")
    try:
        preprocessor = joblib.load(preprocessor_path)
        X_transformed = preprocessor.transform(X)
    except FileNotFoundError:
        logger.error(f"Preprocessor not found at {preprocessor_path}")
        sys.exit(1)

    # ---------------- Model Definition ----------------
    logger.info("Initializing Stacking Architecture (CatBoost + RF)...")
    estimators, meta_model = get_models()

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=False 
    )

    final_pipeline = TransformedTargetRegressor(
        regressor=stacking_regressor,
        transformer=PowerTransformer(),
        check_inverse=False
    )

    # ---------------- Training ----------------
    logger.info("Starting training...")
    final_pipeline.fit(X_transformed, y)
    logger.info("Training completed.")

    # ---------------- Saving ----------------
    model_save_dir.mkdir(parents=True, exist_ok=True)
    # Naming it specifically for CatBoost + RF
    save_path = model_save_dir / "stacking_cat_rf_pipeline.joblib"
    
    joblib.dump(final_pipeline, save_path)
    logger.info(f"Full pipeline saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stacking (CatBoost + RF)")
    parser.add_argument("--root_dir", type=str, default=".", help="Root project directory")
    
    args = parser.parse_args()
    main(args)