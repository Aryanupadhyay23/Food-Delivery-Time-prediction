import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error
)
from scipy.stats import skew

# ==========================================================
# Configuration
# ==========================================================

TARGET = "time_taken"
SEGMENT_COLUMN = None  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==========================================================
# Utility Functions
# ==========================================================

def load_data(path: Path) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path, engine="pyarrow")
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        sys.exit(1)

def split_xy(df: pd.DataFrame, target: str):
    if target not in df.columns:
        logger.error(f"Target column '{target}' not found.")
        sys.exit(1)

    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def compute_metrics(y_true, y_pred, prefix=""):
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        f"{prefix}MAE": round(mean_absolute_error(y_true, y_pred), 4),
        f"{prefix}RMSE": round(rmse, 4),
        f"{prefix}R2": round(r2_score(y_true, y_pred), 4),
        f"{prefix}MedianAE": round(median_absolute_error(y_true, y_pred), 4),
        f"{prefix}MAPE": round(mean_absolute_percentage_error(y_true, y_pred), 4),
        
        # Max Error is critical for reliability
        f"{prefix}Max_Error": round(np.max(abs_residuals), 4),

        # Residual diagnostics
        f"{prefix}Error_Mean": round(np.mean(residuals), 4),
        f"{prefix}Error_Std": round(np.std(residuals), 4),
        f"{prefix}Error_Skewness": round(skew(residuals), 4),

        # Risk metrics
        f"{prefix}P90_Error": round(np.percentile(abs_residuals, 90), 4),
        f"{prefix}P95_Error": round(np.percentile(abs_residuals, 95), 4),
    }

    return metrics

def compute_generalization_gap(train_metrics, test_metrics):
    gap = {}
    for key in train_metrics:
        metric_name = key.replace("train_", "")
        test_key = f"test_{metric_name}"
        if test_key in test_metrics:
            gap[f"gap_{metric_name}"] = round(
                test_metrics[test_key] - train_metrics[key],
                4
            )
    return gap

def segment_performance(df, y_true, y_pred, segment_col):
    results = {}
    df_copy = df.copy()
    df_copy["true"] = y_true.values
    df_copy["pred"] = y_pred

    for segment in df_copy[segment_col].unique():
        seg_df = df_copy[df_copy[segment_col] == segment]
        mae = mean_absolute_error(seg_df["true"], seg_df["pred"])
        results[f"{segment_col}_{segment}_MAE"] = round(mae, 4)

    return results

def save_metrics(metrics: dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved at {save_path}")

# ==========================================================
# Main Evaluation Flow
# ==========================================================

def main(args):

    root_path = Path(args.root_dir)

    train_path = root_path / "data" / "processed" / "train.csv"
    test_path = root_path / "data" / "processed" / "test.csv"
    
    # UPDATED: Pointing to the CatBoost + RF pipeline
    model_path = root_path / "models" / "stacking_cat_rf_pipeline.joblib"
    
    preprocessor_path = root_path / "artifacts" / "preprocessor.pkl"
    metrics_path = root_path / "reports" / "metrics.json"

    # 1. Load data
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    X_train, y_train = split_xy(train_df, TARGET)
    X_test, y_test = split_xy(test_df, TARGET)

    # 2. Load artifacts
    logger.info(f"Loading model from {model_path} and preprocessor...")
    try:
        model_pipeline = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError as e:
        logger.error(f"Artifact not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Artifact loading failed: {e}")
        sys.exit(1)

    # 3. Transform features
    logger.info("Transforming features...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 4. Predictions
    logger.info("Generating predictions...")
    y_train_pred = model_pipeline.predict(X_train_transformed)
    y_test_pred = model_pipeline.predict(X_test_transformed)

    # 5. Metrics
    logger.info("Computing evaluation metrics...")

    train_metrics = compute_metrics(y_train, y_train_pred, prefix="train_")
    test_metrics = compute_metrics(y_test, y_test_pred, prefix="test_")
    gap_metrics = compute_generalization_gap(train_metrics, test_metrics)

    final_metrics = {
        **train_metrics,
        **test_metrics,
        **gap_metrics
    }

    # 6. Optional segmentation
    if SEGMENT_COLUMN and SEGMENT_COLUMN in test_df.columns:
        logger.info(f"Running segment-wise evaluation on {SEGMENT_COLUMN}")
        segment_metrics = segment_performance(
            test_df, y_test, y_test_pred, SEGMENT_COLUMN
        )
        final_metrics.update(segment_metrics)

    # 7. Logging results
    logger.info("Final Evaluation Summary:")
    print(json.dumps(final_metrics, indent=2))

    # 8. Save metrics
    save_metrics(final_metrics, metrics_path)

    logger.info("Evaluation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enterprise-Grade Model Evaluation Script"
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Root project directory"
    )

    args = parser.parse_args()
    main(args)