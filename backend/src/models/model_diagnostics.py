import json
import time
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import mlflow
from mlflow.tracking import MlflowClient

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from backend.src.models.mlflow_utils import (
        configure_mlflow,
        start_pipeline_stage_run
    )
except ImportError:
    from mlflow_utils import (
        configure_mlflow,
        start_pipeline_stage_run
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
    if not path.exists():
        logger.error(f"Dataset not found at {path}")
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path, engine="pyarrow")
        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        return df
    except Exception:
        logger.exception("Failed to load dataset")
        raise


def load_threshold(params_path: Path) -> float:
    logger.info(f"Loading threshold values from {params_path}")
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        threshold = (
            params["promotion_policy"]
            ["staging_thresholds"]
            ["diagnostics"]
            ["extreme_error_threshold"]
        )
        logger.info(f"Extreme error threshold loaded: {threshold}")
        return float(threshold)
    except Exception:
        logger.exception("Failed to load threshold values")
        raise


def plot_residuals(y_true, y_pred, save_dir: Path) -> Path:
    logger.info("Generating residual diagnostic plot")
    try:
        residuals = y_true - y_pred
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.35)
        plt.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--"
        )
        plt.title("Actual vs Predicted Delivery Time (Minutes)")
        plt.xlabel("Actual Time (Minutes)")
        plt.ylabel("Predicted Time (Minutes)")

        plt.subplot(1, 2, 2)
        sns.histplot(residuals, kde=True, bins=30, color="crimson")
        plt.axvline(x=0, color="black", linestyle="--")
        plt.title("Residual Error Distribution (Actual - Predicted)")
        plt.xlabel("Prediction Error (Minutes)")

        save_path = save_dir / "residual_analysis.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        logger.info(f"Residual plot saved at {save_path}")
        return save_path
    except Exception:
        logger.exception("Failed to generate residual plot")
        raise


def benchmark_latency(model, sample_input, iterations: int = 500) -> dict:
    logger.info(f"Benchmarking inference latency over {iterations} iterations")
    try:
        # Warm-up run
        model.predict(sample_input)

        latencies_ms = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            model.predict(sample_input)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        latencies_ms = np.array(latencies_ms)
        avg_ms = float(np.mean(latencies_ms))
        p50_ms = float(np.percentile(latencies_ms, 50))
        p90_ms = float(np.percentile(latencies_ms, 90))
        p95_ms = float(np.percentile(latencies_ms, 95))
        p99_ms = float(np.percentile(latencies_ms, 99))
        throughput_qps = float(1000.0 / avg_ms) if avg_ms > 0 else 0.0

        stats = {
            "avg_latency_ms": round(avg_ms, 4),
            "p50_latency_ms": round(p50_ms, 4),
            "p90_latency_ms": round(p90_ms, 4),
            "p95_latency_ms": round(p95_ms, 4),
            "p99_latency_ms": round(p99_ms, 4),
            "throughput_qps": round(throughput_qps, 2),
            "benchmark_iterations": iterations
        }
        logger.info(f"Latency benchmark: avg={avg_ms:.2f}ms, p95={p95_ms:.2f}ms, QPS={throughput_qps:.1f}")
        return stats
    except Exception:
        logger.exception("Latency test failed")
        raise


def main():
    try:
        configure_mlflow()
        client = MlflowClient()

        root_path = Path(__file__).resolve().parents[2]
        test_path = root_path / "data" / "processed" / "test.csv"
        params_path = root_path / "params.yaml"
        reports_dir = root_path / "reports"
        diagnostics_dir = reports_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = reports_dir / "diagnostics_metrics.json"

        threshold = load_threshold(params_path)

        logger.info("Loading candidate model from registry")
        try:
            model_version_obj = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                CANDIDATE_ALIAS
            )
        except Exception:
            logger.exception("Candidate model not found")
            raise RuntimeError("No candidate model found")

        version_number = str(model_version_obj.version)
        logger.info(f"Running diagnostics for model version {version_number}")

        local_model_path = root_path / "models" / "stacking_lgb_rf_pipeline.joblib"
        if not local_model_path.exists():
            local_model_path = root_path / "models" / "stacking_cat_rf_pipeline.joblib"

        try:
            model = mlflow.pyfunc.load_model(
                f"models:/{REGISTERED_MODEL_NAME}@{CANDIDATE_ALIAS}"
            )
        except Exception as err:
            if local_model_path.exists():
                import joblib
                logger.warning(
                    f"MLflow model load failed ({err}). Falling back to local artifact {local_model_path}"
                )
                model = joblib.load(local_model_path)
            else:
                raise

        test_df = load_data(test_path)
        X_test = test_df.drop(columns=[TARGET])
        y_test = test_df[TARGET]

        logger.info("Generating predictions for diagnostic error analysis")
        y_pred = model.predict(X_test)

        residual_plot_path = plot_residuals(y_test, y_pred, diagnostics_dir)

        sample_row = X_test.iloc[[0]]
        latency_stats = benchmark_latency(model, sample_row, iterations=500)

        residual_errors = np.abs(y_test - y_pred)
        extreme_error_count = int(np.sum(residual_errors > threshold))
        extreme_error_rate_pct = round(
            float(extreme_error_count / len(test_df) * 100.0), 4
        )

        logger.info(
            f"Extreme errors (> {threshold} min): {extreme_error_count} "
            f"({extreme_error_rate_pct}% of test set)"
        )

        diagnostics_metrics = {
            "model_version": version_number,
            "extreme_error_count": extreme_error_count,
            "extreme_error_rate_pct": extreme_error_rate_pct,
            "extreme_error_threshold": threshold,
            **latency_stats
        }

        with open(metrics_path, "w") as file:
            json.dump(diagnostics_metrics, file, indent=4)
        logger.info(f"Diagnostics metrics saved at {metrics_path}")

        # Start nested Stage 03 diagnostics run under parent pipeline
        with start_pipeline_stage_run(
            stage_name="Model_Diagnostics",
            stage_number="03",
            tags={
                "diagnosed_model_version": version_number,
                "diagnosed_alias": CANDIDATE_ALIAS,
                "model_name": REGISTERED_MODEL_NAME
            }
        ) as diag_run:
            logger.info(f"Logging diagnostics metrics to Stage 03 run (ID: {diag_run.info.run_id})")

            mlflow.log_metric("diagnostics_avg_latency_ms", latency_stats["avg_latency_ms"])
            mlflow.log_metric("diagnostics_p50_latency_ms", latency_stats["p50_latency_ms"])
            mlflow.log_metric("diagnostics_p90_latency_ms", latency_stats["p90_latency_ms"])
            mlflow.log_metric("diagnostics_p95_latency_ms", latency_stats["p95_latency_ms"])
            mlflow.log_metric("diagnostics_p99_latency_ms", latency_stats["p99_latency_ms"])
            mlflow.log_metric("diagnostics_throughput_qps", latency_stats["throughput_qps"])
            mlflow.log_metric("diagnostics_extreme_error_count", extreme_error_count)
            mlflow.log_metric("diagnostics_extreme_error_rate_pct", extreme_error_rate_pct)
            mlflow.log_metric("diagnostics_extreme_error_threshold", threshold)

            mlflow.log_artifact(str(residual_plot_path), artifact_path="diagnostic_plots")
            mlflow.log_artifact(str(metrics_path), artifact_path="diagnostic_reports")

            mlflow.set_tag("diagnostics_status", "COMPLETED")

        logger.info("Stage 03: Model Diagnostics completed successfully")

    except Exception:
        logger.exception("Model diagnostics pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()