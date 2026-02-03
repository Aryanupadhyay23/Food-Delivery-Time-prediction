import os
import logging
import boto3
from botocore.exceptions import ClientError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BUCKET_NAME = "ai-ml-datasets-23"
S3_KEY = "food-delivery-time-prediction/Zomato-Dataset.csv"
LOCAL_PATH = "data/raw/Zomato-Dataset.csv"


def download_dataset_from_s3():
    logger.info("Starting data ingestion from S3")

    try:
        # ensure local directory exists
        os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)

        s3 = boto3.client("s3")

        s3.download_file(
            Bucket=BUCKET_NAME,
            Key=S3_KEY,
            Filename=LOCAL_PATH
        )

        logger.info(f"Dataset downloaded successfully to {LOCAL_PATH}")

    except ClientError as e:
        logger.error("Failed to download dataset from S3")
        logger.error(e)
        raise e

if __name__ == "__main__":
    download_dataset_from_s3()
