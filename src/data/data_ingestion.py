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

    logger.info("Starting data ingestion from Amazon S3")

    try:
        local_dir = os.path.dirname(LOCAL_PATH)

        logger.debug(f"Creating local directory: {local_dir}")

        os.makedirs(
            local_dir,
            exist_ok=True
        )

        logger.debug("Initializing S3 client")

        s3 = boto3.client("s3")

        logger.info(
            f"Downloading file from bucket='{BUCKET_NAME}', "
            f"key='{S3_KEY}'"
        )

        s3.download_file(
            Bucket=BUCKET_NAME,
            Key=S3_KEY,
            Filename=LOCAL_PATH
        )

        logger.info(
            f"Dataset downloaded successfully to {LOCAL_PATH}"
        )

    except ClientError:

        logger.exception(
            "AWS client error occurred during data ingestion"
        )
        raise

    except Exception:

        logger.exception(
            "Unexpected error occurred during data ingestion"
        )
        raise


if __name__ == "__main__":
    download_dataset_from_s3()