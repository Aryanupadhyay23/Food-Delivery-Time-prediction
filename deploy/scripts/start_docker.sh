#!/bin/bash

set -e

# Log everything
exec > /home/ubuntu/start_docker.log 2>&1

AWS_REGION="us-east-1"
ECR_REGISTRY="125840290869.dkr.ecr.us-east-1.amazonaws.com"
IMAGE_NAME="food_delivery_time_prediction"
CONTAINER_NAME="delivery_time_pred"
HOST_PORT=80
CONTAINER_PORT=8000
MAX_RETRIES=20
SLEEP_INTERVAL=3

echo "===== Starting Deployment ====="

echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ECR_REGISTRY

echo "Pulling latest Docker image..."
docker pull $ECR_REGISTRY/$IMAGE_NAME:latest

echo "Stopping existing container (if running)..."
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    docker stop $CONTAINER_NAME
fi

echo "Removing existing container (if exists)..."
if docker ps -aq -f name=$CONTAINER_NAME | grep -q .; then
    docker rm $CONTAINER_NAME
fi

# Load environment variables (such as DAGSHUB_TOKEN) from .env / .env.fd files if present
if [ -z "$DAGSHUB_TOKEN" ] && [ -f /home/ubuntu/.env.fd ]; then
    set -a; source /home/ubuntu/.env.fd; set +a
elif [ -z "$DAGSHUB_TOKEN" ] && [ -f /home/ubuntu/.env ]; then
    set -a; source /home/ubuntu/.env; set +a
elif [ -z "$DAGSHUB_TOKEN" ] && [ -f /home/ubuntu/app/.env.fd ]; then
    set -a; source /home/ubuntu/app/.env.fd; set +a
elif [ -z "$DAGSHUB_TOKEN" ] && [ -f /home/ubuntu/app/.env ]; then
    set -a; source /home/ubuntu/app/.env; set +a
elif [ -z "$DAGSHUB_TOKEN" ] && [ -f .env.fd ]; then
    set -a; source .env.fd; set +a
elif [ -z "$DAGSHUB_TOKEN" ] && [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "$DAGSHUB_TOKEN" ]; then
    echo "WARNING: DAGSHUB_TOKEN is not set in environment or .env file."
fi

echo "Starting new container..."

docker run -d \
  --name $CONTAINER_NAME \
  --restart always \
  -p ${HOST_PORT}:${CONTAINER_PORT} \
  -e DAGSHUB_TOKEN="$DAGSHUB_TOKEN" \
  $ECR_REGISTRY/$IMAGE_NAME:latest

echo "Container started. Beginning health check..."

# Retry-based health check
for i in $(seq 1 $MAX_RETRIES); do
    echo "Health check attempt $i..."

    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo "Deployment successful. API is healthy."
        echo "===== Deployment Completed Successfully ====="
        exit 0
    fi

    echo "Health check not ready. Sleeping ${SLEEP_INTERVAL}s..."
    sleep $SLEEP_INTERVAL
done

echo "Health check failed after ${MAX_RETRIES} attempts."

echo "Fetching container logs for debugging..."
docker logs $CONTAINER_NAME

exit 1

