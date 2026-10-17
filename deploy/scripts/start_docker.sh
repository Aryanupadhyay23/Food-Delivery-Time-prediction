#!/bin/bash

set -e

# Log everything
exec > /home/ubuntu/start_docker.log 2>&1

AWS_REGION="ap-southeast-2"
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "308525896139")}
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_NAME="food_delivery_time_prediction"
CONTAINER_NAME="delivery_time_pred"
STREAMLIT_HOST_PORT=${STREAMLIT_HOST_PORT:-8510}
FASTAPI_HOST_PORT=${FASTAPI_HOST_PORT:-8080}
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

# Locate and load environment file specifically for Food Delivery (.env.fd)
# Avoids reading /home/ubuntu/.env which belongs to another project on this EC2 instance
ENV_FILE=""
if [ -f /home/ubuntu/.env.fd ]; then
    ENV_FILE="/home/ubuntu/.env.fd"
elif [ -f /home/ubuntu/app/.env.fd ]; then
    ENV_FILE="/home/ubuntu/app/.env.fd"
elif [ -f .env.fd ]; then
    ENV_FILE=".env.fd"
fi

ENV_FILE_FLAG=""
if [ -n "$ENV_FILE" ]; then
    echo "Found Food Delivery environment file at $ENV_FILE. Sourcing variables..."
    set -a
    source "$ENV_FILE"
    set +a
    ENV_FILE_FLAG="--env-file $ENV_FILE"
else
    echo "WARNING: .env.fd was not found! Please place it at /home/ubuntu/.env.fd on EC2."
fi

if [ -z "$DAGSHUB_TOKEN" ]; then
    echo "WARNING: DAGSHUB_TOKEN is not set in environment or .env.fd file."
fi

echo "Starting new container..."

docker run -d \
  --name $CONTAINER_NAME \
  --restart always \
  -p ${STREAMLIT_HOST_PORT}:8510 \
  -p ${FASTAPI_HOST_PORT}:8080 \
  ${ENV_FILE_FLAG} \
  -e DAGSHUB_TOKEN="$DAGSHUB_TOKEN" \
  -e FASTAPI_PORT=8080 \
  -e STREAMLIT_PORT=8510 \
  -e BACKEND_URL="http://localhost:8080" \
  $ECR_REGISTRY/$IMAGE_NAME:latest

echo "Container started. Beginning health check..."

# Retry-based health check
for i in $(seq 1 $MAX_RETRIES); do
    echo "Health check attempt $i..."

    if curl -f http://localhost:${FASTAPI_HOST_PORT}/health > /dev/null 2>&1; then
        echo "Deployment successful. API is healthy on port ${FASTAPI_HOST_PORT}."
        echo "Streamlit frontend is available on port ${STREAMLIT_HOST_PORT}."
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

