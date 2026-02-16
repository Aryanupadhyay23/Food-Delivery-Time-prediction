#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Log everything
exec > /home/ubuntu/start_docker.log 2>&1

AWS_REGION="us-east-1"
ECR_REGISTRY="125840290869.dkr.ecr.us-east-1.amazonaws.com"
IMAGE_NAME="food_delivery_time_prediction"
CONTAINER_NAME="delivery_time_pred"

echo "===== Starting Deployment ====="

echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ECR_REGISTRY

echo "Pulling latest Docker image..."
docker pull $ECR_REGISTRY/$IMAGE_NAME:latest

echo "Stopping existing container (if running)..."
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    docker stop $CONTAINER_NAME
fi

echo "Removing existing container (if exists)..."
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    docker rm $CONTAINER_NAME
fi

echo "Starting new container..."

docker run -d \
  --name $CONTAINER_NAME \
  -p 80:8000 \
  -e DAGSHUB_TOKEN=16851a314ba5c13f8adb30964e6d87b7c4497394 \
  $ECR_REGISTRY/$IMAGE_NAME:latest

echo "Waiting for container health..."
sleep 10

if curl -f http://localhost/health; then
    echo "Deployment successful and API healthy."
else
    echo "Health check failed."
    exit 1
fi

echo "===== Deployment Completed Successfully ====="

