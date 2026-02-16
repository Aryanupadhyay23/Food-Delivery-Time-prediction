# Food Delivery Time Prediction – MLOps System

This project is an end-to-end production-grade machine learning system for predicting food delivery time. It integrates model training, experiment tracking, containerized inference, CI/CD automation, and AWS-based deployment into a complete and scalable MLOps workflow.

The objective of this project is not just to train a high-performing model, but to design a deployable, maintainable, and version-controlled ML system that reflects real-world engineering practices.

---

## Project Overview

The system includes:

* Data cleaning and preprocessing
* Feature engineering
* Ensemble model training using a Stacking Regressor
* MLflow experiment tracking and model registry (via DagsHub)
* Alias-based model lifecycle management
* Dockerized FastAPI inference service
* CI/CD automation using GitHub Actions
* Deployment to AWS EC2 using CodeDeploy
* Instance lifecycle management using Auto Scaling Groups

This architecture mirrors how production ML systems are built and maintained in industry environments.

---

## Model Architecture

The final model is a **Stacking Regressor**.

Base models:

* CatBoost Regressor
* Random Forest Regressor

Meta model:

* Tree-based regressor

All models are logged and versioned in MLflow Model Registry.

The inference service always loads the model using:

models:/FoodDeliveryTimeModel@production

This alias-based approach enables seamless model updates without modifying application code.

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── interim/
│   └── processed/
├── artifacts/
├── models/
├── reports/
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── inference_app/
├── deploy/
├── appspec.yml
├── Dockerfile
├── params.yaml
└── requirements files
```

The repository is organized to clearly separate data processing, model development, inference logic, and deployment configuration.

---

## Training and Evaluation Pipeline

The training workflow follows this sequence:

Data Cleaning
→ Feature Engineering
→ Train/Test Split
→ Preprocessing
→ Model Training
→ Model Evaluation
→ Model Diagnostics
→ Model Registration

All experiments, metrics, and artifacts are tracked in MLflow.

---

## Model Promotion Workflow

Models move through structured stages:

Candidate
→ Staging
→ Production

Promotion is governed by validation thresholds such as:

* Minimum R²
* Maximum MAE
* Latency constraints
* Extreme error limits

The production alias determines which model version is served in the live API.

---

## CI/CD Architecture

Deployment is triggered on push to the main branch.

### CI Phase

* Install dependencies
* Run validation checks
* Build Docker image
* Push image to Amazon ECR

### CD Phase

* CodeDeploy initiates deployment
* EC2 instance pulls latest Docker image
* Existing container is stopped
* New container is started
* Health endpoint is validated

Deployment succeeds only if the health check passes.

---

## Inference Service

The model is served through a Dockerized FastAPI application.

Endpoints:

GET /health – Returns service status and model load state
GET /docs – Interactive API documentation
POST /predict – Returns estimated delivery time in minutes

The model is loaded during application startup from MLflow using the production alias.

---

## AWS Infrastructure

The deployment stack consists of:

* Amazon ECR – Docker image registry
* Amazon EC2 – Application hosting
* Auto Scaling Group (ASG) – Instance lifecycle management
* AWS CodeDeploy – Deployment orchestration
* IAM – Secure authentication and role-based access

### Deployment Flow

GitHub
→ GitHub Actions
→ Amazon ECR
→ CodeDeploy
→ EC2 (managed by ASG)
→ Docker container
→ FastAPI
→ MLflow production model

The Auto Scaling Group ensures controlled instance provisioning and replacement. CodeDeploy manages application rollout and validates health before marking deployment successful.

---

## Monitoring and Version Control

The system supports:

* MLflow experiment tracking
* Model versioning through Model Registry
* Alias-based production control
* Health-gated deployments
* Container restart policy for resilience

---

## Rollback Strategy

Rollback is handled at the model registry level.

Reassign the production alias to a previous model version in MLflow. No code changes or redeployment are required.

---

## Key Outcomes

* End-to-end MLOps implementation
* Automated CI/CD pipeline
* Containerized inference architecture
* Alias-driven model lifecycle management
* Health-validated production deployment
* Infrastructure lifecycle control using ASG
