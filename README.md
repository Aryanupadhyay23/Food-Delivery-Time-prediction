# Food Delivery Time Prediction – MLOps System

This project is a complete machine learning system built to predict food delivery time. It goes beyond model training and focuses on how machine learning solutions are developed, managed, deployed, and maintained in real production environments.

The project combines data processing, model training, experiment tracking, API deployment, CI/CD automation, and AWS infrastructure into one workflow.

The main goal was to build a practical and reliable ML system that follows industry-style engineering practices.

---

## Project Overview

The system includes:

- Data cleaning and preprocessing  
- Feature engineering  
- Ensemble model training using a stacking model  
- MLflow experiment tracking and model registry through DagsHub  
- Model version management using aliases  
- Dockerized FastAPI inference service  
- Automated CI/CD pipeline with GitHub Actions  
- Deployment on AWS EC2 using CodeDeploy  
- Auto Scaling Group based instance management  

This project is designed to reflect how machine learning systems are handled after model development.

---

## Model Architecture

The final model uses a **Stacking Regressor**.

### Base Models

- CatBoost Regressor  
- Random Forest Regressor  

### Final Meta Model

- Tree-based regressor  

This combination helps improve prediction performance by combining strengths of multiple models.

All trained models are logged and versioned in MLflow.

The live API always loads the current production model using:

```python
models:/FoodDeliveryTimeModel@production
```

This means model upgrades can happen without changing application code.

---

## Project Structure

```text
.
├── data/
│   ├── raw/
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
├── dvc.yaml
├── params.yaml
└── requirements files
```

The structure keeps training, deployment, and data pipeline components organized and easy to manage.

---

## Training Pipeline

The workflow follows these stages:

Data Cleaning  
→ Train/Test Split  
→ Data Preprocessing  
→ Model Training  
→ Model Evaluation  
→ Model Diagnostics  
→ Model Registration  

Each stage is tracked and reproducible using DVC.

MLflow stores:

* Parameters
* Metrics
* Model versions
* Artifacts

---

## Model Promotion Workflow

Every trained model moves through clear lifecycle stages:

Candidate  
→ Staging  
→ Production  

Before promotion, the model must pass quality checks such as:

* Minimum R² score
* Maximum MAE
* Latency limits
* Extreme prediction error checks

Only approved models are promoted to production.

---

## CI/CD Pipeline

Deployment starts automatically when code is pushed to the main branch.

### CI Stage

* Install dependencies
* Run checks and validation
* Pull required artifacts with DVC
* Promote model to staging
* Promote model to production

### CD Stage

* Build Docker image
* Run container health tests
* Push image to Amazon ECR
* Trigger AWS CodeDeploy

This keeps deployment fast and consistent.

---

## Inference Service

The model is served through a FastAPI application inside Docker.

### Available Endpoints

* `GET /health` → Service status
* `GET /docs` → Interactive API documentation
* `POST /predict` → Delivery time prediction

The production model is loaded during startup using MLflow registry aliases.

---

## AWS Infrastructure

The deployment setup uses:

* **Amazon ECR** – stores Docker images
* **Amazon EC2** – runs the application
* **AWS CodeDeploy** – handles deployment process
* **Auto Scaling Group** – manages EC2 lifecycle
* **IAM** – access control and permissions

### Deployment Flow

GitHub  
→ GitHub Actions  
→ Amazon ECR  
→ CodeDeploy  
→ EC2 Instance  
→ Docker Container  
→ FastAPI API  
→ Production Model  

---

## Monitoring and Reliability

The system supports:

* Experiment tracking with MLflow
* Model version history
* Alias-based production control
* Health check based deployments
* Restartable Docker containers
* Repeatable pipeline execution with DVC

---

## Rollback Strategy

Rollback is simple.

If a newer model causes issues, the production alias can be reassigned to any previous stable version inside MLflow.

This avoids retraining or code changes.

---

## Key Outcomes

* Full end-to-end MLOps project
* Real deployment workflow on AWS
* CI/CD automation
* Containerized inference API
* Versioned model lifecycle management
* Production style rollback strategy
* Practical experience with real ML system design