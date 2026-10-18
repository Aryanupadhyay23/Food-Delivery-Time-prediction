# 🍔 Food Delivery Time Prediction – End-to-End MLOps System

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2.svg?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-S3_Remote-945DD6.svg?style=flat&logo=dvc&logoColor=white)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECR%20%7C%20S3%20%7C%20CodeDeploy%20%7C%20EC2-232F3E.svg?style=flat&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions%20%7C%20CodeDeploy-2088FF.svg?style=flat&logo=github-actions&logoColor=white)](https://github.com/Aryanupadhyay23/Food-Delivery-Time-prediction/actions)

An enterprise-grade, production-tested Machine Learning Operations (MLOps) platform predicting food delivery durations using an ensemble **Stacking Regressor** (LightGBM + Random Forest + Decision Tree meta-learner). 

The platform features automated data versioning via DVC with AWS S3, experiment tracking and model governance via DagsHub MLflow Model Registry, a decoupled FastAPI backend and Streamlit UI, multi-stage containerization with OS-level memory optimization, and fully automated CI/CD deployment to AWS EC2 via AWS CodeDeploy.

---

## 🌐 Live Deployment (Available Mon – Fri, 9:00 AM – 5:00 PM)

| Service | Live URL | Description |
| :--- | :--- | :--- |
| **Frontend (Streamlit)** | [https://aryan-projects.duckdns.org/food-delivery/](https://aryan-projects.duckdns.org/food-delivery/) | Real-time map, fleet profile, and delivery estimation UI |
| **Backend (Swagger UI)** | [https://aryan-projects.duckdns.org/food-delivery-backend/docs](https://aryan-projects.duckdns.org/food-delivery-backend/docs) | Interactive OpenAPI testing console and schemas |


---

## 🏛️ System Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              DATA & TRAINING PIPELINE                                  │
 │                                                                                        │
 │   Raw CSV Data  ──[DVC Tracked]──> S3 Remote (s3://ai-ml-datasets-2301)                │
 │        │                                                                               │
 │        ▼                                                                               │
 │   Data Preparation & Feature Pipeline (Haversine Geo-Dist, Categorical, Time Features) │
 │        │                                                                               │
 │        ▼                                                                               │
 │   Stacking Regressor Training (LightGBM + Random Forest + Decision Tree Meta-Learner)  │
 │        │                                                                               │
 │        ▼                                                                               │
 │   Model Governance Gate (R² ≥ 0.82, MAE ≤ 6.0m, Latency ≤ 120ms, Extreme Residuals ≤ 2)│
 │        │                                                                               │
 │        ▼                                                                               │
 │   MLflow Model Registry (models:/FoodDeliveryTimeModel@production)                     │
 └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            │  Dynamic In-Memory Pull via Token
                                            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                         AWS EC2 HOST (ap-southeast-2)                                  │
 │                                                                                        │
 │   Incoming HTTPS Requests (https://aryan-projects.duckdns.org)                         │
 │        │                                                                               │
 │        ▼                                                                               │
 │   ┌────────────────────────────────────────────────────────────────────────────────┐   │
 │   │                   NGINX REVERSE PROXY & SSL TERMINATION                        │   │
 │   │  - /food-delivery/        ─────────────> Proxy to Port 8510                    │   │
 │   │  - /food-delivery-backend/ ────────────> Proxy to Port 8080                    │   │
 │   └───────────────────────────────┬────────────────────────────────────────────────┘   │
 │                                   │                                                    │
 │                                   ▼                                                    │
 │   ┌────────────────────────────────────────────────────────────────────────────────┐   │
 │   │                      PRODUCTION DOCKER CONTAINER                               │   │
 │   │                                                                                │   │
 │   │   ┌───────────────────────────┐    Internal HTTP   ┌────────────────────────┐  │   │
 │   │   │  Streamlit Prediction UI  │  ────────────────> │     FastAPI Engine     │  │   │
 │   │   │        (Port 8510)        │                    │      (Port 8080)       │  │   │
 │   │   │  --baseUrlPath=           │                    │  --root-path=          │  │   │
 │   │   │  food-delivery            │                    │  /food-delivery-backend│  │   │
 │   │   └───────────────────────────┘                    └───────────┬────────────┘  │   │
 │   │                                                                │               │   │
 │   │                                                                ▼               │   │
 │   │                                                     MLflow In-Memory Model     │   │
 │   │                                                     (Pure Remote Registry)     │   │
 │   │                                                     jemalloc + malloc_trim     │   │
 │   └────────────────────────────────────────────────────────────────────────────────┘   │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Architectural Highlights

1. **Exclusively Remote MLflow Model Loading (Zero Local Fallback)**:
   - The FastAPI backend does not bundle or depend on static `.joblib` model files inside the Docker container.
   - On container startup, it authenticates with DagsHub using `DAGSHUB_TOKEN` and dynamically loads `models:/FoodDeliveryTimeModel@production` directly into memory.
   - Updating or rolling back models in production requires only re-pointing the `@production` alias in MLflow—no image rebuilding, no redeployment, and zero downtime.

2. **Container & OS-Level Memory Optimization (~340 MB RAM Saved)**:
   - Disables Joblib multiprocessing and loky background subprocesses (`JOBLIB_MULTIPROCESSING=0`, `LOKY_MAX_CPU_COUNT=1`).
   - Caps thread pool allocations (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).
   - Integrates `libjemalloc.so.2` and programmatic heap trimming via `ctypes` calling `libc.malloc_trim(0)` to immediately release unreferenced C heap pages back to the host operating system.
   - Configures a dedicated 2GB swap partition (`deploy/scripts/setup_swap.sh`) with `vm.swappiness=10` on the EC2 host to safeguard against OOM spikes.

3. **Reverse Proxy Subpath Routing (`/food-delivery` & `/food-delivery-backend`)**:
   - Streamlit runs with `--server.baseUrlPath=food-delivery` to ensure all assets and WebSockets route seamlessly behind Nginx reverse proxy.
   - FastAPI runs with `--root-path /food-delivery-backend` so OpenAPI schemas, Swagger UI (`/docs`), and interactive consoles resolve correctly over HTTPS.

4. **DVC & AWS S3 Data Lineage**:
   - Training datasets and raw data assets are tracked using DVC and stored remotely in AWS S3 (`s3://ai-ml-datasets-2301/zomato-food-delivery-time-prediction/dvc-store`).
   - Ensures full reproducibility of every stage: `data_cleaning` → `data_splitting` → `data_preprocessing` → `model_training` → `model_evaluation` → `model_diagnostics`.

5. **Multi-Stage Slim Docker Architecture**:
   - Uses multi-stage builds (`python:3.12-slim`) with pre-compiled wheels in `/opt/venv` to keep image size minimal and build times fast.
   - Runs with a non-root security profile (`appuser`).
   - Includes a native `HEALTHCHECK` testing `curl -f http://localhost:8080/health`.

6. **Multi-Project EC2 Isolation with Dedicated Ports**:
   - Dedicated ports: **FastAPI on `8080`**, **Streamlit on `8510`** (strictly avoiding ports `80`, `8000`, `8001`, `7860`, `8501`, `8502` used by other projects).
   - Dedicated environment file: Reads strictly from `/home/ubuntu/.env.fd` on EC2, preventing any overlap with standard `/home/ubuntu/.env`.

7. **Dynamic AWS Identity CI/CD**:
   - GitHub Actions workflow dynamically resolves AWS Account ID using `steps.login-ecr.outputs.registry` and `aws sts get-caller-identity`, ensuring seamless execution regardless of differing local and runner AWS accounts.

---

## 📂 Repository Structure

```text
.
├── backend/
│   ├── artifacts/                    # Serialized preprocessor pipeline
│   ├── data/
│   │   ├── raw/                      # Versioned raw data (DVC pointer)
│   │   └── processed/                # Processed train/test data splits
│   ├── models/                       # Local training output directory
│   ├── notebooks/                    # Exploratory data analysis & tuning notebooks
│   ├── reports/                      # Evaluation and diagnostic metrics
│   ├── src/
│   │   ├── data/
│   │   │   └── data_cleaning.py      # Missing value handling, outlier filtering & format normalization
│   │   ├── features/
│   │   │   ├── data_splitting.py     # Stratified train/test dataset partition
│   │   │   └── data_preprocessing.py # ColumnTransformer (StandardScaler, OneHot, Ordinal)
│   │   └── models/
│   │       ├── mlflow_utils.py       # MLflow & DagsHub run orchestration & lifecycle helpers
│   │       ├── model_training.py     # LightGBM + Random Forest Stacking Regressor training
│   │       ├── model_evaluation.py   # Test set evaluation (R², MAE, RMSE, error distribution)
│   │       ├── model_diagnostics.py  # Inference latency testing & extreme residual analysis
│   │       ├── promote_to_staging.py # Automated governance gate (staging validation)
│   │       └── promote_to_production.py # Staging-to-production promotion in MLflow Registry
│   ├── utility_files/
│   │   └── inference_transformer.py  # Production inference feature transformer (Haversine & time features)
│   ├── main.py                       # Asynchronous FastAPI service (Port 8080, root-path /food-delivery-backend)
│   └── params.yaml                   # Hyperparameters, random seeds & promotion gate thresholds
├── frontend/
│   ├── components.py                 # Streamlit UI cards, metrics, and map widgets
│   ├── config.py                     # Frontend configuration & backend URL
│   ├── api_client.py                 # Resilient HTTP client connecting to FastAPI
│   ├── city_registry.py              # Coordinate database for Indian metropolitan areas
│   ├── state.py                      # Session state manager
│   ├── styles.py                     # Custom CSS styling tokens
│   ├── utils.py                      # Time, date, and coordinate helpers
│   └── streamlit_app.py              # Streamlit Prediction Studio (Port 8510, baseUrlPath food-delivery)
├── deploy/
│   └── scripts/
│       ├── install_dependencies.sh   # EC2 setup script (Docker & AWS CLI v2)
│       ├── setup_swap.sh             # EC2 2GB swap creation with swappiness=10
│       ├── start.sh                  # Dual-process container supervisor (FastAPI + Streamlit)
│       └── start_docker.sh           # CodeDeploy container lifecycle & deployment script
├── .dvc/
│   └── config                        # S3 remote storage configuration
├── .github/workflows/
│   └── CI_CD.yaml                    # GitHub Actions CI/CD pipeline
├── Dockerfile                        # Multi-stage production container definition
├── dev-requirements.txt              # Development, training, and CI runner dependencies
├── docker-requirements.txt           # Production runtime container dependencies
├── dvc.yaml                          # DVC automated stage pipeline definition
├── dvc.lock                          # DVC pipeline lockfile
├── appspec.yml                       # AWS CodeDeploy deployment hooks
└── README.md                         # Project documentation
```

---

## 🤖 Machine Learning Model Architecture

The core model is an ensemble **Stacking Regressor** wrapped in a `TransformedTargetRegressor` with `PowerTransformer` target normalization:

```text
                   ┌───────────────────────────────────┐
                   │        Raw Request Payload        │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │    Inference Feature Pipeline     │
                   │  - Haversine Distance Calculation │
                   │  - Categorical Harmonization      │
                   │  - Time-of-Day & Day-of-Week Ext. │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │     ColumnTransformer Pipeline    │
                   │  - StandardScaler (Numerical)     │
                   │  - OneHotEncoder (Nominal)        │
                   │  - OrdinalEncoder (Traffic)       │
                   └─────────────────┬─────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │                               │
                     ▼                               ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │     LightGBM Regressor    │   │  Random Forest Regressor  │
       │    (Base Estimator 1)     │   │    (Base Estimator 2)     │
       └─────────────┬─────────────┘   └─────────────┬─────────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │     Decision Tree Regressor       │
                   │         (Meta-Estimator)          │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │   Estimated Delivery Time (Min)   │
                   └───────────────────────────────────┘
```

### Model Performance Metrics (Test Set)

| Metric | Target | Actual Result | Status |
| :--- | :--- | :--- | :--- |
| **$R^2$ Score** | $\ge 0.80$ | **`0.8396`** | Passed |
| **MAE** | $\le 3.50$ mins | **`3.0409`** mins | Passed |
| **RMSE** | $\le 4.20$ mins | **`3.7695`** mins | Passed |
| **Median AE** | $\le 3.00$ mins | **`2.6693`** mins | Passed |
| **Average Latency** | $\le 25$ ms | **`6.8`** ms | Passed |

---

## ⚙️ DVC Data & Pipeline Reproducibility

Raw data is versioned using DVC and stored in Amazon S3 (`s3://ai-ml-datasets-2301/zomato-food-delivery-time-prediction/dvc-store`).

### 1. Pull Versioned Data from S3
```bash
# Pull raw dataset from S3 remote storage
dvc pull backend/data/raw/Zomato-Dataset.csv.dvc

# Or pull all pipeline artifacts
dvc pull
```

### 2. Reproduce the Entire Training Pipeline
```bash
# Reproduce all 6 DVC pipeline stages defined in dvc.yaml
dvc repro
```

### 3. Track New Dataset Changes & Push to S3
```bash
# 1. Track the updated raw dataset with DVC
dvc add backend/data/raw/Zomato-Dataset.csv

# 2. Push binary data to AWS S3 DVC remote
dvc push

# 3. Commit the updated .dvc pointer and lockfile to Git
git add backend/data/raw/Zomato-Dataset.csv.dvc dvc.lock
git commit -m "Track updated raw dataset via DVC on S3"
git push origin master
```

---

## 📡 REST API Reference

The FastAPI service is deployed at **`https://aryan-projects.duckdns.org/food-delivery-backend`** (or `http://localhost:8080` locally).

### 1. Health Check
```bash
curl -X GET "https://aryan-projects.duckdns.org/food-delivery-backend/health"
```
**Response**:
```json
{
  "status": "healthy",
  "service": "backend",
  "async": true,
  "model_loaded": true
}
```

### 2. Model Information
```bash
curl -X GET "https://aryan-projects.duckdns.org/food-delivery-backend/model-info"
```
**Response**:
```json
{
  "registered_model": "FoodDeliveryTimeModel",
  "alias": "production",
  "model_loaded": true
}
```

### 3. Predict Single Order Delivery Duration
```bash
curl -X POST "https://aryan-projects.duckdns.org/food-delivery-backend/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Restaurant_latitude": 19.0760,
    "Restaurant_longitude": 72.8777,
    "Delivery_location_latitude": 19.1200,
    "Delivery_location_longitude": 72.9000,
    "Delivery_person_Age": 29.0,
    "Delivery_person_Ratings": 4.8,
    "Delivery_person_ID": "MUMRES12345",
    "Type_of_vehicle": "motorcycle",
    "Vehicle_condition": 2,
    "multiple_deliveries": 0.0,
    "Weather_conditions": "Sunny",
    "Road_traffic_density": "Medium",
    "Festival": "No",
    "City": "Metropolitian",
    "ID": "ORD_001",
    "Type_of_order": "Meal",
    "Order_Date": "18-10-2026",
    "Time_Orderd": "19:30"
  }'
```
**Response**:
```json
{
  "estimated_delivery_minutes": 14.3
}
```

### 4. Batch Prediction
```bash
curl -X POST "https://aryan-projects.duckdns.org/food-delivery-backend/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "orders": [
      {
        "Restaurant_latitude": 19.0760,
        "Restaurant_longitude": 72.8777,
        "Delivery_location_latitude": 19.1200,
        "Delivery_location_longitude": 72.9000,
        "Delivery_person_Age": 29.0,
        "Delivery_person_Ratings": 4.8,
        "Delivery_person_ID": "MUMRES12345",
        "Type_of_vehicle": "motorcycle",
        "Vehicle_condition": 2,
        "multiple_deliveries": 0.0,
        "Weather_conditions": "Sunny",
        "Road_traffic_density": "Medium",
        "Festival": "No",
        "City": "Metropolitian",
        "ID": "ORD_001",
        "Type_of_order": "Meal",
        "Order_Date": "18-10-2026",
        "Time_Orderd": "19:30"
      }
    ]
  }'
```
**Response**:
```json
{
  "predictions": [
    {
      "order_id": "ORD_001",
      "estimated_delivery_minutes": 14.3
    }
  ],
  "total_orders": 1
}
```

---

## 🎨 Streamlit Prediction Studio

Access the interactive frontend at **[https://aryan-projects.duckdns.org/food-delivery/](https://aryan-projects.duckdns.org/food-delivery/)**.

### Features:
- **Interactive City & Route Selector**: Preset coordinate registries for top metropolitan hubs (Bangalore, Mumbai, Delhi, Hyderabad, Pune, Kolkata, Chennai).
- **Interactive Route Map**: Visualizes restaurant origin, delivery destination, and Haversine distance straight line or road corridor.
- **Fleet & Driver Profiling**: Real-time sliders for rider age, partner rating, vehicle condition, and active parallel orders.
- **Environmental Modifiers**: Toggle weather dynamics (Sunny, Stormy, Sandstorms, Fog, Windy) and traffic density (Low, Medium, High, Jam).
- **Instant Inference Display**: Displays estimated delivery minutes with traffic delta and status badges.

---

## ☁️ AWS Cloud Infrastructure & Nomenclature

All cloud resources are deployed in the Sydney region:

| AWS Service | Resource / Setting | Configured Value |
| :--- | :--- | :--- |
| **AWS Region** | Default Region | `ap-southeast-2` *(Sydney)* |
| **Amazon ECR** | Repository Name | `food_delivery_time_prediction` |
| **Amazon ECR** | Full Repository URI | `308525896139.dkr.ecr.ap-southeast-2.amazonaws.com/food_delivery_time_prediction` |
| **Amazon S3** | Deployment Bundle Bucket | `fooddeliverytimepredictiondeployments3` |
| **Amazon S3** | DVC Dataset Remote Storage | `ai-ml-datasets-2301` |
| **AWS CodeDeploy**| Application Name | `food-delivery-time-prediction-app` |
| **AWS CodeDeploy**| Deployment Group | `food-delivery-time-prediction-group` |
| **Amazon EC2** | Target User & Directory | `ubuntu` (`/home/ubuntu`) |
| **Amazon EC2** | Environment Config | `/home/ubuntu/.env.fd` *(strictly isolated)* |
| **Amazon EC2** | Swap File Allocation | `2GB` (`/swapfile`, `vm.swappiness=10`) |
| **Inbound Security Group** | Ports Exposed | `8510` (Streamlit), `8080` (FastAPI), `22` (SSH) |
| **Public SSL/TLS Routing** | Domain & Paths | `https://aryan-projects.duckdns.org/food-delivery/`<br>`https://aryan-projects.duckdns.org/food-delivery-backend/` |

---

## 🚦 CI/CD Master Switch & Pipeline Workflow

The repository includes a safety master switch located at the top of [`.github/workflows/CI_CD.yaml`](.github/workflows/CI_CD.yaml):

```yaml
env:
  ENABLE_CI_CD: "true"  # Set to "false" to turn off CI/CD without failing workflow runs
```

### Automated Pipeline Stages:
1. **Gate Switch Evaluation:** Verifies whether `ENABLE_CI_CD == 'true'`. If set to `"false"`, the pipeline terminates gracefully without consuming runner minutes.
2. **Lint & Code Quality:** Runs `flake8` against `backend` and `frontend` to block critical errors, syntax mistakes, or undefined symbols (`E9,F63,F7,F82`).
3. **Data Pull via DVC:** Authenticates with AWS credentials and downloads the versioned raw dataset from S3:
   ```bash
   dvc pull backend/data/raw/Zomato-Dataset.csv.dvc
   ```
4. **Pipeline Reproduction (`dvc repro`):** Runs the full 6-stage training and validation sequence:
   $$\text{data\_cleaning} \rightarrow \text{data\_splitting} \rightarrow \text{data\_preprocessing} \rightarrow \text{model\_training} \rightarrow \text{model\_evaluation} \rightarrow \text{model\_diagnostics}$$
5. **Two-Stage Automated Model Governance:**
   - **Gate 1 – Staging Promotion (`promote_to_staging.py`):** Compares metrics from `backend/reports/` against thresholds in `params.yaml` ($R^2 \ge 0.82$, $\text{MAE} \le 6.0$, latency $\le 120\text{ ms}$, extreme residuals $\le 2$). On success, registers the version and assigns the `@staging` alias in DagsHub MLflow.
   - **Gate 2 – Production Promotion (`promote_to_production.py`):** Promotes the validated `@staging` candidate to `@production`.
6. **Docker Build & Health Verification:** Builds the multi-stage image, launches a temporary container, and validates the live `/health` endpoint before pushing to Amazon ECR.
7. **Automated EC2 Deployment via AWS CodeDeploy:** Bundles `appspec.yml` and deployment hooks into S3, triggering zero-downtime container replacement on the target EC2 host.

---

## 🔐 GitHub Secrets Configuration

To run the automated CI/CD pipeline, configure the following secrets in GitHub under **Settings > Secrets and variables > Actions**:

| Secret Name | Description |
| :--- | :--- |
| `AWS_ACCESS_KEY_ID` | IAM credentials with permissions for ECR, S3 deployment, and CodeDeploy |
| `AWS_SECRET_ACCESS_KEY` | Secret access key corresponding to `AWS_ACCESS_KEY_ID` |
| `DAGSHUB_TOKEN` | Personal access token from DagsHub to pull MLflow models & artifacts |

---

## 🛠️ Local Development & Execution

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Aryanupadhyay23/Food-Delivery-Time-prediction.git
cd Food-Delivery-Time-prediction

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install development dependencies
pip install -r dev-requirements.txt
```

### 2. Configure Environment File (`.env.fd`)
Create a `.env.fd` file in the project root:
```env
DAGSHUB_TOKEN="your_dagshub_token_here"
MLFLOW_TRACKING_USERNAME="Aryanupadhyay23"
MLFLOW_TRACKING_PASSWORD="your_dagshub_token_here"
MLFLOW_TRACKING_URI="https://dagshub.com/Aryanupadhyay23/Food-Delivery-Time-prediction.mlflow"
BACKEND_URL="http://localhost:8080"
```

### 3. Run Locally Without Docker
```bash
# Terminal 1: Run FastAPI backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload

# Terminal 2: Run Streamlit frontend
streamlit run frontend/streamlit_app.py --server.port 8510
```

### 4. Run Locally Using Docker
```bash
# Build multi-stage container
docker build -t fooddeliverytimepredictiondeployment .

# Run with environment file and mapped ports
docker run -d \
  --name delivery_pred_container \
  -p 8510:8510 \
  -p 8080:8080 \
  --env-file .env.fd \
  fooddeliverytimepredictiondeployment:latest
```

---

## 🔄 Instant Zero-Downtime Rollback Strategy

If an updated model version exhibits performance anomalies in production:

1. Navigate to the **DagsHub MLflow Model Registry**.
2. Select `FoodDeliveryTimeModel` and reassign the `@production` alias to the previous verified model version (e.g. `v1`).
3. Re-start the Docker container or wait for CodeDeploy's routine restart.
4. The service will immediately pull and serve the restored version **without code changes or rebuilding container images**.

---

## 📜 License & Acknowledgments

- **Dataset**: Provided by Zomato India via Kaggle.
- **License**: Released under the [MIT License](LICENSE).