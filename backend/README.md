# Food Delivery Time AI — Backend Service

Decoupled FastAPI service providing real-time delivery time predictions using a Stacking Ensemble Regressor (LightGBM + Random Forest) logged in DagsHub MLflow Registry.

---

## Architecture

- **Framework:** FastAPI with Uvicorn
- **Endpoints:**
  - `GET /health`: Async health status & model loading verification
  - `GET /model-info`: Async MLflow model metadata & registry aliases
  - `POST /predict`: Async real-time delivery duration prediction (offloaded to threadpool)
  - `POST /predict/batch`: Async batch delivery duration predictions for multi-order dispatch
- **Concurrency:** Fully asynchronous with CPU-bound feature engineering & model inference offloaded to worker threadpools (`asyncio.to_thread`) to guarantee non-blocking event loop execution.
- **CORS:** Enabled for all origins to serve decoupled frontend clients (Streamlit, Web, Mobile).

---

## Running the Backend

### Local Development
From the workspace root:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```
Or from the `backend/` directory:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```
Interactive API documentation will be available at:
- **Swagger UI:** [http://localhost:8080/docs](http://localhost:8080/docs)
- **ReDoc:** [http://localhost:8080/redoc](http://localhost:8080/redoc)

### Docker Container
```bash
docker build -t food-delivery-app .
docker run -p 8080:8080 -p 8510:8510 -e DAGSHUB_TOKEN=$DAGSHUB_TOKEN food-delivery-app
```
