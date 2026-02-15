FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if required by CatBoost / MLflow)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only inference layer
COPY inference_app ./inference_app
COPY docker-requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r docker-requirements.txt

EXPOSE 8000

CMD ["uvicorn", "inference_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
