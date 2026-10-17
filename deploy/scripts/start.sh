#!/bin/bash
set -e

FASTAPI_PORT=${FASTAPI_PORT:-8080}
STREAMLIT_PORT=${STREAMLIT_PORT:-8510}

echo "Starting FastAPI Backend on port ${FASTAPI_PORT}..."
uvicorn backend.main:app --host 0.0.0.0 --port ${FASTAPI_PORT} &

echo "Waiting for FastAPI Backend to be ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:${FASTAPI_PORT}/health | grep healthy > /dev/null 2>&1; then
        echo "FastAPI Backend is healthy!"
        break
    fi
    sleep 1
done

echo "Starting Streamlit Frontend on port ${STREAMLIT_PORT}..."
exec streamlit run frontend/streamlit_app.py \
    --server.port=${STREAMLIT_PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false

