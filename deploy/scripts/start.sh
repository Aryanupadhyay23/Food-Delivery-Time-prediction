#!/bin/bash
set -e

# Disable joblib/loky background process spawning to conserve memory
export JOBLIB_MULTIPROCESSING=0
export LOKY_MAX_CPU_COUNT=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_TRIM_THRESHOLD_=65536

# Enable jemalloc if installed in the container
if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
fi

FASTAPI_PORT=${FASTAPI_PORT:-8080}
STREAMLIT_PORT=${STREAMLIT_PORT:-8510}

echo "Starting FastAPI Backend on port ${FASTAPI_PORT}..."
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "${FASTAPI_PORT}" \
    --root-path /food-delivery-backend &

echo "Waiting for FastAPI Backend to be ready..."
BACKEND_READY=false

for i in {1..30}; do
    if curl -fsS "http://127.0.0.1:${FASTAPI_PORT}/health" | grep -q healthy; then
        echo "FastAPI Backend is healthy!"
        BACKEND_READY=true
        break
    fi
    sleep 1
done

if [ "${BACKEND_READY}" != "true" ]; then
    echo "ERROR: FastAPI backend failed to become healthy."
    exit 1
fi

echo "Starting Streamlit Frontend on port ${STREAMLIT_PORT}..."
exec streamlit run frontend/streamlit_app.py \
    --server.port="${STREAMLIT_PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.baseUrlPath=food-delivery \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false




# #!/bin/bash
# set -e
#
# # Disable joblib/loky background process spawning to conserve memory (~340 MB savings)
# export JOBLIB_MULTIPROCESSING=0
# export LOKY_MAX_CPU_COUNT=1
# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export MALLOC_TRIM_THRESHOLD_=65536
#
# # Enable jemalloc if installed in the container
# if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
#     export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
# fi
#
# FASTAPI_PORT=${FASTAPI_PORT:-8080}
# STREAMLIT_PORT=${STREAMLIT_PORT:-8510}
#
# echo "Starting FastAPI Backend on port ${FASTAPI_PORT}..."
# uvicorn backend.main:app --host 0.0.0.0 --port ${FASTAPI_PORT} &
#
# echo "Waiting for FastAPI Backend to be ready..."
# for i in {1..30}; do
#     if curl -s http://127.0.0.1:${FASTAPI_PORT}/health | grep healthy > /dev/null 2>&1; then
#         echo "FastAPI Backend is healthy!"
#         break
#     fi
#     sleep 1
# done
#
# echo "Starting Streamlit Frontend on port ${STREAMLIT_PORT}..."
# exec streamlit run frontend/streamlit_app.py \
#     --server.port=${STREAMLIT_PORT} \
#     --server.address=0.0.0.0 \
#     --server.headless=true \
#     --server.fileWatcherType=none \
#     --browser.gatherUsageStats=false