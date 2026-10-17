# Stage 1: Builder - Build Python Dependencies
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install compiler tools for packages requiring C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies into virtualenv
COPY docker-requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r docker-requirements.txt && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete || true && \
    find /opt/venv -type f -name "*.pyo" -delete || true && \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Stage 2: Minimal Runtime - Ultra-Slim Production Image
FROM python:3.12-slim AS runtime

# Create non-root user for production container security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Configure Python, FastAPI, and Streamlit runtime environments
# Dedicated ports avoiding conflicts with other apps:
# FastAPI backend: 8080 | Streamlit frontend: 8510
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTAPI_PORT=8080 \
    STREAMLIT_PORT=8510 \
    STREAMLIT_SERVER_PORT=8510 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    BACKEND_URL=http://localhost:8080

# Install only curl for healthcheck, clean package cache immediately
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/*

# Copy pre-compiled virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy runtime inference serving layer & frontend application with proper ownership
COPY --chown=appuser:appuser backend/main.py ./backend/main.py
COPY --chown=appuser:appuser backend/utility_files ./backend/utility_files
COPY --chown=appuser:appuser frontend ./frontend
COPY --chown=appuser:appuser deploy/scripts/start.sh ./start.sh
RUN chmod +x ./start.sh

# Run as non-root user
USER appuser

# Expose dedicated non-conflicting ports: Streamlit UI (8510) and FastAPI API (8080)
EXPOSE 8080 8510

# Container healthcheck for AWS EC2 self-healing & Docker status
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["./start.sh"]
