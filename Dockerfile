# Multi-stage Dockerfile for ML Model Serving

# Build stage: install Python dependencies into the user site-packages directory.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed by Python packages with native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch wheels plus the rest of the project dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

# Runtime stage: copy only dependencies and app code needed to serve the API.
FROM python:3.11-slim

LABEL maintainer="xuanan437@example.com"
LABEL version="1.0.0"
LABEL description="ML Model Serving API"

WORKDIR /app

# Install curl for Docker health checks, then create a non-root user.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    groupadd -r appuser && \
    useradd -r -g appuser -m appuser && \
    chown -R appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy the FastAPI application source into the image.
COPY --chown=appuser:appuser src/ ./src/

# Runtime defaults for Python, model selection, and API port.
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_NAME=resnet18 \
    DEVICE=cpu \
    PORT=8000

EXPOSE 8000

# Let Docker mark the container unhealthy if the API stops responding.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser

# Start the FastAPI app with Uvicorn.
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
