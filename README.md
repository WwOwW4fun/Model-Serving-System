# Basic Model Serving System

[![CI/CD Pipeline](https://github.com/WwOwW4fun/Model-Serving-System/actions/workflows/ci-cd.yaml/badge.svg)](https://github.com/WwOwW4fun/Model-Serving-System/actions/workflows/ci-cd.yaml)

A small end-to-end machine-learning serving system that exposes a pretrained PyTorch image classifier through FastAPI. The project includes Docker packaging, Kubernetes manifests, Prometheus metrics, Grafana provisioning, automated tests, and a GitHub Actions pipeline that publishes images to GitHub Container Registry.

## Features

- ImageNet classification with pretrained ResNet18
- Top-k prediction responses with confidence scores
- FastAPI OpenAPI, Swagger UI, and ReDoc
- Health, readiness, liveness, and Docker health checks
- Prometheus request, latency, prediction, and error metrics
- Multi-stage Docker image running as a non-root user
- Replicated Kubernetes Deployment, LoadBalancer Service, ConfigMap, and HPA
- Prometheus Operator ServiceMonitor
- Tests across Python 3.10, 3.11, and 3.12
- GitHub Actions linting, testing, image scanning, and GHCR publishing
- Optional Kubernetes deployment and Slack notifications

## Architecture

```text
Client
  │ HTTP
  ▼
LoadBalancer Service :80
  │
  ├──────────────┐
  ▼              ▼
FastAPI Pod   FastAPI Pod       ◀── Horizontal Pod Autoscaler
  │              │
  └──────┬───────┘
         │ /metrics
         ▼
    Prometheus ──────▶ Grafana
```

Each API process loads the model into memory during startup. Prediction requests are decoded, resized, normalized, passed through ResNet, and returned as ranked ImageNet classes. See [Architecture](docs/ARCHITECTURE.md) for component and request-flow details.

## Technology stack

| Area | Technology |
| --- | --- |
| API | Python, FastAPI, Uvicorn, Pydantic |
| ML | PyTorch, torchvision, Pillow |
| Packaging | Docker, Docker Compose |
| Orchestration | Kubernetes, Horizontal Pod Autoscaler |
| Monitoring | Prometheus, Grafana |
| Automation | GitHub Actions, GHCR, Trivy |
| Testing | pytest, pytest-cov |

## Quick start

### Run with Python

Create a virtual environment and install the pinned dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The first real-model startup may download pretrained ResNet weights and ImageNet labels.

Verify the service:

```bash
curl http://localhost:8000/health
```

### Run with Docker

Start Docker Desktop or another Docker engine, then run:

```bash
docker build -t ml-model-serving:v1.0 .
docker run --rm -p 8000:8000 ml-model-serving:v1.0
```

### Run the local monitoring stack

Docker Compose starts the API, Prometheus, and Grafana:

```bash
docker compose up --build -d
docker compose ps
```

| Service | URL |
| --- | --- |
| API | <http://localhost:8000> |
| Swagger UI | <http://localhost:8000/docs> |
| Prometheus | <http://localhost:9090> |
| Grafana | <http://localhost:3000> |

Grafana uses `admin` / `admin` by default for local development. Override `GRAFANA_PASSWORD` before sharing the environment.

## Make a prediction

Upload a JPEG or PNG smaller than 10 MiB. `top_k` accepts values from 1 to 10 and defaults to 5.

```bash
curl -X POST 'http://localhost:8000/predict?top_k=5' \
  -F 'file=@cat.jpg;type=image/jpeg'
```

Example response:

```json
{
  "predictions": [
    {
      "class_id": 281,
      "class_name": "tabby",
      "confidence": 0.7431
    }
  ],
  "inference_time_ms": 42.7,
  "model_version": "1.0.0"
}
```

See the [API reference](docs/API.md) for all endpoints, response fields, and status codes.

## Testing

Run the full suite:

```bash
./venv/bin/pytest
```

Run the same quality checks used by CI:

```bash
./venv/bin/isort --check-only src tests
./venv/bin/black --check src tests
./venv/bin/flake8 src tests --max-line-length=100
./venv/bin/pytest
```

The current suite covers the API, configuration, model pipeline, image utilities, cache behavior, health checks, and error handling. Expensive model downloads are replaced with deterministic test doubles.

## Kubernetes

The Kubernetes manifests create:

- Two API replicas with rolling updates
- A LoadBalancer Service on port 80
- Readiness and liveness probes on `/health`
- CPU and memory requests and limits
- An HPA that scales from 2 to 10 replicas
- A ServiceMonitor for Prometheus Operator

For a local Minikube deployment:

```bash
minikube start --cpus=4 --memory=6144
minikube addons enable metrics-server

docker build -t ml-model-serving:v1.0 .
minikube image load ml-model-serving:v1.0

kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

kubectl rollout status deployment/ml-model-serving-deployment
minikube service ml-model-serving-service --url
```

See the [Deployment guide](docs/DEPLOYMENT.md) for Docker Compose, monitoring, GHCR, automated deployment, rollback, and cleanup instructions.

## CI/CD

The workflow in [`.github/workflows/ci-cd.yaml`](.github/workflows/ci-cd.yaml) runs this sequence:

```text
Code quality → Test matrix → Build and scan image → Push to GHCR
                                                    ├─ Optional Kubernetes deploy
                                                    └─ Optional Slack notification
```

Kubernetes deployment and Slack notifications remain disabled unless their repository variables and secrets are configured. The required validation, testing, Docker build, security scan, and registry publishing stages run independently on GitHub-hosted runners.

Published images use tags such as:

```text
ghcr.io/wwoww4fun/ml-model-serving:main
ghcr.io/wwoww4fun/ml-model-serving:sha-<commit>
ghcr.io/wwoww4fun/ml-model-serving:latest
```

## Project structure

```text
.
├── .github/workflows/ci-cd.yaml  # CI/CD pipeline
├── docs/                         # Project documentation
├── kubernetes/                   # Deployment, Service, ConfigMap, HPA, monitor
├── monitoring/                   # Prometheus and Grafana provisioning
├── src/
│   ├── api.py                    # FastAPI routes and metrics
│   ├── config.py                 # Validated application settings
│   ├── model.py                  # Model loading and inference
│   └── utils.py                  # Shared serving utilities
├── tests/                        # API, model, configuration, and utility tests
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
└── requirements.txt
```

## Documentation

- [API reference](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Deployment guide](docs/DEPLOYMENT.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Current limitations

- The public API has no authentication, authorization, or TLS termination.
- Runtime startup may require internet access for model weights and labels.
- Only single-image JPEG and PNG requests are supported.
- The Grafana datasource and dashboard provider are provisioned, but a finished dashboard JSON is not included yet.
- Automated Kubernetes deployment requires a remotely reachable cluster; laptop Minikube is not reachable from GitHub-hosted runners.
- Performance and availability targets have not been validated under sustained production load.

This repository is intended as a practical model-serving reference and learning project. Additional production hardening should be completed before exposing it to untrusted traffic.
