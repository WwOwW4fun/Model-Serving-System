# Architecture

## Overview

This project serves a pretrained ImageNet classifier through FastAPI. The application can run directly with Python, in Docker, or as replicated pods in Kubernetes. Prometheus collects application metrics, Grafana provides visualization infrastructure, and GitHub Actions validates and publishes each main-branch build.

## System context

```text
Client
  │ HTTP: /predict, /health, /metrics
  ▼
Kubernetes LoadBalancer Service :80
  │
  ├──────────────┐
  ▼              ▼
FastAPI Pod   FastAPI Pod       ◀── Horizontal Pod Autoscaler
  │              │
  └──────┬───────┘
         │ /metrics every 15s
         ▼
    Prometheus ──────▶ Grafana
```

For local development, Docker Compose provides the same API, Prometheus, and Grafana services on one Docker network.

## Application components

### API layer

[`src/api.py`](../src/api.py) owns the HTTP interface and application lifecycle.

- Loads the model before the application begins serving traffic.
- Implements `/`, `/predict`, `/health`, and `/metrics`.
- Validates upload type, size, and `top_k`.
- Records request counts, duration, successful predictions, and errors.
- Releases model and CUDA resources during shutdown.

FastAPI stores the loaded `ModelInference` instance in `app.state.model`. Tests replace it with a deterministic fake model so API behavior can be validated without downloading model weights.

### Model layer

[`src/model.py`](../src/model.py) wraps torchvision models and inference behavior.

1. Create a ResNet model and load pretrained weights.
2. Move it to the selected CPU or CUDA device.
3. Warm it with one dummy tensor.
4. Decode uploads with Pillow.
5. Resize, center-crop, and normalize the image for ImageNet.
6. Run inference without gradient tracking.
7. Apply softmax and return the highest-confidence classes.

ResNet18 is used by the application startup path. ResNet50 is also supported by the model wrapper. Pretrained weights and ImageNet labels are downloaded at runtime when they are not already available; generic class names are used if only the label download fails.

### Configuration

[`src/config.py`](../src/config.py) defines validated settings using Pydantic Settings. Values can come from environment variables or `.env`, with defaults suitable for local CPU execution.

Important settings include the environment, ports, model name, device, upload limit, logging, cache, metrics, and rate-limit feature flags. Kubernetes supplies selected values through [`kubernetes/configmap.yaml`](../kubernetes/configmap.yaml).

The current API startup path uses explicit ResNet18 and CPU defaults; not every available `Settings` field is wired into runtime behavior yet.

### Shared utilities

[`src/utils.py`](../src/utils.py) contains reusable image validation, preprocessing, logging, prediction post-processing, latency statistics, caching, error mapping, model health, and system information helpers.

## Request flow

```text
POST /predict?top_k=5
        │
        ▼
Validate content type, size, and top_k
        │
        ▼
Decode image → RGB → resize → center crop → normalize
        │
        ▼
ResNet forward pass → softmax → top-k classes
        │
        ▼
JSON response + Prometheus counters and latency
```

Invalid client input returns a `4xx` response. Unexpected inference failures return `500` and increment `api_errors_total`.

## Health and availability

The `/health` endpoint returns `200` when the API has a model object and `503` otherwise.

- Docker checks it every 30 seconds after a 40-second startup period.
- Kubernetes readiness checks remove an unhealthy pod from Service traffic.
- Kubernetes liveness checks restart a repeatedly unhealthy container.
- The optional deployment workflow calls `/health` after a rollout and rolls back on failure.

The endpoint is intentionally lightweight; it does not run an inference for every probe.

## Kubernetes topology

The manifests use the current kubectl namespace unless the caller supplies `-n`.

| Resource | Name | Role |
| --- | --- | --- |
| Deployment | `ml-model-serving-deployment` | Runs two API replicas with rolling updates. |
| Service | `ml-model-serving-service` | Exposes port 80 and forwards to container port 8000. |
| ConfigMap | `ml-model-config` | Stores non-secret runtime settings. |
| HPA | `ml-model-serving-hpa` | Scales from 2 to 10 replicas using CPU and memory. |
| ServiceMonitor | `ml-model-serving-monitor` | Tells Prometheus Operator to scrape `/metrics`. |

Each pod requests 500 millicores and 1 GiB of memory and is limited to 1 CPU and 2 GiB.

## Observability

Prometheus scrapes the API every 15 seconds. Grafana is provisioned with Prometheus as its default datasource. The repository currently supplies Grafana provisioning configuration but not a finished dashboard JSON file; dashboards can be created through the Grafana UI or added under `monitoring/grafana/dashboards`.

The ServiceMonitor path requires Prometheus Operator, such as the one installed by `kube-prometheus-stack`. For Docker Compose, Prometheus uses the static target `api:8000`.

## CI/CD flow

```text
Push or pull request
    ▼
Black + isort + flake8
    ▼
pytest on Python 3.10, 3.11, and 3.12
    ▼
Build, scan, and push image to GHCR
    ▼
Optional Kubernetes deployment
    ▼
Optional Slack notification
```

Kubernetes deployment and Slack notification are feature-gated with GitHub repository variables. They remain skipped unless their external services and secrets are configured.

## Security boundaries

The container runs as a non-root user, uploads have type and size checks, dependencies are pinned, and CI scans the built image with Trivy. The current service still needs TLS, authentication, rate limiting, restrictive CORS, network policies, and secret management before public production use.
