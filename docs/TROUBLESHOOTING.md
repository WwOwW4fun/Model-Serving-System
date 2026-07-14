# Troubleshooting

Start with the smallest failing layer: application, container, Kubernetes, monitoring, then CI/CD. Capture the exact command and final error lines before changing configuration.

## Quick diagnostics

Local application:

```bash
curl -i http://localhost:8000/health
curl -i http://localhost:8000/metrics
```

Docker:

```bash
docker compose ps
docker compose logs --tail=200 api
docker inspect --format '{{json .State.Health}}' ml-api
```

Kubernetes:

```bash
kubectl get deployment,pods,service,hpa
kubectl describe deployment ml-model-serving-deployment
kubectl get events --sort-by=.lastTimestamp
kubectl logs deployment/ml-model-serving-deployment --tail=200
```

## Docker daemon is unavailable

Example:

```text
failed to connect to the docker API ... docker.sock ... no such file or directory
```

Docker Desktop is not running or the CLI is using the wrong context. Start Docker Desktop and wait for its engine to become ready:

```bash
docker context ls
docker context use desktop-linux
docker info
```

Docker Desktop is required only for local Docker commands. GitHub Actions uses its own Docker engine.

## Docker build is slow or fails

The image contains PyTorch and torchvision, so the first build downloads large wheels. Later builds should reuse cached layers.

```bash
docker build --progress=plain -t ml-model-serving:v1.0 .
docker system df
```

If disk space is exhausted, inspect usage before deleting caches:

```bash
docker system df
docker builder prune
```

Use `docker system prune` carefully because it removes unused Docker resources.

## Application does not start

View the foreground logs:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

For a container:

```bash
docker logs ml-model-serving
```

Common causes:

- No network access for the initial pretrained-weight download.
- Insufficient memory while importing PyTorch or loading ResNet.
- Port 8000 is already in use.
- A dependency version does not match `requirements.txt`.

Check port usage on macOS or Linux:

```bash
lsof -i :8000
```

Run on a different host port without changing the container:

```bash
docker run --rm -p 8080:8000 ml-model-serving:v1.0
```

## `/health` returns 503

A `503` response means `app.state.model` is not loaded. Review startup logs for the original model-loading error.

```bash
curl -i http://localhost:8000/health
docker logs ml-model-serving
kubectl logs deployment/ml-model-serving-deployment
```

The model is loaded during FastAPI startup. Repeated restarts usually indicate failed weight downloads, memory pressure, or another exception in the startup lifecycle.

## Prediction request fails

Use a JPEG or PNG under 10 MiB and pass `top_k` as a query parameter:

```bash
curl -v -X POST 'http://localhost:8000/predict?top_k=5' \
  -F 'file=@image.jpg;type=image/jpeg'
```

| Response | Likely cause |
| --- | --- |
| `400` | Invalid bytes, unsupported content type, or invalid `top_k`. |
| `413` | Image exceeds 10 MiB. |
| `422` | Multipart `file` field is missing or malformed. |
| `500` | Model inference failed; inspect application logs. |

Do not send `top_k` as a multipart form field; the current endpoint expects `?top_k=N` in the URL.

## Labels appear as `class_123`

The model downloads ImageNet labels during initialization. If that request fails, it intentionally falls back to `class_0` through `class_999`. Predictions still work, but names are generic.

Allow access to the label source or package a local labels file and update the model loader to use it.

## Pod is Pending

Inspect scheduling events:

```bash
kubectl describe pod -l app=ml-model-serving
kubectl top nodes
kubectl get events --sort-by=.lastTimestamp
```

Each pod requests 500 millicores and 1 GiB of memory, and the Deployment starts with two replicas. A small Minikube cluster may not have enough allocatable memory. Increase cluster resources or reduce requests for local testing.

## `ImagePullBackOff` or `ErrImagePull`

The default Deployment refers to the local image `ml-model-serving:v1.0` with `IfNotPresent`.

For Minikube:

```bash
docker build -t ml-model-serving:v1.0 .
minikube image load ml-model-serving:v1.0
kubectl rollout restart deployment/ml-model-serving-deployment
```

For a registry image, use its full reference:

```bash
kubectl set image deployment/ml-model-serving-deployment \
  ml-api=ghcr.io/wwoww4fun/ml-model-serving:latest
```

Private GHCR images also require a Kubernetes image pull secret.

## `CrashLoopBackOff` or failing probes

Inspect the current and previous container logs:

```bash
kubectl logs deployment/ml-model-serving-deployment --tail=200
kubectl logs <pod-name> --previous
kubectl describe pod <pod-name>
```

Readiness starts after 30 seconds and liveness starts after 60 seconds. A slow first-time model download can exceed these windows. Confirm startup time before increasing probe delays. If the container is `OOMKilled`, increase available memory or reduce replica/resource requirements.

## LoadBalancer external IP remains pending

Local clusters normally do not provision cloud load balancers. Use Minikube's service helper or port forwarding:

```bash
minikube service ml-model-serving-service --url
```

or:

```bash
kubectl port-forward service/ml-model-serving-service 8000:80
```

## HPA shows unknown metrics

The HPA depends on Kubernetes Metrics Server and resource requests.

```bash
minikube addons enable metrics-server
kubectl top pods
kubectl describe hpa ml-model-serving-hpa
```

Wait several collection intervals after pod startup. If `kubectl top pods` fails, repair Metrics Server before debugging the HPA.

## Prometheus target is down

Confirm the application exports metrics:

```bash
kubectl port-forward service/ml-model-serving-service 8000:80
curl http://localhost:8000/metrics
```

For Prometheus Operator:

```bash
kubectl get servicemonitor ml-model-serving-monitor -o yaml
kubectl get service ml-model-serving-service --show-labels
```

The ServiceMonitor selector must match the Service labels, its endpoint port must be named `http`, and the `release: monitoring` label must match the Prometheus installation selector.

For Docker Compose, verify that Prometheus can reach `api:8000` on the Compose network:

```bash
docker compose logs prometheus
docker compose exec prometheus wget -qO- http://api:8000/metrics
```

## Grafana displays no data

1. Open Prometheus at `http://localhost:9090` and query `api_requests_total`.
2. Confirm the Grafana datasource URL is `http://prometheus:9090` inside Docker Compose.
3. Generate traffic with `/health` or `/predict`.
4. Select a recent dashboard time range.

The repository provisions the datasource and dashboard provider, but it does not yet include a finished dashboard JSON file.

## GitHub Actions fails

Open the failed job and expand the first red step. The pipeline is ordered, so later jobs are skipped when a required job fails.

Common cases:

- **Formatting or linting:** run Black, isort, and flake8 locally.
- **Tests or coverage:** run pytest and inspect the terminal coverage report.
- **GHCR push:** verify workflow `packages: write` permission.
- **Trivy SARIF upload:** optional and configured not to fail the Docker build.
- **Kubernetes skipped:** set `ENABLE_DEPLOYMENT=true` only after adding `KUBECONFIG` for a reachable cluster.
- **Slack skipped:** set `ENABLE_SLACK_NOTIFICATIONS=true` and add `SLACK_WEBHOOK_URL`.

Local CI-equivalent commands:

```bash
./venv/bin/isort --check-only src tests
./venv/bin/black --check src tests
./venv/bin/flake8 src tests --max-line-length=100
./venv/bin/pytest
```

## Requesting help

Include the following when reporting a problem:

- Exact command that failed
- Complete final error block
- `docker version` or `kubectl version`
- `docker compose ps` or `kubectl get pods -o wide`
- Relevant container or pod logs
- Commit SHA and GitHub Actions run URL for CI failures
