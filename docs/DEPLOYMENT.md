# Deployment guide

This guide covers direct development, Docker Compose, local Kubernetes, and the optional GitHub Actions deployment path.

## Prerequisites

- Python 3.10–3.12 for local development
- Docker Desktop or another Docker engine
- `kubectl` and a Kubernetes cluster for Kubernetes deployment
- Minikube and Metrics Server for local autoscaling tests
- Helm and Prometheus Operator only when using the ServiceMonitor

The first real-model startup may download pretrained ResNet weights and ImageNet class labels. Allow outbound internet access or pre-populate the model cache.

## Run with Python

Create and activate a virtual environment, then install dependencies:

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

Verify it:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

## Run with Docker

Ensure the Docker daemon is running, then build the image:

```bash
docker build -t ml-model-serving:v1.0 .
```

Start a container:

```bash
docker run --rm --name ml-model-serving \
  -p 8000:8000 \
  ml-model-serving:v1.0
```

Check container health and the API:

```bash
docker ps
curl http://localhost:8000/health
```

Stop it with `Ctrl+C` or, from another terminal:

```bash
docker stop ml-model-serving
```

## Run the complete local stack

Docker Compose starts the API, Prometheus, and Grafana:

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

Services:

| Service | URL | Default credentials |
| --- | --- | --- |
| API | `http://localhost:8000` | None |
| Prometheus | `http://localhost:9090` | None |
| Grafana | `http://localhost:3000` | `admin` / `admin` |

Change the Grafana password through `GRAFANA_PASSWORD` before using the stack outside local development.

Stop the stack while retaining named volumes:

```bash
docker compose down
```

Remove the stack and stored Prometheus/Grafana data:

```bash
docker compose down --volumes
```

## Deploy to Minikube

Start a cluster with enough memory for two model-serving pods:

```bash
minikube start --cpus=4 --memory=6144
minikube addons enable metrics-server
kubectl get nodes
```

Build and load the exact image name referenced by the Deployment:

```bash
docker build -t ml-model-serving:v1.0 .
minikube image load ml-model-serving:v1.0
```

Apply the core resources in dependency order:

```bash
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml
```

Wait for the rollout and inspect resources:

```bash
kubectl rollout status deployment/ml-model-serving-deployment
kubectl get pods -l app=ml-model-serving
kubectl get service ml-model-serving-service
kubectl get hpa ml-model-serving-hpa
```

Access the API using either method:

```bash
minikube service ml-model-serving-service --url
```

or:

```bash
kubectl port-forward service/ml-model-serving-service 8000:80
```

Then test it from another terminal:

```bash
curl http://localhost:8000/health
curl -X POST 'http://localhost:8000/predict?top_k=5' \
  -F 'file=@test-image.jpg;type=image/jpeg'
```

## Add Kubernetes monitoring

Install a Prometheus Operator stack if one is not already present:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

Apply the ServiceMonitor in the same namespace as the API Service:

```bash
kubectl apply -f kubernetes/servicemonitor.yaml
kubectl get servicemonitor ml-model-serving-monitor
```

The ServiceMonitor selects `ml-model-serving-service` by labels and scrapes its named `http` port at `/metrics` every 15 seconds. Its `release: monitoring` label must match the Prometheus installation's ServiceMonitor selector.

Access Grafana from the Helm installation:

```bash
kubectl port-forward -n monitoring service/monitoring-grafana 3000:80
```

## Use the image published by CI

Pushes to `main` build and publish images similar to:

```text
ghcr.io/wwoww4fun/ml-model-serving:main
ghcr.io/wwoww4fun/ml-model-serving:sha-<commit>
ghcr.io/wwoww4fun/ml-model-serving:latest
```

Update the Deployment before applying it to a cluster:

```bash
kubectl set image deployment/ml-model-serving-deployment \
  ml-api=ghcr.io/wwoww4fun/ml-model-serving:latest
kubectl rollout status deployment/ml-model-serving-deployment
```

For a private package, configure an image pull secret and reference it with `imagePullSecrets` in the pod specification.

## Enable deployment from GitHub Actions

Automatic deployment is disabled by default. It requires a Kubernetes API endpoint reachable from a GitHub-hosted runner; a Minikube cluster on a laptop is normally not reachable.

In repository **Settings → Secrets and variables → Actions**:

1. Add the secret `KUBECONFIG` containing a base64-encoded kubeconfig.
2. Add the variable `ENABLE_DEPLOYMENT=true`.
3. Ensure the kubeconfig context deploys to the `default` namespace, which the workflow currently uses.

On macOS, produce the secret value with:

```bash
base64 < ~/.kube/config | pbcopy
```

The next push to `main` will build the image, update the Deployment, wait for the rollout, call `/health`, and roll back if the rollout or smoke test fails.

## Rollback and cleanup

Inspect rollout history and undo the most recent Deployment update:

```bash
kubectl rollout history deployment/ml-model-serving-deployment
kubectl rollout undo deployment/ml-model-serving-deployment
kubectl rollout status deployment/ml-model-serving-deployment
```

Remove project resources:

```bash
kubectl delete -f kubernetes/servicemonitor.yaml --ignore-not-found
kubectl delete -f kubernetes/hpa.yaml --ignore-not-found
kubectl delete -f kubernetes/service.yaml --ignore-not-found
kubectl delete -f kubernetes/deployment.yaml --ignore-not-found
kubectl delete -f kubernetes/configmap.yaml --ignore-not-found
```
