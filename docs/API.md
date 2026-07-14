# API reference

The service exposes a small HTTP API for ImageNet image classification, health checks, and Prometheus metrics. FastAPI also generates an OpenAPI schema and interactive documentation.

## Base URL

For local development:

```text
http://localhost:8000
```

When deployed behind the Kubernetes `LoadBalancer` service, replace the host with the service address.

The API does not currently require authentication. Add authentication and TLS before exposing it to an untrusted network.

## Quick example

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
    },
    {
      "class_id": 282,
      "class_name": "tiger cat",
      "confidence": 0.1835
    }
  ],
  "inference_time_ms": 42.7,
  "model_version": "1.0.0"
}
```

Prediction values depend on the uploaded image and loaded model.

## Endpoints

### `GET /`

Returns basic service information.

```bash
curl http://localhost:8000/
```

```json
{
  "name": "ML Model Serving API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs"
}
```

### `GET /health`

Reports whether the API has a model available for inference.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 38.42,
  "device": "cpu"
}
```

Responses:

| Status | Meaning |
| --- | --- |
| `200 OK` | The API is running and a model object is loaded. |
| `503 Service Unavailable` | No model is loaded. |

Docker and Kubernetes use this endpoint for health, readiness, and liveness checks.

### `POST /predict`

Classifies one uploaded JPEG or PNG image.

Request:

- Content type: `multipart/form-data`
- `file`: required image file
- `top_k`: optional query parameter from `1` to `10`; default is `5`
- Maximum file size: 10 MiB

```bash
curl -X POST 'http://localhost:8000/predict?top_k=3' \
  -F 'file=@dog.png;type=image/png'
```

Python example:

```python
import requests

with open("dog.png", "rb") as image:
    response = requests.post(
        "http://localhost:8000/predict",
        params={"top_k": 3},
        files={"file": ("dog.png", image, "image/png")},
        timeout=30,
    )

response.raise_for_status()
print(response.json())
```

Each prediction contains:

| Field | Type | Description |
| --- | --- | --- |
| `class_id` | integer | ImageNet class index. |
| `class_name` | string | ImageNet label, or a fallback `class_N` label if labels could not be downloaded. |
| `confidence` | number | Softmax confidence between `0` and `1`. |

The response also includes inference duration in milliseconds and the application model version.

Common errors:

| Status | Cause |
| --- | --- |
| `400 Bad Request` | Unsupported media type, invalid image bytes, or `top_k` outside `1–10`. |
| `413 Payload Too Large` | Uploaded file exceeds 10 MiB. |
| `422 Unprocessable Entity` | Required multipart `file` field is missing. |
| `500 Internal Server Error` | Model inference failed. |

Error responses use FastAPI's standard format:

```json
{
  "detail": "Invalid image format"
}
```

### `GET /metrics`

Returns Prometheus text-format metrics.

```bash
curl http://localhost:8000/metrics
```

Application metrics include:

| Metric | Type | Description |
| --- | --- | --- |
| `api_requests_total` | Counter | Requests grouped by endpoint, method, and status code. |
| `api_request_duration_seconds` | Histogram | HTTP request duration. |
| `api_predictions_total` | Counter | Successful predictions. |
| `api_errors_total` | Counter | Prediction errors grouped by error type. |

The Python Prometheus client also exposes process and runtime metrics.

## Interactive documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI schema: `http://localhost:8000/openapi.json`

## Current limitations

- No authentication, authorization, or rate limiting is enforced.
- Only one image can be classified per request.
- JPEG and PNG are the only accepted upload formats.
- The first startup requires network access if pretrained weights are not already cached.
- Class labels fall back to generic names if the ImageNet label download fails.
