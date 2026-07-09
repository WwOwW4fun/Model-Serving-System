"""
Tests for ML Model Serving API.
"""

import concurrent.futures
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image


class FakeModel:
    """Small deterministic stand-in for the real ML model."""

    device = "cpu"

    def predict(self, image, top_k: int = 5):
        return [
            {
                "class_id": i,
                "class_name": f"class_{i}",
                "confidence": round(1.0 - (i * 0.05), 4),
            }
            for i in range(top_k)
        ]


@pytest.fixture
def client():
    from src.api import app

    app.state.model = FakeModel()
    app.state.start_time = time.time()
    return TestClient(app)


@pytest.fixture
def test_settings():
    try:
        from src.config import Settings

        return Settings(environment="development", model_name="resnet18", device="cpu")
    except Exception as exc:
        pytest.skip(f"Settings cannot be created yet: {exc}")


@pytest.fixture
def sample_image():
    image = Image.new("RGB", (224, 224), color="red")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    return img_bytes.getvalue()


@pytest.fixture
def invalid_image():
    return b"This is not a valid image"


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "ML Model Serving API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"
    assert data["docs"] == "/docs"


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "unhealthy"]
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"


def test_health_check_response_format(client):
    response = client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert isinstance(data["status"], str)
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["version"], str)
    assert isinstance(data["uptime_seconds"], (int, float))


def test_predict_with_valid_image(client, sample_image):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) > 0

    for pred in data["predictions"]:
        assert set(pred) == {"class_id", "class_name", "confidence"}
        assert isinstance(pred["class_id"], int)
        assert isinstance(pred["class_name"], str)
        assert 0 <= pred["confidence"] <= 1


def test_predict_with_invalid_image(client, invalid_image):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", invalid_image, "image/jpeg")},
    )

    assert response.status_code in [400, 422, 500]
    assert "detail" in response.json()


def test_predict_with_wrong_content_type(client, sample_image):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", sample_image, "text/plain")},
    )

    assert response.status_code in [400, 422]
    assert "detail" in response.json()


def test_predict_without_file(client):
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_with_top_k_parameter(client, sample_image):
    response = client.post(
        "/predict?top_k=3",
        files={"file": ("test.jpg", sample_image, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 3

    confidences = [p["confidence"] for p in data["predictions"]]
    assert confidences == sorted(confidences, reverse=True)


def test_predict_response_includes_metadata(client, sample_image):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["predictions"], list)
    assert isinstance(data["inference_time_ms"], (int, float))
    assert isinstance(data["model_version"], str)


def test_predict_with_large_file(client):
    large_payload = b"x" * (11 * 1024 * 1024)
    response = client.post(
        "/predict",
        files={"file": ("large.jpg", large_payload, "image/jpeg")},
    )

    assert response.status_code in [400, 413, 500]


def test_predict_with_invalid_top_k(client, sample_image):
    for top_k in [0, -1, 10000]:
        response = client.post(
            f"/predict?top_k={top_k}",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )
        assert response.status_code in [400, 422, 500]


def test_404_for_nonexistent_endpoint(client):
    response = client.get("/nonexistent")
    assert response.status_code == 404


def test_method_not_allowed(client):
    response = client.get("/predict")
    assert response.status_code == 405


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "# HELP" in response.text
    assert "api_requests_total" in response.text


def test_metrics_after_prediction(client, sample_image):
    client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image, "image/jpeg")},
    )
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "api_predictions_total" in response.text


def test_openapi_schema(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema
    assert "/predict" in schema["paths"]


def test_docs_endpoint(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_prediction_latency(client, sample_image):
    start = time.time()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image, "image/jpeg")},
    )
    latency_ms = (time.time() - start) * 1000

    assert response.status_code == 200
    assert latency_ms < 5000
    assert response.json()["inference_time_ms"] <= latency_ms


def test_concurrent_predictions(client, sample_image):
    def make_prediction():
        return client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        responses = list(executor.map(lambda _: make_prediction(), range(5)))

    for response in responses:
        assert response.status_code == 200


def test_full_prediction_pipeline(client, sample_image):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    pred = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image, "image/jpeg")},
    )
    assert pred.status_code == 200
    assert len(pred.json()["predictions"]) > 0

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "api_predictions_total" in metrics.text


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield
