"""Tests for shared model-serving utilities."""

import io
import json
import logging

import pytest
import torch
from PIL import Image

from src import utils


def image_bytes(
    image_format: str = "JPEG", size: tuple[int, int] = (32, 24), mode: str = "RGB"
) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=1).save(buffer, format=image_format)
    return buffer.getvalue()


def test_setup_logging_supports_json_and_text():
    json_logger = utils.setup_logging("info", json_format=True)
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
    payload = json.loads(json_logger.handlers[0].formatter.format(record))

    assert json_logger.level == logging.INFO
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"

    text_logger = utils.setup_logging("DEBUG", json_format=False)
    assert "hello" in text_logger.handlers[0].formatter.format(record)

    with pytest.raises(ValueError):
        utils.setup_logging("LOUD")


def test_validate_image_file():
    assert utils.validate_image_file(image_bytes()) == (True, "")
    assert not utils.validate_image_file(b"")[0]
    assert not utils.validate_image_file(b"not-an-image")[0]
    assert not utils.validate_image_file(image_bytes("GIF"))[0]
    assert not utils.validate_image_file(image_bytes("PNG"), max_size_mb=0)[0]
    assert not utils.validate_image_file(b"x" * (1024 * 1024 + 1), max_size_mb=1)[0]
    assert not utils.validate_image_file(image_bytes("PNG", (4097, 1)))[0]


def test_load_resize_and_preprocess_image():
    image = utils.load_image_from_bytes(image_bytes("PNG", mode="L"))
    resized = utils.resize_image(image, (16, 16))
    tensor = utils.preprocess_image(resized)

    assert image.mode == "RGB"
    assert resized.size == (16, 16)
    assert tensor.shape == (1, 3, 16, 16)
    assert tensor.dtype == torch.float32

    with pytest.raises(ValueError):
        utils.load_image_from_bytes(b"")
    with pytest.raises(ValueError):
        utils.load_image_from_bytes(b"invalid")
    with pytest.raises(ValueError):
        utils.resize_image(image, (0, 1))
    with pytest.raises(ValueError):
        utils.preprocess_image(image, std=[1.0, 1.0])


def test_model_path_and_class_labels(tmp_path):
    assert utils.get_model_path("resnet18") == utils.Path("models/resnet18.pth")
    with pytest.raises(ValueError):
        utils.get_model_path("../model")

    text_file = tmp_path / "labels.txt"
    text_file.write_text("cat\n\ndog\n", encoding="utf-8")
    assert utils.load_class_labels(str(text_file)) == ["cat", "dog"]

    json_file = tmp_path / "labels.json"
    json_file.write_text('{"1": "dog", "0": "cat"}', encoding="utf-8")
    assert utils.load_class_labels(str(json_file)) == ["cat", "dog"]

    list_file = tmp_path / "list.json"
    list_file.write_text('["cat", "dog"]', encoding="utf-8")
    assert utils.load_class_labels(str(list_file)) == ["cat", "dog"]

    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text('"cat"', encoding="utf-8")
    with pytest.raises(ValueError):
        utils.load_class_labels(str(invalid_file))

    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    with pytest.raises(ValueError):
        utils.load_class_labels(str(empty_file))
    with pytest.raises(FileNotFoundError):
        utils.load_class_labels(str(tmp_path / "missing.txt"))


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return torch.zeros(batch.shape[0], 2, device=batch.device)


def test_model_warmup_and_health():
    model = CountingModel()
    utils.warm_up_model(model, torch.device("cpu"), num_iterations=2)
    health = utils.check_model_health(model, torch.device("cpu"))

    assert model.calls == 3
    assert health["healthy"]
    assert health["test_inference_ms"] >= 0

    with pytest.raises(ValueError):
        utils.warm_up_model(model, torch.device("cpu"), num_iterations=-1)

    assert not utils.check_model_health(None, torch.device("cpu"))["healthy"]


def test_model_health_reports_unavailable_device_and_errors(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    unavailable = utils.check_model_health(CountingModel(), torch.device("cuda"))

    class BrokenModel(torch.nn.Module):
        def forward(self, batch):
            raise RuntimeError("broken")

    broken = utils.check_model_health(BrokenModel(), torch.device("cpu"))

    assert unavailable["error"] == "device is unavailable: cuda"
    assert broken["error"] == "broken"


def test_top_predictions():
    predictions = utils.get_top_predictions(
        torch.tensor([[1.0, 3.0, 2.0]]), ["a", "b", "c"], top_k=2
    )

    assert [item["class_id"] for item in predictions] == [1, 2]
    assert predictions[0]["confidence"] > predictions[1]["confidence"]
    assert (
        utils.get_top_predictions(
            torch.tensor([1.0, 3.0, 2.0]),
            ["a", "b", "c"],
            top_k=2,
            threshold=0.9,
        )
        == []
    )

    with pytest.raises(ValueError):
        utils.get_top_predictions(torch.zeros(1, 2, 2), ["a", "b"])
    with pytest.raises(ValueError):
        utils.get_top_predictions(torch.zeros(2), ["a"])
    with pytest.raises(ValueError):
        utils.get_top_predictions(torch.zeros(2), ["a", "b"], top_k=3)
    with pytest.raises(ValueError):
        utils.get_top_predictions(torch.zeros(2), ["a", "b"], top_k=1, threshold=-1)


def test_latency_percentiles():
    result = utils.calculate_latency_percentiles([10, 20, 30])

    assert result["min"] == 10
    assert result["max"] == 30
    assert result["mean"] == 20
    assert result["p50"] == 20

    with pytest.raises(ValueError):
        utils.calculate_latency_percentiles([])


def test_prediction_logging_and_error_mapping():
    logger = logging.getLogger("test-utils")
    logger.info = lambda *args, **kwargs: None
    captured = []
    logger.error = lambda *args, **kwargs: captured.append((args, kwargs))

    utils.log_prediction(logger, "cat.jpg", {"class": "cat"}, 12.5)

    cases = [
        (ValueError("bad image"), "invalid_input"),
        (MemoryError("full"), "out_of_memory"),
        (TimeoutError("slow"), "timeout"),
        (RuntimeError("failed"), "inference_error"),
        (Exception("unknown"), "internal_error"),
    ]
    for error, expected_type in cases:
        assert utils.handle_inference_error(error, logger)["error"] == expected_type
    assert len(captured) == len(cases)


def test_prediction_cache(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(utils.time, "monotonic", lambda: now[0])
    cache = utils.PredictionCache(max_size=2, ttl_seconds=10)

    assert cache.get("missing") is None
    cache.set("a", {"value": 1})
    cache.set("b", {"value": 2})
    assert cache.get("a") == {"value": 1}

    cache.set("c", {"value": 3})
    assert cache.get("b") is None
    assert cache.size() == 2

    now[0] = 111.0
    assert cache.get("a") is None
    cache.clear()
    assert cache.size() == 0

    with pytest.raises(ValueError):
        utils.PredictionCache(max_size=0)
    with pytest.raises(ValueError):
        utils.PredictionCache(ttl_seconds=-1)


def test_system_info():
    result = utils.get_system_info()

    assert result["cpu_count"] >= 1
    assert result["disk_total_gb"] > 0
    assert 0 <= result["disk_usage_percent"] <= 100
    assert isinstance(result["gpu_available"], bool)
