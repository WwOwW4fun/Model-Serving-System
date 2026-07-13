"""Reusable utilities for model serving, monitoring, and image handling."""

import io
import json
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError


class JSONFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
            }
        )


def setup_logging(log_level: str = "INFO", json_format: bool = True) -> logging.Logger:
    """Create an idempotently configured application logger."""
    normalized = log_level.upper()
    if normalized not in logging._nameToLevel:
        raise ValueError(f"invalid log level: {log_level}")

    logger = logging.getLogger("model-serving")
    logger.setLevel(normalized)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    logger.addHandler(handler)
    return logger


def validate_image_file(file_content: bytes, max_size_mb: int = 10) -> Tuple[bool, str]:
    """Validate upload size, format, integrity, and dimensions."""
    if max_size_mb <= 0:
        return False, "Maximum image size must be positive"
    if not file_content:
        return False, "Image file is empty"
    if len(file_content) > max_size_mb * 1024 * 1024:
        return False, f"Image exceeds the {max_size_mb} MB limit"

    try:
        with Image.open(io.BytesIO(file_content)) as image:
            if image.format not in {"JPEG", "PNG"}:
                return False, "Only JPEG and PNG images are supported"
            if image.width <= 0 or image.height <= 0:
                return False, "Image dimensions are invalid"
            if image.width > 4096 or image.height > 4096:
                return False, "Image dimensions exceed 4096x4096"
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return False, "Image file is invalid or corrupted"
    return True, ""


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Decode image bytes and return a detached RGB image."""
    if not image_bytes:
        raise ValueError("image data is empty")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("invalid image data") from exc


def resize_image(image: Image.Image, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize and center-crop an image to an exact size."""
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("target dimensions must be positive")
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def preprocess_image(
    image: Image.Image,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """Convert an RGB image into a normalized NCHW float tensor."""
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("mean and std must contain three valid channel values")
    rgb_image = image.convert("RGB")
    pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    return torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)


def get_model_path(model_name: str, models_dir: str = "models") -> Path:
    """Return the conventional local path for a named model."""
    if not model_name or Path(model_name).name != model_name:
        raise ValueError("model_name must be a simple non-empty name")
    return Path(models_dir) / f"{model_name}.pth"


def load_class_labels(labels_file: str = "imagenet_classes.txt") -> List[str]:
    """Load non-empty class labels from a text or JSON file."""
    path = Path(labels_file)
    if not path.is_file():
        raise FileNotFoundError(f"class labels file not found: {labels_file}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = [str(data[key]).strip() for key in sorted(data, key=int)]
        elif isinstance(data, list):
            labels = [str(value).strip() for value in data]
        else:
            raise ValueError("label JSON must be an object or an array")
    else:
        labels = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        ]

    labels = [label for label in labels if label]
    if not labels:
        raise ValueError("class labels file is empty")
    return labels


def warm_up_model(
    model: torch.nn.Module, device: torch.device, num_iterations: int = 10
) -> None:
    """Run dummy inferences to initialize model execution paths."""
    if num_iterations < 0:
        raise ValueError("num_iterations cannot be negative")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(dummy_input)


def get_top_predictions(
    logits: torch.Tensor,
    class_labels: List[str],
    top_k: int = 5,
    threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Convert model logits into sorted, filtered prediction records."""
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.ndim != 1:
        raise ValueError("logits must have shape (classes,) or (1, classes)")
    if len(class_labels) != logits.numel():
        raise ValueError("class label count must match the model output")
    if not 1 <= top_k <= len(class_labels):
        raise ValueError("top_k is outside the available class range")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    probabilities = torch.softmax(logits, dim=0)
    top_values, top_indices = torch.topk(probabilities, top_k)
    return [
        {
            "class_name": class_labels[int(index)],
            "class_id": int(index),
            "confidence": float(probability),
        }
        for probability, index in zip(top_values, top_indices)
        if float(probability) >= threshold
    ]


def calculate_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Summarize a non-empty sequence of latency measurements."""
    if not latencies:
        raise ValueError("at least one latency measurement is required")
    values = np.asarray(latencies, dtype=float)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def log_prediction(
    logger: logging.Logger,
    image_name: str,
    prediction: Dict[str, Any],
    latency_ms: float,
    status: str = "success",
) -> None:
    """Write one structured prediction event."""
    logger.info(
        "Prediction completed",
        extra={
            "image": image_name,
            "prediction": prediction,
            "latency_ms": latency_ms,
            "status": status,
        },
    )


def handle_inference_error(error: Exception, logger: logging.Logger) -> Dict[str, str]:
    """Log an inference exception and return a safe client response."""
    logger.error("Inference failed: %s", error, exc_info=True)
    if isinstance(error, ValueError):
        error_type, message = "invalid_input", str(error)
    elif isinstance(error, MemoryError):
        error_type, message = "out_of_memory", "The server ran out of memory"
    elif isinstance(error, TimeoutError):
        error_type, message = "timeout", "Inference timed out"
    elif isinstance(error, RuntimeError):
        error_type, message = "inference_error", "Model inference failed"
    else:
        error_type, message = "internal_error", "An unexpected error occurred"
    return {"error": error_type, "message": message, "details": str(error)}


class PredictionCache:
    """Thread-safe least-recently-used cache with per-entry expiry."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        if max_size <= 0 or ttl_seconds < 0:
            raise ValueError("max_size must be positive and ttl_seconds non-negative")
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, Tuple[float, Dict[str, Any]]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            created_at, value = item
            if time.monotonic() - created_at >= self.ttl_seconds:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._items)


def check_model_health(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    """Run a small inference and report model availability and latency."""
    result: Dict[str, Any] = {
        "healthy": False,
        "model_loaded": model is not None,
        "device_available": device.type == "cpu" or torch.cuda.is_available(),
        "test_inference_ms": None,
        "error": None,
    }
    if model is None:
        result["error"] = "model is not loaded"
        return result
    if not result["device_available"]:
        result["error"] = f"device is unavailable: {device}"
        return result

    try:
        started = time.perf_counter()
        with torch.no_grad():
            model(torch.zeros(1, 3, 224, 224, device=device))
        result["test_inference_ms"] = (time.perf_counter() - started) * 1000
        result["healthy"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def get_system_info() -> Dict[str, Any]:
    """Return lightweight CPU, disk, and accelerator information."""
    disk = shutil.disk_usage(Path.cwd())
    return {
        "cpu_count": os.cpu_count() or 1,
        "disk_total_gb": disk.total / (1024**3),
        "disk_free_gb": disk.free / (1024**3),
        "disk_usage_percent": (disk.used / disk.total) * 100,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }
