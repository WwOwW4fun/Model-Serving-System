"""
Tests for Model Inference Module.
"""

import io
import time

import pytest
import torch
from PIL import Image


class TinyImageNetModel(torch.nn.Module):
    """Tiny deterministic model with ImageNet-like output shape."""

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        logits = torch.zeros(batch.shape[0], 1000, device=batch.device)
        logits[:, 10] = 10.0
        logits[:, 5] = 8.0
        logits[:, 1] = 6.0
        logits[:, 0] = 4.0
        return logits


def image_to_bytes(image: Image.Image, image_format: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def lightweight_model(monkeypatch):
    import src.model as model_module

    monkeypatch.setattr(
        model_module.ModelInference,
        "_load_class_labels",
        lambda self: [f"class_{i}" for i in range(1000)],
    )
    monkeypatch.setattr(
        model_module.models, "resnet18", lambda pretrained=True: TinyImageNetModel()
    )
    monkeypatch.setattr(
        model_module.models, "resnet50", lambda pretrained=True: TinyImageNetModel()
    )


@pytest.fixture
def model_inference():
    from src.model import ModelInference

    model = ModelInference(model_name="resnet18", device="cpu")
    model.load_model()
    return model


@pytest.fixture
def sample_tensor():
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_image():
    return Image.new("RGB", (224, 224), color="red")


def test_model_init():
    from src.model import ModelInference

    model = ModelInference(model_name="resnet18", device="cpu")

    assert model.model is None
    assert model.model_name == "resnet18"
    assert str(model.device) == "cpu"
    assert len(model.classes) == 1000


def test_model_init_with_invalid_name():
    from src.model import ModelInference

    model = ModelInference(model_name="invalid_model", device="cpu")
    with pytest.raises(RuntimeError):
        model.load_model()


def test_model_init_with_cuda_device():
    from src.model import ModelInference

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = ModelInference(model_name="resnet18", device="cuda")
    assert model.device.type == "cuda"


def test_load_model():
    from src.model import ModelInference

    model = ModelInference(model_name="resnet18", device="cpu")
    model.load_model()

    assert model.model is not None
    assert not model.model.training
    assert next(model.model.parameters()).device.type == "cpu"


def test_load_model_twice():
    from src.model import ModelInference

    model = ModelInference(model_name="resnet18", device="cpu")
    model.load_model()
    model.load_model()

    assert model.model is not None


def test_preprocess_image(sample_image):
    from src.model import ModelInference

    model = ModelInference(model_name="resnet18", device="cpu")
    tensor = model.preprocess_image(image_to_bytes(sample_image))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.min() >= -3.0
    assert tensor.max() <= 3.0


def test_preprocess_with_invalid_input():
    from src.model import ModelInference

    model = ModelInference(model_name="resnet18", device="cpu")

    with pytest.raises((ValueError, TypeError, OSError)):
        model.preprocess_image(b"not an image")


def test_preprocess_with_different_sizes():
    from src.model import ModelInference

    model = ModelInference(model_name="resnet18", device="cpu")

    for size in [(100, 100), (224, 224), (500, 300), (1920, 1080)]:
        image = Image.new("RGB", size, color="blue")
        tensor = model.preprocess_image(image_to_bytes(image))
        assert tensor.shape == (1, 3, 224, 224)


def test_predict(model_inference, sample_image):
    result = model_inference.predict(image_to_bytes(sample_image), top_k=5)

    assert isinstance(result, list)
    assert len(result) == 5
    for pred in result:
        assert set(pred) == {"class_id", "class_name", "confidence"}
        assert 0 <= pred["confidence"] <= 1


def test_predict_with_tensor_input(model_inference, sample_tensor):
    if not hasattr(model_inference, "predict_tensor"):
        pytest.skip("predict_tensor is not implemented")

    result = model_inference.predict_tensor(sample_tensor, top_k=5)
    assert isinstance(result, list)
    assert len(result) == 5


def test_predict_without_loading_model(sample_image):
    from src.model import ModelInference

    model = ModelInference(model_name="resnet18", device="cpu")

    with pytest.raises(RuntimeError):
        model.predict(image_to_bytes(sample_image))


def test_predict_with_different_top_k(model_inference, sample_image):
    image_bytes = image_to_bytes(sample_image)

    for top_k in [1, 3, 5, 10]:
        result = model_inference.predict(image_bytes, top_k=top_k)
        assert len(result) == top_k


def test_predict_with_threshold(model_inference, sample_image):
    if "threshold" not in model_inference.predict.__code__.co_varnames:
        pytest.skip("confidence threshold is not implemented")

    result = model_inference.predict(
        image_to_bytes(sample_image), top_k=10, threshold=0.1
    )
    assert all(pred["confidence"] >= 0.1 for pred in result)


def test_postprocess_output(model_inference):
    if not hasattr(model_inference, "postprocess"):
        pytest.skip("postprocess is not implemented")

    output = torch.randn(1, 1000)
    result = model_inference.postprocess(output, top_k=5)

    assert len(result) == 5
    assert all("class_name" in p for p in result)
    confidences = [p["confidence"] for p in result]
    assert confidences == sorted(confidences, reverse=True)


def test_batch_inference(model_inference):
    if not hasattr(model_inference, "predict_batch"):
        pytest.skip("predict_batch is not implemented")

    images = [
        image_to_bytes(Image.new("RGB", (224, 224), color="red")) for _ in range(4)
    ]
    results = model_inference.predict_batch(images, top_k=5)

    assert len(results) == 4
    assert all(len(result) == 5 for result in results)


def test_batch_inference_performance(model_inference):
    if not hasattr(model_inference, "predict_batch"):
        pytest.skip("predict_batch is not implemented")

    images = [image_to_bytes(Image.new("RGB", (224, 224))) for _ in range(10)]

    start = time.time()
    batch_results = model_inference.predict_batch(images, top_k=5)
    batch_time = time.time() - start

    start = time.time()
    seq_results = [model_inference.predict(image, top_k=5) for image in images]
    seq_time = time.time() - start

    assert len(batch_results) == len(seq_results)
    assert batch_time < seq_time * 1.5


def test_warmup(model_inference):
    if not hasattr(model_inference, "warmup"):
        pytest.skip("warmup is not implemented as a public method")

    model_inference.warmup(num_iterations=2)
    assert model_inference.model is not None


def test_first_inference_without_warmup(sample_image):
    from src.model import ModelInference

    model = ModelInference(model_name="resnet18", device="cpu")
    model.load_model()

    image_bytes = image_to_bytes(sample_image)
    first = model.predict(image_bytes, top_k=1)
    second = model.predict(image_bytes, top_k=1)

    assert first[0]["class_id"] == second[0]["class_id"]


def test_handle_corrupted_image(model_inference):
    with pytest.raises((ValueError, OSError)):
        model_inference.predict(b"This is not an image")


def test_handle_wrong_image_mode(model_inference):
    gray_image = Image.new("L", (224, 224), color=128)
    gray_result = model_inference.predict(image_to_bytes(gray_image), top_k=1)
    assert len(gray_result) == 1

    rgba_image = Image.new("RGBA", (224, 224), color=(255, 0, 0, 255))
    rgba_result = model_inference.predict(image_to_bytes(rgba_image, "PNG"), top_k=1)
    assert len(rgba_result) == 1


def test_memory_cleanup_after_inference(model_inference, sample_image):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.cuda.empty_cache()
    initial_mem = torch.cuda.memory_allocated()

    for _ in range(10):
        model_inference.predict(image_to_bytes(sample_image), top_k=1)

    torch.cuda.empty_cache()
    final_mem = torch.cuda.memory_allocated()

    assert final_mem - initial_mem < 100 * 1024 * 1024


def test_prediction_consistency(model_inference, sample_image):
    image_bytes = image_to_bytes(sample_image)
    results = [model_inference.predict(image_bytes, top_k=1) for _ in range(5)]

    top_classes = [result[0]["class_name"] for result in results]
    assert len(set(top_classes)) == 1


def test_known_image_prediction(model_inference):
    image = Image.new("RGB", (224, 224), color="orange")
    result = model_inference.predict(image_to_bytes(image), top_k=5)

    assert len(result) == 5
    assert all(pred["confidence"] > 0 for pred in result)


def test_switch_models(sample_image):
    from src.model import ModelInference

    model = ModelInference(model_name="resnet18", device="cpu")
    model.load_model()
    result1 = model.predict(image_to_bytes(sample_image), top_k=1)
    assert len(result1) == 1

    model.model_name = "resnet50"
    model.load_model()
    result2 = model.predict(image_to_bytes(sample_image), top_k=1)
    assert len(result2) == 1
