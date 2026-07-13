"""
Model Loading and Inference

This module handles loading the ML model and running inference.
Complete the TODOs to implement model loading and prediction logic.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Any
from PIL import Image
import io
import logging
import os

logger = logging.getLogger(__name__)


class ModelInference:
    """
    Model inference class for image classification

    This class handles:
    - Loading pre-trained model
    - Image preprocessing
    - Running inference
    - Post-processing predictions
    """

    def __init__(self, model_name: str = "resnet18", device: str = "cpu"):
        """
        Initialize the model inference class

        TODO: Implement initialization

        Args:
            model_name: Name of the model to load (resnet18, resnet50, etc.)
            device: Device to run inference on ('cpu' or 'cuda')

        Steps:
        1. Store model_name and device
        2. Initialize model to None (will load in load_model())
        3. Initialize preprocessing transforms
        4. Load ImageNet class labels
        """
        # TODO: Implement initialization
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = None
        self.preprocess = self._create_transforms()
        self.classes = self._load_class_labels()

        logger.info(f"Initialized ModelInference with {model_name} on {device}")

    def load_model(self) -> None:
        """
        TODO: Load the pre-trained model

        Steps:
        1. Check if model_name is valid
        2. Load model from torchvision.models
        3. Set model to evaluation mode
        4. Move model to device
        5. Warm up model (run dummy inference)
        6. Log successful loading

        Example:
        if self.model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif self.model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.model.eval()  # Set to evaluation mode
        self.model.to(self.device)

        # Warm up
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)

        logger.info(f"Model {self.model_name} loaded successfully")

        Error Handling:
        - Invalid model_name → ValueError
        - Model loading fails → RuntimeError
        - CUDA not available when device='cuda' → Warning and fallback to CPU
        """
        # TODO: Implement model loading
        logger.info(f"Loading model: {self.model_name}")

        try:
            if self.model_name == "resnet18":
                self.model = models.resnet18(pretrained=True)
            elif self.model_name == "resnet50":
                self.model = models.resnet50(pretrained=True)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            if not torch.cuda.is_available() and self.device.type == 'cuda':
                logger.warning("CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')

            self.model.eval()  # Set to evaluation mode
            self.model.to(self.device)

            # Warm up
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _create_transforms(self) -> transforms.Compose:
        """
        TODO: Create image preprocessing transforms

        ImageNet models expect:
        - Image size: 224x224
        - Normalized with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        - RGB format

        Returns:
            transforms.Compose object with preprocessing pipeline


        Note: These are ImageNet normalization values.
        If using a different model, adjust accordingly.
        """

        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        

    def _load_class_labels(self) -> List[str]:
        """
        TODO: Load ImageNet class labels

        Load the 1000 ImageNet class names for mapping predictions.

        Options:
        1. Load from file (imagenet_classes.txt)
        2. Download from URL
        3. Hardcode common classes

        Returns:
            List of 1000 class names

        Error Handling:
        - File not found → Try downloading
        - Download fails → Use placeholder classes
        """
        logger.info("Loading ImageNet class labels")

        # TODO: Load class labels from file or URL
        import urllib.request
        try: 
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            with urllib.request.urlopen(url) as response:
                classes = [line.decode('utf-8').strip() for line in response]

        except Exception as e:
            logger.error(f"Failed to download class labels from url {url}: {e}")
        # Placeholder
            classes = [f"class_{i}" for i in range(1000)]
        return classes

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        TODO: Preprocess image bytes for model input

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)

        Returns:
            Preprocessed tensor ready for model input (shape: [1, 3, 224, 224])

        Steps:
        1. Convert bytes to PIL Image
        2. Convert to RGB (handle grayscale, RGBA)
        3. Apply transforms
        4. Add batch dimension
        5. Move to device

        Example:
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        input_tensor = self.preprocess(image)

        # Add batch dimension and move to device
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        return input_batch

        Error Handling:
        - Invalid image bytes → ValueError
        - Corrupt image → IOError
        - Unsupported format → ValueError
        """ 
        from PIL import Image, UnidentifiedImageError

        SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG"}
        try: 
            image = Image.open(io.BytesIO(image_bytes)) 
            image.verify()
            image = Image.open(io.BytesIO(image_bytes)) 
        except UnidentifiedImageError as e:
            raise ValueError("Invalid image bytes") from e
        except IOError as e:
            raise IOError("Corrupt image data") from e

        #check unsupportted format
        if image.format not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError("Unsupported image format")
        

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        input_tensor = self.preprocess(image)

        # Add batch dimension and move to device
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        return input_batch


    def predict(
        self,
        image_bytes: bytes,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        TODO: Run inference and return predictions

        Args:
            image_bytes: Raw image bytes
            top_k: Number of top predictions to return

        Returns:
            List of predictions with class_id, class_name, confidence

        Steps:
        1. Validate inputs
        2. Preprocess image
        3. Run inference (with torch.no_grad())
        4. Apply softmax to get probabilities
        5. Get top-k predictions
        6. Format results
        7. Return predictions 

        Return Format:
        [
            {
                'class_id': 285,
                'class_name': 'Egyptian cat',
                'confidence': 0.8234
            },
            {
                'class_id': 281,
                'class_name': 'tabby cat',
                'confidence': 0.1432
            },
            ...
        ]
        """
        # TODO: Implement prediction
        logger.debug(f"Running prediction with top_k={top_k}")
        # Validate
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        # Preprocess
        input_batch = self.preprocess_image(image_bytes)

        # Inference
        with torch.no_grad():
            output = self.model(input_batch)
        # Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top-k
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # Format results
        predictions = []
        for i in range(top_k):
            class_id = int(top_indices[i])
            predictions.append({
                'class_id': class_id,
                'class_name': self.classes[class_id],
                'confidence': float(top_probs[i])
            })

        return predictions


    def predict_from_url(
        self,
        image_url: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        TODO: Download image from URL and run prediction

        Args:
            image_url: URL to image
            top_k: Number of top predictions

        Returns:
            List of predictions

        Steps:
        1. Download image from URL
        2. Validate image
        3. Call predict() with image bytes

        Example:
        import requests

        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        image_bytes = response.content

        return self.predict(image_bytes, top_k)

        Error Handling:
        - Invalid URL → ValueError
        - Download timeout → requests.Timeout
        - HTTP error → requests.HTTPError
        """
        import requests
        from urllib.parse import urlparse

        if not isinstance(image_url, str) or not image_url.strip():
            raise ValueError("image_url must be a non-empty string")

        parsed_url = urlparse(image_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(
                "Invalid image URL: expected an absolute HTTP or HTTPS URL"
            )

        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
        except requests.Timeout:
            logger.error("Timed out while downloading image from %s", image_url)
            raise
        except requests.HTTPError:
            logger.error(
                "HTTP error while downloading image from %s",
                image_url
            )
            raise
        except (requests.InvalidURL, requests.MissingSchema) as e:
            raise ValueError(f"Invalid image URL: {image_url}") from e

        image_bytes = response.content
        if not image_bytes:
            raise ValueError("Downloaded image is empty")

        return self.predict(image_bytes, top_k)

    def get_model_info(self) -> Dict[str, Any]:
        """
        TODO: Get information about the loaded model

        Returns:
            Dictionary with model metadata

    
        """
        # TODO: Implement model info
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'loaded': self.model is not None,
            'num_classes': len(self.classes) if self.classes else 0,
            'input_size': (224, 224),
            'framework': 'PyTorch',
            'version': torch.__version__
        }


# ==============================================================================
# Helper Functions
# ==============================================================================

def load_model( 
    model_name: str = "resnet18",
    device: str = "cpu"
) -> ModelInference:
    """
    TODO: Convenience function to load model

    Args:
        model_name: Name of model
        device: Device to use

    Returns:
        ModelInference instance with loaded model

    Example:
    model_inference = load_model("resnet18", "cpu")
    predictions = model_inference.predict(image_bytes)
    """
    inference = ModelInference(model_name, device)
    inference.load_model()
    return inference


def validate_image(image_bytes: bytes) -> bool:
    """
    TODO: Validate that bytes represent a valid image

    Args:
        image_bytes: Raw image bytes

    Returns:
        True if valid image, False otherwise

    Example:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        return True
    except Exception:
        return False
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        return True
    except Exception:
        return False


def get_supported_models() -> List[str]:
    """
    TODO: Return list of supported model names

    Returns:
        List of model names that can be loaded

    Example:
    return [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
        'mobilenet_v2',
        'efficientnet_b0',
    ]
    """
    
    return [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
        'mobilenet_v2',
        'efficientnet_b0',
    ]


# ==============================================================================
# Testing / Debug
# ==============================================================================

def test_model_loading():
    """
    TODO: Test function to verify model loading works

    This is useful for debugging and validation.

    Example:
    print("Testing model loading...")

    # Test model loading
    model = ModelInference("resnet18", "cpu")
    model.load_model()

    assert model.model is not None, "Model not loaded"
    assert model.classes is not None, "Classes not loaded"

    # Test with dummy image
    from PIL import Image
    import numpy as np

    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    dummy_image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()

    predictions = model.predict(image_bytes, top_k=5)

    assert len(predictions) == 5, "Should return 5 predictions"
    assert all('class_name' in p for p in predictions), "Missing class_name"
    assert all('confidence' in p for p in predictions), "Missing confidence"

    print("✅ All tests passed!")
    """
    print("Testing model loading...")

    # Test model loading
    model = ModelInference("resnet18", "cpu")
    model.load_model()

    assert model.model is not None, "Model not loaded"
    assert model.classes is not None, "Classes not loaded"

    # Test with dummy image
    from PIL import Image
    import numpy as np

    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    dummy_image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()

    predictions = model.predict(image_bytes, top_k=5)

    assert len(predictions) == 5, "Should return 5 predictions"
    assert all('class_name' in p for p in predictions), "Missing class_name"
    assert all('confidence' in p for p in predictions), "Missing confidence"

    print("✅ All tests passed for real!")


if __name__ == "__main__":
    """
    Test the model loading and inference
    """
    # TODO: Add command-line interface for testing
    # Example:
    #python model.py --model resnet18 --image path/to/image.jpg

    print("Model inference module")
    print("Run test_model_loading() to verify functionality")

    # Uncomment to run tests
    test_model_loading()
