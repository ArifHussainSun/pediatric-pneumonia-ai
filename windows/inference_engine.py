"""
ONNX Runtime Inference Engine for Windows Desktop

High-performance inference engine optimized for Windows tablets.
Uses ONNX Runtime for maximum speed on Intel processors.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
import torchvision.transforms as transforms

# Setup logging
logger = logging.getLogger(__name__)


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine for pneumonia detection.

    Optimized for Windows Surface Pro tablets with Intel processors.
    """

    def __init__(self, model_path: str):
        """
        Initialize inference engine.

        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None

        # Initialize preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Initialize quality tracking
        self.last_quality_info = None
        self.last_user_feedback = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load ONNX model with optimizations."""
        try:
            # Configure ONNX Runtime for optimal performance on Windows
            providers = ['CPUExecutionProvider']

            # Check for DirectML (Windows GPU acceleration)
            available_providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in available_providers:
                providers.insert(0, 'DmlExecutionProvider')
                logger.info("Using DirectML GPU acceleration")

            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            # Create inference session
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )

            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            logger.info(f"ONNX model loaded successfully: {self.model_path}")
            logger.info(f"Providers: {self.session.get_providers()}")

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for inference with quality assessment and enhancement.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Add src to path for imports
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))

            from src.preprocessing import IntelligentPreprocessor

            # Initialize intelligent preprocessor with autoencoder if available
            autoencoder_path = os.getenv('AUTOENCODER_PATH')
            preprocessor = IntelligentPreprocessor(autoencoder_path=autoencoder_path)

            # Load and process image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Process image through intelligent preprocessing pipeline
                result = preprocessor.process_image(img)

                if not result['success']:
                    # Image quality too poor for reliable analysis
                    logger.warning(f"Image rejected due to poor quality: {result['user_message']}")
                    raise ValueError(f"Image quality insufficient for analysis: {result['user_message']}")

                # Use enhanced image if preprocessing was applied
                processed_image = result['enhanced_image']

                # Store quality information for response
                self.last_quality_info = {
                    "overall_quality": result['original_quality'].overall_quality.value,
                    "brightness_score": result['original_quality'].brightness_score,
                    "contrast_score": result['original_quality'].contrast_score,
                    "sharpness_score": result['original_quality'].sharpness_score,
                    "noise_level": result['original_quality'].noise_level,
                    "positioning_score": result['original_quality'].positioning_score,
                    "artifacts_detected": result['original_quality'].artifacts_detected,
                    "enhancement_applied": result['enhancement_applied'],
                    "recommendations": result['original_quality'].recommendations
                }

                self.last_user_feedback = result['user_message']

                # Log quality assessment and enhancements
                if result['enhancement_applied']:
                    logger.info(f"Image enhanced - Original quality: {result['original_quality'].overall_quality.value}")
                    logger.info(f"Enhanced quality: {result['enhanced_quality'].overall_quality.value}")
                    self.last_quality_info["enhanced_quality"] = result['enhanced_quality'].overall_quality.value

                # Convert numpy array back to PIL Image for transforms
                if isinstance(processed_image, np.ndarray):
                    if processed_image.dtype != np.uint8:
                        processed_image = (processed_image * 255).astype(np.uint8)
                    processed_image = Image.fromarray(processed_image, mode='L').convert('RGB')

                # Apply transforms
                tensor_image = self.transform(processed_image)

                # Add batch dimension and convert to numpy
                batch_tensor = tensor_image.unsqueeze(0)
                numpy_image = batch_tensor.numpy()

                return numpy_image

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on image.

        Args:
            image_path: Path to chest X-ray image

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess image
            input_data = self.preprocess_image(image_path)

            # Run inference
            start_time = self._get_time_ms()

            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )

            inference_time = self._get_time_ms() - start_time

            # Process outputs
            logits = outputs[0][0]  # Remove batch dimension

            # Apply softmax
            probabilities = self._softmax(logits)

            # Get prediction
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]

            # Map to labels
            class_labels = ['NORMAL', 'PNEUMONIA']
            prediction = class_labels[predicted_class]

            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': {
                    'NORMAL': float(probabilities[0]),
                    'PNEUMONIA': float(probabilities[1])
                },
                'inference_time_ms': inference_time,
                'model_path': self.model_path,
                'image_quality': self.last_quality_info,
                'user_feedback': self.last_user_feedback
            }

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _get_time_ms(self) -> float:
        """Get current time in milliseconds."""
        import time
        return time.time() * 1000

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        if not self.session:
            return {'error': 'Model not loaded'}

        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape

        return {
            'model_path': self.model_path,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'providers': self.session.get_providers(),
            'input_name': self.input_name,
            'output_name': self.output_name
        }


def test_inference_engine():
    """Test function for inference engine."""
    try:
        # Test with dummy model path
        model_path = "windows_exports/mobilenet_windows.onnx"

        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            print("Run export script first: python scripts/export_windows.py")
            return

        # Initialize engine
        engine = ONNXInferenceEngine(model_path)

        # Get model info
        info = engine.get_model_info()
        print("Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("Inference engine ready!")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_inference_engine()