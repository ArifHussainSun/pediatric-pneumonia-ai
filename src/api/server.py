"""
REST API Server for Pneumonia Detection Inference

This module provides a production-ready FastAPI server for deploying
pneumonia detection models. Designed for clinical integration with
high-performance inference and medical-grade reliability.

Features:
- Async inference endpoints
- Model management and loading
- Health monitoring
- Request validation
- Error handling
- CORS support for web integration

Optimized for deployment on cloud infrastructure with automatic
scaling and load balancing capabilities.
"""

import os
import io
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, validator, Field
from typing import Optional, List, Dict, Any
import uvicorn

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models with comprehensive validation
class PredictionRequest(BaseModel):
    patient_id: Optional[str] = Field(None, max_length=100, pattern=r'^[a-zA-Z0-9_-]*$')
    model_type: str = Field("mobilenet", pattern=r'^(mobilenet|vgg|xception)$')
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    include_metadata: bool = True

    @validator('patient_id')
    def validate_patient_id(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    model_info: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    image_quality: Optional[Dict[str, Any]] = None
    user_feedback: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    memory_usage_mb: float
    uptime_seconds: float
    version: str

class BatchPredictionRequest(BaseModel):
    images: List[str] = Field(..., min_items=1, max_items=10)  # Base64 encoded images
    patient_ids: Optional[List[str]] = Field(None, max_items=10)
    model_type: str = Field("mobilenet", pattern=r'^(mobilenet|vgg|xception)$')
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)

    @validator('patient_ids')
    def validate_patient_ids_length(cls, v, values):
        if v is not None and 'images' in values and len(v) != len(values['images']):
            raise ValueError('Number of patient IDs must match number of images')
        return v

    @validator('images')
    def validate_images(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 images allowed per batch')
        return v

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_size: int
    total_processing_time_ms: float


# File validation functions
def validate_image_file(file: UploadFile) -> None:
    """Comprehensive image file validation."""
    # Check content type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        )

    # Check file size (max 50MB)
    if hasattr(file, 'size') and file.size:
        max_size = 50 * 1024 * 1024  # 50MB
        if file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size // (1024 * 1024)}MB"
            )

    # Check filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Check file extension
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    if f'.{file_ext}' not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
        )

def validate_image_content(image_data: bytes) -> None:
    """Validate actual image content."""
    if len(image_data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        # Try to open image to validate it's a real image
        image = Image.open(io.BytesIO(image_data))

        # Check image dimensions
        min_size = 32
        max_size = 8192
        width, height = image.size

        if width < min_size or height < min_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image too small. Minimum size: {min_size}x{min_size}"
            )

        if width > max_size or height > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Maximum size: {max_size}x{max_size}"
            )

        # Check if image has proper channels
        if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image mode: {image.mode}"
            )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

# Rate limiting functionality
class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(deque)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for client IP."""
        now = time.time()
        client_requests = self.requests[client_ip]

        # Remove old requests outside time window
        while client_requests and client_requests[0] <= now - self.time_window:
            client_requests.popleft()

        # Check if under limit
        if len(client_requests) >= self.max_requests:
            return False

        # Add current request
        client_requests.append(now)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=100, time_window=3600)  # 100 requests per hour

def check_rate_limit(request: Request) -> None:
    """Dependency to check rate limiting."""
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

class InferenceServer:
    """
    High-performance inference server for pneumonia detection models.

    Provides async REST API endpoints with model management,
    monitoring, and clinical integration capabilities.
    """

    def __init__(self, models_dir: str = "models", config_path: Optional[str] = None):
        """
        Initialize inference server.

        Args:
            models_dir: Directory containing model files
            config_path: Path to server configuration file
        """
        self.models_dir = Path(models_dir)
        self.config = self._load_config(config_path)
        self.models = {}
        self.model_info = {}
        self.start_time = None

        # Initialize components
        self._setup_logging()
        self._load_models()

        logger.info("InferenceServer initialized")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Loaded models: {list(self.models.keys())}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load server configuration."""
        default_config = {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 1,
            'max_request_size': 10 * 1024 * 1024,  # 10MB
            'timeout_seconds': 30,
            'cors_origins': ['*'],
            'models': {
                'mobilenet': {'file': 'outputs/dgx_station_experiment/Best_MobilenetV1.pth', 'device': 'cpu'},
                'vgg': {'file': 'vgg_pneumonia.pth', 'device': 'cpu'},
                'xception': {'file': 'xception_pneumonia.pth', 'device': 'cpu'}
            }
        }

        if config_path and Path(config_path).exists():
            import json
            with open(config_path) as f:
                user_config = json.load(f)
            default_config.update(user_config)

        return default_config

    def _setup_logging(self):
        """Setup structured logging for the API server."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format=log_format
        )

    def _load_models(self):
        """Load all configured models."""
        from ..models import create_model

        for model_name, model_config in self.config['models'].items():
            try:
                file_path = model_config['file']

                # Handle both absolute and relative paths
                if file_path.startswith('/') or file_path.startswith('outputs/'):
                    # Absolute path or outputs relative path
                    if file_path.startswith('outputs/'):
                        model_path = Path(file_path)
                    else:
                        model_path = Path(file_path)
                else:
                    # Relative to models directory
                    model_path = self.models_dir / file_path

                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    continue

                # Create model architecture
                model = create_model(model_name, num_classes=2)

                # Load checkpoint
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # Extract model state dict
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                else:
                    model_state = checkpoint

                # Load weights using custom loading for MobileNet (handles architecture mismatches)
                if model_name == 'mobilenet' and hasattr(model, 'load_custom_weights'):
                    # For MobileNet, use custom loading
                    success = model.load_custom_weights(model_path)
                    if not success:
                        logger.error(f"Failed to load {model_name} using custom loader")
                        continue
                else:
                    # Standard loading for other models
                    model.load_state_dict(model_state)

                model.eval()

                # Store model and info
                self.models[model_name] = model
                self.model_info[model_name] = {
                    'file_path': str(model_path),
                    'device': model_config.get('device', 'cpu'),
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'status': 'loaded'
                }

                # Log checkpoint information
                if isinstance(checkpoint, dict):
                    if 'metrics' in checkpoint:
                        metrics = checkpoint['metrics']
                        accuracy = metrics.get('accuracy', 'Unknown')
                        logger.info(f"Loaded model: {model_name} (accuracy: {accuracy:.4f})")
                    elif 'results' in checkpoint and 'accuracy' in checkpoint['results']:
                        accuracy = checkpoint['results']['accuracy']
                        logger.info(f"Loaded model: {model_name} (accuracy: {accuracy:.4f})")
                    else:
                        logger.info(f"Loaded model: {model_name}")
                else:
                    logger.info(f"Loaded model: {model_name}")

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                self.model_info[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }

    async def predict(self,
                     image: Image.Image,
                     model_type: str = "mobilenet",
                     patient_id: Optional[str] = None,
                     confidence_threshold: float = 0.5,
                     include_metadata: bool = True,
                     disable_preprocessing: bool = False) -> PredictionResponse:
        """
        Perform pneumonia prediction on a single image.

        Args:
            image: PIL Image object
            model_type: Type of model to use
            patient_id: Optional patient identifier
            confidence_threshold: Minimum confidence threshold
            include_metadata: Whether to include additional metadata

        Returns:
            PredictionResponse with prediction results
        """
        import time

        start_time = time.perf_counter()

        if model_type not in self.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_type} not available. Available models: {list(self.models.keys())}"
            )

        try:
            # Preprocess image (with option to disable for testing)
            quality_info = None
            user_feedback = None

            if disable_preprocessing:
                processed_image = await self._simple_preprocess_image(image)
            else:
                processed_image, quality_info, user_feedback = await self._preprocess_image_with_feedback(image)

            # Run inference
            model = self.models[model_type]
            with torch.no_grad():
                outputs = model(processed_image)

            # Apply temperature scaling to calibrate confidence (reduce overconfidence)
            temperature = 2.5  # Higher = less confident, more realistic for medical use
            calibrated_outputs = outputs / temperature
            probabilities = torch.softmax(calibrated_outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

            # Map prediction to label (matches training: PNEUMONIA=0, NORMAL=1)
            class_labels = ["PNEUMONIA", "NORMAL"]
            prediction = class_labels[predicted_class]

            processing_time = (time.perf_counter() - start_time) * 1000

            # Build response
            response = PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                probabilities={
                    "PNEUMONIA": float(probabilities[0][0]),
                    "NORMAL": float(probabilities[0][1])
                },
                processing_time_ms=processing_time,
                model_info={
                    "model_type": model_type,
                    "version": "1.0.0",
                    "device": self.model_info[model_type].get('device', 'cpu')
                },
                image_quality=quality_info,
                user_feedback=user_feedback
            )

            # Add metadata if requested
            if include_metadata:
                response.metadata = {
                    "patient_id": patient_id,
                    "image_size": image.size,
                    "confidence_threshold": confidence_threshold,
                    "meets_threshold": bool(confidence >= confidence_threshold),
                    "timestamp": time.time()
                }

            return response

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    async def predict_batch(self,
                           images: List[Image.Image],
                           model_type: str = "mobilenet",
                           patient_ids: Optional[List[str]] = None,
                           confidence_threshold: float = 0.5) -> BatchPredictionResponse:
        """
        Perform batch prediction on multiple images.

        Args:
            images: List of PIL Image objects
            model_type: Type of model to use
            patient_ids: Optional list of patient identifiers
            confidence_threshold: Minimum confidence threshold

        Returns:
            BatchPredictionResponse with batch results
        """
        import time

        start_time = time.perf_counter()

        # Validate inputs
        if patient_ids and len(patient_ids) != len(images):
            raise HTTPException(
                status_code=400,
                detail="Number of patient IDs must match number of images"
            )

        # Process images in parallel
        tasks = []
        for i, image in enumerate(images):
            patient_id = patient_ids[i] if patient_ids else None
            task = self.predict(
                image=image,
                model_type=model_type,
                patient_id=patient_id,
                confidence_threshold=confidence_threshold,
                include_metadata=True
            )
            tasks.append(task)

        # Wait for all predictions
        predictions = await asyncio.gather(*tasks)

        total_processing_time = (time.perf_counter() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(images),
            total_processing_time_ms=total_processing_time
        )

    async def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model inference with intelligent quality assessment and enhancement."""
        from ..data.datasets import get_medical_transforms
        from ..preprocessing import IntelligentPreprocessor

        # Initialize intelligent preprocessor with autoencoder if available
        autoencoder_path = os.getenv('AUTOENCODER_PATH')
        preprocessor = IntelligentPreprocessor(autoencoder_path=autoencoder_path)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image through intelligent preprocessing pipeline
        result = preprocessor.process_image(image)

        if not result['success']:
            # Image quality too poor for reliable analysis
            logger.warning(f"Image rejected due to poor quality: {result['user_message']}")
            raise HTTPException(
                status_code=400,
                detail=f"Image quality insufficient for analysis: {result['user_message']}"
            )

        # Use enhanced image if preprocessing was applied
        processed_image = result['enhanced_image']

        # Log quality assessment and enhancements
        if result['enhancement_applied']:
            logger.info(f"Image enhanced - Original quality: {result['original_quality'].overall_quality.value}")
            logger.info(f"Enhanced quality: {result['enhanced_quality'].overall_quality.value}")

        # Convert numpy array back to PIL Image for transforms
        if isinstance(processed_image, np.ndarray):
            if processed_image.dtype != np.uint8:
                processed_image = (processed_image * 255).astype(np.uint8)
            processed_image = Image.fromarray(processed_image, mode='L').convert('RGB')

        # Apply transforms (inference only, no augmentation)
        transform = get_medical_transforms(image_size=(224, 224), is_training=False)
        tensor = transform(processed_image).unsqueeze(0)

        return tensor

    async def _simple_preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Simple preprocessing without intelligent enhancement for baseline testing."""
        from ..data.datasets import get_medical_transforms

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms only (no quality assessment or enhancement)
        transform = get_medical_transforms(image_size=(224, 224), is_training=False)
        tensor = transform(image).unsqueeze(0)

        return tensor

    async def _preprocess_image_with_feedback(self, image: Image.Image):
        """Preprocess image and return quality assessment and user feedback."""
        from ..data.datasets import get_medical_transforms
        from ..preprocessing import IntelligentPreprocessor

        # Initialize intelligent preprocessor with autoencoder if available
        autoencoder_path = os.getenv('AUTOENCODER_PATH')
        preprocessor = IntelligentPreprocessor(autoencoder_path=autoencoder_path)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image through intelligent preprocessing pipeline
        result = preprocessor.process_image(image)

        if not result['success']:
            # Image quality too poor for reliable analysis
            logger.warning(f"Image rejected due to poor quality: {result['user_message']}")
            raise HTTPException(
                status_code=400,
                detail=f"Image quality insufficient for analysis: {result['user_message']}"
            )

        # Use enhanced image if preprocessing was applied
        processed_image = result['enhanced_image']

        # Prepare quality info for response (convert numpy types to native Python types)
        quality_info = {
            "overall_quality": result['original_quality'].overall_quality.value,
            "brightness_score": float(result['original_quality'].brightness_score),
            "contrast_score": float(result['original_quality'].contrast_score),
            "sharpness_score": float(result['original_quality'].sharpness_score),
            "noise_level": float(result['original_quality'].noise_level),
            "positioning_score": float(result['original_quality'].positioning_score),
            "artifacts_detected": bool(result['original_quality'].artifacts_detected),
            "enhancement_applied": bool(result['enhancement_applied']),
            "recommendations": result['original_quality'].recommendations
        }

        # Get user feedback message
        user_feedback = result['user_message']

        # Log quality assessment and enhancements
        if result['enhancement_applied']:
            logger.info(f"Image enhanced - Original quality: {result['original_quality'].overall_quality.value}")
            logger.info(f"Enhanced quality: {result['enhanced_quality'].overall_quality.value}")
            quality_info["enhanced_quality"] = result['enhanced_quality'].overall_quality.value

        # Convert numpy array back to PIL Image for transforms
        if isinstance(processed_image, np.ndarray):
            if processed_image.dtype != np.uint8:
                processed_image = (processed_image * 255).astype(np.uint8)
            processed_image = Image.fromarray(processed_image, mode='L').convert('RGB')

        # Apply transforms (inference only, no augmentation)
        transform = get_medical_transforms(image_size=(224, 224), is_training=False)
        tensor = transform(processed_image).unsqueeze(0)

        return tensor, quality_info, user_feedback

    def get_health_status(self) -> HealthResponse:
        """Get server health status."""
        import psutil
        import time

        # Calculate uptime
        uptime = time.time() - self.start_time if self.start_time else 0

        # Get memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 2)  # MB

        # Check model status
        loaded_models = [name for name, info in self.model_info.items()
                        if info.get('status') == 'loaded']

        return HealthResponse(
            status="healthy" if loaded_models else "degraded",
            models_loaded=loaded_models,
            memory_usage_mb=memory_usage,
            uptime_seconds=uptime,
            version="1.0.0"
        )

    def start_server(self):
        """Start the inference server."""
        import time
        self.start_time = time.time()

        app = create_app(self)

        uvicorn.run(
            app,
            host=self.config['host'],
            port=self.config['port'],
            workers=self.config['workers'],
            log_level="info"
        )


def create_app(inference_server: InferenceServer) -> FastAPI:
    """
    Create FastAPI application with inference server.

    Args:
        inference_server: Configured inference server instance

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Pediatric Pneumonia Detection API",
        description="REST API for pneumonia detection in pediatric chest X-rays",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=inference_server.config.get('cors_origins', ['*']),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Get server health status."""
        return inference_server.get_health_status()

    # Single prediction endpoint
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_pneumonia(
        request: Request,
        file: UploadFile = File(...),
        model_type: str = "mobilenet",
        patient_id: Optional[str] = None,
        confidence_threshold: float = 0.5,
        include_metadata: bool = True,
        disable_preprocessing: bool = False,
    ):
        """
        Predict pneumonia from chest X-ray image.

        Args:
            file: Uploaded image file
            model_type: Model to use for prediction
            patient_id: Optional patient identifier
            confidence_threshold: Minimum confidence threshold
            include_metadata: Include additional metadata in response

        Returns:
            Prediction results
        """
        try:
            # Comprehensive file validation
            validate_image_file(file)

            # Load and validate image data
            image_data = await file.read()
            validate_image_content(image_data)

            # Load image
            image = Image.open(io.BytesIO(image_data))

            # Perform prediction
            result = await inference_server.predict(
                image=image,
                model_type=model_type,
                patient_id=patient_id,
                confidence_threshold=confidence_threshold,
                include_metadata=include_metadata,
                disable_preprocessing=disable_preprocessing
            )

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction endpoint failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    # Batch prediction endpoint
    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch_pneumonia(
        request: Request,
        files: List[UploadFile] = File(...),
        model_type: str = "mobilenet",
        patient_ids: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Predict pneumonia from multiple chest X-ray images.

        Args:
            files: List of uploaded image files
            model_type: Model to use for prediction
            patient_ids: Optional list of patient identifiers
            confidence_threshold: Minimum confidence threshold

        Returns:
            Batch prediction results
        """
        try:
            # Validate input
            if len(files) > 10:  # Limit batch size
                raise HTTPException(
                    status_code=400,
                    detail="Batch size limited to 10 images"
                )

            # Load and validate images
            images = []
            for i, file in enumerate(files):
                try:
                    # Comprehensive file validation for each file
                    validate_image_file(file)

                    # Load and validate image data
                    image_data = await file.read()
                    validate_image_content(image_data)

                    # Load image
                    image = Image.open(io.BytesIO(image_data))
                    images.append(image)

                except HTTPException as e:
                    # Add file index to error message for batch processing
                    raise HTTPException(
                        status_code=e.status_code,
                        detail=f"File {i+1} ({file.filename}): {e.detail}"
                    )

            # Perform batch prediction
            result = await inference_server.predict_batch(
                images=images,
                model_type=model_type,
                patient_ids=patient_ids,
                confidence_threshold=confidence_threshold
            )

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch prediction endpoint failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    # Model information endpoint
    @app.get("/models")
    async def get_models():
        """Get information about available models."""
        return {
            "available_models": list(inference_server.models.keys()),
            "model_info": inference_server.model_info
        }

    return app

# Create the app instance for uvicorn
inference_server = InferenceServer(models_dir="outputs")
app = create_app(inference_server)

if __name__ == "__main__":
    # Test server setup
    print("Testing inference server...")

    try:
        # Create test server
        server = InferenceServer(models_dir="models")
        print("Inference server ready for deployment!")

        # Optionally start server
        if os.getenv("START_SERVER"):
            server.start_server()

    except Exception as e:
        print(f"Error during server setup: {e}")

    print("Inference server system ready!")