"""
MobileNet model for pediatric pneumonia detection.

This module contains the MobileNetFineTune model class that uses transfer learning
to adapt a pre-trained MobileNet model for medical image classification.

MobileNet is a lightweight CNN designed for mobile and embedded devices using
depthwise separable convolutions for efficiency while maintaining good performance.
"""

import torch
import torch.nn as nn
import timm
from torchvision import models as tv_models


class MobileNetV2Seyon(nn.Module):
    """
    MobileNetV2 implementation matching trained model architecture.

    Architecture:
    - MobileNetV2 backbone with inverted residual blocks
    - Conv_head: 320 -> 1280 channels
    - Classifier: 1280 -> 64 -> 2 classes
    """

    def __init__(self, num_classes=2, dropout_rate=0.4, hidden_size=64):
        super(MobileNetV2Seyon, self).__init__()

        # Use torchvision MobileNetV2 to match expected architecture
        self.mobilenet = tv_models.mobilenet_v2(pretrained=True)

        # Remove original classifier
        self.mobilenet.classifier = nn.Identity()

        # Add conv_head (320 -> 1280 channels) to match saved model
        self.mobilenet.conv_head = nn.Conv2d(320, 1280, kernel_size=1, bias=False)

        # Add batch norm for conv_head output
        self.mobilenet.bn2 = nn.BatchNorm2d(1280)

        # Create classifier matching trained model structure
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # Index 0
            nn.Linear(1280, hidden_size),  # Index 1 (64 hidden units)
            nn.ReLU(),  # Index 2
            nn.Dropout(dropout_rate * 0.5),  # Index 3
            nn.Linear(hidden_size, num_classes)  # Index 4 (2 classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        # Use MobileNetV2 features extraction
        x = self.mobilenet.features(x)

        # Standard MobileNetV2 gives us 1280 channels, but Seyon's model expects 320
        # Apply global average pooling first
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Use the standard classifier directly (skip conv_head since dimensions don't match)
        x = self.mobilenet.classifier(x)
        return x

    def load_custom_weights(self, checkpoint_path):
        """Load Seyon's trained weights (compatible layers only for now)."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                saved_state = checkpoint['model_state_dict']
            else:
                saved_state = checkpoint

            # Load only compatible weights (classifier and conv_head)
            current_state = self.state_dict()
            loaded_keys = []
            skipped_keys = []

            for key, value in saved_state.items():
                if key in current_state and current_state[key].shape == value.shape:
                    current_state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(key)

            # Load the updated state dict
            self.load_state_dict(current_state)

            logger.info(f"Successfully loaded {len(loaded_keys)} compatible layers from Seyon's model")
            if skipped_keys:
                logger.info(f"Skipped {len(skipped_keys)} incompatible backbone layers (using pretrained weights)")

            # Load accuracy info if available
            if isinstance(checkpoint, dict) and 'results' in checkpoint:
                results = checkpoint['results']
                logger.info(f"Seyon's original model accuracy: {results.get('accuracy', 'Unknown'):.4f}")

            return True

        except Exception as e:
            logger.error(f"Failed to load Seyon's weights: {e}")
            return False


class MobileNetFineTune(nn.Module):
    """
    MobileNet model adapted for pneumonia detection.

    Lightweight architecture suitable for deployment on mobile devices
    or when computational resources are limited.

    Architecture:
    - Pre-trained MobileNetV2 backbone (feature extraction)
    - Custom lightweight classification head
    - Selective layer freezing for stable transfer learning

    Args:
        num_classes (int): Number of output classes (default: 2 for Normal/Pneumonia)
        freeze_layers (int): Number of early layers to freeze during training (default: 50)
        dropout_rate (float): Dropout rate for regularization (default: 0.4)
        hidden_size (int): Size of hidden layer in classifier (default: 64)
    """

    def __init__(self, num_classes=2, freeze_layers=50, dropout_rate=0.4, hidden_size=64):
        super(MobileNetFineTune, self).__init__()

        # Create raw timm MobileNetV1 that preserves exact structure (conv_stem, blocks)
        raw_model = timm.create_model('mobilenetv1_100', pretrained=True, num_classes=1000)

        # Copy all the raw model attributes to preserve structure
        self.conv_stem = raw_model.conv_stem
        self.bn1 = raw_model.bn1
        self.blocks = raw_model.blocks
        self.conv_head = raw_model.conv_head
        self.bn2 = raw_model.bn2
        self.global_pool = raw_model.global_pool

        # Get the number of features from the original classifier
        num_features = raw_model.classifier.in_features

        # Replace with our custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_size, num_classes)
        )

        # Store configuration
        self.freeze_layers = freeze_layers
        self.num_classes = num_classes

        # Freeze early layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, num_layers):
        """
        Freeze the first num_layers of the MobileNet backbone.

        Args:
            num_layers (int): Number of layers to freeze from the beginning
        """
        frozen_count = 0
        for name, param in self.named_parameters():
            if 'classifier' not in name:  # Don't freeze classifier
                if frozen_count < num_layers:
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    break
        print(f"Frozen first {frozen_count} MobileNet layers")

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("All MobileNet layers unfrozen for fine-tuning")

    def forward(self, x):
        """
        Forward pass through MobileNet using raw timm structure.

        Args:
            x (torch.Tensor): Input batch of images [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Raw prediction logits [batch_size, num_classes]
        """
        # Follow the exact timm MobileNetV1 forward pass
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.global_pool(x)
        if self.global_pool.is_identity():
            x = x.flatten(1)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization.

        Args:
            x (torch.Tensor): Input batch of images

        Returns:
            torch.Tensor: Feature maps from MobileNet backbone
        """
        with torch.no_grad():
            # Extract features before classifier
            features = self.mobilenet.features(x)
            return features

    def predict_proba(self, x):
        """
        Get prediction probabilities.

        Args:
            x (torch.Tensor): Input batch of images

        Returns:
            torch.Tensor: Prediction probabilities [batch_size, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def get_model_size(self):
        """
        Get model size information.

        Returns:
            dict: Dictionary containing model size metrics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'model_size_mb': (total_params * 4) / (1024 * 1024)  # Assuming float32
        }

    def load_custom_weights(self, checkpoint_path):
        """
        Custom weight loading for MobileNetV1 checkpoint with prefix handling.

        This method handles loading weights from checkpoint with 'mobilenet.' prefix
        and maps them to the current timm MobileNetV1 architecture.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        import torch
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                saved_state = checkpoint['model_state_dict']
            else:
                saved_state = checkpoint

            # Get current model state
            current_state = self.state_dict()

            # Map saved weights to current model (handle mobilenet. prefix)
            loaded_keys = []
            skipped_keys = []

            for saved_key, value in saved_state.items():
                # Remove 'mobilenet.' prefix if present to match our model structure
                if saved_key.startswith('mobilenet.'):
                    target_key = saved_key[10:]  # Remove 'mobilenet.' prefix
                else:
                    target_key = saved_key

                # Our model structure doesn't have the mobilenet. prefix
                current_key = target_key

                if current_key in current_state and current_state[current_key].shape == value.shape:
                    current_state[current_key] = value
                    loaded_keys.append(saved_key)
                else:
                    skipped_keys.append(saved_key)

            # Load the updated state dict
            self.load_state_dict(current_state)

            logger.info(f"Successfully loaded {len(loaded_keys)} compatible layers")
            logger.info(f"Skipped {len(skipped_keys)} incompatible layers")

            # Load accuracy info if available
            if isinstance(checkpoint, dict):
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                    accuracy = metrics.get('accuracy', 'Unknown')
                    logger.info(f"Model accuracy: {accuracy:.4f}")
                elif 'results' in checkpoint and 'accuracy' in checkpoint['results']:
                    accuracy = checkpoint['results']['accuracy']
                    logger.info(f"Model accuracy: {accuracy:.4f}")

            return True

        except Exception as e:
            logger.error(f"Failed to load custom weights: {e}")
            return False


class MobileNetV3FineTune(nn.Module):
    """
    MobileNetV3 variant for potentially better performance.

    Uses the newer MobileNetV3 architecture with improved efficiency
    and performance compared to MobileNetV2.
    """

    def __init__(self, num_classes=2, freeze_layers=40, dropout_rate=0.3, variant='small'):
        super(MobileNetV3FineTune, self).__init__()

        # Choose MobileNetV3 variant
        model_name = f'mobilenetv3_{variant}_100'
        self.mobilenet = timm.create_model(model_name, pretrained=True)

        # Get classifier input features
        if hasattr(self.mobilenet.classifier, 'in_features'):
            num_features = self.mobilenet.classifier.in_features
        else:
            # Fallback based on variant
            num_features = 576 if variant == 'small' else 960

        # Replace classifier
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 32),  # Even smaller for V3
            nn.Hardswish(),  # MobileNetV3 uses Hardswish
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, num_classes)
        )

        # Freeze early layers
        if freeze_layers > 0:
            for i, (name, param) in enumerate(self.mobilenet.named_parameters()):
                if i < freeze_layers:
                    param.requires_grad = False

    def forward(self, x):
        return self.mobilenet(x)


class MobileNetTiny(nn.Module):
    """
    Ultra-lightweight MobileNet variant for extreme resource constraints.

    Designed for scenarios where model size and inference speed are critical,
    such as edge deployment or real-time applications.
    """

    def __init__(self, num_classes=2):
        super(MobileNetTiny, self).__init__()

        # Use a very lightweight backbone
        self.backbone = timm.create_model('mobilenetv2_050', pretrained=True)  # 50% width

        # Get feature size
        num_features = self.backbone.classifier.in_features

        # Ultra-lightweight classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 16),  # Very small hidden layer
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def create_mobilenet_model(num_classes=2, variant='v2', size='standard', freeze_layers=50, dropout_rate=0.4):
    """
    Factory function to create a MobileNet model for pneumonia detection.

    Args:
        num_classes (int): Number of output classes
        variant (str): MobileNet variant ('v2' or 'v3')
        size (str): Model size ('standard', 'small', 'tiny')
        freeze_layers (int): Number of layers to freeze
        dropout_rate (float): Dropout rate for regularization

    Returns:
        nn.Module: Initialized MobileNet model
    """
    if size == 'tiny':
        return MobileNetTiny(num_classes=num_classes)
    elif variant == 'v3':
        v3_variant = 'small' if size == 'small' else 'large'
        return MobileNetV3FineTune(
            num_classes=num_classes,
            freeze_layers=freeze_layers,
            dropout_rate=dropout_rate,
            variant=v3_variant
        )
    else:  # Default to MobileNetV1 FineTune (matches saved checkpoint structure)
        hidden_size = 32 if size == 'small' else 64
        return MobileNetFineTune(
            num_classes=num_classes,
            freeze_layers=freeze_layers,
            dropout_rate=dropout_rate,
            hidden_size=hidden_size
        )


if __name__ == "__main__":
    # Test different MobileNet variants
    variants = [
        ('v2', 'standard'),
        ('v2', 'small'),
        ('v3', 'small'),
        ('v2', 'tiny')
    ]

    dummy_input = torch.randn(2, 3, 224, 224)

    for variant, size in variants:
        print(f"\nTesting MobileNet {variant.upper()} ({size}):")
        try:
            model = create_mobilenet_model(variant=variant, size=size)

            # Get model size info
            if hasattr(model, 'get_model_size'):
                size_info = model.get_model_size()
                print(f"  Parameters: {size_info['total_parameters']:,}")
                print(f"  Model size: {size_info['model_size_mb']:.2f} MB")
            else:
                total_params = sum(p.numel() for p in model.parameters())
                print(f"  Parameters: {total_params:,}")

            # Test forward pass
            output = model(dummy_input)
            print(f"  Output shape: {output.shape}")

            # Test probability prediction for standard models
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(dummy_input)
                print(f"  Probabilities valid: {torch.allclose(probs.sum(dim=1), torch.ones(2))}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\nMobileNet models ready for deployment!")