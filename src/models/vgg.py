"""
VGG16 model for pediatric pneumonia detection.

This module contains the VGG16FineTune model class that uses transfer learning
to adapt a pre-trained VGG16 model for medical image classification.

VGG16 is a classic CNN architecture with 16 layers that uses small 3x3 convolution
filters throughout. It provides a reliable baseline for medical image analysis.
"""

import torch
import torch.nn as nn
import timm


class VGG16FineTune(nn.Module):
    """
    VGG16 model adapted for pneumonia detection.

    Classic CNN architecture providing a reliable baseline
    for medical image classification tasks.

    Architecture:
    - Pre-trained VGG16 backbone (feature extraction)
    - Custom classification head with dropout for regularization
    - Selective layer freezing for stable transfer learning

    Args:
        num_classes (int): Number of output classes (default: 2 for Normal/Pneumonia)
        freeze_layers (int): Number of early feature layers to freeze (default: 10)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
    """

    def __init__(self, num_classes=2, freeze_layers=10, dropout_rate=0.5):
        super(VGG16FineTune, self).__init__()

        # Load pre-trained VGG16 from timm
        self.vgg = timm.create_model('vgg16', pretrained=True)

        # Get number of features from pre-classifier layer
        if hasattr(self.vgg, 'pre_logits'):
            num_features = self.vgg.pre_logits.in_features
        else:
            # Fallback: manually get feature size from classifier
            num_features = self.vgg.classifier.in_features if hasattr(self.vgg.classifier, 'in_features') else 25088

        # Replace the classifier head
        self.vgg.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),  # Intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),  # Reduced dropout for second layer
            nn.Linear(256, num_classes)  # Final prediction
        )

        # Store configuration
        self.freeze_layers = freeze_layers
        self.num_classes = num_classes

        # Freeze early convolutional layers
        if freeze_layers > 0:
            self._freeze_feature_layers(freeze_layers)

    def _freeze_feature_layers(self, num_layers):
        """
        Freeze the first num_layers of the VGG16 feature extractor.

        Args:
            num_layers (int): Number of feature layers to freeze from the beginning
        """
        frozen_count = 0
        for i, (name, param) in enumerate(self.vgg.features.named_parameters()):
            if frozen_count < num_layers:
                param.requires_grad = False
                frozen_count += 1
            else:
                break
        print(f"Frozen first {frozen_count} VGG16 feature layers")

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.vgg.parameters():
            param.requires_grad = True
        print("All VGG16 layers unfrozen for fine-tuning")

    def forward(self, x):
        """
        Forward pass through VGG16.

        Args:
            x (torch.Tensor): Input batch of images [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Raw prediction logits [batch_size, num_classes]
        """
        return self.vgg(x)

    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization.

        Args:
            x (torch.Tensor): Input batch of images

        Returns:
            torch.Tensor: Feature maps from VGG16 features
        """
        with torch.no_grad():
            features = self.vgg.features(x)
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

    def get_classifier_weights(self):
        """
        Get the weights of the final classification layer.

        Returns:
            torch.Tensor: Classification layer weights
        """
        # Navigate to the final linear layer
        if hasattr(self.vgg.head, '__iter__'):
            # Sequential classifier
            for layer in reversed(self.vgg.head):
                if isinstance(layer, nn.Linear):
                    return layer.weight.data
        return None


class VGG16Lightweight(nn.Module):
    """
    Lightweight version of VGG16 with fewer parameters.

    Suitable for environments with limited computational resources
    while maintaining the architectural benefits of VGG.
    """

    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(VGG16Lightweight, self).__init__()

        # Load pre-trained VGG16 but modify classifier
        self.backbone = timm.create_model('vgg16', pretrained=True)

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Lightweight classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),  # Reduce spatial dimensions
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 4, 64),  # Much smaller intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.backbone.features(x)
        x = self.classifier(features)
        return x


def create_vgg16_model(num_classes=2, variant='standard', freeze_layers=10):
    """
    Factory function to create a VGG16 model for pneumonia detection.

    Args:
        num_classes (int): Number of output classes
        variant (str): Model variant ('standard' or 'lightweight')
        freeze_layers (int): Number of layers to freeze (only for standard)

    Returns:
        nn.Module: Initialized VGG16 model
    """
    if variant == 'lightweight':
        model = VGG16Lightweight(num_classes=num_classes)
    else:
        model = VGG16FineTune(
            num_classes=num_classes,
            freeze_layers=freeze_layers
        )
    return model


if __name__ == "__main__":
    # Test standard VGG16 model
    print("Testing VGG16 Standard Model:")
    model = create_vgg16_model(variant='standard')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test probability prediction
    probs = model.predict_proba(dummy_input)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=1), torch.ones(2))}")

    print("\nTesting VGG16 Lightweight Model:")
    lightweight_model = create_vgg16_model(variant='lightweight')
    lightweight_params = sum(p.numel() for p in lightweight_model.parameters())
    print(f"Lightweight model parameters: {lightweight_params:,}")

    lightweight_output = lightweight_model(dummy_input)
    print(f"Lightweight output shape: {lightweight_output.shape}")