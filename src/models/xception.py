"""
Xception model for pediatric pneumonia detection.

This module contains the XceptionFineTune model class that uses transfer learning
to adapt a pre-trained Xception model for medical image classification.

Xception uses depthwise separable convolutions for efficient feature extraction
and is particularly good at capturing fine details in medical images.
"""

import torch
import torch.nn as nn
import timm


class XceptionFineTune(nn.Module):
    """
    Xception model adapted for pneumonia detection.

    Uses transfer learning: starts with ImageNet pre-trained weights
    and adapts them for medical image classification.

    Architecture:
    - Pre-trained Xception backbone (feature extraction)
    - Custom global average pooling
    - Custom classification head with dropout for regularization

    Args:
        num_classes (int): Number of output classes (default: 2 for Normal/Pneumonia)
        freeze_layers (int): Number of early layers to freeze during training (default: 100)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
    """

    def __init__(self, num_classes=2, freeze_layers=100, dropout_rate=0.5):
        super(XceptionFineTune, self).__init__()

        # Load pre-trained Xception from timm library
        self.xception = timm.create_model('xception', pretrained=True)

        # Remove the original classification layers
        self.xception.global_pool = nn.Identity()  # Remove global pooling
        self.xception.fc = nn.Identity()  # Remove final fully connected layer

        # Add our own pooling and classification layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

        # Custom classification head for pneumonia detection
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout prevents overfitting
            nn.Linear(2048, 128),  # Xception features: 2048 -> 128
            nn.ReLU(),  # Activation function
            nn.Dropout(dropout_rate * 0.6),  # Reduced dropout for second layer
            nn.Linear(128, num_classes)  # Final prediction: 128 -> num_classes
        )

        # Freeze early layers for stable training
        self.freeze_layers = freeze_layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, num_layers):
        """
        Freeze the first num_layers of the Xception backbone.

        Args:
            num_layers (int): Number of layers to freeze from the beginning
        """
        for i, (name, param) in enumerate(self.xception.named_parameters()):
            if i < num_layers:
                param.requires_grad = False  # Don't update these weights
        print(f"Frozen first {num_layers} layers of Xception backbone")

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.xception.parameters():
            param.requires_grad = True
        print("All Xception layers unfrozen for fine-tuning")

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input batch of images [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Raw prediction logits [batch_size, num_classes]
        """
        # Extract features using Xception backbone
        x = self.xception(x)  # [batch_size, 2048, 7, 7]

        # Global average pooling: average across spatial dimensions
        x = self.pool(x)  # [batch_size, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 2048]

        # Classification
        x = self.classifier(x)  # [batch_size, num_classes]

        return x

    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization.

        Args:
            x (torch.Tensor): Input batch of images

        Returns:
            torch.Tensor: Feature maps before global pooling [batch_size, 2048, 7, 7]
        """
        with torch.no_grad():
            features = self.xception(x)
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


def create_xception_model(num_classes=2, pretrained=True, freeze_layers=100):
    """
    Factory function to create an Xception model for pneumonia detection.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pre-trained weights
        freeze_layers (int): Number of layers to freeze

    Returns:
        XceptionFineTune: Initialized model
    """
    model = XceptionFineTune(
        num_classes=num_classes,
        freeze_layers=freeze_layers if pretrained else 0
    )
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_xception_model()
    print(f"Created Xception model with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test probability prediction
    probs = model.predict_proba(dummy_input)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=1), torch.ones(2))}")