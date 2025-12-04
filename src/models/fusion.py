"""
Fusion models for pediatric pneumonia detection.

This module contains fusion architectures that combine multiple CNN models
for improved pneumonia detection performance. The main approach combines
Xception and VGG16 models to leverage their complementary strengths.

Key fusion strategies:
- Feature concatenation: Combines feature vectors from different models
- Dual backbone architecture: Processes images through multiple networks
- Weighted fusion: Allows different contribution levels from each model
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm


class XceptionVGGFusion(nn.Module):
    """
    Advanced fusion model combining Xception and VGG16 architectures.

    This model processes each input image through both Xception and VGG16,
    extracts features from each, and combines them for final classification.

    Architecture:
    1. Dual Backbone: Both Xception and VGG16 process the same input image
    2. Feature Extraction: Extract features from each model separately
    3. Feature Fusion: Concatenate features from both models (2048 + 512 = 2560)
    4. Final Classification: Combined features fed to classifier

    Args:
        num_classes (int): Number of output classes (1 for binary with sigmoid, 2 for softmax)
        freeze_early_layers (bool): Whether to freeze early layers for stable training
        dropout_rate (float): Dropout rate for regularization
        use_sigmoid (bool): Whether to use sigmoid activation (for binary classification)
    """

    def __init__(self, num_classes=1, freeze_early_layers=True, dropout_rate=0.5, use_sigmoid=True):
        super(XceptionVGGFusion, self).__init__()

        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid

        # ========== Xception Branch ==========
        self.xception = timm.create_model('xception', pretrained=True)

        # Remove original classification layers
        self.xception.global_pool = nn.Identity()
        self.xception.fc = nn.Identity()

        # Freeze early Xception layers for stable training
        if freeze_early_layers:
            self._freeze_xception_layers(100)

        # ========== VGG16 Branch ==========
        self.vgg = models.vgg16(pretrained=True)

        # Remove original classifier
        self.vgg.classifier = nn.Identity()

        # Freeze early VGG layers
        if freeze_early_layers:
            self._freeze_vgg_layers(10)

        # ========== Feature Processing ==========
        self.xception_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.vgg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ========== Fusion Classifier ==========
        # Combines features: Xception (2048) + VGG16 (512) = 2560 total
        classifier_layers = [
            nn.Dropout(dropout_rate),
            nn.Linear(2048 + 512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, num_classes)
        ]

        if use_sigmoid and num_classes == 1:
            classifier_layers.append(nn.Sigmoid())

        self.fusion_classifier = nn.Sequential(*classifier_layers)

    def _freeze_xception_layers(self, num_layers):
        """Freeze the first num_layers of the Xception backbone."""
        for i, (name, param) in enumerate(self.xception.named_parameters()):
            if i < num_layers:
                param.requires_grad = False
        print(f"Frozen first {num_layers} Xception layers")

    def _freeze_vgg_layers(self, num_layers):
        """Freeze the first num_layers of the VGG16 feature extractor."""
        for i, (name, param) in enumerate(self.vgg.features.named_parameters()):
            if i < num_layers:
                param.requires_grad = False
        print(f"Frozen first {num_layers} VGG16 layers")

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("All fusion model layers unfrozen for fine-tuning")

    def forward(self, x):
        """
        Forward pass through the fusion model.

        Args:
            x (torch.Tensor): Input batch [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Fused predictions [batch_size, num_classes]
        """
        # ========== Xception Path ==========
        xception_features = self.xception(x)  # [batch_size, 2048, 7, 7]
        xception_features = self.xception_pool(xception_features)  # [batch_size, 2048, 1, 1]
        xception_features = xception_features.view(x.size(0), -1)  # [batch_size, 2048]

        # ========== VGG16 Path ==========
        vgg_features = self.vgg.features(x)  # [batch_size, 512, 7, 7]
        vgg_features = self.vgg_pool(vgg_features)  # [batch_size, 512, 1, 1]
        vgg_features = vgg_features.view(x.size(0), -1)  # [batch_size, 512]

        # ========== Feature Fusion ==========
        fused_features = torch.cat((xception_features, vgg_features), dim=1)  # [batch_size, 2560]

        # ========== Final Classification ==========
        predictions = self.fusion_classifier(fused_features)

        return predictions

    def get_individual_features(self, x):
        """
        Extract features from individual model branches.

        Args:
            x (torch.Tensor): Input batch of images

        Returns:
            dict: Dictionary containing features from each branch
        """
        with torch.no_grad():
            # Xception features
            xception_features = self.xception(x)
            xception_features = self.xception_pool(xception_features).view(x.size(0), -1)

            # VGG features
            vgg_features = self.vgg.features(x)
            vgg_features = self.vgg_pool(vgg_features).view(x.size(0), -1)

            return {
                'xception_features': xception_features,
                'vgg_features': vgg_features,
                'fused_features': torch.cat((xception_features, vgg_features), dim=1)
            }

    def get_feature_importance(self, x):
        """
        Analyze the contribution of each model to the prediction.

        Args:
            x (torch.Tensor): Input batch of images

        Returns:
            dict: Feature importance metrics for each model branch
        """
        with torch.no_grad():
            features = self.get_individual_features(x)

            # Calculate feature magnitudes as proxy for importance
            xception_importance = torch.norm(features['xception_features'], dim=1).mean()
            vgg_importance = torch.norm(features['vgg_features'], dim=1).mean()

            total_importance = xception_importance + vgg_importance

            return {
                'xception_importance': xception_importance.item(),
                'vgg_importance': vgg_importance.item(),
                'xception_ratio': (xception_importance / total_importance).item(),
                'vgg_ratio': (vgg_importance / total_importance).item(),
                'total_features': features['fused_features'].shape[1]
            }

    def predict_proba(self, x):
        """
        Get prediction probabilities.

        Args:
            x (torch.Tensor): Input batch of images

        Returns:
            torch.Tensor: Prediction probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            if self.use_sigmoid and self.num_classes == 1:
                return logits  # Already sigmoid
            else:
                return torch.softmax(logits, dim=1)


class WeightedFusionModel(nn.Module):
    """
    Fusion model with learnable weights for combining different model outputs.

    Instead of simple concatenation, this model learns optimal weights
    for combining predictions from different models.
    """

    def __init__(self, num_classes=2, models=None):
        super(WeightedFusionModel, self).__init__()

        if models is None:
            # Default: Create Xception and VGG16 models
            self.model1 = timm.create_model('xception', pretrained=True, num_classes=num_classes)
            self.model2 = timm.create_model('vgg16', pretrained=True, num_classes=num_classes)
        else:
            self.model1, self.model2 = models

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(2))

        # Optional fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_classes * 4, num_classes)
        )

    def forward(self, x):
        # Get predictions from both models
        pred1 = self.model1(x)
        pred2 = self.model2(x)

        # Apply learned weights
        weights = torch.softmax(self.fusion_weights, dim=0)
        weighted_pred1 = pred1 * weights[0]
        weighted_pred2 = pred2 * weights[1]

        # Concatenate and process through fusion MLP
        combined = torch.cat([weighted_pred1, weighted_pred2], dim=1)
        final_prediction = self.fusion_mlp(combined)

        return final_prediction

    def get_fusion_weights(self):
        """Get the current fusion weights."""
        weights = torch.softmax(self.fusion_weights, dim=0)
        return {
            'model1_weight': weights[0].item(),
            'model2_weight': weights[1].item()
        }


def create_fusion_model(model_type='xception_vgg', num_classes=2, **kwargs):
    """
    Factory function to create fusion models.

    Args:
        model_type (str): Type of fusion model to create
        num_classes (int): Number of output classes
        **kwargs: Additional arguments for model creation

    Returns:
        nn.Module: Initialized fusion model
    """
    if model_type == 'xception_vgg':
        # Convert num_classes for binary classification
        if num_classes == 2:
            use_sigmoid = kwargs.get('use_sigmoid', False)
            model_classes = 2 if not use_sigmoid else 1
        else:
            model_classes = num_classes
            use_sigmoid = False

        return XceptionVGGFusion(
            num_classes=model_classes,
            freeze_early_layers=kwargs.get('freeze_early_layers', True),
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            use_sigmoid=use_sigmoid
        )

    elif model_type == 'weighted':
        return WeightedFusionModel(
            num_classes=num_classes,
            models=kwargs.get('models', None)
        )

    else:
        raise ValueError(f"Unknown fusion model type: {model_type}")


if __name__ == "__main__":
    # Test fusion models
    print("Testing Xception + VGG16 Fusion Model:")
    model = create_fusion_model('xception_vgg', num_classes=2)

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test feature importance
    importance = model.get_feature_importance(dummy_input)
    print(f"Feature importance - Xception: {importance['xception_ratio']:.3f}, VGG16: {importance['vgg_ratio']:.3f}")

    # Test individual features
    features = model.get_individual_features(dummy_input)
    print(f"Xception features: {features['xception_features'].shape}")
    print(f"VGG features: {features['vgg_features'].shape}")
    print(f"Fused features: {features['fused_features'].shape}")

    print("\nTesting Weighted Fusion Model:")
    weighted_model = create_fusion_model('weighted', num_classes=2)
    weighted_output = weighted_model(dummy_input)
    print(f"Weighted fusion output shape: {weighted_output.shape}")

    fusion_weights = weighted_model.get_fusion_weights()
    print(f"Fusion weights: {fusion_weights}")

    print("\nFusion models ready for training!")