"""
Xception-LSTM hybrid models for pediatric pneumonia detection.

This module contains experimental CNN-LSTM architectures that combine
convolutional feature extraction with sequential processing for medical
image analysis.

Key innovations:
- Spatial tokenization: Converting 2D feature maps to 1D token sequences
- Sequential processing: Using LSTM to learn relationships between spatial regions
- Hybrid architecture: Combining CNN spatial features with LSTM contextual understanding
"""

import torch
import torch.nn as nn
import timm
import numpy as np


class XceptionLSTM(nn.Module):
    """
    Experimental Xception-LSTM architecture for pneumonia detection.

    This model combines the feature extraction power of Xception with
    the sequential processing capabilities of LSTM for spatial analysis.

    Research Innovation:
    - Treats spatial feature maps as sequences of tokens
    - Uses LSTM to learn relationships between different lung regions
    - Investigates whether sequential processing improves medical image analysis

    Architecture:
    1. Xception Backbone: Extract rich spatial features (2048 channels)
    2. Spatial Tokenization: Convert 7x7 feature map to 49 spatial tokens
    3. LSTM Processing: Sequential analysis of spatial relationships
    4. Feature Integration: Combine spatial and sequential information

    Args:
        freeze_layers (int): Number of early Xception layers to freeze
        use_channel_reduction (bool): Whether to reduce channel dimensions for efficiency
        reduced_dim (int): Target dimensions if using channel reduction
        lstm_units (int): Number of LSTM hidden units
        dropout_rate (float): Dropout rate for regularization
    """

    def __init__(self, freeze_layers=100, use_channel_reduction=False,
                 reduced_dim=512, lstm_units=256, dropout_rate=0.46):
        super(XceptionLSTM, self).__init__()

        self.use_channel_reduction = use_channel_reduction
        self.lstm_units = lstm_units

        # ========== Xception Backbone ==========
        # Use Xception for feature extraction only (no classification head)
        self.xception = timm.create_model("xception", pretrained=True, features_only=True)

        # Freeze early layers for stable training on medical data
        if freeze_layers > 0:
            self._freeze_xception_layers(freeze_layers)

        # ========== Channel Reduction (Optional) ==========
        # Reduce computational complexity while maintaining performance
        if use_channel_reduction:
            self.channel_reducer = nn.Sequential(
                nn.Conv2d(2048, reduced_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduced_dim),
                nn.ReLU(inplace=True)
            )
            lstm_input_dim = reduced_dim
            print(f"Using channel reduction: 2048 -> {reduced_dim} dimensions")
        else:
            lstm_input_dim = 2048
            print("Using full 2048-dimensional features")

        # ========== LSTM Sequential Processor ==========
        # Process spatial tokens sequentially to learn regional relationships
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_units,
            batch_first=True,
            dropout=0.2 if freeze_layers > 50 else 0.0  # LSTM dropout for longer training
        )

        # ========== Classification Head ==========
        # Transform LSTM output to final prediction
        self.classifier = nn.Sequential(
            nn.Linear(lstm_units, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)  # Binary classification
        )

    def _freeze_xception_layers(self, num_layers):
        """Freeze the first num_layers of the Xception backbone."""
        frozen_count = 0
        for name, param in self.xception.named_parameters():
            if frozen_count < num_layers:
                param.requires_grad = False
                frozen_count += 1
            else:
                break
        print(f"Frozen first {frozen_count} Xception layers for stability")

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("All XceptionLSTM layers unfrozen for fine-tuning")

    def forward(self, x):
        """
        Forward pass through the experimental architecture.

        Args:
            x (torch.Tensor): Input batch [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Raw prediction logits [batch_size, 1]
        """
        # ========== Feature Extraction ==========
        # Extract spatial features using Xception backbone
        features = self.xception(x)[-1]  # Take last feature stage: [batch, 2048, 7, 7]

        # ========== Optional Channel Reduction ==========
        if self.use_channel_reduction:
            features = self.channel_reducer(features)  # [batch, reduced_dim, 7, 7]

        # ========== Spatial Tokenization ==========
        # Convert 2D feature maps to sequence of spatial tokens
        batch_size, channels, height, width = features.shape

        # Flatten spatial dimensions and transpose for LSTM input
        # [batch, channels, height*width] -> [batch, height*width, channels]
        spatial_tokens = features.flatten(2).permute(0, 2, 1)
        # Result: [batch_size, 49, channels] - 49 spatial tokens per image

        # ========== Sequential Processing ==========
        # Process spatial tokens through LSTM to learn regional relationships
        lstm_output, (hidden_state, _) = self.lstm(spatial_tokens)

        # Use final hidden state as global representation
        # hidden_state shape: [1, batch_size, lstm_units]
        global_features = hidden_state.squeeze(0)  # [batch_size, lstm_units]

        # ========== Classification ==========
        # Transform global features to final prediction
        logits = self.classifier(global_features)  # [batch_size, 1]

        return logits

    def get_spatial_attention(self, x):
        """
        Analyze spatial attention patterns across the image.
        Returns LSTM attention weights across spatial tokens.

        Args:
            x (torch.Tensor): Input batch of images

        Returns:
            torch.Tensor: Attention weights reshaped to spatial dimensions [batch, 7, 7]
        """
        with torch.no_grad():
            features = self.xception(x)[-1]
            if self.use_channel_reduction:
                features = self.channel_reducer(features)

            spatial_tokens = features.flatten(2).permute(0, 2, 1)
            lstm_output, _ = self.lstm(spatial_tokens)

            # Calculate attention as magnitude of LSTM outputs
            attention = torch.norm(lstm_output, dim=2)  # [batch, num_tokens]
            attention = torch.softmax(attention, dim=1)  # Normalize to probabilities

            return attention.reshape(-1, 7, 7)  # Reshape to spatial dimensions

    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization.

        Args:
            x (torch.Tensor): Input batch of images

        Returns:
            dict: Dictionary containing various feature representations
        """
        with torch.no_grad():
            # CNN features
            cnn_features = self.xception(x)[-1]  # [batch, 2048, 7, 7]

            if self.use_channel_reduction:
                reduced_features = self.channel_reducer(cnn_features)
                features_for_lstm = reduced_features
            else:
                reduced_features = None
                features_for_lstm = cnn_features

            # Spatial tokens
            spatial_tokens = features_for_lstm.flatten(2).permute(0, 2, 1)

            # LSTM output
            lstm_output, (hidden_state, _) = self.lstm(spatial_tokens)

            return {
                'cnn_features': cnn_features,
                'reduced_features': reduced_features,
                'spatial_tokens': spatial_tokens,
                'lstm_output': lstm_output,
                'hidden_state': hidden_state.squeeze(0)
            }

    def predict_proba(self, x):
        """
        Get prediction probabilities.

        Args:
            x (torch.Tensor): Input batch of images

        Returns:
            torch.Tensor: Prediction probabilities [batch_size, 1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def count_parameters(self):
        """Count trainable parameters for model analysis."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total,
            'trainable_parameters': trainable,
            'frozen_parameters': total - trainable
        }


class CustomCNNLSTM(nn.Module):
    """
    Simple 3-layer CNN + LSTM architecture for pneumonia detection.

    This lightweight model serves as a comparison baseline to understand
    whether complex backbones are necessary for CNN-LSTM architectures.

    Research Questions:
    - Can simple CNN + LSTM compete with advanced architectures?
    - What's the minimum complexity needed for effective pneumonia detection?
    - How do computational requirements scale with performance?

    Args:
        dropout_rate (float): Dropout probability for regularization
        lstm_hidden (int): LSTM hidden units (smaller than Xception-LSTM)
        cnn_channels (list): Number of channels for each CNN layer
    """

    def __init__(self, dropout_rate=0.27, lstm_hidden=50, cnn_channels=None):
        super(CustomCNNLSTM, self).__init__()

        if cnn_channels is None:
            cnn_channels = [200, 150, 100]

        self.lstm_hidden = lstm_hidden

        # ========== Simple CNN Backbone ==========
        # 3-layer CNN for basic feature extraction
        self.cnn_backbone = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Reduce spatial dimensions

            # Second convolutional block
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Further spatial reduction

            # Third convolutional block
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Adaptive pooling to ensure 7x7 output (same as Xception-LSTM)
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # ========== Sequential Processor ==========
        # LSTM for processing spatial tokens
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],  # CNN output channels
            hidden_size=lstm_hidden,     # Smaller than Xception-LSTM
            batch_first=True
        )

        # ========== Classification Head ==========
        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)  # Binary classification
        )

    def forward(self, x):
        """
        Forward pass through the lightweight architecture.

        Args:
            x (torch.Tensor): Input batch [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Raw prediction scores [batch_size, 1]
        """
        # ========== Feature Extraction ==========
        # Extract features using simple CNN
        cnn_features = self.cnn_backbone(x)  # [batch_size, channels, 7, 7]

        # ========== Spatial Tokenization ==========
        # Convert to spatial tokens (same approach as Xception-LSTM)
        batch_size, channels, height, width = cnn_features.shape

        # Flatten and transpose: [batch, channels, H*W] -> [batch, H*W, channels]
        spatial_tokens = cnn_features.flatten(2).permute(0, 2, 1)
        # Result: [batch_size, 49, channels] - same token count, fewer features per token

        # ========== Sequential Processing ==========
        # Process spatial tokens through LSTM
        lstm_output, (hidden_state, _) = self.lstm(spatial_tokens)

        # Use final hidden state
        global_features = hidden_state.squeeze(0)  # [batch_size, lstm_hidden]

        # ========== Classification ==========
        logits = self.classifier(global_features)

        return logits

    def get_feature_maps(self, x):
        """Extract feature maps for analysis."""
        with torch.no_grad():
            cnn_features = self.cnn_backbone(x)
            spatial_tokens = cnn_features.flatten(2).permute(0, 2, 1)
            lstm_output, (hidden_state, _) = self.lstm(spatial_tokens)

            return {
                'cnn_features': cnn_features,
                'spatial_tokens': spatial_tokens,
                'lstm_output': lstm_output,
                'hidden_state': hidden_state.squeeze(0)
            }

    def predict_proba(self, x):
        """Get prediction probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def count_parameters(self):
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        return {
            'total_parameters': total,
            'trainable_parameters': total  # All parameters are trainable in this simple model
        }


def create_xception_lstm_model(model_type='advanced', **kwargs):
    """
    Factory function to create Xception-LSTM models.

    Args:
        model_type (str): Type of model ('advanced' or 'lightweight')
        **kwargs: Additional arguments for model creation

    Returns:
        nn.Module: Initialized CNN-LSTM model
    """
    if model_type == 'advanced':
        return XceptionLSTM(
            freeze_layers=kwargs.get('freeze_layers', 100),
            use_channel_reduction=kwargs.get('use_channel_reduction', False),
            reduced_dim=kwargs.get('reduced_dim', 512),
            lstm_units=kwargs.get('lstm_units', 256),
            dropout_rate=kwargs.get('dropout_rate', 0.46)
        )
    elif model_type == 'lightweight':
        return CustomCNNLSTM(
            dropout_rate=kwargs.get('dropout_rate', 0.27),
            lstm_hidden=kwargs.get('lstm_hidden', 50),
            cnn_channels=kwargs.get('cnn_channels', None)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test both models
    print("Testing Xception-LSTM Models:")

    # Test advanced model
    print("\n1. Advanced Xception-LSTM:")
    model = create_xception_lstm_model('advanced')

    param_info = model.count_parameters()
    print(f"   Total parameters: {param_info['total_parameters']:,}")
    print(f"   Trainable parameters: {param_info['trainable_parameters']:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"   Output shape: {output.shape}")

    # Test probability prediction
    probs = model.predict_proba(dummy_input)
    print(f"   Probabilities shape: {probs.shape}")

    # Test spatial attention
    attention = model.get_spatial_attention(dummy_input)
    print(f"   Spatial attention shape: {attention.shape}")

    # Test lightweight model
    print("\n2. Lightweight Custom CNN-LSTM:")
    lightweight_model = create_xception_lstm_model('lightweight')

    lightweight_params = lightweight_model.count_parameters()
    print(f"   Total parameters: {lightweight_params['total_parameters']:,}")

    lightweight_output = lightweight_model(dummy_input)
    print(f"   Output shape: {lightweight_output.shape}")

    # Parameter comparison
    ratio = param_info['total_parameters'] / lightweight_params['total_parameters']
    print(f"\n3. Model Comparison:")
    print(f"   Parameter ratio: {ratio:.1f}x more parameters in advanced model")
    print(f"   Advanced model suitable for: High accuracy requirements")
    print(f"   Lightweight model suitable for: Resource-constrained deployment")

    print("\nXception-LSTM models ready for training!")