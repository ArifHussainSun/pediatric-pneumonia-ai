# Windows Desktop Deployment Guide

Complete guide for deploying MobileNet pneumonia detection on Windows tablets, specifically optimized for Microsoft Surface Pro devices.

## Overview

This deployment provides a native Windows desktop application for offline pneumonia detection using trained MobileNet models. The app is optimized for touch interfaces and designed for clinical environments.

## System Requirements

### Minimum Requirements

- **OS**: Windows 10 (version 1903 or later) or Windows 11
- **RAM**: 8GB (16GB recommended for Surface Pro)
- **Storage**: 2GB free space
- **Processor**: Intel Core i5 or equivalent
- **Graphics**: DirectX 11 compatible (DirectML acceleration)

### Recommended Hardware

- **Microsoft Surface Pro 7+** with Intel Core i5, 16GB RAM
- **Surface Pro 8/9** for optimal performance
- **Touch display** for best user experience

## Installation Guide

### Method 1: Pre-built Executable (Recommended)

1. **Download** the pre-built package
2. **Extract** to desired location
3. **Run** `install.bat` as administrator
4. **Launch** from desktop shortcut

### Method 2: Build from Source

```bash
# 1. Clone repository
git clone <repository-url>
cd pediatric-pneumonia-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Export trained model
python scripts/export_windows.py \
    --model_path outputs/checkpoint_best.pth \
    --model_type mobilenet \
    --output_dir windows_exports

# 4. Build executable
python scripts/build_windows_exe.py

# 5. Install
cd dist
install.bat
```

## Usage Instructions

### Basic Workflow

1. **Launch** Pneumonia Detector from desktop
2. **Select Image** - Click to browse for chest X-ray
3. **Analyze** - Click "Analyze X-Ray" button
4. **Review Results** - View diagnosis and confidence scores

### Image Requirements

**Supported Formats:**

- PNG, JPEG, TIFF, BMP
- Maximum file size: 50MB
- Recommended resolution: 512×512 or higher

**Image Quality Guidelines:**

- Clear chest X-ray images
- Proper contrast and brightness
- No significant artifacts or noise
- Standard radiographic positioning

### Result Interpretation

The application provides:

- **Primary Diagnosis**: NORMAL or PNEUMONIA
- **Confidence Score**: Percentage confidence in diagnosis
- **Detailed Probabilities**: Individual class probabilities
- **Performance Metrics**: Inference time and model info

## Features

### Touch-Optimized Interface

- **Large buttons** designed for touch interaction
- **Responsive layout** adapts to screen orientation
- **Visual feedback** for all user interactions
- **Drag-and-drop** image loading support

### Performance Optimizations

- **ONNX Runtime** for maximum inference speed
- **DirectML acceleration** on compatible hardware
- **Memory optimization** for extended usage
- **Automatic model loading** with fallback paths

### Reliability Features

- **Crash recovery** system with auto-save
- **Session restoration** after unexpected closure
- **Comprehensive error handling** with user feedback
- **Input validation** for file types and sizes

### Clinical Safety

- **Medical disclaimers** prominently displayed
- **Warning notifications** for pneumonia detection
- **Audit trail** of analysis results
- **Offline operation** for data privacy

## Performance Benchmarks

### Surface Pro 7+ (Intel i5, 16GB RAM)

| Metric             | Performance         |
| ------------------ | ------------------- |
| **Inference Time** | 20-50ms             |
| **Model Loading**  | 2-3 seconds         |
| **Memory Usage**   | 200-300MB           |
| **Storage**        | 150MB (app + model) |
| **Startup Time**   | 3-5 seconds         |

### Model Specifications

| Model           | Size | Accuracy | Speed | Memory |
| --------------- | ---- | -------- | ----- | ------ |
| **MobileNetV1** | 25MB | 94.2%    | 25ms  | 250MB  |
| **Quantized**   | 8MB  | 93.1%    | 15ms  | 180MB  |

## Troubleshooting

### Common Issues

**App won't start:**

- Verify Windows version compatibility
- Check antivirus software interference
- Run as administrator

**Model not found:**

- Ensure `.onnx` file is in app directory
- Check file permissions
- Verify model export completed successfully

**Slow performance:**

- Update graphics drivers for DirectML
- Close other memory-intensive applications
- Check Windows power management settings

**Touch interface issues:**

- Calibrate touch screen in Windows settings
- Verify tablet mode is enabled
- Check for Windows updates

### Error Messages

**"Model loading failed":**

- Model file corrupted or incompatible
- Re-export model using latest export script

**"DirectML not available":**

- Graphics drivers need updating
- Hardware doesn't support DirectML (fallback to CPU)

**"Image preprocessing failed":**

- Unsupported image format
- File corrupted or too large

## Advanced Configuration

### Model Paths

The app searches for models in this order:

1. `windows_exports/mobilenet_windows.onnx`
2. `../windows_exports/mobilenet_windows.onnx`
3. `mobilenet_windows.onnx`

### Performance Tuning

For optimal performance on Surface Pro:

- Enable "Best performance" power mode
- Disable Windows animations
- Close unnecessary background apps
- Use SSD storage for model files

### Clinical Deployment

For healthcare environments:

- Configure automatic Windows updates
- Set up centralized model management
- Implement audit logging
- Train users on proper image quality

## File Structure

```
Windows Deployment/
├── PneumoniaDetector.exe      # Main application
├── mobilenet_windows.onnx     # ONNX model file
├── install.bat               # Installation script
└── docs/                     # Documentation
    └── windows_deployment.md # This file
```

## Security Considerations

### Data Privacy

- **All processing local** - no data transmitted
- **No internet required** for inference
- **Temporary files cleaned** on app closure
- **No user data collection**

### Model Security

- Model files should be digitally signed
- Verify model integrity before deployment
- Implement access controls in clinical settings

## Updates and Maintenance

### Model Updates

1. Export new model using export script
2. Replace `.onnx` file in app directory
3. Restart application

### App Updates

1. Download new version
2. Run uninstaller for old version
3. Install new version using `install.bat`

## Support

### Diagnostic Information

The app provides system information for support:

- Model version and path
- Hardware acceleration status
- Performance metrics
- Error logs

### Common Solutions

- **Restart application** for temporary issues
- **Clear temp files** in `%TEMP%` folder
- **Reinstall** for persistent problems
- **Update drivers** for performance issues
