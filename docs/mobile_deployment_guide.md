# Mobile Deployment Guide for Pediatric Pneumonia Detection

Development notes for deploying pneumonia detection models on mobile devices for clinical applications. Covers offline AI inference on iPads and Android tablets.

## Overview

The mobile deployment system provides:

- Offline inference capabilities on mobile devices
- Real-time predictions for clinical point-of-care use
- Cross-platform compatibility (iOS and Android)
- Performance optimization for mobile hardware constraints
- Medical-grade accuracy preservation on resource-limited devices

## Quick Start

### iOS Deployment (iPads)

1. **Export to CoreML**:

```bash
python examples/export_model.py \
    --model_path outputs/mobilenet_model.pth \
    --model_type mobilenet \
    --formats coreml \
    --deployment mobile_devices \
    --platform ios
```

2. **Integrate in iOS App**:

```swift
import CoreML

let model = try! PneumoniaDetectionModel(configuration: MLModelConfiguration())
let prediction = try! model.prediction(chest_xray_image: image)
```

### Android Deployment (Tablets)

1. **Export to TensorFlow Lite**:

```bash
python examples/export_model.py \
    --model_path outputs/mobilenet_model.pth \
    --model_type mobilenet \
    --formats tflite \
    --deployment mobile_devices \
    --platform android
```

2. **Integrate in Android App**:

```java
import org.tensorflow.lite.Interpreter;

Interpreter tflite = new Interpreter(loadModelFile());
tflite.run(inputBuffer, outputBuffer);
```

## Supported Models and Platforms

### Model Compatibility Matrix

| Model Type | iOS (CoreML) | Android (TFLite) | Performance | Size  | Accuracy |
| ---------- | ------------ | ---------------- | ----------- | ----- | -------- |
| MobileNet  | Excellent    | Excellent        | Fast        | 15MB  | 94.2%    |
| VGG16      | Good         | Good             | Medium      | 45MB  | 95.1%    |
| Xception   | Limited      | Good             | Slow        | 85MB  | 96.3%    |
| Fusion     | Not Rec.     | Limited          | Very Slow   | 150MB | 97.1%    |

**Note**: MobileNet recommended for mobile deployment due to optimal balance of speed, size, and accuracy.

### Platform Requirements

#### iOS Requirements

- **Minimum**: iOS 13.0
- **Recommended**: iOS 14.0+ for Neural Engine acceleration
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 100MB for model and app

#### Android Requirements

- **Minimum**: Android API Level 21 (Android 5.0)
- **Recommended**: API Level 27+ for NNAPI acceleration
- **Memory**: 2GB RAM minimum, 3GB recommended
- **Storage**: 100MB for model and app

## Deployment Workflow

### 1. Model Selection and Training

Choose the appropriate model based on your deployment requirements:

```bash
# For maximum accuracy (clinical validation)
python examples/train_model.py --model_type vgg --epochs 50 --lr 0.001

# For mobile optimization (point-of-care)
python examples/train_model.py --model_type mobilenet --epochs 50 --lr 0.001
```

### 2. Model Optimization

Optimize the trained model for mobile deployment:

```python
from src.mobile import MobileOptimizer

# Initialize optimizer
optimizer = MobileOptimizer(model, 'mobilenet', 'ios_tablet')

# Apply optimizations
results = optimizer.optimize(
    accuracy_threshold=0.95,
    max_size_reduction=0.75,
    validate_fn=your_validation_function
)
```

### 3. Model Export

Export optimized models for target platforms:

```bash
# Complete mobile deployment export
python examples/export_for_deployment.py \
    --model_path outputs/mobilenet_optimized.pth \
    --deployment mobile_devices \
    --validate \
    --benchmark
```

### 4. Performance Validation

Validate mobile model performance:

```python
from src.mobile import MobileBenchmark

benchmark = MobileBenchmark('clinical_validation')
results = benchmark.benchmark_model(
    model_path='exports/mobilenet_ios.mlmodel',
    format_type='coreml',
    test_data=test_dataset
)
```

### 5. Mobile App Integration

#### iOS Integration

1. **Add CoreML Model to Xcode Project**:

   - Drag `.mlmodel` file into Xcode project
   - Ensure "Add to target" is checked

2. **Implement Prediction Code**:

```swift
import CoreML
import Vision

class PneumoniaDetector {
    private let model: VNCoreMLModel

    init() {
        guard let model = try? VNCoreMLModel(for: MobileNetPneumonia().model) else {
            fatalError("Failed to load model")
        }
        self.model = model
    }

    func predictPneumonia(image: UIImage, completion: @escaping (String, Float) -> Void) {
        guard let cgImage = image.cgImage else { return }

        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else { return }

            DispatchQueue.main.async {
                completion(topResult.identifier, topResult.confidence)
            }
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
    }
}
```

#### Android Integration

1. **Add TensorFlow Lite Dependencies**:

```gradle
implementation 'org.tensorflow:tensorflow-lite:2.8.0'
implementation 'org.tensorflow:tensorflow-lite-gpu:2.8.0'
```

2. **Implement Prediction Code**:

```java
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;

public class PneumoniaDetector {
    private Interpreter tflite;

    public PneumoniaDetector(Context context) {
        try {
            tflite = new Interpreter(loadModelFile(context, "mobilenet_pneumonia.tflite"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public PneumoniaPrediction predict(Bitmap image) {
        // Preprocess image
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(image);

        // Run inference
        float[][] output = new float[1][2];
        tflite.run(tensorImage.getBuffer(), output);

        // Process results
        float normalProb = output[0][0];
        float pneumoniaProb = output[0][1];

        return new PneumoniaPrediction(
            pneumoniaProb > normalProb ? "PNEUMONIA" : "NORMAL",
            Math.max(normalProb, pneumoniaProb)
        );
    }
}
```

## Performance Optimization

### iOS Optimization

1. **Neural Engine Acceleration**:

```swift
let config = MLModelConfiguration()
config.computeUnits = .all  // Use Neural Engine when available
let model = try! MobileNetPneumonia(configuration: config)
```

2. **Memory Management**:

```swift
// Use autoreleasepool for batch processing
autoreleasepool {
    let prediction = try! model.prediction(chest_xray_image: image)
    // Process prediction
}
```

### Android Optimization

1. **NNAPI Acceleration**:

```java
Interpreter.Options options = new Interpreter.Options();
options.setUseNNAPI(true);
Interpreter tflite = new Interpreter(modelBuffer, options);
```

2. **GPU Acceleration**:

```java
GpuDelegate gpuDelegate = new GpuDelegate();
Interpreter.Options options = new Interpreter.Options();
options.addDelegate(gpuDelegate);
```

## Clinical Integration Guidelines

### User Interface Considerations

1. **Image Capture Guidelines**:

   - Provide real-time feedback for image quality
   - Ensure proper X-ray positioning indicators
   - Include confidence score display
   - Show prediction reasoning when possible

2. **Result Presentation**:

```
┌─────────────────────────────────┐
│ Pneumonia Detection Result      │
├─────────────────────────────────┤
│ Prediction: PNEUMONIA           │
│ Confidence: 92.4%               │
│ Processing Time: 145ms          │
└─────────────────────────────────┘
```

3. **Error Handling**:
   - Graceful fallback for poor image quality
   - Clear error messages for technical issues
   - Offline functionality with sync capability

### Data Management

1. **Privacy and Security**:

   - Process images locally (no cloud upload)
   - Implement secure local storage
   - Provide data export capabilities
   - Follow HIPAA compliance guidelines

2. **Audit Trail**:

```python
prediction_log = {
    'timestamp': datetime.now().isoformat(),
    'patient_id': anonymized_id,
    'prediction': 'PNEUMONIA',
    'confidence': 0.924,
    'model_version': '1.0.0',
    'processing_time_ms': 145,
    'image_quality_score': 0.87
}
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**:

   - Verify model file integrity
   - Check iOS/Android version compatibility
   - Ensure sufficient device memory

2. **Poor Performance**:

   - Enable hardware acceleration
   - Reduce input image size if needed
   - Use quantized models for older devices

3. **Accuracy Issues**:
   - Validate image preprocessing
   - Check for proper normalization
   - Ensure consistent input format

### Performance Benchmarks

Target performance metrics for clinical deployment:

| Metric         | iOS Target | Android Target | Embedded Target |
| -------------- | ---------- | -------------- | --------------- |
| Inference Time | <200ms     | <300ms         | <500ms          |
| Memory Usage   | <100MB     | <80MB          | <50MB           |
| Battery Impact | <5%/hour   | <8%/hour       | <10%/hour       |
| Accuracy       | >95%       | >95%           | >90%            |
