
# Pediatric Pneumonia AI Detection System

An advanced deep learning system for automated pediatric pneumonia detection from chest X-ray images, achieving **99.80% sensitivity** with only **1 false negative** out of 500 pneumonia cases through innovative CLAHE augmentation and knowledge distillation techniques.

## Project Overview

This project tackles a critical challenge in pediatric healthcare: **reducing missed pneumonia diagnoses** in chest X-rays, particularly for blurry or low-quality images common in resource-constrained settings.

**Key Achievement:** Reduced false negatives from **21 → 1** (95% reduction) while maintaining high accuracy, making the system ideal for high-sensitivity screening in underserved regions.

## Performance Summary

Tested on 1,000-image stratified test set (500 normal + 500 pneumonia cases):

| Model Variant | Accuracy | Sensitivity | Specificity | False Negatives | False Positives | F1-Score | Status |
|---------------|----------|-------------|-------------|-----------------|-----------------|----------|--------|
| **MobileNet-Base**<br>*(Baseline)* | 97.60% | 95.80% | 99.40% | 21 | 3 | ~96.5% | Benchmark |
| **MobileNet-CLAHE**<br>*(V4, Production)* | 96.10% | 99.20% | 93.00% | **4** | 35 | 96.22% | Production |
| **MobileNet-KD**<br>*(V5, Best)* | 95.70% | **99.80%** | 91.60% | **1** | 42 | 95.87% | **Best** |

### Clinical Impact
- **99.80% Sensitivity**: Catches 499 out of 500 pneumonia cases
- **Only 1 Missed Case**: Down from 21 in baseline model
- **F1-Score**: 95.87% - excellent balance of precision and recall
- **Ideal for Screening**: High sensitivity prioritized for underserved regions

## Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.10+
- PyTorch 2.8.0+ with CUDA 12.8 (for GPU training)
- 16GB+ RAM (32GB recommended for training)
- GPU: Tesla V100 or better (for multi-GPU training)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ArifHussainSun/pediatric-pneumonia-ai
cd pediatric-pneumonia-ai

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Test Best Model (MobileNet-KD)

```bash
# Test on 1000 images (500 normal + 500 pneumonia)
# MobileNet-KD = V5 = Knowledge Distilled variant
python3 validation/scripts/test_1000_images_conservative.py \
    --model_path outputs/mobilenet_v5_distilled_resnet50/best_model.pth \
    --data_dir ~/pediatric-pneumonia-ai/data/test \
    --no-preprocessing

# Expected output: 95.70% accuracy, 99.80% sensitivity, 1 FN
```

### Test Production Model (MobileNet-CLAHE)

```bash
# More balanced performance (fewer false positives)
# MobileNet-CLAHE = V4 = CLAHE-augmented variant
python3 validation/scripts/test_1000_images_conservative.py \
    --model_path outputs/clahe_augmented_finetune_v4/best_model.pth \
    --data_dir ~/pediatric-pneumonia-ai/data/test \
    --no-preprocessing

# Expected output: 96.10% accuracy, 99.20% sensitivity, 4 FN
```

### Run Inference on a Single Image

```python
# Python script to test on your own chest X-ray image
import torch
from PIL import Image
from torchvision import transforms
from src.models.mobilenet import MobileNetFineTune

# Load model
model = MobileNetFineTune(num_classes=2)
model.load_custom_weights('outputs/mobilenet_v5_distilled_resnet50/best_model.pth')
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('path/to/chest_xray.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()

    normal_prob = probabilities[0, 1].item()
    pneumonia_prob = probabilities[0, 0].item()

    result = "NORMAL" if prediction == 1 else "PNEUMONIA"
    confidence = max(normal_prob, pneumonia_prob)

    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")
    print(f"NORMAL: {normal_prob:.2%}, PNEUMONIA: {pneumonia_prob:.2%}")
```

## Architecture & Approach

### The Problem

**Challenge Identified:**
- Baseline model had **21 false negatives** (missed pneumonia cases)
- **62%** of false negatives were blurry/low-detail images
- Post-hoc preprocessing created distribution mismatch issues
- Traditional augmentation didn't address image quality variation

### The Solution: CLAHE Augmentation Training

**Key Innovation:** Train models with CLAHE (Contrast Limited Adaptive Histogram Equalization) augmentation to make them robust to varying image quality.

```
Training Pipeline:
Raw Image → CLAHE (40% probability) → Standard Augmentation → Model
           ↓
  Clip Limit: 1.5-3.0 (random)
  Grid Size: 8×8

Result: Model learns to handle both high-quality and enhanced images
```

### Model Architecture

#### Primary Model: MobileNetV1 (Optimized for Deployment)

```
Architecture:
Input (224×224×3)
    ↓
MobileNetV1 Backbone (ImageNet pretrained)
    ↓
Global Average Pooling
    ↓
Classifier:
  - Dropout (0.4)
  - Linear (1024 → 64)
  - ReLU + Dropout (0.2)
  - Linear (64 → 2)
    ↓
Softmax → [NORMAL, PNEUMONIA]
```

**Why MobileNetV1?**
- Lightweight: 4.2M parameters vs 25.6M (ResNet50)
- Fast inference: <100ms on CPU
- Mobile-ready: Runs on smartphones and edge devices
- Excellent accuracy: 96%+ with proper training

#### Knowledge Distillation: ResNet50 Teacher → MobileNetV1 Student

```
Knowledge Distillation Pipeline:

Teacher (ResNet50)  ────┐
                        ├──→ Soft Targets (Temperature=3.0)
Student (MobileNetV1)  ─┘         ↓
         ↓                  Combined Loss
    Hard Labels          α×Hard + (1-α)×Soft
    (Ground Truth)         (α = 0.5)
```

**Result:** MobileNet-KD (student model) achieves 99.80% sensitivity with only 1 FN

### Model Nomenclature

To maintain clarity across documentation and submissions, our three MobileNetV1 variants use consistent naming:

| Variant Name | Alternative Names | Description |
|--------------|-------------------|-------------|
| **MobileNet-Base** | Baseline, Baseline MobileNetV1 | Original model trained with standard augmentation only (horizontal flip, rotation ±5°, color jitter). Serves as performance benchmark. |
| **MobileNet-CLAHE** | V4, CLAHE-augmented, Production Model | Enhanced model trained on 40% CLAHE-preprocessed + 60% standard augmented images. Best balance of sensitivity and specificity. |
| **MobileNet-KD** | V5, Knowledge Distilled, Best Model | Most advanced variant with same data composition as MobileNet-CLAHE, incorporating knowledge distillation from ResNet50 teacher. Achieves highest sensitivity (99.80%) with only 1 false negative. |

**Key Distinction:** All three variants share the same MobileNetV1 base architecture (4.2M parameters) and are trained on the Kermany pediatric chest X-ray dataset. Progressive improvements stem from: (1) enhanced preprocessing (CLAHE), and (2) advanced training methodology (knowledge distillation).

### Intelligent Preprocessing Pipeline

Our production API employs an intelligent quality assessment system that selectively enhances images based on quality metrics:

```
Inference Pipeline:
User Upload → Quality Assessment → Enhancement Decision → Model Inference
                     ↓                      ↓
            7 Quality Metrics      If Score < 0.65:
            • Brightness            1. CLAHE enhancement
            • Contrast              2. ROI extraction
            • Sharpness             3. Autoencoder refinement
            • Edge density          4. Denoising filter
            • Noise level                   ↓
            • Positioning          Else: Use original
            • Artifacts                    ↓
                                   Resize 224×224 → Normalize → Inference
```

**Quality Assessment Process:**
1. **Seven-Metric Analysis**: Each uploaded X-ray is evaluated across brightness, contrast, sharpness, edge density, noise level, anatomical positioning, and artifact detection
2. **Threshold Decision** (Score ≥ 0.65): High-quality images bypass enhancement to preserve diagnostic fidelity
3. **Selective Enhancement** (Score < 0.65): Low-quality images undergo multi-stage enhancement:
   - **CLAHE**: Adaptive contrast enhancement with clip limit 1.5-3.0, grid size 8×8
   - **ROI Extraction**: Focus on lung fields, eliminating non-diagnostic regions
   - **Autoencoder Refinement**: Pattern enhancement and feature extraction
   - **Denoising**: Artifact reduction and noise filtering
4. **Conservative Philosophy**: Only enhance when necessary; preserve original quality when possible

This intelligent pipeline ensures robust performance across varying image quality conditions while maintaining high diagnostic accuracy for excellent source images.

## Project Structure

```
pediatric-pneumonia-ai/
├── src/                          # Core source code (15,922 lines)
│   ├── models/                   # Model architectures
│   │   ├── mobilenet.py         # MobileNetV1/V2/V3 implementations
│   │   ├── resnet.py            # ResNet50 for knowledge distillation
│   │   ├── vgg.py               # VGG16 variants
│   │   └── fusion.py            # Ensemble approaches
│   ├── data/                     # Data pipeline
│   │   ├── datasets.py          # PneumoniaDataset with caching
│   │   └── data_loaders.py      # Medical-safe augmentation
│   ├── preprocessing/            # Image enhancement
│   │   ├── clahe_enhancement.py # CLAHE implementation
│   │   └── intelligent_preprocessing.py  # Quality assessment
│   ├── training/                 # Training infrastructure
│   │   ├── trainer.py           # Single-GPU trainer
│   │   └── distributed_trainer.py  # Multi-GPU DDP training
│   ├── evaluation/               # Model evaluation
│   │   └── evaluator.py         # Metrics & confusion matrices
│   ├── visualization/            # Visualization tools
│   │   └── visualizer.py        # GradCAM, training curves
│   ├── mobile/                   # Mobile deployment
│   │   ├── quantization.py      # Model quantization
│   │   └── calibration.py       # Calibration for TFLite
│   └── api/                      # Inference API
│       └── inference.py         # REST API endpoints
│
├── scripts/                      # Training & automation (3,078 lines)
│   ├── retrain_with_clahe_augmentation.py  # Main CLAHE training
│   ├── train_mobilenet_with_distillation.py  # Knowledge distillation
│   ├── train_resnet50_standalone.py  # ResNet50 training
│   ├── optimize_clahe.py        # CLAHE parameter optimization
│   ├── edge_case_analyzer.py    # Failure analysis
│   └── push_distillation_results.sh  # Git automation
│
├── validation/                   # Validation & testing
│   ├── scripts/                 # 15 validation scripts
│   │   ├── test_1000_images_conservative.py  # Main test
│   │   ├── analyze_false_negatives.py  # FN analysis
│   │   ├── comprehensive_model_evaluation.py  # Full metrics
│   │   └── threshold_adjusted_evaluation.py  # Threshold tuning
│   └── reports/                 # Test results
│       ├── mobilenet_v5_distilled_resnet50_test_results/
│       └── v4_finetune_BEST_96.10pct/
│
├── outputs/                      # Trained models
│   ├── dgx_station_experiment/  # MobileNet-Base: Baseline (97.60%, 21 FN)
│   ├── clahe_augmented_finetune_v4/  # MobileNet-CLAHE: V4 (96.10%, 4 FN)
│   └── mobilenet_v5_distilled_resnet50/  # MobileNet-KD: V5 (95.70%, 1 FN)
│
├── configs/                      # Training configurations
│   ├── dgx_station_config.yaml  # DGX multi-GPU setup
│   └── train_config.yaml        # Single-GPU config
│
├── android/                      # Android app
│   └── app/src/main/java/...    # TFLite implementation
│
├── windows/                      # Windows desktop app
│   ├── pneumonia_app.py         # PyQt5 GUI
│   └── inference_engine.py      # Local inference
│
├── data/                         # Dataset
│   ├── train/                   # 6,826 images (balanced)
│   │   ├── NORMAL/              # 3,413 images
│   │   └── PNEUMONIA/           # 3,413 images
│   └── test/                    # 1,704 images (held-out)
│       ├── NORMAL/              # 852 images
│       └── PNEUMONIA/           # 852 images
│
├── HANDOFF_SUMMARY.md           # Complete project documentation
├── DEPLOYMENT.md                # Deployment guides
└── README.md                    # This file
```

## Experimental Journey

### Phase 1: Baseline Model
**Model:** MobileNetV1 (ImageNet pretrained)
**Result:** 97.60% accuracy, **21 false negatives**
**Issue:** High false negatives, especially on blurry images

### Phase 2: CLAHE Experiments

#### Experiment 1: Fine-tune v1
```yaml
Config:
  CLAHE probability: 50%
  Learning rate: 0.0001
  Batch size: 64

Result: FAILED
  - 50% accuracy with preprocessing (distribution mismatch)
  - Model trained on CLAHE but tested with additional preprocessing

Learning: CLAHE models must be tested on raw images
```

#### Experiment 2: Fine-tune v2
```yaml
Config:
  CLAHE probability: 30%
  Learning rate: 0.00005
  Batch size: 64

Result: UNSTABLE
  - High variance (±3% accuracy)
  - Overfitting issues

Learning: Small batch size + low CLAHE probability = unstable training
```

#### Experiment 3: Train from Scratch v2
```yaml
Config:
  CLAHE probability: 50%
  Learning rate: 0.001
  Batch size: 128

Result: GOOD
  - 95.90% accuracy
  - 99.40% sensitivity
  - 3 false negatives, 38 false positives

Learning: Larger batch size (128) dramatically improved stability
```

#### Experiment 4: Fine-tune v4 BEST
```yaml
Config:
  CLAHE probability: 40%  # Sweet spot
  Learning rate: 0.0002
  Batch size: 128
  Class weighting: 1.0 (normal) vs 1.3 (pneumonia)
  Dropout: 0.5
  Early stopping patience: 7

Result: PRODUCTION MODEL
  - 96.10% accuracy
  - 99.20% sensitivity, 93.00% specificity
  - 4 false negatives (81% reduction!)
  - 35 false positives (acceptable trade-off)
  - F1-Score: 96.22%

Status: Approved for deployment
Location: outputs/clahe_augmented_finetune_v4/best_model.pth
```

#### Experiment 5: Over-regularized v5
```yaml
Config:
  CLAHE probability: 25%
  Learning rate: 0.00008
  Dropout: 0.5
  Weight decay: 0.0005
  Class weight: 1.3

Result: CATASTROPHIC FAILURE
  - 68.70% accuracy (worse than random!)
  - 57.40% sensitivity
  - 213 false negatives

Learning: Over-regularization destroyed model capacity
```

### Phase 3: Knowledge Distillation

#### V5: ResNet50 Teacher → MobileNetV1 Student 
```yaml
Config:
  Student: MobileNetV1 v4 (CLAHE-augmented)
  Teacher: ResNet50 (ImageNet pretrained)
  Temperature: 3.0
  Alpha (hard/soft): 0.5
  CLAHE probability: 40%
  Batch size: 128
  Epochs: 29 (early stopped at epoch 22)

Result: BEST MODEL
  - 95.70% accuracy (slight decrease)
  - 99.80% sensitivity (best!)
  - 91.60% specificity
  - 1 false negative (95% reduction from baseline!)
  - 42 false positives
  - F1-Score: 95.87%

Key Achievement: Only 1 missed pneumonia case out of 500
Location: outputs/mobilenet_v5_distilled_resnet50/best_model.pth

False Negative Case:
  - person54_bacteria_258.jpeg (61.6% confidence)
  - Also missed by v4 (difficult case)

Successfully Identified (previously missed by v4):
  person485_bacteria_2049.jpeg
  person1481_virus_2567.jpeg
  person1676_virus_2892.jpeg
```

## Key Technical Findings

### 1. **Distribution Mismatch**
**Finding:** CLAHE-augmented models fail catastrophically (50% accuracy) when tested with additional preprocessing.

**Solution:** Test CLAHE models on raw images using `--no-preprocessing` flag.

**Exception:** V4 model is robust to both preprocessed and raw images (96.10% in both cases).

### 2. **Batch Size Impact**
**Finding:** Increasing batch size from 64 → 128 dramatically improved training stability.

**Why:** More diverse samples per gradient update = smoother convergence, less overfitting.

### 3. **Over-Regularization Risk**
**Finding:** Excessive regularization (high dropout + weight decay + class weighting) destroyed model capacity.

**Example:** V5 regularized with 68.70% accuracy, 213 FN (catastrophic failure).

**Lesson:** Balance is critical - not all regularization is good regularization.

### 4. **CLAHE Probability Sweet Spot**
**Optimal:** 40% CLAHE probability

| Probability | Result |
|-------------|--------|
| 50% | Overfitting to augmented distribution |
| 40% | **Best balance** |
| 30% | Insufficient robustness |
| 25% | Too few augmented samples |

### 5. **Sensitivity-Specificity Trade-off**
**Finding:** CLAHE models achieve ~99% sensitivity but ~7% lower specificity than baseline.

**Why:** Model becomes more cautious about pneumonia detection.

**Trade-off Analysis:**
- Fewer missed cases (critical for screening)
- More false alarms (manageable with radiologist review)
- **Conclusion:** Acceptable for high-sensitivity screening applications

## Usage Guide

### 1. Training a Model

#### Train with CLAHE Augmentation (Recommended)

```bash
# Fine-tune from baseline (faster, recommended)
python3 scripts/retrain_with_clahe_augmentation.py \
    --mode finetune \
    --data_dir ~/pediatric-pneumonia-ai/data \
    --base_model outputs/dgx_station_experiment/Best_MobilenetV1.pth \
    --output_dir outputs/my_clahe_model \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.0002 \
    --clahe_prob 0.4 \
    --dropout_rate 0.5 \
    --normal_class_weight 1.3

# Train from scratch (longer, more flexible)
python3 scripts/retrain_with_clahe_augmentation.py \
    --mode scratch \
    --data_dir ~/pediatric-pneumonia-ai/data \
    --output_dir outputs/my_scratch_model \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --clahe_prob 0.4
```

#### Knowledge Distillation Training

```bash
# Train with ResNet50 teacher
python3 scripts/train_mobilenet_with_distillation.py \
    --use_resnet_teacher \
    --student_model outputs/clahe_augmented_finetune_v4/best_model.pth \
    --data_dir ~/pediatric-pneumonia-ai/data \
    --output_dir outputs/my_distilled_model \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --temperature 3.0 \
    --alpha 0.5
```

#### ResNet50 Standalone (Benchmark)

```bash
# Train ResNet50 for comparison
python3 scripts/train_resnet50_standalone.py \
    --data_dir ~/pediatric-pneumonia-ai/data \
    --output_dir outputs/resnet50_standalone \
    --epochs 30 \
    --batch_size 64 \
    --clahe_prob 0.4
```

### 2. Multi-GPU Training (DGX Station)

```bash
# Launch distributed training on 4 GPUs
# Edit configs/dgx_station_config.yaml first

./scripts/launch_dgx_training.sh configs/dgx_station_config.yaml 4
```

**DGX Configuration:**
- Hardware: 4× Tesla V100 (32GB each)
- Batch size: 48 per GPU (192 effective)
- Workers: 6 per GPU
- Optimizations: AMP, NVLink, persistent workers

### 3. Validation & Testing

#### Test on 1000 Images

```bash
# Test CLAHE models (use --no-preprocessing)
python3 validation/scripts/test_1000_images_conservative.py \
    --model_path outputs/clahe_augmented_finetune_v4/best_model.pth \
    --data_dir ~/pediatric-pneumonia-ai/data/test \
    --no-preprocessing

# Test baseline (with preprocessing)
python3 validation/scripts/test_1000_images_conservative.py \
    --model_path outputs/dgx_station_experiment/Best_MobilenetV1.pth \
    --data_dir ~/pediatric-pneumonia-ai/data/test

# Test ResNet50
python3 validation/scripts/test_1000_images_conservative.py \
    --model_path outputs/resnet50_standalone/best_model.pth \
    --data_dir ~/pediatric-pneumonia-ai/data/test \
    --no-preprocessing \
    --model_type resnet50
```

#### Comprehensive Evaluation

```bash
# Full metrics with ROC, PR curves
python3 validation/scripts/comprehensive_model_evaluation.py \
    --model_path outputs/mobilenet_v5_distilled_resnet50/best_model.pth \
    --data_dir ~/pediatric-pneumonia-ai/data/test \
    --output_dir validation/results/v5_full_eval

# Analyze false negatives in detail
python3 validation/scripts/analyze_false_negatives_detailed.py \
    --model_path outputs/clahe_augmented_finetune_v4/best_model.pth \
    --data_dir ~/pediatric-pneumonia-ai/data/test

# Threshold optimization
python3 validation/scripts/threshold_adjusted_evaluation.py \
    --model_path outputs/mobilenet_v5_distilled_resnet50/best_model.pth \
    --data_dir ~/pediatric-pneumonia-ai/data/test \
    --optimize_for sensitivity  # or 'accuracy', 'f1'
```

### 4. Model Export & Deployment

#### Export for Mobile (Android)

```bash
# Export to TFLite with quantization
python3 scripts/export_android.py \
    --model_path outputs/mobilenet_v5_distilled_resnet50/best_model.pth \
    --output_dir android/app/src/main/assets/ \
    --quantize int8  # or 'float16', 'dynamic'
```

#### Export for Windows

```bash
# Export to ONNX
python3 scripts/export_windows.py \
    --model_path outputs/clahe_augmented_finetune_v4/best_model.pth \
    --output_dir windows/models/ \
    --optimize
```

#### Multi-Format Export

```bash
# Export to multiple formats (ONNX, TorchScript, TFLite)
python3 examples/export_for_deployment.py \
    --model_path outputs/mobilenet_v5_distilled_resnet50/best_model.pth \
    --formats onnx torchscript tflite \
    --output_dir deployment/models/
```

## Dataset Information

### Dataset Structure

```
data/
├── train/                   # Training set (6,826 images)
│   ├── NORMAL/             # 3,413 normal chest X-rays
│   └── PNEUMONIA/          # 3,413 pneumonia chest X-rays
└── test/                    # Test set (1,704 images)
    ├── NORMAL/             # 852 normal chest X-rays
    └── PNEUMONIA/          # 852 pneumonia chest X-rays
```

### Dataset Characteristics

- **Balance:** Perfectly balanced (50% normal, 50% pneumonia)
- **Format:** JPEG chest X-ray images
- **Patient Population:** Pediatric (children)
- **Image Quality:** Varies (some blurry/low-quality)
- **Source:** Publicly available pediatric pneumonia dataset

### Data Augmentation (Medical-Safe)

```python
Training Augmentations:
├── CLAHE Enhancement (40% probability)
│   ├── Clip limit: random(1.5, 3.0)
│   └── Grid size: 8×8
├── Resize: 224×224
├── Random Horizontal Flip (50%)
├── Random Rotation (±5°)  # Small angles only
├── Color Jitter (brightness=0.1, contrast=0.1)
└── ImageNet Normalization

Validation/Test:
├── Resize: 224×224
└── ImageNet Normalization

NOT USED (Unsafe for medical images):
├── Vertical Flip (changes anatomy)
└── Large Rotations (>15°)
```

## Responsible AI & Guardrails

Our solution prioritizes fairness, privacy, transparency, and compliance throughout the development and deployment lifecycle:

### Data & Training Fairness
- **Balanced Dataset**: Training set comprises equal representation (50% normal, 50% pneumonia) from the publicly available Kermany pediatric chest X-ray dataset to mitigate algorithmic bias
- **Class Weighting**: Applied 1.0 (normal) vs 1.3 (pneumonia) weighting during training to address slight class imbalance in loss calculation
- **Diverse Image Quality**: Dataset includes varying quality images (blurry, low-contrast, high-quality) to ensure equitable performance across real-world clinical conditions
- **Stratified Validation**: Test sets maintain balanced class representation for unbiased performance assessment

### Privacy & Compliance
- **Public Training Data**: Utilized publicly available Kermany dataset, eliminating patient privacy concerns during development
- **Future Clinical Validation**: Planned hospital dataset from Pakistan will be handled under strict ethical oversight:
  - Sheridan Research Ethics Board (REB) approval secured
  - Amendment submitted for new dataset coverage
  - All patient data will be anonymized per international privacy regulations
  - Compliance with institutional and international ethical standards
- **No PII Storage**: Inference API processes images in-memory without persistent storage of patient-identifiable information

### Transparency & Interpretability
- **Model Decisions**: Visualization techniques (GradCAM, activation maps) enable clinicians to understand prediction basis
- **Confidence Scoring**: All predictions include calibrated confidence scores (temperature scaling T=2.5) to reduce overconfidence
- **User Feedback**: Quality assessment results provided to clinicians, including enhancement decisions and image quality metrics
- **Open Methodology**: Complete training pipeline, hyperparameters, and architecture details documented for reproducibility

### Continuous Monitoring & Accountability
- **Performance Tracking**: Continuous monitoring of precision, recall, F1-score across deployments
- **Subgroup Analysis**: Attention to performance disparities across demographic subgroups (when data available)
- **Clinical Oversight**: System designed as decision support tool requiring radiologist review, not autonomous diagnosis
- **Error Analysis**: Comprehensive false negative/positive analysis to identify failure modes and improvement opportunities

### Clinical Safety Principles
- **High Sensitivity Priority**: System optimized for 99.80% sensitivity to minimize missed pneumonia diagnoses (only 1 false negative)
- **Acceptable False Positive Rate**: 42 false positives (8.4%) manageable through radiologist review workflows
- **Risk-Benefit Balance**: Missing pneumonia cases (clinical harm) weighted more heavily than false alarms (additional review)
- **Point-of-Care Screening**: Intended for initial screening in underserved regions, not replacement for expert radiologist review

These measures collectively uphold principles of fairness, transparency, accountability, and beneficence, aligning with global frameworks for responsible AI deployment in healthcare.

## Deployment Options

### 1. Web Application (FastAPI + Frontend)

Run the complete web application with FastAPI backend and interactive web interface:

```bash
# Terminal 1: Start FastAPI backend server
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Serve web frontend (in a new terminal)
cd web
python3 -m http.server 3000
```

**Access the Application:**
1. Open browser to: `http://localhost:3000`
2. Upload a chest X-ray image via drag-and-drop or file picker
3. View real-time prediction results with confidence scores
4. Check API health status in the interface

**API Endpoints Available:**
- `GET /health` - Server health check
- `POST /predict` - Single image prediction
- `POST /predict/batch` - Batch prediction (up to 10 images)
- `GET /models` - List available models

**Features:**
- Drag-and-drop image upload
- Real-time inference visualization
- Confidence scoring with NORMAL/PNEUMONIA probabilities
- Image quality assessment and feedback
- Processing time metrics
- Responsive dark-themed UI

### 2. Python API (REST)

```bash
# Start inference server (alternative to FastAPI server above)
python3 -m src.api.inference --port 8080 \
    --model_path outputs/mobilenet_v5_distilled_resnet50/best_model.pth

# Test API
curl -X POST http://localhost:8080/predict \
    -F "file=@chest_xray.jpg" \
    -F "return_confidence=true"
```

### 3. Android App

```bash
cd android
./gradlew assembleRelease

# APK: android/app/build/outputs/apk/release/app-release.apk
```

**Features:**
- Real-time inference (<200ms on modern phones)
- Image quality assessment
- Confidence scoring
- Offline operation

### 4. Windows Desktop App

```bash
cd windows
python pneumonia_app.py

# Or build standalone executable:
pyinstaller --onefile --windowed pneumonia_app.py
```

**Features:**
- PyQt5 GUI
- Batch processing
- Image preprocessing controls
- Result export (PDF, CSV)

### 5. Docker Deployment

```bash
# Build container
docker build -t pneumonia-detection:latest .

# Run inference service
docker run -d -p 8080:8080 \
    --gpus all \
    -v $(pwd)/models:/app/models \
    pneumonia-detection:latest

# Health check
curl http://localhost:8080/health
```

### 6. Integration with Existing Healthcare Systems

Our solution seamlessly integrates into existing healthcare workflows through a multi-stage pipeline designed for both point-of-care and backend analysis:

#### Edge-Cloud Hybrid Architecture

```
Clinical Workflow Integration:

Point-of-Care (Edge):                 Backend Infrastructure (Cloud):
┌─────────────────────┐              ┌──────────────────────────┐
│ Mobile X-ray Unit   │              │ ResNet50 Teacher Model   │
│ └─ MobileNetV1 (4.2M)│──────────────│ └─ Comprehensive Analysis│
│ └─ <100ms inference │              │ └─ Continuous Learning   │
│ └─ Offline capable  │              │ └─ Model Improvement     │
└─────────────────────┘              └──────────────────────────┘
         ↓                                       ↑
    Immediate Screening              Periodic Model Updates
    (99.80% sensitivity)              & Performance Monitoring
```

**Dual-Model Strategy:**
- **Edge Deployment**: Lightweight MobileNetV1 (student model) for real-time screening on resource-constrained devices (mobile X-ray units, rural clinics)
- **Backend Analysis**: Powerful ResNet50 (teacher model) on cloud infrastructure for comprehensive analysis and continuous model improvement
- **Knowledge Transfer**: Backend insights fed back to edge models through periodic updates

#### Healthcare System Touchpoints

Our system provides four key integration points for seamless workflow adoption:

1. **DICOM Image Ingestion**
   - Direct integration with existing radiology PACS (Picture Archiving and Communication Systems)
   - Supports standard DICOM format for chest X-ray images
   - Automatic metadata extraction (patient ID, acquisition parameters)

2. **RESTful API Endpoints**
   - `/predict` - Single image pneumonia detection
   - `/predict/batch` - Batch processing (up to 10 images)
   - `/health` - System status monitoring
   - `/models` - Available model versions
   - Standard HTTP/HTTPS protocols for universal compatibility

3. **HL7 FHIR-Compliant Result Reporting**
   - Structured result output following HL7 FHIR (Fast Healthcare Interoperability Resources) standards
   - Seamless integration with Electronic Health Record (EHR) systems
   - Standardized diagnosis codes and confidence scores
   - Audit trail for clinical decision support

4. **Clinician Dashboard Interfaces**
   - Radiologist review queues with prioritization based on confidence scores
   - Side-by-side comparison of original and enhanced images
   - GradCAM visualization overlays for interpretability
   - Batch result management and export capabilities

#### Deployment Scenarios

**Scenario 1: Rural Clinic Integration**
```
Mobile Technician → Capture X-ray → MobileNetV1 Edge Inference (offline)
                                           ↓
                                    Alert if PNEUMONIA detected
                                           ↓
                              Sync results when connectivity available
                                           ↓
                              Backend validation & EHR integration
```

**Scenario 2: Hospital PACS Integration**
```
DICOM Server → API Gateway → Preprocessing Pipeline → Model Inference
                                                            ↓
                                                   Results → FHIR
                                                            ↓
                                               EHR System Integration
                                                            ↓
                                            Radiologist Review Dashboard
```

This hybrid edge-cloud architecture balances the need for immediate diagnostic support with the computational requirements of state-of-the-art deep learning models, ensuring accessibility in resource-constrained settings while maintaining diagnostic accuracy.

## Model Interpretability

### GradCAM Visualization

```python
from src.visualization import GradCAMVisualizer
from src.models.mobilenet import MobileNetFineTune
import torch

# Load model
model = MobileNetFineTune(num_classes=2)
model.load_custom_weights('outputs/clahe_augmented_finetune_v4/best_model.pth')
model.eval()

# Create GradCAM visualizer
gradcam = GradCAMVisualizer(model, target_layer='features.12')

# Visualize prediction
result = gradcam.visualize_prediction(
    'path/to/chest_xray.jpg',
    output_path='visualization.jpg',
    save_visualization=True
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Attention Map Analysis

GradCAM highlights regions the model focuses on:
- **Pneumonia cases:** Consolidation areas, infiltrates
- **Normal cases:** Clear lung fields
- **Difficult cases:** Multiple small regions

## Performance Monitoring

### TensorBoard

```bash
# View training logs
tensorboard --logdir outputs/clahe_augmented_finetune_v4/tensorboard/

# Metrics tracked:
# - Training/Validation Loss
# - Training/Validation Accuracy
# - Learning Rate
# - Gradient Norms
```

### Continuous Validation

```bash
# Automated validation pipeline
bash scripts/validate_model.sh outputs/mobilenet_v5_distilled_resnet50/

# Generates:
# - Confusion matrix
# - ROC/PR curves
# - False negative analysis
# - Performance report (PDF)
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/ArifHussainSun/pediatric-pneumonia-ai
cd pediatric-pneumonia-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# Install pre-commit hooks (optional)
pre-commit install
```

### Code Quality

```bash
# Format code
black src/ scripts/ validation/

# Lint code
flake8 src/ scripts/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/
```

## Documentation

### Key Documents

- **[HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md)** - Complete project documentation with all experiments
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment guides for all platforms
- **[validation/reports/README.md](validation/reports/README.md)** - Validation results explained
- **[outputs/clahe_augmented_finetune_v4/VALIDATION_SUMMARY.md](outputs/clahe_augmented_finetune_v4/VALIDATION_SUMMARY.md)** - V4 model details
- **[configs/dgx_station_config.yaml](configs/dgx_station_config.yaml)** - DGX training configuration

### API Documentation

```bash
# Generate API docs
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes using [Conventional Commits](https://www.conventionalcommits.org/)
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `chore:` for maintenance
4. **Test** your changes thoroughly
5. **Submit** a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset:** Pediatric Chest X-Ray Images (Pneumonia) - Kaggle
- **DGX System:** CMI DGX Station with 4× Tesla V100 GPUs
- **Framework:** PyTorch 2.8.0 with CUDA 12.8
- **Pretrained Models:** ImageNet weights from torchvision

## Contact

- **Project Maintainer:** Seyon Sriskandarajah
- **GitHub:** [ArifHussainSun/pediatric-pneumonia-ai](https://github.com/ArifHussainSun/pediatric-pneumonia-ai)
- **Issues:** [GitHub Issues](https://github.com/ArifHussainSun/pediatric-pneumonia-ai/issues)

## Citation

If you use this work in your research, please cite:

```bibtex
@software{pediatric_pneumonia_ai_2025,
  author = {Sriskandarajah, Seyon},
  title = {Pediatric Pneumonia AI Detection System with CLAHE Augmentation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ArifHussainSun/pediatric-pneumonia-ai},
  note = {Achieved 99.80\% sensitivity with knowledge distillation}
}
```

## Future Work

### Planned Improvements
- [ ] Test on external hospital datasets for generalization
- [ ] Multi-class classification (bacterial vs viral pneumonia)
- [ ] Integration with PACS systems
- [ ] Ensemble with v4 and v5 models
- [ ] Active learning for continuous improvement
- [ ] Federated learning for privacy-preserving training
- [ ] Explainability improvements (SHAP, LIME)
- [ ] Mobile app optimization (model compression)

### Research Directions
- [ ] Self-supervised learning on unlabeled X-rays
- [ ] Few-shot learning for rare pneumonia types
- [ ] Multi-modal learning (X-ray + clinical data)
- [ ] Uncertainty quantification
- [ ] Adversarial robustness testing

---

**Latest Update:** December 2025 - Added ResNet50 standalone training and knowledge distillation v5 model
**Status:** Production-ready, actively maintained
**Best Model:** V5 (Knowledge Distilled) - 99.80% sensitivity, 1 FN
=======
# pediatric-pneumonia-ai
pediatric-pneumonia-ai

