# Validation Reports

This directory contains the test results for the best performing model.

## Best Model: Fine-tuning v4 ✓

**Directory:** `conservative_1000img_test_20251118_090501/`

### Performance
- **Overall Accuracy:** 96.10%
- **Sensitivity:** 99.20%
- **Specificity:** 93.00%
- **False Negatives:** 4 (81% reduction from baseline's 21)
- **False Positives:** 35

### Configuration
- **Model:** CLAHE-augmented MobileNetV1
- **Training:** Fine-tuned from baseline with 40% CLAHE augmentation
- **Batch Size:** 128
- **Learning Rate:** 0.0002
- **Early Stopping:** Epoch 18 (best at epoch 11)

### Comparison to Baseline

| Metric | Baseline | V4 | Change |
|--------|----------|-----|--------|
| Accuracy | 97.60% | 96.10% | -1.5% |
| Sensitivity | 95.80% | 99.20% | **+3.4%** ✓ |
| Specificity | 99.40% | 93.00% | -6.4% |
| False Negatives | 21 | **4** | **-81%** ✓ |
| False Positives | 3 | 35 | +32 |

### Key Achievement
**Reduced missed pneumonia cases from 21 to 4 (81% reduction)** while maintaining high overall accuracy.

### Test Details
- **Test Set:** 1000 images (500 Normal + 500 Pneumonia)
- **Seed:** 2025 (for reproducibility)
- **Testing Method:** Raw images without preprocessing

### Files in Report
- `results.json` - Overall metrics and confusion matrix
- `false_negatives.json` - Details of 4 missed pneumonia cases
- `full_results.csv` - Complete predictions for all 1000 images

### Deployment
**Model Location:** `outputs/clahe_augmented_finetune_v4/best_model.pth`

**Recommended for:** High-sensitivity pneumonia screening where minimizing false negatives is critical.

---

For complete experiment history and methodology, see project documentation.
