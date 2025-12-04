# REST API Documentation

## Overview

REST API for pneumonia detection in pediatric chest X-rays. Provides endpoints for single and batch image processing with clinical-grade validation.

## Endpoints

### Health Check
```
GET /health
```
Returns API health status and system metrics.

### Single Prediction
```
POST /predict
```
Predict pneumonia from single chest X-ray image.

**Parameters:**
- `file`: Image file (required)
- `model_type`: Model to use (default: mobilenet)
- `patient_id`: Patient identifier (optional)
- `confidence_threshold`: Threshold for prediction (default: 0.5)

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.874,
  "probabilities": {
    "NORMAL": 0.126,
    "PNEUMONIA": 0.874
  },
  "processing_time_ms": 145.2
}
```

### Batch Prediction
```
POST /predict/batch
```
Process multiple images in single request.

**Parameters:**
- `files`: List of image files (required, max 20)
- `model_type`: Model to use (default: mobilenet)
- `patient_ids`: Comma-separated patient IDs (optional)

## Configuration

API behavior controlled via `configs/api_config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

models:
  mobilenet:
    file: "mobilenet_pneumonia.pth"
    device: "cpu"
```

## Deployment

### Local Development
```bash
python -m src.api.server
```

### Production
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

## Monitoring

- Health endpoint: `/health`
- Metrics endpoint: `/metrics`
- Request logging enabled by default