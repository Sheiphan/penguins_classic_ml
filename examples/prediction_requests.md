# Example Prediction Requests

This document provides example API requests for the Enterprise ML Classifier using real penguin data patterns.

## Single Prediction Examples

### Adelie Penguin (Torgersen Island)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "island": "Torgersen",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "MALE",
    "year": 2007
  }'
```

**Expected Response:**
```json
{
  "prediction": "Adelie",
  "confidence": 0.95,
  "model_version": "randomforestclassifier_v1",
  "prediction_time": "2024-01-15T10:30:00Z"
}
```

### Chinstrap Penguin (Dream Island)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "island": "Dream",
    "bill_length_mm": 46.5,
    "bill_depth_mm": 17.9,
    "flipper_length_mm": 192.0,
    "body_mass_g": 3500.0,
    "sex": "FEMALE",
    "year": 2008
  }'
```

**Expected Response:**
```json
{
  "prediction": "Chinstrap",
  "confidence": 0.88,
  "model_version": "randomforestclassifier_v1",
  "prediction_time": "2024-01-15T10:31:00Z"
}
```

### Gentoo Penguin (Biscoe Island)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "island": "Biscoe",
    "bill_length_mm": 48.6,
    "bill_depth_mm": 16.0,
    "flipper_length_mm": 230.0,
    "body_mass_g": 5800.0,
    "sex": "MALE",
    "year": 2009
  }'
```

**Expected Response:**
```json
{
  "prediction": "Gentoo",
  "confidence": 0.92,
  "model_version": "randomforestclassifier_v1",
  "prediction_time": "2024-01-15T10:32:00Z"
}
```

## Batch Prediction Examples

### Mixed Species Batch
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "MALE",
        "year": 2007
      },
      {
        "island": "Dream",
        "bill_length_mm": 46.5,
        "bill_depth_mm": 17.9,
        "flipper_length_mm": 192.0,
        "body_mass_g": 3500.0,
        "sex": "FEMALE",
        "year": 2008
      },
      {
        "island": "Biscoe",
        "bill_length_mm": 48.6,
        "bill_depth_mm": 16.0,
        "flipper_length_mm": 230.0,
        "body_mass_g": 5800.0,
        "sex": "MALE",
        "year": 2009
      }
    ]
  }'
```

**Expected Response:**
```json
{
  "predictions": [
    {
      "prediction": "Adelie",
      "confidence": 0.95,
      "input_index": 0
    },
    {
      "prediction": "Chinstrap", 
      "confidence": 0.88,
      "input_index": 1
    },
    {
      "prediction": "Gentoo",
      "confidence": 0.92,
      "input_index": 2
    }
  ],
  "model_version": "randomforestclassifier_v1",
  "prediction_time": "2024-01-15T10:33:00Z",
  "batch_size": 3
}
```

## Edge Cases and Validation Examples

### Missing Optional Fields
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "island": "Biscoe",
    "bill_length_mm": 45.2,
    "bill_depth_mm": 15.8,
    "flipper_length_mm": 215.0,
    "body_mass_g": 4800.0,
    "sex": "FEMALE",
    "year": 2008
  }'
```

### Invalid Island Value (Error Example)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "island": "InvalidIsland",
    "bill_length_mm": 45.2,
    "bill_depth_mm": 15.8,
    "flipper_length_mm": 215.0,
    "body_mass_g": 4800.0,
    "sex": "FEMALE",
    "year": 2008
  }'
```

**Expected Error Response:**
```json
{
  "detail": [
    {
      "type": "enum",
      "loc": ["body", "island"],
      "msg": "Input should be 'Biscoe', 'Dream' or 'Torgersen'",
      "input": "InvalidIsland"
    }
  ]
}
```

### Out of Range Values (Error Example)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "island": "Biscoe",
    "bill_length_mm": -5.0,
    "bill_depth_mm": 15.8,
    "flipper_length_mm": 215.0,
    "body_mass_g": 4800.0,
    "sex": "FEMALE",
    "year": 2008
  }'
```

**Expected Error Response:**
```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "bill_length_mm"],
      "msg": "Input should be greater than 0",
      "input": -5.0
    }
  ]
}
```

## Health Check and Model Info

### Health Check
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:35:00Z",
  "model_loaded": true,
  "model_version": "randomforestclassifier_v1",
  "uptime_seconds": 3600
}
```

### Model Information
```bash
curl http://localhost:8000/model/info
```

**Expected Response:**
```json
{
  "model_name": "RandomForestClassifier",
  "model_version": "randomforestclassifier_v1",
  "training_date": "2024-01-15T09:00:00Z",
  "features": [
    "island",
    "bill_length_mm",
    "bill_depth_mm", 
    "flipper_length_mm",
    "body_mass_g",
    "sex",
    "year"
  ],
  "target_classes": ["Adelie", "Chinstrap", "Gentoo"],
  "metrics": {
    "accuracy": 0.97,
    "f1_score": 0.97,
    "precision": 0.97,
    "recall": 0.97
  }
}
```

## Python Client Examples

### Using requests library
```python
import requests
import json

# Single prediction
url = "http://localhost:8000/predict"
data = {
    "island": "Biscoe",
    "bill_length_mm": 48.6,
    "bill_depth_mm": 16.0,
    "flipper_length_mm": 230.0,
    "body_mass_g": 5800.0,
    "sex": "MALE",
    "year": 2009
}

response = requests.post(url, json=data)
result = response.json()
print(f"Predicted species: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Using httpx (async)
```python
import httpx
import asyncio

async def predict_species():
    async with httpx.AsyncClient() as client:
        data = {
            "island": "Dream",
            "bill_length_mm": 46.5,
            "bill_depth_mm": 17.9,
            "flipper_length_mm": 192.0,
            "body_mass_g": 3500.0,
            "sex": "FEMALE",
            "year": 2008
        }
        
        response = await client.post(
            "http://localhost:8000/predict",
            json=data
        )
        result = response.json()
        return result

# Run async function
result = asyncio.run(predict_species())
print(result)
```

## Data Ranges and Valid Values

### Feature Constraints
- **island**: Must be one of ["Biscoe", "Dream", "Torgersen"]
- **bill_length_mm**: Positive float, typically 30-60mm
- **bill_depth_mm**: Positive float, typically 13-22mm  
- **flipper_length_mm**: Positive float, typically 170-235mm
- **body_mass_g**: Positive float, typically 2700-6300g
- **sex**: Must be one of ["MALE", "FEMALE"]
- **year**: Integer, typically 2007-2009

### Typical Value Ranges by Species

**Adelie Penguins:**
- Bill length: 32-46mm
- Bill depth: 15-21mm
- Flipper length: 172-210mm
- Body mass: 2850-4775g
- Islands: Torgersen, Biscoe, Dream

**Chinstrap Penguins:**
- Bill length: 40-58mm
- Bill depth: 16-20mm
- Flipper length: 178-212mm
- Body mass: 2700-4800g
- Islands: Dream only

**Gentoo Penguins:**
- Bill length: 40-60mm
- Bill depth: 13-17mm
- Flipper length: 203-231mm
- Body mass: 3950-6300g
- Islands: Biscoe only