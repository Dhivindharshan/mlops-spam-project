# MLOps Spam Classification Project

This project is a simple end-to-end spam detection system built for learning and showcasing MLOps fundamentals. It trains a spam classifier on the SMS Spam Collection dataset, saves the trained model as an artifact, exposes prediction through a FastAPI service, and supports MLflow experiment tracking for both training and inference.

## Project Description

The goal of the project is to classify SMS-style text messages into:

- `spam`
- `ham`

The project covers the core lifecycle of a small machine learning system:

- dataset preparation
- model training
- artifact generation
- API serving
- experiment tracking with MLflow

## Architecture

The project is organized into these main components:

- `data/`
  Contains the dataset used for training. The current `spam.csv` is based on the Kaggle SMS Spam Collection dataset.
- `src/train.py`
  Trains the Naive Bayes spam classifier, evaluates it, saves the model, and logs runs to MLflow.
- `models/`
  Stores the trained model artifact and evaluation metrics.
- `app/main.py`
  Runs the FastAPI application and exposes prediction endpoints.
- `requirements.txt`
  Lists the Python dependencies required for the API and MLflow integration.

## Dataset

The project now uses the larger Kaggle SMS Spam Collection dataset instead of a tiny sample file.

- Dataset: SMS Spam Collection
- Source: Kaggle
- Size: 5,572 rows
- Format used in this project: `label,text`

This improves the quality of training and makes the project stronger for demos, interviews, and placements.

## How To Run

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Train the model

```powershell
python src/train.py
```

Optional MLflow tracking:

```powershell
$env:MLFLOW_EXPERIMENT_NAME="spam-classifier"
python src/train.py
```

Optional MLflow tracking server:

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME="spam-classifier"
python src/train.py
```

### 3. Run the API

```powershell
uvicorn app.main:app --reload
```

The API will start on:

```text
http://127.0.0.1:8000
```

### 4. Check API health

Open:

```text
http://127.0.0.1:8000/health
```

## Sample API Request

### PowerShell

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"Win a free cash prize now"}'
```

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Win a free cash prize now\"}"
```

### Example Response

```json
{
  "label": "spam",
  "score": 0.9987,
  "model_source": "trained_model",
  "keyword_matches": null
}
```

## API Endpoints

- `GET /health`
  Returns service health, MLflow status, model loading status, and model path.
- `POST /predict`
  Accepts message text and returns the predicted label and confidence score.

## MLflow Tracking

MLflow is integrated into:

- training in `src/train.py`
- inference logging in `app/main.py`

Training logs:

- model type
- dataset path
- sample counts
- test size
- random state
- accuracy
- precision
- recall
- f1 score
- model artifact
- metrics artifact

## Current Output Files

After training, the project generates:

- `models/spam_model.pkl`
- `models/metrics.json`

## Tech Stack

- Python
- FastAPI
- MLflow
- Pydantic
- Uvicorn

## Project Structure

```text
mlops-spam-project/
|-- app/
|   `-- main.py
|-- data/
|   `-- spam.csv
|-- models/
|   |-- metrics.json
|   `-- spam_model.pkl
|-- src/
|   `-- train.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```
