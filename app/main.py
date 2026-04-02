from __future__ import annotations

import os
import pickle
from pathlib import Path
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.model import NaiveBayesSpamModel

try:
    import mlflow
except ImportError:
    mlflow = None


SPAM_KEYWORDS = {
    "free",
    "win",
    "winner",
    "claim",
    "prize",
    "cash",
    "offer",
    "urgent",
    "click",
    "buy",
    "limited",
}

MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "false").lower() == "true"
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "spam-classifier-inference",
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/spam_model.pkl"))


@dataclass
class PredictionResult:
    label: str
    score: float
    model_source: str
    keyword_matches: int | None = None


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Message text to classify.")


class PredictionResponse(BaseModel):
    label: str
    score: float
    model_source: str
    keyword_matches: int | None = None


def configure_mlflow() -> bool:
    if not MLFLOW_ENABLED:
        return False

    if mlflow is None:
        return False

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return True


MLFLOW_READY = configure_mlflow()
MODEL_LOAD_ERROR: str | None = None

app = FastAPI(
    title="Spam Classifier API",
    description="API for spam message prediction with optional MLflow tracking.",
    version="1.0.0",
)


def load_trained_model() -> object | None:
    global MODEL_LOAD_ERROR

    if not MODEL_PATH.exists():
        return None

    try:
        with MODEL_PATH.open("rb") as model_file:
            return pickle.load(model_file)
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        return None


TRAINED_MODEL = load_trained_model()


def predict_with_keyword_rules(text: str) -> PredictionResult:
    normalized_text = text.lower()
    matches = sum(keyword in normalized_text for keyword in SPAM_KEYWORDS)
    score = matches / max(len(SPAM_KEYWORDS), 1)
    label = "spam" if matches >= 2 else "ham"
    return PredictionResult(
        label=label,
        score=round(score, 2),
        model_source="keyword_rules",
        keyword_matches=matches,
    )


def predict_with_trained_model(text: str) -> PredictionResult:
    if TRAINED_MODEL is None:
        return predict_with_keyword_rules(text)

    label, confidence = TRAINED_MODEL.predict_with_confidence(text)
    return PredictionResult(
        label=label,
        score=confidence,
        model_source="trained_model",
    )


def predict_spam(text: str) -> PredictionResult:
    return predict_with_trained_model(text)


def log_prediction_to_mlflow(text: str, result: PredictionResult) -> None:
    if not MLFLOW_READY or mlflow is None:
        return

    with mlflow.start_run(run_name="api-prediction"):
        mlflow.log_params(
            {
                "prediction_label": result.label,
                "text_length": len(text),
                "model_source": result.model_source,
            }
        )
        if result.keyword_matches is not None:
            mlflow.log_param("keyword_matches", result.keyword_matches)
        mlflow.log_metrics({"spam_score": result.score})
        mlflow.set_tags(
            {
                "app": "spam-classification-api",
                "model_type": result.model_source,
            }
        )


@app.get("/health")
def health_check() -> dict[str, str | bool | None]:
    return {
        "status": "ok",
        "mlflow_enabled": MLFLOW_READY,
        "model_loaded": TRAINED_MODEL is not None,
        "model_path": str(MODEL_PATH),
        "model_load_error": MODEL_LOAD_ERROR,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    result = predict_spam(text)
    log_prediction_to_mlflow(text, result)
    return PredictionResponse(
        label=result.label,
        score=result.score,
        model_source=result.model_source,
        keyword_matches=result.keyword_matches,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
