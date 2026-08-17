from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.inference import SentimentModel
from app.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)


def create_app(model: SentimentModel | None = None) -> FastAPI:
    model = model or SentimentModel()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not model.loaded:
            model.load()
        yield

    app = FastAPI(title="IndoBERT Sentiment API", version="1.0.0", lifespan=lifespan)
    app.model = model

    @app.get("/health", response_model=HealthResponse)
    def health():
        if not model.loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "ok", "device": str(model.device), "model_loaded": True}

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata():
        return {
            "model_name": "indobert-large-p1",
            "model_type": "BertForSequenceClassification",
            "labels": ["negative", "neutral", "positive"],
            "max_length": 128,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        return model.predict(req.text)

    @app.post("/predict_batch", response_model=PredictBatchResponse)
    def predict_batch(req: PredictBatchRequest):
        return PredictBatchResponse(results=model.predict_batch(req.texts))

    return app


app = create_app()
