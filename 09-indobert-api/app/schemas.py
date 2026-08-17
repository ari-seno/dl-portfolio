from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Teks yang akan diklasifikasi")


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100, description="Daftar teks")

    @field_validator("texts")
    @classmethod
    def texts_not_blank(cls, texts):
        for text in texts:
            if not text.strip():
                raise ValueError("setiap teks harus non-kosong")
        return texts


class PredictResponse(BaseModel):
    text: str
    label: str
    score: float
    probabilities: dict[str, float]


class PredictBatchResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    model_name: str
    model_type: str
    labels: list[str]
    max_length: int
