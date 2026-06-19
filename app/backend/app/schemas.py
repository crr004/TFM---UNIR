from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    name: str = ""
    url: str | None = None


class NewsArticle(BaseModel):
    title: str
    body: str
    body_source: str = "gnews"
    url: str | None = None
    image: str | None = None
    published_at: str | None = None
    source: SourceInfo = Field(default_factory=SourceInfo)


class ModelOption(BaseModel):
    id: str
    name: str
    description: str


class NewsResponse(BaseModel):
    articles: list[NewsArticle]


class ClassificationRequest(BaseModel):
    article: NewsArticle
    model_id: str


class EvaluateRequest(BaseModel):
    article: NewsArticle
    prediction_label: str
    justification: str


class ClassificationResponse(BaseModel):
    prediction: str
    prediction_label: str
    fake_probability: float
    real_probability: float
    justification: str
    processed_text: str


class EvaluationResponse(BaseModel):
    relevancia: int    # 1-5: la justificación se basa en el artículo
    fidelidad: int     # 1-5: las afirmaciones son fieles al texto
    coherencia: int    # 1-5: bien escrita y argumentada
    alineacion: int    # 1-5: encaja con la predicción emitida
    media: float
    razonamiento: str  # cadena de pensamiento del juez (CoT)
