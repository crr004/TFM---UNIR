from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .schemas import ClassificationRequest, ClassificationResponse, ModelOption, NewsResponse
from .services import BetoClassifier, DatasetWriter, JustificationService, MODEL_OPTIONS, NewsService


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    news_service = NewsService(settings)
    classifier = BetoClassifier(settings)
    justification_service = JustificationService(settings)
    dataset_writer = DatasetWriter(settings.dataset_csv_path)

    @app.get(f"{settings.api_prefix}/health")
    async def health():
        return {"status": "ok"}

    @app.get(f"{settings.api_prefix}/models", response_model=list[ModelOption])
    async def list_models():
        return MODEL_OPTIONS

    @app.get(f"{settings.api_prefix}/news", response_model=NewsResponse)
    async def get_news(query: str | None = Query(default=None, min_length=2)):
        articles = await news_service.fetch_articles(query=query)
        return NewsResponse(articles=articles)

    @app.post(f"{settings.api_prefix}/classify", response_model=ClassificationResponse)
    async def classify(payload: ClassificationRequest):
        if payload.model_id != "beto":
            return ClassificationResponse(
                prediction="unsupported",
                prediction_label="No soportado",
                fake_probability=0.0,
                real_probability=0.0,
                justification="El modelo seleccionado no esta disponible.",
                processed_text="",
            )

        result = classifier.predict(payload.article)
        justification = await justification_service.generate(
            payload.article,
            result["prediction_label"],
        )
        dataset_writer.append(payload.article, result["prediction"], justification)

        return ClassificationResponse(
            justification=justification,
            **result,
        )

    dist_dir = settings.frontend_dist_dir
    if dist_dir.exists():
        assets_dir = dist_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            requested = dist_dir / full_path
            if full_path and requested.exists() and requested.is_file():
                return FileResponse(requested)
            return FileResponse(dist_dir / "index.html")

    return app


app = create_app()
