from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = ROOT_DIR / "app"
BACKEND_DIR = APP_DIR / "backend"


class Settings(BaseSettings):
    app_name: str = "DetectIA"
    app_env: str = "development"
    api_prefix: str = "/api"

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    )

    beto_model_path: Path = ROOT_DIR / "BETO" / "artifacts" / "model"
    beto_max_length: int = 512
    dataset_csv_path: Path = BACKEND_DIR / "data" / "classified_news.csv"

    gnews_api_key: str | None = None
    gnews_base_url: str = "https://gnews.io/api/v4"
    gnews_language: str = "es"
    gnews_country: str = "es"
    gnews_max_articles: int = 10
    extract_article_body: bool = True
    article_fetch_user_agent: str = (
        "DetectIA/0.1 (+https://localhost; article extraction)"
    )
    article_min_extracted_chars: int = 120

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_base_url: str = (
        "https://generativelanguage.googleapis.com/v1beta/models"
    )

    request_timeout_seconds: float = 30.0

    frontend_dist_dir: Path = APP_DIR / "frontend" / "dist"

    model_config = SettingsConfigDict(
        env_file=APP_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.dataset_csv_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
