import csv
import json
import re
import threading
from pathlib import Path

import httpx
import torch
import trafilatura
from fastapi import HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import Settings
from .preprocessing import build_full_text, minimal_beto_clean
from .schemas import NewsArticle


CSV_HEADERS = ["titulo", "cuerpo", "prediccion", "justificacion"]
MODEL_OPTIONS = [
    {
        "id": "beto",
        "name": "BETO",
        "description": "Clasificador binario fine-tuned sobre noticias en español.",
    }
]


class NewsService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def fetch_articles(self, query: str | None = None) -> list[NewsArticle]:
        if not self.settings.gnews_api_key:
            raise HTTPException(
                status_code=503,
                detail="Falta configurar GNEWS_API_KEY en app/.env.",
            )

        endpoint = "search" if query else "top-headlines"
        params = {
            "lang": self.settings.gnews_language,
            "country": self.settings.gnews_country,
            "max": self.settings.gnews_max_articles,
            "apikey": self.settings.gnews_api_key,
        }
        if query:
            params["q"] = query
        else:
            params["category"] = "general"

        headers = {"User-Agent": self.settings.article_fetch_user_agent}
        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds,
            headers=headers,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                f"{self.settings.gnews_base_url}/{endpoint}",
                params=params,
            )

            if response.status_code >= 400:
                detail = self._build_error_detail(response)
                raise HTTPException(status_code=response.status_code, detail=detail)

            payload = response.json()
            articles = [
                NewsArticle(
                    title=item.get("title") or "Sin titulo",
                    body=item.get("content") or item.get("description") or "",
                    url=item.get("url"),
                    image=item.get("image"),
                    published_at=item.get("publishedAt"),
                    source={
                        "name": (item.get("source") or {}).get("name", ""),
                        "url": (item.get("source") or {}).get("url"),
                    },
                )
                for item in payload.get("articles", [])
            ]

            if not self.settings.extract_article_body:
                return [self._normalize_article_body(article) for article in articles]

            hydrated_articles = []
            for article in articles:
                hydrated_articles.append(await self._hydrate_article_body(client, article))
            return hydrated_articles

    async def _hydrate_article_body(
        self,
        client: httpx.AsyncClient,
        article: NewsArticle,
    ) -> NewsArticle:
        fallback_body = minimal_beto_clean(article.body)
        if not article.url:
            article.body = fallback_body
            article.body_source = "gnews"
            return article

        try:
            response = await client.get(article.url)
            response.raise_for_status()
        except httpx.HTTPError:
            article.body = fallback_body
            article.body_source = "gnews"
            return article

        extracted_body = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            url=article.url,
        )
        extracted_body = minimal_beto_clean(extracted_body or "")

        if self._is_useful_extraction(extracted_body, fallback_body):
            article.body = extracted_body
            article.body_source = "source_url"
        else:
            article.body = fallback_body
            article.body_source = "gnews"

        return article

    def _normalize_article_body(self, article: NewsArticle) -> NewsArticle:
        article.body = minimal_beto_clean(article.body)
        article.body_source = "gnews"
        return article

    def _is_useful_extraction(self, extracted_body: str, fallback_body: str) -> bool:
        if len(extracted_body) < self.settings.article_min_extracted_chars:
            return False

        if TRUNCATED_GNEWS_PATTERN.search(extracted_body):
            return False

        if len(extracted_body) <= len(fallback_body):
            return False

        return True

    @staticmethod
    def _build_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {}

        message = payload.get("errors") or payload.get("message") or response.text
        if isinstance(message, list):
            message = "; ".join(str(item) for item in message)
        return f"GNews devolvio {response.status_code}: {message}"


class BetoClassifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._tokenizer = None
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_lock = threading.Lock()

    def predict(self, article: NewsArticle) -> dict:
        processed_text = build_full_text(article.title, article.body)
        if not processed_text:
            raise HTTPException(status_code=422, detail="La noticia no contiene texto util.")

        tokenizer, model = self._ensure_loaded()
        encoded = tokenizer(
            processed_text,
            truncation=True,
            max_length=self.settings.beto_max_length,
            padding=False,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()

        real_probability = float(probabilities[0].item())
        fake_probability = float(probabilities[1].item())
        prediction = "fake" if fake_probability >= real_probability else "no fake"

        return {
            "prediction": prediction,
            "prediction_label": "Falsa" if prediction == "fake" else "Real",
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "processed_text": processed_text,
        }

    def _ensure_loaded(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        with self._load_lock:
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model

            model_path = Path(self.settings.beto_model_path)
            if not model_path.exists():
                raise RuntimeError(
                    f"No se encontro el modelo BETO en {model_path}."
                )

            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._model.to(self._device)
            self._model.eval()

        return self._tokenizer, self._model


class JustificationService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(self, article: NewsArticle, prediction_label: str) -> str:
        if not self.settings.gemini_api_key:
            return "Justificacion no generada: falta GEMINI_API_KEY en app/.env."

        prompt = (
            "Eres un asistente que justifica una clasificacion de deteccion de noticias falsas. "
            "Redacta una justificacion breve en espanol, con 2 o 3 frases, prudente y sin afirmar "
            "que el modelo tiene certeza absoluta. Explica que senales textuales pueden haber influido "
            f"en la prediccion '{prediction_label}'. Noticia:\n\n"
            f"Titulo: {article.title}\n"
            f"Cuerpo: {article.body}\n"
        )

        url = (
            f"{self.settings.gemini_base_url}/{self.settings.gemini_model}:generateContent"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 180,
            },
        }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                url,
                params={"key": self.settings.gemini_api_key},
                json=payload,
            )

        if response.status_code >= 400:
            return f"Justificacion no disponible: error Gemini {response.status_code}."

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "Justificacion no disponible: Gemini no devolvio contenido."

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )
        text = " ".join(part.get("text", "").strip() for part in parts if part.get("text"))
        return text or "Justificacion no disponible: respuesta vacia."


class DatasetWriter:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self._lock = threading.Lock()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, article: NewsArticle, prediction: str, justification: str) -> None:
        row = {
            "titulo": article.title,
            "cuerpo": article.body,
            "prediccion": prediction,
            "justificacion": justification,
        }

        with self._lock:
            file_exists = self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)


TRUNCATED_GNEWS_PATTERN = re.compile(r"\[\d+\s+chars\]", re.IGNORECASE)


