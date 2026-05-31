# DetectIA

MVP full-stack para recuperar noticias en español, clasificarlas con BETO y guardar un dataset incremental con justificación generada por un LLM.

## Stack

- `FastAPI` para backend y serving del frontend compilado.
- `Vue 3 + Vite` para una única pantalla.
- `BETO` (`BETO/artifacts/model`) como clasificador binario.
- `GNews` para obtener noticias en español.
- `Gemini` para generar la justificación.

## Configuración

1. Copia `app/.env.example` a `app/.env`.
2. Rellena al menos:
   - `GNEWS_API_KEY`
   - `GEMINI_API_KEY`

Opcionales útiles:

- `EXTRACT_ARTICLE_BODY=true` para intentar extraer el texto completo desde la URL original del medio.
- `ARTICLE_MIN_EXTRACTED_CHARS=120` para exigir una longitud mínima antes de aceptar la extracción.

## Desarrollo local

Backend:

```bash
cd app/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd app/frontend
npm install
npm run dev
```

El frontend usa proxy hacia `http://localhost:8000` durante desarrollo.

## Docker para desarrollo

```bash
cd app
docker compose up -d
```

En desarrollo:

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`

Notas importantes:

- Si solo cambias `app/.env`, no uses `--build`. Basta con `docker compose up -d`.
- Si cambias código del backend, `uvicorn --reload` recarga automáticamente.
- Si cambias código del frontend, Vite recompila en caliente automáticamente.

## Docker para producción

```bash
cd app
docker compose --profile prod up --build -d detectia-prod
```

La versión de producción quedará disponible en `http://localhost:8080`.

## Dataset generado

Las clasificaciones se guardan en `app/backend/data/classified_news.csv` con estas columnas:

- `titulo`
- `cuerpo`
- `prediccion`
- `justificacion`
