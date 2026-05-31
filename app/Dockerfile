FROM python:3.11-slim AS backend-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY app/backend/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

FROM backend-base AS backend-dev

CMD ["uvicorn", "app.main:app", "--app-dir", "/srv/app/backend", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM node:22-bookworm-slim AS frontend-builder
WORKDIR /srv/frontend
COPY app/frontend/package*.json ./
RUN npm install --no-audit --no-fund
COPY app/frontend/ ./
RUN npm run build

FROM backend-base AS backend-runtime

COPY BETO/artifacts/model /srv/BETO/artifacts/model
COPY app/backend /srv/app/backend
COPY --from=frontend-builder /srv/frontend/dist /srv/app/frontend/dist

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "/srv/app/backend", "--host", "0.0.0.0", "--port", "8000"]
