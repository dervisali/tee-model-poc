# TEE-Model uygulama imajı — Vertex AI Gemini cloud yığını ile çalışır.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf-cache \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

# chromadb ve bazı Python paketleri için sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN groupadd --system app && useradd --system --gid app --home-dir /home/app --create-home app \
    && mkdir -p /tmp/chroma_db /tmp/hf-cache /tmp/.cache /app/logs \
    && chown -R app:app /app /home/app /tmp/chroma_db /tmp/hf-cache /tmp/.cache \
    && chmod +x entrypoint.sh

EXPOSE 8080

USER app

# Phase 1 — entrypoint korpusu hazırlar (GCS senkronizasyonu + doğrulama), sonra Streamlit'i başlatır.
ENTRYPOINT ["./entrypoint.sh"]
