# TEE-Model uygulama imajı — Vertex AI Gemini cloud yığını ile çalışır.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/hf

WORKDIR /app

# chromadb ve bazı Python paketleri için sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8080

# Phase 1 — entrypoint korpusu hazırlar (GCS senkronizasyonu + doğrulama), sonra Streamlit'i başlatır.
ENTRYPOINT ["./entrypoint.sh"]
