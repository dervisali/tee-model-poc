# TEE-Model — Tacit-Explicit Entegre Eğitim Modeli

Kamu kurumlarında kurumsal bellek kaybını (corporate amnesia) azaltmak için tasarlanmış, yapay zeka destekli onboarding sistemi.

**Hedef kullanım durumu:** Maaş mutemedi onboarding'i.

> **Cloud sürüm (cloud-vertex-rag-v2):** Bu branch Vertex AI Gemini kullanır — `gemini-2.5-flash` üretim modeli ve `gemini-embedding-001` cloud embedding modeli. Yerel Ollama/e5 sürümü `feature/production-rag-v2` branch'inde korunur.

## Mimari

```
data/                       Ham veri kaynakları (mevzuat + tacit transkript)
  ↓ src/anonymizer.py       KVKK uyumlu PII maskeleme
processed/                  Anonimleştirilmiş metinler
  ↓ src/ingestion.py        PDR iki seviyeli chunker + Vertex AI embeddings
chroma_db/                  Yerel vektör veritabanı (3072 boyut, Gemini embedding)
  ↓ src/retrieval.py        Yoğun + (Phase 1.3) hibrit BM25 retrieval
  ↓ src/generators.py       Vertex AI Gemini ile structured JSON üretimi
app.py                      Streamlit UI (6 sekme)
```

## Kurulum (yerel)

1. **Google Cloud kimliğini hazırla**:
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable aiplatform.googleapis.com
   ```
2. **Python bağımlılıklarını kur**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Yapılandırma:**
   ```bash
   cp .env.example .env
   ```
4. **İlk veri yüklemesi:**
   ```bash
   # Embedding modeli değiştiğinde chroma_db yeniden oluşturulmalıdır
   rm -rf chroma_db/
   python -m src.ingestion
   ```
5. **Uygulamayı başlat:**
   ```bash
   streamlit run app.py
   ```

## Docker ile çalıştırma

```bash
docker compose up
```

`docker-compose.yml` yalnızca Streamlit uygulamasını ayağa kaldırır; model çağrıları Vertex AI'a gider. Cloud Run'da servis hesabına Vertex AI User yetkisi verilmelidir.

## Mock modu

API/inference olmadan UI testi için:

```bash
MOCK_MODE=true streamlit run app.py
```

## Sekmeler

1. **Veri Yükleme** — Belge yükleme durumu ve anonimleştirme logu
2. **Süreç Haritası ve Hata Kartları** — AI tarafından üretilen eğitim materyali
3. **İnteraktif Simülasyon** — Senaryo bazlı pratik sorular
4. **Uzman Onay Paneli** — Tüm içeriklerin uzman kontrolü
5. **Prompt Optimizer** — autoresearch ilhamlı prompt arama
6. **Veritabanı Gezgini** — ChromaDB içerik gezgini

## Ortam Değişkenleri

Tam liste için `src/config.py`. En kritikler:

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | - | Vertex AI projesi |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | Vertex AI bölgesi |
| `GENERATION_MODEL` | `gemini-2.5-flash` | Üretim modeli |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Cloud gömme modeli |
| `EMBEDDING_DIMENSION` | `3072` | Gömme boyutu |
| `MOCK_MODE` | `false` | API çağrısı yapmadan fixture döndür |
| `ENABLE_HYBRID_SEARCH` | `true` | BM25 + dense füzyon |
| `ENABLE_CONTEXTUAL_ENRICHMENT` | `true` | Anthropic 2024 contextual retrieval |
| `ENABLE_CONFIDENCE_SCORING` | `true` | Üretilen içeriğe güven skoru |
| `ENABLE_QUERY_REWRITING` | `false` | Çoklu sorgu (HyDE) — yavaş |
| `RAGAS_ENABLED` | `false` | RAGAS değerlendirme harness'i |
