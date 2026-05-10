# TEE-Model — Tacit-Explicit Entegre Eğitim Modeli

Kamu kurumlarında kurumsal bellek kaybını (corporate amnesia) azaltmak için tasarlanmış, yapay zeka destekli onboarding sistemi.

**Hedef kullanım durumu:** Maaş mutemedi onboarding'i.

> **Sürüm 2 (production-rag-v2):** Sistem tamamen yerel inferans yığınına geçti — Ollama (gemma3:4b) ve sentence-transformers (multilingual-e5-large). Üretim Phase 1–5 yol haritası ile TÜBİTAK uyumlu üretim kalitesine taşınmaktadır.

## Mimari

```
data/                       Ham veri kaynakları (mevzuat + tacit transkript)
  ↓ src/anonymizer.py       KVKK uyumlu PII maskeleme
processed/                  Anonimleştirilmiş metinler
  ↓ src/ingestion.py        PDR iki seviyeli chunker + sentence-transformers
chroma_db/                  Yerel vektör veritabanı (1024 boyut, e5-large)
  ↓ src/retrieval.py        Yoğun + (Phase 1.3) hibrit BM25 retrieval
  ↓ src/generators.py       Ollama gemma3:4b ile structured JSON üretimi
app.py                      Streamlit UI (6 sekme)
```

## Kurulum (yerel)

1. **Ollama'yı kur ve modeli indir** ([ollama.com](https://ollama.com)):
   ```bash
   ollama pull gemma3:4b
   ```
2. **Python bağımlılıklarını kur** (sentence-transformers + torch ~3 GB indirir):
   ```bash
   pip install -r requirements.txt
   ```
3. **Yapılandırma:**
   ```bash
   cp .env.example .env
   ```
4. **İlk veri yüklemesi:**
   ```bash
   # Mevcut chroma_db (3072-boyutlu Gemini gömeleri) artık geçersiz; bir kez sil
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

`docker-compose.yml` Ollama servisini ve Streamlit uygulamasını birlikte ayağa kaldırır; gemma3:4b ilk açılışta otomatik indirilir.

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
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama servisi |
| `GENERATION_MODEL` | `gemma3:4b` | Üretim modeli |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Yerel gömme modeli |
| `MOCK_MODE` | `false` | API çağrısı yapmadan fixture döndür |
| `ENABLE_HYBRID_SEARCH` | `true` | BM25 + dense füzyon |
| `ENABLE_CONTEXTUAL_ENRICHMENT` | `true` | Anthropic 2024 contextual retrieval |
| `ENABLE_CONFIDENCE_SCORING` | `true` | Üretilen içeriğe güven skoru |
| `ENABLE_QUERY_REWRITING` | `false` | Çoklu sorgu (HyDE) — yavaş |
| `RAGAS_ENABLED` | `false` | RAGAS değerlendirme harness'i |
