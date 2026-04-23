# TEE-Model POC — Tacit-Explicit Entegre Eğitim Modeli

Kamu kurumlarında kurumsal bellek kaybını (corporate amnesia) azaltmak için tasarlanmış, yapay zeka destekli onboarding sistemi.

**Hedef kullanım durumu:** Maaş mutemedi onboardingı

## Kurulum

```bash
pip install -r requirements.txt
cp .env.example .env   # API anahtarını .env dosyasına girin
```

## Çalıştırma

```bash
# Uygulamayı başlat
streamlit run app.py

# Mock modda çalıştır (API çağrısı yapmadan UI test)
MOCK_MODE=true streamlit run app.py

# Sadece veri yükle
python3 src/ingestion.py

# Anonymizer'ı test et
python3 src/anonymizer.py
```

## Mimari

```
Veri Kaynakları
  ├── data/explicit_mevzuat.txt    → Resmi mevzuat
  └── data/tacit_interview_raw.txt → Deneyimli personel mülakatı
        ↓
  src/anonymizer.py   → KVKK uyumlu PII maskeleme (regex)
        ↓
  processed/tacit_interview_clean.txt
        ↓
  src/ingestion.py    → Chunking + Embedding (gemini-embedding-001) + ChromaDB
        ↓
  chroma_db/          → Yerel vektör veritabanı
        ↓
  src/retrieval.py    → Anlamsal arama + context oluşturma
        ↓
  src/generators.py   → RAG içerik üretimi (gemini-2.5-flash)
        ↓
  app.py              → Streamlit UI (4 sekme)
```

## Sekmeler

1. **Veri Yükleme** — Belge yükleme durumu ve anonimleştirme logu
2. **Süreç Haritası ve Hata Kartları** — AI tarafından üretilen eğitim materyali
3. **İnteraktif Simülasyon** — Senaryo bazlı pratik sorular
4. **Uzman Onay Paneli** — Tüm içeriklerin uzman kontrolü

## Ortam Değişkenleri

| Değişken | Açıklama |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio API anahtarı |
| `MOCK_MODE` | `true` → API çağrısı yapmadan fixture veriler kullanılır |
