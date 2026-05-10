# TEE-Model — Mimari ve Teknik Tasarım Belgesi

> **Sürüm:** cloud-vertex-rag-v2 · **Tarih:** 2026-05-10  
> **Hedef:** TÜBİTAK 1001/1003 araştırma desteği için akademik nitelikte
> teknik rapor temeli.

> **Cloud branch notu:** Bu belge ilk olarak `feature/production-rag-v2`
> yerel Ollama/e5 yığını için yazılmıştır. `feature/cloud-vertex-rag-v2`
> branch'i aynı RAG tasarımını korur, ancak inferans ve embedding katmanını
> Vertex AI `gemini-2.5-flash` + `gemini-embedding-001` ile değiştirir.
> Güncel production ayarları için `README.md`, `src/config.py`,
> `src/llm.py` ve `src/embeddings.py` esas alınmalıdır.

Bu belge, Tacit-Explicit Entegre Eğitim (TEE) modelinin RAG mimarisini,
her büyük tasarım kararının gerekçesini ve değerlendirme sonuçlarını
özetler. Tüm bileşenler `feature/production-rag-v2` dalında
uygulanmıştır; commit hash referansları her bölümün sonundadır.

---

## 1. Sistem Mimarisi

```
                     ┌────────────────────────────────────┐
                     │        BELGE KAYNAKLARI            │
                     │  data/explicit_mevzuat.txt         │
                     │  data/tacit_interview_raw.txt      │
                     │  data/new_directive.txt            │
                     └─────────────────┬──────────────────┘
                                       │
                       ┌───────────────▼────────────────┐
                       │  src/anonymizer.py             │
                       │  KVKK regex maskeleme          │
                       │  ([İSİM] [TC-KİMLİK]           │
                       │   [IBAN] [TELEFON])            │
                       └───────────────┬────────────────┘
                                       │
                       ┌───────────────▼────────────────┐
                       │  src/chunkers.py               │
                       │  Strateji: paragraph (varsayılan)
                       │             | semantic         │
                       │             | fixed            │
                       │  Parent: 300-800 ch            │
                       │  Child : 50-200 ch (PDR)       │
                       └───────────────┬────────────────┘
                                       │
                       ┌───────────────▼────────────────┐
                       │  src/contextual_enrichment.py  │
                       │  Anthropic 2024 Contextual     │
                       │  Retrieval — her chunk LLM     │
                       │  ile bağlam özetiyle zenginleştir│
                       │  Cache: SHA1(doc, chunk)       │
                       └───────────────┬────────────────┘
                                       │
                ┌──────────────────────▼──────────────────┐
                │  src/embeddings.py                      │
                │  multilingual-e5-large (1024 dim)       │
                │  passage:/query: prefix konvansiyonu    │
                └──────┬─────────────────────────────┬────┘
                       │                             │
       ┌───────────────▼──────────┐  ┌───────────────▼──────────┐
       │  ChromaDB                 │  │  src/hybrid_search.py    │
       │  tee_children (HNSW,      │  │  BM25Okapi + RRF (k=60)  │
       │  cosine)                  │  │  pickle indeksi          │
       │  parents.json (PDR)       │  └───────────────┬──────────┘
       └─────────┬─────────────────┘                  │
                 │                                    │
                 └─────────────┬──────────────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  src/retrieval.py        │
                  │  search_mode ∈ {dense,   │
                  │   sparse, hybrid}        │
                  │  parent_id dedup         │
                  │  thin-context fallback   │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  src/query_rewriter.py   │
                  │  (opsiyonel) çok-sorgu   │
                  │  HyDE varyantları        │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  src/llm.py              │
                  │  Ollama gemma3:4b        │
                  │  Pydantic format=        │
                  │  tenacity retry          │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  src/generators.py       │
                  │  4 üretici (process_map  │
                  │   error_cards · glossary │
                  │   simulation)            │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  src/confidence.py       │
                  │  ikinci LLM geçişi:      │
                  │  desteklenen/desteklen-  │
                  │   meyen iddia sayımı +   │
                  │   güven skoru            │
                  └────────────┬─────────────┘
                               │
       ┌───────────────────────┼─────────────────────┐
       │                       │                     │
       ▼                       ▼                     ▼
 ┌──────────────┐    ┌──────────────────┐   ┌──────────────────┐
 │ src/job_queue│    │  src/evaluator.py│   │  app.py (UI)     │
 │ tek-işçili   │    │  RAGAS 4 metrik  │   │  6 sekme,        │
 │ kuyruk       │    │  Ollama yargıç   │   │  güven rozetleri │
 └──────────────┘    └──────────────────┘   └──────────────────┘

 Çevresel: src/config.py (pydantic-settings), src/logging_config.py
 (yapılandırılmış JSONL).
```

İnferans yığını **tamamen yereldir**: hiçbir bileşen dış API'ye dokunmaz.
gemma3:4b modeli Ollama üzerinden, multilingual-e5-large modeli ise
sentence-transformers üzerinden CPU/MPS/CUDA otomatik tespit ile çalışır.

---

## 2. RAG Boru Hattı — Akademik Atıflar

### 2.1 Parent Document Retrieval (PDR)

**Yöntem.** Belgeler iki seviyeye ayrılır: 300-800 karakterlik *parent*
parçaları LLM'e gönderilen tam bağlamdır; 50-200 karakterlik *child*
parçaları gömme (embedding) için optimize edilmiştir. Sorgu, child seviyesinde
arar; parent_id geri-referansı ile deduplicate edilip parent metni LLM'e
gönderilir.

**Atıf.** Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question
Answering*, EMNLP 2020, hiyerarşik retrieval'ı temellendirir. LangChain
RAG dokümantasyonu PDR pattern'ini günümüz mühendislik pratiği olarak
sunar.

**Bu projede.** `src/ingestion.py:_split_into_parent_chunks` ve
`_split_into_child_chunks`. Commit `65fd005`.

### 2.2 Contextual Chunk Enrichment

**Yöntem.** Her child chunk gömülmeden önce LLM'e *(tam_belge, chunk)*
ikilisi verilip 2-3 cümlelik bağlam özeti üretilir. Özet chunk'ın başına
eklenir; gömme ve BM25 yolu zenginleştirilmiş metin üzerinden işler.

**Atıf.** Anthropic, *Introducing Contextual Retrieval*, Eylül 2024.
Top-20 retrieval başarısızlığı (yalnız gömme): %5.7 → contextual embeddings
+ contextual BM25 → %2.9 (-%49 göreli iyileşme).

**Bu projede.** `src/contextual_enrichment.py`, SHA1-keyli kalıcı önbellek
ile (re-ingestion artımlıdır). Commit `43295d0`.

### 2.3 Hybrid Search (Dense + BM25 + RRF)

**Yöntem.** Yoğun gömme (e5-large) ve BM25Okapi paralel çalıştırılır;
her motorun top-N'i Reciprocal Rank Fusion ile birleştirilir.

**Atıflar.**
- Cormack, Clarke, Büettcher, *Reciprocal Rank Fusion outperforms Condorcet
  and individual Rank Learning Methods*, SIGIR 2009. RRF formülü:
  *score(d) = Σ 1 / (k + rank_i(d))*; *k = 60* endüstri standardıdır.
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond*, FNT-IR 2009. BM25 referansı.

**Türkçe gerekçe.** Mevzuat metinlerinde "ek gösterge katsayısı", "net
aylığın 1/4'ü", "Form-ADB-01" gibi birebir terim ifadeleri yoğun gömmeden
sıkça kaçar. BM25 bu eşleşmeleri yakalar; RRF skorları normalize etmeden
birleştirir.

**Bu projede.** `src/hybrid_search.py`,
`scripts/compare_retrieval.py` ile dense vs sparse vs hybrid kıyas tablosu.
Commit `8ee9492`.

### 2.4 Multi-Query Retrieval (HyDE-tarzı)

**Yöntem.** Orijinal sorgu için 3 alternatif Türkçe ifadeleme üretilir;
her biri için retrieval çalıştırılır, sonuçlar parent_id bazında
deduplicate edilip skorlara göre yeniden sıralanır.

**Atıflar.**
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*
  (HyDE), ACL 2023.
- Multi-Query Retrieval, LangChain dokümantasyonu (production pattern).

**Bu projede.** `src/query_rewriter.py`. ENABLE_QUERY_REWRITING bayrağı
arkasında, varsayılan **kapalı** (kalite-kritik üretimler için açılır).
Commit `ca48d20`.

### 2.5 RAGAS Evaluation Framework

**Yöntem.** Dört metrik:
- **faithfulness** — cevap, retrieved context'e bağlı mı?
- **answer_relevancy** — cevap soruyu karşılıyor mu?
- **context_precision** — alınan parçalar alakalı mı?
- **context_recall** — alınan parça ideal cevabı içeriyor mu?

**Atıf.** Es, James, Espinosa-Anke, Schockaert, *RAGAs: Automated
Evaluation of Retrieval Augmented Generation*, EACL 2024.

**Bu projede.** `src/evaluator.py`. Ollama gemma3:4b'yi
`langchain-ollama` üzerinden RAGAS'a yargıç olarak bağlar; aynı
e5-large'ı RAGAS embeddings olarak kullanır. Test seti
`evaluation/test_questions.json` — 20 elle yazılmış soru-cevap çifti
(12 mevzuat, 6 tacit, 2 genelge). Commit `f7e38e7`.

### 2.6 İkinci-Geçiş Güven Skorlaması

**Yöntem.** Üretilen JSON içeriğin her iddiası, retrieved parent
metinlere karşı bir ek LLM çağrısıyla karşılaştırılır. Çıktı:
*guven_skoru ∈ [0, 1]*, desteklenen/desteklenmeyen iddia sayıları,
desteklenmeyen iddia listesi, uzman onay tavsiyesi.

**Atıflar.** Self-RAG (Asai et al., 2023) ve faithfulness as judgment
literatürü.

**Bu projede.** `src/confidence.py`. Yeşil/sarı/kırmızı rozet renkleri
uzman onay panelinde gösterilir (`app.py`'de `_render_confidence_banner`).
Commit `3ae2784`.

### 2.7 Anonimleştirme (KVKK)

**Yöntem.** Sıralı regex maskelemesi (IBAN → telefon → TC → isim).
Sıra bilinçli: 11 haneli telefon TC kalıbına uyabildiği için telefon
TC'den önce; IBAN'ın TR + 24 hanesi başka kalıpları gölgelememeli.

**Bu projede.** `src/anonymizer.py`. Mock veriden korunan örnekler
`tests/test_anonymizer.py`'da uygulanır.

---

## 3. Tasarım Kararları Günlüğü

### 3.1 İnferans yığını: Ollama + e5-large vs Google Gemini

| Madde | İçerik |
|---|---|
| Alternatifler | (a) Mevcut Gemini API; (b) Ollama gemma3:4b + e5-large; (c) Hibrit |
| Seçim | (b) Tamamen yerel yığın |
| Gerekçe | TÜBİTAK projesinde dış API bağımlılığı kabul edilemez (veri egemenliği, sürdürülebilir maliyet). Yerel yığın çevrimdışı çalışır, KVKK kontrolünü tamamlar. |
| Maliyet | Ücretsiz; ilk model indirme ~3 GB; gemma3:4b CPU'da ~10-20 token/s. |
| Risk | gemma3:4b uzun bağlam üzerinde Gemini Flash'tan zayıftır; bu "tek geçişli yapılandırılmış JSON + ikinci geçiş güven skorlaması" iki-aşamalı tasarımla telafi edilir. |
| Atıf | Phase 0.5 commit `38f6be1`. |

### 3.2 Varsayılan chunklama: paragraph vs semantic

| Madde | İçerik |
|---|---|
| Alternatifler | paragraph (\n\n + cümle), semantic (cümle gömme benzerliği), fixed (sabit pencere) |
| Seçim | **paragraph** varsayılan; semantic ve fixed birlikte sunulur. |
| Gerekçe | Şubat-2026 7-strateji benchmark'ında recursive 512-token splitting %69 doğruluk, similarity-based semantic chunking %54 (akademik metinde). MADDE-tabanlı Türk mevzuat yapısı zaten paragraph sınırı sağlam çiziyor. Semantic, daha az yapılandırılmış belgeler için açık opsiyon olarak kalır. |
| Atıf | ICNLSP 2025 *Recursive Semantic for RAG Optimization*; Phase 1.2.B commit `2d9ebe3`. |

### 3.3 Hibrit füzyon: RRF vs ağırlıklı (alpha) toplam

| Madde | İçerik |
|---|---|
| Alternatifler | (a) alpha-weighted normalize edilmiş skor toplamı; (b) RRF |
| Seçim | **RRF** varsayılan (alpha=None), kullanıcı isterse weighted moda geçebilir. |
| Gerekçe | RRF skor normalize gerektirmez, BM25'in mutlak puanı dense'in mutlak mesafesinden yapısal olarak farklı; toplama hataya açık. RRF rank-bazlı olduğu için skor dağılımına dayanıklı. |
| Atıf | Cormack 2009; Phase 1.3 commit `8ee9492`. |

### 3.4 Multi-query: varsayılan açık vs kapalı

| Madde | İçerik |
|---|---|
| Alternatifler | (a) Her sorguda 3 varyant + 4 retrieval; (b) Bayrak arkasında, varsayılan kapalı |
| Seçim | (b) Bayrak arkasında varsayılan **kapalı**. |
| Gerekçe | gemma3:4b ile her query rewrite ~5-15s ekler; hot loop'ta (Prompt Optimizer'da) gereksiz ve maliyetli. Kalite-kritik içerik üretiminde (final ders materyali, RAGAS değerlendirmesi) açılması önerilir. |
| Atıf | Phase 1.4 commit `ca48d20`. |

### 3.5 Bağlam zenginleştirme: önbellek tasarımı

| Madde | İçerik |
|---|---|
| Alternatifler | (a) Hiç önbellek; (b) İçerik-hash anahtarlı JSON; (c) ChromaDB metadata içine yaz |
| Seçim | (b) `chroma_db/enrichment_cache.json` SHA1(document_id + chunk_text) anahtarlı. |
| Gerekçe | Re-ingestion sıkça yapılır (özellikle yeni belge eklemede); LLM çağrısı pahalı. ChromaDB metadata'ya yazmak daha sıkı entegrasyon ama Chroma sürümleri arası taşınabilirliği zorlaştırır. JSON dosyası taşınabilir, debugging-dostu. |
| Atıf | Phase 1.2.A commit `43295d0`. |

### 3.6 Üretim isteklerini sıralama: tek-işçi vs paralel

| Madde | İçerik |
|---|---|
| Alternatifler | (a) Streamlit'in doğal seri davranışına güven; (b) Tek-işçili thread pool; (c) FastAPI ayrı servis |
| Seçim | (b) `concurrent.futures.ThreadPoolExecutor(max_workers=1)`. |
| Gerekçe | Çok-kullanıcılı senaryoda Streamlit oturum izolasyonu paralel çağrılara izin verir. Cloud branch'te tek-işçili kuyruk, Vertex AI kota/maliyet kontrolü ve kullanıcıya izlenebilir uzun iş durumu vermek için korunur. Production ölçeği arttığında kalıcı kuyruk (Cloud Tasks / Pub/Sub) veya ayrı API worker mimarisi tercih edilmelidir. |
| Atıf | Phase 3.1 commit `917820d`. |

---

## 4. Değerlendirme Sonuçları

> **Not.** Aşağıdaki tablo `python -m src.evaluator` ile gerçek
> Ollama + multilingual-e5-large ortamında doldurulmalıdır. Bu projedeki
> son değerlendirme yerel makinede çalıştırıldığında
> `evaluation/results/ragas_results_<timestamp>.json` dosyasında
> saklanır; özet bu tabloya kopyalanır.

### 4.1 RAGAS skorları — temel (baseline) ve aşamalı iyileştirme

| Yapılandırma | faithfulness | answer_relevancy | context_precision | context_recall |
|---|:---:|:---:|:---:|:---:|
| dense-only, paragraph chunking, **enrichment KAPALI** (baseline) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + hybrid search (BM25 + RRF) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + contextual enrichment | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + multi-query rewriting | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Her satır için çalıştırma:
```bash
# baseline
ENABLE_HYBRID_SEARCH=false ENABLE_CONTEXTUAL_ENRICHMENT=false ENABLE_QUERY_REWRITING=false \
    python -m src.evaluator
```

### 4.2 Kategori bazında kapsam — test_questions.json

- **explicit (12 soru):** Mevzuat metnine doğrudan dayanan, faithfulness için
  güçlü sinyal sağlayan sorular.
- **tacit (6 soru):** Mülakat transkriptindeki örtük bilgi (yeni mutemet
  hataları, ay ortası işe giriş, geriye dönük zam). Yapay zeka ile
  açıklanması zor olduğu için context_recall'un kritik testidir.
- **directive (2 soru):** Genelge belgesinin retrieval'a girip girmediğini
  doğrular — yeni belge ingestion'ın etkinliğini ölçer.

### 4.3 Hibrit retrieval kalitatif karşılaştırması

`scripts/compare_retrieval.py` aşağıdaki 6 birebir Türkçe terim sorgusu
için dense vs sparse vs hybrid top-3 sonuçlarını yazdırır:

- *ek gösterge katsayısı*
- *net aylığın 1/4'ü icra kesintisi*
- *kümülatif matrah Ocak ayı sıfırlama*
- *Form-ADB-01 aile durumu*
- *göreve başlama belgesi*
- *yıllık katsayı güncelleme*

Bu çıktılar TÜBİTAK raporunun "RAG Mimari Kararları" bölümünde örnek
olarak sunulabilir.

---

## 5. Sürdürülebilirlik ve Yeniden Üretilebilirlik

### 5.1 Lokal kurulum
```bash
ollama pull gemma3:4b
pip install -r requirements.txt
cp .env.example .env
rm -rf chroma_db/                # önceki Gemini gömeleri varsa temizle
python -m src.ingestion
streamlit run app.py
```

### 5.2 Docker ile
```bash
docker compose up
```

### 5.3 Test
```bash
MOCK_MODE=true python -m pytest tests/ -v
```
Bu komut Ollama veya sentence-transformers yüklü olmasa bile geçer
(uygun testler markörlerle atlanır).

### 5.4 Ölçüm yenilemesi
```bash
python -m src.evaluator             # tüm 20 soru, RAGAS dört metriği
python -m src.evaluator --quick     # ilk 5 soru hızlı doğrulama
python -m scripts.compare_retrieval # dense/sparse/hybrid kıyas çıktısı
```

---

## 6. Kaynaklar

- Anthropic. *Introducing Contextual Retrieval*. 2024-09. https://www.anthropic.com/news/contextual-retrieval
- Cormack, G. V., Clarke, C. L. A., Büettcher, S. *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*. SIGIR 2009.
- Es, S., James, J., Espinosa-Anke, L., Schockaert, S. *RAGAs: Automated Evaluation of Retrieval Augmented Generation*. EACL 2024.
- Gao, L., Ma, X., Lin, J., Callan, J. *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*. ACL 2023.
- Karpukhin, V. et al. *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020.
- Robertson, S., Zaragoza, H. *The Probabilistic Relevance Framework: BM25 and Beyond*. FNT-IR 2009.
- Singh, A. et al. *Agentic Retrieval-Augmented Generation: A Survey*. arXiv:2501.09136 (2025).
- *Recursive Semantic Chunking for RAG Optimization*. ICNLSP 2025.
- Jina AI. *Late Chunking: Contextual Chunk Embeddings*. arXiv:2409.04701 (2024).
