# TEE-Model POC (Maaş Mutemedi Eğitim Sistemi) - Architecture & Context

## 🎯 Project Purpose
The **TEE-Model POC** is a Retrieval-Augmented Generation (RAG) system designed to onboard and train public institution payroll officers (Maaş Mutemedi). It tackles the critical issue of knowledge loss by bridging two types of information:
1. **Explicit Knowledge:** Formal regulations, laws, and official procedures (Mevzuat).
2. **Tacit Knowledge:** Unwritten rules, practical tips, and common pitfalls gathered from interviews with veteran payroll officers.

By combining these, the system acts as an expert assistant to prevent costly payroll errors and speed up the training of new personnel.

---

## 🏗️ Core Architecture & Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/) (`app.py`), providing an interactive, multi-tab web interface.
- **LLM Engine:** Google Cloud Gemini API (`gemini-3-flash-preview`), utilized via the modern `google-genai` SDK.
- **Vector Storage:** Local [ChromaDB](https://www.trychroma.com/) (`chroma_db/`).
- **Deployment:** Containerized via Docker and hosted serverless on **Google Cloud Run**.

---

## 🧠 Data Pipeline: Parent Document Retrieval (PDR)
The system uses an advanced RAG design pattern called **Parent Document Retrieval** to maximize accuracy while minimizing hallucinations.
- **Ingestion (`src/ingestion.py`):** Large documents are split into massive "Parent" chunks (stored in a local `parents.json` dictionary). These parents are then subdivided into smaller "Child" chunks, which are embedded and stored in ChromaDB.
- **Retrieval (`src/retrieval.py`):** When a user asks a question, the system performs a semantic search against the small, highly specific Child chunks. However, instead of sending the small child to the LLM, it looks up and sends the entire Parent chunk. This gives Gemini the full, broad context it needs to generate a comprehensive answer.

---

## ⚙️ Generation Layer (`src/generators.py`)
This is the core logic that communicates with Gemini. We applied Google Cloud Agent Platform best practices here:
1. **Native Structured Outputs:** Uses Pydantic (`BaseModel`) to forcefully constrain the LLM to return 100% valid JSON matching exact schemas (e.g., `ProcessMapSchema`, `ErrorCardsSchema`).
2. **System Grounding:** Uses the `system_instruction` parameter to establish a strict persona and forbid hallucinations outside of the provided context.
3. **Deterministic RAG:** Forces a `temperature=0.1` to ensure highly factual and repeatable answers.

It generates four distinct components:
- **Süreç Haritası (Process Map):** Step-by-step guides for tasks like calculating severance.
- **Hata Kartları (Error Cards):** Common mistakes, root causes, and correct practices.
- **Terim Sözlüğü (Glossary):** Definitions of domain-specific jargon.
- **Simülasyon (Simulation):** Interactive "What would you do?" scenarios for trainees.

---

## 🧪 Auto-Evaluation (`src/prompt_optimizer.py`)
An autonomous evaluation loop that runs multiple prompt variants, scores them using a "Gemini LLM Judge" (based on grounding, completeness, and usefulness), and persists the winning prompts to `optimized_prompts.json` to be used in production.

---

## 🚀 Known Limitations & Future Roadmap
1. **Stateless Vector DB Issue:** Currently, `chroma_db` is baked into the Docker image. For true production readiness on Cloud Run, the vector database must be migrated to a managed solution like **Cloud SQL (pgvector)**.
2. **Security:** The current Cloud Run deployment is public. It requires **Google Identity-Aware Proxy (IAP)** to enforce organization-level authentication. 
3. **Context Length:** If the document base grows significantly, implementing Gemini Context Caching will be required to maintain low latency.
