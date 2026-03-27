# RAG Assistant (Production-Ready)

A portfolio-ready **Retrieval-Augmented Generation (RAG)** app built with **Python + Streamlit + Hugging Face embeddings**, with an optional **FastAPI** backend. Upload a document, ask questions, and see the sources used to answer.

## Architecture

```mermaid
flowchart LR
  U[User] --> Q[Query]
  Q --> QR["Query Rewriting (LLM)"]
  QR --> R[Hybrid Retrieval]
  R --> V[Vector Search (Chroma)]
  R --> K[Keyword Search (BM25)]
  V --> M[Merge + Dedupe]
  K --> M
  M --> RR[Cross-Encoder Re-ranking]
  RR --> CF[Context Filtering / Compression (LLM)]
  CF --> A[Answer (LLM)]
  A --> UI[Streamlit UI / FastAPI Response]
  CF --> S[Sources]
  S --> UI
```

## Features

- **Document ingestion**: PDF / Website / GitHub README
- **Hybrid search**: semantic (vector) + keyword (BM25)
- **Re-ranking**: cross-encoder improves relevance
- **Context compression**: drops irrelevant text before answering
- **Source transparency**: shows which chunks were used
- **Multilingual-friendly embeddings**: works well for English + Hindi/Hinglish
- **Optional API**: `GET /health`, `POST /upload`, `POST /query`

## Tech stack

- **UI**: Streamlit (`app.py`)
- **API**: FastAPI (`app/main.py`)
- **RAG framework**: LangChain
- **Vector DB**: Chroma (local persistent)
- **Embeddings**: `intfloat/multilingual-e5-large` (configurable)
- **Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Security (best practices)

- **Never commit secrets**: `.env` is in `.gitignore` ✅
- **Use env vars**: store API keys/tokens in environment variables (or `.env` locally)
- **Commit only `.env.example`**: safe template without real secrets
- **Rotate keys** if you accidentally pushed them

## Clean project structure (recommended)

```text
rag/
├─ app/                    # core RAG modules (API + services)
├─ docs/screenshots/       # README images (placeholders)
├─ samples/                # safe sample data for demos
├─ data/                   # local vector store (gitignored)
├─ app.py                  # Streamlit UI
├─ Dockerfile              # optional container build
├─ docker-compose.yml      # optional local run
├─ requirements.txt
├─ .env.example
└─ README.md
```

## Setup (stable, step-by-step)

### 1) Create venv

```bash
python -m venv venv
```

### 2) Activate venv (Windows)

```powershell
venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment

```powershell
copy .env.example .env
```

Then edit `.env`:
- **Ollama**: `LLM_PROVIDER=ollama`, `LLM_MODEL=mistral`
- **OpenAI**: `LLM_PROVIDER=openai`, set `OPENAI_API_KEY=...`

### 5) Run Streamlit (recommended)

```bash
streamlit run app.py
```

### 6) Run FastAPI (optional)

```bash
uvicorn app.main:app --reload --port 8000
```

Open Swagger UI at `http://localhost:8000/docs`.

## Usage guide (what to click)

1. Choose **Source type** (PDF / website / GitHub)
2. Ingest your source
3. Ask a question
4. Review:
   - answer
   - rewritten query
   - retrieved sources

## Demo screenshots (placeholders)

Add images under `docs/screenshots/` and reference them here:

- `docs/screenshots/chat-ui.png`
- `docs/screenshots/sources.png`
- `docs/screenshots/api-docs.png`

## 1–2 minute demo script (what to say)

- **0:00–0:10**: “This is my production-ready RAG assistant. It grounds answers in your documents and shows sources.”
- **0:10–0:30**: Upload a PDF (or ingest a URL). “It chunks the doc, embeds it, and stores it in Chroma.”
- **0:30–0:55**: Ask a question. “The pipeline rewrites the query, performs hybrid retrieval (vector + BM25), re-ranks, compresses context, then answers.”
- **0:55–1:15**: Show **retrieved sources** and explain transparency. “You can verify exactly where the answer came from.”
- **1:15–1:30**: Mention extensibility: “FastAPI endpoints exist for integration and deployment.”

## Sample data (what to include)

- Include a **small, safe, redistributable** sample so reviewers can run the project immediately.
- This repo includes `samples/sample.txt` to demonstrate question-answering without private data.
- Recommended sample PDF ideas:
  - your own **1–2 page project FAQ** (architecture + features)
  - a short **product manual excerpt** you wrote
  - a **policy doc** you authored (like the sample text), exported to PDF

Why it matters: it makes your GitHub repo **self-contained**, improves first-run UX, and helps reviewers validate results quickly.

## Advanced (high-impact) improvements

- **Docker** (already included): `docker compose up --build`
- **Streaming responses** (token-wise):
  - Streamlit: use `st.write_stream(...)` and LLM streaming callbacks
  - FastAPI: use `StreamingResponse` for SSE/websocket style streaming
- **Deployment ideas**:
  - Streamlit Community Cloud (simple UI demos)
  - Render / Railway (FastAPI + persistent volume for Chroma)
  - Hugging Face Spaces (Streamlit app + small demo data)
- **UI polish**:
  - chat history + message bubbles
  - “copy answer” button
  - collapsible “sources” panel with chunk previews

## Future improvements

- Better GitHub ingestion (full repo traversal + file-type filters)
- Auth + rate limiting for the API
- Background ingestion queue for large docs
- Automated eval dashboards (RAGAS regression tests)

