# Customer_Bot

Production-oriented RAG assistant for Hajster controller documentation.

The service ingests a technical PDF manual, indexes chunks in Pinecone, and generates Ukrainian answers using Gemini with source grounding.

## Highlights
- Retrieval-Augmented Generation (RAG) pipeline for technical Q&A
- Semantic search via Pinecone + multilingual embeddings (`intfloat/multilingual-e5-large`)
- Answer generation with Google Gemini (`gemini-2.5-flash`)
- Session-aware chat memory support
- Flask API + web UI + optional Streamlit UI
- Docker + Google Cloud Run deployment flow

## Architecture
1. PDF ingestion and parsing (`docling` with `PyMuPDF` fallback)
2. Chunking with metadata enrichment
3. Embedding generation
4. Vector indexing/search in Pinecone
5. Prompt assembly + LLM response generation
6. Response payload with source references

Core modules:
- `src/pdf_loader.py` — PDF text/image extraction
- `src/text_splitter.py` — chunking and chunk metadata
- `src/vector_store.py` — embeddings + Pinecone operations
- `src/rag_chain.py` — retrieval + prompt + generation
- `api.py` — REST API (`/`, `/status`, `/chat`)
- `reindex.py` — rebuilds vector index from PDF

## Repository Structure
- `api.py` — Flask backend
- `app.py` — Streamlit UI
- `templates/index.html` — browser chat UI
- `src/` — RAG pipeline code
- `data/pdf/` — source PDF documentation
- `deploy.sh` — Cloud Run deployment script
- `download_models.py` — pre-cache model artifacts
- `verify_model_cache.py` — runtime cache + env checks
- `test_model_cache.sh` — local environment verification
- `rebuild_model_cache.sh` — clear and rebuild model cache

## Prerequisites
- Python 3.10+
- Pinecone account + API key
- Google Gemini API key
- Hugging Face token (for model artifact download)

## Environment Variables
Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

### Example `.env`

```dotenv
# Gemini / Google AI
GEMINI_API_KEY=your_api_key
# Pinecone
PINECONE_API_KEY=your_key
PINECONE_INDEX=bot-hajster

# Hugging Face (required for model download/cache scripts)
HF_TOKEN=your_token

# Runtime
PORT=8080
PYTHONUNBUFFERED=1
LOG_LEVEL=INFO
MAX_SESSIONS=500

# Optional cache controls (recommended in containerized environments)
HF_HOME=/app/.cache/hf
TRANSFORMERS_CACHE=/app/.cache/hf
SENTENCE_TRANSFORMERS_HOME=/app/.cache/hf
```

Notes:
- Do not commit `.env`.
- Keep `.env.example` sanitized (no real secrets).
- `HF_TOKEN` is required by cache/download scripts and Docker build-time model prefetch.

## Local Development
### 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Validate local environment

```bash
./test_model_cache.sh
```

### 3) Build / refresh vector index

```bash
python3 reindex.py
```

### 4) Run API

```bash
python3 api.py
```

Server default: `http://localhost:8080`

### 5) (Optional) Run Streamlit UI

```bash
streamlit run app.py
```

## Deployment (Google Cloud Run)
Use the provided script:

```bash
./deploy.sh
```

The script performs:
- environment validation
- model cache verification
- Cloud Build image build
- Cloud Run deploy

## Docker
Build locally:

```bash
docker build \
  --build-arg HF_TOKEN=$HF_TOKEN \
  --build-arg GEMINI_API_KEY=$GEMINI_API_KEY \
  --build-arg PINECONE_API_KEY=$PINECONE_API_KEY \
  -t customer-bot:local .
```

Run locally:

```bash
docker run --rm -p 8080:8080 --env-file .env customer-bot:local
```

## Operational Notes
- Re-run `python3 reindex.py` after updating the source PDF.
- Keep Pinecone index name stable between indexing and runtime.
- For production, prefer structured logging and external session persistence over in-memory sessions.

## Security
- Never commit API keys or tokens.
- Rotate secrets if exposed.
- Limit CORS and tighten network/security policies before public deployment.
