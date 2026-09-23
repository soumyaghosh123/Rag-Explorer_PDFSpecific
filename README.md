# RAG Explorer

A lightweight, local Retrieval-Augmented Generation (RAG) demo that ingests a PDF, chunks it, embeds it, stores it in a vector database, and lets you search/chat against it with an LLM.

## Architecture

```
PDF -> extract -> normalise -> chunk -> embed -> vector store -> (search) -> LLM -> answer
```

## Components & Technology

### Frontend (`frontend/`)
| Purpose | Technology |
|---|---|
| UI framework | React (via Vite) |
| Build tool / dev server | Vite |
| Icons | lucide-react |
| Styling | Plain CSS (`src/styles.css`), no framework |
| State management | React `useState`/`useEffect` (no external state library) |

Runs at `http://localhost:5173`.

### Backend (`backend/`)
| Purpose | Technology |
|---|---|
| Web framework | FastAPI |
| ASGI server | Uvicorn |
| PDF text extraction | pypdf |
| Embedding model | `fastembed` (ONNX runtime, no PyTorch) running `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local, open-source, no API calls, ~200MB total runtime footprint — chosen to fit free-tier hosting RAM limits) |
| Vector database | Qdrant (`qdrant-client`, embedded/local mode — no separate server needed) |
| LLM (answer generation) | `openai/gpt-oss-120b` served via the Groq API (`GROQ_API_KEY` from `.env`) |
| Config / secrets | `python-dotenv` (reads `.env` at project root) |
| File uploads | `python-multipart` |
| Numerical ops | NumPy |

Runs at `http://localhost:8000`.

### Vector Database
| Detail | Value |
|---|---|
| Engine | [Qdrant](https://qdrant.tech/) via `qdrant-client` |
| Mode | Embedded/local — runs in-process, no separate Qdrant server or Docker container needed |
| Storage location | `backend/.qdrant/` (created automatically on first run) |
| Collection name | `rag_explorer_chunks` |
| Distance metric | Cosine similarity |
| Vector size | 384 (matches the all-MiniLM-L6-v2 output dimension) |
| Point ID | Same as the chunk id (integer) |
| Point payload | `page` (source PDF page number), `word_count` (chunk size) |
| Lifecycle | Collection is dropped and recreated on every ingest/reindex, so it always reflects the current PDF + chunk settings |

### Pipeline stages (as shown in the "Ingest & Chunks" tab)
1. **Extract** — pypdf, per page
2. **Normalise** — whitespace/text repair via regex
3. **Chunk** — sliding window over words (configurable size/overlap via UI sliders)
4. **Embed** — all-MiniLM-L6-v2, 384 dimensions, computed locally on CPU
5. **Store** — Qdrant collection with cosine similarity

### Data flow per query
1. Query text -> embedded with the same embedding model
2. Top-k nearest chunks retrieved from Qdrant (cosine similarity)
3. Retrieved chunks + question -> prompt sent to `openai/gpt-oss-120b` on Groq
4. LLM's grounded, chunk-cited answer returned to the UI along with token usage and per-stage timing (embed/search/LLM ms)

## Environment variables

**Backend (`.env` locally, or host env vars in production):**
- `GROQ_API_KEY` — required for LLM answer generation
- `EMBEDDING_MODEL` — optional override (defaults to `sentence-transformers/all-MiniLM-L6-v2`; note nomic-family models need >512MB RAM and won't fit Render's free tier)
- `GROQ_MODEL` — optional override (defaults to `openai/gpt-oss-120b`)
- `ALLOWED_ORIGINS` — comma-separated extra CORS origins (e.g. your deployed Vercel URL)

**Frontend (Vite build-time env var):**
- `VITE_API_URL` — base URL of the backend API (e.g. `https://rag-explorer-backend.onrender.com`). Defaults to `http://localhost:8000` for local dev.

## Running locally

**Backend:**
```
cd backend
.venv\Scripts\activate
uvicorn main:app
```

**Frontend:**
```
cd frontend
npm run dev
```

Then open `http://localhost:5173`.

## Deployment

The backend (FastAPI + local ML model + embedded Qdrant) needs a host with a real, long-running Python process and persistent-enough disk — it cannot run as a Vercel serverless function. The frontend is a static Vite build and deploys to Vercel easily.

**Backend on Render:**
1. Push this repo to GitHub (already done).
2. In Render, "New +" -> "Blueprint", point it at this repo. It will read `render.yaml` and create a free web service (`rag-explorer-backend`) with `rootDir: backend`.
3. In the Render dashboard, set the environment variables it prompts for:
   - `GROQ_API_KEY` — your Groq key
   - `ALLOWED_ORIGINS` — your Vercel frontend URL once you have it (e.g. `https://rag-explorer.vercel.app`)
4. Deploy. First boot takes a bit since it downloads the embedding model on startup; free-tier services also spin down after 15 minutes idle and cold-start on the next request.

**Frontend on Vercel:**
1. From the `frontend/` directory, run `vercel` (or `vercel --prod`) and follow the prompts.
2. Set the `VITE_API_URL` environment variable in the Vercel project settings to your Render backend's public URL.
3. Redeploy after setting the env var so the build picks it up.
