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
| Embedding model | `sentence-transformers` running `nomic-ai/nomic-embed-text-v1.5` (768-dim, local, open-source, no API calls) |
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
| Vector size | 768 (matches the nomic-embed-text-v1.5 output dimension) |
| Point ID | Same as the chunk id (integer) |
| Point payload | `page` (source PDF page number), `word_count` (chunk size) |
| Lifecycle | Collection is dropped and recreated on every ingest/reindex, so it always reflects the current PDF + chunk settings |

### Pipeline stages (as shown in the "Ingest & Chunks" tab)
1. **Extract** — pypdf, per page
2. **Normalise** — whitespace/text repair via regex
3. **Chunk** — sliding window over words (configurable size/overlap via UI sliders)
4. **Embed** — nomic-embed-text-v1.5, 768 dimensions, computed locally on CPU
5. **Store** — Qdrant collection with cosine similarity

### Data flow per query
1. Query text -> embedded with the same nomic model (`search_query` prefix)
2. Top-k nearest chunks retrieved from Qdrant (cosine similarity)
3. Retrieved chunks + question -> prompt sent to `openai/gpt-oss-120b` on Groq
4. LLM's grounded, chunk-cited answer returned to the UI along with token usage and per-stage timing (embed/search/LLM ms)

## Environment variables (`.env`)
- `GROQ_API_KEY` — required for LLM answer generation
- `EMBEDDING_MODEL` — optional override (defaults to `nomic-ai/nomic-embed-text-v1.5`)
- `GROQ_MODEL` — optional override (defaults to `openai/gpt-oss-120b`)

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
