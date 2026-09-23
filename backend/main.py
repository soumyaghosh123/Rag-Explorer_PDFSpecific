from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = next(ROOT.glob("*.pdf"), None)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
GENERATION_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
QDRANT_DIR = ROOT / "backend" / ".qdrant"
DEFAULT_CHUNK_SIZE = 180
DEFAULT_CHUNK_OVERLAP = 40
COLLECTION_NAME = "rag_explorer_chunks"

DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
EXTRA_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]

app = FastAPI(title="RAG Explorer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS + EXTRA_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class IngestSettings(BaseModel):
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


class RagStore:
    def __init__(self) -> None:
        self.document_name = ""
        self.pages = 0
        self.chunks: list[dict[str, Any]] = []
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.chunk_overlap = DEFAULT_CHUNK_OVERLAP
        self.embedder: SentenceTransformer | None = None
        self.embedding_status = "not loaded"
        self.embedding_dimensions = 0
        self.pdf_path: Path | None = None

        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        self.qdrant_client = QdrantClient(path=str(QDRANT_DIR))

        if DEFAULT_PDF:
            try:
                self.ingest(DEFAULT_PDF, self.chunk_size, self.chunk_overlap)
            except Exception as exc:  # keep API usable even if first ingest fails
                self.embedding_status = f"startup ingest failed: {exc}"

    def _reset_collection(self, vector_size: int) -> None:
        if self.qdrant_client.collection_exists(COLLECTION_NAME):
            self.qdrant_client.delete_collection(COLLECTION_NAME)
        self.qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def load_embedder(self) -> None:
        if self.embedder is not None:
            return
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        self.embedding_dimensions = self.embedder.get_sentence_embedding_dimension()
        self.embedding_status = "loaded"

    def embed(self, texts: list[str], prefix: str) -> np.ndarray:
        self.load_embedder()
        prefixed = [f"{prefix}: {text}" for text in texts]
        vectors = self.embedder.encode(prefixed, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def ingest(self, pdf_path: Path | None, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
        if not pdf_path or not pdf_path.exists():
            raise FileNotFoundError("No PDF was found in the project folder.")
        chunk_size = max(20, chunk_size)
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

        reader = PdfReader(str(pdf_path))
        words: list[str] = []
        page_ranges: list[tuple[int, int, int]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            normalised = re.sub(r"\s+", " ", raw_text).strip()
            page_words = normalised.split(" ") if normalised else []
            start = len(words)
            words.extend(page_words)
            page_ranges.append((start, len(words), page_number))

        chunks: list[dict[str, Any]] = []
        spans: list[tuple[int, int]] = []
        start = 0
        chunk_id = 1
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            pages = [page for page_start, page_end, page in page_ranges if page_start < end and page_end > start]
            chunks.append({
                "id": chunk_id,
                "text": " ".join(chunk_words),
                "page": pages[0] if pages else 1,
                "pages": pages,
                "word_count": len(chunk_words),
            })
            spans.append((start, end))
            if end == len(words):
                break
            start = end - chunk_overlap
            chunk_id += 1

        for index, chunk in enumerate(chunks):
            this_start, this_end = spans[index]
            prev_overlap = 0
            prev_text = ""
            if index > 0:
                prev_start, prev_end = spans[index - 1]
                prev_overlap = max(0, prev_end - this_start)
                if prev_overlap:
                    prev_text = " ".join(words[this_start:this_start + prev_overlap])
            next_overlap = 0
            next_text = ""
            if index < len(chunks) - 1:
                next_start, _next_end = spans[index + 1]
                next_overlap = max(0, this_end - next_start)
                if next_overlap:
                    next_text = " ".join(words[this_end - next_overlap:this_end])
            chunk["overlap_prev_words"] = prev_overlap
            chunk["overlap_prev_text"] = prev_text
            chunk["overlap_next_words"] = next_overlap
            chunk["overlap_next_text"] = next_text

        vectors = self.embed([chunk["text"] for chunk in chunks], prefix="search_document")

        self._reset_collection(vector_size=vectors.shape[1])
        self.qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=chunk["id"],
                    vector=vectors[index].tolist(),
                    payload={"page": chunk["page"], "word_count": chunk["word_count"]},
                )
                for index, chunk in enumerate(chunks)
            ],
        )

        for index, chunk in enumerate(chunks):
            chunk["vector_id"] = chunk["id"]
            chunk["vector_dims"] = int(vectors.shape[1])
            chunk["vector_preview"] = [round(float(value), 4) for value in vectors[index][:6]]
            chunk["collection"] = COLLECTION_NAME

        self.document_name = pdf_path.name
        self.pages = len(reader.pages)
        self.chunks = chunks
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_path = pdf_path
        return self.stats()

    def search(self, query: str, top_k: int = 3) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if not self.chunks:
            return [], {"embed_ms": 0.0, "search_ms": 0.0}
        embed_start = time.perf_counter()
        query_vector = self.embed([query], prefix="search_query")[0]
        embed_ms = (time.perf_counter() - embed_start) * 1000

        search_start = time.perf_counter()
        hits = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=max(1, min(top_k, len(self.chunks))),
        ).points
        search_ms = (time.perf_counter() - search_start) * 1000

        chunk_lookup = {chunk["id"]: chunk for chunk in self.chunks}
        matches: list[dict[str, Any]] = []
        for hit in hits:
            chunk = chunk_lookup.get(hit.id)
            if not chunk:
                continue
            matches.append({**chunk, "score": round(float(hit.score), 4)})
        return matches, {"embed_ms": round(embed_ms, 2), "search_ms": round(search_ms, 2)}

    def stats(self) -> dict[str, Any]:
        return {
            "document": self.document_name,
            "pages": self.pages,
            "chunks": len(self.chunks),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimensions": self.embedding_dimensions,
            "chunk_size_words": self.chunk_size,
            "chunk_overlap_words": self.chunk_overlap,
            "embedding_status": self.embedding_status,
            "generation_model": GENERATION_MODEL,
            "vector_store": "qdrant · cosine",
        }


store = RagStore()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    return store.stats()


@app.get("/api/chunks")
def chunks() -> dict[str, Any]:
    return {"chunks": store.chunks}


@app.post("/api/ingest")
async def ingest(
    file: UploadFile = File(...),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    upload_path = ROOT / Path(file.filename).name
    upload_path.write_bytes(await file.read())
    try:
        return store.ingest(upload_path, chunk_size, chunk_overlap)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not ingest PDF: {exc}") from exc


@app.post("/api/reindex")
def reindex(settings: IngestSettings) -> dict[str, Any]:
    if not store.pdf_path:
        raise HTTPException(status_code=400, detail="No PDF ingested yet.")
    try:
        return store.ingest(store.pdf_path, settings.chunk_size, settings.chunk_overlap)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not reindex PDF: {exc}") from exc


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Enter a question first.")
    results, metrics = store.search(request.query.strip(), request.top_k)
    answer = None
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and results:
        context = "\n\n".join(f"[Chunk {item['id']}, page {item['page']}] {item['text']}" for item in results)
        prompt = (
            "Answer the user's question using only the supplied document context. "
            "Format the answer as markdown: start with a short bold headline line, then bullet points "
            "where each bullet begins with a bold term followed by ' - ' and a short explanation. "
            "After each bullet, cite the chunk(s) it came from in the exact form [Chunk N] (one bracket per chunk id). "
            "If the context is insufficient, say so plainly instead of guessing.\n\n"
            f"Context:\n{context}\n\nQuestion: {request.query}"
        )
        llm_start = time.perf_counter()
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={"model": GENERATION_MODEL, "temperature": 0.1, "max_tokens": 1200, "messages": [{"role": "user", "content": prompt}]},
            timeout=45,
        )
        metrics["llm_ms"] = round((time.perf_counter() - llm_start) * 1000, 2)
        if response.ok:
            payload = response.json()
            answer = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage", {})
            metrics["prompt_tokens"] = usage.get("prompt_tokens", 0)
            metrics["output_tokens"] = usage.get("completion_tokens", 0)
        else:
            answer = f"Groq could not answer ({response.status_code}); inspect the retrieved chunks below."
    return {"query": request.query, "results": results, "answer": answer, "llm_configured": bool(groq_key), "metrics": metrics}
