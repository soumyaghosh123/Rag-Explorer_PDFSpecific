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
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDFS = sorted(ROOT.glob("*.pdf"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
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
        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.chunk_overlap = DEFAULT_CHUNK_OVERLAP
        self.embedder: TextEmbedding | None = None
        self.embedding_status = "not loaded"
        self.embedding_dimensions = 0
        self.pdf_paths: list[Path] = []

        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        self.qdrant_client = QdrantClient(path=str(QDRANT_DIR))

        if DEFAULT_PDFS:
            try:
                self.add_documents(DEFAULT_PDFS, self.chunk_size, self.chunk_overlap)
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
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        sample = next(self.embedder.embed(["dimension probe"]))
        self.embedding_dimensions = len(sample)
        self.embedding_status = "loaded"

    def embed(self, texts: list[str], prefix: str) -> np.ndarray:
        self.load_embedder()
        vectors = np.array(list(self.embedder.embed(texts)), dtype=np.float32)
        return vectors

    def _chunk_document(self, pdf_path: Path, chunk_size: int, chunk_overlap: int, chunk_id_start: int) -> tuple[list[dict[str, Any]], int]:
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
        chunk_id = chunk_id_start
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            pages = [page for page_start, page_end, page in page_ranges if page_start < end and page_end > start]
            chunks.append({
                "id": chunk_id,
                "document": pdf_path.name,
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

        self.documents.append({"name": pdf_path.name, "pages": len(reader.pages)})
        return chunks, chunk_id + 1

    def _rebuild(self, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
        chunk_size = max(20, chunk_size)
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

        self.documents = []
        all_chunks: list[dict[str, Any]] = []
        next_id = 1
        for pdf_path in self.pdf_paths:
            doc_chunks, next_id = self._chunk_document(pdf_path, chunk_size, chunk_overlap, next_id)
            all_chunks.extend(doc_chunks)

        vectors = self.embed([chunk["text"] for chunk in all_chunks], prefix="search_document") if all_chunks else np.zeros((0, self.embedding_dimensions or 1), dtype=np.float32)

        if all_chunks:
            self._reset_collection(vector_size=vectors.shape[1])
            self.qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=chunk["id"],
                        vector=vectors[index].tolist(),
                        payload={"document": chunk["document"], "page": chunk["page"], "word_count": chunk["word_count"]},
                    )
                    for index, chunk in enumerate(all_chunks)
                ],
            )

        for index, chunk in enumerate(all_chunks):
            chunk["vector_id"] = chunk["id"]
            chunk["vector_dims"] = int(vectors.shape[1]) if all_chunks else 0
            chunk["vector_preview"] = [round(float(value), 4) for value in vectors[index][:6]] if all_chunks else []
            chunk["collection"] = COLLECTION_NAME

        self.chunks = all_chunks
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        return self.stats()

    def add_documents(self, pdf_paths: list[Path], chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                raise FileNotFoundError(f"{pdf_path.name} was not found.")
            if pdf_path not in self.pdf_paths:
                self.pdf_paths.append(pdf_path)
        return self._rebuild(chunk_size, chunk_overlap)

    def remove_document(self, name: str, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
        self.pdf_paths = [p for p in self.pdf_paths if p.name != name]
        return self._rebuild(chunk_size, chunk_overlap)

    def clear_documents(self) -> dict[str, Any]:
        self.pdf_paths = []
        return self._rebuild(self.chunk_size, self.chunk_overlap)

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
            "documents": self.documents,
            "document_count": len(self.documents),
            "document": self.documents[0]["name"] if len(self.documents) == 1 else (f"{len(self.documents)} documents" if self.documents else ""),
            "pages": sum(doc["pages"] for doc in self.documents),
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
    files: list[UploadFile] = File(...),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    saved_paths: list[Path] = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename or 'upload'} is not a PDF file.")
        upload_path = ROOT / Path(file.filename).name
        upload_path.write_bytes(await file.read())
        saved_paths.append(upload_path)
    try:
        return store.add_documents(saved_paths, chunk_size, chunk_overlap)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not ingest PDF(s): {exc}") from exc


@app.post("/api/reindex")
def reindex(settings: IngestSettings) -> dict[str, Any]:
    if not store.pdf_paths:
        raise HTTPException(status_code=400, detail="No PDF ingested yet.")
    try:
        return store._rebuild(settings.chunk_size, settings.chunk_overlap)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not reindex: {exc}") from exc


@app.delete("/api/documents/{name}")
def remove_document(name: str) -> dict[str, Any]:
    try:
        return store.remove_document(name, store.chunk_size, store.chunk_overlap)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not remove document: {exc}") from exc


@app.delete("/api/documents")
def clear_documents() -> dict[str, Any]:
    return store.clear_documents()


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Enter a question first.")
    results, metrics = store.search(request.query.strip(), request.top_k)
    answer = None
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and results:
        context = "\n\n".join(f"[Chunk {item['id']}, {item['document']}, page {item['page']}] {item['text']}" for item in results)
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
