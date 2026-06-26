# vector_store.py
"""
Vector Store
Handles ChromaDB operations, embeddings, and LLM chat.
All constants come from saas_config.json via config_loader.
"""

import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import httpx

from config_loader import (
    get_chroma_dir,
    get_collection_name,
    get_similarity_metric,
    get_chunk_size,
    get_chunk_overlap,
    get_kb_path,
    get_embed_url,
    get_embed_model,
    get_chat_url,
    get_chat_model,
    get_llm_timeout,
)

# =========================
# ChromaDB client (persistent)
# =========================
client = chromadb.PersistentClient(path=str(get_chroma_dir()))


def get_collection():
    """Get or create the knowledge collection."""
    return client.get_or_create_collection(
        name=get_collection_name(),
        metadata={"hnsw:space": get_similarity_metric()},
    )


# =========================
# Embedding helper
# =========================
async def ollama_embed(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Ollama."""
    async with httpx.AsyncClient(timeout=float(get_llm_timeout())) as http_client:
        resp = await http_client.post(
            get_embed_url(),
            json={"model": get_embed_model(), "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


# =========================
# Chunking
# =========================
def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Character-based chunking with overlap. Sizes from config."""
    if chunk_size is None:
        chunk_size = get_chunk_size()
    if overlap is None:
        overlap = get_chunk_overlap()

    text = text.strip()
    if not text:
        return []

    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + chunk_size)
        chunks.append(text[i:end])
        if end == len(text):
            break
        i = end - overlap
        if i < 0:
            i = 0
    return chunks


# =========================
# Ingest KB into ChromaDB
# =========================
async def ingest_kb(force_reload: bool = False) -> int:
    """
    Load KB file, chunk, embed, and store in ChromaDB.
    Returns the number of chunks stored.
    """
    collection = get_collection()

    if not force_reload and collection.count() > 0:
        print(f"[ChromaDB] Collection already has {collection.count()} chunks. Skipping ingest.")
        return collection.count()

    if force_reload and collection.count() > 0:
        print("[ChromaDB] Force reload — clearing existing data.")
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    kb_path = get_kb_path()
    if not kb_path.exists():
        print(f"[ChromaDB] No KB file found at {kb_path}.")
        return 0

    docs = json.loads(kb_path.read_text(encoding="utf-8"))

    all_chunks: List[str] = []
    all_ids: List[str] = []
    all_metadata: List[Dict[str, str]] = []

    for d in docs:
        source_id = d.get("id", "unknown")
        title = d.get("title", source_id)
        category = d.get("category", "general")
        text = d.get("text", "")

        for idx, ch in enumerate(chunk_text(text), start=1):
            chunk_id = f"{source_id}::chunk{idx}"
            all_chunks.append(ch)
            all_ids.append(chunk_id)
            all_metadata.append({
                "source_id": source_id,
                "title": title,
                "category": category,
                "chunk_index": str(idx),
            })

    if not all_chunks:
        print("[ChromaDB] No text chunks found in KB.")
        return 0

    embeddings = await ollama_embed(all_chunks)

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
    )

    print(f"[ChromaDB] Ingested {len(docs)} docs -> {len(all_chunks)} chunks stored.")
    return len(all_chunks)


# =========================
# Retrieve similar chunks
# =========================
async def retrieve(
    query: str,
    k: int = None,
    where_filter: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve top-k similar chunks from ChromaDB.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    if k is None:
        from config_loader import get_default_top_k
        k = get_default_top_k()

    query_embedding = (await ollama_embed([query]))[0]

    query_params: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(k, collection.count()),
    }

    if where_filter:
        query_params["where"] = where_filter

    results = collection.query(**query_params)

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
        })

    return chunks


# =========================
# Ingest a single document
# =========================
async def ingest_document(
    source_id: str,
    title: str,
    text: str,
    category: str = "general",
) -> Dict[str, Any]:
    """Chunk, embed, and store a single document into ChromaDB."""
    collection = get_collection()

    chunks = chunk_text(text)
    if not chunks:
        return {"status": "skipped", "reason": "empty text", "chunks_added": 0}

    ids = []
    documents = []
    metadatas = []

    for idx, ch in enumerate(chunks, start=1):
        chunk_id = f"{source_id}::chunk{idx}"
        ids.append(chunk_id)
        documents.append(ch)
        metadatas.append({
            "source_id": source_id,
            "title": title,
            "category": category,
            "chunk_index": str(idx),
        })

    embeddings = await ollama_embed(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "status": "success",
        "source_id": source_id,
        "title": title,
        "category": category,
        "chunks_added": len(chunks),
        "total_chunks_in_db": collection.count(),
    }


# =========================
# Delete a document
# =========================
def delete_document(source_id: str) -> Dict[str, Any]:
    """Remove all chunks belonging to a source_id."""
    collection = get_collection()

    results = collection.get(where={"source_id": source_id})

    if not results["ids"]:
        return {"status": "not_found", "source_id": source_id, "chunks_deleted": 0}

    collection.delete(ids=results["ids"])

    return {
        "status": "deleted",
        "source_id": source_id,
        "chunks_deleted": len(results["ids"]),
        "total_chunks_in_db": collection.count(),
    }


# =========================
# List all documents
# =========================
def list_documents() -> List[Dict[str, str]]:
    """List all unique documents in the collection."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    all_data = collection.get()
    seen = {}

    for meta in all_data["metadatas"]:
        sid = meta.get("source_id", "unknown")
        if sid not in seen:
            seen[sid] = {
                "source_id": sid,
                "title": meta.get("title", "unknown"),
                "category": meta.get("category", "general"),
            }

    return list(seen.values())


# =========================
# Collection info
# =========================
def get_collection_info() -> Dict[str, Any]:
    """Return collection stats."""
    return {
        "name": get_collection_name(),
        "count": get_collection().count(),
        "persist_dir": str(get_chroma_dir()),
    }


# =========================
# LLM Chat
# =========================
async def ollama_chat(messages: List[Dict[str, str]]) -> str:
    """Send messages to LLM and get response."""
    async with httpx.AsyncClient(timeout=float(get_llm_timeout())) as http_client:
        resp = await http_client.post(
            get_chat_url(),
            json={"model": get_chat_model(), "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# =========================
# Context formatter
# =========================
def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as context for LLM."""
    if not chunks:
        return "CONTEXT: (none)\n"
    lines = ["CONTEXT:"]
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        title = meta.get("title", "unknown")
        source = meta.get("source_id", "unknown")
        lines.append(f"[{i}] {title} ({source}): {c['text']}")
    return "\n".join(lines) + "\n"


# =========================
# JSON parser
# =========================
def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "intent": "unclear",
            "reply": raw_text,
            "confidence": 0.0,
        }