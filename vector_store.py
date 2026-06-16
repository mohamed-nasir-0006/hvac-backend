# vector_store.py
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import httpx
import asyncio

# =========================
# Config
# =========================
CHROMA_DIR = Path(__file__).parent / "chroma_data"
COLLECTION_NAME = "hvac_knowledge"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
KB_PATH = Path(__file__).parent / "kb_docs.json"
# =========================
# Ollama config
# =========================
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GEN_MODEL = "llama3"


# =========================
# ChromaDB client (persistent)
# =========================
client = chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Get or create the HVAC knowledge collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )


# =========================
# Embedding helper
# =========================
async def ollama_embed(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Ollama."""
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        resp = await http_client.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


# =========================
# Chunking
# =========================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Character-based chunking with overlap."""
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
# Ingest documents into ChromaDB
# =========================
async def ingest_kb(force_reload: bool = False) -> int:
    """
    Load kb_docs.json, chunk, embed, and store in ChromaDB.
    Returns the number of chunks stored.
    Skips if collection already has data (unless force_reload=True).
    """
    collection = get_collection()

    # Skip if already populated
    if not force_reload and collection.count() > 0:
        print(f"[ChromaDB] Collection already has {collection.count()} chunks. Skipping ingest.")
        return collection.count()

    # Clear existing data if force reloading
    if force_reload and collection.count() > 0:
        print("[ChromaDB] Force reload — clearing existing data.")
        # Delete all existing IDs
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    if not KB_PATH.exists():
        print(f"[ChromaDB] No KB file found at {KB_PATH}.")
        return 0

    docs = json.loads(KB_PATH.read_text(encoding="utf-8"))

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

    # Embed all chunks
    embeddings = await ollama_embed(all_chunks)

    # Upsert into ChromaDB
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
    k: int = 3,
    where_filter: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve top-k similar chunks from ChromaDB.
    Optionally filter by metadata (e.g., {"category": "hvac_basics"}).
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = (await ollama_embed([query]))[0]

    query_params: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(k, collection.count()),
    }

    if where_filter:
        query_params["where"] = where_filter

    results = collection.query(**query_params)

    # Format results
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
# Ingest a single document via API
# =========================
async def ingest_document(
    source_id: str,
    title: str,
    text: str,
    category: str = "general",
) -> Dict[str, Any]:
    """
    Chunk, embed, and store a single document into ChromaDB.
    Returns info about what was stored.
    """
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

    # Embed
    embeddings = await ollama_embed(documents)

    # Upsert (update if same ID exists, insert if new)
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
# Delete a document by source_id
# =========================
def delete_document(source_id: str) -> Dict[str, Any]:
    """
    Remove all chunks belonging to a source_id from ChromaDB.
    """
    collection = get_collection()

    # Find all chunks with this source_id
    results = collection.get(
        where={"source_id": source_id},
    )

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
# List all documents (unique source_ids)
# =========================
def list_documents() -> List[Dict[str, str]]:
    """
    List all unique documents in the collection.
    """
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
# Utility: list all chunks (for debugging)
# =========================
def get_collection_info() -> Dict[str, Any]:
    """Return collection stats."""
    collection = get_collection()
    return {
        "name": COLLECTION_NAME,
        "count": collection.count(),
        "persist_dir": str(CHROMA_DIR),
    }

async def ollama_chat(messages: List[Dict[str, str]]) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            OLLAMA_CHAT_URL,
            json={"model": GEN_MODEL, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

def format_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "CONTEXT: (none)\n"
    lines = ["CONTEXT:"]
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        title = meta.get("title", "unknown")
        source = meta.get("source_id", "unknown")
        lines.append(f"[{i}] {title} ({source}): {c['text']}")
    return "\n".join(lines) + "\n"

def parse_llm_json(raw_text: str) -> Dict[str, Any]:
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