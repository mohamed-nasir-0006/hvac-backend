# app.py
"""
Generic Chat Backend
All configuration comes from saas_config.json.
This code works for ANY domain.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
import httpx
import json

from intents import get_intent_prompt_block
from file_parser import extract_text
from saas_bridge import execute_intent
from config_loader import (
    get_app_name,
    get_debug_mode,
    get_cors_origins,
    get_allowed_upload_types,
    get_default_category,
    get_chat_url,
    get_chat_model,
    get_llm_timeout,
    get_default_top_k,
    build_system_prompt,
    extract_parsed_fields,
    is_actionable_intent,
)
from vector_store import (
    ingest_kb,
    ingest_document,
    retrieve,
    get_collection_info,
    delete_document,
    list_documents,
    format_context,
    ollama_chat,
    parse_llm_json,
)
from memory import (
    init_db,
    create_session,
    save_message,
    get_session_messages,
    get_session_history,
    list_sessions,
    delete_session,
)

# =========================
# FastAPI app + CORS (from config)
# =========================
app = FastAPI(debug=get_debug_mode(), title=f"{get_app_name()} Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic models (Generic)
# =========================
class Msg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Msg] = []
    session_id: str | None = None

class IngestRequest(BaseModel):
    source_id: str = Field(..., description="Unique ID for this document")
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Document content")
    category: str = Field(default="general", description="Category for filtering")

class BulkIngestRequest(BaseModel):
    documents: List[IngestRequest]

class DeleteRequest(BaseModel):
    source_id: str


# =========================
# Build system prompt from config
# =========================
INTENT_BLOCK = get_intent_prompt_block()
SYSTEM_PROMPT = build_system_prompt(INTENT_BLOCK)


# =========================
# Startup
# =========================
@app.on_event("startup")
async def startup():
    count = await ingest_kb()
    init_db()
    print(f"[Startup] {get_app_name()} Backend ready with {count} chunks.")


# =========================
# Routes
# =========================
@app.get("/")
async def root():
    return {
        "message": f"{get_app_name()} Backend running",
        "bridge": "enabled",
    }


@app.get("/kb/info")
async def kb_info():
    return get_collection_info()


@app.post("/kb/reload")
async def kb_reload():
    count = await ingest_kb(force_reload=True)
    return {"message": f"Reloaded {count} chunks into ChromaDB."}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    # Session management
    session_id = req.session_id
    if not session_id:
        session = create_session()
        session_id = session["session_id"]

    if req.session_id:
        history_dicts = get_session_history(session_id)
        history_msgs = [Msg(role=h["role"], content=h["content"]) for h in history_dicts]
    else:
        history_msgs = list(req.history)

    save_message(session_id, "user", req.message)

    # Build messages
    top_chunks = await retrieve(query=req.message)
    context_block = format_context(top_chunks)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": context_block},
    ]
    messages += [m.model_dump() for m in history_msgs]
    messages.append({"role": "user", "content": req.message})

    # Stream from LLM
    async def generate():
        full_reply = ""
        try:
            async with httpx.AsyncClient(timeout=get_llm_timeout()) as client:
                async with client.stream(
                    "POST",
                    get_chat_url(),
                    json={"model": get_chat_model(), "messages": messages, "stream": True},
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            full_reply += token

                            yield f"data: {json.dumps({'token': token, 'session_id': session_id})}\n\n"

                            if chunk.get("done", False):
                                break

            # Parse — dynamic field extraction from config
            parsed_dict = parse_llm_json(full_reply)
            parsed = extract_parsed_fields(parsed_dict)
            human_reply = parsed_dict.get("reply", full_reply)

            # ========== 🌉 BRIDGE ==========
            bridge_result = None
            if is_actionable_intent(parsed.get("intent", "")):
                try:
                    bridge_result = execute_intent(parsed)
                    if bridge_result.get("success"):
                        human_reply = bridge_result["reply"]
                        print(f"[Bridge] ✅ {parsed['intent']} executed successfully")
                    elif bridge_result.get("error"):
                        human_reply = bridge_result["reply"]
                        print(f"[Bridge] ❌ {parsed['intent']} failed: {bridge_result['error']}")
                except Exception as e:
                    print(f"[Bridge] ⚠️ Bridge error: {e}")
            # ========== END BRIDGE ==========

            save_message(session_id, "assistant", human_reply, parsed)

            yield f"data: {json.dumps({'done': True, 'reply': human_reply, 'parsed': parsed, 'session_id': session_id})}\n\n"

        except Exception as e:
            fallback = f"(Fallback) Could not reach local LLM: {e}"
            save_message(session_id, "assistant", fallback)
            yield f"data: {json.dumps({'token': fallback, 'done': True, 'session_id': session_id})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# =========================
# Document Management Routes
# =========================
@app.post("/ingest")
async def ingest_single(req: IngestRequest):
    result = await ingest_document(
        source_id=req.source_id,
        title=req.title,
        text=req.text,
        category=req.category,
    )
    return result


@app.post("/ingest/bulk")
async def ingest_bulk(req: BulkIngestRequest):
    results = []
    for doc in req.documents:
        result = await ingest_document(
            source_id=doc.source_id,
            title=doc.title,
            text=doc.text,
            category=doc.category,
        )
        results.append(result)

    total_chunks = sum(r.get("chunks_added", 0) for r in results)
    return {
        "status": "success",
        "documents_processed": len(results),
        "total_chunks_added": total_chunks,
        "details": results,
    }


@app.get("/documents")
async def get_documents():
    docs = list_documents()
    return {"documents": docs, "count": len(docs)}


@app.delete("/documents/{source_id}")
async def remove_document(source_id: str):
    result = delete_document(source_id)
    return result


# =========================
# File Upload Route
# =========================
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(default=get_default_category()),
):
    allowed_types = get_allowed_upload_types()
    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_types:
        return {
            "status": "error",
            "message": f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_types)}",
        }

    file_bytes = await file.read()

    try:
        text = extract_text(file.filename, file_bytes)
    except Exception as e:
        return {"status": "error", "message": f"Failed to extract text: {str(e)}"}

    if not text.strip():
        return {"status": "error", "message": "No text could be extracted from the file."}

    source_id = file.filename.rsplit(".", 1)[0].replace(" ", "-").lower()

    result = await ingest_document(
        source_id=source_id,
        title=file.filename,
        text=text,
        category=category,
    )

    result["extracted_text_length"] = len(text)
    result["original_filename"] = file.filename

    return result


# =========================
# Session Management Routes
# =========================
@app.post("/sessions")
async def new_session():
    return create_session()


@app.get("/sessions")
async def get_sessions():
    sessions = list_sessions()
    return {"sessions": sessions}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    messages = get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}