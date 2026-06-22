from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any, Tuple
import httpx
import json
from intents import get_intent_prompt_block
from file_parser import extract_text
from vector_store import (
    ingest_kb,
    ingest_document,
    retrieve,
    get_collection_info,
    delete_document,
    list_documents,
    format_context, ollama_chat, parse_llm_json
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
# FastAPI app + CORS
# =========================
app = FastAPI(debug=True)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic models
# =========================
class Msg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Msg] = []
    session_id: str | None = None 

class ParsedIntent(BaseModel):
    intent: str = "unclear"
    zone: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    mode: Optional[str] = None
    schedule: Optional[str] = None
    confidence: Optional[float] = None

class ChatResponse(BaseModel):
    reply: str
    parsed: ParsedIntent | None
    history: List[Msg]
    context_used: list | None = None
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

INTENT_BLOCK = get_intent_prompt_block()

SYSTEM_PROMPT = f"""You are an HVAC control assistant.

Your job:
1. Understand the user's intent from their message and conversation history.
2. Extract structured data.
3. Respond with a JSON object (and ONLY a JSON object, no extra text).

{INTENT_BLOCK}

RESPONSE FORMAT (strict JSON, no markdown, no explanation outside the JSON):
{{
  "intent": "<one of the known intents>",
  "zone": "<zone name or null>",
  "value": <number or null>,
  "unit": "<C or F or null>",
  "mode": "<cooling|heating|auto|off|setback or null>",
  "schedule": "<ISO datetime string or natural language time or null>",
  "confidence": <0.0 to 1.0>,
  "reply": "<short human-friendly response>"
}}

RULES:
- Always respond with valid JSON only.
- If the user says "yes" or confirms, use conversation history to determine the intent.
- If intent is unclear, set intent to "unclear" and ask a clarifying question in "reply".
- For general HVAC knowledge questions, set intent to "general_question".
- Use the CONTEXT (retrieved documents) to answer knowledge questions.
"""

# =========================
# Startup — ingest KB into ChromaDB
# =========================
@app.on_event("startup")
async def startup():
    count = await ingest_kb()
    init_db()
    print(f"[Startup] ChromaDB ready with {count} chunks.")

# =========================
# Routes
# =========================
@app.get("/")
async def root():
    return {"message": "HVAC backend (Ollama + ChromaDB + Intent) running"}

@app.get("/kb/info")
async def kb_info():
    """Check ChromaDB collection status."""
    return get_collection_info()

@app.post("/kb/reload")
async def kb_reload():
    """Force reload KB into ChromaDB."""
    count = await ingest_kb(force_reload=True)
    return {"message": f"Reloaded {count} chunks into ChromaDB."}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # ===== NEW: Session management =====
    session_id = req.session_id
    if not session_id:
        session = create_session()
        session_id = session["session_id"]
    # Load history from DB if resuming, otherwise use frontend history
    if req.session_id:
        history_dicts = get_session_history(session_id)
        history_msgs = [Msg(role=h["role"], content=h["content"]) for h in history_dicts]
    else:
        history_msgs = list(req.history)
    # Save user message to DB
    save_message(session_id, "user", req.message)
    # ===== END NEW =====
    # 1) Retrieve context from ChromaDB (YOUR EXISTING CODE)
    top_chunks = await retrieve(query=req.message, k=3)
    context_block = format_context(top_chunks)
    # 2) Build messages (YOUR EXISTING CODE)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": context_block},
    ]
    messages += [m.model_dump() for m in history_msgs]    # ← Changed from req.history
    messages.append({"role": "user", "content": req.message})
    try:
        # 3) Call LLM (YOUR EXISTING CODE)
        reply_text = await ollama_chat(messages)
        # 4) Parse structured JSON (YOUR EXISTING CODE)
        parsed_dict = parse_llm_json(reply_text)
        parsed = ParsedIntent(
            intent=parsed_dict.get("intent", "unclear"),
            zone=parsed_dict.get("zone"),
            value=parsed_dict.get("value"),
            unit=parsed_dict.get("unit"),
            mode=parsed_dict.get("mode"),
            schedule=parsed_dict.get("schedule"),
            confidence=parsed_dict.get("confidence"),
        )
        human_reply = parsed_dict.get("reply", reply_text)
        # 5) Update history (YOUR EXISTING CODE)
        new_history = list(history_msgs)    # ← Changed from req.history
        new_history.append(Msg(role="user", content=req.message))
        new_history.append(Msg(role="assistant", content=human_reply))
        # ===== NEW: Save assistant reply to DB =====
        save_message(session_id, "assistant", human_reply, parsed.model_dump())
        # ===== END NEW =====
        return ChatResponse(
            reply=human_reply,
            parsed=parsed,
            history=new_history,
            context_used=[
                {
                    "chunk_id": c["chunk_id"],
                    "title": c["metadata"].get("title"),
                    "distance": c.get("distance"),
                }
                for c in top_chunks
            ],
            session_id=session_id,          # ← ADD this
        )
    except Exception as e:
        new_history = list(history_msgs)    # ← Changed from req.history
        new_history.append(Msg(role="user", content=req.message))
        fallback = f"(Fallback) Could not reach local LLM: {e}"
        new_history.append(Msg(role="assistant", content=fallback))
        # ===== NEW: Save fallback to DB =====
        save_message(session_id, "assistant", fallback)
        # ===== END NEW =====
        return ChatResponse(
            reply=fallback,
            parsed=None,
            history=new_history,
            context_used=None,
            session_id=session_id,          # ← ADD this
        )
    
# =========================
# Document Management Routes
# =========================

@app.post("/ingest")
async def ingest_single(req: IngestRequest):
    """Add a single document to the knowledge base."""
    result = await ingest_document(
        source_id=req.source_id,
        title=req.title,
        text=req.text,
        category=req.category,
    )
    return result


@app.post("/ingest/bulk")
async def ingest_bulk(req: BulkIngestRequest):
    """Add multiple documents at once."""
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
    """List all documents in the knowledge base."""
    docs = list_documents()
    return {"documents": docs, "count": len(docs)}


@app.delete("/documents/{source_id}")
async def remove_document(source_id: str):
    """Delete a document and all its chunks."""
    result = delete_document(source_id)
    return result    

# =========================
# File Upload Route
# =========================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(default="general"),
):
    """
    Upload a PDF, TXT, or DOCX file.
    Extracts text → chunks → embeds → stores in ChromaDB.
    """
    # Validate file type
    allowed_types = [".pdf", ".docx", ".txt"]
    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_types:
        return {
            "status": "error",
            "message": f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_types)}"
        }

    # Read file content
    file_bytes = await file.read()

    # Extract text
    try:
        text = extract_text(file.filename, file_bytes)
    except Exception as e:
        return {"status": "error", "message": f"Failed to extract text: {str(e)}"}

    if not text.strip():
        return {"status": "error", "message": "No text could be extracted from the file."}

    # Generate source_id from filename
    source_id = file.filename.rsplit(".", 1)[0].replace(" ", "-").lower()

    # Ingest into ChromaDB
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
    """Create a new chat session."""
    return create_session()


@app.get("/sessions")
async def get_sessions():
    """List all chat sessions."""
    sessions = list_sessions()
    return {"sessions": sessions}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get all messages for a session."""
    messages = get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a chat session."""
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
