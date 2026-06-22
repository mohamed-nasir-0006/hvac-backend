# memory.py
"""
Conversation memory using SQLite.
Stores chat history per session so users can continue conversations.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = "conversations.db"


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT DEFAULT 'New Chat',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            parsed_intent TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    conn.commit()
    conn.close()
    print("[Memory] SQLite database ready.")


def create_session() -> Dict:
    """Create a new chat session."""
    conn = get_db()
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    conn.execute(
        "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, "New Chat", now, now),
    )
    conn.commit()
    conn.close()

    return {"session_id": session_id, "title": "New Chat", "created_at": now}


def save_message(session_id: str, role: str, content: str, parsed_intent: Optional[Dict] = None):
    """Save a message to a session."""
    conn = get_db()
    now = datetime.now().isoformat()

    # Save message
    conn.execute(
        "INSERT INTO messages (session_id, role, content, parsed_intent, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, json.dumps(parsed_intent) if parsed_intent else None, now),
    )

    # Update session timestamp and title (use first user message as title)
    if role == "user":
        # Check if this is the first user message
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user'",
            (session_id,),
        ).fetchone()[0]

        if count == 1:  # Just inserted the first user message
            title = content[:50] + ("..." if len(content) > 50 else "")
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title, now, session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    conn.commit()
    conn.close()


def get_session_messages(session_id: str) -> List[Dict]:
    """Get all messages for a session."""
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, parsed_intent, created_at FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()

    messages = []
    for row in rows:
        msg = {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        if row["parsed_intent"]:
            msg["parsed_intent"] = json.loads(row["parsed_intent"])
        messages.append(msg)

    return messages


def get_session_history(session_id: str) -> List[Dict]:
    """Get messages in Ollama history format (role + content only)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()

    return [{"role": row["role"], "content": row["content"]} for row in rows]


def list_sessions(limit: int = 20) -> List[Dict]:
    """List recent chat sessions."""
    conn = get_db()
    rows = conn.execute(
        "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def delete_session(session_id: str) -> bool:
    """Delete a session and all its messages."""
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return True