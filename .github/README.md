# HVAC AI Backend

FastAPI backend for an AI-powered HVAC control assistant.

## Features
- 🤖 LLM-powered chat (Ollama + Llama3)
- 🎯 Structured intent extraction (NLP)
- 📚 RAG with ChromaDB (persistent vector store)
- 🔍 Knowledge base with semantic search

## Tech Stack
- FastAPI
- Ollama (llama3 + nomic-embed-text)
- ChromaDB
- Python 3.11+

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn httpx chromadb pydantic

# Start Ollama
ollama serve
ollama pull llama3
ollama pull nomic-embed-text

# Run server
uvicorn app:app --reload --port 8000