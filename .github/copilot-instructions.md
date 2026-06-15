# Copilot instructions for HVAC training workspace

## Scope and architecture
- This workspace is split into two apps: `hvac-backend/` (FastAPI) and `hvac-chat-frontend/` (Vite + React + TypeScript).
- Frontend sends chat text to backend `POST /chat`; backend returns `{ "reply": string }`.
- Keep this contract stable unless both sides are updated together.
- Current backend behavior is intentionally a placeholder (keyword matching in `app.py`) and is designed to be swapped later with LLM/tool calls.

## Backend patterns (`hvac-backend`)
- Main app lives in a single file: `app.py`.
- Request/response models use Pydantic classes `ChatRequest` and `ChatResponse`.
- API routes:
  - `GET /` health/info message
  - `POST /chat` returns `ChatResponse`
- CORS is explicitly limited to Vite dev origins (`http://localhost:5173`, `http://127.0.0.1:5173`).
  - If frontend port/origin changes, update CORS list.
- Keep response shape explicit via `response_model=ChatResponse`.

## Frontend patterns (`hvac-chat-frontend`)
- `src/api.ts` is the only backend gateway; keep HTTP calls centralized there.
- Backend URL is hardcoded as `http://localhost:8000` in `src/api.ts`.
  - If backend host/port changes, update this constant (or refactor to env var consistently).
- `src/Chat.tsx` owns chat UI state (`messages`, `input`, `loading`) and submit flow.
- Message schema in UI is local and minimal:
  - `{ role: "user" | "assistant", text: string }`
- UI styling is currently inline in `Chat.tsx` plus base styles in `src/index.css`; avoid mixing in a third styling pattern unless doing a deliberate refactor.
- `sendChatMessage()` returns a fallback string on errors instead of throwing; UI relies on that behavior.

## Developer workflows
- Backend setup (Python venv): install from `requirements.txt`, then run FastAPI with Uvicorn.
- Typical backend run: `uvicorn app:app --reload --port 8000` from `hvac-backend/`.
- Frontend setup: `npm install` then `npm run dev` from `hvac-chat-frontend/` (Vite port 5173).
- Frontend production build: `npm run build`; preview via `npm run preview`.
- No automated tests are configured in this training project; validate by running both apps and exercising `/chat` through the UI.

## Change guidance for agents
- For API changes, update both:
  - backend models/route handler in `app.py`
  - frontend API parsing in `src/api.ts` and render logic in `src/Chat.tsx`
- Preserve beginner-friendly readability: small files, explicit types, minimal abstraction.
- Prefer incremental changes over framework-heavy restructuring unless explicitly requested.
