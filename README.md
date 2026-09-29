# MediSense AI

**Full Academic Title:** MediSense AI — A Multimodal AI-Powered Health Assessment and Healthcare Navigation System

AI-powered multimodal health assessment system using NLP, OCR, deep learning, X-ray analysis, RAG, and LLM-based healthcare assistance.

> **Medical disclaimer:** MediSense AI is an academic decision-support / health-assessment project. It is **not** a replacement for a doctor, does **not** provide definitive diagnosis, and does **not** prescribe medicines.

---

## Features

- Multimodal assessment inputs: **Symptoms**, **Medical Report**, **X-Ray** (any combination)
- Structured evidence engine with missing / conflicting evidence handling
- Rule-based safety / triage pathways: **Emergency**, **Consultation**, **Mild** (LLM cannot override triage)
- Specialty mapping from controlled rules
- Context-aware AI Assistant (RAG + LLM with Groq → Gemini → Ollama fallback)
- Healthcare navigation architecture for doctors, facilities, medical shops (legitimate sources only)
- In-app **demo** appointments (not real provider confirmations)
- Assessment history and PDF report generation
- JWT authentication and ownership-based authorization

---

## Architecture

```
USER → React (Vite) → FastAPI → Input Processing
  ├── Symptoms → NLP
  ├── Medical Report → PDF/OCR → NLP
  └── X-Ray → OpenCV → PyTorch/ResNet (+ Grad-CAM)
→ Structured Evidence → Missing/Contradiction Check
→ Safety/Triage Rules Engine → EMERGENCY | CONSULTATION | MILD
→ Specialty Mapping → Final Outcome → Healthcare Navigation

AI Assistant: Question → Context + RAG → LLM → Grounding/Safety → Answer
```

---

## Technology Stack

| Layer | Stack |
|-------|--------|
| Frontend | React, Vite, React Router, CSS |
| Backend | Python, FastAPI, Pydantic, SQLAlchemy |
| Database | PostgreSQL |
| Auth | JWT + secure password hashing |
| AI/ML | PyTorch, OpenCV, Grad-CAM, Transformers/spaCy, OCR |
| RAG/LLM | Vector retrieval; Groq / Gemini / Ollama |
| PDF | ReportLab |
| Deploy | Docker Compose (optional) |

---

## Folder Structure

```
MediSense-AI/
├── frontend/          # React + Vite SPA
├── backend/           # FastAPI application
├── docs/              # Documentation
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## Prerequisites

- Python 3.11+ (3.12 recommended)
- Node.js 18+
- PostgreSQL 14+
- (Optional) Docker Desktop for compose-based Postgres / full stack

---

## Environment Variables

```bash
cp .env.example .env
cp .env.example backend/.env
# Frontend uses VITE_API_BASE_URL — copy into frontend/.env if desired
```

Set at minimum:

- `DATABASE_URL`
- `JWT_SECRET_KEY`
- `CORS_ORIGINS`
- `VITE_API_BASE_URL`

LLM keys (`GROQ_API_KEY`, `GEMINI_API_KEY`) are optional until the assistant stage.

---

## Database Setup

1. Create a PostgreSQL role and database:

```sql
CREATE USER medisense WITH PASSWORD 'your_secure_password';
CREATE DATABASE medisense_ai OWNER medisense;
GRANT ALL PRIVILEGES ON DATABASE medisense_ai TO medisense;
```

2. Set `DATABASE_URL` in `backend/.env`:

```
DATABASE_URL=postgresql+psycopg2://medisense:your_secure_password@localhost:5432/medisense_ai
```

3. Tables are created on API startup via SQLAlchemy (`Base.metadata.create_all`). Alembic migrations can be added later under `backend/app/database/migrations/`.

**Docker Postgres (if Docker is installed):**

```bash
docker compose up -d postgres
```

---

## Backend Setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy ..\.env.example .env   # then edit DATABASE_URL and JWT_SECRET_KEY

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Frontend Setup

```bash
cd frontend
npm install
# Create frontend/.env with:
# VITE_API_BASE_URL=http://localhost:8000/api
npm run dev
```

App: [http://localhost:5173](http://localhost:5173)

---

## Running Locally

1. Start PostgreSQL and ensure `DATABASE_URL` is valid.
2. Start backend on port **8000**.
3. Start frontend on port **5173**.
4. Open the app, register a user, and sign in.

---

## AI Model Setup

X-ray models live under `backend/models/xray/`. Place trained checkpoints there and set paths in env (e.g. `XRAY_CHEST_CHECKPOINT`).

If no checkpoint is configured, the API returns a clear **unavailable** status — it does **not** invent findings.

---

## LLM Configuration

```
LLM_PROVIDER=groq
LLM_FALLBACK=gemini
LLM_OFFLINE=ollama
GROQ_API_KEY=
GEMINI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
```

Only one provider answers a request. Failures fall through to a safe unavailable message.

---

## Testing

```bash
cd backend
pytest
```

---

## Security

- Passwords hashed (bcrypt / passlib)
- JWT-protected routes
- File type/size validation; private upload storage; ownership checks
- No API keys in source code
- CORS limited to configured origins

---

## Limitations

- Academic / decision-support prototype — not clinically validated for deployment
- Triage rules are rule-based for safety demonstration; not claimed as clinically validated
- Provider / facility / shop lists require legitimate authorized data sources; otherwise the UI shows unavailable/empty states
- Appointments are **demo** records unless a real booking integration is added
- X-ray interpretation only for regions with configured trained models

---

## Medical Disclaimer

This software produces AI-assisted assessment summaries for educational purposes. It is **not** a substitute for professional medical evaluation, diagnosis, or treatment. Seek emergency care immediately for urgent symptoms.
