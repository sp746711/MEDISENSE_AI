# MediSense AI — Architecture Overview

See the root README for full product documentation.

## Layers

1. React frontend (`frontend/`) — JWT auth, assessment wizard, navigation UI
2. FastAPI backend (`backend/app/`) — orchestration, auth, AI adapters
3. PostgreSQL — application data
4. AI modules — NLP, OCR, X-ray (PyTorch), evidence, triage rules, RAG, LLM

## Safety

- Triage pathway is decided only by `triage_engine.py`
- LLM explains; it does not invent providers, findings, or pathways
- Missing model/provider data returns explicit unavailable states
