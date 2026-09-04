# Knowledge-Graph-Based Intelligent Symptom Triage Chatbot

An educational full-stack prototype that turns a natural-language symptom description into KG-linked evidence, asks high-value Yes/No follow-up questions, and returns mock disease and drug results. It is **not a medical device** and must not be used for diagnosis or treatment.

## Architecture

- `backend/`: FastAPI orchestration, NLP/KG mapping, in-memory sessions, and replaceable teammate-service interfaces.
- `frontend/`: Vue 3, TypeScript, Element Plus, Pinia, and Axios chat client.
- `nlp/`: the original standalone NLP work-in-progress retained for reference.

The backend always sends the complete accumulated positive and negative symptom state to the recall service. Teammate algorithms implement the abstract contracts in `backend/app/services/interfaces.py`; the frontend API does not need to change when mocks are replaced.

## Run the backend

Python 3.11 is recommended.

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

By default the NLP service uses a deterministic, offline fallback whose vocabulary is loaded directly from `backend/app/data/kg_nodes.csv`. To enable the requested biomedical NER and semantic entity mapper after dependencies/models are available:

```powershell
pip install -r requirements-nlp.txt
$env:NLP_USE_TRANSFORMERS = "true"
uvicorn app.main:app --reload
```

Replace the demo KG CSV with the teammate's export. Supported logical columns include `node_id`/`id`, `node_name`/`name`/`label`, and `node_type`/`type`.

## Run the frontend

Node.js 20+ is recommended.

```powershell
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`. Set `VITE_API_URL` when the backend is hosted elsewhere.

## Test

```powershell
cd backend
pytest
```

The test suite covers negation, KG-backed extraction, invalid follow-up answers, accumulated session evidence, and the complete mock multi-turn flow.

## API summary

- `POST /api/chat/start` — start from a free-text symptom description.
- `POST /api/chat/answer` — record Yes/No for the current KG symptom.
- `GET /api/chat/session/{session_id}` — inspect development session state.
- `GET /health` — service health check.
