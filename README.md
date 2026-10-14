# Symptom Guide — Full-Catalog NLP & Dialogue Lab

GC1 module: NLP, chatbot orchestration, FastAPI backend and Vue frontend. The current local profile uses **Qwen3-4B-Instruct-2507 with NF4 quantization**, native multi-turn messages, open-ended goal descriptions and validated evidence proposals. It does not dispatch chat through a fixed intent enum. Concept retrieval covers **15,135 symptom nodes**; teammate reasoning remains replaceable through interfaces.

**Ollama models supported:** `run.ps1 -Profile ollama` uses a model served by local Ollama (default `qwen3:8b`) instead of the bundled Qwen3-4B; no PyTorch needed. See [Ollama本地模型接入](docs/Ollama本地模型接入.md).

**KG reasoning integrated (`REASONING_MODE=kg`, now the `run.ps1` default):** PrimeKG recall, the GroupWork informed search and contraindication-filtered drug relations run in-process behind the existing adapter interfaces. See [KG推理集成记录](docs/KG推理集成记录.md).

**v9 performance and editable symptom panel:** [响应速度与症状侧栏改造记录](docs/响应速度与症状侧栏改造记录.md).

**v8 architecture, manual setup, evaluation and limitations:** [NLP 对话架构重构记录](docs/NLP对话架构重构记录.md). Earlier change records below describe historical versions. Their model and intent-routing results do not certify the current engine.

The historical v8 [content review](evaluation/conversation-agent-human-review.md) records repeated guidance and unnecessary follow-ups. The v9 change record contains the latest pipeline comparison and remaining limitations. Automatic test success is not a general understanding accuracy score.

## Current change record

[Full-catalog integration, both interaction fixes, commands and test evidence (Chinese)](docs/全量症状与交互修复记录.md).

- Free-text answering hides Yes / No / Not sure, with a switch back to choices. New questions have their own answer mode.
- The knowledge button follows the latest recorded/answered symptom, rather than a hard-coded fever question. Topic state survives reload.
- Topics without retrieved passages now use local LLM general-knowledge explanations, with no fabricated citations. See [change record](docs/LLM知识问答修复记录.md).
- Random disease fixtures and random symptom selection support standalone integration. Fixed seeds make tests reproducible; `remote` remains the teammate integration mode.
- All nodes participate in concept retrieval and deterministic coverage tests. UI search renders at most 50 matching names at once.

## Run with the existing environment

```powershell
conda activate gc1
cd E:\Study\StudyAtNUS\GC1_Project
$env:AI_PROFILE='local'
$env:DIALOGUE_ENGINE='agent'
$env:LOCAL_MODEL_PATH='E:/Study/StudyAtNUS/GC1_Project/models/Qwen3-4B-Instruct-2507'
$env:LOCAL_MODEL_4BIT='true'
$env:NLP_ENGINE='rules'
$env:NLP_USE_TRANSFORMERS='false'
$env:REASONING_MODE='random'
$env:RANDOM_SEED='20260915'
$env:KG_NODE_FILE='E:/Study/StudyAtNUS/GC1_Project/backend/app/data/kg_nodes.csv'
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --no-access-log
```

Install `backend/requirements-agent.txt` and download the pinned model using `scripts/download_agent_model.py` before first use (already done on the delivery machine). Open http://127.0.0.1:8000. Initial model/index loading takes time. `/api/health` should show 15135 nodes, random mode, dialogue_engine=agent and the Qwen3 generator. Run one GPU service at a time. `AI_PROFILE=offline` uses rules and source extracts without model dependencies.

Default sessions use `backend/runtime/sessions-fullkg.sqlite3`, preserving the old DEMO database separately. If an old browser session is not found, choose Start new local view. No automatic medical-ID migration is attempted.

## Verify

```powershell
python -m pytest backend/tests -q
python scripts/evaluate_full_kg.py --check
# With the full local AI service running:
python scripts/evaluate_full_kg_llm.py --count 20 --seed 20260915
cd frontend
$env:TEST_BASE_URL='http://127.0.0.1:8000'
node test-agent.cjs
node review-agent.cjs
node browser-agent.cjs
```

`evaluate_full_kg_llm.py --count 0` requests real API/LLM execution for every CSV node and may take hours. The actual delivered exhaustive test is 75,675 deterministic cases; real LLM checks are separately reported seeded samples. Do not confuse these scopes.

- [Full coverage](evaluation/full_kg_report.json), [per-case JSONL](evaluation/full_kg_cases.jsonl)
- [Real LLM sample](evaluation/full_kg_llm_report.json)
- [Real full-catalog browser tests](evaluation/browser-fullkg-report.json)
- [Dependency setup / manual commands](docs/AI升级运行手册.md)
- [Teammate interface contract](docs/接口契约.md)
- [AI technical and career material](docs/AI技术详解与求职材料.md) — earlier small-set metrics retain their original scope

Legacy `mock`, `browser-smoke.cjs`, `browser-ai.cjs`, `evaluate.py` and `evaluate_ai.py` retain the explicit ten-node DEMO fixtures for historical regression. They are not the current full-catalog acceptance suite.

## Scope

The CSV supplies symptom labels, not 15,135 medical definitions or a disease catalog. Random diseases are synthetic SIM fixtures and scores are not probabilities. Real disease retrieval and informed search belong to teammates. No clinical accuracy, training/fine-tuning or production deployment is claimed. The project uses AI-assisted development; review the code and rerun the evidence before claiming independent mastery.


LLM intent routing and contextual dialogue: [implementation and tests](docs/LLM意图路由与对话升级.md).


Continuous conversation repair: [implementation and real-model results](docs/连续对话上下文修复记录.md).


