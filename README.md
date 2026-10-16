# Symptom Guide — KG-Enhanced Intelligent Diagnostic Chatbot

Symptom Guide is a symptom-checking chatbot. A patient describes how they feel, the system asks targeted follow-up questions, and it ends with the three most likely conditions, the symptoms that support each one, and related medications from a medical knowledge graph.

> **Not a medical device.** Results are information to discuss with a doctor, not a diagnosis.

## How it works

1. **Understanding.** Rules, a lay-term lexicon and typo tolerance find symptoms in the user's message. An optional local LLM (Qwen3-8B via Ollama) handles freer phrasing. Each symptom is recorded as present, absent or unknown.
2. **Disease recall.** A tri-state two-tower model ranks 773 diseases from the present and absent symptoms. It runs with NumPy, so PyTorch is not needed.
3. **Informed search.** The next question is the symptom that splits the current candidates most evenly.
4. **Stopping rule.** The interview ends when the top three diseases hold at least 90% of the probability, or after 12 questions.
5. **Drug retrieval.** Drug relations come from PrimeKG. Drugs contraindicated for any listed condition are removed.

## Repository layout

```
ProjectReport/          Group report
Video/                  Demo video
SystemCode/
  backend/              FastAPI app, reasoning services, models and data, tests
  frontend/             Vue 3 + TypeScript user interface
  scripts/              Start-up, evaluation and data-preparation scripts
  evaluation/           Test cases and evaluation reports
```

## Requirements

- Python 3.11 or later
- Node.js 18 or later (only needed to build the frontend)
- Optional: [Ollama](https://ollama.com) with the `qwen3:8b` model, for better understanding of everyday language

## Quick start (Windows PowerShell)

Run all commands from the `SystemCode` folder.

```powershell
cd SystemCode

# 1. Install backend dependencies
pip install -r backend/requirements.txt numpy

# 2. Build the frontend (first time only)
cd frontend
npm ci
npm run build
cd ..

# 3a. Start with the local LLM (Ollama must be running and have qwen3:8b)
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 -Profile ollama -Reasoning recall773

# 3b. Or start without any LLM
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 -Profile offline -Reasoning recall773
```

Open http://127.0.0.1:8000. You can check the setup at http://127.0.0.1:8000/api/health.

To install the model for 3a, run `ollama pull qwen3:8b` (about 5 GB). Without a model, use 3b. Disease ranking, questioning, stopping and drug retrieval work the same way in both. Only free-text understanding and reply wording are simpler in 3b.

## Reasoning modes

Select a mode with `-Reasoning`.

| Mode | Description |
|---|---|
| `recall773` | **Main mode.** 773-disease recall model, informed search, PrimeKG drugs |
| `ddxplus` | Recall model trained on DDXPlus (49 diseases), kept for comparison |
| `kg` | Symptom-overlap ranking over PrimeKG, kept for comparison |
| `random` | Seeded test fixtures for integration testing |

Useful settings (environment variables):

- `RECALL773_STOP_TOP3_MASS`: stopping threshold (default `0.9`)
- `MAX_ROUNDS`: maximum number of follow-up questions (default `12`)
- `LOG_LEVEL`: set to `DEBUG` for detailed logs

Logs are written to the terminal and to `backend/logs/chatbot.log`.

## Tests and evaluation

```powershell
python -m pytest backend/tests -q
python scripts/evaluate_recall773.py --profile offline --check
python scripts/validate_stop_rule.py "<path to dataset CSV>" --check
```

The two evaluation scripts write their results to `evaluation/`. The dataset CSV is the [Disease and symptoms dataset 2023](https://data.mendeley.com/datasets/2cxccsxydc/1), which is not included in the repository.

## Data and models

- **Knowledge graph:** PrimeKG (Chandak, Huang & Zitnik, *Scientific Data* 2023). Phenotype vocabulary from the Human Phenotype Ontology.
- **Recall model training data:** Disease and symptoms dataset 2023 (Mendeley Data). The data are synthetic.
- **LLM:** Qwen3-8B, served locally by Ollama. Conversation text never leaves the user's computer.

## Team (Group 12)

| Name | Student ID |
|---|---|
| Yang Ruihan | A0356537U |
| Liu Hengyu | A0353701M |
| Yang Peixin | A0357227Y |
