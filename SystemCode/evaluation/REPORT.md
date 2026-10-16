# NLP evaluation report

Author-curated English functional regression and challenge cases; not a held-out clinical dataset. Gold labels describe text semantics only. Written during development.

Metric: exact (canonical concept, assertion status) tuples. Other-person and historical mentions are excluded.
These are development regression results, not out-of-sample generalization, clinical accuracy, or retrieval accuracy.

| Engine | Slice | Cases | Micro-F1 | Exact match |
|---|---|---:|---:|---:|
| keywords-v1 | functional | 46 | 0.3960 | 0.2826 |
| keywords-v1 | challenge | 8 | 0.1176 | 0.0000 |
| keywords-v1 | all | 54 | 0.3559 | 0.2407 |
| rules-v2 | functional | 46 | 1.0000 | 1.0000 |
| rules-v2 | challenge | 8 | 1.0000 | 1.0000 |
| rules-v2 | all | 54 | 1.0000 | 1.0000 |

## Remaining rules-v2 failures


## Scope

- No teammate model or external model is used by this evaluation.
- The transformer option is not measured here.
- Latencies in report.json are single-pass local NLP timing, not a load-test or end-to-end SLA.
- Candidate fixtures are used in integration tests only. Do not report their scores as diagnostic performance.
- Preserve this challenge slice when improving rules; collect a new independently annotated test set before claiming generalization.
