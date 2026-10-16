"""Validate the recall773 stop rule (top-3 probability mass) on held-out records of the training dataset.

1. Rebuilds the teammate's split exactly (deduplicate, group identical symptom vectors, seed 42, 80/10/10,
   train-coverage repair) and checks it against the bundle audit (dataset sha256, split sizes).
2. Samples N records from the TEST split (never seen in training). Each record is a simulated patient:
   the interview opens with 2 random present symptoms; questions are answered Yes when the asked symptom is 1 in
   the record, otherwise No. Gold = the record's disease label.
3. Records each interview for 12 rounds without stopping (10 candidates every turn), then replays stop rules on
   the recorded trajectories: the current rule and top-3 mass >= each threshold. The same is done for patients who
   answer No to every question.
4. Runs the implemented rule end to end (RECALL773_STOP_TOP3_MASS, default 0.8) on the same records.

Usage: python scripts/validate_stop_rule.py "path/to/Final_Augmented_dataset_Diseases_and_Symptoms.csv" [--n 500] [--check]
  --check fails (exit 1) when the implemented rule drops below the regression floors in FLOORS.
Outputs: evaluation/stop_rule_validation.json and .md
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "backend/app/recall_model_773/artifacts/disease_retrieval_773_tristate_v2/audit.json"
THRESHOLDS = (0.6, 0.7, 0.75, 0.8, 0.85, 0.9)
# Regression floors for the implemented rule (0.9 measured 2026-10-15: Hits@3 85%, 5.0 questions, 99% when fired).
FLOORS = {"hits@3": (">=", 0.80), "avg_questions": ("<=", 6.0), "hits@3_when_rule_fired": (">=", 0.95)}


def teammate_split(path):
    import pandas as pd
    raw = pd.read_csv(path)
    features = raw.columns[1:].tolist()
    unique = raw.drop_duplicates().reset_index(drop=True)
    diseases = sorted(unique.diseases.unique())
    x = unique[features].to_numpy(dtype=np.float32)
    y = unique.diseases.map({d: i for i, d in enumerate(diseases)}).to_numpy(dtype=np.int64)
    hashes = pd.util.hash_pandas_object(unique[features], index=False).to_numpy()
    groups, group_keys = pd.factorize(hashes, sort=True)
    rng = np.random.default_rng(42)
    draws = rng.random(len(group_keys))
    assignments = np.where(draws < 0.8, 0, np.where(draws < 0.9, 1, 2))
    sizes = np.bincount(groups)
    for label in rng.permutation(len(diseases)):
        candidates = np.unique(groups[y == label])
        if not (assignments[candidates] == 0).any():
            assignments[int(candidates[np.argmin(sizes[candidates])])] = 0
    split = {name: np.flatnonzero(assignments[groups] == i) for i, name in enumerate(("train", "val", "test"))}
    return x, y, features, diseases, split, len(raw)


def configure():
    os.environ.update(REASONING_MODE="recall773", AI_PROFILE="offline", NLP_ENGINE="rules", LOG_FILE="",
                      LOG_LEVEL="WARNING", DB_FILE=str(Path(tempfile.mkdtemp()) / "v.sqlite3"), MAX_ROUNDS="12")
    os.environ.setdefault("KG_NODE_FILE", str(ROOT / "backend/app/data/kg_nodes.csv"))
    sys.path.insert(0, str(ROOT / "backend"))
    sys.path.insert(0, str(ROOT / "scripts"))


def interview(svc, ev, data, nodes, opening, positives, mode):
    from app.models.chat_models import AnswerChatRequest, StartChatRequest, Symptom
    patient = ev.Patient(data, positives, mode, 0)
    r = svc.start(StartChatRequest(message=ev.opening_text(data, nodes, opening)))
    traj = [[(c.name, c.score) for c in r.candidate_diseases]]
    while getattr(r.status, "value", r.status) == "RUNNING" and r.next_symptom and r.symptom_question_active:
        q = r.next_symptom
        r = svc.answer(AnswerChatRequest(session_id=r.session_id, expected_version=r.version, request_id=uuid4().hex,
                                         answer=patient.answer(q), symptom=Symptom(kg_id=q.kg_id, kg_name=q.kg_name)))
        traj.append([(c.name, c.score) for c in r.candidate_diseases])
    return traj, r


def replay(trajs, golds, rule):
    stops, h1, h3, fired, h3_fired, h3_not = [], 0, 0, 0, 0, 0
    for T, gold in zip(trajs, golds):
        t = next((i for i in range(len(T)) if rule(T[i])), None)
        hit3 = None
        if t is not None:
            fired += 1
        t = len(T) - 1 if t is None else t
        names = [n for n, _ in T[t]]
        hit3 = gold in names[:3]
        h1 += names[:1] == [gold]
        h3 += hit3
        if rule(T[t]):
            h3_fired += hit3
        else:
            h3_not += hit3
        stops.append(t)
    n = len(trajs)
    return {"stopped_by_rule": round(fired / n, 3), "avg_questions": round(sum(stops) / n, 2),
            "hits@1": round(h1 / n, 3), "hits@3": round(h3 / n, 3),
            "hits@3_when_rule_fired": round(h3_fired / fired, 3) if fired else None,
            "hits@3_when_max_rounds": round(h3_not / (n - fired), 3) if n - fired else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if sha != audit["dataset_sha256"]:
        raise SystemExit(f"Dataset sha256 {sha[:12]}... differs from the one the model was trained on "
                         f"({audit['dataset_sha256'][:12]}...).")
    print("Rebuilding the teammate's train/val/test split ...")
    x, y, features, diseases, split, raw_rows = teammate_split(args.dataset)
    sizes = {k: int(len(v)) for k, v in split.items()}
    if sizes != audit["split_sizes"]:
        raise SystemExit(f"Split {sizes} does not match the bundle audit {audit['split_sizes']}")
    print(f"Split matches the audit: {sizes}")

    configure()
    import evaluate_recall773 as ev
    from app.dependencies import _build_chatbot_service
    from app.services.session_service import InMemorySessionService
    svc = _build_chatbot_service()
    svc.sessions = InMemorySessionService()
    data, nodes = svc.recall.data, svc.nlp.nodes
    if list(features) != data.model.features:
        raise SystemExit("Dataset columns differ from the model symptom order")

    rng = np.random.default_rng(args.seed)
    rows = rng.choice(split["test"], size=min(args.n, len(split["test"])), replace=False)
    cases = []
    for i in rows:
        pos = [features[j] for j in np.flatnonzero(x[i])]
        opening = list(rng.choice(pos, size=min(2, len(pos)), replace=False))
        cases.append({"row": int(i), "gold": diseases[y[i]], "positives": pos, "opening": opening})

    # Record full trajectories: no stop rule, 10 candidates every turn.
    rule_state = (svc.recall.stop_top3_mass, svc.recall.mass, svc.finish_threshold)
    svc.recall.stop_top3_mass, svc.recall.mass, svc.finish_threshold = None, 1.01, 0
    print(f"Recording {len(cases)} interviews x 2 (answers from the record / all No) ...")
    clean = [interview(svc, ev, data, nodes, c["opening"], c["positives"], "clean")[0] for c in cases]
    allno = [interview(svc, ev, data, nodes, c["opening"], c["positives"], "allno")[0] for c in cases]
    svc.recall.stop_top3_mass, svc.recall.mass, svc.finish_threshold = rule_state
    golds = [c["gold"] for c in cases]

    top = lambda T, k: sum(s for _, s in T[:k])
    rules = {"current: top1>=0.80 and top1+top2>=0.95": lambda T: bool(T) and T[0][1] >= 0.8 and top(T, 2) >= 0.95}
    rules.update({f"top-3 mass >= {t:.2f}": (lambda T, t=t: top(T, 3) >= t) for t in THRESHOLDS})
    table = {}
    for name, rule in rules.items():
        res = replay(clean, golds, rule)
        res["all_no_stopped_by_rule"] = round(sum(any(rule(T[t]) for t in range(len(T))) for T in allno) / len(allno), 3)
        table[name] = res
    turn0 = {f"hits@{k}": round(sum(g in [n for n, _ in T[0][:k]] for T, g in zip(clean, golds)) / len(cases), 3)
             for k in (1, 3, 10)}

    # End-to-end check of the implemented rule.
    e2e = []
    for c in cases:
        _, r = interview(svc, ev, data, nodes, c["opening"], c["positives"], "clean")
        names = [d.name for d in r.candidate_diseases]
        e2e.append((r.stop_reason, c["gold"] in names[:3], names[:1] == [c["gold"]], r.round))
    implemented = {"top3_mass_threshold": svc.recall.stop_top3_mass,
                   "stopped_by_rule": round(sum(s == "candidate_threshold" for s, *_ in e2e) / len(e2e), 3),
                   "avg_questions": round(sum(q for *_, q in e2e) / len(e2e), 2),
                   "hits@1": round(sum(h for _, _, h, _ in e2e) / len(e2e), 3),
                   "hits@3": round(sum(h for _, h, _, _ in e2e) / len(e2e), 3),
                   "stop_reasons": {k: sum(str(s) == k for s, *_ in e2e) for k in sorted({str(s) for s, *_ in e2e})}}
    fired = [h for s, h, _, _ in e2e if s == "candidate_threshold"]
    implemented["hits@3_when_rule_fired"] = round(sum(fired) / len(fired), 3) if fired else None
    checks = {k: (implemented[k] >= v if op == ">=" else implemented[k] <= v) if implemented[k] is not None else False
              for k, (op, v) in FLOORS.items()}

    report = {"dataset_sha256": sha, "raw_rows": raw_rows, "split_sizes": sizes, "test_records_sampled": len(cases),
              "sample_seed": args.seed, "diseases_in_sample": len(set(golds)),
              "avg_present_symptoms_per_record": round(float(np.mean([len(c["positives"]) for c in cases])), 2),
              "model": f"{data.source_tag}/{data.model.backend}", "disease_symptom_table": data.table_source,
              "opening_turn": turn0, "stop_rules_replayed": table, "implemented_rule_end_to_end": implemented,
              "regression_floors": {k: f"{op} {v}" for k, (op, v) in FLOORS.items()}, "regression_checks": checks,
              "duration_seconds": round(time.perf_counter() - started, 1),
              "notes": ("Held-out TEST records of the synthetic Disease-Symptom Dataset; the simulated patient answers "
                        "from the record (1 = Yes, 0 = No, the teammate's synthetic-negative assumption). Gold is the "
                        "exact record label (no acceptable alternatives). In-distribution, so expect higher accuracy "
                        "than the hand-written clinical vignettes.")}
    out = ROOT / "evaluation" / "stop_rule_validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# Stop rule validation ({len(cases)} held-out test records)", "",
             f"Split rebuilt and verified against the bundle audit {sizes}; dataset sha256 `{sha[:16]}...`. "
             f"{len(set(golds))} diseases in the sample, {report['avg_present_symptoms_per_record']} present symptoms per record.", "",
             f"Opening turn (2 symptoms): Hits@1 {turn0['hits@1']:.0%}, Hits@3 {turn0['hits@3']:.0%}, Hits@10 {turn0['hits@10']:.0%}", "",
             "| Rule | Stopped by rule | Avg questions | Hits@1 | Hits@3 | Hits@3 when rule fired | Hits@3 at 12 rounds | All-No stopped by rule |",
             "|---|---|---|---|---|---|---|---|"]
    pf = lambda v: "-" if v is None else f"{v:.0%}"
    for name, r in table.items():
        lines.append(f"| {name} | {pf(r['stopped_by_rule'])} | {r['avg_questions']} | {pf(r['hits@1'])} | {pf(r['hits@3'])} | "
                     f"{pf(r['hits@3_when_rule_fired'])} | {pf(r['hits@3_when_max_rounds'])} | {pf(r['all_no_stopped_by_rule'])} |")
    lines += ["", f"Implemented rule end to end (top-3 mass >= {implemented['top3_mass_threshold']}): stopped by rule "
              f"{pf(implemented['stopped_by_rule'])}, {implemented['avg_questions']} questions on average, Hits@1 "
              f"{pf(implemented['hits@1'])}, Hits@3 {pf(implemented['hits@3'])} ({pf(implemented['hits@3_when_rule_fired'])} when the "
              f"rule fired); stop reasons {implemented['stop_reasons']}.", "",
              "Regression floors: " + ", ".join(f"{k} {FLOORS[k][0]} {FLOORS[k][1]}: {'PASS' if ok else 'FAIL'}" for k, ok in checks.items()),
              "", report["notes"], ""]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {out.relative_to(ROOT)} (.md) in {report['duration_seconds']} s")
    if args.check and not all(checks.values()):
        raise SystemExit("Stop-rule regression floors failed: " + ", ".join(k for k, ok in checks.items() if not ok))


if __name__ == "__main__":
    main()
