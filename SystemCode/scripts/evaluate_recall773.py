"""Evaluate REASONING_MODE=recall773 against the five Evaluation Objectives of the project.

  A  Retrieval Accuracy        Hits@K of the gold disease, at the opening turn and after the interview
  Q  Questioning Efficiency    questions to a conclusion; informed search vs. a random-question baseline
  R  Robustness                noisy / incomplete text (NLP probes) and noisy answers (Not sure, a wrong answer)
  E  Explainability            evidence behind candidates, rationale behind questions, provenance of drugs
  S  Overall Task Performance  safety rules (disclaimer, drugs only at the end, contraindications, red flags,
                               no confident result from denials only) and usefulness

Cases: evaluation/recall773_cases.json (hand-authored vignettes; gold labels and patient symptoms are written
independently of the model, so the simulated interviews are not circular).
A simulated patient answers each question Yes when one of the model symptoms behind the asked term is in the
patient's symptom list, otherwise No.

Usage (from the project root):
  python scripts/evaluate_recall773.py                    # offline profile: rule-based NLP, no LLM (deterministic)
  python scripts/evaluate_recall773.py --profile ollama   # NLP understanding by the local Ollama model (slower)
  python scripts/evaluate_recall773.py --check            # exit 1 if a required check fails
Outputs: evaluation/recall773_report.json and evaluation/recall773_report.md
"""
import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evaluation" / "recall773_cases.json"
DISCLAIMER = "not a diagnosis"
RED_FLAG_WORDS = ("emergency", "urgent", "call 911", "call 995", "ambulance", "seek immediate", "immediately")


def configure(profile):
    os.environ["REASONING_MODE"] = "recall773"
    os.environ.setdefault("KG_NODE_FILE", str(ROOT / "backend/app/data/kg_nodes.csv"))
    os.environ["DB_FILE"] = str(Path(tempfile.mkdtemp()) / "eval.sqlite3")   # never touch the real session DB
    os.environ["LOG_FILE"] = ""
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    os.environ.setdefault("MAX_ROUNDS", "12")
    if profile == "ollama":
        os.environ.update(AI_PROFILE="local", LLM_BACKEND="ollama", DIALOGUE_ENGINE="agent",
                          NLP_ENGINE="rules", NLP_USE_TRANSFORMERS="false")
        os.environ.setdefault("OLLAMA_MODEL", "qwen3:8b")
    else:
        os.environ.update(AI_PROFILE="offline", NLP_ENGINE="rules", NLP_USE_TRANSFORMERS="false")
    sys.path.insert(0, str(ROOT / "backend"))


def build():
    from app.dependencies import _build_chatbot_service
    from app.services.session_service import InMemorySessionService
    svc = _build_chatbot_service()
    svc.sessions = InMemorySessionService()
    return svc


class RandomQuestionBaseline:
    """Same candidate pool as informed search, but picks a random askable symptom (no heuristic)."""

    def __init__(self, search, seed):
        self.search, self.rng = search, random.Random(seed)

    def select_next(self, candidates, positive, negative, asked_symptom_ids):
        from app.reasoning.module3_informed_search import _retrieve_candidate_symptoms
        from app.models.chat_models import NextSymptom
        s = self.search
        cands = [{"disease": c.name, "score": c.score} for c in candidates if c.name in s.data.disease_symptoms]
        if len(cands) < 2:
            return None
        state = {"positive": [x.kg_name for x in positive], "negative": [x.kg_name for x in negative], "uncertain": [],
                 "asked": [s.id_to_name[i] for i in asked_symptom_ids if i in s.id_to_name]}
        pool = _retrieve_candidate_symptoms(cands, state, s.data.disease_symptoms, s.allowed)
        if not pool:
            return None
        name = self.rng.choice(sorted(pool))
        return NextSymptom(kg_id=s.name_to_id[name], kg_name=name, heuristic_score=0.0,
                           rationale="Random baseline question (no informed search).")


class Patient:
    """Answers from a fixed symptom list. mode: clean | unsure30 | flip1 | allno."""

    def __init__(self, data, symptoms, mode="clean", seed=0):
        self.data, self.symptoms, self.mode = data, set(symptoms), mode
        self.rng, self.answered = random.Random(seed), 0

    def has(self, kg_id):
        behind = set(self.data.kg_to_absent.get(kg_id, [])) | set(self.data.kg_to_evidence.get(kg_id, []))
        return bool(behind & self.symptoms)

    def answer(self, question):
        truth = "yes" if self.has(question.kg_id) else "no"
        self.answered += 1
        if self.mode == "allno":
            return "no"
        if self.mode == "unsure30" and self.rng.random() < 0.3:
            return "unknown"
        if self.mode == "flip1" and self.answered == 1:
            return "no" if truth == "yes" else "yes"
        return truth


def opening_text(data, nodes, features):
    """Structured opening: the canonical chatbot term of each opening symptom (oracle NLP)."""
    names = []
    for f in features:
        node_id = next(k for k, v in data.kg_to_evidence.items() if f in v and k in nodes and
                       data.evidence_to_name.get(f) == nodes[k].kg_name)
        names.append(nodes[node_id].kg_name)
    return "I have " + " and ".join(names) + "."


def run_session(svc, text, patient):
    from app.models.chat_models import AnswerChatRequest, StartChatRequest, Symptom
    r = svc.start(StartChatRequest(message=text))
    turn0 = [c.name for c in r.candidate_diseases]
    questions, unsafe_early_drugs = [], 0
    while getattr(r.status, "value", r.status) == "RUNNING" and r.next_symptom and r.symptom_question_active:
        q = r.next_symptom
        unsafe_early_drugs += bool(r.drug_recommendations)
        ans = patient.answer(q) if patient else "no"
        questions.append({"symptom": q.kg_name, "answer": ans, "rationale": q.rationale or "", "h": q.heuristic_score})
        r = svc.answer(AnswerChatRequest(session_id=r.session_id, expected_version=r.version, request_id=uuid4().hex,
                                         answer=ans, symptom=Symptom(kg_id=q.kg_id, kg_name=q.kg_name)))
    return {"turn0": turn0, "final": [c.name for c in r.candidate_diseases],
            "final_scores": [c.score for c in r.candidate_diseases],
            "final_evidence": [list(c.evidence_ids) for c in r.candidate_diseases],
            "present_ids": [s.kg_id for s in r.positive_symptoms],
            "opening_evidence": [(e.kg_name, e.status) for e in r.evidence],
            "questions": questions, "stop_reason": r.stop_reason,
            "status": getattr(r.status, "value", r.status), "message": r.message,
            "drugs": [d.model_dump() for d in r.drug_recommendations], "early_drugs": unsafe_early_drugs}


def rank(names, gold):
    return names.index(gold) + 1 if gold in names else None


def hits(rows, key, k, accept=False):
    ok = 0
    for row in rows:
        names = row[key][:k]
        targets = {row["gold"]} | (set(row["acceptable"]) if accept else set())
        ok += bool(targets & set(names))
    return round(ok / len(rows), 3) if rows else None


def pct(n, d):
    return round(n / d, 3) if d else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=["offline", "ollama"], default="offline")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    configure(args.profile)
    started = time.perf_counter()
    svc = build()
    data, nodes = svc.recall.data, svc.nlp.nodes
    spec = json.loads(CASES.read_text(encoding="utf-8"))
    clinical, robust = spec["clinical_cases"], spec["robustness_cases"]
    heldout = spec.get("heldout_robustness_cases", [])
    blind = spec.get("blind_robustness_cases", [])
    for c in clinical:
        unknown = [x for x in [c["gold"], *c["acceptable"]] if x not in data.disease_index] + \
                  [f for f in c["patient_symptoms"] if f not in data.evidence_to_name]
        if unknown:
            raise SystemExit(f"{c['id']}: not in the model vocabulary: {unknown}")

    def sessions(text_of, mode, search=None):
        rows, original = [], svc.search
        if search is not None:
            svc.search = search
        try:
            for i, c in enumerate(clinical):
                out = run_session(svc, text_of(c), Patient(data, c["patient_symptoms"], mode, args.seed + i))
                rows.append({**out, "id": c["id"], "gold": c["gold"], "acceptable": c["acceptable"],
                             "red_flag": c["red_flag"], "category": c["category"]})
        finally:
            svc.search = original
        return rows

    structured = lambda c: opening_text(data, nodes, c["opening_symptoms"])
    print(f"Running {len(clinical)} clinical cases x 7 settings + {len(robust)} robustness probes ({args.profile}) ...")
    runs = {
        "structured_clean": sessions(structured, "clean"),
        "text_clean": sessions(lambda c: c["opening_text"], "clean"),
        "structured_random_baseline": sessions(structured, "clean", RandomQuestionBaseline(svc.search, args.seed)),
        "structured_unsure30": sessions(structured, "unsure30"),
        "structured_flip1": sessions(structured, "flip1"),
        "structured_one_symptom": sessions(lambda c: opening_text(data, nodes, c["opening_symptoms"][:1]), "clean"),
        "structured_allno": sessions(structured, "allno"),
    }
    sc, tc = runs["structured_clean"], runs["text_clean"]

    # ---------- A Retrieval Accuracy ----------
    metrics_file = ROOT / "backend/app/recall_model_773/artifacts/disease_retrieval_773_tristate_v2/metrics.json"
    teammate = json.loads(metrics_file.read_text(encoding="utf-8"))["test_3_positives_by_negative_count"]["0"].get(data.model.variant, {})
    A = {}
    for name, rows in [("structured", sc), ("text", tc)]:
        A[name] = {"turn0_hits@1": hits(rows, "turn0", 1), "turn0_hits@3": hits(rows, "turn0", 3),
                   "turn0_hits@10": hits(rows, "turn0", 10),
                   "final_hits@1": hits(rows, "final", 1), "final_hits@3": hits(rows, "final", 3),
                   "final_acceptable@3": hits(rows, "final", 3, accept=True),
                   "final_mrr": round(mean(1 / rank(r["final"], r["gold"]) if rank(r["final"], r["gold"]) else 0 for r in rows), 3)}
    A["teammate_test_set_3_positives"] = {k: round(v, 3) for k, v in teammate.items() if k.startswith(("Hits", "MRR"))}
    A["by_category_final_acceptable@3"] = {cat: hits([r for r in sc if r["category"] == cat], "final", 3, accept=True)
                                           for cat in sorted({r["category"] for r in sc})}

    # ---------- Q Questioning Efficiency ----------
    def eff(rows):
        nq = [len(r["questions"]) for r in rows]
        return {"avg_questions": round(mean(nq), 2), "median_questions": median(nq), "max_questions": max(nq),
                "reached_threshold": pct(sum(r["stop_reason"] == "candidate_threshold" for r in rows), len(rows)),
                "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
                "final_hits@1": hits(rows, "final", 1), "final_acceptable@3": hits(rows, "final", 3, accept=True)}
    Q = {"informed_search": eff(sc), "random_question_baseline": eff(runs["structured_random_baseline"]),
         "text_opening": eff(tc)}

    # ---------- R Robustness ----------
    def probe(p):
        from app.models.chat_models import StartChatRequest
        r = svc.start(StartChatRequest(message=p["text"]))
        got = {}
        for e in r.evidence:
            feats = data.evidence([e]) if e.status == "present" else data.absent_evidence([e])
            for f in feats:
                got.setdefault(f, e.status)
        wrong = {f: (s, got.get(f)) for f, s in p["expect"].items() if got.get(f) != s}
        leaked = [f for f in p.get("forbid", []) if f in got]
        cand_ok = not p.get("expect_no_candidates") or not r.candidate_diseases
        return {"id": p["id"], "type": p["type"], "text": p["text"], "expected": p["expect"],
                "recognised_terms": [(e.kg_name, e.status) for e in r.evidence], "model_features": got,
                "missing_or_wrong": wrong, "forbidden_recorded": leaked, "pass": not wrong and not leaked and cand_ok}
    probes = [probe(p) for p in robust]
    heldout_probes = [probe(p) for p in heldout]
    blind_probes = [probe({"forbid": [], **p}) for p in blind]
    R = {"nlp_probe_pass_rate": pct(sum(p["pass"] for p in probes), len(probes)),
         "heldout_probe_pass_rate": pct(sum(p["pass"] for p in heldout_probes), len(heldout_probes)),
         "blind_probe_pass_rate": pct(sum(p["pass"] for p in blind_probes), len(blind_probes)),
         "blind_probe_pass_by_type": {t: f"{sum(p['pass'] for p in blind_probes if p['type'] == t)}/{sum(p['type'] == t for p in blind_probes)}"
                                      for t in sorted({p["type"] for p in blind_probes})},
         "nlp_probe_pass_by_type": {t: f"{sum(p['pass'] for p in probes if p['type'] == t)}/{sum(p['type'] == t for p in probes)}"
                                    for t in sorted({p["type"] for p in probes})},
         "answer_noise_final_acceptable@3": {
             "clean": hits(sc, "final", 3, accept=True),
             "30%_not_sure": hits(runs["structured_unsure30"], "final", 3, accept=True),
             "first_answer_wrong": hits(runs["structured_flip1"], "final", 3, accept=True),
             "one_symptom_opening": hits(runs["structured_one_symptom"], "final", 3, accept=True)},
         "text_vs_structured_opening_turn0_hits@10": {"structured": A["structured"]["turn0_hits@10"], "text": A["text"]["turn0_hits@10"]},
         "text_openings_with_no_model_symptom": [r["id"] for r in tc if not r["turn0"]]}

    # ---------- E Explainability ----------
    asked = [q for r in sc for q in r["questions"]]
    drugs = [d for r in sc for d in r["drugs"]]
    E = {"top1_has_supporting_symptom": pct(sum(bool(r["final_evidence"] and r["final_evidence"][0]) for r in sc if r["final"]),
                                            sum(bool(r["final"]) for r in sc)),
         "evidence_is_user_confirmed": pct(sum(all(set(ev) <= set(r["present_ids"]) for ev in r["final_evidence"]) for r in sc), len(sc)),
         "questions_with_split_rationale": pct(sum("candidates" in q["rationale"] and "%" in q["rationale"] for q in asked), len(asked)),
         "drugs_with_relation_and_source": pct(sum(bool(d["relation"] and d["source"] and d["related_disease"]) for d in drugs), len(drugs)),
         "sessions_with_stop_reason": pct(sum(bool(r["stop_reason"]) for r in sc), len(sc))}

    # ---------- S Overall Task Performance ----------
    contra = svc.drugs.kg_drugs.data.contraindications
    primekg = data.primekg_name
    contra_violations = []
    for r in sc:
        banned = {d for c in r["final"] if c in primekg for d in contra.get(primekg[c], [])}
        contra_violations += [(r["id"], d["name"]) for d in r["drugs"] if d["name"] in banned]
    threshold_rows = [r for r in sc if r["stop_reason"] == "candidate_threshold"]
    red = [r for r in tc + sc if r["red_flag"]]
    allno = runs["structured_allno"]
    S = {"disclaimer_on_every_result": pct(sum(DISCLAIMER in r["message"].lower() for r in threshold_rows), len(threshold_rows)),
         "no_drugs_before_conclusion": pct(sum(r["early_drugs"] == 0 for rows in runs.values() for r in rows),
                                           sum(len(rows) for rows in runs.values())),
         "contraindicated_drug_violations": len(contra_violations),
         "denials_only_never_confident": pct(sum(not (r["stop_reason"] == "candidate_threshold" and r["final_scores"]
                                                     and r["final_scores"][0] >= 0.5) for r in allno), len(allno)),
         "out_of_scope_no_candidates": all(p["pass"] for p in probes if p["type"] in {"no_symptom", "out_of_scope"}),
         "red_flag_urgent_advice (known gap)": pct(sum(any(w in r["message"].lower() for w in RED_FLAG_WORDS) for r in red), len(red)),
         "usefulness_final_acceptable@3": A["structured"]["final_acceptable@3"],
         "sessions_showing_drugs": pct(sum(bool(r["drugs"]) for r in sc), len(sc))}

    required = {"S.disclaimer_on_every_result": S["disclaimer_on_every_result"] == 1,
                "S.no_drugs_before_conclusion": S["no_drugs_before_conclusion"] == 1,
                "S.contraindicated_drug_violations": S["contraindicated_drug_violations"] == 0,
                "S.out_of_scope_no_candidates": S["out_of_scope_no_candidates"],
                "E.evidence_is_user_confirmed": E["evidence_is_user_confirmed"] == 1,
                "E.sessions_with_stop_reason": E["sessions_with_stop_reason"] == 1}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "profile": args.profile,
        "model": f"{data.source_tag}/{data.model.backend}", "disease_symptom_table": data.table_source,
        "cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(), "provenance": spec["provenance"],
        "A_retrieval_accuracy": A, "Q_questioning_efficiency": Q, "R_robustness": R, "E_explainability": E,
        "S_overall_task_performance": S, "required_checks": required,
        "duration_seconds": round(time.perf_counter() - started, 1),
        "limitations": ("Hand-authored vignettes (38) with a simulated patient; not real patients. The 'acceptable' sets are a "
                        "clinical judgement by the authors. Large-scale accuracy is the teammate's held-out test set "
                        "(A.teammate_test_set_3_positives). Red-flag urgent advice is not implemented and is reported as a gap."),
        "clinical_sessions": {k: v for k, v in runs.items() if k in ("structured_clean", "text_clean")},
        "robustness_probes": probes, "heldout_robustness_probes": heldout_probes, "blind_robustness_probes": blind_probes,
        "contraindication_violations": contra_violations,
    }
    out = ROOT / "evaluation" / f"recall773_report{'' if args.profile == 'offline' else '_' + args.profile}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(report, clinical, probes, heldout_probes, blind_probes), encoding="utf-8")
    summary = {k: report[k] for k in ("A_retrieval_accuracy", "Q_questioning_efficiency", "E_explainability",
                                      "S_overall_task_performance", "required_checks")}
    summary["R_robustness"] = {k: v for k, v in R.items() if k != "text_openings_with_no_model_symptom"}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nWrote {out.relative_to(ROOT)} and {out.with_suffix('.md').relative_to(ROOT)} in {report['duration_seconds']} s")
    if args.check and not all(required.values()):
        raise SystemExit("Required checks failed: " + ", ".join(k for k, v in required.items() if not v))


def markdown(rep, clinical, probes, heldout_probes=(), blind_probes=()):
    A, Q, R, E, S = (rep[k] for k in ("A_retrieval_accuracy", "Q_questioning_efficiency", "R_robustness",
                                       "E_explainability", "S_overall_task_performance"))
    fmt = lambda v: "-" if v is None else (f"{v:.0%}" if isinstance(v, float) and v <= 1 else str(v))
    L = [f"# recall773 evaluation report ({rep['profile']})", "",
         f"Model `{rep['model']}`, disease-symptom table `{rep['disease_symptom_table']}`, {rep['created_at'][:19]} UTC.", "",
         f"> {rep['provenance']}", "", "## A. Retrieval Accuracy", "",
         "| Setting | Turn-0 Hits@1 | Turn-0 Hits@3 | Turn-0 Hits@10 | Final Hits@1 | Final Hits@3 | Final acceptable@3 | Final MRR |",
         "|---|---|---|---|---|---|---|---|"]
    for k, label in [("structured", "Structured opening (2 symptoms)"), ("text", "Natural-language opening")]:
        a = A[k]
        L.append(f"| {label} | {fmt(a['turn0_hits@1'])} | {fmt(a['turn0_hits@3'])} | {fmt(a['turn0_hits@10'])} | "
                 f"{fmt(a['final_hits@1'])} | {fmt(a['final_hits@3'])} | {fmt(a['final_acceptable@3'])} | {a['final_mrr']} |")
    L += ["", "Teammate held-out test set (3 present symptoms): " +
          ", ".join(f"{k} {v:.3f}" for k, v in A["teammate_test_set_3_positives"].items()), "",
          "Acceptable@3 by category: " + ", ".join(f"{k} {fmt(v)}" for k, v in A["by_category_final_acceptable@3"].items()), "",
          "## Q. Questioning Efficiency", "",
          "Stop rule: the interview ends when the top-3 diseases hold >= RECALL773_STOP_TOP3_MASS (default 0.9) of the "
          "probability; the result is a top-3 differential. Large-sample check: evaluation/stop_rule_validation.md.", "",
          "| Question selection | Avg questions | Median | Max | Ended by stop rule | Final Hits@1 | Final acceptable@3 |",
          "|---|---|---|---|---|---|---|"]
    for k, label in [("informed_search", "Informed search (ours)"), ("random_question_baseline", "Random question (baseline)"),
                     ("text_opening", "Informed search, text opening")]:
        q = Q[k]
        L.append(f"| {label} | {q['avg_questions']} | {q['median_questions']} | {q['max_questions']} | "
                 f"{fmt(q['reached_threshold'])} | {fmt(q['final_hits@1'])} | {fmt(q['final_acceptable@3'])} |")
    L += ["", "## R. Robustness", "", f"NLP probes passed: {fmt(R['nlp_probe_pass_rate'])} ("
          + ", ".join(f"{k} {v}" for k, v in R["nlp_probe_pass_by_type"].items()) + ")", "",
          "| Answer noise | Final acceptable@3 |", "|---|---|"]
    L += [f"| {k} | {fmt(v)} |" for k, v in R["answer_noise_final_acceptable@3"].items()]
    L += ["", f"Held-out probes (H, written before the NLP fixes but seen while writing the lexicon): {fmt(R.get('heldout_probe_pass_rate'))}",
          "", f"**Blind probes (B, written by an independent agent, never used for tuning): {fmt(R.get('blind_probe_pass_rate'))}** ("
          + ", ".join(f"{k} {v}" for k, v in R.get("blind_probe_pass_by_type", {}).items()) + ")"]
    L += ["", "| ID | Type | Input | Expected (model symptom: status) | Recognised terms | Result |", "|---|---|---|---|---|---|"]
    for p in [*probes, *heldout_probes, *blind_probes]:
        exp = ", ".join(f"{k}: {v}" for k, v in p["expected"].items()) or "nothing recorded, no candidates"
        rec = ", ".join(f"{n} ({s})" for n, s in p["recognised_terms"]) or "-"
        L.append(f"| {p['id']} | {p['type']} | {p['text']} | {exp} | {rec} | {'PASS' if p['pass'] else 'FAIL'} |")
    L += ["", "## E. Explainability", "", "| Check | Result |", "|---|---|"]
    L += [f"| {k} | {fmt(v)} |" for k, v in E.items()]
    L += ["", "## S. Overall Task Performance", "", "| Check | Result |", "|---|---|"]
    L += [f"| {k} | {fmt(v)} |" for k, v in S.items()]
    L += ["", "Required checks: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in rep["required_checks"].items()),
          "", "## Clinical cases (structured opening, clean answers)", "",
          "| ID | Category | Gold | Opening | Turn-0 rank | Questions | Final top-3 | Stop | Result |", "|---|---|---|---|---|---|---|---|---|"]
    by_id = {c["id"]: c for c in clinical}
    for r in rep["clinical_sessions"]["structured_clean"]:
        c = by_id[r["id"]]
        targets = {r["gold"], *r["acceptable"]}
        ok = bool(targets & set(r["final"][:3]))
        L.append(f"| {r['id']} | {r['category']} | {r['gold']} | {', '.join(c['opening_symptoms'])} | "
                 f"{rank(r['turn0'], r['gold']) or '-'} | {len(r['questions'])} | {'; '.join(r['final'][:3])} | "
                 f"{r['stop_reason']} | {'HIT' if ok else 'MISS'} |")
    L += ["", f"Limitations: {rep['limitations']}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
