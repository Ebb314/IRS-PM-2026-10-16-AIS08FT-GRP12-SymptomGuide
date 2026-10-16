"""Run from any directory: python scripts/evaluate.py. No network or teammate modules."""
import sys, json, statistics, platform, argparse
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))
from app.services.nlp_service import NLPService

def metric(rows):
    tp=fp=fn=exact=0
    for row in rows:
        gold=set(row["expected"].items()); pred=set(row["predicted"].items())
        tp+=len(gold&pred); fp+=len(pred-gold); fn+=len(gold-pred); exact+=gold==pred
    return {"cases":len(rows),"tp":tp,"fp":fp,"fn":fn,
        "micro_f1":round(2*tp/(2*tp+fp+fn),4) if (2*tp+fp+fn) else 1.0,
        "exact_match":round(exact/len(rows),4) if rows else 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if any rules regression case fails")
    args = parser.parse_args()
    source=json.loads((ROOT/"evaluation/cases.json").read_text(encoding="utf-8-sig"))
    nlp=NLPService(ROOT/"backend/app/data/demo_symptoms.csv",use_models=False)
    results={}
    for engine in ("keywords-v1","rules-v2"):
        rows=[]; times=[]
        for case in source["cases"]:
            t=perf_counter()
            items=nlp.baseline(case["text"]) if engine=="keywords-v1" else nlp.analyze(case["text"]).evidence
            times.append((perf_counter()-t)*1000)
            rows.append({**case,"predicted":{e.kg_name:e.status for e in items}})
        results[engine]={"all":metric(rows),
            "functional":metric([r for r in rows if r["category"]!="challenge"]),
            "challenge":metric([r for r in rows if r["category"]=="challenge"]),
            "by_category":{k:metric([r for r in rows if r["category"]==k]) for k in sorted({r["category"] for r in rows})},
            "single_pass_latency_ms":{"median":round(statistics.median(times),3),"max":round(max(times),3)},
            "failures":[r for r in rows if r["expected"]!=r["predicted"]]}
    report={"generated_at":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
        "dataset_note":source["description"],"results":results}
    (ROOT/"evaluation/report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# NLP evaluation report","",source["description"],"",
        "Metric: exact (canonical concept, assertion status) tuples. Other-person and historical mentions are excluded.",
        "These are development regression results, not out-of-sample generalization, clinical accuracy, or retrieval accuracy.","",
        "| Engine | Slice | Cases | Micro-F1 | Exact match |","|---|---|---:|---:|---:|"]
    for engine,result in results.items():
        for group in ("functional","challenge","all"):
            m=result[group]
            lines.append(f"| {engine} | {group} | {m['cases']} | {m['micro_f1']:.4f} | {m['exact_match']:.4f} |")
    lines+=["","## Remaining rules-v2 failures",""]
    for row in results["rules-v2"]["failures"]:
        lines += [f"- {row['id']}: {row['text']}",f"  - Expected: {row['expected']}",f"  - Predicted: {row['predicted']}"]
    lines+=["","## Scope","",
        "- No teammate model or external model is used by this evaluation.",
        "- The transformer option is not measured here.",
        "- Latencies in report.json are single-pass local NLP timing, not a load-test or end-to-end SLA.",
        "- Candidate fixtures are used in integration tests only. Do not report their scores as diagnostic performance.",
        "- Preserve this challenge slice when improving rules; collect a new independently annotated test set before claiming generalization."]
    (ROOT/"evaluation/REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:{s:v[s] for s in ("functional","challenge","all")} for k,v in results.items()},indent=2))
    if args.check and results["rules-v2"]["failures"]:
        raise SystemExit("Rules regression failed; inspect evaluation/REPORT.md")

if __name__=="__main__": main()


