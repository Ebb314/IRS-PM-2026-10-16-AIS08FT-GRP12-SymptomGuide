"""Export the disease -> symptom frequency table for REASONING_MODE=recall773.

Input: the dataset the 773 model was trained on (teammate bundle, PROJECT_OVERVIEW.md section 2):
  Disease-Symptom Dataset/Final_Augmented_dataset_Diseases_and_Symptoms.csv
  (public data card: https://data.mendeley.com/datasets/2cxccsxydc/1)
  One row per record: a disease label column plus 377 symptom columns of 0/1.

Output: backend/app/data/recall773/disease_symptom_frequency.csv with columns disease,symptom,frequency,records
  frequency = share of that disease's distinct records with the symptom = 1 (exact duplicate rows removed first,
  as in the teammate's training). Only frequency > 0 is written.
The chatbot picks this file up automatically in recall773 mode (or set DISEASE_SYMPTOM_TABLE to another path);
a symptom counts as belonging to a disease when frequency >= 0.2.

Usage: python scripts/export_disease_symptom_table.py "path/to/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "backend/app/recall_model_773/artifacts/disease_retrieval_773_tristate_v2/vocabulary.json"
OUT = ROOT / "backend/app/data/recall773/disease_symptom_frequency.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    symptoms, diseases = vocab["symptoms"], set(vocab["diseases"])

    with open(args.dataset, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Duplicate column names get pandas-style suffixes (regurgitation -> regurgitation.1), as in the model vocabulary.
        seen, names = defaultdict(int), []
        for h in header:
            h = h.strip()
            names.append(h if not seen[h] else f"{h}.{seen[h]}")
            seen[h] += 1
        label_col = next((i for i, h in enumerate(names) if h.lower() in ("diseases", "disease", "prognosis", "label")), 0)
        index = {h: i for i, h in enumerate(names)}
        missing = [s for s in symptoms if s not in index]
        if missing:
            raise SystemExit(f"{len(missing)} model symptoms are not columns of this file, e.g. {missing[:5]}. Wrong dataset?")
        cols = [index[s] for s in symptoms]
        unique, total = set(), 0
        for row in reader:
            total += 1
            unique.add((row[label_col].strip(), tuple(row[i].strip() in ("1", "1.0") for i in cols)))

    count, has = defaultdict(int), defaultdict(lambda: [0] * len(symptoms))
    for disease, vec in unique:
        count[disease] += 1
        acc = has[disease]
        for j, v in enumerate(vec):
            acc[j] += v
    unknown = sorted(set(count) - diseases)
    rows = [{"disease": d, "symptom": s, "frequency": round(has[d][j] / count[d], 4), "records": count[d]}
            for d in sorted(count) if d in diseases for j, s in enumerate(symptoms) if has[d][j]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["disease", "symptom", "frequency", "records"])
        w.writeheader()
        w.writerows(rows)
    covered = len({r["disease"] for r in rows})
    print(f"{total} rows, {len(unique)} after removing exact duplicates; {covered}/{len(diseases)} model diseases covered")
    if unknown:
        print(f"note: {len(unknown)} labels in the file are not model diseases, e.g. {unknown[:5]}")
    print(f"wrote {len(rows)} disease-symptom pairs to {args.out}")


if __name__ == "__main__":
    main()
