"""Package the teammate KG lookup tables into one file for REASONING_MODE=kg.

Inputs come from the GroupWork repository (run its build scripts first):
  Informed Search/data/nodes.csv                     project KG nodes (disease ids)
  Informed Search/data/disease_symptom_map.json      build_kg_maps.py (non-symptom terms removed)
  Informed Search/data/disease_lacks_symptom_map.json
  Drug_Retrival/data/primekg_drug_disease_subset.csv extract_step3_fixed.py
  Drug_Retrival/data/disease_contraindicated_drugs.json

Output: backend/app/data/kg_reasoning.json.gz

Only symptoms present in the chatbot vocabulary (KG_NODE_FILE) are kept, so the
search adapter can never return an out-of-vocabulary question.

Usage:
  python scripts/build_kg_reasoning_data.py --groupwork path/to/GroupWork
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = ROOT / "backend" / "app" / "data" / "kg_nodes.csv"
DEFAULT_OUT = ROOT / "backend" / "app" / "data" / "kg_reasoning.json.gz"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groupwork", required=True, type=Path, help="GroupWork folder")
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    search_dir = args.groupwork / "Informed Search" / "data"
    drug_dir = args.groupwork / "Drug_Retrival" / "data"

    with open(args.vocab, encoding="utf-8", newline="") as f:
        vocab = {row["node_name"] for row in csv.DictReader(f)}

    disease_ids = {}
    with open(search_dir / "nodes.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["node_type"] == "disease":
                disease_ids[row["name"]] = f"PRIMEKG:{row['node_index']}"

    def keep_vocab(mapping):
        out, dropped = {}, set()
        for disease, symptoms in mapping.items():
            kept = [s for s in symptoms if s in vocab]
            dropped.update(s for s in symptoms if s not in vocab)
            if kept:
                out[disease] = kept
        return out, dropped

    disease_symptoms, dropped = keep_vocab(read_json(search_dir / "disease_symptom_map.json"))
    disease_lacks, _ = keep_vocab(read_json(search_dir / "disease_lacks_symptom_map.json"))

    disease_drugs = defaultdict(list)
    with open(drug_dir / "primekg_drug_disease_subset.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            disease_drugs[row["disease_name"]].append([row["drug_name"], row["relation"]])
    contraindications = read_json(drug_dir / "disease_contraindicated_drugs.json")

    missing = [d for d in disease_symptoms if d not in disease_ids]
    if missing:
        raise SystemExit(f"{len(missing)} diseases have no node id, e.g. {missing[:3]}")

    used = set(disease_symptoms) | set(disease_drugs)
    payload = {
        "version": 1,
        "source": "PrimeKG project KG (GroupWork build_kg_maps.py + extract_step3_fixed.py)",
        "disease_ids": {d: disease_ids[d] for d in sorted(used) if d in disease_ids},
        "disease_symptoms": disease_symptoms,
        "disease_lacks": disease_lacks,
        "disease_drugs": {d: sorted(v, key=lambda x: (x[1] != "indication", x[0])) for d, v in disease_drugs.items()},
        "contraindications": contraindications,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    print(f"diseases with symptoms : {len(disease_symptoms):,}")
    print(f"disease-symptom pairs  : {sum(map(len, disease_symptoms.values())):,}")
    print(f"symptoms outside vocab : {len(dropped)} {sorted(dropped)[:5]}")
    print(f"diseases with drugs    : {len(disease_drugs):,}")
    print(f"written                : {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
