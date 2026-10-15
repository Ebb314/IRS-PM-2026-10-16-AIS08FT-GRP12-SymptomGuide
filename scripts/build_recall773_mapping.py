"""Build the chatbot <-> 773-disease recall model mapping tables (REASONING_MODE=recall773).

Inputs:  backend/app/recall_model_773/ (teammate bundle: vocabulary.json with the shared-KG catalog
         links, negative_support.json), backend/app/data/kg_nodes.csv (chatbot vocabulary),
         backend/app/data/kg_reasoning.json.gz (PrimeKG drug data), GroupWork nodes.csv
         (KG node_index -> name; the same KG version, sha256 6a4ce3a4...).
Outputs: backend/app/data/recall773/feature_map.csv        377 model symptoms -> chatbot nodes
         backend/app/data/recall773/extra_symptom_nodes.csv model-only symptom nodes
         backend/app/data/recall773/disease_map.csv        773 diseases -> PrimeKG names (drugs)

Rules (see docs/773召回模型接入.md):
  1. A model symptom whose name equals a chatbot term, or whose catalog KG node is used by no other
     model symptom, links one-to-one to that chatbot node.
  2. Several model symptoms sharing one broad KG node form a group under that node ("head").
     A NO to the head applies to every member. A YES applies only to the group's representative
     (REPRESENTATIVE below); groups without one are incoherent (e.g. "Pain") and are never asked.
  3. Every model symptom without a precise chatbot node gets its own model-only node
     ("DSS_xxxx", named after the symptom), loaded into the vocabulary only in recall773 mode.
Usage: python scripts/build_recall773_mapping.py --groupwork path/to/GroupWork
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "backend" / "app"
BUNDLE = APP / "recall_model_773" / "artifacts" / "disease_retrieval_773_tristate_v2"
OUT = APP / "data" / "recall773"

# YES to the broad KG term -> this model symptom. Groups not listed have no representative.
REPRESENTATIVE = {
    "Muscle spasm": "cramps and spasms", "Edema": "skin swelling", "Muscle weakness": "weakness",
    "Muscle stiffness": "muscle stiffness or tightness", "Pruritus": "itching of skin",
    "Abdominal pain": "sharp abdominal pain", "Gastroesophageal reflux": "heartburn",
    "Joint swelling": "joint swelling", "Chest pain": "sharp chest pain", "Ocular pain": "pain in eye",
    "Abdominal distention": "stomach bloating", "Dyspnea": "shortness of breath", "Lower limb pain": "leg pain",
    "Hematochezia": "blood in stool", "Headache": "headache", "Localized skin lesion": "skin lesion",
    "Oral ulcer": "mouth ulcer", "Abnormality of eye movement": "eye moves abnormally",
    "Bone pain": "bones are painful", "Abnormal stool composition": "changes in stool appearance",
    "Sinusitis": "sinus congestion", "Dry skin": "skin dryness, peeling, scaliness, or roughness",
    "Limb muscle weakness": "leg weakness",
}
# Display names for pandas-style duplicate columns.
DISPLAY = {"regurgitation.1": "Regurgitation (second variant)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groupwork", required=True, type=Path)
    args = parser.parse_args()

    vocab = json.loads((BUNDLE / "vocabulary.json").read_text(encoding="utf-8"))
    with open(args.groupwork / "Informed Search" / "data" / "nodes.csv", encoding="utf-8", newline="") as f:
        kg_name = {r["node_index"]: r["name"] for r in csv.DictReader(f)}
    with open(APP / "data" / "kg_nodes.csv", encoding="utf-8", newline="") as f:
        chat = {r["node_name"]: r["node_id"] for r in csv.DictReader(f)}
    chat_lower = {k.casefold(): k for k in chat}
    kg = json.load(gzip.open(APP / "data" / "kg_reasoning.json.gz", "rt", encoding="utf-8"))

    # KG node usage across model symptoms (to detect many-to-one groups).
    links = {s: [kg_name[n] for n in m["kg_node_indices"].split(";") if n] for s, m in vocab["symptom_mapping"].items()}
    users = defaultdict(list)
    for s, names in links.items():
        for name in names:
            users[name].append(s)

    rows, extra = [], []
    for i, s in enumerate(vocab["symptoms"]):
        m = vocab["symptom_mapping"][s]
        row = {"feature": s, "feature_index": i, "node_id": "", "node_name": "", "role": "",
               "group_head_id": "", "group_head_name": "", "represents_head": ""}
        exact = chat_lower.get(s.casefold())
        solo = [n for n in links[s] if len(users[n]) == 1 and n in chat]
        heads = [n for n in links[s] if len(users[n]) > 1 and n in chat]
        if heads:
            row["group_head_name"] = heads[0]
            row["group_head_id"] = chat[heads[0]]
            if REPRESENTATIVE.get(heads[0]) == s:
                row["represents_head"] = heads[0]   # YES to the head term -> this symptom
        if exact:
            row.update(node_id=chat[exact], node_name=exact, role="exact_name")
        elif solo:
            row.update(node_id=chat[solo[0]], node_name=solo[0], role="one_to_one")
        elif heads and REPRESENTATIVE.get(heads[0]) == s:
            row.update(node_id=chat[heads[0]], node_name=heads[0], role="group_representative")
        else:
            name = DISPLAY.get(s, s[:1].upper() + s[1:])
            if name.casefold() in chat_lower:
                raise SystemExit(f"Model-only node name collides with the vocabulary: {name}")
            row.update(node_id=m["source_id"], node_name=name, role="model_only_node")
            extra.append({"node_id": m["source_id"], "node_name": name, "node_type": "symptom"})
        rows.append(row)

    reps = {r["represents_head"] for r in rows if r["represents_head"]}
    missing = set(REPRESENTATIVE) - reps
    if missing:
        raise SystemExit(f"Representatives not applied: {sorted(missing)}")

    diseases = []
    for i, d in enumerate(vocab["diseases"]):
        m = vocab["disease_mapping"][d]
        name, link = "", ""
        for kind in ("kg_direct_nodes", "kg_exact_name_nodes"):
            for n in filter(None, m[kind].split(";")):
                candidate = kg_name.get(n, "")
                if candidate in kg["disease_drugs"] or (not name and candidate in kg["disease_ids"]):
                    name, link = candidate, kind.replace("kg_", "").replace("_nodes", "")
            if name in kg["disease_drugs"]:
                break
        diseases.append({"disease": d, "disease_index": i, "primekg_name": name,
                         "primekg_id": kg["disease_ids"].get(name, ""), "link": link,
                         "has_drug_relations": name in kg["disease_drugs"]})

    OUT.mkdir(parents=True, exist_ok=True)
    for fname, data in [("feature_map.csv", rows), ("extra_symptom_nodes.csv", extra), ("disease_map.csv", diseases)]:
        with open(OUT / fname, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, list(data[0]))
            w.writeheader(); w.writerows(data)

    from collections import Counter
    print("feature roles:", dict(Counter(r["role"] for r in rows)))
    print("model-only nodes:", len(extra))
    print("diseases linked to PrimeKG:", sum(bool(d["primekg_name"]) for d in diseases),
          "| with drug relations:", sum(d["has_drug_relations"] for d in diseases))


if __name__ == "__main__":
    main()
