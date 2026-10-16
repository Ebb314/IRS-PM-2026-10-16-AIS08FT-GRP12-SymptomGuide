"""In-process knowledge-graph reasoning for REASONING_MODE=kg.

Implements the three teammate interfaces (services/interfaces.py) on top of the
GroupWork modules vendored in app/reasoning/:

  KGRecallService  interim KG recall: candidates are the diseases that explain the
                   most positive symptoms; denied symptoms remove diseases
                   (state_update.filter_candidates). Replace with the trained
                   recall model once it exists; the interface stays the same.
  KGSearchService  module3_informed_search.select_best_next_symptom
  KGDrugService    drug_retrieval.retrieve_drugs_for_diseases (+ contraindication filter)

Data: backend/app/data/kg_reasoning.json.gz, built by scripts/build_kg_reasoning_data.py.
Disease names are the join key with the GroupWork lookup tables; symptom names
are the canonical names of the chatbot vocabulary (kg_nodes.csv).
"""
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import logging

from app.logging_setup import fmt_scores, get_logger
from app.models.chat_models import DiseaseCandidate, DrugRecommendation, NextSymptom
from app.reasoning.drug_retrieval import retrieve_drugs_for_diseases
from app.reasoning.module3_informed_search import (
    _compute_heuristics, _normalized_weights, _retrieve_candidate_symptoms, select_best_next_symptom,
)
from app.reasoning.state_update import filter_candidates
from app.services.interfaces import DrugService, InformedSearchService, RecallService

_recall_log, _search_log, _drug_log = get_logger("recall"), get_logger("search"), get_logger("drugs")

RECALL_SOURCE = "primekg-kg-recall-v1"
DRUG_SOURCE = "PrimeKG project KG"


class KGData:
    """Lookup tables shared by the three services (loaded once per process)."""

    def __init__(self, payload: dict):
        self.disease_ids: Dict[str, str] = payload["disease_ids"]
        self.id_to_disease = {v: k for k, v in self.disease_ids.items()}
        self.disease_symptoms: Dict[str, List[str]] = payload["disease_symptoms"]
        self.disease_lacks: Dict[str, List[str]] = payload["disease_lacks"]
        self.disease_drug_pairs: Dict[str, List[List[str]]] = payload["disease_drugs"]
        self.disease_drugs = {d: [drug for drug, _ in pairs] for d, pairs in self.disease_drug_pairs.items()}
        self.relations = {(d, drug): rel for d, pairs in self.disease_drug_pairs.items() for drug, rel in pairs}
        self.contraindications: Dict[str, List[str]] = payload["contraindications"]

        self.symptom_diseases: Dict[str, List[str]] = defaultdict(list)
        for disease, symptoms in self.disease_symptoms.items():
            for s in symptoms:
                self.symptom_diseases[s].append(disease)
        # Rare symptoms are stronger evidence than ones shared by thousands of diseases.
        n = len(self.disease_symptoms)
        self.idf = {s: math.log((n + 1) / (len(ds) + 1)) + 1.0 for s, ds in self.symptom_diseases.items()}


@lru_cache(maxsize=4)
def load_kg_data(path: str) -> KGData:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return KGData(json.load(f))


def load_symptom_whitelist(path: Optional[str]) -> Optional[set]:
    """Optional recall-model vocabulary (symptom_hpo_matched.csv, column matched_name)."""
    if not path:
        return None
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["matched_name"].strip() for row in csv.DictReader(f) if row.get("matched_name")}


class KGRecallService(RecallService):
    def __init__(self, data: KGData, top_k: int = 10):
        self.data, self.top_k = data, top_k

    def recall(self, positive, negative):
        positive_ids = {s.kg_name: s.kg_id for s in positive}
        if not positive_ids:
            _recall_log.info("kg: no present symptom yet -> no candidates")
            return []  # only denials: nothing to explain yet
        counts: Counter = Counter()
        for name in positive_ids:
            counts.update(self.data.symptom_diseases.get(name, []))
        if not counts:
            _recall_log.info("kg: no PrimeKG disease has %s -> no candidates", sorted(positive_ids))
            return []

        state = {"positive": list(positive_ids), "negative": [s.kg_name for s in negative]}
        pool: List[str] = []
        # Prefer diseases that explain the most positive symptoms; if every one of them
        # was removed by a denial, fall back to the next level.
        for level in sorted(set(counts.values()), reverse=True):
            level_pool = [{"disease": d, "score": 0.0} for d, c in counts.items() if c == level]
            pool = [c["disease"] for c in filter_candidates(
                level_pool, state, self.data.disease_symptoms, self.data.disease_lacks)]
            _recall_log.info("kg: %d diseases match %d/%d present symptoms; %d removed by No answers / KG 'lacks'",
                             len(level_pool), level, len(positive_ids), len(level_pool) - len(pool))
            if pool:
                break
        if not pool:
            _recall_log.info("kg: every matching disease was removed -> no candidates")
            return []

        def raw_score(disease):
            matched = set(self.data.disease_symptoms[disease]) & positive_ids.keys()
            # Specificity: the same evidence counts more for a disease with fewer annotated symptoms.
            return sum(self.data.idf[s] for s in matched) / math.log2(2 + len(self.data.disease_symptoms[disease]))

        scored = sorted(((raw_score(d), d) for d in pool), key=lambda x: (-x[0], x[1]))[: self.top_k]
        total = sum(r for r, _ in scored) or 1.0
        _recall_log.info("kg: kept top %d of %d by IDF specificity: %s", len(scored), len(pool),
                         fmt_scores((d, r / total) for r, d in scored))
        return [
            DiseaseCandidate(
                id=self.data.disease_ids[d],
                name=d,
                score=round(r / total, 4),
                evidence_ids=sorted(positive_ids[s] for s in set(self.data.disease_symptoms[d]) & positive_ids.keys()),
                source=RECALL_SOURCE,
            )
            for r, d in scored
        ]


class KGSearchService(InformedSearchService):
    def __init__(self, data: KGData, nodes: dict, whitelist: Optional[Iterable[str]] = None):
        self.data = data
        self.id_to_name = {kg_id: node.kg_name for kg_id, node in nodes.items()}
        self.name_to_id = {name: kg_id for kg_id, name in self.id_to_name.items()}
        allowed = set(self.name_to_id)  # never ask outside the chatbot vocabulary
        self.allowed = allowed & set(whitelist) if whitelist is not None else allowed

    def select_next(self, candidates, positive, negative, asked_symptom_ids):
        cands = [{"disease": c.name, "score": c.score} for c in candidates
                 if c.name in self.data.disease_symptoms]
        state = {
            "positive": [s.kg_name for s in positive],
            "negative": [s.kg_name for s in negative],
            "uncertain": [],
            "asked": [self.id_to_name[i] for i in asked_symptom_ids if i in self.id_to_name],
        }
        if len(cands) < len(candidates):
            _search_log.info("%d candidates have no disease-symptom entry and are ignored", len(candidates) - len(cands))
        if _search_log.isEnabledFor(logging.DEBUG):
            pool = _retrieve_candidate_symptoms(cands, state, self.data.disease_symptoms, self.allowed)
            table = sorted(_compute_heuristics(pool, cands, self.data.disease_symptoms), key=lambda x: (-x[1], x[0]))
            _search_log.debug("%d askable symptoms; top h(s)=min(P_yes,1-P_yes): %s", len(pool),
                              ", ".join(f"{s} {h:.3f}" for s, h in table[:8]))
        result = select_best_next_symptom(cands, state, self.data.disease_symptoms,
                                          allowed_symptoms=self.allowed)
        name = result["symptom"]
        if name is None:
            _search_log.info("no informative question left among %d candidates (all h=0, or < 2 candidates)", len(cands))
            return None
        weights = _normalized_weights(cands)
        share = sum(w for d, w in weights.items() if name in self.data.disease_symptoms.get(d, []))
        having = sum(1 for c in cands if name in self.data.disease_symptoms[c["disease"]])
        _search_log.info("chose %s: h=%.3f, %d/%d candidates have it (%.0f%% of weight)",
                         name, result["heuristic_score"], having, len(cands), share * 100)
        return NextSymptom(
            kg_id=self.name_to_id[name],
            kg_name=name,
            heuristic_score=round(result["heuristic_score"], 4),
            rationale=(f"Asked because it best splits the {len(cands)} candidates: {having} of them "
                       f"({share:.0%} of the ranking weight) have this symptom, so a Yes or a No "
                       f"narrows the list. Informed search: weighted balanced split, greedy best-first."),
        )


class KGDrugService(DrugService):
    def __init__(self, data: KGData, per_disease: int = 5):
        self.data, self.per_disease = data, per_disease

    def recommend(self, candidates):
        cands = [{"disease": c.name, "score": c.score} for c in candidates]
        safe = retrieve_drugs_for_diseases(cands, self.data.disease_drugs, self.data.contraindications)
        out = []
        for c in cands:
            disease = c["disease"]
            total, kept = len(self.data.disease_drugs.get(disease, [])), safe.get(disease, [])
            removed = total - len(kept)
            _drug_log.info("%s: %d related drugs, %d contraindicated, showing %d", disease, total, removed,
                           min(len(kept), self.per_disease))
            for drug in kept[: self.per_disease]:
                note = (f"Showing {min(len(kept), self.per_disease)} of {len(kept)} related drugs."
                        + (f" {removed} excluded: contraindicated for a remaining candidate." if removed else ""))
                out.append(DrugRecommendation(
                    name=drug,
                    related_disease=disease,
                    relation=self.data.relations.get((disease, drug), "indication"),
                    source=DRUG_SOURCE,
                    note=note,
                ))
        return out


def build_kg_services(path: str, nodes: dict, top_k: int = 10, per_disease: int = 5,
                      whitelist_path: Optional[str] = None):
    if not Path(path).exists():
        raise ValueError(f"KG reasoning data not found: {path}. Run scripts/build_kg_reasoning_data.py.")
    if all(k.startswith("DEMO:") for k in nodes):
        raise ValueError("REASONING_MODE=kg requires the full KG vocabulary (kg_nodes.csv), not the DEMO list.")
    data = load_kg_data(str(path))
    whitelist = load_symptom_whitelist(whitelist_path)
    return (KGRecallService(data, top_k), KGSearchService(data, nodes, whitelist),
            KGDrugService(data, per_disease))
