"""REASONING_MODE=ddxplus: teammate's DDXPlus tri-state recall model + GroupWork informed search.

  DDXPlusRecallService  chatbot symptoms (HPO vocabulary) -> DDXPlus evidence IDs -> model
                        probabilities over 49 diseases. Candidates = the smallest set holding
                        DDXPLUS_MASS (0.95) of the probability, capped at DDXPLUS_TOP_K, so the
                        list shrinks as the model becomes confident and the "<= 2 candidates"
                        stop rule is meaningful. While the top disease is below DDXPLUS_CONFIDENT
                        (0.8) at least 3 candidates are kept, so two equally likely diseases lead
                        to a question rather than an early stop. "Not sure" is never sent
                        (tri-state "unknown").
  DDXPlusSearchService  module3_informed_search over a disease -> symptom table:
                        official DDXPlus release_conditions.json when DDXPLUS_CONDITIONS_FILE is
                        set, otherwise a provisional table derived from the model itself.
  DDXPlusDrugService    maps DDXPlus diseases to PrimeKG names, then KG drug relations with the
                        contraindication filter (KGDrugService).

Mapping tables (drafts for team review): data/ddxplus_mapping/symptom_map.csv, disease_map.csv.
"Absent" answers are handled softly by the model (teammate design). Hard deletion
(state_update.filter_candidates) is opt-in via DDXPLUS_HARD_FILTER=true, because a
DDXPlus patient usually lacks several symptoms that their disease *can* have.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from app.logging_setup import fmt_scores, get_logger
from app.models.chat_models import DiseaseCandidate
from app.reasoning.state_update import filter_candidates
from app.services.ddxplus_runtime import TristateRecallModel, load_ddxplus_model
from app.services.interfaces import DrugService, InformedSearchService, RecallService
from app.services.kg_reasoning_service import KGDrugService, KGSearchService, load_kg_data

MAPPING_DIR = Path(__file__).resolve().parents[1] / "data" / "ddxplus_mapping"
_recall_log, _drug_log = get_logger("recall"), get_logger("drugs")


class DDXPlusData:
    # Shared with Recall773Data: how candidates and questions are labelled.
    id_prefix, id_width, source_tag, question_label = "DDX", 3, "ddxplus-tristate-v1", "DDXPlus question"

    def __init__(self, model: TristateRecallModel, nodes: dict, conditions_file: Optional[str] = None):
        self.model = model
        vocab_ids = {n.kg_name: k for k, n in nodes.items()}
        self.kg_to_evidence: Dict[str, List[str]] = defaultdict(list)   # chatbot kg_id -> E ids
        self.evidence_to_name: Dict[str, str] = {}                       # E id -> chatbot kg_name
        self.questions: Dict[str, List[str]] = defaultdict(list)         # kg_name -> DDXPlus questions
        with open(MAPPING_DIR / "symptom_map.csv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["mapping_status"] == "unmapped":
                    continue
                if vocab_ids.get(row["kg_name"]) != row["kg_id"]:
                    raise ValueError(f"symptom_map.csv row {row['evidence_id']} does not match the chatbot vocabulary")
                self.kg_to_evidence[row["kg_id"]].append(row["evidence_id"])
                # The first row is the term the search asks; later "synonym" rows only map user input.
                self.evidence_to_name.setdefault(row["evidence_id"], row["kg_name"])
                self.questions[row["kg_name"]].append(row["question_en"])
        self.primekg_name: Dict[str, str] = {}
        with open(MAPPING_DIR / "disease_map.csv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["primekg_name"]:
                    self.primekg_name[row["ddx_name"]] = row["primekg_name"]
        self.disease_index = {d: i for i, d in enumerate(model.diseases)}
        if conditions_file:
            self.disease_symptoms, self.table_source = self._from_conditions(conditions_file), "ddxplus-release-conditions"
        else:
            self.disease_symptoms, self.table_source = self._from_model(), "model-derived-provisional"

    def _from_model(self):
        """Provisional: symptom s belongs to disease d if s alone gives d >= 2x uniform probability."""
        table = defaultdict(set)
        threshold = 2.0 / len(self.model.diseases)
        for evidence, name in self.evidence_to_name.items():
            probs = self.model.probabilities([evidence])
            for i, p in enumerate(probs):
                if p >= threshold:
                    table[self.model.diseases[i]].add(name)
        return {d: sorted(s) for d, s in table.items()}

    def _from_conditions(self, path):
        conditions = json.loads(Path(path).read_text(encoding="utf-8"))
        table = {}
        for key, info in conditions.items():
            names = {key, info.get("condition_name", ""), info.get("cond-name-eng", "")}
            disease = next((n for n in names if n in self.disease_index), None)
            if disease is None:
                continue
            codes = list(info.get("symptoms", {}))
            if codes and not all(str(c).startswith("E_") for c in codes):
                raise ValueError("release_conditions.json must use E_ evidence codes (English DDXPlus release)")
            table[disease] = sorted({self.evidence_to_name[c] for c in codes if c in self.evidence_to_name})
        missing = set(self.model.diseases) - set(table)
        if missing:
            raise ValueError(f"release_conditions.json lacks {len(missing)} model diseases, e.g. {sorted(missing)[:3]}")
        return table

    def evidence(self, symptoms):
        """Model features implied by a YES to these chatbot symptoms."""
        return sorted({e for s in symptoms for e in self.kg_to_evidence.get(s.kg_id, [])})

    def absent_evidence(self, symptoms):
        """Model features implied by a NO (DDXPlus: same as YES mapping)."""
        return self.evidence(symptoms)


class DDXPlusRecallService(RecallService):
    def __init__(self, data: DDXPlusData, top_k=10, mass=0.95, temperature=1.0, hard_filter=False,
                 confident=0.8, min_candidates=3):
        self.data, self.top_k, self.mass, self.temperature, self.hard_filter = data, top_k, mass, temperature, hard_filter
        self.confident, self.min_candidates = confident, min(min_candidates, top_k)
        self.source = f"{data.source_tag}/{data.model.backend}"

    def recall(self, positive, negative):
        present = self.data.evidence(positive)
        unmapped = ([s.kg_name for s in positive if not self.data.evidence([s])]
                    + [s.kg_name for s in negative if not self.data.absent_evidence([s])])
        if unmapped:
            _recall_log.info("not in the model's symptom list (ignored): %s", unmapped)
        if not present:
            _recall_log.info("no present symptom the model understands -> no candidates")
            return []  # nothing the model understands yet (or only denials)
        absent = sorted(set(self.data.absent_evidence(negative)) - set(present))
        _recall_log.info("model input (%s): present=%s absent=%s", self.source, present, absent)
        probs = self.data.model.probabilities(present, absent, self.temperature)
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
        ranked = [{"disease": self.data.model.diseases[i], "score": float(probs[i])} for i in order]
        _recall_log.info("model top-5 of %d: %s", len(ranked), fmt_scores((r["disease"], r["score"]) for r in ranked))
        if self.hard_filter:
            state = {"positive": [s.kg_name for s in positive], "negative": [s.kg_name for s in negative]}
            before = len(ranked)
            ranked = filter_candidates(ranked, state, self.data.disease_symptoms)
            _recall_log.info("hard filter removed %d diseases with a denied symptom", before - len(ranked))
        chosen, total = [], 0.0
        keep_at_least = self.min_candidates if ranked and ranked[0]["score"] < self.confident else 1
        for item in ranked[: self.top_k]:
            chosen.append(item)
            total += item["score"]
            if total >= self.mass and len(chosen) >= keep_at_least:
                break
        if ranked:
            _recall_log.info("candidates=%d (cumulative mass %.3f; top-1 %.3f %s confident %.2f -> keep at least %d; cap %d)",
                             len(chosen), total, ranked[0]["score"], "<" if ranked[0]["score"] < self.confident else ">=",
                             self.confident, keep_at_least, self.top_k)
        positive_by_name = {s.kg_name: s.kg_id for s in positive}
        return [
            DiseaseCandidate(
                id=f"{self.data.id_prefix}:{self.data.disease_index[c['disease']]:0{self.data.id_width}d}",
                name=c["disease"],
                score=round(c["score"], 4),
                evidence_ids=sorted(positive_by_name[n] for n in self.data.disease_symptoms.get(c["disease"], [])
                                    if n in positive_by_name),
                source=self.source,
            )
            for c in chosen
        ]


class DDXPlusSearchService(KGSearchService):
    """GroupWork informed search over the DDXPlus disease table; asks only model-mapped symptoms."""

    def __init__(self, data: DDXPlusData, nodes: dict, whitelist=None):
        mapped = set(data.evidence_to_name.values())
        super().__init__(data, nodes, whitelist=mapped & set(whitelist) if whitelist is not None else mapped)

    def select_next(self, candidates, positive, negative, asked_symptom_ids):
        question = super().select_next(candidates, positive, negative, asked_symptom_ids)
        if question is None:
            return None
        asked = " / ".join(self.data.questions.get(question.kg_name, [])[:2])
        return question.model_copy(update={"rationale": question.rationale + (
            f' {self.data.question_label}: "{asked}"' if asked else "")})


class DDXPlusDrugService(DrugService):
    def __init__(self, data: DDXPlusData, kg_drugs: KGDrugService):
        self.data, self.kg_drugs = data, kg_drugs

    def recommend(self, candidates):
        mapped = [DiseaseCandidate(id=c.id, name=self.data.primekg_name[c.name], score=c.score)
                  for c in candidates if c.name in self.data.primekg_name]
        back = {self.data.primekg_name[c.name]: c.name for c in candidates if c.name in self.data.primekg_name}
        _drug_log.info("PrimeKG mapping: %s | no PrimeKG disease: %s",
                       [f"{c.name} -> {self.data.primekg_name[c.name]}" for c in candidates if c.name in self.data.primekg_name],
                       [c.name for c in candidates if c.name not in self.data.primekg_name])
        out = []
        for drug in self.kg_drugs.recommend(mapped):
            ddx = back[drug.related_disease]
            label = ddx if ddx.casefold() == drug.related_disease.casefold() else f"{ddx} (PrimeKG: {drug.related_disease})"
            out.append(drug.model_copy(update={"related_disease": label}))
        return out


def build_ddxplus_services(nodes: dict, kg_reasoning_file: str, top_k=10, mass=0.95, temperature=1.0,
                           hard_filter=False, per_disease=5, conditions_file: Optional[str] = None,
                           whitelist_path: Optional[str] = None, backend="auto", confident=0.8):
    if all(k.startswith("DEMO:") for k in nodes):
        raise ValueError("REASONING_MODE=ddxplus requires the full KG vocabulary (kg_nodes.csv), not the DEMO list.")
    from app.services.kg_reasoning_service import load_symptom_whitelist
    data = DDXPlusData(load_ddxplus_model(backend), nodes, conditions_file or None)
    kg = load_kg_data(str(kg_reasoning_file))
    return (DDXPlusRecallService(data, top_k, mass, temperature, hard_filter, confident),
            DDXPlusSearchService(data, nodes, load_symptom_whitelist(whitelist_path)),
            DDXPlusDrugService(data, KGDrugService(kg, per_disease)))
