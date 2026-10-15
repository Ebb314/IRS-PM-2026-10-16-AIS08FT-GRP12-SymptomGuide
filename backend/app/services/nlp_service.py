"""Bounded English evidence extraction. Rules are an inspectable baseline, not clinical NLP."""
import csv
import re
from pathlib import Path
from app.config import settings
from app.logging_setup import fmt_states, get_logger
from app.models.chat_models import ExtractedSymptom, NLPResult, Symptom

_log = get_logger("nlp")

BOUNDARY = re.compile(r"[.!?;\n]|\b(?:but|however|yet)\b|,\s*(?=(?:I|he|she|my|just)\b)|\band\s+(?=(?:I|he|she)\s+(?:have|feel|am|has|is)\b)", re.I)

class NLPService:
    def __init__(self, kg_node_file: Path | None = None, use_models: bool | None = None,
                 extra_vocab_file: str | None = None):
        self.path = kg_node_file or settings.kg_node_file
        self.nodes = {}
        self.aliases = {}
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row.get("node_type") or row.get("type", "")).lower() not in {"symptom","phenotype","sign","sign_symptom"}:
                    continue
                node_id = row.get("node_id") or row.get("id")
                name = row.get("node_name") or row.get("name") or row.get("label")
                if not node_id or not name:
                    continue
                if node_id in self.nodes:
                    raise ValueError(f"Duplicate vocabulary ID: {node_id}")
                self.nodes[node_id] = Symptom(kg_id=node_id, kg_name=name)
                for alias in [name] + row.get("aliases", "").split("|"):
                    if alias.strip():
                        self.aliases.setdefault(alias.strip().casefold(), []).append(node_id)
        # Optional extra nodes (recall773 mode: model-only symptoms such as Diarrhea that have no
        # precise KG phenotype node). Same CSV columns as the vocabulary; existing names win.
        extra = settings.extra_vocab_file if extra_vocab_file is None else extra_vocab_file
        if extra and Path(extra).is_file():
            with Path(extra).open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    node_id, name = row.get("node_id", ""), row.get("node_name", "")
                    if not node_id or not name or node_id in self.nodes or name.casefold() in self.aliases:
                        continue
                    self.nodes[node_id] = Symptom(kg_id=node_id, kg_name=name)
                    self.aliases.setdefault(name.casefold(), []).append(node_id)
        # Optional colloquial aliases (data/symptom_aliases.csv, "kg_name,aliases" with a|b|c).
        # They only add phrases; an existing vocabulary name or alias always wins.
        alias_file = settings.symptom_alias_file
        if alias_file and Path(alias_file).is_file():
            by_name = {n.kg_name: k for k, n in self.nodes.items()}
            with Path(alias_file).open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    node_id = by_name.get(row.get("kg_name", ""))
                    for alias in (row.get("aliases") or "").split("|") if node_id else []:
                        key = alias.strip().casefold()
                        if key and key not in self.aliases:
                            self.aliases[key] = [node_id]
        if not self.nodes:
            raise ValueError("Vocabulary contains no supported symptom nodes")
        self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(self.aliases, key=len, reverse=True)) + r")(?!\w)", re.I)
        self._ner = self._encoder = None
        self._node_ids = list(self.nodes)
        self.engine = "rules-v2"
        if settings.use_transformer_models if use_models is None else use_models:
            self._load_models()

    def _load_models(self):
        from transformers import pipeline
        from sentence_transformers import SentenceTransformer
        self._ner = pipeline("token-classification", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
        self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._embeddings = self._encoder.encode([self.nodes[k].kg_name for k in self._node_ids], normalize_embeddings=True)
        self.engine = "biomedical-ner+rules-v2"

    def _mentions(self, text):
        mentions = [(m.start(), m.end(), self.aliases[m.group().casefold()], 1.0) for m in self.pattern.finditer(text)]
        if self._ner is not None:
            import numpy as np
            for item in self._ner(text):
                if str(item.get("entity_group", "")).lower() not in {"sign_symptom", "symptom", "sign/symptom"}:
                    continue
                start, end = int(item["start"]), int(item["end"])
                if any(start < b and end > a for a,b,_,_ in mentions):
                    continue
                query = self._encoder.encode([text[start:end]], normalize_embeddings=True)[0]
                scores = np.dot(self._embeddings, query)
                ranking = np.argsort(scores)[::-1]
                best = int(ranking[0])
                # Abstain when two concepts are almost equally plausible.
                if scores[best] >= .65 and (len(ranking) == 1 or scores[best] - scores[int(ranking[1])] >= .08):
                    mentions.append((start, end, [self._node_ids[best]], min(float(item["score"]), float(scores[best]))))
        return sorted(mentions, key=lambda m: m[0])

    def analyze(self, text: str, turn: int = 0) -> NLPResult:
        text = text.replace("’", "'")
        result = NLPResult(engine=self.engine)
        for start, end, ids, score in self._mentions(text):
            if len(ids) != 1:
                result.warnings.append(f"Ambiguous vocabulary alias: {text[start:end]}. Please clarify.")
                continue
            left = max([m.end() for m in BOUNDARY.finditer(text) if m.end() <= start] or [0])
            right = min([m.start() for m in BOUNDARY.finditer(text) if m.start() >= end] or [len(text)])
            prefix, suffix = text[left:start].lower(), text[end:right].lower()
            # Ontology labels may themselves contain "history of". Assertion cues
            # belong to the surrounding report, not to the matched concept name.
            clause = (text[left:start]+' '+text[end:right]).lower()
            other = bool(re.search(r"\b(?:my (?:mother|father|sister|brother|friend|child|son|daughter|wife|husband)|he|she|they)\b", prefix))
            historical = bool(re.search(r"\b(?:used to|(?<!since )last (?:year|month)|previously|history of)\b", clause))
            uncertain = bool(re.search(r"\b(?:not sure|unsure|uncertain|maybe|perhaps|might|possibly|whether|don't know|do not know)\b", prefix))
            neg_prefix = re.sub(r"\bnot only\b", "", prefix)
            negative = bool(re.search(r"\b(?:no|not|never|without|denies|deny|denied)\b|\b(?:don't|doesn't|didn't)\b", neg_prefix))
            negative |= bool(re.match(r"(?:-free\b|\s+(?:(?:is|was)\s+)?(?:absent|gone|denied)\b|\s+(?:isn't|is not)\s+present\b|\s+(?:yesterday,?\s+)?which is gone\b)", suffix))
            negative |= bool(re.match(r"\?\s*(?:I\s+)?(?:don't think so|do not think so|no)\b", text[end:], re.I))
            item = ExtractedSymptom(**self.nodes[ids[0]].model_dump(), original_text=text[start:end],
                confidence=round(score,4), status="uncertain" if uncertain else "absent" if negative else "present",
                experiencer="other" if other else "self", temporality="historical" if historical else "current",
                start=start, end=end, turn=turn, source=self.engine)
            if other or historical:
                result.excluded.append(item)
            else:
                result.evidence.append(item)
        # Resolve repeated mentions by textual order. Contradictory assertions need clarification.
        latest = {}
        for item in result.evidence:
            old = latest.get(item.kg_id)
            if old and old.status != item.status:
                between = text[old.end:item.start].lower()
                if not re.search(r"\b(?:actually|correction|sorry|meant|rather)\b", between):
                    item = item.model_copy(update={"status": "uncertain"})
                    result.warnings.append(f"Conflicting statements about {item.kg_name}; please confirm in the evidence panel.")
            latest[item.kg_id] = item
        result.evidence = list(latest.values())
        if not result.evidence:
            result.warnings.append("No current self-reported symptom matched the supported vocabulary.")
        _log.info("rules: %s | excluded(other person / past)=%s warnings=%d", fmt_states(result.evidence),
                  [e.kg_name for e in result.excluded], len(result.warnings))
        return result

    def extract(self, text):
        """Compatibility method; use analyze() to preserve uncertainty and provenance."""
        result = self.analyze(text)
        return ([e for e in result.evidence if e.status == "present"], [e for e in result.evidence if e.status == "absent"])

    def baseline(self, text):
        """Canonical-only positive keyword baseline. Deliberately no assertion handling."""
        return [ExtractedSymptom(**node.model_dump(), original_text=m.group(), start=m.start(), end=m.end(), source="keywords-v1")
            for node in self.nodes.values()
            for m in re.finditer(r"\b" + re.escape(node.kg_name) + r"\b", text, re.I)]

