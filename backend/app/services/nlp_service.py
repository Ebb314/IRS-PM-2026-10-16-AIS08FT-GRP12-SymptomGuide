import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from app.config import settings
from app.models.chat_models import ExtractedSymptom, Symptom


@dataclass
class Mention:
    text: str
    start: int
    end: int
    confidence: float


class NLPService:
    """
    Free text -> NER mentions -> negation -> KG mapping.

    Heavy models are enabled with NLP_USE_TRANSFORMERS=true. The development
    fallback derives its vocabulary exclusively from the KG CSV, so tests and
    UI work do not depend on model downloads.
    """

    def __init__(self, kg_node_file: Optional[Path] = None, use_models: Optional[bool] = None) -> None:
        self.kg_node_file = kg_node_file or settings.kg_node_file
        self.use_models = settings.use_transformer_models if use_models is None else use_models
        self._kg_nodes = self._load_symptom_nodes(self.kg_node_file)
        self._ner = None
        self._encoder = None
        self._kg_embeddings = None
        if self.use_models:
            self._load_models()

    @staticmethod
    def _load_symptom_nodes(path: Path) -> List[Symptom]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        nodes = []
        for row in rows:
            node_type = (row.get("node_type") or row.get("type") or "").strip().lower()
            if node_type not in {"symptom", "phenotype", "sign", "sign_symptom"}:
                continue
            node_id = (row.get("node_id") or row.get("id") or "").strip()
            name = (row.get("node_name") or row.get("name") or row.get("label") or "").strip()
            if node_id and name:
                nodes.append(Symptom(kg_id=node_id, kg_name=name))
        if not nodes:
            raise ValueError("The KG node file contains no symptom-like nodes")
        return nodes

    def _load_models(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "NLP_USE_TRANSFORMERS=true requires transformers and sentence-transformers"
            ) from exc
        self._ner = pipeline(
            "token-classification",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
        )
        self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._kg_embeddings = self._encoder.encode(
            [node.kg_name for node in self._kg_nodes], normalize_embeddings=True
        )

    def extract(self, text: str) -> Tuple[List[ExtractedSymptom], List[ExtractedSymptom]]:
        text = text.strip()
        mentions = self._model_mentions(text) if self._ner else self._kg_mentions(text)
        positive: List[ExtractedSymptom] = []
        negative: List[ExtractedSymptom] = []
        seen = set()
        for mention in mentions:
            node, similarity = self._map_to_kg(mention.text)
            if node is None or node.kg_id in seen:
                continue
            seen.add(node.kg_id)
            item = ExtractedSymptom(
                kg_id=node.kg_id,
                kg_name=node.kg_name,
                original_text=mention.text,
                confidence=round(min(mention.confidence, similarity), 4),
            )
            (negative if self._is_negated(text, mention.start) else positive).append(item)
        return positive, negative

    def _model_mentions(self, text: str) -> List[Mention]:
        accepted = {"sign_symptom", "symptom", "sign/symptom"}
        mentions = []
        for entity in self._ner(text):
            label = str(entity.get("entity_group", entity.get("entity", ""))).lower().replace("-", "_")
            if label in accepted and float(entity["score"]) >= 0.5:
                mentions.append(
                    Mention(str(entity["word"]).strip(), int(entity["start"]), int(entity["end"]), float(entity["score"]))
                )
        return mentions

    def _kg_mentions(self, text: str) -> List[Mention]:
        """Offline fallback: locate canonical KG symptom labels in user text."""
        mentions = []
        for node in sorted(self._kg_nodes, key=lambda item: len(item.kg_name), reverse=True):
            pattern = r"\b" + re.escape(node.kg_name) + r"\b"
            mentions.extend(Mention(match.group(), match.start(), match.end(), 1.0) for match in re.finditer(pattern, text, re.I))
        return mentions

    def _map_to_kg(self, text: str) -> Tuple[Optional[Symptom], float]:
        if self._encoder is not None:
            import numpy as np

            query = self._encoder.encode([text], normalize_embeddings=True)[0]
            scores = np.dot(self._kg_embeddings, query)
            index = int(np.argmax(scores))
            score = float(scores[index])
            return (self._kg_nodes[index], score) if score >= 0.55 else (None, score)
        normalized = text.casefold().strip()
        best = max(self._kg_nodes, key=lambda node: SequenceMatcher(None, normalized, node.kg_name.casefold()).ratio())
        score = SequenceMatcher(None, normalized, best.kg_name.casefold()).ratio()
        return (best, score) if score >= 0.72 else (None, score)

    @staticmethod
    def _is_negated(text: str, symptom_start: int) -> bool:
        context = text[max(0, symptom_start - 60):symptom_start].lower()
        boundaries = [context.rfind(mark) for mark in (".", ";", ", but", " however ", " but ")]
        context = context[max(boundaries) + 1:]
        patterns: Iterable[str] = (
            r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bwithout\b",
            r"\bden(?:y|ies|ied)\b", r"don['’]t have", r"doesn['’]t have",
        )
        return any(re.search(pattern, context) for pattern in patterns)
