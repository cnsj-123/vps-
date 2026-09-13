from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ContextRelevanceFilter:
    threshold: float = 0.65

    def score(self, item: dict[str, Any]) -> float:
        direct = item.get("context_relevance")
        if isinstance(direct, (int, float)) and not isinstance(direct, bool):
            return max(0.0, min(1.0, float(direct)))
        candidates = []
        for key in ("semantic_relevance", "semantic_score", "vector_score", "bm25_relevance", "lexical_relevance", "topic_relevance", "topic_score", "relevance"):
            value = item.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                candidates.append(max(0.0, min(1.0, float(value))))
        if not candidates:
            return 0.0
        return max(candidates)

    def include(self, item: dict[str, Any]) -> bool:
        return self.score(item) >= self.threshold

    def filter(self, items: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
        accepted = []
        for item in items:
            relevance = self.score(item)
            if relevance >= self.threshold:
                enriched = dict(item)
                enriched["context_relevance"] = round(relevance, 4)
                accepted.append(enriched)
        accepted.sort(key=lambda item: float(item.get("context_relevance", 0.0)), reverse=True)
        return tuple(accepted)
