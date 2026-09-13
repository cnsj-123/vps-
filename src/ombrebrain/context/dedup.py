from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class DedupObservation:
    candidates: int = 0
    duplicate_ids: int = 0
    exact_text_duplicates: int = 0
    unique: int = 0
    def to_dict(self) -> dict[str, int]:
        return {"candidates": self.candidates, "duplicate_ids": self.duplicate_ids, "exact_text_duplicates": self.exact_text_duplicates, "unique": self.unique}

class ShadowDedupObserver:
    @staticmethod
    def _text(item: dict[str, Any]) -> str:
        value = item.get("content")
        return str(value).strip() if isinstance(value, str) else ""
    def observe(self, items: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> DedupObservation:
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        duplicate_ids = 0
        exact_text_duplicates = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            bucket_id = str(item.get("id") or "").strip()
            if bucket_id:
                if bucket_id in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(bucket_id)
            text = self._text(item)
            if text:
                if text in seen_text:
                    exact_text_duplicates += 1
                else:
                    seen_text.add(text)
        return DedupObservation(candidates=len(items), duplicate_ids=duplicate_ids, exact_text_duplicates=exact_text_duplicates, unique=max(0, len(items) - max(duplicate_ids, exact_text_duplicates)))
