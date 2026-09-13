from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ombrebrain.context.anti_echo import AntiEchoObserver
from ombrebrain.context.dedup import ShadowDedupObserver

@dataclass
class ContextRetrievalAdapter:
    bucket_mgr: Any
    embedding_engine: Any = None
    relevance_threshold: float = 0.65
    semantic_top_k: int = 50
    last_telemetry: dict[str, Any] | None = None
    anti_echo_observer: AntiEchoObserver = AntiEchoObserver()
    dedup_observer: ShadowDedupObserver = ShadowDedupObserver()

    async def _semantic_scores(self, query: str) -> dict[str, float]:
        engine = self.embedding_engine
        if not engine or not getattr(engine, "enabled", False):
            return {}
        try:
            strict_search = getattr(engine, "search_similar_strict", None)
            pairs = await strict_search(query, top_k=max(self.semantic_top_k, 1)) if callable(strict_search) else await engine.search_similar(query, top_k=max(self.semantic_top_k, 1))
            return {str(bucket_id): max(0.0, min(1.0, float(score))) for bucket_id, score in pairs}
        except Exception:
            return {}

    @staticmethod
    def _context_relevance(bucket: dict[str, Any], vector_scores: dict[str, float]) -> float:
        semantic = vector_scores.get(str(bucket.get("id") or ""))
        if semantic is None:
            return 0.0
        recall_floor = 0.55
        if semantic < recall_floor:
            return 0.0
        relevance = 0.65 + ((semantic - recall_floor) / (1.0 - recall_floor)) * 0.35
        return max(0.65, min(1.0, relevance))

    async def retrieve(self, query: str, *, max_results: int = 8, domain_filter: list[str] | None = None, query_valence: float | None = None, query_arousal: float | None = None) -> list[dict[str, Any]]:
        query = str(query or "").strip()
        if not query:
            return []
        vector_scores = await self._semantic_scores(query)
        try:
            matches = await self.bucket_mgr.search(query, limit=max(max_results, 20), domain_filter=domain_filter, query_valence=query_valence, query_arousal=query_arousal, vector_scores=vector_scores, include_archive=False)
        except TypeError:
            matches = await self.bucket_mgr.search(query, limit=max(max_results, 20), domain_filter=domain_filter, query_valence=query_valence, query_arousal=query_arousal, vector_scores=vector_scores)
        results = []
        candidate_count = 0
        relevance_rejected = 0

        for bucket in matches:

            if not isinstance(bucket, dict):
                continue

            candidate_count += 1
            relevance = self._context_relevance(bucket, vector_scores)
            if relevance < self.relevance_threshold:
                relevance_rejected += 1
                continue
            item = dict(bucket)
            item["context_relevance"] = round(relevance, 4)
            results.append(item)
            if len(results) >= max_results:
                break
        anti_echo = self.anti_echo_observer.observe(results)
        dedup = self.dedup_observer.observe(results)
        self.last_telemetry = {"candidate_count": candidate_count, "relevance_rejected": relevance_rejected, "included_count": len(results), "anti_echo_candidates": anti_echo.candidates, "anti_echo_recent_24h": anti_echo.recent_24h, "anti_echo_recent_24_72h": anti_echo.recent_24_72h, "anti_echo_normal": anti_echo.normal, "anti_echo_missing_last_active": anti_echo.missing_last_active, "dedup_candidates": dedup.candidates, "dedup_duplicate_ids": dedup.duplicate_ids, "dedup_exact_text_duplicates": dedup.exact_text_duplicates, "dedup_unique": dedup.unique}
        return results
