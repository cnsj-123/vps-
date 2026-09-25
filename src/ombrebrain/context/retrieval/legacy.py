from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ombrebrain.context.anti_echo import (
    AntiEchoObserver,
)
from ombrebrain.context.dedup import (
    ShadowDedupObserver,
)
from ombrebrain.context.retrieval_decision_shadow import (
    RetrievalDecisionShadowObserver,
)


_SEMANTIC_RECALL_FLOOR = 0.55


@dataclass
class ContextRetrievalAdapter:
    bucket_mgr: Any
    embedding_engine: Any = None
    relevance_threshold: float = 0.65
    semantic_top_k: int = 50
    last_telemetry: dict[str, Any] | None = None
    anti_echo_observer: AntiEchoObserver = (
        AntiEchoObserver()
    )
    dedup_observer: ShadowDedupObserver = (
        ShadowDedupObserver()
    )
    decision_shadow_observer: (
        RetrievalDecisionShadowObserver
    ) = RetrievalDecisionShadowObserver()

    async def _semantic_scores(
        self,
        query: str,
    ) -> dict[str, float]:

        engine = self.embedding_engine

        if (
            not engine
            or not getattr(
                engine,
                "enabled",
                False,
            )
        ):
            return {}

        try:
            strict_search = getattr(
                engine,
                "search_similar_strict",
                None,
            )

            if callable(strict_search):
                pairs = await strict_search(
                    query,
                    top_k=max(
                        self.semantic_top_k,
                        1,
                    ),
                )
            else:
                pairs = await engine.search_similar(
                    query,
                    top_k=max(
                        self.semantic_top_k,
                        1,
                    ),
                )

            return {
                str(bucket_id):
                    max(
                        0.0,
                        min(
                            1.0,
                            float(score),
                        ),
                    )
                for bucket_id, score
                in pairs
            }

        except Exception:
            # Existing fail-open behavior is preserved.
            return {}

    @staticmethod
    def _context_relevance(
        bucket: dict[str, Any],
        vector_scores: dict[str, float],
    ) -> float:

        semantic = vector_scores.get(
            str(
                bucket.get("id")
                or ""
            )
        )

        if semantic is None:
            return 0.0

        recall_floor = (
            _SEMANTIC_RECALL_FLOOR
        )

        if semantic < recall_floor:
            return 0.0

        relevance = (
            0.65
            + (
                (
                    semantic
                    - recall_floor
                )
                / (
                    1.0
                    - recall_floor
                )
            )
            * 0.35
        )

        return max(
            0.65,
            min(
                1.0,
                relevance,
            ),
        )

    def _store_telemetry(
        self,
        *,
        embedding_enabled: bool,
        semantic_score_count: int,
        raw_match_count: int,
        processed_candidate_count: int,
        relevance_rejected: int,
        included: list[dict[str, Any]],
        max_semantic_score: float | None,
        outcome: str,
    ) -> None:

        anti_echo = (
            self.anti_echo_observer.observe(
                included
            )
        )

        dedup = (
            self.dedup_observer.observe(
                included
            )
        )

        decision_shadow = (
            self.decision_shadow_observer.observe(
                included
            )
        )

        context_relevances = [
            float(
                item[
                    "context_relevance"
                ]
            )
            for item in included
            if isinstance(
                item,
                dict,
            )
            and isinstance(
                item.get(
                    "context_relevance"
                ),
                (int, float),
            )
            and not isinstance(
                item.get(
                    "context_relevance"
                ),
                bool,
            )
        ]

        min_context_relevance = (
            min(context_relevances)
            if context_relevances
            else None
        )

        max_context_relevance = (
            max(context_relevances)
            if context_relevances
            else None
        )

        self.last_telemetry = {
            # Retrieval pipeline.
            "embedding_enabled":
                embedding_enabled,
            "semantic_score_count":
                semantic_score_count,
            "raw_match_count":
                raw_match_count,
            # Backward-compatible name.
            "candidate_count":
                processed_candidate_count,
            "processed_candidate_count":
                processed_candidate_count,
            "relevance_rejected":
                relevance_rejected,
            "included_count":
                len(included),
            "max_semantic_score":
                (
                    round(
                        max_semantic_score,
                        4,
                    )
                    if max_semantic_score
                    is not None
                    else None
                ),

            # Calibration:
            # raw semantic score is first mapped through
            # _context_relevance(), then compared against
            # relevance_threshold. These values make that
            # distinction explicit without changing behavior.
            "semantic_recall_floor":
                _SEMANTIC_RECALL_FLOOR,
            "min_context_relevance":
                (
                    round(
                        min_context_relevance,
                        4,
                    )
                    if min_context_relevance
                    is not None
                    else None
                ),
            "max_context_relevance":
                (
                    round(
                        max_context_relevance,
                        4,
                    )
                    if max_context_relevance
                    is not None
                    else None
                ),
            "context_relevance_threshold":
                float(
                    self.relevance_threshold
                ),

            # Backward-compatible telemetry key.
            "relevance_threshold":
                float(
                    self.relevance_threshold
                ),

            # Explicitly documents that anti-echo/dedup
            # are still observers and never filters.
            "filter_mode":
                "observe_only",

            "outcome":
                outcome,

            # Observation only.
            # These DO NOT filter retrieval.
            "anti_echo_candidates":
                anti_echo.candidates,
            "anti_echo_recent_24h":
                anti_echo.recent_24h,
            "anti_echo_recent_24_72h":
                anti_echo.recent_24_72h,
            "anti_echo_normal":
                anti_echo.normal,
            "anti_echo_missing_last_active":
                anti_echo.missing_last_active,

            "dedup_candidates":
                dedup.candidates,
            "dedup_duplicate_ids":
                dedup.duplicate_ids,
            "dedup_exact_text_duplicates":
                dedup.exact_text_duplicates,
            "dedup_unique":
                dedup.unique,

            # Hypothetical future filter decisions only.
            # Retrieval results above are NOT changed.
            "decision_shadow":
                decision_shadow.to_dict(),
        }

    async def retrieve(
        self,
        query: str,
        *,
        max_results: int = 8,
        domain_filter: list[str] | None = None,
        query_valence: float | None = None,
        query_arousal: float | None = None,
    ) -> list[dict[str, Any]]:

        query = str(
            query or ""
        ).strip()

        engine = self.embedding_engine

        embedding_enabled = bool(
            engine
            and getattr(
                engine,
                "enabled",
                False,
            )
        )

        if not query:
            self._store_telemetry(
                embedding_enabled=
                    embedding_enabled,
                semantic_score_count=0,
                raw_match_count=0,
                processed_candidate_count=0,
                relevance_rejected=0,
                included=[],
                max_semantic_score=None,
                outcome="empty_query",
            )

            return []

        vector_scores = (
            await self._semantic_scores(
                query
            )
        )

        try:
            matches = (
                await self.bucket_mgr.search(
                    query,
                    limit=max(
                        max_results,
                        20,
                    ),
                    domain_filter=
                        domain_filter,
                    query_valence=
                        query_valence,
                    query_arousal=
                        query_arousal,
                    vector_scores=
                        vector_scores,
                    include_archive=False,
                )
            )

        except TypeError:
            # Existing compatibility fallback.
            matches = (
                await self.bucket_mgr.search(
                    query,
                    limit=max(
                        max_results,
                        20,
                    ),
                    domain_filter=
                        domain_filter,
                    query_valence=
                        query_valence,
                    query_arousal=
                        query_arousal,
                    vector_scores=
                        vector_scores,
                )
            )

        matches = list(
            matches or []
        )

        raw_match_count = sum(
            1
            for item in matches
            if isinstance(
                item,
                dict,
            )
        )

        max_semantic_score = (
            max(
                vector_scores.values()
            )
            if vector_scores
            else None
        )

        results: list[
            dict[str, Any]
        ] = []

        processed_candidate_count = 0
        relevance_rejected = 0

        for bucket in matches:

            if not isinstance(
                bucket,
                dict,
            ):
                continue

            processed_candidate_count += 1

            relevance = (
                self._context_relevance(
                    bucket,
                    vector_scores,
                )
            )

            if (
                relevance
                < self.relevance_threshold
            ):
                relevance_rejected += 1
                continue

            item = dict(bucket)

            item[
                "context_relevance"
            ] = round(
                relevance,
                4,
            )

            results.append(item)

            if (
                len(results)
                >= max_results
            ):
                break

        if results:
            outcome = "included"

        elif raw_match_count == 0:
            outcome = (
                "no_search_matches"
            )

        elif not embedding_enabled:
            outcome = (
                "embedding_disabled"
            )

        elif not vector_scores:
            outcome = (
                "no_semantic_scores"
            )

        elif relevance_rejected:
            outcome = (
                "below_relevance_threshold"
            )

        else:
            outcome = (
                "no_included_results"
            )

        self._store_telemetry(
            embedding_enabled=
                embedding_enabled,
            semantic_score_count=
                len(vector_scores),
            raw_match_count=
                raw_match_count,
            processed_candidate_count=
                processed_candidate_count,
            relevance_rejected=
                relevance_rejected,
            included=
                results,
            max_semantic_score=
                max_semantic_score,
            outcome=
                outcome,
        )

        return results
