from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from ombrebrain.context.retrieval.candidate import (
    RetrievalCandidate,
)
from ombrebrain.context.retrieval.reranker import (
    RetrievalReranker,
)
from ombrebrain.context.retrieval.scorer import (
    RetrievalScorer,
    RetrievalScoringContext,
)

_VERSION = "retrieval-quality-shadow.v1"
_MODE = "shadow_only"

LOG_TAG = "[gateway.retrieval_quality_v2]"

# Same logging channel as the other gateway.* shadow tags so
# operators collect them uniformly. The gateway module itself
# stays untouched.
_logger = logging.getLogger(
    "ombre_brain.gateway"
)

_SCORE_BUCKET_WIDTH = 0.2
_SCORE_BUCKET_COUNT = 5

# Distribution buckets, low-inclusive:
#   0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
_SCORE_BUCKET_LABELS = tuple(
    f"{i * _SCORE_BUCKET_WIDTH:.1f}"
    f"-{(i + 1) * _SCORE_BUCKET_WIDTH:.1f}"
    for i in range(_SCORE_BUCKET_COUNT)
)


def _empty_distribution() -> (
    dict[str, int]
):
    return {
        label: 0
        for label in _SCORE_BUCKET_LABELS
    }


def _distribution(
    scores: Iterable[float],
) -> dict[str, int]:
    buckets = _empty_distribution()

    for score in scores:
        try:
            numeric = float(score)
        except (TypeError, ValueError, OverflowError):
            continue

        if numeric != numeric:
            continue

        index = min(
            _SCORE_BUCKET_COUNT - 1,
            max(
                0,
                int(
                    numeric
                    / _SCORE_BUCKET_WIDTH
                ),
            ),
        )

        buckets[
            _SCORE_BUCKET_LABELS[index]
        ] += 1

    return buckets


def _round4(value: float) -> float:
    return round(float(value), 4)


class RetrievalQualityShadow:
    """Retrieval v2 shadow observer.

    Runs the new score + rerank pipeline next to the old
    retrieval result and reports what WOULD change.

    Privacy contract — the report and the log line never
    contain: the query, memory content, conversation_id
    or bucket_id. Only counts, score statistics and the
    shadow decision.
    """

    def __init__(
        self,
        scorer: RetrievalScorer
        | None = None,
        reranker: RetrievalReranker
        | None = None,
    ):
        self.scorer = (
            scorer or RetrievalScorer()
        )
        self.reranker = (
            reranker
            or RetrievalReranker(
                scorer=self.scorer
            )
        )

    def observe(
        self,
        old_candidates: Iterable[Any],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Compare old retrieval output with the shadow rerank.

        ``old_candidates`` are the raw result dicts produced by
        the existing retrieval path (with context_relevance).
        """

        old_normalized = [
            candidate
            for candidate in (
                RetrievalCandidate.from_bucket(
                    item
                )
                if not isinstance(
                    item,
                    RetrievalCandidate,
                )
                else item
                for item in old_candidates
                or []
            )
        ]

        context = RetrievalScoringContext(
            now=(
                now
                or datetime.now(
                    timezone.utc
                )
            ),
            max_semantic_score=(
                max(
                    (
                        candidate
                        .semantic_score
                        for candidate in old_normalized
                    ),
                    default=0.0,
                )
            ),
        )

        scored = self.reranker.rank(
            old_normalized,
            context,
        )

        scores = [
            score
            for _candidate, score in scored
        ]

        # Identity comparison stays internal: ids are used
        # only to compute the delta counts below.
        old_order = [
            candidate.id
            for candidate in old_normalized
        ]

        new_order = [
            candidate.id
            for candidate, _score in scored
        ]

        would_change = (
            old_order != new_order
        )

        return {
            "version": _VERSION,
            "mode": _MODE,
            "candidate_count": len(
                old_normalized
            ),
            "selected_count": len(new_order),
            "score_distribution": (
                _distribution(scores)
            ),
            "top_score": (
                _round4(max(scores))
                if scores
                else None
            ),
            "average_score": (
                _round4(
                    sum(scores)
                    / len(scores)
                )
                if scores
                else None
            ),
            "would_change": would_change,
            "shadow_only": True,
            "selection_delta": (
                _selection_delta(
                    old_order,
                    new_order,
                )
            ),
        }

    def emit(
        self,
        report: dict[str, Any],
    ) -> None:
        """Log one privacy-safe shadow report."""

        _logger.info(
            "%s %s",
            LOG_TAG,
            json.dumps(
                report,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    @classmethod
    def observe_failed(
        cls,
        reason: str = "observe_failed",
    ) -> dict[str, Any]:
        """Fail-open report used when observe() raises.

        Keeps the shadow contract: never breaks the real
        retrieval path, never leaks anything.
        """

        return {
            "version": _VERSION,
            "mode": _MODE,
            "candidate_count": 0,
            "selected_count": 0,
            "score_distribution": (
                _empty_distribution()
            ),
            "top_score": None,
            "average_score": None,
            "would_change": False,
            "shadow_only": True,
            "reason": reason,
            "selection_delta": {
                "overlap_count": 0,
                "added_count": 0,
                "dropped_count": 0,
                "reordered_count": 0,
            },
        }


def _selection_delta(
    old_order: list[str],
    new_order: list[str],
) -> dict[str, int]:
    old_ids = set(old_order)
    new_ids = set(new_order)

    old_index = {
        value: index
        for index, value in enumerate(
            old_order
        )
    }

    new_index = {
        value: index
        for index, value in enumerate(
            new_order
        )
    }

    common = old_ids & new_ids

    reordered = sum(
        1
        for value in common
        if old_index.get(value)
        != new_index.get(value)
    )

    return {
        "overlap_count": len(common),
        "added_count": len(
            new_ids - old_ids
        ),
        "dropped_count": len(
            old_ids - new_ids
        ),
        "reordered_count": reordered,
    }
