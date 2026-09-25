from __future__ import annotations

from typing import Any, Iterable

from ombrebrain.context.retrieval.candidate import (
    RetrievalCandidate,
)
from ombrebrain.context.retrieval.scorer import (
    RetrievalScorer,
    RetrievalScoringContext,
)

# Shadow-only reranker.
#
# Sorts pre-scored candidates by the shadow score. It never
# mutates the real retrieval result: callers use it only to
# compute what a hypothetical reorder WOULD produce.


class RetrievalReranker:
    """Deterministic, stable reranker.

    Sorting is descending by score; Python's stable sort
    keeps candidates with equal scores in their original
    relative order, so identical inputs always produce an
    identical output ordering.
    """

    def __init__(
        self,
        scorer: RetrievalScorer
        | None = None,
        *,
        top_k: int | None = None,
        score_threshold: float
        | None = None,
    ):
        self.scorer = (
            scorer or RetrievalScorer()
        )
        self.top_k = _validated_top_k(
            top_k
        )
        self.score_threshold = (
            score_threshold
        )

    def rank(
        self,
        candidates: Iterable[Any],
        context: RetrievalScoringContext,
    ) -> list[
        tuple[
            RetrievalCandidate,
            float,
        ]
    ]:
        """Score every candidate and return (candidate, score)
        pairs sorted by descending score."""

        normalized = [
            (
                candidate
                if isinstance(
                    candidate,
                    RetrievalCandidate,
                )
                else RetrievalCandidate
                .from_bucket(
                    candidate
                )
            )
            for candidate in candidates
        ]

        scored = [
            (
                candidate,
                self.scorer.score(
                    candidate,
                    context,
                ),
            )
            for candidate in normalized
        ]

        # Stable sort: equal scores keep input order.
        scored.sort(
            key=lambda pair: pair[1],
            reverse=True,
        )

        return scored

    def rerank(
        self,
        candidates: Iterable[Any],
        context: RetrievalScoringContext,
    ) -> list[RetrievalCandidate]:
        """Sorted candidate list.

        Applies, in order:
          1. descending shadow-score sort (stable)
          2. score-threshold filter (>= threshold)
          3. top_k truncation

        The input iterable is never mutated.
        """

        scored = self.rank(
            candidates,
            context,
        )

        result: list[
            RetrievalCandidate
        ] = []

        for candidate, score in scored:
            if (
                self.score_threshold
                is not None
                and score
                < self.score_threshold
            ):
                continue

            result.append(candidate)

            if (
                self.top_k is not None
                and len(result)
                >= self.top_k
            ):
                break

        return result


def _validated_top_k(
    value: Any,
) -> int | None:
    if value is None:
        return None

    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if numeric < 1:
        return None

    return numeric
