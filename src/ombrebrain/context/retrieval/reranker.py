from __future__ import annotations

from typing import Any, Iterable

from ombrebrain.context.retrieval.candidate import (
    RetrievalCandidate,
    normalize_candidate,
)
from ombrebrain.context.retrieval.scorer import (
    RetrievalScorer,
    RetrievalScorerWeights,
    RetrievalScoringContext,
    score_candidate,
)

# Shadow-only reranker.
#
# Pipeline: candidate list -> score -> sort -> top_k.
#
# It never accesses a database, never mutates a memory and
# never changes the real retrieval result: callers use it
# only to compute what a hypothetical reorder WOULD produce.


def rerank_candidates(
    candidates: Iterable[Any],
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    context: RetrievalScoringContext | None = None,
    scorer: RetrievalScorer | None = None,
    weights: RetrievalScorerWeights | None = None,
) -> list[RetrievalCandidate]:
    """Return a NEW sorted candidate list.

    Steps, in order:
      1. normalize each input (pure, never raises)
      2. score with the shadow scorer
      3. stable descending sort by score
      4. score_threshold filter (>= threshold)
      5. top_k truncation

    The input iterable and its items are never mutated.
    """

    normalized = [
        normalize_candidate(item)
        for item in candidates or []
    ]

    active_context = (
        context
        if context is not None
        else RetrievalScoringContext()
    )

    def _score(
        candidate: RetrievalCandidate,
    ) -> float:
        if scorer is not None:
            return scorer.score(
                candidate,
                active_context,
            )

        return score_candidate(
            candidate,
            active_context,
            weights=weights,
        )

    scored = [
        (candidate, _score(candidate))
        for candidate in normalized
    ]

    # Stable sort: equal scores keep input order, so the
    # same inputs always produce the same ordering.
    scored.sort(
        key=lambda pair: pair[1],
        reverse=True,
    )

    limit = _validated_top_k(top_k)

    result: list[RetrievalCandidate] = []

    for candidate, score in scored:
        if (
            score_threshold is not None
            and score < score_threshold
        ):
            continue

        result.append(candidate)

        if limit is not None and len(result) >= limit:
            break

    return result


class RetrievalReranker:
    """Configured reranker.

    Holds top_k / threshold / scorer and delegates to
    rerank_candidates(), so a caller can either inject
    configuration or call the function directly.
    """

    def __init__(
        self,
        scorer: RetrievalScorer | None = None,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ):
        self.scorer = (
            scorer or RetrievalScorer()
        )
        self.top_k = _validated_top_k(top_k)
        self.score_threshold = score_threshold

    def rank(
        self,
        candidates: Iterable[Any],
        context: RetrievalScoringContext,
    ) -> list[tuple[RetrievalCandidate, float]]:
        """Score every candidate and return (candidate, score)
        pairs sorted by descending score."""

        normalized = [
            normalize_candidate(item)
            for item in candidates or []
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
        return rerank_candidates(
            candidates,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            context=context,
            scorer=self.scorer,
        )


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
