from __future__ import annotations

from typing import Any, Iterable

from ombrebrain.context.retrieval.candidate import (
    ShadowCandidate,
    normalize_candidate,
)
from ombrebrain.context.retrieval.scorer import (
    ShadowScorer,
    ShadowScoringWeights,
    ShadowScoringContext,
    score_candidate,
)

# Shadow-only ranking for the Retrieval v2 observation pipeline.
#
# RANK ONLY. This module intentionally performs no selection:
#   input candidates
#     -> score
#     -> stable sort
#     -> return ALL ranked candidates
#
# It does NOT do threshold filtering, top_k truncation, admission
# decisions, confidence decisions or token-budget selection.
# Selection stays with the legacy live retrieval path; a future
# selector will be designed together with confidence / token
# budget / anti-echo / source balancing and is deliberately not
# introduced here (problem D).
#
# It never accesses a database and never mutates memory.


def rank_candidates(
    candidates: Iterable[Any],
    *,
    context: ShadowScoringContext | None = None,
    scorer: ShadowScorer | None = None,
    weights: ShadowScoringWeights | None = None,
) -> list[tuple[ShadowCandidate, float]]:
    """Return ALL candidates ranked by descending shadow score.

    Every input candidate is scored and returned: nothing is
    dropped and nothing is truncated. Ties keep their input order
    (stable sort), so identical inputs always produce an identical
    ranking. The input iterable and its items are never mutated.
    """

    normalized = [
        normalize_candidate(item)
        for item in candidates or []
    ]

    active_context = (
        context
        if context is not None
        else ShadowScoringContext()
    )

    def _score(
        candidate: ShadowCandidate,
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

    scored.sort(
        key=lambda pair: pair[1],
        reverse=True,
    )

    return scored


def rank_ids(
    candidates: Iterable[Any],
    *,
    context: ShadowScoringContext | None = None,
    scorer: ShadowScorer | None = None,
) -> list[str]:
    """Ordered candidate ids after ranking. Identity stays local."""

    return [
        candidate.id
        for candidate, _score in rank_candidates(
            candidates,
            context=context,
            scorer=scorer,
        )
    ]


class ShadowRanker:
    """Configured shadow ranker.

    Holds a scorer and delegates to rank_candidates(). Ranking
    only: no filtering, no truncation.
    """

    def __init__(
        self,
        scorer: ShadowScorer | None = None,
    ):
        self.scorer = (
            scorer or ShadowScorer()
        )

    def rank(
        self,
        candidates: Iterable[Any],
        context: ShadowScoringContext,
    ) -> list[tuple[ShadowCandidate, float]]:
        return rank_candidates(
            candidates,
            context=context,
            scorer=self.scorer,
        )
