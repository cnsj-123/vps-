from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from ombrebrain.retrieval import RetrievalCandidate

from ombrebrain.context.retrieval.scorer import (
    score_candidate,
)

# Retrieval v2 shadow ranking — **rank only**.
#
#   input candidates
#     -> score
#     -> stable sort
#     -> return ALL ranked candidates
#
# It does NOT do threshold filtering, top_k truncation, admission
# decisions, confidence decisions or token-budget selection. There is
# no ranker class and no selector: selection stays with the legacy
# live retrieval path, and a future selector will be designed together
# with confidence / token budget / anti-echo / source balancing.
#
# Input and output are the canonical retrieval domain type
# (``ombrebrain.retrieval.RetrievalCandidate``). The shadow adds no
# candidate domain of its own.

_DEFAULT_RECENCY_HALF_LIFE_HOURS = 72.0

__all__ = ["rank_candidates"]


def rank_candidates(
    candidates: Iterable[Any],
    *,
    now: datetime,
    max_semantic_similarity: Any,
    recency_half_life_hours: float = (
        _DEFAULT_RECENCY_HALF_LIFE_HOURS
    ),
) -> list[tuple[RetrievalCandidate, float]]:
    """Score and order every canonical candidate. Nothing is dropped.

    Ties keep their input order (stable sort), so identical inputs
    always produce an identical ranking. The input iterable and its
    items are never mutated.
    """

    items = list(candidates or [])

    scored = [
        (
            candidate,
            score_candidate(
                candidate,
                now=now,
                max_semantic_similarity=(
                    max_semantic_similarity
                ),
                recency_half_life_hours=(
                    recency_half_life_hours
                ),
            ),
        )
        for candidate in items
    ]

    scored.sort(
        key=lambda pair: pair[1],
        reverse=True,
    )

    return scored
