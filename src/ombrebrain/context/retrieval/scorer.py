from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ombrebrain.context.retrieval.candidate import (
    RetrievalCandidate,
    normalize_candidate,
)

# Shadow-only retrieval scorer (context layer).
#
# Fixed formula:
#   score =
#     0.5 * semantic_score
#   + 0.2 * recency_score
#   + 0.2 * importance_score
#   + 0.1 * context_match_score
#
# All weights live in RetrievalScorerWeights and default to
# the formula above. This module is a pure function layer:
# it never mutates a candidate and never touches the real
# retrieval path.

_DEFAULT_RECENCY_HALF_LIFE_HOURS = 72.0

# A missing timestamp is neutral: memories without
# last_active are neither promoted nor punished.
_MISSING_TIMESTAMP_RECENCY = 0.5


@dataclass(frozen=True)
class RetrievalScorerWeights:
    """Configurable score weights.

    Defaults implement the fixed formula and sum to 1.0.
    """

    semantic: float = 0.5
    recency: float = 0.2
    importance: float = 0.2
    context_match: float = 0.1

    def to_dict(self) -> dict[str, float]:
        return {
            "semantic": float(self.semantic),
            "recency": float(self.recency),
            "importance": float(self.importance),
            "context_match": float(self.context_match),
        }


@dataclass(frozen=True)
class RetrievalScoringContext:
    """Per-request scoring context.

    Carries only what scoring needs. It never contains the
    query, conversation id or any raw text.
    """

    now: datetime = field(
        default_factory=lambda: (
            datetime.now(timezone.utc)
        )
    )

    # Best semantic score in the current candidate set.
    # Used for the relative context-match component.
    max_semantic_score: float = 0.0


def recency_score(
    candidate: Any,
    context: RetrievalScoringContext,
    *,
    recency_half_life_hours: float = (
        _DEFAULT_RECENCY_HALF_LIFE_HOURS
    ),
) -> float:
    """Exponential decay with a configurable half-life.

    age 0      -> 1.0
    half-life  -> 0.5
    older      -> decays toward 0.0
    """

    candidate = normalize_candidate(
        candidate
    )

    if candidate.timestamp is None:
        return _MISSING_TIMESTAMP_RECENCY

    half_life = max(
        1.0,
        _float(
            recency_half_life_hours,
            default=(
                _DEFAULT_RECENCY_HALF_LIFE_HOURS
            ),
        ),
    )

    now = context.now

    if now.tzinfo is None:
        now = now.replace(
            tzinfo=timezone.utc
        )

    age_hours = max(
        0.0,
        (
            now - candidate.timestamp
        ).total_seconds()
        / 3600.0,
    )

    return 0.5 ** (age_hours / half_life)


def importance_score(
    candidate: Any,
) -> float:
    """Normalized importance. Already 0..1 on the candidate."""

    candidate = normalize_candidate(
        candidate
    )

    return _clamp01(candidate.importance)


def context_match_score(
    candidate: Any,
    context: RetrievalScoringContext,
) -> float:
    """Relative match against the current query context.

    Initial definition: how close this candidate's
    semantic score is to the best candidate in the same
    set. Future phases (confidence gate, feedback) can
    supply a richer value through the context without
    changing this contract.
    """

    candidate = normalize_candidate(
        candidate
    )

    best = _clamp01(
        context.max_semantic_score
    )

    if best <= 0.0:
        return 0.0

    return _clamp01(
        candidate.semantic_score / best
    )


def score_candidate(
    candidate: Any,
    context: RetrievalScoringContext,
    *,
    weights: RetrievalScorerWeights
    | None = None,
    recency_half_life_hours: float = (
        _DEFAULT_RECENCY_HALF_LIFE_HOURS
    ),
) -> float:
    """Unified scoring entry point. Returns a 0..1 float.

    Pure function of (candidate, context, weights): the
    same inputs always produce the same float. It never
    modifies the candidate and never raises for malformed
    input.
    """

    candidate = normalize_candidate(
        candidate
    )

    active = weights or RetrievalScorerWeights()

    combined = (
        _clamp01(active.semantic)
        * _clamp01(candidate.semantic_score)
        + _clamp01(active.recency)
        * recency_score(
            candidate,
            context,
            recency_half_life_hours=(
                recency_half_life_hours
            ),
        )
        + _clamp01(active.importance)
        * importance_score(candidate)
        + _clamp01(active.context_match)
        * context_match_score(
            candidate,
            context,
        )
    )

    return _clamp01(combined)


class RetrievalScorer:
    """Configured scorer.

    Holds weights / half-life and delegates every
    computation to the module-level pure functions, so a
    caller can either inject configuration or call the
    functions directly.
    """

    def __init__(
        self,
        weights: RetrievalScorerWeights
        | None = None,
        *,
        recency_half_life_hours: float = (
            _DEFAULT_RECENCY_HALF_LIFE_HOURS
        ),
    ):
        self.weights = (
            weights
            or RetrievalScorerWeights()
        )
        self.recency_half_life_hours = (
            recency_half_life_hours
        )

    def recency_score(
        self,
        candidate: Any,
        context: RetrievalScoringContext,
    ) -> float:
        return recency_score(
            candidate,
            context,
            recency_half_life_hours=(
                self.recency_half_life_hours
            ),
        )

    def importance_score(
        self,
        candidate: Any,
    ) -> float:
        return importance_score(candidate)

    def context_match_score(
        self,
        candidate: Any,
        context: RetrievalScoringContext,
    ) -> float:
        return context_match_score(
            candidate,
            context,
        )

    def score(
        self,
        candidate: Any,
        context: RetrievalScoringContext,
    ) -> float:
        return score_candidate(
            candidate,
            context,
            weights=self.weights,
            recency_half_life_hours=(
                self.recency_half_life_hours
            ),
        )


def _float(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if not math.isfinite(numeric):
        return default

    return numeric


def _clamp01(value: Any) -> float:
    return max(
        0.0,
        min(1.0, _float(value)),
    )
