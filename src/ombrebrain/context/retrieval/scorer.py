from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ombrebrain.context.retrieval.candidate import (
    normalize_candidate,
)

# Shadow-only scoring for the Retrieval v2 observation pipeline.
#
# Canonical scoring domain lives in ``ombrebrain.retrieval``
# (RetrievalFeatures.candidate_score / PolicyGatedRetrievalScorer).
# This module is a small, fixed shadow formula; it is NOT a second
# scoring framework and is named to make that explicit.
#
# Formula (unchanged weights):
#   score =
#     0.5 * semantic_similarity
#   + 0.2 * recency_score
#   + 0.2 * importance_score
#   + 0.1 * relative_semantic
#
# ``relative_semantic`` is the candidate's raw semantic similarity
# relative to the best candidate in the same set. It was previously
# misnamed ``context_match``, which implied an independent context
# signal. It is not one, so it is named for what it actually is
# (problem F). Weights are unchanged and still sum to 1.0.

_DEFAULT_RECENCY_HALF_LIFE_HOURS = 72.0

# A missing timestamp is neutral: memories without
# last_active are neither promoted nor punished.
_MISSING_TIMESTAMP_RECENCY = 0.5


@dataclass(frozen=True)
class ShadowScoringWeights:
    """Fixed shadow weights. Defaults sum to 1.0."""

    semantic: float = 0.5
    recency: float = 0.2
    importance: float = 0.2
    relative_semantic: float = 0.1

    def to_dict(self) -> dict[str, float]:
        return {
            "semantic": float(self.semantic),
            "recency": float(self.recency),
            "importance": float(self.importance),
            "relative_semantic": float(
                self.relative_semantic
            ),
        }


@dataclass(frozen=True)
class ShadowScoringContext:
    """Per-request scoring context.

    Carries only what scoring needs. It never contains the
    query, conversation id or any raw text.
    """

    now: datetime = field(
        default_factory=lambda: (
            datetime.now(timezone.utc)
        )
    )

    # Best raw semantic similarity in the observed set.
    max_semantic_similarity: float = 0.0


def recency_score(
    candidate: Any,
    context: ShadowScoringContext,
    *,
    recency_half_life_hours: float = (
        _DEFAULT_RECENCY_HALF_LIFE_HOURS
    ),
) -> float:
    """Exponential decay with a configurable half-life."""

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
    """Normalized importance. Already 0..1 on the view."""

    candidate = normalize_candidate(
        candidate
    )

    return _clamp01(candidate.importance)


def relative_semantic_score(
    candidate: Any,
    context: ShadowScoringContext,
) -> float:
    """Candidate similarity relative to the best in the set.

    This is NOT an independent context signal: it is derived
    purely from ``semantic_similarity``. Formerly named
    ``context_match_score``.
    """

    candidate = normalize_candidate(
        candidate
    )

    best = _clamp01(
        context.max_semantic_similarity
    )

    if best <= 0.0:
        return 0.0

    return _clamp01(
        candidate.semantic_similarity / best
    )


def score_candidate(
    candidate: Any,
    context: ShadowScoringContext,
    *,
    weights: ShadowScoringWeights
    | None = None,
    recency_half_life_hours: float = (
        _DEFAULT_RECENCY_HALF_LIFE_HOURS
    ),
) -> float:
    """Unified shadow scoring entry point. Returns a 0..1 float.

    Pure function of (candidate, context, weights): the same
    inputs always produce the same float. It never modifies the
    candidate and never raises for malformed input.
    """

    candidate = normalize_candidate(
        candidate
    )

    active = weights or ShadowScoringWeights()

    combined = (
        _clamp01(active.semantic)
        * _clamp01(candidate.semantic_similarity)
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
        + _clamp01(active.relative_semantic)
        * relative_semantic_score(
            candidate,
            context,
        )
    )

    return _clamp01(combined)


class ShadowScorer:
    """Configured shadow scorer.

    Holds weights / half-life and delegates every computation to
    the module-level pure functions.
    """

    def __init__(
        self,
        weights: ShadowScoringWeights
        | None = None,
        *,
        recency_half_life_hours: float = (
            _DEFAULT_RECENCY_HALF_LIFE_HOURS
        ),
    ):
        self.weights = (
            weights or ShadowScoringWeights()
        )
        self.recency_half_life_hours = (
            recency_half_life_hours
        )

    def recency_score(
        self,
        candidate: Any,
        context: ShadowScoringContext,
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

    def relative_semantic_score(
        self,
        candidate: Any,
        context: ShadowScoringContext,
    ) -> float:
        return relative_semantic_score(
            candidate,
            context,
        )

    def score(
        self,
        candidate: Any,
        context: ShadowScoringContext,
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
