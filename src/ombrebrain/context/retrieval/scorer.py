from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from ombrebrain.retrieval import RetrievalCandidate

from ombrebrain.context.retrieval.candidate import (
    read_active_timestamp,
    read_importance,
)

# Retrieval v2 shadow scoring — **pure functions only**.
#
# This is a shadow *experiment*, not a second scoring framework:
# there is no scorer class, no weights class and no context class.
# The canonical data model is reused (``RetrievalCandidate`` /
# ``RetrievalFeatures``); the canonical ``PolicyGatedRetrievalScorer``
# is deliberately NOT called here.
#
# Fixed formula (weights must not change in this round):
#
#   score =
#       0.5 * semantic_similarity
#     + 0.2 * recency_score
#     + 0.2 * importance_score
#     + 0.1 * relative_semantic_score
#
# ``relative_semantic_score`` is the candidate's raw semantic
# similarity relative to the best candidate in the same set. It is
# derived purely from ``semantic_similarity`` and is NOT an
# independent context signal (it used to be misnamed ``context_match``).
#
# Every function here is pure and total: the same inputs always
# produce the same float, and malformed input degrades to a neutral
# value instead of raising.

_SEMANTIC_WEIGHT = 0.5
_RECENCY_WEIGHT = 0.2
_IMPORTANCE_WEIGHT = 0.2
_RELATIVE_SEMANTIC_WEIGHT = 0.1

_DEFAULT_RECENCY_HALF_LIFE_HOURS = 72.0

# A missing timestamp is neutral: memories without last_active are
# neither promoted nor punished.
_MISSING_TIMESTAMP_RECENCY = 0.5

# Real OB importance lives in metadata["importance"] as 1..10 and is
# normalized to 0..1 via (n - 1) / 9.
_DEFAULT_IMPORTANCE = 5
_MIN_IMPORTANCE = 1
_MAX_IMPORTANCE = 10


def semantic_similarity(
    candidate: Any,
) -> float:
    """Raw semantic similarity from the canonical candidate."""

    if isinstance(
        candidate,
        RetrievalCandidate,
    ):
        return _clamp01(
            candidate.features.semantic_similarity
        )

    return 0.0


def recency_score(
    candidate: Any,
    *,
    now: datetime,
    recency_half_life_hours: float = (
        _DEFAULT_RECENCY_HALF_LIFE_HOURS
    ),
) -> float:
    """Exponential decay with a configurable half-life.

    The timestamp is read from the real OB bucket metadata:
    ``last_active`` first, then ``created``.
    """

    timestamp = read_active_timestamp(
        candidate
    )

    if timestamp is None:
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

    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        now = now.replace(
            tzinfo=timezone.utc
        )

    age_hours = max(
        0.0,
        (now - timestamp).total_seconds()
        / 3600.0,
    )

    return 0.5 ** (age_hours / half_life)


def importance_score(
    candidate: Any,
) -> float:
    """Normalized real OB importance (metadata.importance, 1..10).

    Missing or unparsable importance uses the default 5. A top-level
    bucket ``importance`` never overrides ``metadata.importance``.
    """

    try:
        raw = int(read_importance(candidate))
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        raw = _DEFAULT_IMPORTANCE

    raw = max(
        _MIN_IMPORTANCE,
        min(_MAX_IMPORTANCE, raw),
    )

    return (raw - _MIN_IMPORTANCE) / (
        _MAX_IMPORTANCE - _MIN_IMPORTANCE
    )


def relative_semantic_score(
    candidate: Any,
    *,
    max_semantic_similarity: Any,
) -> float:
    """Candidate similarity relative to the best in the set.

    This is NOT an independent context signal: it is derived purely
    from ``semantic_similarity``. Formerly named ``context_match``.
    """

    best = _clamp01(
        max_semantic_similarity
    )

    if best <= 0.0:
        return 0.0

    return _clamp01(
        semantic_similarity(candidate) / best
    )


def score_candidate(
    candidate: Any,
    *,
    now: datetime,
    max_semantic_similarity: Any,
    recency_half_life_hours: float = (
        _DEFAULT_RECENCY_HALF_LIFE_HOURS
    ),
) -> float:
    """Fixed shadow score for one canonical candidate. 0..1.

    Pure: never modifies the candidate, never raises for malformed
    input.
    """

    combined = (
        _SEMANTIC_WEIGHT
        * semantic_similarity(candidate)
        + _RECENCY_WEIGHT
        * recency_score(
            candidate,
            now=now,
            recency_half_life_hours=(
                recency_half_life_hours
            ),
        )
        + _IMPORTANCE_WEIGHT
        * importance_score(candidate)
        + _RELATIVE_SEMANTIC_WEIGHT
        * relative_semantic_score(
            candidate,
            max_semantic_similarity=(
                max_semantic_similarity
            ),
        )
    )

    return _clamp01(combined)


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
