from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ombrebrain.context.retrieval.candidate import (
    RetrievalCandidate,
)

# Shadow-only retrieval scorer (context layer).
#
# score =
#   0.5 * semantic_score
# + 0.2 * recency_score
# + 0.2 * importance_score
# + 0.1 * context_match_score
#
# All weights are configurable through RetrievalScorerWeights.
# Defaults match the initial formula and never modify the
# real retrieval path: this scorer is used only by the
# shadow pipeline.


@dataclass(frozen=True)
class RetrievalScorerWeights:
    """Configurable score weights.

    Defaults implement the initial Retrieval v2 formula and
    sum to 1.0. No environment flags are involved.
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


class RetrievalScorer:
    """Deterministic shadow scorer.

    Pure function of (candidate, context): the same inputs
    always produce the same float. Never raises for
    malformed candidates.
    """

    def __init__(
        self,
        weights: RetrievalScorerWeights
        | None = None,
        *,
        recency_half_life_hours: float = 72.0,
    ):
        self.weights = (
            weights
            or RetrievalScorerWeights()
        )
        self.recency_half_life_hours = (
            max(
                1.0,
                float(
                    recency_half_life_hours
                ),
            )
        )

    def recency_score(
        self,
        candidate: RetrievalCandidate,
        context: RetrievalScoringContext,
    ) -> float:
        """Exponential decay with a configurable half-life.

        age 0      -> 1.0
        half-life  -> 0.5
        older      -> decays toward 0.0

        A missing timestamp is neutral (0.5): memories
        without last_active are neither promoted nor
        punished by the shadow.
        """

        if candidate.timestamp is None:
            return 0.5

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

        return 0.5 ** (
            age_hours
            / self.recency_half_life_hours
        )

    def importance_score(
        self,
        candidate: RetrievalCandidate,
    ) -> float:
        # Already normalized to 0..1 by the candidate.
        return _clamp01(
            candidate.importance
        )

    def context_match_score(
        self,
        candidate: RetrievalCandidate,
        context: RetrievalScoringContext,
    ) -> float:
        """Relative match against the current query context.

        Initial definition: how close this candidate's
        semantic score is to the best candidate in the
        same set. Future phases (confidence gate,
        feedback) can supply a richer value through the
        context without changing this contract.
        """

        best = _clamp01(
            context.max_semantic_score
        )

        if best <= 0.0:
            return 0.0

        return _clamp01(
            candidate.semantic_score
            / best
        )

    def score(
        self,
        candidate: Any,
        context: RetrievalScoringContext,
    ) -> float:
        """Unified scoring entry point. Returns a 0..1 float."""

        if not isinstance(
            candidate,
            RetrievalCandidate,
        ):
            candidate = (
                RetrievalCandidate
                .from_bucket(
                    candidate
                )
            )

        w = self.weights

        combined = (
            _clamp01(w.semantic)
            * _clamp01(
                candidate.semantic_score
            )
            + _clamp01(w.recency)
            * self.recency_score(
                candidate,
                context,
            )
            + _clamp01(w.importance)
            * self.importance_score(
                candidate
            )
            + _clamp01(w.context_match)
            * self.context_match_score(
                candidate,
                context,
            )
        )

        return _clamp01(combined)


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0

    if not math.isfinite(numeric):
        return 0.0

    return max(0.0, min(1.0, numeric))
