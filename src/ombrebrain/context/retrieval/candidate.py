from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

# Context-layer retrieval candidate.
#
# This deliberately differs from the engine-layer
# ombrebrain.retrieval.scoring.RetrievalCandidate:
#   - engine layer wraps the raw bucket Mapping;
#   - this context layer exposes ONLY normalized,
#     storage-independent fields, so downstream
#     shadow scoring can never depend on bucket
#     internals.
#
# The candidate id is used internally for identity
# comparison. It must never be logged or persisted
# in telemetry.

_METADATA_WHITELIST = (
    "type",
    "domain",
    "name",
)

_DEFAULT_IMPORTANCE = 5

# importance is stored as 1..10; normalized to 0..1.
_MIN_IMPORTANCE = 1
_MAX_IMPORTANCE = 10


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0

    if numeric != numeric:  # NaN
        return 0.0

    return max(0.0, min(1.0, numeric))


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None

    try:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _normalized_importance(value: Any) -> float:
    try:
        raw = int(value)
    except (TypeError, ValueError, OverflowError):
        raw = _DEFAULT_IMPORTANCE

    raw = max(_MIN_IMPORTANCE, min(_MAX_IMPORTANCE, raw))

    return (raw - _MIN_IMPORTANCE) / (
        _MAX_IMPORTANCE - _MIN_IMPORTANCE
    )


def _safe_metadata(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}

    safe: dict[str, str] = {}

    for key in _METADATA_WHITELIST:
        item = value.get(key)

        if (
            isinstance(item, str)
            and item.strip()
        ):
            safe[key] = item.strip()

    return safe


@dataclass(frozen=True)
class RetrievalCandidate:
    """Normalized, storage-independent memory candidate.

    Shadow-only value object. Never exposes raw bucket
    content, created_at strings or storage internals.
    """

    id: str
    semantic_score: float = 0.0
    timestamp: datetime | None = None
    importance: float = 0.5
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_bucket(
        cls,
        bucket: Any,
    ) -> "RetrievalCandidate":
        """Normalize one retrieved memory bucket.

        Never raises: any malformed input degrades to a
        neutral candidate instead of breaking the shadow.
        """

        if not isinstance(bucket, Mapping):
            return cls(id="")

        metadata = bucket.get("metadata")
        raw_metadata = (
            metadata
            if isinstance(metadata, Mapping)
            else {}
        )

        timestamp = _parse_timestamp(
            raw_metadata.get("last_active")
        )

        if timestamp is None:
            timestamp = _parse_timestamp(
                bucket.get("created_at")
            )

        return cls(
            id=str(
                bucket.get("id") or ""
            ),
            # The old pipeline attaches its calibrated
            # context_relevance to included results; the
            # shadow builds on that same signal.
            semantic_score=_clamp01(
                bucket.get(
                    "context_relevance"
                )
            ),
            timestamp=timestamp,
            importance=(
                _normalized_importance(
                    bucket.get(
                        "importance",
                        _DEFAULT_IMPORTANCE,
                    )
                )
            ),
            metadata=_safe_metadata(
                raw_metadata
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for tests and local shadow state."""

        return {
            "id": self.id,
            "semantic_score": round(
                self.semantic_score,
                6,
            ),
            "timestamp": (
                self.timestamp.isoformat()
                if self.timestamp
                is not None
                else None
            ),
            "importance": round(
                self.importance,
                6,
            ),
            "metadata": dict(
                self.metadata
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> "RetrievalCandidate":
        if not isinstance(payload, Mapping):
            return cls(id="")

        return cls(
            id=str(
                payload.get("id") or ""
            ),
            semantic_score=_clamp01(
                payload.get(
                    "semantic_score"
                )
            ),
            timestamp=_parse_timestamp(
                payload.get("timestamp")
            ),
            importance=_clamp01(
                payload.get("importance")
            ),
            metadata=_safe_metadata(
                payload.get("metadata")
            ),
        )


def normalize_candidate(
    value: Any,
) -> RetrievalCandidate:
    """Convert one old-retrieval result into a candidate.

    Pure and total: never raises, never ranks, never
    filters, never selects. An already-normalized
    candidate is returned unchanged.
    """

    if isinstance(
        value,
        RetrievalCandidate,
    ):
        return value

    return RetrievalCandidate.from_bucket(
        value
    )
