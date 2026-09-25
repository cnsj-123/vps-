from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

# Context-layer shadow projection of a legacy retrieval candidate.
#
# Canonical retrieval domain lives in ``ombrebrain.retrieval``
# (RetrievalCandidate / RetrievalFeatures / PolicyGatedRetrievalScorer).
# This module intentionally does NOT define a second candidate
# domain: it is a read-only *adapter view* used only by the
# Retrieval v2 shadow pipeline, and it is named accordingly.
#
# Signal naming (problem G):
#   semantic_similarity
#       raw embedding similarity, i.e. the engine's vector score.
#       Named to match the upstream RetrievalFeatures contract.
#   legacy_context_relevance
#       the legacy adapter's *calibrated* value (its
#       _context_relevance() mapping of the raw score). This is a
#       different concept and is NOT read here, so the shadow can
#       never mistake one for the other.
#
# The candidate id is used internally for identity comparison.
# It must never be logged or persisted in telemetry.

_METADATA_WHITELIST = (
    "type",
    "domain",
    "name",
)

_DEFAULT_IMPORTANCE = 5

# importance is stored 1..10; normalized to 0..1.
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
class ShadowCandidate:
    """Shadow-only, storage-independent view of a candidate.

    Not a retrieval domain type: the canonical one is
    ``ombrebrain.retrieval.RetrievalCandidate``. This view never
    exposes raw bucket content, created_at strings, storage
    internals or the legacy calibrated relevance.
    """

    id: str
    semantic_similarity: float = 0.0
    timestamp: datetime | None = None
    importance: float = 0.5
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_bucket(
        cls,
        bucket: Any,
        *,
        semantic_similarity: float | None = None,
    ) -> "ShadowCandidate":
        """Project one legacy retrieval candidate.

        ``semantic_similarity`` must be the raw vector score for
        this bucket (from the adapter's vector score map), not the
        legacy calibrated context relevance.

        Never raises: malformed input degrades to a neutral view.
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
            semantic_similarity=_clamp01(
                semantic_similarity
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
            "semantic_similarity": round(
                self.semantic_similarity,
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
    ) -> "ShadowCandidate":
        if not isinstance(payload, Mapping):
            return cls(id="")

        return cls(
            id=str(
                payload.get("id") or ""
            ),
            semantic_similarity=_clamp01(
                payload.get(
                    "semantic_similarity"
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
    *,
    semantic_similarity: float | None = None,
) -> ShadowCandidate:
    """Project one legacy candidate into a shadow view.

    Pure and total: never raises, never ranks, never filters,
    never selects. An already-projected view is returned
    unchanged (its similarity is preserved).
    """

    if isinstance(
        value,
        ShadowCandidate,
    ):
        return value

    return ShadowCandidate.from_bucket(
        value,
        semantic_similarity=(
            semantic_similarity
        ),
    )
