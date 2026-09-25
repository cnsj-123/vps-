from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from ombrebrain.retrieval import (
    RetrievalCandidate,
    RetrievalFeatures,
)

# Context-layer adapter onto the **canonical** retrieval domain.
#
# Canonical candidate / features / scoring / policy live in
# ``ombrebrain.retrieval``. This module deliberately defines no
# second candidate domain: it only *projects* one legacy
# Ombre-Brain memory bucket into a canonical ``RetrievalCandidate``
# for the Retrieval v2 shadow.
#
# Real Ombre-Brain memory bucket shape — this is the only shape read
# here:
#
#   {
#       "id": "...",
#       "content": "...",
#       "metadata": {
#           "importance": 1..10,
#           "last_active": "<iso>",
#           "created": "<iso>",
#           ...
#       },
#   }
#
# importance / last_active / created live inside ``metadata``. A
# top-level ``importance`` or ``created_at`` is NOT the real OB
# bucket contract and is never used as a normal path.
#
# timestamp / importance are deliberately NOT candidate fields: when
# scoring needs them it reads them through the readers below, which
# are the single place that knows the real OB bucket shape.

_SOURCE = "context_retrieval_v2_shadow"


def bucket_metadata(value: Any) -> Mapping[str, Any]:
    """Return the real OB ``metadata`` mapping of a bucket/candidate.

    Accepts either a raw bucket mapping or a canonical
    ``RetrievalCandidate``. Pure and total: anything unexpected
    degrades to an empty mapping.
    """

    bucket: Any = value

    if isinstance(value, RetrievalCandidate):
        bucket = value.bucket

    if not isinstance(bucket, Mapping):
        return {}

    metadata = bucket.get("metadata")

    if not isinstance(metadata, Mapping):
        return {}

    return metadata


def read_importance(value: Any) -> Any:
    """Real OB importance, read from ``metadata.importance`` only.

    A top-level ``importance`` is not the OB contract and must not
    override the metadata value.
    """

    return bucket_metadata(value).get("importance")


def read_active_timestamp(
    value: Any,
) -> datetime | None:
    """Last activity as a UTC datetime.

    ``metadata.last_active`` first; if it is missing, empty or
    unparsable, fall back to ``metadata.created``. Top-level
    ``created_at`` is never used.
    """

    metadata = bucket_metadata(value)

    last_active = parse_timestamp(
        metadata.get("last_active")
    )

    if last_active is not None:
        return last_active

    return parse_timestamp(
        metadata.get("created")
    )


def parse_timestamp(
    value: Any,
) -> datetime | None:
    """Parse one ISO timestamp into an aware UTC datetime.

    Pure and total: anything unparsable returns ``None``.
    """

    if not value:
        return None

    try:
        text = str(value).strip().replace(
            "Z",
            "+00:00",
        )
        parsed = datetime.fromisoformat(
            text
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(
            timezone.utc
        )
    except Exception:
        return None


def project_candidate(
    bucket: Any,
    *,
    semantic_similarity: Any = None,
) -> RetrievalCandidate:
    """Project one legacy OB memory bucket into the canonical domain.

    ``semantic_similarity`` must be the raw embedding similarity for
    this bucket (the adapter's vector score map), never the legacy
    calibrated context relevance.

    The bucket mapping is held by reference — it is not copied and no
    content is extracted here. Privacy is enforced at the
    telemetry/output boundary, not at this internal boundary.

    Pure and total: never raises, never ranks, never filters, never
    selects.
    """

    if isinstance(bucket, RetrievalCandidate):
        safe_bucket: Mapping[str, Any] = bucket.bucket
    elif isinstance(bucket, Mapping):
        safe_bucket = bucket
    else:
        safe_bucket = {}

    return RetrievalCandidate(
        bucket=safe_bucket,
        features=RetrievalFeatures(
            semantic_similarity=_clamp01(
                semantic_similarity
            ),
        ),
        source=_SOURCE,
    )


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0

    if numeric != numeric:  # NaN
        return 0.0

    return max(0.0, min(1.0, numeric))
