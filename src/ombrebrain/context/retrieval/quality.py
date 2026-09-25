from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from ombrebrain.retrieval import RetrievalCandidate

from ombrebrain.context.retrieval.candidate import (
    project_candidate,
)
from ombrebrain.context.retrieval.reranker import (
    rank_candidates,
)

# Retrieval v2 quality shadow — observability only.
#
# This is NOT part of the retrieval domain: it is a privacy-safe
# telemetry observer that reuses the canonical candidate model
# (``ombrebrain.retrieval.RetrievalCandidate``) plus the shadow
# project / rank helpers. It never participates in selection: the
# legacy live result is the source of truth and is never modified.
#
# Metric semantics — names must not describe behaviour that does not
# exist:
#   candidate_count        candidates the V2 shadow observed
#                          (the pre-selection pool, not the live result)
#   ranked_count           candidates V2 successfully scored/ranked
#   legacy_selected_count  what the legacy live path finally returned
#   order_changed          whether V2 ranking of the legacy-selected
#                          set differs from the legacy ordering
#   ranking_delta          ordering-only delta (no add/drop claim)

_VERSION = "retrieval-quality-shadow.v2"
_MODE = "shadow_only"

LOG_TAG = "[gateway.retrieval_quality_v2]"

# Same logging channel as the other gateway.* shadow tags so
# operators collect them uniformly.
_logger = logging.getLogger(
    "ombre_brain.gateway"
)

_SCORE_BUCKET_WIDTH = 0.2
_SCORE_BUCKET_COUNT = 5

# Distribution buckets, low-inclusive:
#   0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
_SCORE_BUCKET_LABELS = tuple(
    f"{i * _SCORE_BUCKET_WIDTH:.1f}"
    f"-{(i + 1) * _SCORE_BUCKET_WIDTH:.1f}"
    for i in range(_SCORE_BUCKET_COUNT)
)

# Canonical fields of a privacy-safe quality event.
# Exported so tests can assert the contract without
# duplicating the list.
PRIVACY_SAFE_FIELDS = (
    "candidate_count",
    "ranked_count",
    "legacy_selected_count",
    "top_score",
    "average_score",
    "score_distribution",
    "order_changed",
    "shadow_only",
)


def _empty_distribution() -> dict[str, int]:
    return {
        label: 0
        for label in _SCORE_BUCKET_LABELS
    }


def _distribution(
    scores: Iterable[float],
) -> dict[str, int]:
    buckets = _empty_distribution()

    for score in scores:
        try:
            numeric = float(score)
        except (TypeError, ValueError, OverflowError):
            continue

        if numeric != numeric:
            continue

        index = min(
            _SCORE_BUCKET_COUNT - 1,
            max(
                0,
                int(
                    numeric
                    / _SCORE_BUCKET_WIDTH
                ),
            ),
        )

        buckets[
            _SCORE_BUCKET_LABELS[index]
        ] += 1

    return buckets


def _round4(value: float) -> float:
    return round(float(value), 4)


def _candidate_id(value: Any) -> str:
    if isinstance(
        value,
        RetrievalCandidate,
    ):
        return str(
            value.bucket.get("id") or ""
        )

    if isinstance(value, Mapping):
        return str(value.get("id") or "")

    return ""


def _semantic_from(views: list[Any]) -> float:
    return max(
        (
            view.features.semantic_similarity
            for view in views
        ),
        default=0.0,
    )


class RetrievalQualityShadow:
    """Shadow statistics observer for the new retrieval path.

    Privacy contract — neither the event nor the log line ever
    contains: query, memory content, conversation_id or bucket_id.
    Only counts, score statistics and the shadow decision leave
    this module.
    """

    def observe(
        self,
        candidates: Iterable[Any],
        *,
        vector_scores: Mapping[str, Any]
        | None = None,
        legacy_selected: Iterable[Any] = (),
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Compare the pre-selection pool with the legacy live result.

        ``candidates`` is the pre-selection candidate pool: the shadow
        observes the same candidates the legacy path saw *before* its
        own selection, not the legacy-selected subset.

        ``vector_scores`` maps bucket id -> raw embedding similarity
        for those candidates. It is the raw signal; the legacy
        calibrated context relevance is deliberately not used here.

        ``legacy_selected`` is the legacy live result, used only to
        measure whether V2 ordering would differ. Neither input is
        modified.
        """

        scores_map = (
            dict(vector_scores)
            if isinstance(
                vector_scores,
                Mapping,
            )
            else {}
        )

        views = [
            project_candidate(
                candidate,
                semantic_similarity=(
                    scores_map.get(
                        _candidate_id(
                            candidate
                        )
                    )
                ),
            )
            for candidate in candidates or []
        ]

        ranked = rank_candidates(
            views,
            now=(
                now
                or datetime.now(
                    timezone.utc
                )
            ),
            max_semantic_similarity=(
                _semantic_from(views)
            ),
        )

        scores = [
            score
            for _candidate, score in ranked
        ]

        ranked_ids = [
            _candidate_id(candidate)
            for candidate, _score in ranked
        ]

        legacy_ids = [
            _candidate_id(item)
            for item in legacy_selected or []
        ]

        order_changed, ranking_delta = (
            _ordering_delta(
                legacy_ids,
                ranked_ids,
            )
        )

        return {
            "version": _VERSION,
            "mode": _MODE,
            "candidate_count": len(views),
            "ranked_count": len(ranked),
            "legacy_selected_count": len(
                legacy_ids
            ),
            "top_score": (
                _round4(max(scores))
                if scores
                else None
            ),
            "average_score": (
                _round4(
                    sum(scores) / len(scores)
                )
                if scores
                else None
            ),
            "score_distribution": (
                _distribution(scores)
            ),
            "order_changed": order_changed,
            "shadow_only": True,
            "ranking_delta": ranking_delta,
        }

    def emit(
        self,
        report: dict[str, Any],
    ) -> None:
        """Log one privacy-safe shadow event."""

        _logger.info(
            "%s %s",
            LOG_TAG,
            json.dumps(
                report,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    @classmethod
    def observe_failed(
        cls,
        reason: str = "observe_failed",
    ) -> dict[str, Any]:
        """Fail-open event used when observe() raises.

        Keeps the shadow contract: never breaks the real retrieval
        path, never leaks anything.
        """

        return {
            "version": _VERSION,
            "mode": _MODE,
            "candidate_count": 0,
            "ranked_count": 0,
            "legacy_selected_count": 0,
            "top_score": None,
            "average_score": None,
            "score_distribution": (
                _empty_distribution()
            ),
            "order_changed": False,
            "shadow_only": True,
            "reason": reason,
            "ranking_delta": {
                "overlap_count": 0,
                "reordered_count": 0,
            },
        }


def _ordering_delta(
    legacy_ids: list[str],
    ranked_ids: list[str],
) -> tuple[bool, dict[str, int]]:
    """Ordering-only delta between legacy order and V2 order.

    Restricted to ids present in both, so it never claims that V2
    added or dropped a candidate.
    """

    ranked_set = set(ranked_ids)

    legacy_overlap = [
        value
        for value in legacy_ids
        if value in ranked_set
    ]

    legacy_set = set(legacy_ids)

    v2_overlap = [
        value
        for value in ranked_ids
        if value in legacy_set
    ]

    common = set(legacy_overlap) & set(
        v2_overlap
    )

    legacy_index = {
        value: index
        for index, value in enumerate(
            legacy_overlap
        )
    }

    v2_index = {
        value: index
        for index, value in enumerate(
            v2_overlap
        )
    }

    reordered = sum(
        1
        for value in common
        if legacy_index.get(value)
        != v2_index.get(value)
    )

    return (
        legacy_overlap != v2_overlap,
        {
            "overlap_count": len(common),
            "reordered_count": reordered,
        },
    )
