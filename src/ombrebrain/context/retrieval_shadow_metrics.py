from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_VERSION = "retrieval-shadow-metrics.v1"
_DEFAULT_ROOT = "/app/buckets/.context"
_WINDOW_SIZE = 100

_LOCK = threading.RLock()

_ALLOWED_OUTCOMES = frozenset({
    "included",
    "no_search_matches",
    "no_semantic_scores",
    "embedding_disabled",
    "below_relevance_threshold",
    "no_included_results",
    "empty_query",
    "unknown",
})


def _root() -> Path:
    return Path(
        (
            os.environ.get(
                "OMBRE_CONTEXT_STATE_DIR"
            )
            or _DEFAULT_ROOT
        ).strip()
    )


def _path() -> Path:
    return (
        _root()
        / "retrieval_shadow_metrics.json"
    )


def _now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _nonnegative_int(
    value: Any,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        return 0

    return max(
        0,
        int(value),
    )


def _sample_key(
    conversation_id: str,
    revision: int,
) -> str:
    payload = (
        f"{conversation_id}:{revision}"
        .encode("utf-8")
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def _read() -> dict[str, Any]:
    p = _path()

    if not p.is_file():
        return {}

    try:
        value = json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}

    return (
        value
        if isinstance(
            value,
            dict,
        )
        else {}
    )


def _atomic_write(
    payload: dict[str, Any],
) -> None:
    p = _path()

    p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        p.parent.chmod(0o700)
    except OSError:
        pass

    temp = p.with_name(
        p.name + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    try:
        temp.chmod(0o600)
    except OSError:
        pass

    os.replace(
        temp,
        p,
    )

    try:
        p.chmod(0o600)
    except OSError:
        pass


def _sample(
    *,
    key: str,
    telemetry: dict[str, Any],
    retrieval_query_used: bool,
) -> dict[str, Any]:

    quality = (
        telemetry.get(
            "retrieval_quality"
        )
        or {}
    )

    if not isinstance(
        quality,
        dict,
    ):
        quality = {}

    shadow = (
        quality.get(
            "decision_shadow"
        )
        or {}
    )

    if not isinstance(
        shadow,
        dict,
    ):
        shadow = {}

    outcome = str(
        quality.get(
            "outcome",
            "unknown",
        )
    )

    if outcome not in _ALLOWED_OUTCOMES:
        outcome = "unknown"

    return {
        # Hash only. No conversation id is persisted.
        "key":
            key,
        "observed_at":
            _now(),
        "retrieval_query_used":
            bool(
                retrieval_query_used
            ),
        "included_count":
            _nonnegative_int(
                quality.get(
                    "included_count"
                )
            ),
        "raw_match_count":
            _nonnegative_int(
                quality.get(
                    "raw_match_count"
                )
            ),
        "relevance_rejected":
            _nonnegative_int(
                telemetry.get(
                    "relevance_rejected"
                )
            ),
        "outcome":
            outcome,
        "would_drop_total":
            _nonnegative_int(
                shadow.get(
                    "would_drop_total"
                )
            ),
        "would_drop_recent_24h":
            _nonnegative_int(
                shadow.get(
                    "would_drop_recent_24h"
                )
            ),
        "would_drop_duplicate_id":
            _nonnegative_int(
                shadow.get(
                    "would_drop_duplicate_id"
                )
            ),
        "would_drop_exact_text_duplicate":
            _nonnegative_int(
                shadow.get(
                    "would_drop_exact_text_duplicate"
                )
            ),
    }


def _summary(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:

    outcomes: dict[str, int] = {}

    retrieved_memories = 0
    raw_matches = 0
    relevance_rejected = 0

    would_drop_total = 0
    drop_recent = 0
    drop_id = 0
    drop_text = 0

    query_used_requests = 0
    hit_requests = 0
    affected_requests = 0

    for sample in samples:
        if not isinstance(
            sample,
            dict,
        ):
            continue

        included = _nonnegative_int(
            sample.get(
                "included_count"
            )
        )

        retrieved_memories += included

        raw_matches += _nonnegative_int(
            sample.get(
                "raw_match_count"
            )
        )

        relevance_rejected += (
            _nonnegative_int(
                sample.get(
                    "relevance_rejected"
                )
            )
        )

        drops = _nonnegative_int(
            sample.get(
                "would_drop_total"
            )
        )

        would_drop_total += drops

        drop_recent += _nonnegative_int(
            sample.get(
                "would_drop_recent_24h"
            )
        )

        drop_id += _nonnegative_int(
            sample.get(
                "would_drop_duplicate_id"
            )
        )

        drop_text += _nonnegative_int(
            sample.get(
                "would_drop_exact_text_duplicate"
            )
        )

        if sample.get(
            "retrieval_query_used"
        ) is True:
            query_used_requests += 1

        if included > 0:
            hit_requests += 1

        if drops > 0:
            affected_requests += 1

        outcome = str(
            sample.get(
                "outcome",
                "unknown",
            )
        )

        if outcome not in _ALLOWED_OUTCOMES:
            outcome = "unknown"

        outcomes[outcome] = (
            outcomes.get(
                outcome,
                0,
            )
            + 1
        )

    drop_rate = (
        (
            would_drop_total
            / retrieved_memories
        )
        * 100.0
        if retrieved_memories > 0
        else 0.0
    )

    return {
        "sample_count":
            len(samples),
        "query_used_requests":
            query_used_requests,
        "retrieval_hit_requests":
            hit_requests,
        "retrieved_memories":
            retrieved_memories,
        "raw_matches":
            raw_matches,
        "relevance_rejected":
            relevance_rejected,
        "shadow_affected_requests":
            affected_requests,
        "shadow_would_drop_total":
            would_drop_total,
        "shadow_would_drop_recent_24h":
            drop_recent,
        "shadow_would_drop_duplicate_id":
            drop_id,
        "shadow_would_drop_exact_text_duplicate":
            drop_text,
        "shadow_drop_rate_percent":
            round(
                drop_rate,
                1,
            ),
        "outcomes":
            outcomes,
    }


def record_retrieval_shadow_metrics(
    *,
    conversation_id: str,
    revision: Any,
    telemetry: dict[str, Any],
    retrieval_query_used: bool,
) -> dict[str, Any]:
    """Record one privacy-safe retrieval shadow sample.

    The sample key is a SHA256 of conversation id + Unified revision.
    Conversation id, query, memory content and bucket ids are never stored.

    Repeated calls for the same Unified revision are idempotent.
    """

    if (
        not isinstance(
            revision,
            int,
        )
        or isinstance(
            revision,
            bool,
        )
        or revision <= 0
    ):
        return {
            "stored": False,
            "reason":
                "invalid_revision",
        }

    key = _sample_key(
        conversation_id,
        revision,
    )

    with _LOCK:
        previous = _read()

        raw_samples = previous.get(
            "samples"
        )

        samples = (
            list(raw_samples)
            if isinstance(
                raw_samples,
                list,
            )
            else []
        )

        duplicate = any(
            isinstance(item, dict)
            and item.get("key")
            == key
            for item in samples
        )

        if not duplicate:
            samples.append(
                _sample(
                    key=key,
                    telemetry=(
                        telemetry
                        if isinstance(
                            telemetry,
                            dict,
                        )
                        else {}
                    ),
                    retrieval_query_used=
                        retrieval_query_used,
                )
            )

            samples = samples[
                -_WINDOW_SIZE:
            ]

        summary = _summary(
            samples
        )

        payload = {
            "version":
                _VERSION,
            "window_size":
                _WINDOW_SIZE,
            "updated_at":
                _now(),
            "samples":
                samples,
            "summary":
                summary,
        }

        # Rewrite on duplicate too, so a previous partial/old-format
        # file is self-healed without double-counting.
        _atomic_write(
            payload
        )

    return {
        "stored": True,
        "duplicate":
            duplicate,
        "window_size":
            _WINDOW_SIZE,
        **summary,
    }


def retrieval_shadow_metrics_status() -> dict[str, Any]:
    value = _read()

    if not value:
        return {
            "exists": False,
        }

    summary = value.get(
        "summary"
    )

    return {
        "exists": True,
        "version":
            value.get(
                "version"
            ),
        "window_size":
            value.get(
                "window_size"
            ),
        **(
            dict(summary)
            if isinstance(
                summary,
                dict,
            )
            else {}
        ),
    }
