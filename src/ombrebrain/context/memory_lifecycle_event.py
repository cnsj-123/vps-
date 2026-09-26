from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    is_valid_recall_id,
    now_iso,
    read_json,
    state_root,
)


# Memory Lifecycle Event v1 — immutable reinforcement evidence.
#
#     retrieved       -> reinforcement  NO
#     surfaced_as_flash -> reinforcement  NO
#     recall_requested -> reinforcement  NO
#     memory_loaded   -> reinforcement  NO
#     used            -> reinforcement  YES
#
# This module is the event store of the Memory Lifecycle shadow: the
# single immutable receipt that one ``used`` Memory Usage Signal
# produced. It answers exactly one question -- "which explicit,
# validated usages happened to this memory?" -- and it never touches
# a canonical memory source, never touches Retrieval, never touches
# the Surfacing Policy and never calls a model.
#
# Only the ``used`` stage exists here. ``retrieved`` / ``surfaced`` /
# ``loaded`` are deliberately NOT representable: a lifecycle event is
# reinforcement evidence, and none of those three stages is
# reinforcement.
#
# Events are append-only and content-addressed:
#
#   - ``event_id`` is a deterministic digest of the usage provenance,
#     so reprocessing the same usage event 100 times yields exactly
#     one lifecycle event;
#   - an event file is never rewritten, never merged and never
#     deleted, so the history stays recomputable forever;
#   - a reader re-derives the identity from the artifact itself, so a
#     correct path never implies a correct content.

LIFECYCLE_EVENT_VERSION = (
    "memory-lifecycle-event.v1"
)

MODE = "shadow_only"

# The only lifecycle stage v1 accepts.
STAGE_USED = "used"

# The Usage Signal contract every event must have been derived from.
SOURCE_USAGE_VERSION = "memory-usage-signal.v1"

EVENT_ID_RE = re.compile(r"^mlcevt_[0-9a-f]{64}$")

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_REASON_INVALID = "invalid_lifecycle_event"
_REASON_CONFLICT = (
    "existing_lifecycle_event_invalid"
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def parse_iso_utc(
    value: Any,
) -> datetime | None:
    """Strict, timezone-aware ISO-8601 parsing.

    Only a timezone-aware timestamp is accepted; a naive string is
    rejected. A malformed usage timestamp is never silently replaced
    by "now", because a wrong time would poison the decay formulas.
    """

    if not isinstance(value, str):
        return None

    text = value.strip()

    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except ValueError:
        return None

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        return None

    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def iso_utc(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def memory_key(memory_id: Any) -> str | None:
    """Stable directory key for one memory id.

    A raw memory id is never used as a path component: it may come
    from external data. Only the sha256 hex digest is used, and the
    real id is kept inside the artifact.
    """

    if (
        not isinstance(memory_id, str)
        or not memory_id
        or memory_id != memory_id.strip()
    ):
        return None

    return hashlib.sha256(
        memory_id.encode("utf-8")
    ).hexdigest()


def lifecycle_events_dir(
    key: Any,
) -> Path | None:
    if (
        not isinstance(key, str)
        or not _HEX64_RE.fullmatch(key)
    ):
        return None

    return (
        state_root()
        / "memory_lifecycle_events"
        / key
    )


def lifecycle_event_path(
    key: Any,
    event_id: Any,
) -> Path | None:
    directory = lifecycle_events_dir(key)

    if directory is None:
        return None

    if (
        not isinstance(event_id, str)
        or not EVENT_ID_RE.fullmatch(event_id)
    ):
        return None

    return directory / (event_id + ".json")


def derive_lifecycle_event_id(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
    memory_id: str,
    stage: str,
    source_usage_event_at: str,
    source_usage_event_index: int,
) -> str:
    """Deterministic identity of one reinforcement observation.

    The digest covers the full usage provenance plus the exact usage
    event slot, so the same persisted usage event always folds onto
    the same lifecycle event id -- no random uuid is ever the
    idempotency key.
    """

    payload = "\0".join(
        [
            str(conversation_id or ""),
            str(cognitive_request_id or ""),
            str(recall_id or ""),
            str(memory_id or ""),
            str(stage or ""),
            str(source_usage_event_at or ""),
            str(source_usage_event_index),
        ]
    )

    return "mlcevt_" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def build_lifecycle_event(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
    memory_id: str,
    source_usage_event_at: str,
    source_usage_event_index: int,
) -> dict[str, Any]:
    """Build (do not persist) one immutable lifecycle event."""

    key = memory_key(memory_id)

    if key is None:
        raise ValueError("invalid memory_id")

    event_id = derive_lifecycle_event_id(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
        memory_id=memory_id,
        stage=STAGE_USED,
        source_usage_event_at=(
            source_usage_event_at
        ),
        source_usage_event_index=(
            source_usage_event_index
        ),
    )

    return {
        "version":
            LIFECYCLE_EVENT_VERSION,
        "mode": MODE,
        "event_id": event_id,
        "stage": STAGE_USED,
        "memory_id": memory_id,
        "memory_key": key,
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "recall_id": recall_id,
        "source_usage_version":
            SOURCE_USAGE_VERSION,
        "source_usage_event_index":
            source_usage_event_index,
        "source_usage_event_at":
            source_usage_event_at,
        "created_at": now_iso(),
    }


def is_valid_lifecycle_event(
    artifact: Any,
    *,
    key: Any = None,
    event_id: Any = None,
    memory_id: Any = None,
) -> bool:
    """Structural identity of one lifecycle event.

    The artifact's own identity is RECOMPUTED -- memory key and event
    id are re-derived from the provenance it carries -- so a tampered
    memory id, request binding, stage, index or timestamp is rejected
    and never counted.
    """

    if not isinstance(artifact, dict):
        return False

    if (
        artifact.get("version")
        != LIFECYCLE_EVENT_VERSION
        or artifact.get("mode") != MODE
    ):
        return False

    if artifact.get("stage") != STAGE_USED:
        return False

    artifact_memory_id = artifact.get(
        "memory_id"
    )

    derived_key = memory_key(artifact_memory_id)

    if derived_key is None:
        return False

    if artifact.get("memory_key") != derived_key:
        return False

    if key is not None and derived_key != key:
        return False

    if (
        memory_id is not None
        and artifact_memory_id != memory_id
    ):
        return False

    conversation_id = artifact.get(
        "conversation_id"
    )

    cognitive_request_id = artifact.get(
        "cognitive_request_id"
    )

    if not is_valid_conversation_id(
        conversation_id
    ):
        return False

    if not is_valid_cognitive_request_id(
        cognitive_request_id
    ):
        return False

    if not is_valid_recall_id(
        artifact.get("recall_id")
    ):
        return False

    if (
        artifact.get("source_usage_version")
        != SOURCE_USAGE_VERSION
    ):
        return False

    index = artifact.get(
        "source_usage_event_index"
    )

    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
    ):
        return False

    source_at = artifact.get(
        "source_usage_event_at"
    )

    if parse_iso_utc(source_at) is None:
        return False

    if parse_iso_utc(
        artifact.get("created_at")
    ) is None:
        return False

    artifact_event_id = artifact.get("event_id")

    if (
        not isinstance(artifact_event_id, str)
        or not EVENT_ID_RE.fullmatch(
            artifact_event_id
        )
    ):
        return False

    expected = derive_lifecycle_event_id(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=artifact.get("recall_id"),
        memory_id=artifact_memory_id,
        stage=STAGE_USED,
        source_usage_event_at=source_at,
        source_usage_event_index=index,
    )

    if artifact_event_id != expected:
        return False

    if (
        event_id is not None
        and artifact_event_id != event_id
    ):
        return False

    return True


def persist_lifecycle_event(
    artifact: Any,
) -> dict[str, Any]:
    """Persist one lifecycle event. Immutable and idempotent.

    A valid event already on disk is never overwritten and never
    re-counted -- the same usage event can only ever produce one
    reinforcement receipt. An invalid file that already occupies the
    deterministic path is never deleted and never overwritten: the
    observation fails closed instead.
    """

    if not is_valid_lifecycle_event(artifact):
        return {
            "stored": False,
            "duplicate": False,
            "reason": _REASON_INVALID,
        }

    path = lifecycle_event_path(
        artifact["memory_key"],
        artifact["event_id"],
    )

    if path is None:
        return {
            "stored": False,
            "duplicate": False,
            "reason": _REASON_INVALID,
        }

    with LOCK:
        existing = read_json(path)

        if existing is not None:
            if is_valid_lifecycle_event(
                existing,
                key=artifact["memory_key"],
                event_id=artifact["event_id"],
                memory_id=artifact["memory_id"],
            ):
                return {
                    "stored": True,
                    "duplicate": True,
                    "reason":
                        "duplicate_lifecycle_event",
                }

            return {
                "stored": False,
                "duplicate": False,
                "reason": _REASON_CONFLICT,
            }

        atomic_write(path, artifact)

    return {
        "stored": True,
        "duplicate": False,
        "reason": "lifecycle_event_recorded",
    }


def read_lifecycle_events(
    memory_id: Any,
) -> dict[str, Any]:
    """All readable events of one memory, identity re-validated.

    Only ``memory_lifecycle_events/<memory_key>/`` is scanned -- never
    the whole event store -- and every file is validated against its
    own path (``path.stem == event_id``) and its own provenance. A
    corrupt event is ignored, counted and left in place; it is never
    deleted and never counted as a use.
    """

    empty = {
        "events": [],
        "event_file_count": 0,
        "invalid_event_count": 0,
    }

    key = memory_key(memory_id)

    if key is None:
        return empty

    directory = lifecycle_events_dir(key)

    if directory is None or not directory.is_dir():
        return empty

    events: list[dict[str, Any]] = []
    file_count = 0
    invalid_count = 0

    for path in sorted(directory.glob("*.json")):
        file_count += 1

        artifact = read_json(path)

        if not is_valid_lifecycle_event(
            artifact,
            key=key,
            memory_id=memory_id,
        ):
            invalid_count += 1
            continue

        # The filename must agree with the artifact's own identity:
        # path correctness never implies content correctness.
        if path.stem != artifact.get("event_id"):
            invalid_count += 1
            continue

        events.append(artifact)

    return {
        "events": events,
        "event_file_count": file_count,
        "invalid_event_count": invalid_count,
    }


def lifecycle_use_facts(
    events: list[dict[str, Any]],
    *,
    as_of: datetime,
) -> dict[str, Any]:
    """Derive the time facts a state computation needs.

    Events may land in any order, so ``last_used_at`` is the maximum
    real usage timestamp -- never a filename ordering artifact.
    Future-dated events (small clock drift) count as a real use but
    contribute zero age, so decay can never go negative.
    """

    use_count = len(events)

    last_used = None
    future_event_count = 0

    for artifact in events:
        at = parse_iso_utc(
            artifact.get(
                "source_usage_event_at"
            )
        )

        if at is None:
            continue

        if at > as_of:
            future_event_count += 1

        if last_used is None or at > last_used:
            last_used = at

    return {
        "use_count": use_count,
        "last_used_at": last_used,
        "future_event_count":
            future_event_count,
    }


def event_age_hours(
    *,
    at: datetime | None,
    as_of: datetime,
) -> float:
    """Non-negative age in hours. Never negative, never NaN."""

    if at is None:
        return 0.0

    seconds = (as_of - at).total_seconds()

    if not math.isfinite(seconds) or seconds <= 0:
        return 0.0

    return seconds / 3600.0