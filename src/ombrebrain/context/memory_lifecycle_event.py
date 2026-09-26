from __future__ import annotations

import hashlib
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ombrebrain.context.memory_usage_signal import (
    read_memory_usage,
)
from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    is_valid_recall_id,
    now_iso,
    read_json,
    state_root,
    validate_cognitive_request_id,
    validate_conversation_id,
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
# There is exactly ONE authorized write path:
#
#     record_lifecycle_event_from_usage(
#         conversation_id, cognitive_request_id, recall_id, index
#     )
#
# It re-reads the real, persisted ``memory-usage-signal.v1`` artifact
# itself and derives EVERY provenance field -- memory id, timestamp,
# stage, request binding -- from ``usage["events"][index]``. A caller
# can never name a memory, a time or a stage, so a "phantom"
# reinforcement receipt is not representable:
#
#     artifact structurally valid  !=  authorized to persist
#
# That path is ALSO gated by the lifecycle feature flag, read here
# directly (no import of ``memory_reinforcement``, so there is no
# cycle): while the shadow is OFF a direct call refuses with
# ``lifecycle_disabled`` and touches nothing at all. The observer
# keeps its own early flag check as defense in depth.
#
# ``build_lifecycle_event`` stays available as a pure builder for unit
# tests, and ``_persist_verified_lifecycle_event`` stays available as
# the low-level, content-addressed storage primitive, but neither is
# an authorization boundary.
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
#
# Authorization happens at the moment a receipt is written; once
# written, the event store is the history's own source of truth, so a
# later cleanup of an old Usage artifact never erases reinforcement
# provenance.

LIFECYCLE_EVENT_VERSION = (
    "memory-lifecycle-event.v1"
)

MODE = "shadow_only"

# The only lifecycle stage v1 accepts.
STAGE_USED = "used"

# The Usage Signal contract every event must have been derived from.
SOURCE_USAGE_VERSION = "memory-usage-signal.v1"

# The lifecycle feature flag. Defined here as well as in
# ``memory_reinforcement`` so the authorized write path can be gated
# without importing that module (no cycle). Default OFF.
LIFECYCLE_SHADOW_ENV = (
    "OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW"
)

EVENT_ID_RE = re.compile(r"^mlcevt_[0-9a-f]{64}$")

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_REASON_INVALID = "invalid_lifecycle_event"
_REASON_CONFLICT = (
    "existing_lifecycle_event_invalid"
)

# Provenance-gate reasons. Each one is a distinct, structural fact.
_REASON_INVALID_BINDING = "invalid_request_binding"
_REASON_INVALID_INDEX = "invalid_usage_event_index"
_REASON_USAGE_INVALID = (
    "usage_not_found_or_invalid"
)
_REASON_NOT_USED = "source_event_not_used"
_REASON_DISABLED = "lifecycle_disabled"

# Public: the observer classifies this reason separately so a bad
# timestamp is never reported as a storage conflict.
REASON_INVALID_USAGE_EVENT_TIME = (
    "invalid_usage_event_time"
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _lifecycle_shadow_enabled() -> bool:
    """Is the Memory Lifecycle shadow allowed to write at all?

    Default OFF: unset, false, 0 or off all mean "no".
    """

    return _truthy(
        os.environ.get(LIFECYCLE_SHADOW_ENV)
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


def _persist_verified_lifecycle_event(
    artifact: Any,
) -> dict[str, Any]:
    """Low-level, content-addressed storage primitive.

    This is NOT an authorization boundary: it only checks that the
    artifact is internally valid and then writes it once. Only
    ``record_lifecycle_event_from_usage`` -- which derives every
    provenance field from a real, persisted Usage artifact -- may
    treat its output as authorized reinforcement.

    Immutable and idempotent: a valid event already on disk is never
    overwritten and never re-counted, so the same usage event can only
    ever produce one reinforcement receipt. An invalid file that
    already occupies the deterministic path is never deleted and never
    overwritten; the observation fails closed instead.
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


def _no_record(
    reason: str,
    *,
    index: Any = None,
) -> dict[str, Any]:
    return {
        "version": LIFECYCLE_EVENT_VERSION,
        "mode": MODE,
        "stored": False,
        "duplicate": False,
        "decision": "no_record",
        "reason": reason,
        "event_id": None,
        "memory_id": None,
        "memory_key": None,
        "source_usage_event_index": index,
    }


def _valid_event_binding(
    conversation_id: Any,
    cognitive_request_id: Any,
    recall_id: Any,
) -> bool:
    try:
        validate_conversation_id(conversation_id)

        validate_cognitive_request_id(
            cognitive_request_id
        )
    except ValueError:
        return False

    return is_valid_recall_id(recall_id)


def record_lifecycle_event_from_usage(
    conversation_id: Any,
    cognitive_request_id: Any,
    recall_id: Any,
    source_usage_event_index: Any,
) -> dict[str, Any]:
    """The ONLY authorized way to create a reinforcement receipt.

    It reads the real, persisted ``memory-usage-signal.v1`` artifact
    through ``read_memory_usage`` and locates the exact event slot
    ``usage["events"][source_usage_event_index]``. Every provenance
    field -- conversation, request, recall, memory id, stage and
    timestamp -- is taken from THAT persisted event, so the caller can
    name nothing but the slot:

      - lifecycle shadow OFF        -> ``lifecycle_disabled``
      - no Usage artifact           -> ``usage_not_found_or_invalid``
      - invalid / corrupt Usage     -> ``usage_not_found_or_invalid``
      - bad slot                    -> ``invalid_usage_event_index``
      - slot is not ``used``        -> ``source_event_not_used``
      - illegal event timestamp     -> ``invalid_usage_event_time``
      - structurally valid artifact -> the ONLY write path

    The feature flag is checked FIRST, before anything is read or
    written, so a direct call while the shadow is OFF creates no
    event, no event directory and no state.

    Reinforcing a memory that was never explicitly used, at a time the
    Usage artifact never recorded, is therefore not representable.
    """

    if not _lifecycle_shadow_enabled():
        return _no_record(_REASON_DISABLED)

    if not _valid_event_binding(
        conversation_id,
        cognitive_request_id,
        recall_id,
    ):
        return _no_record(_REASON_INVALID_BINDING)

    index = source_usage_event_index

    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
    ):
        return _no_record(
            _REASON_INVALID_INDEX
        )

    usage = read_memory_usage(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if (
        not isinstance(usage, dict)
        or usage.get("version")
        != SOURCE_USAGE_VERSION
    ):
        return _no_record(
            _REASON_USAGE_INVALID, index=index
        )

    events = usage.get("events")

    if (
        not isinstance(events, list)
        or index >= len(events)
    ):
        return _no_record(
            _REASON_INVALID_INDEX, index=index
        )

    source = events[index]

    if not isinstance(source, dict):
        return _no_record(
            _REASON_INVALID_INDEX, index=index
        )

    # Only an explicit ``used`` stage is reinforcement. A
    # ``recall_requested`` / ``memory_loaded`` slot is refused.
    if source.get("stage") != STAGE_USED:
        return _no_record(
            _REASON_NOT_USED, index=index
        )

    memory_id = source.get("memory_id")

    if (
        not isinstance(memory_id, str)
        or not memory_id
        or memory_id != memory_id.strip()
    ):
        # Unreachable for a validated Usage artifact; fail closed.
        return _no_record(
            _REASON_INVALID_INDEX, index=index
        )

    at = source.get("at")

    if parse_iso_utc(at) is None:
        return _no_record(
            REASON_INVALID_USAGE_EVENT_TIME,
            index=index,
        )

    # Fully derived from the persisted artifact, never from the caller.
    artifact = build_lifecycle_event(
        conversation_id=usage.get(
            "conversation_id"
        ),
        cognitive_request_id=usage.get(
            "cognitive_request_id"
        ),
        recall_id=usage.get("recall_id"),
        memory_id=memory_id,
        source_usage_event_at=at,
        source_usage_event_index=index,
    )

    outcome = _persist_verified_lifecycle_event(
        artifact
    )

    return {
        "version": LIFECYCLE_EVENT_VERSION,
        "mode": MODE,
        "stored": bool(outcome.get("stored")),
        "duplicate": bool(
            outcome.get("duplicate")
        ),
        "decision": (
            "recorded"
            if outcome.get("stored")
            else "no_record"
        ),
        "reason": outcome.get("reason"),
        "event_id": artifact.get("event_id"),
        "memory_id": memory_id,
        "memory_key": artifact.get(
            "memory_key"
        ),
        "source_usage_event_index": index,
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