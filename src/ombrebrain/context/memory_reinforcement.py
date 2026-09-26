from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from ombrebrain.context.memory_lifecycle_event import (
    MODE,
    build_lifecycle_event,
    memory_key,
    parse_iso_utc,
    persist_lifecycle_event,
    read_lifecycle_events,
)
from ombrebrain.context.memory_lifecycle_state import (
    STATE_VERSION,
    build_lifecycle_state,
    persist_lifecycle_state,
    resolve_lifecycle_config,
)
from ombrebrain.context.memory_usage_signal import (
    read_memory_usage,
)
from ombrebrain.context.recall_types import (
    is_valid_recall_id,
    validate_cognitive_request_id,
    validate_conversation_id,
)
from ombrebrain.context.unified_context_candidate import (
    bound_context_service,
)


# Memory Reinforcement / Lifecycle Observer v1 — shadow only.
#
#     used -> immutable reinforcement evidence
#          -> long-term strength increases
#
#     time passes -> accessibility decreases
#                -> salience decreases
#
#     but the memory still EXISTS.
#
# The ONLY reinforcement source is a persisted, validated Memory Usage
# Signal whose stage is exactly ``used``. Nothing here infers usage
# from response text, keyword matching, semantic similarity or an LLM
# judge, and nothing here treats ``retrieved`` / ``surfaced_as_flash``
# / ``recall_requested`` / ``memory_loaded`` as reinforcement.
#
# The observer reads the usage artifact itself -- a caller can never
# hand it a raw ``{"memory_id": ..., "used": True}`` dict.
#
# Everything this phase produces lives under ``OMBRE_CONTEXT_STATE_DIR``
# as lifecycle events and lifecycle state. A canonical memory source is
# read-only (``BucketManager.get`` only): the observer NEVER calls
# ``touch`` / ``archive`` / ``update`` / ``delete`` / ``create``, never
# writes ``activation_count`` / ``last_active`` / ``importance``, never
# changes Retrieval, never changes the Surfacing Policy and never runs
# a background decay job. Decay is derived at ``as_of``.
#
# The observer is an explicit internal callable. It is NOT wired into
# the Gateway and NOT run by ``run_context_pipeline()``, and it is
# default OFF.

_REPORT_VERSION = (
    "memory-lifecycle-observation.v1"
)

_ENV_LIFECYCLE_SHADOW = (
    "OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW"
)

_STAGE_USED = "used"

_REASON_DISABLED = "lifecycle_disabled"
_REASON_INVALID_BINDING = (
    "invalid_request_binding"
)
_REASON_INVALID_MEMORY_ID = (
    "invalid_memory_id"
)
_REASON_INVALID_AS_OF = "invalid_as_of"
_REASON_USAGE_INVALID = (
    "usage_not_found_or_invalid"
)
_REASON_NO_USED = "no_used_events"
_REASON_INVALID_EVENT_TIME = (
    "invalid_usage_event_time"
)
_REASON_EVENT_CONFLICT = (
    "lifecycle_event_conflict"
)
_REASON_OBSERVED = "lifecycle_observed"
_REASON_DERIVED = "lifecycle_state_derived"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def memory_lifecycle_shadow_enabled() -> bool:
    """Is the Memory Lifecycle shadow allowed to run at all?

    Default OFF. Even an explicit internal call is a no-op while the
    flag is off, so a rollout can never create lifecycle events or
    state by accident.
    """

    return _truthy(
        os.environ.get(
            _ENV_LIFECYCLE_SHADOW
        )
    )


def _empty_report(
    reason: str,
    *,
    stored: bool = False,
    decision: str = "no_observation",
) -> dict[str, Any]:
    return {
        "version": _REPORT_VERSION,
        "mode": MODE,
        "stored": stored,
        "decision": decision,
        "reason": reason,
        "used_event_count": 0,
        "new_event_count": 0,
        "duplicate_event_count": 0,
        "invalid_event_count": 0,
        "state_updated_count": 0,
        "exists_count": 0,
        "low_accessibility_count": 0,
    }


def _resolve_as_of(
    as_of: Any,
) -> datetime | None:
    """``None`` -> now UTC. A fixed, timezone-aware value otherwise.

    Tests inject a fixed ``as_of``; no test ever sleeps.
    """

    if as_of is None:
        return datetime.now(timezone.utc)

    if isinstance(as_of, datetime):
        if (
            as_of.tzinfo is None
            or as_of.utcoffset() is None
        ):
            return None

        return as_of.astimezone(timezone.utc)

    return parse_iso_utc(as_of)


def _resolve_bucket_manager(
    bucket_manager: Any,
) -> Any:
    """Reuse the bound canonical repository. Never creates one."""

    if bucket_manager is not None:
        return bucket_manager

    service = bound_context_service()

    if service is None:
        return None

    return getattr(service, "bucket_mgr", None)


async def _canonical_source(
    bucket_manager: Any,
    memory_id: str,
) -> dict[str, Any]:
    """Read-only existence / created check of one canonical memory."""

    result = {
        "exists": False,
        "checked": False,
        "created": None,
    }

    getter = getattr(
        bucket_manager, "get", None
    )

    if not callable(getter):
        return result

    try:
        bucket = await getter(memory_id)
    except Exception:
        bucket = None

    result["checked"] = True

    if not isinstance(bucket, dict):
        return result

    result["exists"] = True

    metadata = bucket.get("metadata")

    if isinstance(metadata, dict):
        created = metadata.get("created")

        if (
            isinstance(created, str)
            and created.strip()
        ):
            result["created"] = created.strip()

    return result


async def _derive_state(
    memory_id: str,
    *,
    bucket_manager: Any,
    as_of: datetime,
    persist: bool,
    config: dict[str, float],
) -> tuple[dict[str, Any], bool]:
    """Recompute one state snapshot from its events, then cache it.

    The snapshot is always re-derived -- a corrupt snapshot is never
    used as an incremental base -- and a failed state write still
    returns the freshly computed result.
    """

    canonical = await _canonical_source(
        bucket_manager, memory_id
    )

    store = read_lifecycle_events(memory_id)

    state = build_lifecycle_state(
        memory_id=memory_id,
        events=store["events"],
        as_of=as_of,
        config=config,
        exists=canonical["exists"],
        source_exists_checked=canonical[
            "checked"
        ],
        source_created_at=canonical["created"],
        event_file_count=store[
            "event_file_count"
        ],
        invalid_event_count=store[
            "invalid_event_count"
        ],
    )

    if not persist:
        return state, False

    try:
        written = persist_lifecycle_state(state)
    except Exception:
        return state, False

    return state, bool(written.get("stored"))


async def derive_memory_lifecycle_state(
    memory_id: Any,
    *,
    bucket_manager: Any = None,
    as_of: Any = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Derive one memory's lifecycle state from its event receipts.

    ``as_of`` is injectable: production passes ``None`` (now UTC) and
    tests pass a fixed instant, so decay can be verified without ever
    waiting. The state returned is a derived cache; the events remain
    the source of truth.
    """

    if not memory_lifecycle_shadow_enabled():
        return _refused_state(_REASON_DISABLED)

    if memory_key(memory_id) is None:
        return _refused_state(
            _REASON_INVALID_MEMORY_ID
        )

    resolved_as_of = _resolve_as_of(as_of)

    if resolved_as_of is None:
        return _refused_state(
            _REASON_INVALID_AS_OF
        )

    resolved_bucket = _resolve_bucket_manager(
        bucket_manager
    )

    state, stored = await _derive_state(
        memory_id,
        bucket_manager=resolved_bucket,
        as_of=resolved_as_of,
        persist=persist,
        config=resolve_lifecycle_config(),
    )

    return {
        "version": STATE_VERSION,
        "mode": MODE,
        "decision": "derived",
        "reason": _REASON_DERIVED,
        "state_stored": stored,
        "state": state,
    }


def _refused_state(
    reason: str,
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "mode": MODE,
        "decision": "refused",
        "reason": reason,
        "state_stored": False,
        "state": None,
    }


def _used_events(
    artifact: dict[str, Any],
) -> tuple[list[tuple[int, str, str]], int]:
    """Extract only the explicit ``used`` stage from a Usage artifact.

    ``retrieved`` / ``surfaced`` / ``recall_requested`` /
    ``memory_loaded`` are skipped by construction. An event with an
    illegal timestamp is rejected -- a bad time would poison decay --
    and is never replaced by "now".
    """

    used: list[tuple[int, str, str]] = []
    invalid = 0

    events = artifact.get("events")

    if not isinstance(events, list):
        return used, invalid

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue

        if event.get("stage") != _STAGE_USED:
            continue

        memory_id = event.get("memory_id")

        if (
            not isinstance(memory_id, str)
            or not memory_id
            or memory_id != memory_id.strip()
        ):
            invalid += 1
            continue

        at = event.get("at")

        if parse_iso_utc(at) is None:
            invalid += 1
            continue

        used.append((index, memory_id, at))

    return used, invalid


async def observe_memory_usage_lifecycle(
    conversation_id: Any,
    cognitive_request_id: Any,
    recall_id: Any,
    *,
    bucket_manager: Any = None,
    as_of: Any = None,
) -> dict[str, Any]:
    """Turn one request's persisted Usage Signal into lifecycle state.

    The usage artifact is read from disk and re-validated here -- the
    caller can never pass raw usage data. For each explicit ``used``
    event a deterministic, immutable lifecycle event is written (once,
    ever); then each affected memory's state is recomputed from ALL of
    its events and cached. A memory that was loaded but never used
    gets no reinforcement at all.
    """

    if not memory_lifecycle_shadow_enabled():
        return _empty_report(_REASON_DISABLED)

    try:
        validate_conversation_id(
            conversation_id
        )

        validate_cognitive_request_id(
            cognitive_request_id
        )
    except ValueError:
        return _empty_report(
            _REASON_INVALID_BINDING
        )

    if not is_valid_recall_id(recall_id):
        return _empty_report(
            _REASON_INVALID_BINDING
        )

    resolved_as_of = _resolve_as_of(as_of)

    if resolved_as_of is None:
        return _empty_report(
            _REASON_INVALID_AS_OF
        )

    usage = read_memory_usage(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if not isinstance(usage, dict):
        return _empty_report(
            _REASON_USAGE_INVALID
        )

    used, invalid = _used_events(usage)

    report = _empty_report(_REASON_NO_USED)

    report["used_event_count"] = len(used)

    resolved_bucket = _resolve_bucket_manager(
        bucket_manager
    )

    affected: list[str] = []

    for index, memory_id, at in used:
        result = persist_lifecycle_event(
            build_lifecycle_event(
                conversation_id=(
                    conversation_id
                ),
                cognitive_request_id=(
                    cognitive_request_id
                ),
                recall_id=recall_id,
                memory_id=memory_id,
                source_usage_event_at=at,
                source_usage_event_index=index,
            )
        )

        if not result.get("stored"):
            invalid += 1
            continue

        if result.get("duplicate"):
            report[
                "duplicate_event_count"
            ] += 1
        else:
            report[
                "new_event_count"
            ] += 1

        if memory_id not in affected:
            affected.append(memory_id)

    report["invalid_event_count"] = invalid

    _update_decision(
        report,
        used_count=len(used),
        invalid=invalid,
    )

    if not affected:
        return report

    config = resolve_lifecycle_config()

    for memory_id in affected:
        state, stored = await _derive_state(
            memory_id,
            bucket_manager=resolved_bucket,
            as_of=resolved_as_of,
            persist=True,
            config=config,
        )

        if stored:
            report["state_updated_count"] += 1

        if state.get("exists"):
            report["exists_count"] += 1

        if state.get("low_accessibility"):
            report[
                "low_accessibility_count"
            ] += 1

    return report


def _update_decision(
    report: dict[str, Any],
    *,
    used_count: int,
    invalid: int,
) -> None:
    recorded = (
        report["new_event_count"]
        + report["duplicate_event_count"]
    )

    if recorded > 0:
        report["stored"] = True
        report["decision"] = "observed"
        report["reason"] = _REASON_OBSERVED
        return

    if invalid > 0:
        report["reason"] = (
            _REASON_INVALID_EVENT_TIME
            if used_count == 0
            else _REASON_EVENT_CONFLICT
        )


async def process_completed_usage_for_lifecycle(
    conversation_id: Any,
    cognitive_request_id: Any,
    recall_id: Any,
    *,
    bucket_manager: Any = None,
    as_of: Any = None,
) -> dict[str, Any]:
    """Thin coordinator hook for a future integration.

    Provided only so a later phase can decide WHERE a completed usage
    should be observed. It is never run automatically, is not wired
    into the Gateway or the request pipeline, and affects no request
    body, no streaming and no cache.
    """

    return await observe_memory_usage_lifecycle(
        conversation_id,
        cognitive_request_id,
        recall_id,
        bucket_manager=bucket_manager,
        as_of=as_of,
    )