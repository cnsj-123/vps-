from __future__ import annotations

from typing import Any

from ombrebrain.context.recall_request import (
    read_recall_request,
)
from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    is_valid_recall_id,
    memory_usage_path,
    now_iso,
    read_json,
    validate_cognitive_request_id,
    validate_conversation_id,
)


# Memory Usage Signal v1 — observation only.
#
#     recall_requested != memory_loaded
#     memory_loaded    != used
#     used             != reinforced
#
# This module records exactly three event stages, through three
# separate APIs so the three states can never be conflated:
#
#     record_recall_requested()  the AI explicitly asked for an anchor
#     record_memory_loaded()     a persisted Recall result was returned
#     record_usage()             the AI explicitly declared usage
#
# ``recall_requested`` is independent of loading: a Recall Request
# that was authorized and then could not be satisfied still records
# ``recall_requested`` with zero loaded memories.
#
# ``used`` is NEVER inferred. There is no keyword matching, no answer
# text scanning and no heuristic "the model probably used M17". A
# usage event exists only when an explicit model / tool control signal
# reaches ``record_usage()``.
#
# ``loaded`` is NEVER ``used``: every memory a persisted Recall result
# returned is recorded as loaded, and ``used_memory_ids`` stays empty
# until an explicit signal arrives.
#
# ``memory_loaded`` is NEVER taken from a retrieval candidate: it is
# derived from a persisted Related Recall artifact, so only memories
# that were really returned can ever be loaded.
#
# Nothing here reinforces anything. There is no activation_count, no
# last_active, no importance, no strength, no weight, no score, no
# decay, no archive and no salience write -- not to a memory source
# file and not to an in-memory memory object. A usage artifact only
# carries internal ids, stage names and timestamps, so a future phase
# can compute reinforcement from it. Memory content is never copied
# into it.

_VERSION = "memory-usage-signal.v1"
_MODE = "shadow_only"

_STAGE_RECALL_REQUESTED = "recall_requested"
_STAGE_MEMORY_LOADED = "memory_loaded"
_STAGE_USED = "used"

_REASON_MEMORY_NOT_LOADED = "memory_not_loaded"
_REASON_INVALID_ID = "invalid_memory_id"


def _no_record(
    reason: str,
) -> dict[str, Any]:
    return {
        "stored": False,
        "mode": _MODE,
        "decision": "no_record",
        "reason": reason,
        "duplicate": False,
        "loaded_count": 0,
        "used_count": 0,
    }


def _valid_id_list(value: Any) -> bool:
    """A list of unique, non-empty string memory ids."""

    if not isinstance(value, list):
        return False

    seen: set[str] = set()

    for item in value:
        if (
            not isinstance(item, str)
            or not item.strip()
        ):
            return False

        if item in seen:
            return False

        seen.add(item)

    return True


def is_valid_usage_artifact(
    artifact: Any,
    *,
    conversation_id: Any = None,
    cognitive_request_id: Any = None,
    recall_id: Any = None,
) -> bool:
    """Structural identity of one persisted Usage Signal artifact.

    A correct path proves nothing, so a persisted artifact is only
    trusted after re-validating the contract, the mode, the full
    request binding, the id-list shapes, the anchor binding and the
    invariant ``used ⊆ loaded``.

    The event log and the state arrays must agree EXACTLY:

      - exactly one ``recall_requested`` event, naming the anchor;
      - ``memory_loaded`` events' unique ids == ``loaded_memory_ids``;
      - ``used`` events' unique ids == ``used_memory_ids``.

    A malformed or inconsistent artifact is never merged into.
    """

    if not isinstance(artifact, dict):
        return False

    if (
        artifact.get("version") != _VERSION
        or artifact.get("mode") != _MODE
    ):
        return False

    artifact_conversation = artifact.get(
        "conversation_id"
    )

    artifact_request = artifact.get(
        "cognitive_request_id"
    )

    artifact_recall = artifact.get("recall_id")

    if (
        conversation_id is not None
        and artifact_conversation
        != conversation_id
    ):
        return False

    if (
        cognitive_request_id is not None
        and artifact_request
        != cognitive_request_id
    ):
        return False

    if (
        recall_id is not None
        and artifact_recall != recall_id
    ):
        return False

    try:
        validate_conversation_id(
            artifact_conversation
        )

        validate_cognitive_request_id(
            artifact_request
        )
    except ValueError:
        return False

    if not is_valid_recall_id(artifact_recall):
        return False

    anchor = artifact.get("anchor_memory_id")

    if (
        not isinstance(anchor, str)
        or not anchor.strip()
    ):
        return False

    anchor = anchor.strip()

    loaded = artifact.get("loaded_memory_ids")

    used = artifact.get("used_memory_ids")

    if not _valid_id_list(loaded):
        return False

    if not _valid_id_list(used):
        return False

    if not set(used).issubset(set(loaded)):
        return False

    if artifact.get("loaded_count") != len(loaded):
        return False

    if artifact.get("used_count") != len(used):
        return False

    events = artifact.get("events")

    if not isinstance(events, list) or not events:
        return False

    requested_ids: list[str] = []
    loaded_event_ids: list[str] = []
    used_event_ids: list[str] = []

    for event in events:
        if not isinstance(event, dict):
            return False

        stage = event.get("stage")

        memory_id = event.get("memory_id")

        if stage not in (
            _STAGE_RECALL_REQUESTED,
            _STAGE_MEMORY_LOADED,
            _STAGE_USED,
        ):
            return False

        if (
            not isinstance(memory_id, str)
            or not memory_id.strip()
        ):
            return False

        memory_id = memory_id.strip()

        if stage == _STAGE_RECALL_REQUESTED:
            requested_ids.append(memory_id)
        elif stage == _STAGE_MEMORY_LOADED:
            loaded_event_ids.append(memory_id)
        else:
            used_event_ids.append(memory_id)

    # Exactly one recall_requested event, and it must name the anchor.
    # Two requested events, or a requested event bound to another
    # memory, mean the artifact cannot be trusted.
    if len(requested_ids) != 1:
        return False

    if requested_ids[0] != anchor:
        return False

    # The event log must agree with the state arrays exactly -- and
    # neither side may contain a duplicate.
    if len(set(loaded_event_ids)) != len(
        loaded_event_ids
    ):
        return False

    if set(loaded_event_ids) != set(loaded):
        return False

    if len(set(used_event_ids)) != len(
        used_event_ids
    ):
        return False

    return set(used_event_ids) == set(used)


def _loaded_ids(
    recall_artifact: dict[str, Any],
) -> list[str]:
    loaded: list[str] = []
    seen: set[str] = set()

    for item in (
        recall_artifact.get("memories")
        or []
    ):
        if not isinstance(item, dict):
            continue

        memory_id = item.get("memory_id")

        if (
            not isinstance(memory_id, str)
            or not memory_id.strip()
        ):
            continue

        memory_id = memory_id.strip()

        if memory_id in seen:
            continue

        seen.add(memory_id)
        loaded.append(memory_id)

    return loaded


def _base_artifact(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
    anchor_memory_id: str,
) -> dict[str, Any]:
    """Requested-only state: one event, zero loaded, zero used."""

    requested_at = now_iso()

    events = [
        {
            "stage": _STAGE_RECALL_REQUESTED,
            "memory_id": anchor_memory_id,
            "at": requested_at,
        }
    ]

    return {
        "version": _VERSION,
        "mode": _MODE,
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "recall_id": recall_id,
        "anchor_memory_id": anchor_memory_id,
        "requested_at": requested_at,
        "loaded_memory_ids": [],
        "used_memory_ids": [],
        "loaded_count": 0,
        "used_count": 0,
        "events": events,
    }


def _summary(
    artifact: dict[str, Any],
    *,
    duplicate: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "stored": True,
        "mode": _MODE,
        "decision": "recorded",
        "reason": (
            reason
            or (
                "duplicate_usage_signal"
                if duplicate
                else "usage_recorded"
            )
        ),
        "duplicate": duplicate,
        "loaded_count": artifact.get(
            "loaded_count"
        ),
        "used_count": artifact.get(
            "used_count"
        ),
        "loaded_memory_ids": list(
            artifact.get(
                "loaded_memory_ids"
            )
            or []
        ),
        "used_memory_ids": list(
            artifact.get("used_memory_ids")
            or []
        ),
    }


def record_recall_requested(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    recall_id: Any,
) -> dict[str, Any]:
    """Record that the AI explicitly requested this recall.

    The event is derived from a REAL, persisted Recall Request:
    ``read_recall_request`` must return a validated
    ``memory-recall-request.v1`` artifact for exactly this
    ``conversation_id`` + ``cognitive_request_id`` + ``recall_id``.
    The anchor is read from that artifact, never from the caller. A
    request that was never persisted therefore cannot manufacture a
    phantom ``recall_requested`` event.

    This is written BEFORE any anchor load is attempted, so

        recall_requested = 1
        loaded_count     = 0
        used_count       = 0

    is a real, representable state for a recall that could not be
    satisfied. Idempotent per recall id: an existing, valid artifact
    bound to the same anchor is returned as ``duplicate`` and its
    events are never double-counted. A malformed existing artifact
    fails closed and is never merged into.
    """

    try:
        validate_conversation_id(conversation_id)

        validate_cognitive_request_id(
            cognitive_request_id
        )
    except ValueError:
        return _no_record(
            "invalid_request_binding"
        )

    if not is_valid_recall_id(recall_id):
        return _no_record("invalid_recall_id")

    recall_request_artifact = read_recall_request(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if not isinstance(
        recall_request_artifact, dict
    ):
        return _no_record(
            "recall_request_not_found"
        )

    anchor_memory_id = (
        recall_request_artifact.get(
            "anchor_memory_id"
        )
    )

    if (
        not isinstance(anchor_memory_id, str)
        or not anchor_memory_id.strip()
    ):
        return _no_record(
            "invalid_anchor_memory_id"
        )

    anchor_memory_id = anchor_memory_id.strip()

    path = memory_usage_path(
        conversation_id,
        cognitive_request_id,
        recall_id,
    )

    with LOCK:
        existing = read_json(path)

        if existing is not None:
            if (
                is_valid_usage_artifact(
                    existing,
                    conversation_id=(
                        conversation_id
                    ),
                    cognitive_request_id=(
                        cognitive_request_id
                    ),
                    recall_id=recall_id,
                )
                and existing.get(
                    "anchor_memory_id"
                )
                == anchor_memory_id
            ):
                return _summary(
                    existing, duplicate=True
                )

            return _no_record(
                "usage_artifact_invalid"
            )

        artifact = _base_artifact(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            recall_id=recall_id,
            anchor_memory_id=anchor_memory_id,
        )

        atomic_write(path, artifact)

    return _summary(
        artifact,
        reason="recall_requested_recorded",
    )


def record_memory_loaded(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    recall_id: Any,
) -> dict[str, Any]:
    """Record ``memory_loaded`` from a PERSISTED Recall result.

    The loaded ids are read from the real, on-disk related-recall
    artifact via ``read_related_recall`` -- which fully re-validates
    identity, fingerprint, revision and memory shape. A synthetic
    dict is never accepted, and a persisted artifact whose identity /
    fingerprint / anchor / scope / source revision has been tampered
    with reads as missing, so no memory can ever be marked loaded
    without a real recall result.

    Idempotent per recall id; never writes a ``used`` event.
    """

    if not _valid_binding(
        conversation_id,
        cognitive_request_id,
        recall_id,
    ):
        return _no_record(
            "invalid_request_binding"
        )

    # Imported lazily: ``related_recall`` imports this module, so a
    # top-level import would be circular.
    from ombrebrain.context.related_recall import (
        read_related_recall,
    )

    recall_artifact = read_related_recall(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if not isinstance(recall_artifact, dict):
        return _no_record(
            "related_recall_not_found"
        )

    path = memory_usage_path(
        conversation_id,
        cognitive_request_id,
        recall_id,
    )

    with LOCK:
        artifact = read_json(path)

        if artifact is None:
            return _no_record("usage_not_found")

        if not is_valid_usage_artifact(
            artifact,
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            recall_id=recall_id,
        ):
            return _no_record(
                "usage_artifact_invalid"
            )

        merged = list(
            artifact["loaded_memory_ids"]
        )

        new_ids: list[str] = []

        for memory_id in _loaded_ids(
            recall_artifact
        ):
            if memory_id not in merged:
                merged.append(memory_id)
                new_ids.append(memory_id)

        events = list(artifact["events"])

        at = now_iso()

        for memory_id in new_ids:
            events.append(
                {
                    "stage": _STAGE_MEMORY_LOADED,
                    "memory_id": memory_id,
                    "at": at,
                }
            )

        artifact["loaded_memory_ids"] = merged
        artifact["loaded_count"] = len(merged)
        artifact["events"] = events

        atomic_write(path, artifact)

        summary = _summary(
            artifact,
            reason="memory_loaded_recorded",
        )

        summary["newly_loaded_count"] = len(
            new_ids
        )

        return summary


def record_usage(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
    used_memory_ids: Any = (),
) -> dict[str, Any]:
    """Record an EXPLICIT usage signal for one recall.

    The signal must be bound to all three of ``conversation_id``,
    ``cognitive_request_id`` and ``recall_id``; a report against
    another request's recall is refused. Membership is enforced:
    ``used`` must be a SUBSET of the memories that recall actually
    loaded. A never-loaded id is rejected -- it is never added to
    ``used`` -- with reason ``memory_not_loaded``.
    """

    if not _valid_binding(
        conversation_id,
        cognitive_request_id,
        recall_id,
    ):
        return _no_record(
            "invalid_request_binding"
        )

    # Read the real, persisted recall artifact this usage must belong
    # to. ``read_related_recall`` fully re-validates it, so Request B
    # can never report usage against Request A's recall, and a
    # tampered recall result cannot authorize a usage report.
    from ombrebrain.context.related_recall import (
        read_related_recall,
    )

    recall_artifact = read_related_recall(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if not isinstance(recall_artifact, dict):
        return _no_record("recall_not_found")

    path = memory_usage_path(
        conversation_id,
        cognitive_request_id,
        recall_id,
    )

    with LOCK:
        artifact = read_json(path)

        if artifact is None:
            return _no_record("usage_not_found")

        if not is_valid_usage_artifact(
            artifact,
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            recall_id=recall_id,
        ):
            return _no_record(
                "usage_artifact_invalid"
            )

        loaded_set = set(
            artifact["loaded_memory_ids"]
        )

        accepted: list[str] = []
        rejected: list[dict[str, str]] = []

        for value in (
            used_memory_ids
            if isinstance(
                used_memory_ids, (list, tuple)
            )
            else []
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                rejected.append(
                    {
                        "memory_id": str(value),
                        "reason":
                            _REASON_INVALID_ID,
                    }
                )
                continue

            memory_id = value.strip()

            if memory_id not in loaded_set:
                # Used must be a subset of loaded. A never-loaded id
                # is rejected and never added to used.
                rejected.append(
                    {
                        "memory_id": memory_id,
                        "reason": (
                            _REASON_MEMORY_NOT_LOADED
                        ),
                    }
                )
                continue

            if memory_id not in accepted:
                accepted.append(memory_id)

        merged = list(
            artifact["used_memory_ids"]
        )

        new_ids: list[str] = []

        for memory_id in accepted:
            if memory_id not in merged:
                merged.append(memory_id)
                new_ids.append(memory_id)

        events = list(artifact["events"])

        at = now_iso()

        for memory_id in new_ids:
            events.append(
                {
                    "stage": _STAGE_USED,
                    "memory_id": memory_id,
                    "at": at,
                }
            )

        artifact["used_memory_ids"] = merged
        artifact["used_count"] = len(merged)
        artifact["events"] = events

        atomic_write(path, artifact)

        summary = _summary(
            artifact,
            reason="explicit_usage_signal",
        )

        summary["rejected"] = rejected
        summary["newly_used_count"] = len(new_ids)

        return summary


def _valid_binding(
    conversation_id: Any,
    cognitive_request_id: Any,
    recall_id: Any,
) -> bool:
    try:
        validate_conversation_id(
            conversation_id
        )

        validate_cognitive_request_id(
            cognitive_request_id
        )
    except ValueError:
        return False

    return is_valid_recall_id(recall_id)


def read_memory_usage(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
) -> dict[str, Any] | None:
    """Read one raw usage artifact, identity-validated.

    Returns None when the artifact is missing OR malformed: a corrupt
    usage artifact is never handed to a caller as if it were valid.
    """

    if not _valid_binding(
        conversation_id,
        cognitive_request_id,
        recall_id,
    ):
        return None

    artifact = read_json(
        memory_usage_path(
            conversation_id,
            cognitive_request_id,
            recall_id,
        )
    )

    if not is_valid_usage_artifact(
        artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    ):
        return None

    return artifact


def memory_usage_status(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
) -> dict[str, Any]:
    """Privacy-safe status of one usage artifact."""

    artifact = read_memory_usage(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if not isinstance(artifact, dict):
        return {"exists": False}

    return {
        "exists": True,
        "version": artifact.get("version"),
        "mode": artifact.get("mode"),
        "loaded_count":
            artifact.get("loaded_count"),
        "used_count":
            artifact.get("used_count"),
        "event_count": len(
            artifact.get("events") or []
        ),
    }