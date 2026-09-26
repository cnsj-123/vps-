from __future__ import annotations

from typing import Any

from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    is_valid_recall_id,
    memory_usage_path,
    now_iso,
    read_json,
    related_recall_path,
    validate_cognitive_request_id,
    validate_conversation_id,
)


# Memory Usage Signal v1 — observation only.
#
#     recall_requested != memory_loaded
#     memory_loaded    != used
#     used             != reinforced
#
# This module records exactly three event stages:
#
#     recall_requested   the AI asked to recall an anchor
#     memory_loaded      a memory was returned by that recall
#     used               the AI explicitly declared it used a memory
#
# ``used`` is NEVER inferred. There is no keyword matching, no answer
# text scanning and no heuristic "the model probably used M17". A
# usage event exists only when an explicit model / tool control signal
# reaches ``record_usage()``.
#
# ``loaded`` is NEVER ``used``: every memory a recall returns is
# recorded as loaded, and ``used_memory_ids`` stays empty until an
# explicit signal arrives.
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

_RECALL_VERSION = "related-memory-recall.v1"
_RECALL_MODE = "shadow_only"

_STAGE_RECALL_REQUESTED = "recall_requested"
_STAGE_MEMORY_LOADED = "memory_loaded"
_STAGE_USED = "used"

_REASON_MEMORY_NOT_LOADED = "memory_not_loaded"
_REASON_INVALID_ID = "invalid_memory_id"


def _recall_binding(
    recall_artifact: Any,
) -> dict[str, Any] | None:
    """Validate a related-recall artifact's structural identity."""

    if (
        not isinstance(recall_artifact, dict)
        or recall_artifact.get("version")
        != _RECALL_VERSION
        or recall_artifact.get("mode")
        != _RECALL_MODE
    ):
        return None

    conversation_id = recall_artifact.get(
        "conversation_id"
    )

    cognitive_request_id = (
        recall_artifact.get(
            "cognitive_request_id"
        )
    )

    recall_id = recall_artifact.get(
        "recall_id"
    )

    try:
        validate_conversation_id(
            conversation_id
        )

        validate_cognitive_request_id(
            cognitive_request_id
        )
    except ValueError:
        return None

    if not is_valid_recall_id(recall_id):
        return None

    return {
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "recall_id": recall_id,
    }


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
    anchor_memory_id: Any,
    loaded: list[str],
) -> dict[str, Any]:
    requested_at = now_iso()

    events: list[dict[str, Any]] = []

    if (
        isinstance(anchor_memory_id, str)
        and anchor_memory_id.strip()
    ):
        events.append(
            {
                "stage": _STAGE_RECALL_REQUESTED,
                "memory_id":
                    anchor_memory_id.strip(),
                "at": requested_at,
            }
        )

    for memory_id in loaded:
        events.append(
            {
                "stage": _STAGE_MEMORY_LOADED,
                "memory_id": memory_id,
                "at": requested_at,
            }
        )

    return {
        "version": _VERSION,
        "mode": _MODE,
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "recall_id": recall_id,
        "anchor_memory_id": (
            anchor_memory_id.strip()
            if isinstance(
                anchor_memory_id, str
            )
            and anchor_memory_id.strip()
            else None
        ),
        "requested_at": requested_at,
        "loaded_memory_ids": loaded,
        "used_memory_ids": [],
        "loaded_count": len(loaded),
        "used_count": 0,
        "events": events,
    }


def _summary(
    artifact: dict[str, Any],
    *,
    duplicate: bool = False,
) -> dict[str, Any]:
    return {
        "stored": True,
        "mode": _MODE,
        "decision": "recorded",
        "reason": (
            "duplicate_usage_signal"
            if duplicate
            else "usage_recorded"
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


def record_recall(
    *,
    recall_artifact: Any,
) -> dict[str, Any]:
    """Record ``recall_requested`` + ``memory_loaded`` for one recall.

    Loaded != used: this never writes a ``used`` event and never
    writes a ``used_memory_ids`` entry. Idempotent: re-recording the
    same recall returns the existing artifact as ``duplicate`` and
    never double-counts the events.
    """

    binding = _recall_binding(recall_artifact)

    if binding is None:
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason": "malformed_recall_artifact",
            "duplicate": False,
            "loaded_count": 0,
            "used_count": 0,
        }

    path = memory_usage_path(
        binding["conversation_id"],
        binding["cognitive_request_id"],
        binding["recall_id"],
    )

    with LOCK:
        existing = read_json(path)

        if isinstance(existing, dict):
            return _summary(
                existing,
                duplicate=True,
            )

        artifact = _base_artifact(
            conversation_id=(
                binding["conversation_id"]
            ),
            cognitive_request_id=(
                binding[
                    "cognitive_request_id"
                ]
            ),
            recall_id=binding["recall_id"],
            anchor_memory_id=(
                recall_artifact.get(
                    "anchor_memory_id"
                )
            ),
            loaded=_loaded_ids(
                recall_artifact
            ),
        )

        atomic_write(path, artifact)

    return _summary(artifact)


def _load_or_build(
    *,
    recall_artifact: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    path = memory_usage_path(
        binding["conversation_id"],
        binding["cognitive_request_id"],
        binding["recall_id"],
    )

    existing = read_json(path)

    if isinstance(
        existing, dict
    ) and existing.get("version") == _VERSION:
        return existing

    return _base_artifact(
        conversation_id=(
            binding["conversation_id"]
        ),
        cognitive_request_id=(
            binding["cognitive_request_id"]
        ),
        recall_id=binding["recall_id"],
        anchor_memory_id=(
            recall_artifact.get(
                "anchor_memory_id"
            )
        ),
        loaded=_loaded_ids(recall_artifact),
    )


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
    loaded. An id that was never loaded is ignored (never silently
    added) with reason ``memory_not_loaded``.
    """

    if not _valid_binding(
        conversation_id,
        cognitive_request_id,
        recall_id,
    ):
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason": "invalid_request_binding",
            "duplicate": False,
            "loaded_count": 0,
            "used_count": 0,
            "rejected": [],
        }

    # Read the recall artifact this usage must belong to. The binding
    # is verified field by field, so Request B can never report usage
    # against Request A's recall.
    recall_artifact = read_json(
        related_recall_path(
            conversation_id,
            cognitive_request_id,
            recall_id,
        )
    )

    binding = _recall_binding(recall_artifact)

    if binding is None:
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason": "recall_not_found",
            "duplicate": False,
            "loaded_count": 0,
            "used_count": 0,
            "rejected": [],
        }

    if (
        binding["conversation_id"]
        != conversation_id
        or binding["cognitive_request_id"]
        != cognitive_request_id
        or binding["recall_id"] != recall_id
    ):
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason": "recall_binding_mismatch",
            "duplicate": False,
            "loaded_count": 0,
            "used_count": 0,
            "rejected": [],
        }

    path = memory_usage_path(
        conversation_id,
        cognitive_request_id,
        recall_id,
    )

    with LOCK:
        artifact = _load_or_build(
            recall_artifact=recall_artifact,
            binding=binding,
        )

        loaded_set = set(
            artifact.get(
                "loaded_memory_ids"
            )
            or []
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
                # Used must be a subset of loaded. An id that was
                # never loaded is remembered as requested-and-used
                # even though it was never loaded.
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

        existing_used = artifact.get(
            "used_memory_ids"
        )

        if not isinstance(existing_used, list):
            existing_used = []

        merged = list(existing_used)

        new_ids: list[str] = []

        for memory_id in accepted:
            if memory_id not in merged:
                merged.append(memory_id)
                new_ids.append(memory_id)

        events = artifact.get("events")

        if not isinstance(events, list):
            events = []

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

    summary = _summary(artifact)

    summary["decision"] = "usage_recorded"
    summary["reason"] = "explicit_usage_signal"
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
    """Read one raw usage artifact (internal ids / events only)."""

    if not _valid_binding(
        conversation_id,
        cognitive_request_id,
        recall_id,
    ):
        return None

    return read_json(
        memory_usage_path(
            conversation_id,
            cognitive_request_id,
            recall_id,
        )
    )


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