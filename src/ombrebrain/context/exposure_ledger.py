from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ombrebrain.context.validators.freshness import (
    is_valid_revision,
    validate_context_freshness,
)


# Memory Exposure Ledger — shadow only.
#
# retrieved != surfaced
# surfaced   != noticed
# noticed    != recall_requested
# recall_requested != full_memory_loaded
# full_memory_loaded != used_in_response
#
# This phase records exactly two stages and nothing more:
#
#     retrieved
#     surfaced_as_flash
#
# The later stages (noticed / recall_requested / full_memory_loaded /
# used_in_response) are deliberately NOT implemented and are NOT
# written as false -- they simply do not exist yet.
#
# No reinforcement occurs in this phase. This ledger only observes:
# it never writes activation_count, last_active, importance,
# strength, weight, score, decay, archive state or any memory
# metadata. Source memory files stay byte-for-byte unchanged.
#
# Privacy: memory ids are kept internally because a future Recall
# needs to know what they point at, but no cue text, raw memory text,
# query text, conversation text, current user text or rendered
# Context is ever stored here, and no memory id / request id /
# conversation id / cue is ever logged.

_VERSION = "memory-exposure-ledger.v1"
_MODE = "shadow_only"

# The Unified observation shape this ledger may attribute to a
# request. A snapshot with the right revision but the wrong version
# or the wrong conversation must never be recorded against this
# conversation's ledger.
_UNIFIED_VERSION = "unified-context-candidate.v1"

_STAGE_RETRIEVED = "retrieved"
_STAGE_SURFACED = "surfaced_as_flash"

_DEFAULT_ROOT = "/app/buckets/.context"

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_COGNITIVE_REQUEST_ID_RE = re.compile(
    r"^ctxreq_[0-9a-f]{32}$"
)

_LOCK = threading.RLock()


def _root() -> Path:
    return Path(
        (
            os.environ.get(
                "OMBRE_CONTEXT_STATE_DIR"
            )
            or _DEFAULT_ROOT
        ).strip()
    )


def _validate_conversation_id(
    conversation_id: str,
) -> None:
    if (
        not isinstance(
            conversation_id,
            str,
        )
        or not _CONVERSATION_ID_RE.fullmatch(
            conversation_id
        )
    ):
        raise ValueError(
            "invalid conversation_id"
        )


def _validate_cognitive_request_id(
    cognitive_request_id: str,
) -> None:
    if (
        not isinstance(
            cognitive_request_id,
            str,
        )
        or not _COGNITIVE_REQUEST_ID_RE.fullmatch(
            cognitive_request_id
        )
    ):
        raise ValueError(
            "invalid cognitive_request_id"
        )


def _path(
    conversation_id: str,
    cognitive_request_id: str,
) -> Path:
    """One ledger file per (conversation, request)."""

    _validate_conversation_id(
        conversation_id
    )

    _validate_cognitive_request_id(
        cognitive_request_id
    )

    return (
        _root()
        / "exposure_ledger"
        / conversation_id
        / (cognitive_request_id + ".json")
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


def _read_json(
    path: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    return (
        value
        if isinstance(
            value,
            dict,
        )
        else None
    )


def _atomic_write(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        path.parent.chmod(0o700)
    except OSError:
        pass

    temp = path.with_name(
        path.name + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        temp.chmod(0o600)
    except OSError:
        pass

    os.replace(
        temp,
        path,
    )

    try:
        path.chmod(0o600)
    except OSError:
        pass


def retrieved_memory_ids(
    unified: Any,
) -> list[str]:
    """Stable ids of the candidate set the Context layer exposed.

    This is the canonical Context retrieval candidate observation --
    never the raw vector pool, the embedding pool or the
    threshold-rejected candidates. Order and de-duplication follow
    the candidates.
    """

    if not isinstance(
        unified,
        dict,
    ):
        return []

    sections = unified.get("sections")

    if not isinstance(
        sections,
        dict,
    ):
        return []

    memories = sections.get("memories")

    if not isinstance(
        memories,
        list,
    ):
        return []

    ids: list[str] = []
    seen: set[str] = set()

    for candidate in memories:
        if not isinstance(
            candidate,
            dict,
        ):
            continue

        memory_id = candidate.get("id")

        if (
            not isinstance(
                memory_id,
                str,
            )
            or not memory_id.strip()
        ):
            continue

        memory_id = memory_id.strip()

        if memory_id in seen:
            continue

        seen.add(memory_id)

        ids.append(memory_id)

    return ids


def build_exposure_ledger(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    source_unified_revision: Any,
    retrieved_ids: Any,
    surfaced_ids: Any,
) -> dict[str, Any]:
    """Build one privacy-safe ledger artifact.

    ``surfaced_as_flash`` is always a subset of ``retrieved``: an id
    that never entered the Flash artifact is never recorded as
    surfaced. Each (memory, stage) pair appears at most once, so
    rebuilding the ledger for the same request is idempotent.
    """

    _validate_conversation_id(
        conversation_id
    )

    _validate_cognitive_request_id(
        cognitive_request_id
    )

    retrieved: list[str] = []

    for value in (
        retrieved_ids
        if isinstance(
            retrieved_ids,
            list,
        )
        else []
    ):
        if (
            isinstance(
                value,
                str,
            )
            and value.strip()
            and value.strip() not in retrieved
        ):
            retrieved.append(
                value.strip()
            )

    retrieved_set = set(retrieved)

    surfaced: list[str] = []

    for value in (
        surfaced_ids
        if isinstance(
            surfaced_ids,
            list,
        )
        else []
    ):
        if not (
            isinstance(
                value,
                str,
            )
            and value.strip()
        ):
            continue

        value = value.strip()

        if (
            value not in retrieved_set
            or value in surfaced
        ):
            continue

        surfaced.append(value)

    events: list[dict[str, str]] = []

    for memory_id in retrieved:
        events.append(
            {
                "memory_id": memory_id,
                "stage": _STAGE_RETRIEVED,
            }
        )

    for memory_id in surfaced:
        events.append(
            {
                "memory_id": memory_id,
                "stage": _STAGE_SURFACED,
            }
        )

    return {
        "version":
            _VERSION,
        "mode":
            _MODE,
        "conversation_id":
            conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "source_unified_revision":
            source_unified_revision,
        "decision":
            "recorded",
        "reason":
            "exposure_recorded",
        "retrieved_count":
            len(retrieved),
        "surfaced_count":
            len(surfaced),
        "retrieved_memory_ids":
            retrieved,
        "surfaced_memory_ids":
            surfaced,
        "events":
            events,
    }


def update_exposure_ledger(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    unified: Any,
    surfaced_ids: Any = (),
    expected_unified_revision: Any,
) -> dict[str, Any]:
    """Persist the ledger for this request.

    Bound to the Unified revision this request produced: a ledger
    written from a Unified file another request already overwrote
    would attribute the wrong candidates to this request, so a
    mismatch is reported with ``stored=False`` and nothing is
    written. Never raises.
    """

    _validate_conversation_id(
        conversation_id
    )

    _validate_cognitive_request_id(
        cognitive_request_id
    )

    if not is_valid_revision(
        expected_unified_revision
    ):
        return {
            "stored":
                False,
            "mode":
                _MODE,
            "decision":
                "no_record",
            "reason":
                "invalid_expected_unified_revision",
        }

    if not isinstance(
        unified,
        dict,
    ):
        return {
            "stored":
                False,
            "mode":
                _MODE,
            "decision":
                "no_record",
            "reason":
                "unified_observation_invalid",
        }

    if (
        unified.get("version")
        != _UNIFIED_VERSION
    ):
        return {
            "stored":
                False,
            "mode":
                _MODE,
            "decision":
                "no_record",
            "reason":
                "unified_observation_invalid",
        }

    if (
        unified.get("conversation_id")
        != conversation_id
    ):
        return {
            "stored":
                False,
            "mode":
                _MODE,
            "decision":
                "no_record",
            "reason":
                "unified_conversation_mismatch",
        }

    observed_revision = unified.get(
        "revision"
    )

    freshness = validate_context_freshness(
        checked_revision=observed_revision,
        expected_revision=(
            expected_unified_revision
        ),
        invalid_reason=(
            "invalid_observed_unified_revision"
        ),
        mismatch_reason=(
            "ledger_unified_request_revision_mismatch"
        ),
    )

    if not freshness["valid"]:
        return {
            "stored":
                False,
            "mode":
                _MODE,
            "decision":
                "no_record",
            "reason":
                freshness["reason"],
        }

    artifact = build_exposure_ledger(
        conversation_id=conversation_id,
        cognitive_request_id=
            cognitive_request_id,
        source_unified_revision=(
            observed_revision
        ),
        retrieved_ids=(
            retrieved_memory_ids(
                unified
            )
        ),
        surfaced_ids=surfaced_ids,
    )

    artifact["created_at"] = _now()

    with _LOCK:
        _atomic_write(
            _path(
                conversation_id,
                cognitive_request_id,
            ),
            artifact,
        )

    output = dict(
        artifact
    )

    output["stored"] = True

    output.pop(
        "created_at",
        None,
    )

    return output


def exposure_ledger_status(
    *,
    conversation_id: str,
    cognitive_request_id: str,
) -> dict[str, Any]:
    """Read back one ledger artifact (privacy-safe summary only)."""

    state = _read_json(
        _path(
            conversation_id,
            cognitive_request_id,
        )
    )

    if state is None:
        return {
            "exists": False,
        }

    return {
        "exists": True,
        "version": state.get("version"),
        "mode": state.get("mode"),
        "stored": True,
        "retrieved_count":
            state.get("retrieved_count"),
        "surfaced_count":
            state.get("surfaced_count"),
        "source_unified_revision":
            state.get(
                "source_unified_revision"
            ),
    }