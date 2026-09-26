from __future__ import annotations

import os
import secrets
from typing import Any

from ombrebrain.context.recall_types import (
    bound_text,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    is_valid_memref,
    now_iso,
    read_json,
    recall_surface_path,
    atomic_write,
    LOCK,
)


# Memory Recall Surface v1 — model-facing, shadow only.
#
# This is the safety boundary for a FUTURE live exposure. It turns one
# request's Memory Flash artifact into the smallest possible
# model-facing representation:
#
#     memref_<32 hex>  ->  an opaque reference
#     cue              ->  the already-bounded Flash cue
#
# The model never sees a real memory id, a bucket path, metadata, a
# revision graph, an activation count or a score. Internally the
# mapping ``memref -> real memory_id`` is persisted, but it is bound
# to exactly one ``conversation_id`` + ``cognitive_request_id``, so a
# guessed or cross-request memref is refused.
#
# It is SHADOW only: it is never written into the live request body,
# never enters Unified / Preview / Injection / a cache key, and it
# never changes what the model currently sees. Default OFF.
#
# It performs no retrieval, loads no memory content, calls no model
# and never mutates a memory source file.

_VERSION = "memory-recall-surface.v1"
_MODE = "shadow_only"

_ENV_SURFACE_SHADOW = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW"
)

_FLASH_VERSION = "memory-flash.v1"
_FLASH_MODE = "shadow_only"

# Model-facing cues stay as bounded as the Flash contract.
_MAX_CUE_CHARS = 320


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def recall_surface_shadow_enabled() -> bool:
    return _truthy(
        os.environ.get(_ENV_SURFACE_SHADOW)
    )


def new_memref() -> str:
    return "memref_" + secrets.token_hex(16)


def _not_stored(
    reason: str,
) -> dict[str, Any]:
    return {
        "stored": False,
        "mode": _MODE,
        "decision": "no_surface",
        "reason": reason,
        "memory_count": 0,
    }


def _model_surfaces(
    surfaces: Any,
) -> list[dict[str, Any]]:
    """Strip every internal field from a surface list.

    Only ``memref`` and ``cue`` (plus the harmless ``source_type``
    enum) may leave this function. A real memory id can never
    survive here.
    """

    output: list[dict[str, Any]] = []

    for item in (
        surfaces
        if isinstance(surfaces, list)
        else []
    ):
        if not isinstance(item, dict):
            continue

        memref = item.get("memref")
        cue = item.get("cue")

        if not is_valid_memref(memref):
            continue

        if not isinstance(cue, str) or not cue:
            continue

        entry: dict[str, Any] = {
            "memref": memref,
            "cue": cue,
        }

        source_type = item.get("source_type")

        if (
            isinstance(source_type, str)
            and source_type.strip()
        ):
            entry["source_type"] = (
                source_type.strip()
            )

        output.append(entry)

    return output


def build_recall_surface(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    flash_report: Any,
) -> dict[str, Any]:
    """Build the internal surface artifact for one request.

    Refuses (without raising) unless ``flash_report`` is a well-formed
    ``memory-flash.v1`` ``shadow_only`` report for THIS conversation
    AND THIS request. Each surfaced memory gets a fresh opaque memref;
    the real id is kept only in the internal mapping.
    """

    if (
        not isinstance(flash_report, dict)
        or flash_report.get("version")
        != _FLASH_VERSION
        or flash_report.get("mode") != _FLASH_MODE
    ):
        return _not_stored(
            "malformed_flash_report"
        )

    if (
        flash_report.get("conversation_id")
        != conversation_id
        or flash_report.get(
            "cognitive_request_id"
        )
        != cognitive_request_id
    ):
        return _not_stored(
            "flash_identity_mismatch"
        )

    flashes = flash_report.get("flashes")

    if not isinstance(flashes, list):
        return _not_stored(
            "no_surfaced_memories"
        )

    mapping: dict[str, str] = {}
    surfaces: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for item in flashes:
        if not isinstance(item, dict):
            continue

        memory_id = item.get("memory_id")

        if (
            not isinstance(memory_id, str)
            or not memory_id.strip()
        ):
            continue

        memory_id = memory_id.strip()

        if memory_id in seen_ids:
            continue

        cue = bound_text(
            item.get("cue"),
            max_chars=_MAX_CUE_CHARS,
        )

        if not cue:
            continue

        seen_ids.add(memory_id)

        memref = new_memref()

        mapping[memref] = memory_id

        surface: dict[str, Any] = {
            "memref": memref,
            "cue": cue,
        }

        source_type = item.get("source_type")

        if (
            isinstance(source_type, str)
            and source_type.strip()
        ):
            surface["source_type"] = (
                source_type.strip()
            )

        surfaces.append(surface)

    if not surfaces:
        return _not_stored(
            "no_surfaced_memories"
        )

    return {
        "version": _VERSION,
        "mode": _MODE,
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "source_flash_unified_revision":
            flash_report.get(
                "source_unified_revision"
            ),
        "created_at": now_iso(),
        "memory_count": len(surfaces),
        "mapping": mapping,
        "surfaces": surfaces,
    }


def update_recall_surface(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    flash_report: Any,
) -> dict[str, Any]:
    """Persist the shadow Recall Surface for one request."""

    if not recall_surface_shadow_enabled():
        return _not_stored("surface_disabled")

    if not is_valid_conversation_id(
        conversation_id
    ) or not is_valid_cognitive_request_id(
        cognitive_request_id
    ):
        return _not_stored(
            "invalid_request_identity"
        )

    artifact = build_recall_surface(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        flash_report=flash_report,
    )

    if artifact.get("stored") is False:
        return artifact

    path = recall_surface_path(
        conversation_id,
        cognitive_request_id,
    )

    with LOCK:
        atomic_write(path, artifact)

    return {
        "stored": True,
        "mode": _MODE,
        "decision": "surface_built",
        "reason": "recall_surface_built",
        "memory_count": artifact[
            "memory_count"
        ],
        "source_flash_unified_revision":
            artifact.get(
                "source_flash_unified_revision"
            ),
    }


def recall_surface_for_model(
    *,
    conversation_id: str,
    cognitive_request_id: str,
) -> dict[str, Any]:
    """The model-facing representation: memref + cue only.

    No real memory id, no mapping, no bucket path, no metadata, no
    revision and no score is ever returned. Reading is request
    scoped.
    """

    if not is_valid_conversation_id(
        conversation_id
    ) or not is_valid_cognitive_request_id(
        cognitive_request_id
    ):
        return {
            "version": _VERSION,
            "mode": _MODE,
            "surfaces": [],
        }

    artifact = read_json(
        recall_surface_path(
            conversation_id,
            cognitive_request_id,
        )
    )

    if (
        not isinstance(artifact, dict)
        or artifact.get("version") != _VERSION
    ):
        return {
            "version": _VERSION,
            "mode": _MODE,
            "surfaces": [],
        }

    return {
        "version": _VERSION,
        "mode": _MODE,
        "surfaces": _model_surfaces(
            artifact.get("surfaces")
        ),
    }


def resolve_recall_ref(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    memref: Any,
) -> str | None:
    """Resolve one opaque memref for THIS request only.

    A memref from another conversation, another request or a random
    guess returns None. The mapping is never global and never
    guessable.
    """

    if not is_valid_memref(memref):
        return None

    if not is_valid_conversation_id(
        conversation_id
    ) or not is_valid_cognitive_request_id(
        cognitive_request_id
    ):
        return None

    artifact = read_json(
        recall_surface_path(
            conversation_id,
            cognitive_request_id,
        )
    )

    if (
        not isinstance(artifact, dict)
        or artifact.get("version") != _VERSION
    ):
        return None

    mapping = artifact.get("mapping")

    if not isinstance(mapping, dict):
        return None

    memory_id = mapping.get(memref)

    if (
        isinstance(memory_id, str)
        and memory_id.strip()
    ):
        return memory_id.strip()

    return None


def recall_surface_status(
    *,
    conversation_id: str,
    cognitive_request_id: str,
) -> dict[str, Any]:
    """Privacy-safe status of one surface artifact."""

    artifact = read_json(
        recall_surface_path(
            conversation_id,
            cognitive_request_id,
        )
    )

    if not isinstance(artifact, dict):
        return {"exists": False}

    return {
        "exists": True,
        "version": artifact.get("version"),
        "mode": artifact.get("mode"),
        "memory_count":
            artifact.get("memory_count"),
        "source_flash_unified_revision":
            artifact.get(
                "source_flash_unified_revision"
            ),
    }