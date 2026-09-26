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
from ombrebrain.context.validators.freshness import (
    is_valid_revision,
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
_FLASH_SURFACED = "surfaced"

# Refusal reasons.
_REASON_MALFORMED = "malformed_flash_report"
_REASON_IDENTITY = "flash_identity_mismatch"
_REASON_NOT_SURFACED = "flash_not_surfaced"
_REASON_REVISION = (
    "invalid_flash_source_unified_revision"
)
_REASON_NO_MEMORIES = "no_surfaced_memories"

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
    AND THIS request that actually SURFACED memories:

      - ``decision`` must be ``surfaced``. A ``no_surface`` artifact is
        never a cue source, even if a malformed one still carries a
        stale ``flashes`` list -- no memref, no cue and no mapping may
        be derived from it;
      - ``source_unified_revision`` must be a valid revision under the
        shared freshness contract. The model-facing safety boundary is
        never built on a Flash with no legal source binding.

    Each surfaced memory gets a fresh opaque memref; the real id is
    kept only in the internal mapping.
    """

    if (
        not isinstance(flash_report, dict)
        or flash_report.get("version")
        != _FLASH_VERSION
        or flash_report.get("mode") != _FLASH_MODE
    ):
        return _not_stored(_REASON_MALFORMED)

    if (
        flash_report.get("conversation_id")
        != conversation_id
        or flash_report.get(
            "cognitive_request_id"
        )
        != cognitive_request_id
    ):
        return _not_stored(_REASON_IDENTITY)

    # Only a real surfaced Flash may become a model-facing surface.
    if (
        flash_report.get("decision")
        != _FLASH_SURFACED
    ):
        return _not_stored(_REASON_NOT_SURFACED)

    # The shared freshness contract, not a local validator.
    if not is_valid_revision(
        flash_report.get(
            "source_unified_revision"
        )
    ):
        return _not_stored(_REASON_REVISION)

    flashes = flash_report.get("flashes")

    if not isinstance(flashes, list):
        return _not_stored(_REASON_NO_MEMORIES)

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
        return _not_stored(_REASON_NO_MEMORIES)

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


def is_valid_recall_surface_artifact(
    artifact: Any,
    *,
    conversation_id: Any = None,
    cognitive_request_id: Any = None,
) -> bool:
    """Structural identity of one persisted Recall Surface.

    Reading the file at the right path proves nothing, so every
    reader re-validates the contract, the mode, the request identity,
    the Flash revision binding and the mapping / surfaces shape.
    Anything off makes the whole artifact unusable: a corrupt surface
    must never resolve a memref or reach the model.
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
        not is_valid_conversation_id(
            artifact_conversation
        )
        or not is_valid_cognitive_request_id(
            artifact_request
        )
    ):
        return False

    if not is_valid_revision(
        artifact.get(
            "source_flash_unified_revision"
        )
    ):
        return False

    mapping = artifact.get("mapping")

    surfaces = artifact.get("surfaces")

    if (
        not isinstance(mapping, dict)
        or not isinstance(surfaces, list)
        or not surfaces
    ):
        return False

    # mapping and surfaces must be EXACTLY the same set: one opaque
    # ref per model-facing surface, no hidden extra ref.
    if len(mapping) != len(surfaces):
        return False

    if artifact.get("memory_count") != len(
        surfaces
    ):
        return False

    memory_ids: list[str] = []

    for memref, memory_id in mapping.items():
        if not is_valid_memref(memref):
            return False

        if (
            not isinstance(memory_id, str)
            or not memory_id.strip()
        ):
            return False

        memory_ids.append(memory_id)

    # build_recall_surface() de-duplicates memory ids, so a valid
    # artifact never maps two different refs to one memory.
    if len(set(memory_ids)) != len(memory_ids):
        return False

    surface_memrefs: list[str] = []

    for surface in surfaces:
        if not isinstance(surface, dict):
            return False

        memref = surface.get("memref")

        if not is_valid_memref(memref):
            return False

        cue = surface.get("cue")

        if not isinstance(cue, str) or not cue:
            return False

        surface_memrefs.append(memref)

    if len(set(surface_memrefs)) != len(
        surface_memrefs
    ):
        return False

    if set(surface_memrefs) != set(mapping):
        return False

    return True


def update_recall_surface(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    flash_report: Any,
) -> dict[str, Any]:
    """Persist the shadow Recall Surface for one request.

    Idempotent and memref-stable: if a structurally valid surface for
    the SAME conversation, request and Flash revision already exists,
    it is reused (``duplicate=True``) and the existing memref mapping
    is kept. A model that already saw a memref must not see it
    silently replaced because an observer ran twice.
    """

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
        existing = read_json(path)

        if (
            is_valid_recall_surface_artifact(
                existing,
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
            )
            and existing.get(
                "source_flash_unified_revision"
            )
            == artifact.get(
                "source_flash_unified_revision"
            )
        ):
            return {
                "stored": True,
                "mode": _MODE,
                "decision": "surface_reused",
                "reason":
                    "duplicate_recall_surface",
                "duplicate": True,
                "memory_count": existing.get(
                    "memory_count"
                ),
                "source_flash_unified_revision":
                    existing.get(
                        "source_flash_unified_revision"
                    ),
            }

        atomic_write(path, artifact)

    return {
        "stored": True,
        "mode": _MODE,
        "decision": "surface_built",
        "reason": "recall_surface_built",
        "duplicate": False,
        "memory_count": artifact[
            "memory_count"
        ],
        "source_flash_unified_revision":
            artifact.get(
                "source_flash_unified_revision"
            ),
    }


def _empty_model_view() -> dict[str, Any]:
    return {
        "version": _VERSION,
        "mode": _MODE,
        "surfaces": [],
    }


def recall_surface_for_model(
    *,
    conversation_id: str,
    cognitive_request_id: str,
) -> dict[str, Any]:
    """The model-facing representation: memref + cue only.

    No real memory id, no mapping, no bucket path, no metadata, no
    revision and no score is ever returned. Reading is request
    scoped, and the persisted artifact is re-validated in full: a
    tampered conversation / request / mode / revision returns an empty
    surface instead of a usable one.
    """

    if not is_valid_conversation_id(
        conversation_id
    ) or not is_valid_cognitive_request_id(
        cognitive_request_id
    ):
        return _empty_model_view()

    artifact = read_json(
        recall_surface_path(
            conversation_id,
            cognitive_request_id,
        )
    )

    if not is_valid_recall_surface_artifact(
        artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
    ):
        return _empty_model_view()

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

    A memref from another conversation, another request, a random
    guess or a corrupted surface artifact returns None. The mapping is
    never global and never guessable.
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

    if not is_valid_recall_surface_artifact(
        artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
    ):
        return None

    memory_id = artifact["mapping"].get(
        memref
    )

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

    if not is_valid_recall_surface_artifact(
        artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
    ):
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