from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from ombrebrain.context.live_recall_authorization import (
    is_valid_live_recall_result,
    project_live_recall_result,
)
from ombrebrain.context.memory_recall_surface import (
    resolve_recall_ref,
)
from ombrebrain.context.live_recall_usage_ref_receipt import (
    build_usage_ref_receipt,
    persist_usage_ref_receipt,
    read_usage_ref_receipt,
    receipt_matches_binding,
    useref_sha256,
)
from ombrebrain.context.memory_usage_signal import (
    read_memory_usage,
)
from ombrebrain.context.recall_request import (
    find_recall_request_by_fingerprint,
    is_valid_recall_request_artifact,
    read_recall_request,
    recall_request_fingerprint,
)
from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    estimate_tokens,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    is_valid_fingerprint,
    is_valid_recall_id,
    now_iso,
    parse_positive_int,
    read_json,
    state_root,
)
from ombrebrain.context.related_recall import (
    is_valid_related_recall_artifact,
    read_related_recall,
)


# Memory Live Recall Usage Surface v1 — explicit model attribution.
#
#     memory_loaded            != attributable
#     attributable(useref)     != used
#     used                     != reinforced
#
# This module builds exactly ONE new artifact per cognitive request: a
# request-scoped, opaque ``useref`` handle for every recalled item the
# model REALLY saw in a Live Recall projection.
#
# It is NOT a usage record:
#
#   - it never calls ``record_usage()``;
#   - it never creates a lifecycle event;
#   - the model never sees a raw memory id, a recall id, a CID, a RID
#     or a memref -- only ``useref_<32 hex>``.
#
# A ``useref`` is minted with ``secrets.token_hex(16)`` and encodes
# nothing: not a memory id, not a recall id, not the request identity.
# Rank is never an authorization and a memref is never a usage
# authorization.
#
# The surface covers ONLY the items of the validated model projection
# (``is_valid_live_recall_result``). A Related Recall may have loaded
# 5 memories while the bounded projection output 3: the 2 dropped
# items can never obtain a ``useref``, because the model never saw
# them.
#
# The provenance chain is exact, never a guess:
#
#     anchor_ref (memref)
#       -> resolve_recall_ref(CID, RID, memref)   -> raw anchor id
#         -> recall_request_fingerprint(...)
#           -> find_recall_request_by_fingerprint  -> real recall id
#             -> read_related_recall(...)          -> real artifact
#               -> read_memory_usage(...)          -> real loaded set
#
# A STRUCTURAL valid surface proves nothing about its content, so every
# mapping is re-validated against the real persisted sources before a
# ``useref`` may ever become ``used``:
#
#   - the real Recall Request is re-read and its fingerprint is
#     RECOMPUTED (a stored 64-hex string is never trusted as
#     provenance);
#   - the real Related Recall is re-read and re-validated, and the
#     mapping's ``rank`` must re-bind to the SAME raw ``memory_id``
#     whose internal ``reason`` matches the mapping's ``kind``;
#   - the real Usage Signal is re-read and the memory must still be in
#     ``loaded_memory_ids`` (``used ⊆ loaded``);
#   - the model projection is RECOMPUTED from the real artifact and
#     the mapping's ``source_model_result_sha256`` must equal its
#     canonical digest, and the mapping's ``rank`` + ``kind`` must
#     exist in that projection.
#
# So a structurally-valid tamper (a swapped ``memory_id``, a rewritten
# 64-hex fingerprint / digest, a shifted rank, a flipped kind or a
# mapping borrowed from another real recall of the same request) can
# never become an attribution handle.
#
# No glob, no rglob, no "most recent related recall", no global
# ``useref`` index, no retrieval, no memory content load, no model
# call, no answer text, no similarity and no lifecycle mutation.
#
# The internal artifact may carry ``recall_id`` + ``memory_id`` because
# it is a 0600, request-scoped provenance file. It never carries
# memory content, a cue, user text, assistant text or a query.

_VERSION = "memory-live-recall-usage-surface.v1"
_MODE = "explicit_model_attribution"

_USAGEREF_RE = re.compile(r"^useref_[0-9a-f]{32}$")

# Model-facing refs budget. Small by design, and the hard caps can
# never be raised from the environment.
_ENV_REFS_MAX = (
    "OMBRE_GATEWAY_CONTEXT_USAGE_REFS_MAX"
)
_DEFAULT_REFS_MAX = 5
_HARD_REFS_MAX = 8

_ENV_REFS_TOKEN_BUDGET = (
    "OMBRE_GATEWAY_CONTEXT_USAGE_REFS_TOKEN_BUDGET"
)
_DEFAULT_REFS_TOKEN_BUDGET = 160
_HARD_REFS_TOKEN_BUDGET = 192

_KIND_ANCHOR = "anchor"
_KIND_RELATED = "related"

_REASON_ANCHOR = "anchor_memory"
_REASON_RELATED = "related_memory"

_KIND_TO_REASON = {
    _KIND_ANCHOR: _REASON_ANCHOR,
    _KIND_RELATED: _REASON_RELATED,
}

_MAPPING_KEYS = frozenset(
    {
        "useref",
        "recall_id",
        "memory_id",
        "rank",
        "kind",
        "source_request_fingerprint",
        "source_model_result_sha256",
    }
)

# Safe build reasons — a closed enum. No filesystem error, exception
# message, memory id or request identity is ever surfaced.
_REASON_BUILT = "usage_surface_built"
_REASON_REUSED = "duplicate_usage_surface"
_REASON_INVALID_IDENTITY = "invalid_request_identity"
_REASON_INVALID_MODEL_RESULT = "invalid_model_result"
_REASON_SURFACE_BINDING = (
    "recall_surface_binding_invalid"
)
_REASON_REQUEST_NOT_FOUND = "recall_request_not_found"
_REASON_RECALL_NOT_FOUND = "related_recall_not_found"
_REASON_USAGE_NOT_FOUND = "usage_signal_not_found"
_REASON_RANK_NOT_FOUND = "rank_not_found"
_REASON_KIND_MISMATCH = "kind_mismatch"
_REASON_MEMORY_NOT_LOADED = "memory_not_loaded"
_REASON_PROJECTION_UNAVAILABLE = (
    "model_projection_unavailable"
)
_REASON_PROJECTION_MISMATCH = (
    "model_projection_mismatch"
)
_REASON_ARTIFACT_INVALID = "usage_surface_invalid"
_REASON_PERSISTENCE_FAILED = (
    "usage_surface_persistence_failed"
)
_REASON_UNAVAILABLE = "usage_surface_unavailable"
_REASON_REF_BINDING_INVALID = (
    "usage_ref_binding_invalid"
)

SAFE_BUILD_REASONS = frozenset(
    {
        _REASON_BUILT,
        _REASON_REUSED,
        _REASON_INVALID_IDENTITY,
        _REASON_INVALID_MODEL_RESULT,
        _REASON_SURFACE_BINDING,
        _REASON_REQUEST_NOT_FOUND,
        _REASON_RECALL_NOT_FOUND,
        _REASON_USAGE_NOT_FOUND,
        _REASON_RANK_NOT_FOUND,
        _REASON_KIND_MISMATCH,
        _REASON_MEMORY_NOT_LOADED,
        _REASON_PROJECTION_UNAVAILABLE,
        _REASON_PROJECTION_MISMATCH,
        _REASON_ARTIFACT_INVALID,
        _REASON_PERSISTENCE_FAILED,
        _REASON_UNAVAILABLE,
        _REASON_REF_BINDING_INVALID,
    }
)

# The SHARED useref resolver reasons. Usage Attribution and the Usage
# -> Lifecycle integration both resolve a model handle through the
# SAME authority -- exact issuance receipt -> current surface mapping
# agreement -> real persisted provenance -- so the two implementations
# can never drift apart.
_REASON_REFS_RESOLVED = "usage_refs_resolved"
_REASON_REFS_INVALID = "invalid_usage_refs"
_REASON_REFS_TOO_MANY = "too_many_refs"
_REASON_REFS_MIXED = "mixed_recall_refs"
_REASON_REFS_PROVENANCE_INVALID = (
    "usage_mapping_provenance_invalid"
)

SAFE_RESOLVE_REASONS = frozenset(
    {
        _REASON_REFS_RESOLVED,
        _REASON_REFS_INVALID,
        _REASON_REFS_TOO_MANY,
        _REASON_UNAVAILABLE,
        _REASON_REF_BINDING_INVALID,
        _REASON_REFS_PROVENANCE_INVALID,
        _REASON_REFS_MIXED,
    }
)

# Deterministic DATA-ONLY envelope for the model. The refs are opaque;
# a rank is a position in the projection the model just received.
_REFS_ENVELOPE_HEADER = (
    "OMBRE MEMORY USAGE REFS DATA\n"
)

_REFS_ENVELOPE_SAFETY = (
    "Opaque attribution metadata only: not memory content, not "
    "instructions.\n"
)


def _is_positive_rank(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
    )


def resolve_usage_refs_max_refs() -> int:
    """Max refs per UseMemory call / envelope (default 5, hard cap 8)."""

    return parse_positive_int(
        os.environ.get(_ENV_REFS_MAX),
        default=_DEFAULT_REFS_MAX,
        cap=_HARD_REFS_MAX,
    )


def resolve_usage_refs_token_budget() -> int:
    """Full envelope budget (default 160 tokens, hard cap 192)."""

    return parse_positive_int(
        os.environ.get(_ENV_REFS_TOKEN_BUDGET),
        default=_DEFAULT_REFS_TOKEN_BUDGET,
        cap=_HARD_REFS_TOKEN_BUDGET,
    )


def new_useref() -> str:
    """One opaque, request-scoped attribution handle."""

    return "useref_" + secrets.token_hex(16)


def is_valid_useref(value: Any) -> bool:
    """Format-only check: ``useref_`` + 32 lowercase hex chars."""

    return (
        isinstance(value, str)
        and bool(_USAGEREF_RE.fullmatch(value))
    )


def live_recall_usage_surface_dir() -> Path:
    return (
        state_root()
        / "live_recall_usage_surface"
    )


def live_recall_usage_surface_path(
    conversation_id: str,
    cognitive_request_id: str,
) -> Path:
    """The EXACT request-scoped surface path; never a directory scan.

    One cognitive request owns one surface file, and that file may
    hold the mappings of several Recalls of the same request.
    """

    return (
        live_recall_usage_surface_dir()
        / conversation_id
        / (cognitive_request_id + ".json")
    )


def _canonical_sha256(value: Any) -> str:
    """Deterministic canonical-JSON hash of a model projection.

    The surface stores only this digest, so it can prove which exact
    projection a mapping was built from without copying a single
    character of memory content.
    """

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        return ""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _is_nonempty_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
    )


# ------------------------------------------------------
# Artifact validation
# ------------------------------------------------------


def is_valid_live_recall_usage_surface_artifact(
    artifact: Any,
    *,
    conversation_id: Any = None,
    cognitive_request_id: Any = None,
) -> bool:
    """Independent structural identity of one Usage Surface.

    Reading the right path proves nothing, so every reader
    re-validates the contract, the mode, the request identity, the
    timestamps and every mapping:

      - ``useref`` format, never a duplicate;
      - ``recall_id`` format;
      - a non-empty ``memory_id``;
      - a positive ``rank``, unique per ``recall_id``;
      - the ``anchor`` / ``related`` kind enum;
      - a full sha256 request fingerprint and model result digest.

    A duplicate ``useref`` or a duplicate ``recall_id`` + ``rank``
    makes the whole artifact invalid: a tampered handle table must
    never resolve to a memory.
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
        not is_valid_conversation_id(
            artifact_conversation
        )
        or not is_valid_cognitive_request_id(
            artifact_request
        )
    ):
        return False

    if (
        conversation_id is not None
        and artifact_conversation != conversation_id
    ):
        return False

    if (
        cognitive_request_id is not None
        and artifact_request != cognitive_request_id
    ):
        return False

    created_at = artifact.get("created_at")
    updated_at = artifact.get("updated_at")

    if (
        not _is_nonempty_text(created_at)
        or not _is_nonempty_text(updated_at)
    ):
        return False

    mappings = artifact.get("mappings")

    if not isinstance(mappings, list) or not mappings:
        return False

    seen_refs: set[str] = set()
    seen_rank_keys: set[tuple[str, int]] = set()

    for mapping in mappings:
        if not _is_valid_mapping_shape(mapping):
            return False

        useref = mapping["useref"]

        if useref in seen_refs:
            return False

        seen_refs.add(useref)

        rank_key = (mapping["recall_id"], mapping["rank"])

        if rank_key in seen_rank_keys:
            return False

        seen_rank_keys.add(rank_key)

    return True


def _is_valid_mapping_shape(
    mapping: Any,
) -> bool:
    """The structural shape of ONE mapping, as persisted.

    This is only the SHAPE: it can never prove that the mapping came
    from the real sources, so a reader that is about to act on a
    mapping must also pass ``validate_live_recall_usage_
    mapping_provenance`` (or the surface-level equivalent).
    """

    if not isinstance(mapping, dict):
        return False

    if set(mapping.keys()) != _MAPPING_KEYS:
        return False

    if not is_valid_useref(mapping.get("useref")):
        return False

    if not is_valid_recall_id(mapping.get("recall_id")):
        return False

    if not _is_nonempty_text(mapping.get("memory_id")):
        return False

    if not _is_positive_rank(mapping.get("rank")):
        return False

    if mapping.get("kind") not in (
        _KIND_ANCHOR,
        _KIND_RELATED,
    ):
        return False

    if not is_valid_fingerprint(
        mapping.get("source_request_fingerprint")
    ):
        return False

    return is_valid_fingerprint(
        mapping.get("source_model_result_sha256")
    )


def read_live_recall_usage_surface(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
) -> dict[str, Any] | None:
    """Read THIS request's surface, fully re-validated.

    Returns None when the surface is missing OR malformed. A corrupt
    surface is never handed out as if it were valid, and it is never
    overwritten by the builder.
    """

    if (
        not is_valid_conversation_id(conversation_id)
        or not is_valid_cognitive_request_id(
            cognitive_request_id
        )
    ):
        return None

    artifact = read_json(
        live_recall_usage_surface_path(
            conversation_id,
            cognitive_request_id,
        )
    )

    if not is_valid_live_recall_usage_surface_artifact(
        artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
    ):
        return None

    return artifact


def live_recall_usage_surface_status(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
) -> dict[str, Any]:
    """Privacy-safe status of one surface artifact."""

    artifact = read_live_recall_usage_surface(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
    )

    if not isinstance(artifact, dict):
        return {"exists": False}

    mappings = artifact.get("mappings") or []

    return {
        "exists": True,
        "version": artifact.get("version"),
        "mode": artifact.get("mode"),
        "mapping_count": len(mappings),
        "recall_count": len(
            {
                mapping.get("recall_id")
                for mapping in mappings
                if isinstance(mapping, dict)
            }
        ),
    }


# ------------------------------------------------------
# Provenance (re-read the REAL sources)
# ------------------------------------------------------


def _verified_sources(
    *,
    request_fingerprint: str,
    recall_artifact: dict[str, Any],
    loaded_memory_ids: set[str],
    projection: dict[str, Any],
    model_result_sha256: str,
) -> dict[str, Any]:
    """The verified authority a mapping must match."""

    return {
        "request_fingerprint": request_fingerprint,
        "related_by_rank": {
            item["rank"]: item
            for item in recall_artifact["memories"]
            if isinstance(item, dict)
            and _is_positive_rank(item.get("rank"))
        },
        "loaded_memory_ids": set(loaded_memory_ids),
        "projection_by_rank": {
            item["rank"]: item
            for item in projection["memories"]
            if isinstance(item, dict)
            and _is_positive_rank(item.get("rank"))
        },
        "model_result_sha256": model_result_sha256,
    }


def _mapping_matches_sources(
    mapping: Any,
    sources: dict[str, Any],
) -> bool:
    """Does ONE mapping re-bind to the verified real sources?

    Every provenance field is re-derived, never trusted:
    ``source_request_fingerprint`` must equal the RECOMPUTED request
    fingerprint, ``source_model_result_sha256`` must equal the
    RECOMPUTED projection digest, the ``rank`` must re-bind to the
    SAME raw ``memory_id`` (with a matching internal ``reason``), the
    memory must still be loaded and the ``rank`` + ``kind`` must exist
    in the recomputed model projection.
    """

    if not _is_valid_mapping_shape(mapping):
        return False

    if (
        mapping["source_request_fingerprint"]
        != sources.get("request_fingerprint")
    ):
        return False

    if (
        mapping["source_model_result_sha256"]
        != sources.get("model_result_sha256")
    ):
        return False

    item = sources["related_by_rank"].get(
        mapping["rank"]
    )

    if not isinstance(item, dict):
        return False

    if item.get("memory_id") != mapping["memory_id"]:
        return False

    if item.get("reason") != _KIND_TO_REASON.get(
        mapping["kind"]
    ):
        return False

    if (
        mapping["memory_id"]
        not in sources["loaded_memory_ids"]
    ):
        return False

    projected = sources["projection_by_rank"].get(
        mapping["rank"]
    )

    if not isinstance(projected, dict):
        return False

    return projected.get("kind") == mapping["kind"]


def _read_verified_sources(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    recall_id: Any,
) -> dict[str, Any] | None:
    """Re-read and re-validate the REAL persisted provenance chain.

    Recall Request -> recomputed fingerprint -> Related Recall ->
    Usage Signal -> recomputed model projection. Any missing or
    corrupt link returns None, so a mapping can never be validated
    against a guess.
    """

    if (
        not is_valid_conversation_id(conversation_id)
        or not is_valid_cognitive_request_id(
            cognitive_request_id
        )
        or not is_valid_recall_id(recall_id)
    ):
        return None

    request_artifact = read_recall_request(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if not is_valid_recall_request_artifact(
        request_artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
    ):
        return None

    # The fingerprint is RECOMPUTED from the real artifact; the
    # string a surface stores about itself is never trusted.
    fingerprint = recall_request_fingerprint(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        anchor_memory_id=request_artifact.get(
            "anchor_memory_id"
        ),
        requested_scope=request_artifact.get(
            "requested_scope"
        ),
    )

    if request_artifact.get(
        "request_fingerprint"
    ) != fingerprint:
        return None

    recall_artifact = read_related_recall(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if not is_valid_related_recall_artifact(
        recall_artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
        request_fingerprint=fingerprint,
    ):
        return None

    usage = read_memory_usage(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if not isinstance(usage, dict):
        return None

    loaded = {
        memory_id
        for memory_id in (
            usage.get("loaded_memory_ids") or []
        )
        if _is_nonempty_text(memory_id)
    }

    # The projection is RECOMPUTED from the real artifact.
    projection = project_live_recall_result(
        artifact=recall_artifact
    )

    if (
        not isinstance(projection, dict)
        or not is_valid_live_recall_result(projection)
    ):
        return None

    projection_sha = _canonical_sha256(projection)

    if not is_valid_fingerprint(projection_sha):
        return None

    return _verified_sources(
        request_fingerprint=fingerprint,
        recall_artifact=recall_artifact,
        loaded_memory_ids=loaded,
        projection=projection,
        model_result_sha256=projection_sha,
    )


def validate_live_recall_usage_mapping_provenance(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    mapping: Any,
) -> bool:
    """Re-validate ONE mapping against the real persisted sources.

    A structurally valid mapping is NOT trustworthy: it may have been
    edited after it was written. This re-reads the real Recall
    Request, Related Recall and Usage Signal, recomputes the request
    fingerprint and the model projection, and requires the mapping to
    re-bind to all of them. Nothing the surface declares about itself
    is treated as authority.
    """

    if not _is_valid_mapping_shape(mapping):
        return False

    try:
        sources = _read_verified_sources(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            recall_id=mapping["recall_id"],
        )
    except Exception:
        return False

    if not isinstance(sources, dict):
        return False

    return _mapping_matches_sources(mapping, sources)


def validate_live_recall_usage_surface_provenance(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    artifact: Any = None,
    mappings: Any = None,
) -> bool:
    """Re-validate every selected mapping of one Usage Surface.

    The real sources are read ONCE per recall id and reused for that
    recall's mappings. Any invalid mapping makes the whole selection
    invalid -- the caller then fails closed and records nothing.
    """

    if artifact is None:
        artifact = read_live_recall_usage_surface(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
        )

    if not is_valid_live_recall_usage_surface_artifact(
        artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
    ):
        return False

    if mappings is None:
        selected = list(artifact["mappings"])
    elif isinstance(mappings, (list, tuple)):
        selected = list(mappings)
    else:
        return False

    if not selected:
        return False

    cache: dict[str, dict[str, Any] | None] = {}

    for mapping in selected:
        if not _is_valid_mapping_shape(mapping):
            return False

        recall_id = mapping["recall_id"]

        if recall_id not in cache:
            try:
                cache[recall_id] = _read_verified_sources(
                    conversation_id=conversation_id,
                    cognitive_request_id=(
                        cognitive_request_id
                    ),
                    recall_id=recall_id,
                )
            except Exception:
                return False

        sources = cache[recall_id]

        if not isinstance(sources, dict):
            return False

        if not _mapping_matches_sources(mapping, sources):
            return False

    return True


# ------------------------------------------------------
# Receipt-bound useref resolution (the shared authority)
# ------------------------------------------------------


def _mapping_binding(
    mapping: Any,
) -> dict[str, Any] | None:
    """The comparable binding of ONE surface mapping, or None."""

    if not _is_valid_mapping_shape(mapping):
        return None

    return {
        "recall_id": mapping["recall_id"],
        "memory_id": mapping["memory_id"],
        "rank": mapping["rank"],
        "kind": mapping["kind"],
        "source_request_fingerprint": mapping[
            "source_request_fingerprint"
        ],
        "source_model_result_sha256": mapping[
            "source_model_result_sha256"
        ],
    }


def _mapping_has_matching_receipt(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    mapping: Any,
) -> bool:
    """Is there a valid, EXACTLY matching issuance receipt?

    The receipt is read by its exact ``sha256(useref)`` path -- never
    a scan -- and must agree with the mapping on every binding field.
    A missing, corrupt, copied or edited receipt means False.
    """

    binding = _mapping_binding(mapping)

    if binding is None:
        return False

    ref_sha = useref_sha256(mapping["useref"])

    if not is_valid_fingerprint(ref_sha):
        return False

    receipt = read_usage_ref_receipt(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        ref_sha256=ref_sha,
    )

    if not isinstance(receipt, dict):
        return False

    return receipt_matches_binding(receipt, binding)


def _persist_mapping_receipt(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    mapping: Any,
) -> bool:
    """Persist, then re-read, the issuance receipt of one mapping.

    Persistence-before-exposure: the receipt is written first and only
    a re-read that proves it really exists and matches is accepted.
    """

    binding = _mapping_binding(mapping)

    if binding is None:
        return False

    ref_sha = useref_sha256(mapping["useref"])

    if not is_valid_fingerprint(ref_sha):
        return False

    receipt = build_usage_ref_receipt(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        ref_sha256=ref_sha,
        recall_id=binding["recall_id"],
        memory_id=binding["memory_id"],
        rank=binding["rank"],
        kind=binding["kind"],
        source_request_fingerprint=binding[
            "source_request_fingerprint"
        ],
        source_model_result_sha256=binding[
            "source_model_result_sha256"
        ],
    )

    try:
        outcome = persist_usage_ref_receipt(receipt)
    except Exception:
        return False

    if not outcome.get("stored"):
        return False

    return _mapping_has_matching_receipt(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        mapping=mapping,
    )


def resolve_verified_live_recall_usage_refs(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    userefs: Any,
) -> dict[str, Any]:
    """The ONE trusted way a model ``useref`` becomes a real mapping.

    Usage Attribution and the Usage -> Lifecycle integration both call
    this, so a handle resolves through the SAME authority:

        format -> exact current surface
          -> real persisted provenance (fingerprint + projection
             RECOMPUTED from the real artifacts)
            -> independent write-once issuance receipt agreement

    A surface-only tamper (a swapped ``useref`` included) is refused:
    the surface can never be its own authority for the ORIGINAL
    binding. Returns a privacy-safe report whose ``mappings`` are the
    verified surface mappings in surface order, plus the single
    ``recall_id`` they agree on.
    """

    requested = (
        len(userefs)
        if isinstance(userefs, (list, tuple))
        else 0
    )

    report: dict[str, Any] = {
        "version": _VERSION,
        "mode": _MODE,
        "ok": False,
        "reason": _REASON_REFS_INVALID,
        "requested_ref_count": requested,
        "resolved_ref_count": 0,
        "recall_ids": (),
        "recall_id": None,
        "mappings": (),
    }

    if not isinstance(userefs, (list, tuple)):
        return report

    if requested == 0:
        return report

    if requested > resolve_usage_refs_max_refs():
        report["reason"] = _REASON_REFS_TOO_MANY
        return report

    wanted: set[str] = set()

    for value in userefs:
        if not is_valid_useref(value):
            return report

        wanted.add(value)

    surface = read_live_recall_usage_surface(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
    )

    if not isinstance(surface, dict):
        report["reason"] = _REASON_UNAVAILABLE
        return report

    mappings = surface.get("mappings") or []

    # Surface order, never argument order; a ref that is not in THIS
    # request's exact surface is simply invalid.
    selected = [
        mapping
        for mapping in mappings
        if isinstance(mapping, dict)
        and mapping.get("useref") in wanted
    ]

    if len(
        {mapping.get("useref") for mapping in selected}
    ) != len(wanted):
        return report

    # The REAL persisted provenance first: a surface-only tamper is
    # still reported exactly as before.
    if not validate_live_recall_usage_surface_provenance(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        artifact=surface,
        mappings=selected,
    ):
        report["reason"] = (
            _REASON_REFS_PROVENANCE_INVALID
        )
        return report

    # Then the independent, write-once issuance receipt must agree
    # EXACTLY with the current surface mapping.
    for mapping in selected:
        if not _mapping_has_matching_receipt(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            mapping=mapping,
        ):
            report["reason"] = (
                _REASON_REF_BINDING_INVALID
            )
            return report

    ordered_recall_ids: list[Any] = []

    for mapping in selected:
        recall_id = mapping.get("recall_id")

        if recall_id not in ordered_recall_ids:
            ordered_recall_ids.append(recall_id)

    if len(ordered_recall_ids) != 1:
        report["reason"] = _REASON_REFS_MIXED
        return report

    report["ok"] = True
    report["reason"] = _REASON_REFS_RESOLVED
    report["resolved_ref_count"] = len(selected)
    report["recall_ids"] = tuple(ordered_recall_ids)
    report["recall_id"] = ordered_recall_ids[0]
    report["mappings"] = tuple(selected)

    return report


# ------------------------------------------------------
# Builder
# ------------------------------------------------------


def _no_surface(
    reason: str,
) -> dict[str, Any]:
    return {
        "version": _VERSION,
        "mode": _MODE,
        "stored": False,
        "decision": "refused",
        "reason": reason,
        "usage_refs": [],
        "mapped_count": 0,
        "reused_count": 0,
    }


def _targets(
    *,
    model_result: dict[str, Any],
    recall_artifact: dict[str, Any],
    projection: dict[str, Any],
    loaded_memory_ids: set[str],
    recall_id: str,
) -> tuple[list[dict[str, Any]] | None, str]:
    """Map model ranks onto the REAL Related Recall artifact.

    Only the exact same rank may be mapped, and the model-facing
    ``kind`` must agree with the internal ``reason`` -- rank alone is
    never enough. The (rank, kind) pair must also exist in the
    projection RECOMPUTED from the real artifact, so an item the
    Recall loaded but the bounded projection dropped can never obtain
    a ``useref`` -- and the builder can never mint a mapping the
    provenance validator would have to reject.

    Returns ``(targets, reason)``; ``targets`` is None on refusal and
    the reason is one of the safe enum values.
    """

    by_rank = {
        item["rank"]: item
        for item in recall_artifact["memories"]
        if isinstance(item, dict)
        and _is_positive_rank(item.get("rank"))
    }

    projection_by_rank = {
        item["rank"]: item
        for item in projection["memories"]
        if isinstance(item, dict)
        and _is_positive_rank(item.get("rank"))
    }

    targets: list[dict[str, Any]] = []

    for item in model_result["memories"]:
        rank = item["rank"]

        source = by_rank.get(rank)

        if not isinstance(source, dict):
            return None, _REASON_RANK_NOT_FOUND

        if source.get("reason") != _KIND_TO_REASON.get(
            item["kind"]
        ):
            return None, _REASON_KIND_MISMATCH

        projected = projection_by_rank.get(rank)

        if (
            not isinstance(projected, dict)
            or projected.get("kind") != item["kind"]
        ):
            return None, _REASON_PROJECTION_MISMATCH

        memory_id = source.get("memory_id")

        if not _is_nonempty_text(memory_id):
            return None, _REASON_MEMORY_NOT_LOADED

        memory_id = memory_id.strip()

        if memory_id not in loaded_memory_ids:
            return None, _REASON_MEMORY_NOT_LOADED

        targets.append(
            {
                "rank": rank,
                "kind": item["kind"],
                "memory_id": memory_id,
                "recall_id": recall_id,
            }
        )

    if not targets:
        return None, _REASON_INVALID_MODEL_RESULT

    return targets, _REASON_BUILT


def _unique_useref(
    taken: set[str],
) -> str:
    while True:
        candidate = new_useref()

        if candidate not in taken:
            return candidate


def build_live_recall_usage_surface(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    anchor_ref: Any,
    model_result: Any,
) -> dict[str, Any]:
    """Build (and persist) the request-scoped Usage Surface.

    Returns a privacy-safe report whose ``usage_refs`` carry only
    ``rank`` + ``useref``. The refs are returned ONLY after the
    artifact was atomically persisted AND re-read as valid; a
    persistence failure returns no refs while the Recall itself stays
    successful.

    This never records ``used``, never reinforces anything and never
    creates a lifecycle event: it only turns a loaded-and-exposed
    recalled item into an opaque attribution handle.
    """

    if (
        not is_valid_conversation_id(conversation_id)
        or not is_valid_cognitive_request_id(
            cognitive_request_id
        )
    ):
        return _no_surface(_REASON_INVALID_IDENTITY)

    try:
        # The projection must be a validated model result; a caller
        # can never smuggle its own ranks or content in.
        if not is_valid_live_recall_result(
            model_result
        ):
            return _no_surface(
                _REASON_INVALID_MODEL_RESULT
            )

        # The model names the anchor only as an opaque memref; the raw
        # anchor id is resolved internally for THIS request.
        anchor_memory_id = resolve_recall_ref(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            memref=anchor_ref,
        )

        if not _is_nonempty_text(anchor_memory_id):
            return _no_surface(
                _REASON_SURFACE_BINDING
            )

        anchor_memory_id = anchor_memory_id.strip()

        fingerprint = recall_request_fingerprint(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            anchor_memory_id=anchor_memory_id,
            requested_scope="related",
        )

        request_artifact = (
            find_recall_request_by_fingerprint(
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
                fingerprint=fingerprint,
            )
        )

        if (
            not is_valid_recall_request_artifact(
                request_artifact,
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
            )
            or request_artifact.get(
                "anchor_memory_id"
            )
            != anchor_memory_id
        ):
            return _no_surface(
                _REASON_REQUEST_NOT_FOUND
            )

        recall_id = request_artifact.get("recall_id")

        if not is_valid_recall_id(recall_id):
            return _no_surface(
                _REASON_REQUEST_NOT_FOUND
            )

        recall_artifact = read_related_recall(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            recall_id=recall_id,
        )

        if not is_valid_related_recall_artifact(
            recall_artifact,
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            recall_id=recall_id,
            request_fingerprint=fingerprint,
        ):
            return _no_surface(
                _REASON_RECALL_NOT_FOUND
            )

        usage = read_memory_usage(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            recall_id=recall_id,
        )

        if not isinstance(usage, dict):
            return _no_surface(
                _REASON_USAGE_NOT_FOUND
            )

        loaded = {
            memory_id
            for memory_id in (
                usage.get("loaded_memory_ids") or []
            )
            if _is_nonempty_text(memory_id)
        }

        # The model projection is RECOMPUTED from the real artifact:
        # the surface may only ever describe the projection this
        # Recall really yields, and the digest it stores is the
        # RECOMPUTED one -- never a digest a caller supplied.
        projection = project_live_recall_result(
            artifact=recall_artifact
        )

        if (
            not isinstance(projection, dict)
            or not is_valid_live_recall_result(projection)
        ):
            return _no_surface(
                _REASON_PROJECTION_UNAVAILABLE
            )

        source_sha = _canonical_sha256(projection)

        if not is_valid_fingerprint(source_sha):
            return _no_surface(
                _REASON_PROJECTION_UNAVAILABLE
            )

        targets, targets_reason = _targets(
            model_result=model_result,
            recall_artifact=recall_artifact,
            projection=projection,
            loaded_memory_ids=loaded,
            recall_id=recall_id,
        )

        if targets is None:
            return _no_surface(targets_reason)

        path = live_recall_usage_surface_path(
            conversation_id,
            cognitive_request_id,
        )

        reused = 0

        with LOCK:
            existing = read_json(path)

            if existing is None:
                artifact: dict[str, Any] = {
                    "version": _VERSION,
                    "mode": _MODE,
                    "conversation_id": conversation_id,
                    "cognitive_request_id": (
                        cognitive_request_id
                    ),
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                    "mappings": [],
                }
            elif is_valid_live_recall_usage_surface_artifact(
                existing,
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
            ):
                artifact = existing
            else:
                # A corrupt surface is never overwritten / repaired.
                return _no_surface(
                    _REASON_ARTIFACT_INVALID
                )

            by_key = {
                (
                    mapping["recall_id"],
                    mapping["memory_id"],
                    mapping["rank"],
                ): mapping
                for mapping in artifact["mappings"]
            }

            taken_refs = {
                mapping["useref"]
                for mapping in artifact["mappings"]
            }

            new_mappings: list[dict[str, Any]] = []

            for target in targets:
                key = (
                    target["recall_id"],
                    target["memory_id"],
                    target["rank"],
                )

                found = by_key.get(key)

                if isinstance(found, dict):
                    # A duplicate Recall reuses the original useref, so
                    # a retry yields identical attribution refs -- but
                    # ONLY while the independent issuance receipt still
                    # exists and exactly matches. A missing / corrupt /
                    # conflicting receipt is never re-minted or
                    # repaired: the whole build fails closed.
                    if not _mapping_has_matching_receipt(
                        conversation_id=conversation_id,
                        cognitive_request_id=(
                            cognitive_request_id
                        ),
                        mapping=found,
                    ):
                        return _no_surface(
                            _REASON_REF_BINDING_INVALID
                        )

                    reused += 1
                    target["useref"] = found["useref"]
                    continue

                useref = _unique_useref(taken_refs)

                taken_refs.add(useref)

                mapping = {
                    "useref": useref,
                    "recall_id": target["recall_id"],
                    "memory_id": target["memory_id"],
                    "rank": target["rank"],
                    "kind": target["kind"],
                    "source_request_fingerprint": (
                        fingerprint
                    ),
                    "source_model_result_sha256": (
                        source_sha
                    ),
                }

                # Persistence-before-exposure: the write-once issuance
                # receipt is persisted (and re-read) BEFORE the ref can
                # ever be added to the surface or returned.
                if not _persist_mapping_receipt(
                    conversation_id=conversation_id,
                    cognitive_request_id=(
                        cognitive_request_id
                    ),
                    mapping=mapping,
                ):
                    return _no_surface(
                        _REASON_REF_BINDING_INVALID
                    )

                by_key[key] = mapping

                new_mappings.append(mapping)

                target["useref"] = useref

            if new_mappings:
                candidate = dict(artifact)

                candidate["mappings"] = (
                    list(artifact["mappings"])
                    + new_mappings
                )

                candidate["updated_at"] = now_iso()

                if not (
                    is_valid_live_recall_usage_surface_artifact(
                        candidate,
                        conversation_id=(
                            conversation_id
                        ),
                        cognitive_request_id=(
                            cognitive_request_id
                        ),
                    )
                ):
                    return _no_surface(
                        _REASON_ARTIFACT_INVALID
                    )

                try:
                    atomic_write(path, candidate)
                except Exception:
                    return _no_surface(
                        _REASON_PERSISTENCE_FAILED
                    )

            # Persistence-before-exposure: the refs may only leave
            # after a re-read proves they are really persisted.
            persisted = read_json(path)

            if not (
                is_valid_live_recall_usage_surface_artifact(
                    persisted,
                    conversation_id=conversation_id,
                    cognitive_request_id=(
                        cognitive_request_id
                    ),
                )
            ):
                return _no_surface(
                    _REASON_PERSISTENCE_FAILED
                )

            persisted_by_key = {
                (
                    mapping["recall_id"],
                    mapping["memory_id"],
                    mapping["rank"],
                ): mapping
                for mapping in persisted["mappings"]
            }

            usage_refs: list[dict[str, Any]] = []

            for target in targets:
                persisted_mapping = persisted_by_key.get(
                    (
                        target["recall_id"],
                        target["memory_id"],
                        target["rank"],
                    )
                )

                if (
                    not isinstance(
                        persisted_mapping, dict
                    )
                    or persisted_mapping.get("useref")
                    != target.get("useref")
                ):
                    return _no_surface(
                        _REASON_PERSISTENCE_FAILED
                    )

                # The persisted mapping's independent issuance receipt
                # must still exist and agree exactly before ANY ref is
                # exposed to the model.
                if not _mapping_has_matching_receipt(
                    conversation_id=conversation_id,
                    cognitive_request_id=(
                        cognitive_request_id
                    ),
                    mapping=persisted_mapping,
                ):
                    return _no_surface(
                        _REASON_REF_BINDING_INVALID
                    )

                usage_refs.append(
                    {
                        "rank": target["rank"],
                        "useref": persisted_mapping[
                            "useref"
                        ],
                    }
                )

        return {
            "version": _VERSION,
            "mode": _MODE,
            "stored": True,
            "decision": (
                "surface_reused"
                if reused == len(targets)
                else "surface_built"
            ),
            "reason": (
                _REASON_REUSED
                if reused == len(targets)
                else _REASON_BUILT
            ),
            "usage_refs": usage_refs,
            "mapped_count": len(usage_refs),
            "reused_count": reused,
        }

    except Exception:
        return _no_surface(_REASON_UNAVAILABLE)


# ------------------------------------------------------
# Model-facing refs envelope
# ------------------------------------------------------


def _render_refs(
    entries: list[dict[str, Any]],
) -> str:
    return (
        _REFS_ENVELOPE_HEADER
        + _REFS_ENVELOPE_SAFETY
        + json.dumps(
            {"usage_refs": entries},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def render_live_recall_usage_refs(
    usage_refs: Any,
) -> str:
    """Render the DATA-ONLY refs envelope appended after Recall data.

    Refuses to emit anything invalid: a malformed entry, a rank that
    is not a positive int or a non-``useref`` handle returns "" (no
    envelope at all). Over budget, only TRAILING entries are dropped --
    a ``useref`` is never truncated, and if even rank 1 does not fit
    there is no envelope (the original Recall result still returns).
    """

    if not isinstance(usage_refs, list):
        return ""

    max_refs = resolve_usage_refs_max_refs()

    entries: list[dict[str, Any]] = []

    for item in usage_refs:
        if not isinstance(item, dict):
            return ""

        rank = item.get("rank")
        useref = item.get("useref")

        if not _is_positive_rank(rank):
            return ""

        if not is_valid_useref(useref):
            return ""

        if any(
            entry["rank"] == rank or entry["useref"] == useref
            for entry in entries
        ):
            return ""

        entries.append(
            {"rank": int(rank), "useref": useref}
        )

        if len(entries) >= max_refs:
            break

    budget = resolve_usage_refs_token_budget()

    while entries:
        rendered = _render_refs(entries)

        if estimate_tokens(rendered) <= budget:
            return rendered

        entries = entries[:-1]

    return ""


def append_live_recall_usage_refs(
    rendered: Any,
    usage_refs: Any,
) -> str:
    """The frozen Recall data, plus the refs envelope when one fits.

    The Recall envelope itself is NEVER modified -- the opaque refs
    are appended after it. With no (or no fitting) refs the original
    rendered Recall data is returned unchanged, so a Usage Attribution
    failure can never fail a Recall.
    """

    base = rendered if isinstance(rendered, str) else ""

    envelope = render_live_recall_usage_refs(usage_refs)

    if not envelope:
        return base

    return base + envelope


def compose_live_recall_result_with_usage_refs(
    *,
    rendered: Any,
    conversation_id: Any,
    cognitive_request_id: Any,
    anchor_ref: Any,
    model_result: Any,
) -> str:
    """The transport Recall result: Recall data + usage refs.

    Usage Attribution is ADDITIVE: whenever the surface cannot be
    built (or cannot be persisted) the original rendered Recall data
    is returned unchanged -- a Recall never fails because of usage
    attribution, and no ref is ever emitted for a surface that was not
    really persisted and re-read.
    """

    base = rendered if isinstance(rendered, str) else ""

    try:
        report = build_live_recall_usage_surface(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            anchor_ref=anchor_ref,
            model_result=model_result,
        )
    except Exception:
        return base

    if not isinstance(report, dict):
        return base

    return append_live_recall_usage_refs(
        base, report.get("usage_refs") or []
    )


__all__ = [
    "SAFE_BUILD_REASONS",
    "SAFE_RESOLVE_REASONS",
    "append_live_recall_usage_refs",
    "build_live_recall_usage_surface",
    "compose_live_recall_result_with_usage_refs",
    "is_valid_live_recall_usage_surface_artifact",
    "is_valid_useref",
    "live_recall_usage_surface_path",
    "live_recall_usage_surface_status",
    "new_useref",
    "read_live_recall_usage_surface",
    "render_live_recall_usage_refs",
    "resolve_usage_refs_max_refs",
    "resolve_usage_refs_token_budget",
    "resolve_verified_live_recall_usage_refs",
    "validate_live_recall_usage_mapping_provenance",
    "validate_live_recall_usage_surface_provenance",
]