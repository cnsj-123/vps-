from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    is_valid_fingerprint,
    is_valid_recall_id,
    now_iso,
    read_json,
    state_root,
)


# Memory Live Recall Usage Ref Receipt v1 — write-once issuance receipt.
#
# The Usage Surface alone must NEVER be the authority for "which
# opaque ``useref`` was originally bound to which recalled item". A
# Surface is mutable and can be tampered with after the refs were
# handed to the model: a surfaced-only provenance check cannot prove
# that ``useref_A`` really belonged to rank 1.
#
#     useref_A -> rank1 / mem-1
#     useref_B -> rank2 / mem-2
#
# This module adds an INDEPENDENT, request-scoped, write-once receipt
# per minted ``useref``. The receipt is persisted BEFORE the ref is
# exposed and is never overwritten, so a later Surface tamper (a
# swapped ``useref`` included) can never change the original binding.
#
#   - the file name is ``sha256(useref).json``: the raw ``useref`` is
#     NEVER a path component and is NEVER stored, so a leaked receipt
#     can never be turned back into a model-visible handle;
#   - the receipt carries only opaque provenance: recall id, memory
#     id, rank, kind, the request fingerprint and the model projection
#     digest. It never carries memory content, a cue, a query, user
#     text or assistant text;
#   - lookup is EXACT-path only: no glob, no rglob, no global useref
#     index, no "most recent" guess;
#   - an existing receipt path is never repaired and never overwritten:
#     a valid receipt with the exact same binding is reused, anything
#     else fails closed.
#
# A receipt is an authorization input, never an authorization result:
# a structurally valid receipt still has to agree with the current
# Usage Surface AND re-bind to the real persisted Recall Request /
# Related Recall / Usage Signal before a ``useref`` may resolve.

RECEIPT_VERSION = (
    "memory-live-recall-usage-ref-receipt.v1"
)

MODE = "opaque_usage_ref_binding"

_KIND_ANCHOR = "anchor"
_KIND_RELATED = "related"

_RECEIPT_KEYS = frozenset(
    {
        "version",
        "mode",
        "conversation_id",
        "cognitive_request_id",
        "useref_sha256",
        "recall_id",
        "memory_id",
        "rank",
        "kind",
        "source_request_fingerprint",
        "source_model_result_sha256",
        "created_at",
    }
)

# The binding fields that must agree, exactly, between a receipt and
# the Usage Surface mapping it was minted for.
BINDING_FIELDS = (
    "recall_id",
    "memory_id",
    "rank",
    "kind",
    "source_request_fingerprint",
    "source_model_result_sha256",
)

_REASON_STORED = "usage_ref_receipt_recorded"
_REASON_REUSED = "duplicate_usage_ref_receipt"
_REASON_INVALID = "invalid_usage_ref_receipt"
_REASON_CONFLICT = (
    "existing_usage_ref_receipt_conflict"
)


def useref_sha256(useref: Any) -> str:
    """Deterministic digest of an opaque ``useref``.

    The raw handle is never used as a path component and is never
    persisted. A non-string / empty value degrades to "" (an invalid
    digest), so a caller can never accidentally path-traverse.
    """

    if not isinstance(useref, str) or not useref:
        return ""

    return hashlib.sha256(
        useref.encode("utf-8")
    ).hexdigest()


def usage_ref_receipt_dir(
    conversation_id: Any,
    cognitive_request_id: Any,
) -> Path | None:
    if (
        not is_valid_conversation_id(conversation_id)
        or not is_valid_cognitive_request_id(
            cognitive_request_id
        )
    ):
        return None

    return (
        state_root()
        / "live_recall_usage_ref_receipt"
        / conversation_id
        / cognitive_request_id
    )


def usage_ref_receipt_path(
    conversation_id: Any,
    cognitive_request_id: Any,
    ref_sha256: Any,
) -> Path | None:
    """The EXACT request-scoped receipt path for one ref digest."""

    directory = usage_ref_receipt_dir(
        conversation_id, cognitive_request_id
    )

    if directory is None:
        return None

    if not is_valid_fingerprint(ref_sha256):
        return None

    return directory / (ref_sha256 + ".json")


def _is_positive_rank(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
    )


def _is_nonempty_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
    )


def is_valid_usage_ref_receipt(
    artifact: Any,
    *,
    conversation_id: Any = None,
    cognitive_request_id: Any = None,
    ref_sha256: Any = None,
) -> bool:
    """Independent structural identity of one issuance receipt.

    Reading the right path proves nothing: every field is re-checked,
    so a copied, renamed or edited receipt is refused.
    """

    if not isinstance(artifact, dict):
        return False

    if set(artifact.keys()) != _RECEIPT_KEYS:
        return False

    if (
        artifact.get("version") != RECEIPT_VERSION
        or artifact.get("mode") != MODE
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

    artifact_sha = artifact.get("useref_sha256")

    if not is_valid_fingerprint(artifact_sha):
        return False

    if (
        ref_sha256 is not None
        and artifact_sha != ref_sha256
    ):
        return False

    if not is_valid_recall_id(
        artifact.get("recall_id")
    ):
        return False

    if not _is_nonempty_text(
        artifact.get("memory_id")
    ):
        return False

    if not _is_positive_rank(artifact.get("rank")):
        return False

    if artifact.get("kind") not in (
        _KIND_ANCHOR,
        _KIND_RELATED,
    ):
        return False

    if not is_valid_fingerprint(
        artifact.get("source_request_fingerprint")
    ):
        return False

    if not is_valid_fingerprint(
        artifact.get(
            "source_model_result_sha256"
        )
    ):
        return False

    return _is_nonempty_text(
        artifact.get("created_at")
    )


def build_usage_ref_receipt(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    ref_sha256: str,
    recall_id: str,
    memory_id: str,
    rank: int,
    kind: str,
    source_request_fingerprint: str,
    source_model_result_sha256: str,
) -> dict[str, Any]:
    """Build (do not persist) one issuance receipt."""

    return {
        "version": RECEIPT_VERSION,
        "mode": MODE,
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "useref_sha256": ref_sha256,
        "recall_id": recall_id,
        "memory_id": memory_id,
        "rank": rank,
        "kind": kind,
        "source_request_fingerprint":
            source_request_fingerprint,
        "source_model_result_sha256":
            source_model_result_sha256,
        "created_at": now_iso(),
    }


def receipt_binding(artifact: Any) -> dict[str, Any] | None:
    """The comparable binding of a receipt, or None."""

    if not isinstance(artifact, dict):
        return None

    return {
        field: artifact.get(field)
        for field in BINDING_FIELDS
    }


def _same_binding(
    left: Any,
    right: Any,
) -> bool:
    left_binding = receipt_binding(left)

    right_binding = receipt_binding(right)

    if left_binding is None or right_binding is None:
        return False

    return left_binding == right_binding


def persist_usage_ref_receipt(
    artifact: Any,
) -> dict[str, Any]:
    """Write-once persist of one issuance receipt.

    A valid receipt already at the deterministic path is reused only
    when its binding is EXACTLY the same. An invalid or conflicting
    receipt is never repaired and never overwritten.
    """

    if not is_valid_usage_ref_receipt(artifact):
        return {
            "stored": False,
            "duplicate": False,
            "reason": _REASON_INVALID,
        }

    path = usage_ref_receipt_path(
        artifact["conversation_id"],
        artifact["cognitive_request_id"],
        artifact["useref_sha256"],
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
            if is_valid_usage_ref_receipt(
                existing,
                conversation_id=artifact[
                    "conversation_id"
                ],
                cognitive_request_id=artifact[
                    "cognitive_request_id"
                ],
                ref_sha256=artifact[
                    "useref_sha256"
                ],
            ) and _same_binding(existing, artifact):
                return {
                    "stored": True,
                    "duplicate": True,
                    "reason": _REASON_REUSED,
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
        "reason": _REASON_STORED,
    }


def read_usage_ref_receipt(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    ref_sha256: Any,
) -> dict[str, Any] | None:
    """Read one issuance receipt by EXACT path, fully re-validated.

    Returns None when the receipt is missing OR malformed OR belongs
    to another request: a corrupt or borrowed receipt is never handed
    out as if it were valid.
    """

    path = usage_ref_receipt_path(
        conversation_id,
        cognitive_request_id,
        ref_sha256,
    )

    if path is None:
        return None

    artifact = read_json(path)

    if not is_valid_usage_ref_receipt(
        artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        ref_sha256=ref_sha256,
    ):
        return None

    return artifact


def receipt_matches_binding(
    receipt: Any,
    binding: Any,
) -> bool:
    """Does a receipt carry EXACTLY this binding?"""

    if not isinstance(binding, dict):
        return False

    receipt_view = receipt_binding(receipt)

    if receipt_view is None:
        return False

    return all(
        receipt_view.get(field) == binding.get(field)
        for field in BINDING_FIELDS
    )


__all__ = [
    "BINDING_FIELDS",
    "MODE",
    "RECEIPT_VERSION",
    "build_usage_ref_receipt",
    "is_valid_usage_ref_receipt",
    "persist_usage_ref_receipt",
    "read_usage_ref_receipt",
    "receipt_binding",
    "receipt_matches_binding",
    "usage_ref_receipt_dir",
    "usage_ref_receipt_path",
    "useref_sha256",
]