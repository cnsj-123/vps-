from __future__ import annotations

import json
import os
from typing import Any

from ombrebrain.context import memory_usage_signal
from ombrebrain.context.live_recall_usage_surface import (
    is_valid_useref,
    resolve_usage_refs_max_refs,
    resolve_verified_live_recall_usage_refs,
)
from ombrebrain.context.memory_usage_signal import (
    read_memory_usage,
)
from ombrebrain.context.recall_types import (
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
)


# Memory Live Recall Usage Attribution v1 — explicit model attribution.
#
#     memory_loaded            != used
#     loaded_but_not_exposed   != attributable
#     used                     != reinforced
#
# This module implements exactly ONE new rung:
#
#     memory_loaded + explicit model-declared usage -> used
#
# The ONLY accepted signal is an explicit tool call that names one or
# more opaque ``useref`` handles of THIS request's Usage Surface. The
# trusted ``conversation_id`` + ``cognitive_request_id`` are injected
# by the transport from the trusted ContextVar binding; the model can
# never supply them, and it never sees a raw memory id or a recall id.
#
# A ``useref`` is resolved to a mapping by the request-scoped surface,
# and that mapping is then re-validated against the REAL persisted
# provenance chain (Recall Request + Related Recall + Usage Signal)
# before anything may be recorded. The surface's own structural
# validity is never authority: a structurally-valid tamper -- a
# swapped ``memory_id``, a rewritten 64-hex fingerprint / digest, a
# shifted ``rank``, a flipped ``kind`` or a mapping borrowed from
# another real recall of the same request -- fails the WHOLE call
# closed, so ``record_usage()`` is never reached and ``used`` never
# changes.
#
# This API is also the FINAL kill-switch for
# ``OMBRE_GATEWAY_CONTEXT_USAGE_ATTRIBUTION``: with the flag OFF the
# call is refused before any surface is even read, so an old but
# still-unexpired v2 capability can never append a usage ref.
#
# ``used`` is NEVER inferred. There is no answer-text scan, no keyword
# matching, no embedding similarity, no LLM judge and no "the Recall
# succeeded, so it must have been used" shortcut. A ref that does not
# resolve in the exact request-scoped surface file is refused, so a
# guessed, cross-request or cross-conversation ref can never become a
# usage signal.
#
# One call may only reference ONE recall: mixed refs of two recalls
# are refused as a whole, so the frozen ``record_usage()`` keeps its
# simple atomic per-recall semantics. The model may simply call again.
#
# Everything below delegates the write to the frozen Usage Signal:
# ``memory_usage_signal.record_usage()`` re-reads and re-validates the
# Related Recall and enforces ``used ⊆ loaded``. This module never
# writes a ``memory_usage`` JSON itself, never performs retrieval,
# never loads memory content and never mutates a memory or a lifecycle
# artifact.
#
# Usage Attribution records explicit usage evidence only. Lifecycle /
# reinforcement consumption is intentionally deferred to a later
# integration phase: this module never calls
# ``observe_memory_usage_lifecycle()``,
# ``process_completed_usage_for_lifecycle()``,
# ``record_lifecycle_event_from_usage()`` or
# ``derive_memory_lifecycle_state()``.

_VERSION = "memory-live-recall-usage-attribution.v1"
_MODE = "explicit_model_attribution"

_ENV_USAGE_ATTRIBUTION = (
    "OMBRE_GATEWAY_CONTEXT_USAGE_ATTRIBUTION"
)

_REASON_RECORDED = "usage_recorded"
_REASON_DUPLICATE = "duplicate_usage_refs"
_REASON_INVALID_REFS = "invalid_usage_refs"
_REASON_MIXED_RECALLS = "mixed_recall_refs"
_REASON_TOO_MANY = "too_many_refs"
_REASON_SURFACE_UNAVAILABLE = (
    "usage_surface_unavailable"
)
_REASON_PROVENANCE_INVALID = (
    "usage_mapping_provenance_invalid"
)
_REASON_DISABLED = "usage_attribution_disabled"
_REASON_RECORD_FAILED = "usage_record_failed"
_REASON_REF_BINDING_INVALID = (
    "usage_ref_binding_invalid"
)

SAFE_ATTRIBUTION_REASONS = frozenset(
    {
        _REASON_RECORDED,
        _REASON_DUPLICATE,
        _REASON_INVALID_REFS,
        _REASON_MIXED_RECALLS,
        _REASON_TOO_MANY,
        _REASON_SURFACE_UNAVAILABLE,
        _REASON_PROVENANCE_INVALID,
        _REASON_DISABLED,
        _REASON_RECORD_FAILED,
        _REASON_REF_BINDING_INVALID,
    }
)

# Deterministic DATA-ONLY acknowledgement. It never echoes a useref, a
# memory id, a recall id, a CID or a RID.
_ACK_HEADER = "OMBRE MEMORY USAGE ACK DATA\n"

_ACK_STATUS_RECORDED = "recorded"
_ACK_STATUS_REFUSED = "refused"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def live_recall_usage_attribution_enabled() -> bool:
    """The Usage Attribution rollout gate. Default OFF.

    OFF keeps the frozen baseline behavior: no Usage Surface is built,
    ``UseMemory`` is neither registered nor visible, and no model
    result carries usage refs.
    """

    return _truthy(
        os.environ.get(_ENV_USAGE_ATTRIBUTION)
    )


def _report(
    *,
    reason: str,
    requested_ref_count: int = 0,
    accepted_ref_count: int = 0,
    newly_used_count: int = 0,
    duplicate_count: int = 0,
    recorded: bool = False,
) -> dict[str, Any]:
    """Privacy-safe internal report.

    Exactly the allowed fields: no raw id, no CID, no RID, no recall
    id, no useref value and no memory id can appear here.
    """

    return {
        "version": _VERSION,
        "mode": _MODE,
        "recorded": bool(recorded),
        "reason": (
            reason
            if reason in SAFE_ATTRIBUTION_REASONS
            else _REASON_INVALID_REFS
        ),
        "requested_ref_count": int(
            requested_ref_count
        ),
        "accepted_ref_count": int(
            accepted_ref_count
        ),
        "newly_used_count": int(newly_used_count),
        "duplicate_count": int(duplicate_count),
    }


def _nonnegative_int(value: Any) -> int:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    return 0


def attribute_live_recall_usage(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    userefs: Any,
) -> dict[str, Any]:
    """Record explicit model-declared usage for ONE recall.

    The trusted caller binds ``conversation_id`` and
    ``cognitive_request_id``; the model may only pass ``userefs``.

    The rollout flag is the FINAL authority: with
    ``OMBRE_GATEWAY_CONTEXT_USAGE_ATTRIBUTION`` OFF the call is
    refused before any surface is read, so an already-minted, still
    unexpired v2 capability can never append a usage ref.

    Resolution is exact and request-scoped:

        THIS CID/RID surface file
          -> every useref must exist in it (unknown / fake /
             cross-request / cross-conversation refs are refused)
            -> every selected mapping re-validated against the REAL
               persisted Recall Request / Related Recall / Usage
               Signal (a structurally-valid tamper fails the WHOLE
               call closed)
              -> all refs must belong to ONE recall id
                -> frozen ``record_usage()`` (used ⊆ loaded)

    Fail-closed: any refusal records nothing and never raises.
    """

    requested = (
        len(userefs)
        if isinstance(userefs, (list, tuple))
        else 0
    )

    if not live_recall_usage_attribution_enabled():
        return _report(
            reason=_REASON_DISABLED,
            requested_ref_count=requested,
        )

    if not isinstance(userefs, (list, tuple)):
        return _report(reason=_REASON_INVALID_REFS)

    if (
        not is_valid_conversation_id(conversation_id)
        or not is_valid_cognitive_request_id(
            cognitive_request_id
        )
    ):
        return _report(
            reason=_REASON_INVALID_REFS,
            requested_ref_count=requested,
        )

    if requested == 0:
        return _report(
            reason=_REASON_INVALID_REFS,
            requested_ref_count=requested,
        )

    # Bounded by design: the model can never pass an unbounded array.
    if requested > resolve_usage_refs_max_refs():
        return _report(
            reason=_REASON_TOO_MANY,
            requested_ref_count=requested,
        )

    for value in userefs:
        if not is_valid_useref(value):
            return _report(
                reason=_REASON_INVALID_REFS,
                requested_ref_count=requested,
            )

    try:
        # The SHARED, receipt-bound resolver: a handle only resolves
        # through the exact current surface, the REAL persisted
        # provenance (fingerprint + projection recomputed) AND its
        # independent, write-once issuance receipt. A surface-only
        # tamper -- a swapped useref included -- can never record
        # ``used``.
        resolution = (
            resolve_verified_live_recall_usage_refs(
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
                userefs=userefs,
            )
        )

        if (
            not isinstance(resolution, dict)
            or resolution.get("ok") is not True
        ):
            reason = (
                resolution.get("reason")
                if isinstance(resolution, dict)
                else None
            )

            return _report(
                reason=(
                    reason
                    if reason
                    in SAFE_ATTRIBUTION_REASONS
                    else _REASON_PROVENANCE_INVALID
                ),
                requested_ref_count=requested,
            )

        recall_id = resolution.get("recall_id")

        resolved = list(
            resolution.get("mappings") or []
        )

        usage = read_memory_usage(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            recall_id=recall_id,
        )

        if not isinstance(usage, dict):
            return _report(
                reason=_REASON_RECORD_FAILED,
                requested_ref_count=requested,
            )

        loaded = {
            memory_id
            for memory_id in (
                usage.get("loaded_memory_ids") or []
            )
            if isinstance(memory_id, str)
            and memory_id.strip()
        }

        used_memory_ids: list[str] = []

        for mapping in resolved:
            memory_id = mapping.get("memory_id")

            if (
                not isinstance(memory_id, str)
                or not memory_id.strip()
                or memory_id.strip() not in loaded
            ):
                # Only a really loaded recalled item can ever be
                # declared used.
                return _report(
                    reason=_REASON_INVALID_REFS,
                    requested_ref_count=requested,
                )

            memory_id = memory_id.strip()

            if memory_id not in used_memory_ids:
                used_memory_ids.append(memory_id)

        # The frozen Usage Signal stays the final authority: it
        # re-validates the Related Recall, re-checks the loaded
        # membership and enforces ``used ⊆ loaded`` itself.
        result = memory_usage_signal.record_usage(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            recall_id=recall_id,
            used_memory_ids=used_memory_ids,
        )

        accepted = len(used_memory_ids)

        if (
            not isinstance(result, dict)
            or result.get("stored") is not True
        ):
            return _report(
                reason=_REASON_RECORD_FAILED,
                requested_ref_count=requested,
                accepted_ref_count=accepted,
            )

        newly_used = _nonnegative_int(
            result.get("newly_used_count")
        )

        if newly_used > accepted:
            newly_used = accepted

        duplicate = accepted - newly_used

        return _report(
            reason=(
                _REASON_RECORDED
                if newly_used > 0
                else _REASON_DUPLICATE
            ),
            requested_ref_count=requested,
            accepted_ref_count=accepted,
            newly_used_count=newly_used,
            duplicate_count=duplicate,
            recorded=True,
        )

    except Exception:
        return _report(
            reason=_REASON_RECORD_FAILED,
            requested_ref_count=requested,
        )


def render_live_recall_usage_ack(
    report: Any,
) -> str:
    """Render the model-facing, DATA-ONLY usage acknowledgement.

    A refusal never explains internals: the reason is a safe enum.
    Nothing is echoed back -- not a useref, not a memory id, not a
    recall id, not the request identity.
    """

    if isinstance(report, dict) and (
        report.get("recorded") is True
    ):
        accepted = _nonnegative_int(
            report.get("accepted_ref_count")
        )

        payload = {
            "status": _ACK_STATUS_RECORDED,
            "used_count": accepted,
            "newly_used_count": _nonnegative_int(
                report.get("newly_used_count")
            ),
            "duplicate": (
                accepted > 0
                and _nonnegative_int(
                    report.get("duplicate_count")
                )
                == accepted
            ),
        }
    else:
        reason = (
            report.get("reason")
            if isinstance(report, dict)
            else None
        )

        payload = {
            "status": _ACK_STATUS_REFUSED,
            "reason": (
                reason
                if reason in SAFE_ATTRIBUTION_REASONS
                else _REASON_INVALID_REFS
            ),
        }

    return (
        _ACK_HEADER
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


__all__ = [
    "SAFE_ATTRIBUTION_REASONS",
    "attribute_live_recall_usage",
    "live_recall_usage_attribution_enabled",
    "render_live_recall_usage_ack",
]