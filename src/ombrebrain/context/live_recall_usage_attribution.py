from __future__ import annotations

import json
import os
from typing import Any

from ombrebrain.context import memory_usage_signal
from ombrebrain.context.live_recall_usage_surface import (
    is_valid_useref,
    read_live_recall_usage_surface,
    resolve_usage_refs_max_refs,
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
_REASON_RECORD_FAILED = "usage_record_failed"

SAFE_ATTRIBUTION_REASONS = frozenset(
    {
        _REASON_RECORDED,
        _REASON_DUPLICATE,
        _REASON_INVALID_REFS,
        _REASON_MIXED_RECALLS,
        _REASON_TOO_MANY,
        _REASON_SURFACE_UNAVAILABLE,
        _REASON_RECORD_FAILED,
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

    Resolution is exact and request-scoped:

        THIS CID/RID surface file
          -> every useref must exist in it (unknown / fake /
             cross-request / cross-conversation refs are refused)
            -> all refs must belong to ONE recall id
              -> frozen ``record_usage()`` (used ⊆ loaded)

    Fail-closed: any refusal records nothing and never raises.
    """

    if not isinstance(userefs, (list, tuple)):
        return _report(reason=_REASON_INVALID_REFS)

    requested = len(userefs)

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
        surface = read_live_recall_usage_surface(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
        )

        if not isinstance(surface, dict):
            # No usable surface for THIS request: nothing to resolve,
            # and no memory_usage / recall / lifecycle artifact is
            # created by the refusal.
            return _report(
                reason=_REASON_SURFACE_UNAVAILABLE,
                requested_ref_count=requested,
            )

        mappings = surface.get("mappings") or []

        by_ref: dict[str, dict[str, Any]] = {}

        surface_order: dict[str, int] = {}

        for index, mapping in enumerate(mappings):
            if not isinstance(mapping, dict):
                continue

            useref = mapping.get("useref")

            if isinstance(useref, str):
                by_ref[useref] = mapping
                surface_order[useref] = index

        resolved: list[dict[str, Any]] = []

        seen: set[str] = set()

        for value in userefs:
            mapping = by_ref.get(value)

            if not isinstance(mapping, dict):
                return _report(
                    reason=_REASON_INVALID_REFS,
                    requested_ref_count=requested,
                )

            if value in seen:
                # Duplicates are folded; used events are never
                # double-counted.
                continue

            seen.add(value)

            resolved.append(mapping)

        # Surface rank order, never the model's argument order.
        resolved.sort(
            key=lambda mapping: surface_order[
                mapping["useref"]
            ]
        )

        recall_ids = {
            mapping.get("recall_id")
            for mapping in resolved
        }

        # One call, one recall: a mixed call records nothing at all.
        if len(recall_ids) != 1:
            return _report(
                reason=_REASON_MIXED_RECALLS,
                requested_ref_count=requested,
            )

        recall_id = next(iter(recall_ids))

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