from __future__ import annotations

import json
import logging
import os
from typing import Any

from ombrebrain.context.live_recall_usage_attribution import (
    attribute_live_recall_usage,
)
from ombrebrain.context.live_recall_usage_surface import (
    is_valid_useref,
    read_live_recall_usage_surface,
    validate_live_recall_usage_surface_provenance,
)
from ombrebrain.context.memory_reinforcement import (
    memory_lifecycle_shadow_enabled,
    process_completed_usage_for_lifecycle,
)
from ombrebrain.context.recall_types import (
    is_valid_recall_id,
)


# Memory Live Recall Usage → Lifecycle Reinforcement Integration v1.
#
#     retrieved        != reinforced
#     surfaced         != reinforced
#     Recall           != reinforced
#     memory_loaded    != reinforced
#     used             != reinforced
#
# This module adds exactly ONE rung to the already-frozen chain:
#
#     memory_loaded
#       -> useref
#         -> UseMemory(useref)
#           -> persisted validated used
#             -> lifecycle reinforcement evidence   (THIS module)
#               -> derived lifecycle state
#
# It is a pure coordinator. It NEVER re-implements Usage Attribution
# and it NEVER re-implements the reinforcement algorithm:
#
#   - ``used`` is recorded by the frozen attribution authority
#     (``attribute_live_recall_usage``); this module never writes a
#     usage artifact, never calls ``record_usage()`` and never infers
#     usage from response text, similarity or an LLM judge;
#   - reinforcement is produced by the frozen observer
#     (``process_completed_usage_for_lifecycle``); this module never
#     calls ``record_lifecycle_event_from_usage()``, never enumerates
#     ``used_memory_ids``, never computes a strength and never copies
#     the reinforcement formula.
#
# The model can never reach a lifecycle path directly. The ONLY model
# action stays ``UseMemory(useref)``; the coordinator runs strictly
# AFTER Usage Attribution, inside the trusted backend, and the model
# never sees a CID, a RID, a recall id, a memory id, a strength, a
# salience or a reinforcement count.
#
# Usage Attribution keeps its authority. If it does not report
# ``recorded == True`` -- invalid ref, mixed recall, cross request,
# invalid provenance, feature disabled -- the observer is NEVER called
# (call count stays 0) and nothing downstream happens.
#
# Lifecycle reinforcement needs a TRIPLE gate, all read on every call
# so a runtime kill-switch takes effect immediately:
#
#     OMBRE_GATEWAY_CONTEXT_USAGE_ATTRIBUTION=1
#     OMBRE_GATEWAY_CONTEXT_USAGE_LIFECYCLE_INTEGRATION=1
#     OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW=1
#
# any one OFF: no new lifecycle event and no derived state, while the
# ``used`` write (Usage Attribution) still succeeds unchanged.
#
# ``used`` is a NECESSARY but NOT sufficient evidence of reinforcement:
#
#     used == True  AND  integration OFF   -> used, NO reinforcement
#     used == True  AND  shadow OFF        -> used, NO reinforcement
#     used == True  AND  observer failure  -> used, NO reinforcement
#
# The integration is a SECOND, independent downstream trust boundary,
# so the exact request-scoped Usage Surface is re-read and every
# selected mapping is re-validated against the real persisted
# provenance chain. The ``recall_id`` is derived INTERNALLY from
# ``current CID/RID`` + the exact surface + the selected refs; the
# model can never supply it, and it is never passed to the observer
# alongside a memory id (the frozen observer re-derives the memory id
# from the persisted ``used`` events itself).
#
# The observer processes ALL persisted ``used`` events of the CURRENT
# recall only, which allows a bounded same-recall catch-up when an
# earlier observation was disabled or failed. It is idempotent through
# the frozen deterministic event id, so a repeated ``UseMemory``
# retries a missing observation without ever reinforcing twice.
# Global history scans, background workers, cron jobs and migrations
# are deliberately NOT part of this phase.
#
# Lifecycle is a downstream observer: a lifecycle-side exception or a
# partial failure must NEVER turn a real ``used`` write into a usage
# failure. The usage report is the ONLY authority for the model-facing
# acknowledgement, which stays byte-identical to the frozen baseline.

logger = logging.getLogger("ombre_brain.gateway")

_VERSION = (
    "memory-live-recall-usage-lifecycle-integration.v1"
)

_MODE = "explicit_usage_reinforcement"

# The independent rollout flag for this phase. Default OFF. It is NOT
# the lifecycle shadow flag: an old deployment may already have the
# shadow ON, and this flag alone decides whether ``used`` may become
# lifecycle reinforcement.
_ENV_INTEGRATION = (
    "OMBRE_GATEWAY_CONTEXT_USAGE_LIFECYCLE_INTEGRATION"
)

# Safe, closed enum of lifecycle integration reasons. No exception
# string, memory id, recall id or request identity is ever surfaced.
_REASON_INTEGRATION_DISABLED = "integration_disabled"
_REASON_USAGE_NOT_RECORDED = "usage_not_recorded"
_REASON_USAGE_ATTRIBUTION_DISABLED = (
    "usage_attribution_disabled"
)
_REASON_LIFECYCLE_DISABLED = "lifecycle_disabled"
_REASON_USAGE_SURFACE_UNAVAILABLE = (
    "usage_surface_unavailable"
)
_REASON_USAGE_PROVENANCE_INVALID = (
    "usage_provenance_invalid"
)
_REASON_MIXED_RECALL_REFS = "mixed_recall_refs"
_REASON_LIFECYCLE_OBSERVED = "lifecycle_observed"
_REASON_LIFECYCLE_UNAVAILABLE = "lifecycle_unavailable"

SAFE_LIFECYCLE_INTEGRATION_REASONS = frozenset(
    {
        _REASON_INTEGRATION_DISABLED,
        _REASON_USAGE_NOT_RECORDED,
        _REASON_USAGE_ATTRIBUTION_DISABLED,
        _REASON_LIFECYCLE_DISABLED,
        _REASON_USAGE_SURFACE_UNAVAILABLE,
        _REASON_USAGE_PROVENANCE_INVALID,
        _REASON_MIXED_RECALL_REFS,
        _REASON_LIFECYCLE_OBSERVED,
        _REASON_LIFECYCLE_UNAVAILABLE,
    }
)

_LIFECYCLE_COUNT_FIELDS = (
    "used_event_count",
    "new_event_count",
    "duplicate_event_count",
    "state_updated_count",
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def live_recall_usage_lifecycle_integration_enabled() -> bool:
    """The Usage → Lifecycle integration rollout gate. Default OFF."""

    return _truthy(os.environ.get(_ENV_INTEGRATION))


def _nonnegative_int(value: Any) -> int:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    return 0


def _lifecycle_report(
    *,
    enabled: bool,
    observed: bool,
    reason: str,
    observation: Any = None,
) -> dict[str, Any]:
    """Privacy-safe lifecycle summary.

    Exactly the allowed fields: no CID, RID, recall id, memory id,
    useref or memref can appear here.
    """

    counts = {
        field: 0 for field in _LIFECYCLE_COUNT_FIELDS
    }

    if isinstance(observation, dict):
        for field in _LIFECYCLE_COUNT_FIELDS:
            counts[field] = _nonnegative_int(
                observation.get(field)
            )

    return {
        "enabled": bool(enabled),
        "observed": bool(observed),
        "reason": (
            reason
            if reason
            in SAFE_LIFECYCLE_INTEGRATION_REASONS
            else _REASON_LIFECYCLE_UNAVAILABLE
        ),
        "used_event_count": counts[
            "used_event_count"
        ],
        "new_event_count": counts[
            "new_event_count"
        ],
        "duplicate_event_count": counts[
            "duplicate_event_count"
        ],
        "state_updated_count": counts[
            "state_updated_count"
        ],
    }


def _combined(
    usage: Any,
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": _VERSION,
        "mode": _MODE,
        "usage": (
            usage
            if isinstance(usage, dict)
            else {"recorded": False}
        ),
        "lifecycle": lifecycle,
    }


def _log(summary: dict[str, Any]) -> None:
    """Privacy-safe telemetry: counts, booleans and a reason only."""

    logger.info(
        "[gateway.context_usage_lifecycle_integration] %s",
        json.dumps(
            {
                "integration_enabled":
                    summary.get("enabled"),
                "usage_recorded":
                    summary.get("usage_recorded"),
                "lifecycle_observed":
                    summary.get("observed"),
                "used_event_count":
                    summary.get("used_event_count"),
                "new_event_count":
                    summary.get("new_event_count"),
                "duplicate_event_count":
                    summary.get(
                        "duplicate_event_count"
                    ),
                "state_updated_count":
                    summary.get(
                        "state_updated_count"
                    ),
                "reason": summary.get("reason"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _resolve_selected_recall_id(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    userefs: Any,
) -> tuple[str | None, str]:
    """Re-derive the ONE recall id of the SELECTED refs, internally.

    Independent downstream trust boundary: this re-reads the EXACT
    request-scoped surface (never a glob, never a global ref index)
    and re-validates every selected mapping against the real persisted
    provenance chain. The model can never supply the recall id.

    Returns ``(recall_id, reason)``; ``recall_id`` is None on refusal
    and the reason is one of the safe enum values.
    """

    if (
        not isinstance(userefs, (list, tuple))
        or not userefs
    ):
        return None, _REASON_USAGE_SURFACE_UNAVAILABLE

    try:
        surface = read_live_recall_usage_surface(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
        )
    except Exception:
        return None, _REASON_USAGE_SURFACE_UNAVAILABLE

    if not isinstance(surface, dict):
        return None, _REASON_USAGE_SURFACE_UNAVAILABLE

    by_ref: dict[str, dict[str, Any]] = {}

    for mapping in surface.get("mappings") or []:
        if not isinstance(mapping, dict):
            continue

        useref = mapping.get("useref")

        if isinstance(useref, str):
            by_ref[useref] = mapping

    selected: list[dict[str, Any]] = []

    seen: set[str] = set()

    for value in userefs:
        if not is_valid_useref(value):
            return None, _REASON_USAGE_SURFACE_UNAVAILABLE

        mapping = by_ref.get(value)

        if not isinstance(mapping, dict):
            return None, _REASON_USAGE_SURFACE_UNAVAILABLE

        if value in seen:
            continue

        seen.add(value)

        selected.append(mapping)

    if not selected:
        return None, _REASON_USAGE_SURFACE_UNAVAILABLE

    recall_ids = {
        mapping.get("recall_id")
        for mapping in selected
    }

    # One observer call, one recall: a mixed selection is refused.
    if len(recall_ids) != 1:
        return None, _REASON_MIXED_RECALL_REFS

    recall_id = next(iter(recall_ids))

    if not is_valid_recall_id(recall_id):
        return None, _REASON_USAGE_SURFACE_UNAVAILABLE

    try:
        valid = (
            validate_live_recall_usage_surface_provenance(
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
                artifact=surface,
                mappings=selected,
            )
        )
    except Exception:
        return None, _REASON_USAGE_PROVENANCE_INVALID

    if not valid:
        return None, _REASON_USAGE_PROVENANCE_INVALID

    return recall_id, _REASON_LIFECYCLE_OBSERVED


async def attribute_live_recall_usage_with_lifecycle(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    userefs: Any,
    bucket_manager: Any = None,
    as_of: Any = None,
) -> dict[str, Any]:
    """UseMemory coordinator: attribute usage, then observe lifecycle.

    Order is fixed and load-bearing:

      1. the FROZEN ``attribute_live_recall_usage`` records ``used``;
      2. if usage did not record, STOP (the observer is never called);
      3. read the integration flag (runtime kill-switch);
      4. read the lifecycle shadow flag (runtime kill-switch);
      5. re-resolve the selected refs to ONE recall id (internal);
      6. re-validate the exact Usage Surface provenance;
      7. the FROZEN ``process_completed_usage_for_lifecycle`` runs;
      8. return a privacy-safe combined report.

    Usage Attribution keeps its authority and its report is the only
    source of the model-facing ACK. A lifecycle-side failure is
    fail-open and can never turn a real ``used`` write into a usage
    failure.
    """

    usage = attribute_live_recall_usage(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        userefs=userefs,
    )

    usage_recorded = (
        isinstance(usage, dict)
        and usage.get("recorded") is True
    )

    if not usage_recorded:
        reason = _REASON_USAGE_NOT_RECORDED

        if (
            isinstance(usage, dict)
            and usage.get("reason")
            == _REASON_USAGE_ATTRIBUTION_DISABLED
        ):
            reason = _REASON_USAGE_ATTRIBUTION_DISABLED

        lifecycle = _lifecycle_report(
            enabled=False,
            observed=False,
            reason=reason,
        )

        _log({**lifecycle, "usage_recorded": False})

        return _combined(usage, lifecycle)

    # ``used`` is a historical fact now. Everything below is a
    # downstream observer and is fail-open.
    if not live_recall_usage_lifecycle_integration_enabled():
        lifecycle = _lifecycle_report(
            enabled=False,
            observed=False,
            reason=_REASON_INTEGRATION_DISABLED,
        )

        _log({**lifecycle, "usage_recorded": True})

        return _combined(usage, lifecycle)

    if not memory_lifecycle_shadow_enabled():
        lifecycle = _lifecycle_report(
            enabled=False,
            observed=False,
            reason=_REASON_LIFECYCLE_DISABLED,
        )

        _log({**lifecycle, "usage_recorded": True})

        return _combined(usage, lifecycle)

    try:
        recall_id, reason = (
            _resolve_selected_recall_id(
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
                userefs=userefs,
            )
        )
    except Exception:
        recall_id, reason = (
            None,
            _REASON_LIFECYCLE_UNAVAILABLE,
        )

    if recall_id is None:
        lifecycle = _lifecycle_report(
            enabled=True,
            observed=False,
            reason=reason,
        )

        _log({**lifecycle, "usage_recorded": True})

        return _combined(usage, lifecycle)

    # The ONLY lifecycle call. The observer reads the persisted Usage
    # Signal itself and re-derives every memory id; this coordinator
    # never names a memory and never enumerates used ids.
    try:
        observation = (
            await process_completed_usage_for_lifecycle(
                conversation_id,
                cognitive_request_id,
                recall_id,
                bucket_manager=bucket_manager,
                as_of=as_of,
            )
        )
    except Exception:
        observation = None

    if not isinstance(observation, dict):
        lifecycle = _lifecycle_report(
            enabled=True,
            observed=False,
            reason=_REASON_LIFECYCLE_UNAVAILABLE,
        )

        _log({**lifecycle, "usage_recorded": True})

        return _combined(usage, lifecycle)

    observed = observation.get("stored") is True

    lifecycle = _lifecycle_report(
        enabled=True,
        observed=observed,
        reason=(
            _REASON_LIFECYCLE_OBSERVED
            if observed
            else _REASON_LIFECYCLE_UNAVAILABLE
        ),
        observation=observation,
    )

    _log({**lifecycle, "usage_recorded": True})

    return _combined(usage, lifecycle)


__all__ = [
    "SAFE_LIFECYCLE_INTEGRATION_REASONS",
    "attribute_live_recall_usage_with_lifecycle",
    "live_recall_usage_lifecycle_integration_enabled",
]