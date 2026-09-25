from __future__ import annotations

from typing import Any

from ombrebrain.context.pipeline_parser import (
    from_stage_payload,
)
from ombrebrain.context.pipeline_state import (
    CHAIN_STAGE,
    FRESHNESS_FRESH,
    FRESHNESS_STALE,
    STAGE_GATE,
    STAGE_PREVIEW,
    STAGE_UNIFIED,
    ContextPipelineState,
)
from ombrebrain.context.validators.freshness import (
    validate_context_freshness,
)


def to_event(
    state: ContextPipelineState,
    *,
    reason: str | None = None,
    freshness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified internal telemetry event.

    Only booleans, counts, enums, revisions, token counts and
    reasons are ever present.
    """

    metrics = state.stage_metrics()

    if freshness is not None:
        metrics["freshness"] = dict(
            freshness
        )

    return {
        "stage": state.stage,
        "decision": state.decision,
        "revision": state.revision,
        "estimated_tokens":
            state.estimated_tokens,
        "reason": reason,
        "metrics": metrics,
    }


def build_context_chain_event(
    *,
    unified: Any,
    preview: Any,
    gate: Any,
) -> dict[str, Any]:
    """Aggregate the refreshed chain into one internal event.

    Observation-only. The revision chain is re-checked with the
    shared freshness validator so a drifted chain is reported
    instead of silently accepted. Nothing here mutates a request.
    """

    states = {
        STAGE_UNIFIED: from_stage_payload(
            STAGE_UNIFIED,
            unified,
        ),
        STAGE_PREVIEW: from_stage_payload(
            STAGE_PREVIEW,
            preview,
        ),
        STAGE_GATE: from_stage_payload(
            STAGE_GATE,
            gate,
        ),
    }

    # Preview must have been rendered from the Unified revision
    # this request just stored...
    freshness = validate_context_freshness(
        checked_revision=states[
            STAGE_PREVIEW
        ].source_revision,
        expected_revision=states[
            STAGE_UNIFIED
        ].revision,
        invalid_reason=(
            "invalid_preview_source_revision"
        ),
        mismatch_reason=(
            "preview_source_revision_mismatch"
        ),
    )

    # ...and the Gate must have evaluated that exact Preview.
    if freshness["valid"] is True:
        gate_freshness = (
            validate_context_freshness(
                checked_revision=states[
                    STAGE_GATE
                ].source_revision,
                expected_revision=states[
                    STAGE_PREVIEW
                ].revision,
                invalid_reason=(
                    "invalid_gate_source_revision"
                ),
                mismatch_reason=(
                    "preview_revision_mismatch"
                ),
            )
        )

        if gate_freshness["valid"] is not True:
            freshness = gate_freshness

    fresh = freshness["valid"] is True

    states[
        STAGE_PREVIEW
    ] = states[
        STAGE_PREVIEW
    ].replace(
        freshness_state=(
            FRESHNESS_FRESH
            if fresh
            else FRESHNESS_STALE
        )
    )

    chain_state = ContextPipelineState(
        stage=CHAIN_STAGE,
        conversation_id=states[
            STAGE_UNIFIED
        ].conversation_id,
        revision=states[
            STAGE_UNIFIED
        ].revision,
        estimated_tokens=states[
            STAGE_PREVIEW
        ].estimated_tokens,
        freshness_state=(
            FRESHNESS_FRESH
            if fresh
            else FRESHNESS_STALE
        ),
        decision=(
            FRESHNESS_FRESH
            if fresh
            else FRESHNESS_STALE
        ),
    )

    event = to_event(
        chain_state,
        reason=freshness["reason"],
        freshness=freshness,
    )

    event["metrics"]["stages"] = {
        stage: state.stage_metrics()
        for stage, state in states.items()
    }

    return event
