from __future__ import annotations

from typing import Any

from ombrebrain.context.pipeline_state import (
    STAGE_GATE,
    STAGE_PREVIEW,
    STAGE_UNIFIED,
    ContextPipelineState,
)


def _as_int(
    value: Any,
) -> int | None:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    return None


def _as_str(
    value: Any,
) -> str | None:
    if isinstance(value, str) and value.strip():
        return value

    return None


def _as_dict(
    value: Any,
) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(
    value: Any,
) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        item
        for item in value
        if isinstance(item, str)
    ]


def _parse_unified_state(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Stage-specific Unified fields.

    source_revision = the Candidate revision this Unified
    candidate was derived from.
    """

    source_revisions = _as_dict(
        payload.get("source_revisions")
    )

    source_revision = _as_int(
        payload.get("source_revision")
    )

    if source_revision is None:
        source_revision = _as_int(
            source_revisions.get(
                "conversation_candidate"
            )
        )

    return {
        "source_revision":
            source_revision,
        "metadata": {
            "source_revisions": dict(
                source_revisions
            )
        },
    }


def _parse_preview_state(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Stage-specific Preview fields."""

    eligible = (
        payload.get("eligible") is True
    )

    return {
        "decision": (
            "eligible"
            if eligible
            else "ineligible"
        ),
        "metadata": {
            "eligible": eligible,
            "reason": _as_str(
                payload.get("reason")
            ),
            "section_names": (
                _as_str_list(
                    payload.get(
                        "section_names"
                    )
                )
            ),
        },
    }


def _parse_gate_state(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Stage-specific Gate fields.

    source_revision = the Preview revision this Gate decision
    was evaluated against.
    """

    source_revision = _as_int(
        payload.get("source_revision")
    )

    if source_revision is None:
        source_revision = _as_int(
            payload.get(
                "source_preview_revision"
            )
        )

    return {
        "source_revision":
            source_revision,
        "decision": _as_str(
            payload.get("decision")
        ),
        "metadata": {
            "allowed": (
                payload.get("allowed")
                is True
            ),
            "source_candidate_revision": (
                _as_int(
                    payload.get(
                        "source_candidate_revision"
                    )
                )
            ),
            "source_unified_revision": (
                _as_int(
                    payload.get(
                        "source_unified_revision"
                    )
                )
            ),
            "reasons": _as_str_list(
                payload.get("reasons")
            ),
        },
    }


# Stage dispatch. Unknown stages (e.g. the generic Candidate
# payload) keep the common extraction only.
_STAGE_PARSERS = {
    STAGE_UNIFIED:
        _parse_unified_state,
    STAGE_PREVIEW:
        _parse_preview_state,
    STAGE_GATE:
        _parse_gate_state,
}


def _parse_common_state(
    stage: str,
    payload: dict[str, Any],
) -> ContextPipelineState:
    """Fields shared by every stage payload shape."""

    telemetry = _as_dict(
        payload.get("telemetry")
    )

    estimated_tokens = _as_int(
        payload.get("estimated_tokens")
    )

    if estimated_tokens is None:
        estimated_tokens = _as_int(
            telemetry.get(
                "estimated_tokens"
            )
        )

    token_budget = _as_int(
        payload.get("token_budget")
    )

    if token_budget is None:
        token_budget = _as_int(
            telemetry.get("token_budget")
        )

    return ContextPipelineState(
        stage=stage,
        conversation_id=_as_str(
            payload.get("conversation_id")
        ),
        revision=_as_int(
            payload.get("revision")
        ),
        source_revision=_as_int(
            payload.get("source_revision")
        ),
        sections=_as_dict(
            payload.get("sections")
        ),
        estimated_tokens=estimated_tokens,
        token_budget=token_budget,
    )


def from_stage_payload(
    stage: str,
    payload: Any,
) -> ContextPipelineState:
    """Normalize a persisted stage payload. Never raises."""

    if not isinstance(payload, dict):
        return ContextPipelineState(
            stage=stage
        )

    state = _parse_common_state(
        stage,
        payload,
    )

    parser = _STAGE_PARSERS.get(
        stage
    )

    if parser is None:
        return state

    return state.replace(
        **parser(payload)
    )
