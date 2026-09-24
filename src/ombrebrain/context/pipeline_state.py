from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ombrebrain.context.validators.freshness import (
    validate_context_freshness,
)

# Canonical stage names for the Context pipeline:
#   Candidate -> Unified -> Preview -> Gate
STAGE_CANDIDATE = "candidate"
STAGE_UNIFIED = "unified"
STAGE_PREVIEW = "preview"
STAGE_GATE = "gate"

STAGES = (
    STAGE_CANDIDATE,
    STAGE_UNIFIED,
    STAGE_PREVIEW,
    STAGE_GATE,
)

# Freshness vocabulary. "unknown" means no per-request latch has
# been evaluated for this stage yet.
FRESHNESS_UNKNOWN = "unknown"
FRESHNESS_FRESH = "fresh"
FRESHNESS_STALE = "stale"

CHAIN_STAGE = "context_chain"


def _as_int(value: Any) -> int | None:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    return None


def _as_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value

    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        item
        for item in value
        if isinstance(item, str)
    ]


@dataclass(frozen=True)
class ContextPipelineState:
    """Read-only normalized view of one persisted Context stage.

    This is a compatibility layer, not a replacement: the existing
    stage modules keep their own on-disk payloads and signatures.
    ``from_stage_payload`` only re-shapes an already-loaded payload
    into the fields the pipeline reasons about, so callers stop
    hand-picking the same keys in several places.
    """

    stage: str

    conversation_id: str | None = None
    revision: int | None = None
    source_revision: int | None = None

    sections: dict[str, Any] = field(
        default_factory=dict
    )

    estimated_tokens: int | None = None
    token_budget: int | None = None

    freshness_state: str = FRESHNESS_UNKNOWN
    decision: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def replace(
        self,
        **changes: Any,
    ) -> "ContextPipelineState":
        return replace(self, **changes)

    def stage_metrics(self) -> dict[str, Any]:
        """Privacy-safe per-stage summary.

        Never contains sections or rendered Context text.
        """

        return {
            "conversation_id":
                self.conversation_id,
            "revision":
                self.revision,
            "source_revision":
                self.source_revision,
            "estimated_tokens":
                self.estimated_tokens,
            "token_budget":
                self.token_budget,
            "decision":
                self.decision,
            "freshness_state":
                self.freshness_state,
        }

    def to_event(
        self,
        *,
        reason: str | None = None,
        freshness: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Unified internal telemetry event.

        Only booleans, counts, enums, revisions, token counts and
        reasons are ever present.
        """

        metrics = self.stage_metrics()

        if freshness is not None:
            metrics["freshness"] = dict(
                freshness
            )

        return {
            "stage": self.stage,
            "decision": self.decision,
            "revision": self.revision,
            "estimated_tokens":
                self.estimated_tokens,
            "reason": reason,
            "metrics": metrics,
        }


def from_stage_payload(
    stage: str,
    payload: Any,
) -> ContextPipelineState:
    """Normalize a persisted stage payload. Never raises."""

    if not isinstance(payload, dict):
        return ContextPipelineState(
            stage=stage
        )

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

    source_revision = _as_int(
        payload.get("source_revision")
    )

    decision: str | None = None
    metadata: dict[str, Any] = {}

    if stage == STAGE_UNIFIED:
        # source_revision = the Candidate revision this Unified
        # candidate was derived from.
        source_revisions = _as_dict(
            payload.get("source_revisions")
        )

        if source_revision is None:
            source_revision = _as_int(
                source_revisions.get(
                    "conversation_candidate"
                )
            )

        metadata["source_revisions"] = dict(
            source_revisions
        )

    elif stage == STAGE_PREVIEW:
        eligible = (
            payload.get("eligible") is True
        )

        decision = (
            "eligible"
            if eligible
            else "ineligible"
        )

        metadata["eligible"] = eligible
        metadata["reason"] = _as_str(
            payload.get("reason")
        )
        metadata["section_names"] = (
            _as_str_list(
                payload.get("section_names")
            )
        )

    elif stage == STAGE_GATE:
        decision = _as_str(
            payload.get("decision")
        )

        # source_revision = the Preview revision this Gate
        # decision was evaluated against.
        if source_revision is None:
            source_revision = _as_int(
                payload.get(
                    "source_preview_revision"
                )
            )

        metadata["allowed"] = (
            payload.get("allowed") is True
        )
        metadata["source_candidate_revision"] = (
            _as_int(
                payload.get(
                    "source_candidate_revision"
                )
            )
        )
        metadata["source_unified_revision"] = (
            _as_int(
                payload.get(
                    "source_unified_revision"
                )
            )
        )
        metadata["reasons"] = _as_str_list(
            payload.get("reasons")
        )

    return ContextPipelineState(
        stage=stage,
        conversation_id=_as_str(
            payload.get("conversation_id")
        ),
        revision=_as_int(
            payload.get("revision")
        ),
        source_revision=source_revision,
        sections=_as_dict(
            payload.get("sections")
        ),
        estimated_tokens=estimated_tokens,
        token_budget=token_budget,
        decision=decision,
        metadata=metadata,
    )


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

    states[
        STAGE_PREVIEW
    ] = states[
        STAGE_PREVIEW
    ].replace(
        freshness_state=(
            FRESHNESS_FRESH
            if freshness["valid"]
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
            if freshness["valid"]
            else FRESHNESS_STALE
        ),
        decision=(
            FRESHNESS_FRESH
            if freshness["valid"]
            else FRESHNESS_STALE
        ),
    )

    event = chain_state.to_event(
        reason=freshness["reason"],
        freshness=freshness,
    )

    event["metrics"]["stages"] = {
        stage: state.stage_metrics()
        for stage, state in states.items()
    }

    return event
