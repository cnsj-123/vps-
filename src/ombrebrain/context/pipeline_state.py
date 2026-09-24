from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

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


@dataclass(frozen=True)
class ContextPipelineState:
    """Read-only normalized view of one persisted Context stage.

    This is a compatibility layer, not a replacement: the existing
    stage modules keep their own on-disk payloads and signatures.

    Parsing lives in ``pipeline_parser.from_stage_payload`` and the
    telemetry event lives in ``pipeline_events``; this module only
    defines the shared state object itself.
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


# Backward-compatible re-exports. These functions moved to
# pipeline_parser / pipeline_events, but the original import path
# ``from ombrebrain.context.pipeline_state import ...`` must keep
# working. Lazy resolution avoids an import cycle between the
# state object and the modules that consume it.
_COMPAT_EXPORTS = {
    "from_stage_payload":
        "pipeline_parser",
    "build_context_chain_event":
        "pipeline_events",
}


def __getattr__(name: str):
    if name in _COMPAT_EXPORTS:
        from importlib import (
            import_module,
        )

        value = getattr(
            import_module(
                "ombrebrain.context."
                + _COMPAT_EXPORTS[name]
            ),
            name,
        )

        # Cache so later lookups skip this hook.
        globals()[name] = value

        return value

    raise AttributeError(
        "module %r has no attribute %r"
        % (
            __name__,
            name,
        )
    )
