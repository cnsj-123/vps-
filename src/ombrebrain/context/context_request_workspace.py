from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Request-local Cognitive Workspace (v0).
#
# One gateway request owns exactly one workspace. It exists in
# memory for the lifetime of a single Context pipeline pass and it
# is bound to the exact ``conversation_id`` / ``cognitive_request_id``
# that pass was given.
#
# It is NOT canonical memory, NOT a database, NOT a persisted
# artifact, NOT model-visible, NOT a cache key, NOT an event bus and
# NOT a universal memory schema. It carries only the exact
# request-local stage outputs THIS request already produced, so that
# downstream runtime decisions can prefer the artifact produced for
# the request currently being forwarded instead of re-reading a
# conversation-scoped "latest" file that a concurrent request may
# already have overwritten.
#
# Conversation-scoped persisted artifacts remain the "latest
# persisted observation" for observability, debugging and
# compatibility. They are NOT interchangeable with the runtime
# authority carried here.
#
# Sensitive by construction: ``conversation_candidate`` and
# ``unified`` carry request/conversation text. The workspace is never
# serialized, never logged and never added to a request, a header, a
# cache key or a persisted artifact.
#
# The fields are the minimum the CURRENT chain needs. No future
# Episodic / Semantic / Relation / Prospective field is added
# speculatively.


@dataclass
class ContextRequestWorkspace:
    """Request-local runtime authority for one Context pipeline pass."""

    conversation_id: str | None = None
    cognitive_request_id: str | None = None

    # Exact Conversation Candidate THIS request captured before it
    # awaited retrieval.
    conversation_candidate: dict[str, Any] | None = None

    # Exact Unified artifact THIS request built.
    unified: dict[str, Any] | None = None

    # Request-local Memory observation snapshot THIS request produced.
    memory_shadow_snapshot: dict[str, Any] | None = None

    # Confidence Gate report for THIS request (observer only).
    confidence_report: dict[str, Any] | None = None

    # Exact Preview / Gate artifacts built for THIS request. The
    # Preview artifact still carries its rendered text; it is never
    # logged from here.
    preview: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None

    # Per-request freshness latch, mirrored from the pipeline pass.
    context_chain_fresh: bool = False

    @property
    def has_request_local_chain(self) -> bool:
        """True when THIS request produced its own Candidate + Unified.

        Only then may downstream stages treat the workspace as the
        runtime authority. When it is False (e.g. a request that did
        not refresh the chain) every stage falls back to its existing
        behaviour.
        """

        return (
            isinstance(
                self.conversation_candidate,
                dict,
            )
            and isinstance(
                self.unified,
                dict,
            )
        )

    @property
    def has_preview_gate_evidence(self) -> bool:
        """True when THIS request stored its own Preview + Gate.

        Distinguishes "this request produced a Preview/Gate decision"
        (even a deny, or a binding that turns out to be malformed) from
        "this request produced no Preview/Gate at all". Downstream
        stages use it to fail closed on a malformed / foreign binding
        instead of silently falling back to a conversation-scoped
        "latest" pair a concurrent request may own.
        """

        return (
            isinstance(
                self.preview,
                dict,
            )
            and isinstance(
                self.gate,
                dict,
            )
        )

    def request_local_preview_gate(
        self,
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        """Return THIS request's exact Preview/Gate runtime evidence.

        The pair is returned only when both artifacts are present, are
        bound to the same ``conversation_id`` this workspace was created
        for, and therefore may act as the request-local runtime
        authority. When anything is missing or mismatched the caller
        gets ``(None, None)`` and keeps its existing conversation-scoped
        behaviour: a malformed or foreign binding must never become the
        authority for this request.
        """

        preview = self.preview
        gate = self.gate

        if (
            not isinstance(
                preview,
                dict,
            )
            or not isinstance(
                gate,
                dict,
            )
            or not isinstance(
                self.conversation_id,
                str,
            )
        ):
            return None, None

        if (
            preview.get("conversation_id")
            != self.conversation_id
            or gate.get("conversation_id")
            != self.conversation_id
        ):
            return None, None

        return preview, gate


def extract_request_local_chain(
    unified_result: Any,
) -> dict[str, Any] | None:
    """Return the request-local chain evidence of a Unified result.

    ``update_unified_context_candidate_from_runtime`` attaches this
    in-memory-only evidence to the privacy-safe report it returns. It
    is deliberately absent from any other caller (and from every
    on-disk artifact), so a request that did not produce its own
    Candidate + Unified is never mistaken for one that did.

    Pure and read-only; never raises.
    """

    if not isinstance(
        unified_result,
        dict,
    ):
        return None

    chain = unified_result.get(
        "request_local_chain"
    )

    if not isinstance(
        chain,
        dict,
    ):
        return None

    candidate = chain.get(
        "conversation_candidate"
    )

    unified = chain.get(
        "unified"
    )

    if (
        not isinstance(
            candidate,
            dict,
        )
        or not isinstance(
            unified,
            dict,
        )
    ):
        return None

    return chain
