from __future__ import annotations

import hashlib
import json
import logging
import os

from ombrebrain.context.context_injection_gate import (
    update_context_injection_gate,
)
from ombrebrain.context.context_injection_preview import (
    update_context_injection_preview,
)
from ombrebrain.context.context_real_injection import (
    select_context_injected_body,
)
from ombrebrain.context.context_request_mutation_shadow import (
    build_context_request_mutation_shadow_from_runtime,
)
from ombrebrain.context.pipeline_events import (
    build_context_chain_event,
)
from ombrebrain.context.unified_context_candidate import (
    update_unified_context_candidate_from_runtime,
)

# Context Pipeline Coordinator.
#
# Owns the Context chain that used to be orchestrated inline by
# src/web/gateway.py:
#
#   Unified
#     -> Preview
#       -> Gate
#         -> per-request freshness latch
#           -> Mutation Shadow
#             -> Real Injection selector
#               -> selected_body
#
# The gateway only forwards the bytes returned by
# ``run_context_pipeline()``. It no longer decides when a stage
# runs, what the revision order is, whether the chain is fresh, or
# whether real injection is eligible.
#
# Safety invariants (unchanged, owned here):
#   - real injection is default OFF;
#   - every stage is fail-open: the live request keeps forward_body;
#   - the Mutation Shadow and the Real Injection selector may only
#     read Preview/Gate refreshed by THIS request;
#   - no HTTP error is produced and the request is never blocked;
#   - only privacy-safe telemetry (counts, revisions, hashes,
#     booleans, reason codes) is logged — never query, memory,
#     fact, rendered Context or conversation text.
#
# This module performs no retrieval and no selection: it composes
# existing context stages only.

logger = logging.getLogger("ombre_brain.gateway")


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


async def observe_unified_preview_gate(
    conversation_id: str | None,
) -> bool:
    """Refresh the Unified -> Preview -> Gate chain for this request.

    Returns a per-request freshness latch:

      True  - this request successfully refreshed the chain end to end
              (Unified stored, Preview stored, Gate stored). A fresh
              deny is still fresh: freshness means "refreshed now",
              not "injection allowed".
      False - the chain was not refreshed for this request, so any
              Preview/Gate already on disk is stale and must not be
              used for real injection.

    Shadow-only callers may ignore the return value.
    """

    unified_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW"
        )
    )

    preview_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW"
        )
    )

    gate_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW"
        )
    )

    mutation_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW"
        )
    )

    # Phase 4A-3D: an enabled real injection must refresh the
    # Unified -> Preview -> Gate chain for THIS request, even when
    # every observability shadow flag is off. The selector itself
    # re-runs the mutation shadow, so REQUEST_MUTATION_SHADOW is not
    # required for real injection.
    real_injection_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION"
        )
    )

    if not (
        unified_enabled
        or preview_enabled
        or gate_enabled
        or mutation_enabled
        or real_injection_enabled
    ):
        return False

    if not isinstance(
        conversation_id,
        str,
    ):
        return False

    try:
        unified = await (
            update_unified_context_candidate_from_runtime(
                conversation_id
            )
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_unified_candidate] "
            "store_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return False

    # Privacy-safe telemetry only.
    # No query, plan, memory, fact or conversation text is logged.
    logger.info(
        "[gateway.context_unified_candidate] %s",
        json.dumps(
            {
                "stored":
                    unified.get(
                        "stored"
                    ),
                "conversation_id":
                    conversation_id,
                "revision":
                    unified.get(
                        "revision"
                    ),
                "duplicate":
                    unified.get(
                        "duplicate"
                    ),
                "estimated_tokens":
                    unified.get(
                        "estimated_tokens"
                    ),
                "token_budget":
                    unified.get(
                        "token_budget"
                    ),
                "truncated":
                    unified.get(
                        "truncated"
                    ),
                "budget_rejected":
                    unified.get(
                        "budget_rejected"
                    ),
                "dedup_rejected":
                    unified.get(
                        "dedup_rejected"
                    ),
                "excluded_text_rejected":
                    unified.get(
                        "excluded_text_rejected"
                    ),
                "has_current_task":
                    unified.get(
                        "has_current_task"
                    ),
                "trusted_fact_count":
                    unified.get(
                        "trusted_fact_count"
                    ),
                "constraint_count":
                    unified.get(
                        "constraint_count"
                    ),
                "decision_count":
                    unified.get(
                        "decision_count"
                    ),
                "open_item_count":
                    unified.get(
                        "open_item_count"
                    ),
                "state_included":
                    unified.get(
                        "state_included"
                    ),
                "plan_count":
                    unified.get(
                        "plan_count"
                    ),
                "memory_count":
                    unified.get(
                        "memory_count"
                    ),
                "recent_context_count":
                    unified.get(
                        "recent_context_count"
                    ),
                "current_user_excluded":
                    unified.get(
                        "current_user_excluded"
                    ),
                "retrieval_query_used":
                    unified.get(
                        "retrieval_query_used"
                    ),
                "retrieval_candidate_count":
                    unified.get(
                        "retrieval_candidate_count"
                    ),
                "relevance_rejected":
                    unified.get(
                        "relevance_rejected"
                    ),
                "retrieval_quality":
                    unified.get(
                        "retrieval_quality"
                    ),
                "anti_echo":
                    unified.get(
                        "anti_echo"
                    ),
                "retrieval_dedup":
                    unified.get(
                        "retrieval_dedup"
                    ),
                "retrieval_metrics":
                    unified.get(
                        "retrieval_metrics"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


    if not (
        preview_enabled
        or gate_enabled
        or mutation_enabled
        or real_injection_enabled
    ):
        return False

    if not unified.get(
        "stored"
    ):
        return False

    try:
        preview = (
            update_context_injection_preview(
                conversation_id
            )
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_injection_preview] "
            "store_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return False

    # Privacy-safe summary only.
    # Never log preview["rendered"] or any candidate text.
    logger.info(
        "[gateway.context_injection_preview] %s",
        json.dumps(
            {
                "stored":
                    preview.get(
                        "stored"
                    ),
                "conversation_id":
                    conversation_id,
                "revision":
                    preview.get(
                        "revision"
                    ),
                "source_revision":
                    preview.get(
                        "source_revision"
                    ),
                "duplicate":
                    preview.get(
                        "duplicate"
                    ),
                "eligible":
                    preview.get(
                        "eligible"
                    ),
                "reason":
                    preview.get(
                        "reason"
                    ),
                "estimated_tokens":
                    preview.get(
                        "estimated_tokens"
                    ),
                "token_budget":
                    preview.get(
                        "token_budget"
                    ),
                "section_names":
                    preview.get(
                        "section_names"
                    ),
                "render_sha256":
                    preview.get(
                        "render_sha256"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


    if not (
        gate_enabled
        or mutation_enabled
        or real_injection_enabled
    ):
        return False

    if not preview.get(
        "stored"
    ):
        return False

    try:
        gate = (
            update_context_injection_gate(
                conversation_id
            )
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_injection_gate] "
            "store_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return False

    # Privacy-safe decision summary only.
    # Never log rendered Preview or candidate text.
    logger.info(
        "[gateway.context_injection_gate] %s",
        json.dumps(
            {
                "stored":
                    gate.get(
                        "stored"
                    ),
                "conversation_id":
                    conversation_id,
                "revision":
                    gate.get(
                        "revision"
                    ),
                "duplicate":
                    gate.get(
                        "duplicate"
                    ),
                "mode":
                    gate.get(
                        "mode"
                    ),
                "decision":
                    gate.get(
                        "decision"
                    ),
                "allowed":
                    gate.get(
                        "allowed"
                    ),
                "reason":
                    gate.get(
                        "reason"
                    ),
                "reasons":
                    gate.get(
                        "reasons"
                    ),
                "source_candidate_revision":
                    gate.get(
                        "source_candidate_revision"
                    ),
                "source_unified_revision":
                    gate.get(
                        "source_unified_revision"
                    ),
                "source_preview_revision":
                    gate.get(
                        "source_preview_revision"
                    ),
                "estimated_tokens":
                    gate.get(
                        "estimated_tokens"
                    ),
                "token_budget":
                    gate.get(
                        "token_budget"
                    ),
                "section_names":
                    gate.get(
                        "section_names"
                    ),
                "render_sha256":
                    gate.get(
                        "render_sha256"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )

    if not gate.get(
        "stored"
    ):
        return False

    # Unified internal Context event. Observation-only: it re-reads
    # nothing from disk and never affects forwarding, so it stays
    # fail-open. Existing per-stage logs are unchanged.
    try:
        logger.info(
            "[gateway.context_event] %s",
            json.dumps(
                build_context_chain_event(
                    unified=unified,
                    preview=preview,
                    gate=gate,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    except Exception:
        pass

    # This request refreshed the whole chain. A fresh deny is still
    # fresh; the selector decides whether injection is allowed.
    return True


def observe_request_mutation(
    conversation_id: str | None,
    forward_body: bytes,
) -> None:
    """Build a hypothetical injected request without sending it."""

    if not _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW"
        )
    ):
        return

    if not isinstance(
        conversation_id,
        str,
    ):
        return

    if not isinstance(
        forward_body,
        bytes,
    ):
        return

    try:
        _mutated_body, report = (
            build_context_request_mutation_shadow_from_runtime(
                forward_body,
                conversation_id=
                    conversation_id,
            )
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_request_mutation_shadow] "
            "build_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return

    # IMPORTANT:
    # _mutated_body is intentionally discarded here.
    # The actual upstream request continues to use forward_body.
    logger.info(
        "[gateway.context_request_mutation_shadow] %s",
        json.dumps(
            {
                "conversation_id":
                    conversation_id,
                **report,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def select_real_injection(
    conversation_id: str | None,
    forward_body: bytes,
    *,
    context_chain_fresh: bool,
) -> bytes:
    """Phase 4A-3D limited real injection.

    Returns the body that may be sent upstream.

    This is the only place where a Context-injected body can replace
    the cache-stabilized forward_body.

    Default-OFF contract:
      - without OMBRE_GATEWAY_CONTEXT_REAL_INJECTION=1 the selector is
        not even called and the exact forward_body is returned;
      - the selector itself re-checks the master flag, the Preview,
        the Gate and the mutation invariants;
      - any exception, deny or invalid report returns the exact
        forward_body;
      - the request is never blocked and no HTTP error is produced;
      - only privacy-safe telemetry is logged.

    Freshness contract:
      - context_chain_fresh must be True, i.e. this very request must
        have refreshed Unified -> Preview -> Gate. Otherwise any
        Preview/Gate on disk belongs to an earlier request and must
        not be injected, so the selector is not even called.
    """

    if not _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION"
        )
    ):
        return forward_body

    if (
        not isinstance(
            conversation_id,
            str,
        )
        or not isinstance(
            forward_body,
            bytes,
        )
    ):
        return forward_body

    if context_chain_fresh is not True:
        # Never read stale Preview/Gate from disk to attempt injection.
        logger.info(
            "[gateway.context_real_injection] %s",
            json.dumps(
                {
                    "enabled": True,
                    "applied": False,
                    "reason":
                        "prerequisite_chain_not_fresh",
                    "conversation_id":
                        conversation_id,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        return forward_body

    try:
        selected_body, report = (
            select_context_injected_body(
                forward_body,
                conversation_id=
                    conversation_id,
            )
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_real_injection] "
            "select_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return forward_body

    if (
        not isinstance(
            selected_body,
            bytes,
        )
        or not isinstance(
            report,
            dict,
        )
    ):
        logger.warning(
            "[gateway.context_real_injection] "
            "invalid_selection fail_open=true"
        )
        return forward_body

    applied = report.get(
        "applied"
    )

    if applied is not True:
        if selected_body != forward_body:
            # Safety net: the body must never change without an
            # explicit applied=true decision.
            logger.warning(
                "[gateway.context_real_injection] "
                "unapplied_body_change fail_open=true"
            )
            return forward_body
    else:
        # Defense-in-depth: never blindly trust an applied=true
        # report from the selector.
        if (
            selected_body == forward_body
            or report.get("version")
            != "context-real-injection.v1"
            or report.get("enabled")
            is not True
            or report.get("reason")
            != "injection_applied"
            or report.get(
                "original_sha256"
            )
            != hashlib.sha256(
                forward_body
            ).hexdigest()
            or report.get(
                "selected_sha256"
            )
            != hashlib.sha256(
                selected_body
            ).hexdigest()
        ):
            logger.warning(
                "[gateway.context_real_injection] "
                "invalid_applied_selection "
                "fail_open=true"
            )
            return forward_body

    # Privacy-safe telemetry only.
    # No rendered Context, memory, fact or user text is logged.
    logger.info(
        "[gateway.context_real_injection] %s",
        json.dumps(
            {
                "conversation_id":
                    conversation_id,
                **report,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )

    return selected_body


async def run_context_pipeline(
    conversation_id: str | None,
    forward_body: bytes,
) -> bytes:
    """Run the whole Context chain for one request and return the body.

    This is the single entry point the gateway uses. It owns the
    fixed ordering, the per-request freshness latch and the real
    injection eligibility rule, so the gateway no longer orchestrates
    Unified / Preview / Gate / Mutation / Real Injection itself.

    Ordering matters: Unified -> Preview -> Gate -> freshness ->
    Mutation Shadow -> Real Injection. The Mutation Shadow must never
    read Preview/Gate left on disk by an earlier request.

    Default-OFF and fail-open: with
    OMBRE_GATEWAY_CONTEXT_REAL_INJECTION unset (or on any deny /
    exception / invalid report inside the selector) this returns the
    exact ``forward_body``.
    """

    context_chain_fresh = (
        await observe_unified_preview_gate(
            conversation_id
        )
    )

    if context_chain_fresh is True:
        observe_request_mutation(
            conversation_id,
            forward_body,
        )

    return select_real_injection(
        conversation_id,
        forward_body,
        context_chain_fresh=
            context_chain_fresh,
    )
