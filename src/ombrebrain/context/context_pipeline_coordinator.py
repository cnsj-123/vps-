from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
from typing import Any

from ombrebrain.context.context_confidence_gate import (
    update_context_confidence_gate,
)
from ombrebrain.context.exposure_ledger import (
    update_exposure_ledger,
)
from ombrebrain.context.memory_flash import (
    update_memory_flash,
)
from ombrebrain.context.memory_recall_surface import (
    update_recall_surface,
)
from ombrebrain.context.memory_surfacing_policy import (
    evaluate_surfacing_policy,
)
from ombrebrain.context.context_injection_gate import (
    update_context_injection_gate,
)
from ombrebrain.context.context_injection_preview import (
    update_context_injection_preview,
)
from ombrebrain.context.context_observation_pipeline import (
    observe_context_sources,
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
#     -> Confidence Gate (shadow only)
#       -> Memory Surfacing Policy (shadow only)
#         -> Memory Flash (shadow only)
#           -> Exposure Ledger (shadow only)
#             -> Preview
#               -> Gate
#                 -> per-request freshness latch
#                   -> Mutation Shadow
#                     -> Real Injection selector
#                       -> selected_body
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
#   - the Confidence Gate is an observer: its decision is never a
#     prerequisite and never changes the forwarded body;
#   - Memory Surfacing / Memory Flash / Exposure Ledger are observers
#     too: flash is a cue, not a full memory, it never enters
#     Unified / Preview / Mutation / Real Injection, and no
#     reinforcement of any memory ever occurs here;
#   - no HTTP error is produced and the request is never blocked;
#   - the NEW Memory Flash / Exposure Ledger log lines are
#     privacy-safe (counts, revisions, booleans, reason codes) and
#     never include a conversation id, a cognitive request id, a
#     memory id or any cue / memory / query text. That guarantee is
#     scoped to those new stages: the pre-existing Unified /
#     Preview / Gate / Mutation / Real Injection log lines are
#     unchanged, and some of them do still include conversation_id.
#
# This module performs no retrieval and no selection: it composes
# existing context stages only. All surfacing rules live in
# memory_surfacing_policy, all cue construction and Flash
# persistence live in memory_flash, and all ledger persistence
# lives in exposure_ledger.

logger = logging.getLogger("ombre_brain.gateway")


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def observe_context_confidence(
    conversation_id: str | None,
    *,
    expected_unified_revision: Any,
) -> dict[str, Any] | None:
    """Observe only whether existing Context evidence is trustworthy.

    Confidence Gate, shadow-only first version:

      - it evaluates evidence already produced by the Context chain
        (Conversation Candidate + Unified telemetry), it does not
        re-run retrieval, re-embed or copy Context text;
      - it is bound to THIS request: the caller passes the Unified
        revision this request just produced, so a Unified file
        already overwritten by a concurrent request is rejected
        (stored=False) instead of polluting shadow telemetry;
      - its decision is NOT a prerequisite: a deny_shadow never stops
        the Preview / Injection Gate / Mutation Shadow / Real
        Injection stages and never changes the forwarded body;
      - it is fail-open: a failure here logs only the exception type
        and the existing pipeline behaviour is unchanged;
      - only privacy-safe telemetry (revisions, counts, booleans,
        enums, reason codes) is logged.

    Returns the privacy-safe report for downstream shadow observers
    (Memory Surfacing Policy), or None when no reliable observation
    was produced. The return value never influences the live body.
    """

    if not _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW"
        )
    ):
        return None

    if not isinstance(
        conversation_id,
        str,
    ):
        return None

    try:
        report = update_context_confidence_gate(
            conversation_id,
            expected_unified_revision=
                expected_unified_revision,
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_confidence_gate] "
            "build_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return None

    if not isinstance(
        report,
        dict,
    ):
        logger.warning(
            "[gateway.context_confidence_gate] "
            "invalid_report fail_open=true"
        )
        return None

    # Explicit privacy-safe allowlist. The report is never spread
    # into the log line, so an unexpected extra field (text, ids)
    # can never leak, and conversation_id is deliberately not
    # logged here.
    logger.info(
        "[gateway.context_confidence_gate] %s",
        json.dumps(
            {
                "mode": report.get("mode"),
                "decision": report.get("decision"),
                "allowed": report.get("allowed"),
                "reason": report.get("reason"),
                "reasons": report.get("reasons"),
                "stored": report.get("stored"),
                "duplicate": report.get("duplicate"),
                "revision": report.get("revision"),
                "source_candidate_revision":
                    report.get(
                        "source_candidate_revision"
                    ),
                "source_unified_revision":
                    report.get(
                        "source_unified_revision"
                    ),
                "expected_unified_revision":
                    report.get(
                        "expected_unified_revision"
                    ),
                "observed_unified_revision":
                    report.get(
                        "observed_unified_revision"
                    ),
                "current_user_excluded":
                    report.get(
                        "current_user_excluded"
                    ),
                "retrieval_observation_available":
                    report.get(
                        "retrieval_observation_available"
                    ),
                "retrieval_candidate_count":
                    report.get(
                        "retrieval_candidate_count"
                    ),
                "usable_context_evidence":
                    report.get(
                        "usable_context_evidence"
                    ),
                "has_current_task":
                    report.get(
                        "has_current_task"
                    ),
                "state_included":
                    report.get(
                        "state_included"
                    ),
                "trusted_fact_count":
                    report.get(
                        "trusted_fact_count"
                    ),
                "constraint_count":
                    report.get(
                        "constraint_count"
                    ),
                "decision_count":
                    report.get(
                        "decision_count"
                    ),
                "open_item_count":
                    report.get(
                        "open_item_count"
                    ),
                "plan_count":
                    report.get(
                        "plan_count"
                    ),
                "memory_count":
                    report.get(
                        "memory_count"
                    ),
                "recent_context_count":
                    report.get(
                        "recent_context_count"
                    ),
                "estimated_tokens":
                    report.get(
                        "estimated_tokens"
                    ),
                "token_budget":
                    report.get(
                        "token_budget"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )

    return report


def observe_memory_exposure_shadow(
    conversation_id: str | None,
    *,
    cognitive_request_id: str | None,
    snapshot: Any,
    expected_unified_revision: Any,
    confidence_report: Any,
) -> None:
    """Observe which memories were retrieved / surfaced as a Flash cue.

    Shadow-only and fail-open. The ladder this phase implements is
    exactly one rung:

        retrieved -> surfaced_as_flash

    Being retrieved is not being surfaced, being surfaced is not
    being noticed, and none of it causes any reinforcement. No memory
    source file, activation count, importance, strength, weight,
    score, decay or archive state is ever written.

    It consumes the request-local snapshot THIS request produced
    (Unified artifact + conservative.v1 candidate evidence). It never
    re-reads the shared conversation-level Unified file -- which a
    concurrent same-conversation request may already have overwritten
    -- and never re-runs retrieval, re-embeds, re-vector-searches,
    re-ranks or calls a model / external API.

    Responsibilities are split: all eligibility rules live in
    memory_surfacing_policy, all cue construction and Flash
    persistence live in memory_flash, all ledger persistence lives in
    exposure_ledger. This function only orchestrates them.

    The Flash stage and the Ledger stage are independent. Because
    retrieved != surfaced, a Surfacing Policy / Flash failure still
    lets the Ledger record the ``retrieved`` stage for this request;
    only the Ledger's own failure (or the flag being OFF) skips it.

    Only privacy-safe telemetry is logged (mode, stored, decision,
    reason, counts, estimated tokens, token budget, revisions,
    booleans). A memory id, a cue, a conversation_id or a
    cognitive_request_id is never logged.
    """

    flash_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW"
        )
    )

    ledger_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_EXPOSURE_LEDGER_SHADOW"
        )
    )

    # Shadow-only Recall Surface. It is a model-facing safety boundary
    # built strictly from THIS request's real Flash: it never changes
    # the live body and never triggers a Recall. It needs the Flash,
    # so it only runs when the Flash stage ran.
    surface_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW"
        )
    )

    if not (
        flash_enabled
        or ledger_enabled
        or surface_enabled
    ):
        return

    if (
        not isinstance(conversation_id, str)
        or not conversation_id
        or not isinstance(
            cognitive_request_id,
            str,
        )
        or not cognitive_request_id
    ):
        return

    # Request-local Unified observation for THIS request only. The
    # conversation-level Unified file is shared state, so it is never
    # the source of a Memory observation.
    unified = (
        snapshot.get("unified")
        if (
            isinstance(snapshot, dict)
            and snapshot.get("version")
            == "memory-shadow-snapshot.v1"
            and snapshot.get(
                "conversation_id"
            )
            == conversation_id
        )
        else None
    )

    if (
        not isinstance(unified, dict)
        or unified.get("revision")
        != expected_unified_revision
    ):
        if flash_enabled:
            logger.info(
                "[gateway.context_memory_flash] %s",
                json.dumps(
                    {
                        "mode": "shadow_only",
                        "stored": False,
                        "decision": "no_surface",
                        "reason":
                            "unified_observation_unavailable",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

        if ledger_enabled:
            logger.info(
                "[gateway.context_exposure_ledger] %s",
                json.dumps(
                    {
                        "mode": "shadow_only",
                        "stored": False,
                        "decision": "no_record",
                        "reason":
                            "unified_observation_unavailable",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

        return

    # --------------------------------------------------
    # 1. Surfacing Policy -> Memory Flash
    # --------------------------------------------------

    flash_output = None

    if flash_enabled:
        policy = None

        # Surfacing Policy: deterministic, rule-based, canonical
        # order, and fed the real request-local evidence.
        try:
            policy = evaluate_surfacing_policy(
                conversation_id=conversation_id,
                unified=unified,
                confidence_report=(
                    confidence_report
                ),
                shadow_evidence=(
                    snapshot.get("evidence")
                    if isinstance(
                        snapshot,
                        dict,
                    )
                    else None
                ),
            )
        except Exception as exc:
            logger.warning(
                "[gateway.context_memory_flash] "
                "policy_failed=%s fail_open=true",
                type(exc).__name__,
            )

        if isinstance(policy, dict):
            try:
                flash_output = update_memory_flash(
                    conversation_id=(
                        conversation_id
                    ),
                    cognitive_request_id=(
                        cognitive_request_id
                    ),
                    policy_report=policy,
                    expected_unified_revision=(
                        expected_unified_revision
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "[gateway.context_memory_flash] "
                    "store_failed=%s fail_open=true",
                    type(exc).__name__,
                )
                flash_output = None

        if isinstance(flash_output, dict):
            logger.info(
                "[gateway.context_memory_flash] %s",
                json.dumps(
                    {
                        "mode":
                            flash_output.get(
                                "mode"
                            ),
                        "stored":
                            flash_output.get(
                                "stored"
                            ),
                        "decision":
                            flash_output.get(
                                "decision"
                            ),
                        "reason":
                            flash_output.get(
                                "reason"
                            ),
                        "retrieved_candidate_count":
                            flash_output.get(
                                "retrieved_candidate_count"
                            ),
                        "eligible_candidate_count":
                            flash_output.get(
                                "eligible_candidate_count"
                            ),
                        "surfaced_count":
                            flash_output.get(
                                "surfaced_count"
                            ),
                        "estimated_tokens":
                            flash_output.get(
                                "estimated_tokens"
                            ),
                        "token_budget":
                            flash_output.get(
                                "token_budget"
                            ),
                        "source_unified_revision":
                            flash_output.get(
                                "source_unified_revision"
                            ),
                        "source_confidence_revision":
                            flash_output.get(
                                "source_confidence_revision"
                            ),
                        "source_confidence_binding":
                            flash_output.get(
                                "source_confidence_binding"
                            ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

    # --------------------------------------------------
    # 1b. Recall Surface (shadow only, derived from the Flash)
    #
    # It turns the real Flash cues into opaque, request-scoped
    # memrefs. It is a model-facing safety boundary for a FUTURE live
    # exposure -- it does NOT put anything into the live body, does
    # not expose a raw memory id and does not trigger any recall.
    # --------------------------------------------------

    if (
        surface_enabled
        and isinstance(flash_output, dict)
    ):
        try:
            surface = update_recall_surface(
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
                flash_report=flash_output,
            )
        except Exception as exc:
            logger.warning(
                "[gateway.context_recall_surface] "
                "store_failed=%s fail_open=true",
                type(exc).__name__,
            )
            surface = None

        if isinstance(surface, dict):
            logger.info(
                "[gateway.context_recall_surface] %s",
                json.dumps(
                    {
                        "mode": surface.get("mode"),
                        "stored": surface.get(
                            "stored"
                        ),
                        "decision": surface.get(
                            "decision"
                        ),
                        "reason": surface.get(
                            "reason"
                        ),
                        "memory_count": surface.get(
                            "memory_count"
                        ),
                        "source_flash_unified_revision":
                            surface.get(
                                "source_flash_unified_revision"
                            ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

    # --------------------------------------------------
    # 2. Exposure Ledger (independent of Surfacing / Flash)
    #
    # retrieved != surfaced: this stage records that retrieval
    # happened for THIS request even when Surfacing or Flash failed,
    # was refused, or is turned off. Only its own failure skips it.
    # --------------------------------------------------

    if not ledger_enabled:
        return

    # Only a memory that actually entered the Flash artifact is
    # surfaced. An eligible memory rejected by the item / token
    # budget never counts as surfaced, and with Flash OFF nothing is.
    surfaced_ids: list[str] = []

    if isinstance(flash_output, dict):
        for item in (
            flash_output.get("flashes") or []
        ):
            if (
                isinstance(item, dict)
                and isinstance(
                    item.get("memory_id"),
                    str,
                )
                and item["memory_id"]
            ):
                surfaced_ids.append(
                    item["memory_id"]
                )

    try:
        ledger = update_exposure_ledger(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            unified=unified,
            surfaced_ids=surfaced_ids,
            expected_unified_revision=(
                expected_unified_revision
            ),
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_exposure_ledger] "
            "store_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return

    logger.info(
        "[gateway.context_exposure_ledger] %s",
        json.dumps(
            {
                "mode": ledger.get("mode"),
                "stored": ledger.get("stored"),
                "decision": ledger.get("decision"),
                "reason": ledger.get("reason"),
                "retrieved_count":
                    ledger.get("retrieved_count"),
                "surfaced_count":
                    ledger.get("surfaced_count"),
                "source_unified_revision":
                    ledger.get(
                        "source_unified_revision"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


async def observe_unified_preview_gate(
    conversation_id: str | None,
    *,
    cognitive_request_id: str | None = None,
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

    ``cognitive_request_id`` is the internal, per-request identity
    used only by the Memory Flash / Exposure Ledger observers. It is
    never added to the body, a header, a cache key or a log.
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

    confidence_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW"
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

    # Memory Flash / Exposure Ledger observers. Default OFF. When
    # either is on they also refresh the Unified chain for THIS
    # request, because they consume the retrieval observation this
    # request produced.
    memory_flash_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW"
        )
    )

    exposure_ledger_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_EXPOSURE_LEDGER_SHADOW"
        )
    )

    recall_surface_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW"
        )
    )

    memory_observer_enabled = (
        memory_flash_enabled
        or exposure_ledger_enabled
        or recall_surface_enabled
    )

    if not (
        unified_enabled
        or preview_enabled
        or gate_enabled
        or mutation_enabled
        or confidence_enabled
        or real_injection_enabled
        or memory_observer_enabled
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

    # Confidence Gate shadow observer. It sits right after Unified and
    # before Preview, but it is an observer only: its decision never
    # gates the Preview, the Injection Gate, the Mutation Shadow or
    # Real Injection, and it never changes the forwarded body.
    #
    # It is bound to the Unified revision THIS request just produced,
    # so a concurrent request that already overwrote the persisted
    # Unified cannot be mistaken for this request's evidence.
    confidence_report = None

    if unified.get(
        "stored"
    ):
        confidence_report = (
            observe_context_confidence(
                conversation_id,
                expected_unified_revision=(
                    unified.get("revision")
                ),
            )
        )

    # Memory Surfacing Policy -> Memory Flash -> Exposure Ledger.
    #
    # Shadow-only observers that sit after Confidence and before
    # Preview. They consume the retrieval observation THIS request
    # already produced (the persisted Unified candidate) and never
    # re-run retrieval, re-rank, embed or call a model. retrieved !=
    # surfaced: being retrieved is not being surfaced, and neither
    # causes any reinforcement. Their failure is fail-open and never
    # changes Preview / Gate / Mutation / Real Injection.
    if (
        memory_observer_enabled
        and unified.get("stored")
        and isinstance(
            cognitive_request_id,
            str,
        )
    ):
        try:
            observe_memory_exposure_shadow(
                conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
                snapshot=(
                    unified.get(
                        "memory_shadow_snapshot"
                    )
                ),
                expected_unified_revision=(
                    unified.get("revision")
                ),
                confidence_report=(
                    confidence_report
                ),
            )
        except Exception as exc:
            logger.warning(
                "[gateway.context_memory_flash] "
                "observer_failed=%s fail_open=true",
                type(exc).__name__,
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
    forward_body: bytes,
) -> bytes:
    """Run the whole Context chain for one request and return the body.

    This is the single entry point the gateway uses. It owns the
    whole Context chain, including the conversation observation
    stages, so the gateway no longer computes a conversation_id or
    orchestrates Unified / Preview / Gate / Mutation / Real
    Injection itself.

    Ordering matters: conversation sources -> Unified -> Confidence
    Shadow -> Memory Surfacing Policy Shadow -> Memory Flash Shadow ->
    Exposure Ledger Shadow -> Preview -> Injection Gate -> freshness
    -> Mutation Shadow -> Real Injection. The Mutation Shadow must
    never read Preview/Gate left on disk by an earlier request.

    The Confidence Shadow stage is observation-only: it is bound to
    the Unified revision this request just produced, and neither its
    result nor its presence is a prerequisite for any later stage.

    A per-request cognitive request id (``ctxreq_<32 hex>``) is
    minted here. It is used only by the Memory Flash / Exposure
    Ledger shadows; it is never added to the body, a header, a cache
    key, a request hash or a log, and it never reaches the model or
    affects Real Injection.

    Default-OFF and fail-open: with
    OMBRE_GATEWAY_CONTEXT_REAL_INJECTION unset (or on any deny /
    exception / invalid report inside the selector) this returns the
    exact ``forward_body``.

    There is also a top-level guard: a Context pipeline failure must
    never break live forwarding, so any unexpected error here returns
    ``forward_body`` unchanged and logs only the exception *type*.
    """

    try:
        # Internal, per-request identity for the shadow Memory
        # observers only. Never leaves shadow state.
        cognitive_request_id = (
            "ctxreq_"
            + secrets.token_hex(16)
        )

        conversation_id = (
            observe_context_sources(
                forward_body
            )
        )

        context_chain_fresh = (
            await observe_unified_preview_gate(
                conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
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

    except Exception as exc:
        # Fail-open: never break live forwarding. Only the exception
        # type is logged — never the message, query, memory, prompt
        # or rendered Context.
        logger.warning(
            "[gateway.context_pipeline] "
            "pipeline_failed=%s fail_open=true",
            type(exc).__name__,
        )

        return forward_body
