from __future__ import annotations

import json
import logging
import os

from ombrebrain.context.conversation_shadow import (
    observe_conversation_shadow,
)
from ombrebrain.context.conversation_snapshot import (
    update_conversation_snapshot,
)
from ombrebrain.context.conversation_compact import (
    update_conversation_compact,
)
from ombrebrain.context.conversation_trusted_facts import (
    update_conversation_trusted_facts,
)
from ombrebrain.context.conversation_semantic import (
    update_conversation_semantic,
)
from ombrebrain.context.conversation_semantic_state import (
    update_semantic_state,
)
from ombrebrain.context.conversation_context_candidate import (
    update_conversation_context_candidate,
)

logger = logging.getLogger("ombre_brain.gateway")


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def observe_context_sources(body: bytes) -> str | None:
    """
    Phase 4A shadow context pipeline.

    Request flow:
      conversation identity
        -> bounded snapshot
        -> deterministic compact
        -> conservative semantic frame

    Every stage is fail-open and none mutates the upstream request.
    """

    context_log_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_SHADOW"
        )
    )

    snapshot_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_SNAPSHOT_SHADOW"
        )
    )

    compact_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_COMPACT_SHADOW"
        )
    )

    semantic_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_SEMANTIC_SHADOW"
        )
    )

    semantic_state_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_SEMANTIC_STATE_SHADOW"
        )
    )

    trusted_facts_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_TRUSTED_FACTS_SHADOW"
        )
    )

    candidate_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_CANDIDATE_SHADOW"
        )
    )

    unified_candidate_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW"
        )
    )

    injection_preview_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW"
        )
    )

    injection_gate_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW"
        )
    )

    request_mutation_shadow_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW"
        )
    )

    # Phase 4A-3D: real injection is not an observability flag.
    # When it is enabled it must drive the same Context pipeline,
    # otherwise the selector could read stale Preview/Gate state
    # from disk.
    real_injection_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION"
        )
    )

    # Downstream Context consumers. Like Real Injection they are not
    # observability-only flags: the Confidence Gate and the Memory
    # Flash / Exposure Ledger observers consume Conversation Candidate
    # + Unified evidence, so enabling any of them must drive the
    # required upstream chain (Conversation Shadow -> Snapshot ->
    # Compact -> Trusted Facts / Semantic -> Semantic State ->
    # Conversation Candidate) instead of silently doing nothing.
    confidence_gate_enabled = _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW"
        )
    )

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

    memory_downstream_enabled = (
        confidence_gate_enabled
        or memory_flash_enabled
        or exposure_ledger_enabled
    )

    if not (
        context_log_enabled
        or snapshot_enabled
        or compact_enabled
        or semantic_enabled
        or semantic_state_enabled
        or trusted_facts_enabled
        or candidate_enabled
        or unified_candidate_enabled
        or injection_preview_enabled
        or injection_gate_enabled
        or request_mutation_shadow_enabled
        or real_injection_enabled
        or memory_downstream_enabled
    ):
        return

    # --------------------------------------------------------
    # 1. Conversation continuity
    # --------------------------------------------------------
    try:
        summary = observe_conversation_shadow(
            body
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_shadow] "
            "observer_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return

    if not summary.get("observed"):
        if context_log_enabled:
            logger.info(
                "[gateway.context_shadow] %s",
                json.dumps(
                    {
                        "observed": False,
                        "reason": summary.get(
                            "reason",
                            "unknown",
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        return

    conversation_id = summary.get(
        "conversation_id"
    )

    if context_log_enabled:
        logger.info(
            "[gateway.context_shadow] %s",
            json.dumps(
                {
                    "observed": True,
                    "conversation_id":
                        conversation_id,
                    "round": summary.get(
                        "round"
                    ),
                    "new_conversation":
                        summary.get(
                            "new_conversation"
                        ),
                    "duplicate":
                        summary.get(
                            "duplicate"
                        ),
                    "continuity":
                        summary.get(
                            "continuity"
                        ),
                    "messages_count":
                        summary.get(
                            "messages_count"
                        ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    if not isinstance(
        conversation_id,
        str,
    ):
        return

    # Semantic depends on Compact, which depends on Snapshot.
    if not (
        snapshot_enabled
        or compact_enabled
        or semantic_enabled
        or semantic_state_enabled
        or trusted_facts_enabled
        or candidate_enabled
        or unified_candidate_enabled
        or injection_preview_enabled
        or injection_gate_enabled
        or request_mutation_shadow_enabled
        or real_injection_enabled
        or memory_downstream_enabled
    ):
        return

    # --------------------------------------------------------
    # 2. Bounded Snapshot
    # --------------------------------------------------------
    try:
        snapshot = update_conversation_snapshot(
            body,
            conversation_id=conversation_id,
            boundary_prefix_sha256=summary.get(
                "boundary_prefix_sha256"
            ),
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_snapshot] "
            "store_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return

    logger.info(
        "[gateway.context_snapshot] %s",
        json.dumps(
            {
                "stored":
                    snapshot.get(
                        "stored"
                    ),
                "conversation_id":
                    conversation_id,
                "revision":
                    snapshot.get(
                        "revision"
                    ),
                "duplicate":
                    snapshot.get(
                        "duplicate"
                    ),
                "included_messages":
                    snapshot.get(
                        "included_messages"
                    ),
                "excluded_segments":
                    snapshot.get(
                        "excluded_segments"
                    ),
                "total_chars":
                    snapshot.get(
                        "total_chars"
                    ),
                "budget_truncated":
                    snapshot.get(
                        "budget_truncated"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )

    if not snapshot.get(
        "stored"
    ):
        return

    if not (
        compact_enabled
        or semantic_enabled
        or semantic_state_enabled
        or trusted_facts_enabled
        or candidate_enabled
        or unified_candidate_enabled
        or injection_preview_enabled
        or injection_gate_enabled
        or request_mutation_shadow_enabled
        or real_injection_enabled
        or memory_downstream_enabled
    ):
        return

    # --------------------------------------------------------
    # 3. Deterministic Compact
    # --------------------------------------------------------
    try:
        compact = update_conversation_compact(
            conversation_id
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_compact] "
            "store_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return

    logger.info(
        "[gateway.context_compact] %s",
        json.dumps(
            {
                "stored":
                    compact.get(
                        "stored"
                    ),
                "conversation_id":
                    conversation_id,
                "source_revision":
                    compact.get(
                        "source_revision"
                    ),
                "duplicate":
                    compact.get(
                        "duplicate"
                    ),
                "source_messages":
                    compact.get(
                        "source_messages"
                    ),
                "older_messages":
                    compact.get(
                        "older_messages"
                    ),
                "recent_messages":
                    compact.get(
                        "recent_messages"
                    ),
                "source_chars":
                    compact.get(
                        "source_chars"
                    ),
                "compact_chars":
                    compact.get(
                        "compact_chars"
                    ),
                "compaction_ratio":
                    compact.get(
                        "compaction_ratio"
                    ),
                "compaction_applied":
                    compact.get(
                        "compaction_applied"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )

    if not compact.get(
        "stored"
    ):
        return

    # --------------------------------------------------------
    # 4. Explicit Trusted Facts
    # --------------------------------------------------------
    if (
        trusted_facts_enabled
        or candidate_enabled
        or unified_candidate_enabled
        or injection_preview_enabled
        or injection_gate_enabled
        or request_mutation_shadow_enabled
        or real_injection_enabled
        or memory_downstream_enabled
    ):
        try:
            trusted_facts = (
                update_conversation_trusted_facts(
                    conversation_id
                )
            )
        except Exception as exc:
            logger.warning(
                "[gateway.context_trusted_facts] "
                "store_failed=%s fail_open=true",
                type(exc).__name__,
            )
        else:
            # Privacy-safe telemetry only; no fact text.
            logger.info(
                "[gateway.context_trusted_facts] %s",
                json.dumps(
                    {
                        "stored":
                            trusted_facts.get(
                                "stored"
                            ),
                        "conversation_id":
                            conversation_id,
                        "revision":
                            trusted_facts.get(
                                "revision"
                            ),
                        "source_revision":
                            trusted_facts.get(
                                "source_revision"
                            ),
                        "duplicate":
                            trusted_facts.get(
                                "duplicate"
                            ),
                        "fact_count":
                            trusted_facts.get(
                                "fact_count"
                            ),
                        "history_count":
                            trusted_facts.get(
                                "history_count"
                            ),
                        "commands_this_revision":
                            trusted_facts.get(
                                "commands_this_revision"
                            ),
                        "added_this_revision":
                            trusted_facts.get(
                                "added_this_revision"
                            ),
                        "confirmed_this_revision":
                            trusted_facts.get(
                                "confirmed_this_revision"
                            ),
                        "revoked_this_revision":
                            trusted_facts.get(
                                "revoked_this_revision"
                            ),
                        "replaced_this_revision":
                            trusted_facts.get(
                                "replaced_this_revision"
                            ),
                        "unmatched_this_revision":
                            trusted_facts.get(
                                "unmatched_this_revision"
                            ),
                        "inference_enabled":
                            trusted_facts.get(
                                "inference_enabled"
                            ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

    if not (
        semantic_enabled
        or semantic_state_enabled
        or candidate_enabled
        or unified_candidate_enabled
        or injection_preview_enabled
        or injection_gate_enabled
        or request_mutation_shadow_enabled
        or real_injection_enabled
        or memory_downstream_enabled
    ):
        return

    # --------------------------------------------------------
    # 5. Conservative Semantic Frame
    # --------------------------------------------------------
    try:
        semantic = update_conversation_semantic(
            conversation_id
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_semantic] "
            "store_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return

    # No semantic text is logged here.
    logger.info(
        "[gateway.context_semantic] %s",
        json.dumps(
            {
                "stored":
                    semantic.get(
                        "stored"
                    ),
                "conversation_id":
                    conversation_id,
                "source_revision":
                    semantic.get(
                        "source_revision"
                    ),
                "duplicate":
                    semantic.get(
                        "duplicate"
                    ),
                "source_messages":
                    semantic.get(
                        "source_messages"
                    ),
                "has_current_task":
                    semantic.get(
                        "has_current_task"
                    ),
                "current_task_carried_forward":
                    semantic.get(
                        "current_task_carried_forward"
                    ),
                "latest_user_ack_only":
                    semantic.get(
                        "latest_user_ack_only"
                    ),
                "task_recovered_from_history":
                    semantic.get(
                        "task_recovered_from_history"
                    ),
                "decision_count":
                    semantic.get(
                        "decision_count"
                    ),
                "constraint_count":
                    semantic.get(
                        "constraint_count"
                    ),
                "open_item_count":
                    semantic.get(
                        "open_item_count"
                    ),
                "fact_count":
                    semantic.get(
                        "fact_count"
                    ),
                "facts_deferred":
                    semantic.get(
                        "facts_deferred"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


    if not semantic.get(
        "stored"
    ):
        return

    if not (
        semantic_state_enabled
        or candidate_enabled
        or unified_candidate_enabled
        or injection_preview_enabled
        or injection_gate_enabled
        or request_mutation_shadow_enabled
        or real_injection_enabled
        or memory_downstream_enabled
    ):
        return

    # --------------------------------------------------------
    # 6. Persistent Semantic State
    # --------------------------------------------------------
    try:
        semantic_state = update_semantic_state(
            conversation_id
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_semantic_state] "
            "store_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return

    # No semantic text is logged.
    logger.info(
        "[gateway.context_semantic_state] %s",
        json.dumps(
            {
                "stored":
                    semantic_state.get(
                        "stored"
                    ),
                "conversation_id":
                    conversation_id,
                "revision":
                    semantic_state.get(
                        "revision"
                    ),
                "source_revision":
                    semantic_state.get(
                        "source_revision"
                    ),
                "duplicate":
                    semantic_state.get(
                        "duplicate"
                    ),
                "constraint_count":
                    semantic_state.get(
                        "constraint_count"
                    ),
                "decision_count":
                    semantic_state.get(
                        "decision_count"
                    ),
                "open_item_count":
                    semantic_state.get(
                        "open_item_count"
                    ),
                "history_count":
                    semantic_state.get(
                        "history_count"
                    ),
                "added_this_revision":
                    semantic_state.get(
                        "added_this_revision"
                    ),
                "refreshed_this_revision":
                    semantic_state.get(
                        "refreshed_this_revision"
                    ),
                "archived_this_revision":
                    semantic_state.get(
                        "archived_this_revision"
                    ),
                "purged_closed_echoes_this_revision":
                    semantic_state.get(
                        "purged_closed_echoes_this_revision"
                    ),
                "lifecycle_skipped_carried_forward":
                    semantic_state.get(
                        "lifecycle_skipped_carried_forward"
                    ),
                "lifecycle_command_detected":
                    semantic_state.get(
                        "lifecycle_command_detected"
                    ),
                "lifecycle_action":
                    semantic_state.get(
                        "lifecycle_action"
                    ),
                "lifecycle_kind":
                    semantic_state.get(
                        "lifecycle_kind"
                    ),
                "lifecycle_matched":
                    semantic_state.get(
                        "lifecycle_matched"
                    ),
                "lifecycle_replacement_added":
                    semantic_state.get(
                        "lifecycle_replacement_added"
                    ),
                "facts_deferred":
                    semantic_state.get(
                        "facts_deferred"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )

    if not semantic_state.get(
        "stored"
    ):
        return

    if not (
        candidate_enabled
        or unified_candidate_enabled
        or injection_preview_enabled
        or injection_gate_enabled
        or request_mutation_shadow_enabled
        or real_injection_enabled
        or memory_downstream_enabled
    ):
        return conversation_id

    # --------------------------------------------------------
    # 7. Conversation Context Candidate
    # --------------------------------------------------------
    try:
        candidate = (
            update_conversation_context_candidate(
                conversation_id
            )
        )
    except Exception as exc:
        logger.warning(
            "[gateway.context_candidate] "
            "store_failed=%s fail_open=true",
            type(exc).__name__,
        )
        return

    # Privacy-safe telemetry only.
    # Candidate text is never written to gateway logs.
    logger.info(
        "[gateway.context_candidate] %s",
        json.dumps(
            {
                "stored":
                    candidate.get(
                        "stored"
                    ),
                "conversation_id":
                    conversation_id,
                "revision":
                    candidate.get(
                        "revision"
                    ),
                "source_revision":
                    candidate.get(
                        "source_revision"
                    ),
                "duplicate":
                    candidate.get(
                        "duplicate"
                    ),
                "estimated_tokens":
                    candidate.get(
                        "estimated_tokens"
                    ),
                "token_budget":
                    candidate.get(
                        "token_budget"
                    ),
                "truncated":
                    candidate.get(
                        "truncated"
                    ),
                "budget_rejected":
                    candidate.get(
                        "budget_rejected"
                    ),
                "dedup_rejected":
                    candidate.get(
                        "dedup_rejected"
                    ),
                "control_rejected":
                    candidate.get(
                        "control_rejected"
                    ),
                "has_current_task":
                    candidate.get(
                        "has_current_task"
                    ),
                "trusted_fact_count":
                    candidate.get(
                        "trusted_fact_count"
                    ),
                "constraint_count":
                    candidate.get(
                        "constraint_count"
                    ),
                "decision_count":
                    candidate.get(
                        "decision_count"
                    ),
                "open_item_count":
                    candidate.get(
                        "open_item_count"
                    ),
                "recent_context_count":
                    candidate.get(
                        "recent_context_count"
                    ),
                "current_user_excluded":
                    candidate.get(
                        "current_user_excluded"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


    if not candidate.get(
        "stored"
    ):
        return None

    return conversation_id

