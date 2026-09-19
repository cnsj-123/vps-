from __future__ import annotations

import hashlib
import json
import logging
import os

import httpx
from starlette.responses import JSONResponse, StreamingResponse

from ombrebrain.gateway import (
    cache_move_shadow_summary_from_body,
    cache_plan_summary_from_body,
    canonical_summary_from_body,
    fingertips_ownership_summary_from_body,
    rewrite_body_for_cache,
    rewrite_cache_stable_body,
    rewrite_shadow_summary_from_body,
)

from ombrebrain.gateway.cache_fingerprint import cache_fingerprint_summary_from_body
from ombrebrain.gateway.response_usage import ResponseUsageObserver
from ombrebrain.context.conversation_shadow import (
    observe_conversation_shadow,
)
from ombrebrain.context.conversation_snapshot import (
    update_conversation_snapshot,
)
from ombrebrain.context.conversation_compact import (
    update_conversation_compact,
)
from ombrebrain.context.conversation_semantic import (
    update_conversation_semantic,
)
from ombrebrain.context.conversation_semantic_state import (
    update_semantic_state,
)
from ombrebrain.context.conversation_trusted_facts import (
    update_conversation_trusted_facts,
)
from ombrebrain.context.conversation_context_candidate import (
    update_conversation_context_candidate,
)
from ombrebrain.context.unified_context_candidate import (
    update_unified_context_candidate_from_runtime,
)
from ombrebrain.context.context_injection_preview import (
    update_context_injection_preview,
)
from ombrebrain.context.context_injection_gate import (
    update_context_injection_gate,
)
from ombrebrain.gateway.gateway_runtime import (
    record_cache_usage,
    resolve_upstream_base,
)

logger = logging.getLogger("ombre_brain.gateway")

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

_REQUEST_DROP = _HOP_BY_HOP | {
    "host",
    "content-length",
}

_RESPONSE_DROP = _HOP_BY_HOP | {
    "content-length",
}

_OBSERVE_MAX_DEPTH = 8
_OBSERVE_MAX_ITEMS = 64
_SAFE_ENUM_KEYS = {"role", "type"}


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _upstream_base() -> str:
    base, _source = resolve_upstream_base(
        os.environ.get(
            "OMBRE_GATEWAY_UPSTREAM"
        )
    )

    return base


def _forward_request_headers(request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() in _REQUEST_DROP:
            continue
        headers[key] = value
    return headers


def _forward_response_headers(response: httpx.Response) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in response.headers.items():
        if key.lower() in _RESPONSE_DROP:
            continue
        headers[key] = value
    return headers


def _text_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _json_shape(value, *, key: str = "", depth: int = 0):
    """Describe JSON structure without logging scalar user content."""
    if depth >= _OBSERVE_MAX_DEPTH:
        return {"kind": "depth_limit"}

    if isinstance(value, dict):
        fields = {}
        items = list(value.items())
        for raw_key, child in items[:_OBSERVE_MAX_ITEMS]:
            child_key = str(raw_key)
            fields[child_key] = _json_shape(
                child,
                key=child_key,
                depth=depth + 1,
            )
        result = {
            "kind": "object",
            "field_count": len(items),
            "fields": fields,
        }
        if len(items) > _OBSERVE_MAX_ITEMS:
            result["truncated"] = True
        return result

    if isinstance(value, list):
        result = {
            "kind": "array",
            "count": len(value),
            "items": [
                _json_shape(child, key=key, depth=depth + 1)
                for child in value[:_OBSERVE_MAX_ITEMS]
            ],
        }
        if len(value) > _OBSERVE_MAX_ITEMS:
            result["truncated"] = True
        return result

    if isinstance(value, str):
        result = {
            "kind": "string",
            "chars": len(value),
            "sha256": _text_fingerprint(value),
        }
        if (
            key in _SAFE_ENUM_KEYS
            and len(value) <= 64
            and value.isprintable()
        ):
            result["enum"] = value
        return result

    if value is None:
        return {"kind": "null"}
    if isinstance(value, bool):
        return {"kind": "boolean"}
    if isinstance(value, (int, float)):
        return {"kind": "number"}

    return {"kind": type(value).__name__}


def _observe_canonical_request(body: bytes) -> None:
    if not _truthy(
        os.environ.get("OMBRE_GATEWAY_CANONICAL_OBSERVE")
    ):
        return

    summary = canonical_summary_from_body(body)

    if summary is None:
        logger.info(
            "[gateway.canonical] unsupported_or_non_json"
        )
        return

    logger.info(
        "[gateway.canonical] %s",
        json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _observe_rewrite_shadow(body: bytes) -> None:
    if not _truthy(
        os.environ.get("OMBRE_GATEWAY_REWRITE_SHADOW")
    ):
        return

    summary = rewrite_shadow_summary_from_body(body)

    if summary is None:
        logger.info(
            "[gateway.rewrite_shadow] unsupported_or_non_json"
        )
        return

    logger.info(
        "[gateway.rewrite_shadow] %s",
        json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _observe_cache_plan(body: bytes) -> None:
    if not _truthy(
        os.environ.get("OMBRE_GATEWAY_CACHE_PLAN_OBSERVE")
    ):
        return

    summary = cache_plan_summary_from_body(body)

    if summary is None:
        logger.info(
            "[gateway.cache_plan] unsupported_or_non_json"
        )
        return

    logger.info(
        "[gateway.cache_plan] %s",
        json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _observe_cache_move_shadow(
    body: bytes,
) -> None:
    if not _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CACHE_MOVE_SHADOW"
        )
    ):
        return

    summary = (
        cache_move_shadow_summary_from_body(
            body
        )
    )

    if summary is None:
        logger.info(
            "[gateway.cache_move_shadow] "
            "unsupported_or_non_json"
        )
        return

    logger.info(
        "[gateway.cache_move_shadow] %s",
        json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _observe_fingertips_ownership(
    body: bytes,
) -> None:
    if not _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_FINGERTIPS_PROBE"
        )
    ):
        return

    summary = (
        fingertips_ownership_summary_from_body(
            body
        )
    )

    if summary is None:
        logger.info(
            "[gateway.fingertips_probe] "
            "unsupported_or_non_json"
        )
        return

    logger.info(
        "[gateway.fingertips_probe] %s",
        json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _rewrite_upstream_body(body: bytes) -> bytes:
    if _truthy(
        os.environ.get(
            "OMBRE_GATEWAY_CACHE_STABLE"
        )
    ):
        try:
            transformed_body, report = (
                rewrite_cache_stable_body(body)
            )
        except Exception as exc:
            logger.warning(
                "[gateway.cache_stable] "
                "failed type=%s fail_open=true",
                type(exc).__name__,
            )
            return body

        logger.info(
            "[gateway.cache_stable] %s",
            json.dumps(
                report,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

        return transformed_body

    if not _truthy(
        os.environ.get("OMBRE_GATEWAY_REWRITE")
    ):
        return body

    try:
        rewritten_body, report = rewrite_body_for_cache(
            body
        )
    except Exception as exc:
        logger.warning(
            "[gateway.rewrite] failed type=%s fail_open=true",
            type(exc).__name__,
        )
        return body

    logger.info(
        "[gateway.rewrite] %s",
        json.dumps(
            report,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )

    return rewritten_body


def _observe_request(body: bytes, content_type: str) -> None:
    if not _truthy(os.environ.get("OMBRE_GATEWAY_OBSERVE")):
        return

    try:
        payload = json.loads(body)
    except Exception:
        logger.info(
            "[gateway.observe] non_json content_type=%s body_bytes=%d sha256=%s",
            str(content_type or "")[:100],
            len(body),
            hashlib.sha256(body).hexdigest()[:12],
        )
        return

    shape = _json_shape(payload)
    logger.info(
        "[gateway.observe] %s",
        json.dumps(shape, ensure_ascii=False, separators=(",", ":")),
    )



def _observe_context_shadow(body: bytes) -> None:
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


async def _observe_unified_context_shadow(
    conversation_id: str | None,
) -> None:

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

    if not (
        unified_enabled
        or preview_enabled
        or gate_enabled
    ):
        return

    if not isinstance(
        conversation_id,
        str,
    ):
        return

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
        return

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
    ):
        return

    if not unified.get(
        "stored"
    ):
        return

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
        return

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


    if not gate_enabled:
        return

    if not preview.get(
        "stored"
    ):
        return

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
        return

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



def register(mcp) -> None:

    @mcp.custom_route(
        "/gateway/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def gateway_proxy(request):
        try:
            upstream = _upstream_base()
        except RuntimeError as exc:
            logger.error("Gateway configuration error: %s", exc)
            return JSONResponse(
                {"error": {"type": "gateway_config_error", "message": str(exc)}},
                status_code=503,
            )

        path = str(request.path_params.get("path") or "").lstrip("/")

        if path:
            url = f"{upstream}/{path}"
        else:
            url = upstream
        if request.url.query:
            url += "?" + request.url.query

        body = await request.body()
        headers = _forward_request_headers(request)

        _observe_request(
            body,
            request.headers.get("content-type", ""),
        )

        _observe_canonical_request(body)
        _observe_rewrite_shadow(body)
        _observe_cache_plan(body)
        _observe_cache_move_shadow(body)
        _observe_fingertips_ownership(body)

        forward_body = _rewrite_upstream_body(body)

        # Phase 4A-1: observation-only conversation continuity.
        # Important: observe the cache-stabilized body so the stable
        # boundary used for continuity matches what is actually sent upstream.
        context_conversation_id = (
            _observe_context_shadow(
                forward_body
            )
        )

        await _observe_unified_context_shadow(
            context_conversation_id
        )

        if _truthy(
            os.environ.get(
                "OMBRE_GATEWAY_CACHE_FINGERPRINT_OBSERVE"
            )
        ):
            try:
                fingerprint = (
                    cache_fingerprint_summary_from_body(
                        forward_body
                    )
                )

                if fingerprint is not None:
                    logger.info(
                        "[gateway.cache_fingerprint] %s",
                        json.dumps(
                            fingerprint,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
            except Exception as exc:
                logger.warning(
                    "[gateway.cache_fingerprint] "
                    "observer_failed=%s",
                    type(exc).__name__,
                )

        logger.info(
            "[gateway] %s /%s body_bytes=%d",
            request.method,
            path,
            len(body),
        )

        client = httpx.AsyncClient(
            timeout=None,
            follow_redirects=False,
            trust_env=False,
        )

        try:
            upstream_request = client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                content=forward_body,
            )
            upstream_response = await client.send(upstream_request, stream=True)
        except Exception as exc:
            await client.aclose()
            logger.warning(
                "[gateway] upstream request failed: %s",
                type(exc).__name__,
            )
            return JSONResponse(
                {
                    "error": {
                        "type": "gateway_upstream_error",
                        "message": "Upstream request failed",
                    }
                },
                status_code=502,
            )

        response_headers = _forward_response_headers(upstream_response)

        usage_observer = None

        usage_log_enabled = _truthy(
            os.environ.get(
                "OMBRE_GATEWAY_RESPONSE_USAGE_OBSERVE"
            )
        )

        try:
            usage_observer = ResponseUsageObserver(
                upstream_response.headers.get(
                    "content-type",
                    "",
                )
            )
        except Exception as exc:
            logger.warning(
                "[gateway.response_usage] "
                "observer_init_failed=%s",
                type(exc).__name__,
            )

        async def relay_body():
            nonlocal usage_observer

            stream_completed = False

            try:
                async for chunk in upstream_response.aiter_raw():
                    if usage_observer is not None:
                        try:
                            usage_observer.feed(chunk)
                        except Exception as exc:
                            logger.warning(
                                "[gateway.response_usage] "
                                "observer_feed_failed=%s",
                                type(exc).__name__,
                            )

                            # Observation must never affect
                            # upstream streaming.
                            usage_observer = None

                    # Critical invariant:
                    # forward exactly the original raw chunk.
                    yield chunk

                stream_completed = True

            finally:
                if (
                    stream_completed
                    and usage_observer is not None
                ):
                    try:
                        summary = usage_observer.finish()

                        try:
                            record_cache_usage(
                                summary,
                                upstream=upstream,
                            )
                        except Exception as exc:
                            logger.warning(
                                "[gateway.cache_metrics] "
                                "record_failed=%s",
                                type(exc).__name__,
                            )

                        if usage_log_enabled:
                            logger.info(
                                "[gateway.response_usage] %s",
                                json.dumps(
                                    summary,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            )
                    except Exception as exc:
                        logger.warning(
                            "[gateway.response_usage] "
                            "observer_finish_failed=%s",
                            type(exc).__name__,
                        )

                await upstream_response.aclose()
                await client.aclose()

        return StreamingResponse(
            relay_body(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=None,
        )
