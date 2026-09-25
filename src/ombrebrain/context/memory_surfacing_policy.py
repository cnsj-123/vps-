from __future__ import annotations

from typing import Any

from ombrebrain.context.memory_flash import (
    build_flash_cue,
    candidate_memory_id,
)
from ombrebrain.context.validators.freshness import (
    is_valid_revision,
)


# Memory Surfacing Policy Shadow — shadow only.
#
# retrieved != surfaced: being retrieved is not being surfaced. This
# policy answers one question only: which already-retrieved memory
# candidates are eligible to become a Memory Flash cue?
#
# It is deterministic and rule based. It adds no LLM judge, no
# classifier, no learned confidence, no new embedding, no new
# relevance model, no new score and no new threshold. It never
# re-ranks and never re-runs retrieval: the candidate order is the
# order the canonical Context pipeline already produced.
#
# Real evidence only. Eligibility consumes the request-local
# ``shadow_evidence`` sidecar produced for THIS request from the real
# retrieval candidate set by the existing conservative.v1
# ``RetrievalDecisionShadowObserver`` (recent-24h echo, duplicate id,
# exact text duplicate). There is deliberately NO synthetic
# per-candidate flag contract: the Unified memory item shape stays
# ``id`` / ``content`` / ``context_relevance`` / ``metadata`` and is
# never widened with shadow-only fields.
#
# Deduplication by memory id is the policy's own real check over the
# candidate ids; the admission-time dedup (duplicate id / exact
# duplicate content) has already removed candidates before they reach
# ``sections.memories``.
#
# Confidence is a signal, never a live gate. A persisted
# deny_shadow Confidence decision must NOT globally block every
# candidate: a valid memory candidate can still be worth surfacing.
# Only an unreliable Confidence observation (missing identity, an
# invalid / structural / out-of-order observation that was not
# stored) stops this shadow from running.

_VERSION = "memory-surfacing-policy.v1"
_MODE = "shadow_only"

_ALLOWED_DECISION = "allow_shadow"
_NO_SURFACE = "no_surface"

_CUE_PROBE_MAX_CHARS = 160

# The real conservative.v1 drop reasons mapped to the reason code this
# policy reports. Nothing else is inferred from the evidence.
_EVIDENCE_REASONS = {
    "recent_24h": "anti_echo_recent_24h",
    "duplicate_id": "retrieval_dedup_duplicate_id",
    "exact_text_duplicate":
        "retrieval_dedup_exact_text_duplicate",
}


def _evidence_index(
    shadow_evidence: Any,
) -> dict[str, dict[str, Any]]:
    """First-wins id -> evidence entry map. Total and read-only."""

    index: dict[str, dict[str, Any]] = {}

    if not isinstance(
        shadow_evidence,
        list,
    ):
        return index

    for entry in shadow_evidence:
        if not isinstance(
            entry,
            dict,
        ):
            continue

        memory_id = entry.get(
            "memory_id"
        )

        if (
            not isinstance(
                memory_id,
                str,
            )
            or not memory_id.strip()
        ):
            continue

        memory_id = memory_id.strip()

        if memory_id in index:
            continue

        index[memory_id] = entry

    return index


def evaluate_surfacing_policy(
    *,
    conversation_id: str,
    unified: Any = None,
    confidence_report: Any = None,
    shadow_evidence: Any = None,
) -> dict[str, Any]:
    """Decide which retrieved candidates are eligible to be surfaced.

    Pure and read-only: it touches no file, no request and no live
    pipeline. It never raises on malformed input -- it reports
    ``no_surface`` instead.

    ``shadow_evidence`` is the request-local conservative.v1 evidence
    for this request's retrieval candidate set. Without it there is
    no honest anti-echo / dedup evidence, so the policy surfaces
    nothing rather than pretending.
    """

    report: dict[str, Any] = {
        "version":
            _VERSION,
        "mode":
            _MODE,
        "conversation_id":
            conversation_id,
        "source_unified_revision":
            None,
        "source_confidence_revision":
            None,
        "confidence_decision":
            None,
        "confidence_reason":
            None,
        "decision":
            _NO_SURFACE,
        "reason":
            "no_eligible_candidates",
        "reasons": [],
        "retrieved_candidate_count":
            0,
        "eligible_candidate_count":
            0,
        "evidence_count":
            0,
        "eligible": [],
        "ineligible": [],
    }

    if not isinstance(
        conversation_id,
        str,
    ) or not conversation_id:
        report["reason"] = (
            "invalid_conversation_id"
        )

        report["reasons"] = [
            "invalid_conversation_id"
        ]

        return report

    # --------------------------------------------------
    # 1. Unified observation must be the real, current one
    #
    # It is validated before Confidence so that a valid, bound
    # ``source_unified_revision`` is always recorded -- even when the
    # Confidence observation later turns out to be unusable. That is
    # what lets the Flash report ``no_surface`` with
    # ``confidence_observation_invalid`` instead of refusing to build
    # an artifact at all, while the Ledger records ``retrieved``.
    # --------------------------------------------------

    if (
        not isinstance(
            unified,
            dict,
        )
        or unified.get("version")
        != "unified-context-candidate.v1"
    ):
        report["reason"] = (
            "unified_observation_invalid"
        )

        report["reasons"] = [
            "unified_observation_invalid"
        ]

        return report

    if (
        unified.get("conversation_id")
        != conversation_id
    ):
        report["reason"] = (
            "unified_conversation_mismatch"
        )

        report["reasons"] = [
            "unified_conversation_mismatch"
        ]

        return report

    unified_revision = unified.get(
        "revision"
    )

    if not is_valid_revision(
        unified_revision
    ):
        report["reason"] = (
            "invalid_unified_revision"
        )

        report["reasons"] = [
            "invalid_unified_revision"
        ]

        return report

    report["source_unified_revision"] = (
        unified_revision
    )

    telemetry = unified.get("telemetry")

    if not isinstance(
        telemetry,
        dict,
    ):
        report["reason"] = "malformed_telemetry"

        report["reasons"] = [
            "malformed_telemetry"
        ]

        return report

    # The current user's own content must not be carried into the
    # surfaced set.
    if (
        telemetry.get(
            "current_user_excluded"
        )
        is not True
    ):
        report["reason"] = (
            "current_user_not_excluded"
        )

        report["reasons"] = [
            "current_user_not_excluded"
        ]

        return report

    sections = unified.get("sections")

    if not isinstance(
        sections,
        dict,
    ):
        report["reason"] = "malformed_telemetry"

        report["reasons"] = [
            "malformed_telemetry"
        ]

        return report

    memories = sections.get("memories")

    if not isinstance(
        memories,
        list,
    ):
        report["reason"] = "malformed_telemetry"

        report["reasons"] = [
            "malformed_telemetry"
        ]

        return report

    # --------------------------------------------------
    # 2. Confidence observation must be reliable
    # --------------------------------------------------

    if not isinstance(
        confidence_report,
        dict,
    ):
        report["reason"] = (
            "confidence_observation_missing"
        )

        report["reasons"] = [
            "confidence_observation_missing"
        ]

        return report

    confidence = confidence_report

    report["source_confidence_revision"] = (
        confidence.get("revision")
    )

    report["confidence_decision"] = (
        confidence.get(
            "decision"
        )
    )

    report["confidence_reason"] = (
        confidence.get("reason")
    )

    if confidence.get("stored") is not True:
        # A structural / identity / revision / out-of-order
        # observation is not a valid evidence chain, so this shadow
        # does not use it.
        report["reason"] = (
            "confidence_observation_invalid"
        )

        report["reasons"] = [
            "confidence_observation_invalid"
        ]

        return report

    # --------------------------------------------------
    # 3. Real request-local anti-echo / dedup evidence
    # --------------------------------------------------

    if not isinstance(
        shadow_evidence,
        list,
    ):
        report["reason"] = (
            "shadow_evidence_unavailable"
        )

        report["reasons"] = [
            "shadow_evidence_unavailable"
        ]

        return report

    evidence = _evidence_index(
        shadow_evidence
    )

    report["evidence_count"] = len(
        evidence
    )

    # --------------------------------------------------
    # 4. Per-candidate eligibility, in canonical order
    # --------------------------------------------------

    eligible: list[Any] = []
    ineligible: list[dict[str, Any]] = []
    reasons: list[str] = []
    seen_ids: set[str] = set()

    for candidate in memories:
        memory_id, reason = (
            _candidate_eligibility(
                candidate,
                seen_ids=seen_ids,
                evidence=evidence,
            )
        )

        if reason == "surfacing_eligible":
            seen_ids.add(memory_id)

            eligible.append(candidate)

            continue

        ineligible.append(
            {
                "memory_id": memory_id,
                "reason": reason,
            }
        )

        if reason not in reasons:
            reasons.append(reason)

    report["retrieved_candidate_count"] = len(
        memories
    )

    report["eligible_candidate_count"] = len(
        eligible
    )

    report["eligible"] = eligible

    report["ineligible"] = ineligible

    if eligible:
        report["decision"] = _ALLOWED_DECISION

        report["reason"] = (
            "surfacing_eligible"
        )

        report["reasons"] = [
            "surfacing_eligible"
        ]

    else:
        report["reason"] = (
            reasons[0]
            if reasons
            else "no_eligible_candidates"
        )

        report["reasons"] = (
            reasons
            if reasons
            else ["no_eligible_candidates"]
        )

    return report


def _candidate_eligibility(
    candidate: Any,
    *,
    seen_ids: set[str],
    evidence: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    """Return (memory_id, reason). Deterministic and total."""

    if not isinstance(
        candidate,
        dict,
    ):
        return None, "malformed_candidate"

    memory_id = candidate_memory_id(
        candidate
    )

    if memory_id is None:
        return None, "missing_memory_id"

    if memory_id in seen_ids:
        return memory_id, "duplicate_memory_id"

    entry = evidence.get(memory_id)

    if entry is None:
        # No real evidence for this candidate: surface nothing rather
        # than inventing a decision.
        return memory_id, "missing_shadow_evidence"

    if entry.get("would_keep") is not True:
        raw_reason = entry.get("reason")

        return (
            memory_id,
            _EVIDENCE_REASONS.get(
                raw_reason
                if isinstance(
                    raw_reason,
                    str,
                )
                else "",
                "shadow_evidence_rejected",
            ),
        )

    if (
        build_flash_cue(
            candidate,
            max_chars=_CUE_PROBE_MAX_CHARS,
        )
        is None
    ):
        return memory_id, "no_flashable_cue"

    return memory_id, "surfacing_eligible"