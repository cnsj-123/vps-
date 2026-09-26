from __future__ import annotations

from typing import Any

from ombrebrain.context.memory_flash import (
    build_flash_cue,
    candidate_memory_id,
)
from ombrebrain.context.retrieval_decision_shadow import (
    candidate_fingerprint,
)
from ombrebrain.context.validators.freshness import (
    is_valid_revision,
    validate_context_freshness,
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
# Evidence is bound per (memory_id, candidate_fingerprint), never by
# memory id alone: the same id can appear twice with different content
# and different conservative decisions, and bounding by id alone would
# let one candidate's "keep" leak onto another candidate's "drop".
# When several evidence entries share one identity and their decisions
# disagree, the policy fails closed with
# ``ambiguous_shadow_evidence`` instead of guessing.
#
# Admission-time dedup in the Unified builder is content based (its
# ``dedup_text``), so a duplicate *id* with a different content still
# reaches ``sections.memories`` and is deduplicated here by the
# policy's own memory-id check.
#
# Confidence is a signal, never a live gate. A persisted
# deny_shadow Confidence decision must NOT globally block every
# candidate: a valid memory candidate can still be worth surfacing.
# But the Confidence observation must be a well-formed report for THIS
# conversation and THIS Unified revision; otherwise it is not a
# usable evidence chain and this shadow does not run.

_VERSION = "memory-surfacing-policy.v1"
_MODE = "shadow_only"

_ALLOWED_DECISION = "allow_shadow"
_NO_SURFACE = "no_surface"

_CUE_PROBE_MAX_CHARS = 160

_CONFIDENCE_VERSION = "context-confidence-gate.v1"
_CONFIDENCE_MODE = "shadow_only"

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
) -> dict[
    tuple[str, str],
    list[dict[str, Any]],
]:
    """(memory_id, fingerprint) -> evidence entries. Total, read-only."""

    index: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

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

        fingerprint = entry.get(
            "candidate_fingerprint"
        )

        if (
            not isinstance(
                memory_id,
                str,
            )
            or not memory_id.strip()
            or not isinstance(
                fingerprint,
                str,
            )
            or not fingerprint.strip()
        ):
            continue

        key = (
            memory_id.strip(),
            fingerprint.strip(),
        )

        index.setdefault(
            key,
            [],
        ).append(entry)

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

    The Unified observation is validated before Confidence on purpose:
    a valid, bound ``source_unified_revision`` is then always recorded,
    so a structural Confidence failure still produces a ``no_surface``
    report a Flash artifact can be built from (while the Exposure
    Ledger independently records ``retrieved``).
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
        "confidence_binding":
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

    # The retrieved count is real from here on, whatever happens to
    # the Confidence observation or the evidence below: retrieval
    # already happened for THIS request.
    report["retrieved_candidate_count"] = len(
        memories
    )

    # --------------------------------------------------
    # 2. Confidence observation must be usable for THIS request
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
        report["confidence_binding"] = (
            "confidence_observation_invalid"
        )

        report["reason"] = (
            "confidence_observation_invalid"
        )

        report["reasons"] = [
            "confidence_observation_invalid"
        ]

        return report

    if (
        confidence.get("version")
        != _CONFIDENCE_VERSION
        or confidence.get("mode")
        != _CONFIDENCE_MODE
    ):
        report["confidence_binding"] = (
            "confidence_contract_mismatch"
        )

        report["reason"] = (
            "confidence_observation_invalid"
        )

        report["reasons"] = [
            "confidence_observation_invalid"
        ]

        return report

    if (
        confidence.get("conversation_id")
        != conversation_id
    ):
        report["confidence_binding"] = (
            "confidence_conversation_mismatch"
        )

        report["reason"] = (
            "confidence_observation_invalid"
        )

        report["reasons"] = [
            "confidence_observation_invalid"
        ]

        return report

    if not is_valid_revision(
        confidence.get("revision")
    ):
        report["confidence_binding"] = (
            "confidence_revision_invalid"
        )

        report["reason"] = (
            "confidence_observation_invalid"
        )

        report["reasons"] = [
            "confidence_observation_invalid"
        ]

        return report

    binding = validate_context_freshness(
        checked_revision=(
            confidence.get(
                "source_unified_revision"
            )
        ),
        expected_revision=(
            unified_revision
        ),
        invalid_reason=(
            "confidence_source_unified_revision_invalid"
        ),
        mismatch_reason=(
            "confidence_unified_revision_mismatch"
        ),
    )

    if not binding["valid"]:
        report["confidence_binding"] = (
            binding["reason"]
        )

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
    evidence: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ],
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

    entries = evidence.get(
        (
            memory_id,
            candidate_fingerprint(
                candidate
            ),
        )
    )

    if not entries:
        # No real evidence for this exact candidate: surface nothing
        # rather than inventing a decision.
        return memory_id, "missing_shadow_evidence"

    decisions = {
        (
            entry.get(
                "would_keep"
            )
            is True,
            (
                entry.get("reason")
                if isinstance(
                    entry.get("reason"),
                    str,
                )
                else ""
            ),
        )
        for entry in entries
    }

    if len(decisions) != 1:
        # Same identity, disagreeing conservative decisions. Do not
        # guess which one is this candidate.
        return (
            memory_id,
            "ambiguous_shadow_evidence",
        )

    would_keep, raw_reason = decisions.pop()

    if not would_keep:
        return (
            memory_id,
            _EVIDENCE_REASONS.get(
                raw_reason,
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