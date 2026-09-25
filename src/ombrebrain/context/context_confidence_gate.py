from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ombrebrain.context.validators.freshness import (
    is_valid_revision,
    validate_context_freshness,
)


# Context Confidence Gate — shadow only.
#
# Scope: the Confidence Gate answers exactly one question --
#   "Is the Context / Retrieval evidence we already have trustworthy
#    enough to be surfaced / recalled / injected in a future phase?"
#
# It is NOT the Injection Gate. It never re-checks revision
# freshness of the rendered chain, render SHA, mutation safety, body
# invariants, the token envelope or HTTP safety: those stay owned by
# the Injection Gate and the Real Injection selector.
#
# What it does own is the legality of the evidence it observes:
#   - every revision it reads (candidate revision / candidate
#     source_revision / unified revision) must be a valid 1-based
#     revision, checked with the shared is_valid_revision();
#   - the Candidate -> Unified source chain must match, checked with
#     the shared validate_context_freshness().
# It never validates Preview, the Injection Gate or the mutation
# chain.
#
# Per-request binding: the gate is told the Unified revision THIS
# request just produced, then reads the persisted Unified file and
# requires that revision to match. A mismatch means a concurrent
# request already overwrote it, so the observation is reported with
# stored=False and is never persisted as a decision.
#
# This first version is deliberately rule / reason based:
#   - deterministic, read-only, no model call, no LLM judge,
#     no external API, no new database, no live selector;
#   - it evaluates EXISTING evidence (persisted Context telemetry),
#     it does not generate new evidence, re-run retrieval, re-embed
#     or copy Context text;
#   - it never controls the pipeline: a "deny_shadow" decision does
#     NOT stop the request, change the forwarded body or act as a
#     prerequisite for the Preview / Injection Gate / Real
#     Injection stages;
#   - it is fail-open: any exception logs only the exception type and
#     keeps the existing pipeline behaviour.
#
# Privacy: the report and every log line contain counts, revisions,
# booleans, enums, hashes and reason codes only. No query, memory,
# trusted fact, conversation, prompt, rendered Context or current
# user text is ever stored or logged.

_VERSION = "context-confidence-gate.v1"
_MODE = "shadow_only"

_DEFAULT_ROOT = "/app/buckets/.context"

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_LOCK = threading.RLock()

# Section counts that count as usable, non-memory Context evidence.
_EVIDENCE_COUNT_FIELDS = (
    "trusted_fact_count",
    "constraint_count",
    "decision_count",
    "open_item_count",
    "plan_count",
    "memory_count",
    "recent_context_count",
)

# Section presence flags that count as usable Context evidence.
_EVIDENCE_BOOL_FIELDS = (
    "has_current_task",
    "state_included",
)

# Existing Conversation Candidate freshness flags. "stale" / "ahead"
# mean the carried-forward source is not the one this candidate was
# built from, so the evidence is not trustworthy yet.
_FRESHNESS_FLAGS = (
    "semantic_source_stale",
    "semantic_source_ahead",
    "trusted_facts_source_stale",
    "trusted_facts_source_ahead",
)

# The real legacy retrieval outcome contract
# (ombrebrain.context.service.get_candidates). The Confidence Gate
# only validates that the retrieval observation telemetry carries
# one of these values; it never interprets the outcome itself as a
# confidence deny. In particular embedding_disabled /
# no_semantic_scores / below_relevance_threshold / no_search_matches
# are valid observations, not evidence that the Context is invalid.
_KNOWN_RETRIEVAL_OUTCOMES = frozenset(
    (
        "empty_query",
        "included",
        "no_search_matches",
        "embedding_disabled",
        "no_semantic_scores",
        "below_relevance_threshold",
        "no_included_results",
    )
)


def _root() -> Path:
    return Path(
        (
            os.environ.get(
                "OMBRE_CONTEXT_STATE_DIR"
            )
            or _DEFAULT_ROOT
        ).strip()
    )


def _validate_conversation_id(
    conversation_id: str,
) -> None:
    if (
        not isinstance(
            conversation_id,
            str,
        )
        or not _CONVERSATION_ID_RE.fullmatch(
            conversation_id
        )
    ):
        raise ValueError(
            "invalid conversation_id"
        )


def _path(
    kind: str,
    conversation_id: str,
) -> Path:
    _validate_conversation_id(
        conversation_id
    )

    return (
        _root()
        / kind
        / (conversation_id + ".json")
    )


def _now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _read_json(
    path: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    return (
        value
        if isinstance(
            value,
            dict,
        )
        else None
    )


def _atomic_write(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        path.parent.chmod(0o700)
    except OSError:
        pass

    temp = path.with_name(
        path.name + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        temp.chmod(0o600)
    except OSError:
        pass

    os.replace(
        temp,
        path,
    )

    try:
        path.chmod(0o600)
    except OSError:
        pass


def _valid_nonnegative_int(
    value: Any,
) -> bool:
    return (
        isinstance(
            value,
            int,
        )
        and not isinstance(
            value,
            bool,
        )
        and value >= 0
    )


def _valid_positive_int(
    value: Any,
) -> bool:
    return (
        isinstance(
            value,
            int,
        )
        and not isinstance(
            value,
            bool,
        )
        and value >= 1
    )


def _as_dict(
    value: Any,
) -> dict[str, Any]:
    return (
        value
        if isinstance(
            value,
            dict,
        )
        else {}
    )


def evaluate_context_confidence(
    *,
    conversation_id: str,
    conversation_candidate: dict[str, Any],
    unified: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate whether existing Context evidence is trustworthy.

    Shadow decision only. Nothing here mutates a request, runs
    retrieval or touches the live pipeline.

    Reason codes describe *evidence quality*, never the live
    injection safety of the chain. An empty memory retrieval is not
    on its own a deny: a Context built from current_task,
    trusted_facts, constraints, decisions, state, plans or recent
    context is still usable evidence.
    """

    _validate_conversation_id(
        conversation_id
    )

    reasons: list[str] = []

    # --------------------------------------------------
    # 1. Identity / versions — missing or malformed input
    # --------------------------------------------------

    if (
        not isinstance(
            conversation_candidate,
            dict,
        )
        or conversation_candidate.get(
            "version"
        )
        != "conversation-context-candidate.v1"
    ):
        reasons.append(
            "invalid_conversation_candidate"
        )

    if (
        not isinstance(
            unified,
            dict,
        )
        or unified.get(
            "version"
        )
        != "unified-context-candidate.v1"
    ):
        reasons.append(
            "invalid_unified_candidate"
        )

    for name, payload in (
        (
            "conversation_candidate",
            conversation_candidate,
        ),
        (
            "unified",
            unified,
        ),
    ):
        if (
            isinstance(
                payload,
                dict,
            )
            and payload.get(
                "conversation_id"
            )
            != conversation_id
        ):
            reasons.append(
                name
                + "_conversation_mismatch"
            )

    # --------------------------------------------------
    # 2. Revision legality — shared freshness validator
    #
    # Missing / None / bool / 0 / negative / wrongly-typed
    # revisions are all invalid, so a malformed evidence chain can
    # never reach allow_shadow. The single shared validator is
    # reused; this module never re-implements a revision rule.
    # --------------------------------------------------

    candidate_revision = (
        conversation_candidate.get(
            "revision"
        )
        if isinstance(
            conversation_candidate,
            dict,
        )
        else None
    )

    candidate_source_revision = (
        conversation_candidate.get(
            "source_revision"
        )
        if isinstance(
            conversation_candidate,
            dict,
        )
        else None
    )

    unified_revision = (
        unified.get(
            "revision"
        )
        if isinstance(
            unified,
            dict,
        )
        else None
    )

    raw_source_revisions = (
        unified.get(
            "source_revisions"
        )
        if isinstance(
            unified,
            dict,
        )
        else None
    )

    if not is_valid_revision(
        candidate_revision
    ):
        reasons.append(
            "invalid_candidate_revision"
        )

    if not is_valid_revision(
        candidate_source_revision
    ):
        reasons.append(
            "invalid_candidate_source_revision"
        )

    if not is_valid_revision(
        unified_revision
    ):
        reasons.append(
            "invalid_unified_revision"
        )

    source_revisions = (
        raw_source_revisions
        if isinstance(
            raw_source_revisions,
            dict,
        )
        else {}
    )

    if not isinstance(
        raw_source_revisions,
        dict,
    ):
        reasons.append(
            "invalid_unified_source_revisions"
        )

    # --------------------------------------------------
    # 3. Candidate -> Unified source chain
    #
    # The Unified candidate must have been built from the
    # Conversation Candidate this gate is observing. Only this
    # evidence chain is verified: Preview / Injection Gate /
    # Mutation / Real Injection freshness stays owned by their own
    # stages and is never touched here.
    # --------------------------------------------------

    candidate_chain_freshness = (
        validate_context_freshness(
            checked_revision=(
                source_revisions.get(
                    "conversation_candidate"
                )
            ),
            expected_revision=(
                candidate_revision
            ),
            invalid_reason=(
                "invalid_unified_source_revisions"
            ),
            mismatch_reason=(
                "unified_source_candidate_revision_mismatch"
            ),
        )
    )

    if not candidate_chain_freshness[
        "valid"
    ]:
        reasons.append(
            candidate_chain_freshness[
                "reason"
            ]
        )

    conversation_source_freshness = (
        validate_context_freshness(
            checked_revision=(
                source_revisions.get(
                    "conversation_source"
                )
            ),
            expected_revision=(
                candidate_source_revision
            ),
            invalid_reason=(
                "invalid_unified_source_revisions"
            ),
            mismatch_reason=(
                "unified_source_conversation_revision_mismatch"
            ),
        )
    )

    if not conversation_source_freshness[
        "valid"
    ]:
        reasons.append(
            conversation_source_freshness[
                "reason"
            ]
        )

    # --------------------------------------------------
    # 4. Telemetry shape — malformed telemetry
    # --------------------------------------------------

    raw_candidate_telemetry = (
        conversation_candidate.get(
            "telemetry"
        )
        if isinstance(
            conversation_candidate,
            dict,
        )
        else None
    )

    candidate_telemetry = _as_dict(
        raw_candidate_telemetry
    )

    if not isinstance(
        raw_candidate_telemetry,
        dict,
    ):
        reasons.append(
            "malformed_candidate_telemetry"
        )

    raw_unified_telemetry = (
        unified.get(
            "telemetry"
        )
        if isinstance(
            unified,
            dict,
        )
        else None
    )

    unified_telemetry = _as_dict(
        raw_unified_telemetry
    )

    if not isinstance(
        raw_unified_telemetry,
        dict,
    ):
        reasons.append(
            "malformed_telemetry"
        )

    # --------------------------------------------------
    # 5. Current user must be excluded from carried Context
    # --------------------------------------------------

    if (
        candidate_telemetry.get(
            "current_user_excluded"
        )
        is not True
    ):
        reasons.append(
            "candidate_current_user_not_excluded"
        )

    if (
        unified_telemetry.get(
            "current_user_excluded"
        )
        is not True
    ):
        reasons.append(
            "unified_current_user_not_excluded"
        )

    # --------------------------------------------------
    # 6. Carried-forward source freshness
    # --------------------------------------------------

    for flag in _FRESHNESS_FLAGS:
        value = candidate_telemetry.get(
            flag
        )

        if value is True:
            reasons.append(flag)

        elif value is not False:
            reasons.append(
                flag + "_unknown"
            )

    # --------------------------------------------------
    # 7. Retrieval observation availability
    #
    # Only the STRUCTURE of the retrieval observation telemetry is
    # validated here: the outcome must be one of the known legacy
    # retrieval outcomes. Missing / non-string / empty / unknown is
    # an unavailable or malformed observation, never a judgement
    # about the memories themselves. "Unavailable" does NOT mean
    # "no memories matched": an empty result (empty_query /
    # no_search_matches / ...) is a valid observation, and
    # memory_count == 0 is never a deny on its own.
    # --------------------------------------------------

    retrieval_quality = (
        unified_telemetry.get(
            "retrieval_quality"
        )
    )

    retrieval_outcome = None

    if not isinstance(
        retrieval_quality,
        dict,
    ):
        retrieval_observation_available = (
            False
        )

        reasons.append(
            "retrieval_observation_unavailable"
        )

    else:
        retrieval_outcome = (
            retrieval_quality.get(
                "outcome"
            )
        )

        if retrieval_outcome is None:
            retrieval_observation_available = (
                False
            )

            reasons.append(
                "retrieval_observation_unavailable"
            )

        elif (
            not isinstance(
                retrieval_outcome,
                str,
            )
            or not retrieval_outcome.strip()
            or retrieval_outcome
            not in _KNOWN_RETRIEVAL_OUTCOMES
        ):
            retrieval_observation_available = (
                False
            )

            reasons.append(
                "malformed_telemetry"
            )

        else:
            retrieval_observation_available = (
                True
            )

    # --------------------------------------------------
    # 8. Section evidence
    # --------------------------------------------------

    # The stable Unified writer always emits every evidence count, so a
    # missing count is a malformed observation, never a silent 0.
    section_counts: dict[str, int | None] = {}

    for field in _EVIDENCE_COUNT_FIELDS:
        value = unified_telemetry.get(
            field
        )

        if _valid_nonnegative_int(
            value
        ):
            section_counts[field] = value

        else:
            section_counts[field] = None

            reasons.append(
                "malformed_telemetry"
            )

    # The stable Unified writer always emits these flags, so a
    # missing flag is malformed too. When present they must be a real
    # bool: "true" / "yes" / 1 / 0 / [] / {} are never coerced.
    section_presence: dict[str, bool | None] = {}

    for field in _EVIDENCE_BOOL_FIELDS:
        if field not in unified_telemetry:
            section_presence[field] = None

            reasons.append(
                "malformed_telemetry"
            )

            continue

        value = unified_telemetry[field]

        if not isinstance(
            value,
            bool,
        ):
            section_presence[field] = None

            reasons.append(
                "malformed_telemetry"
            )

        else:
            section_presence[field] = value

    usable_context_evidence = (
        any(
            isinstance(count, int)
            and count > 0
            for count in section_counts.values()
        )
        or any(
            value is True
            for value in section_presence.values()
        )
    )

    if not usable_context_evidence:
        reasons.append(
            "no_usable_context_evidence"
        )

    # --------------------------------------------------
    # 9. Numeric telemetry contract
    #
    # Only the numeric fields Confidence itself uses for its decision
    # and report are validated. This is deliberately not a generic
    # Unified schema framework: anti_echo / retrieval_dedup and the
    # like are left untouched.
    # --------------------------------------------------

    retrieval_candidate_count = (
        unified_telemetry.get(
            "retrieval_candidate_count"
        )
    )

    if not _valid_nonnegative_int(
        retrieval_candidate_count
    ):
        retrieval_candidate_count = None

        reasons.append(
            "malformed_telemetry"
        )

    estimated_tokens = (
        unified_telemetry.get(
            "estimated_tokens"
        )
    )

    if not _valid_nonnegative_int(
        estimated_tokens
    ):
        estimated_tokens = None

        reasons.append(
            "malformed_telemetry"
        )

    token_budget = (
        unified_telemetry.get(
            "token_budget"
        )
    )

    if not _valid_positive_int(
        token_budget
    ):
        token_budget = None

        reasons.append(
            "malformed_telemetry"
        )

    # --------------------------------------------------
    # Decision
    # --------------------------------------------------

    # One reason code per evidence problem, in first-seen order.
    reasons = list(
        dict.fromkeys(reasons)
    )

    decision = (
        "allow_shadow"
        if not reasons
        else "deny_shadow"
    )

    return {
        "version":
            _VERSION,
        "mode":
            _MODE,
        "decision":
            decision,
        "allowed":
            decision
            == "allow_shadow",
        "reason":
            (
                reasons[0]
                if reasons
                else None
            ),
        "reasons":
            reasons,
        "source_candidate_revision":
            candidate_revision,
        "source_unified_revision":
            unified_revision,
        "current_user_excluded":
            (
                candidate_telemetry.get(
                    "current_user_excluded"
                )
                is True
                and unified_telemetry.get(
                    "current_user_excluded"
                )
                is True
            ),
        "semantic_source_stale":
            candidate_telemetry.get(
                "semantic_source_stale"
            ),
        "trusted_facts_source_stale":
            candidate_telemetry.get(
                "trusted_facts_source_stale"
            ),
        "retrieval_observation_available":
            retrieval_observation_available,
        "retrieval_candidate_count":
            retrieval_candidate_count,
        "usable_context_evidence":
            usable_context_evidence,
        "has_current_task":
            section_presence[
                "has_current_task"
            ],
        "state_included":
            section_presence[
                "state_included"
            ],
        **section_counts,
        "estimated_tokens":
            estimated_tokens,
        "token_budget":
            token_budget,
    }


def update_context_confidence_gate(
    conversation_id: str,
    *,
    expected_unified_revision: Any = None,
) -> dict[str, Any]:
    """Evaluate and persist a privacy-safe confidence shadow report.

    Per-request binding: the caller must pass the Unified revision
    THIS request just produced. The persisted Unified file is read
    first and its revision must equal ``expected_unified_revision``;
    otherwise a concurrent request has already overwritten it and the
    observation belongs to someone else.

    A cross-request mismatch / malformed revision is NOT a valid
    evidence decision: it is reported with ``stored=False`` and is
    never written to the confidence_gate state. It also never raises,
    so the caller stays fail-open.

    Read-only over the existing Context state. The persisted file
    follows the same directory / atomic-write / monotonic revision
    conventions as the other Context stages and intentionally
    contains no rendered text.
    """

    _validate_conversation_id(
        conversation_id
    )

    if not is_valid_revision(
        expected_unified_revision
    ):
        return {
            "stored":
                False,
            "mode":
                _MODE,
            "decision":
                "deny_shadow",
            "allowed":
                False,
            "reason":
                "invalid_expected_unified_revision",
            "reasons": [
                "invalid_expected_unified_revision",
            ],
            "expected_unified_revision":
                None,
            "observed_unified_revision":
                None,
            "conversation_id":
                conversation_id,
        }

    # Read order: Unified first, verify it belongs to this request,
    # then the Candidate. If the Candidate is refreshed afterwards,
    # the existing Candidate -> Unified source chain check denies it.
    unified = _read_json(
        _path(
            "unified_context_candidate",
            conversation_id,
        )
    )

    if unified is None:
        return {
            "stored":
                False,
            "mode":
                _MODE,
            "decision":
                "deny_shadow",
            "allowed":
                False,
            "reason":
                "unified_candidate_not_found",
            "reasons": [
                "unified_candidate_not_found",
            ],
            "expected_unified_revision":
                expected_unified_revision,
            "observed_unified_revision":
                None,
            "conversation_id":
                conversation_id,
        }

    observed_unified_revision = (
        unified.get(
            "revision"
        )
        if isinstance(
            unified,
            dict,
        )
        else None
    )

    observed_freshness = (
        validate_context_freshness(
            checked_revision=
                observed_unified_revision,
            expected_revision=
                expected_unified_revision,
            invalid_reason=(
                "invalid_observed_unified_revision"
            ),
            mismatch_reason=(
                "confidence_unified_request_revision_mismatch"
            ),
        )
    )

    if not observed_freshness["valid"]:
        return {
            "stored":
                False,
            "mode":
                _MODE,
            "decision":
                "deny_shadow",
            "allowed":
                False,
            "reason":
                observed_freshness["reason"],
            "reasons": [
                observed_freshness["reason"],
            ],
            "expected_unified_revision":
                expected_unified_revision,
            "observed_unified_revision":
                observed_unified_revision,
            "conversation_id":
                conversation_id,
        }

    candidate = _read_json(
        _path(
            "context_candidate",
            conversation_id,
        )
    )

    if candidate is None:
        return {
            "stored":
                False,
            "mode":
                _MODE,
            "decision":
                "deny_shadow",
            "allowed":
                False,
            "reason":
                "conversation_candidate_not_found",
            "reasons": [
                "conversation_candidate_not_found",
            ],
            "expected_unified_revision":
                expected_unified_revision,
            "observed_unified_revision":
                observed_unified_revision,
            "conversation_id":
                conversation_id,
        }

    result = (
        evaluate_context_confidence(
            conversation_id=
                conversation_id,
            conversation_candidate=
                candidate,
            unified=
                unified,
        )
    )

    gate_path = _path(
        "confidence_gate",
        conversation_id,
    )

    with _LOCK:
        previous = _read_json(
            gate_path
        )

        if (
            isinstance(
                previous,
                dict,
            )
            and previous.get(
                "version"
            )
            == _VERSION
            and previous.get(
                "source_candidate_revision"
            )
            == result.get(
                "source_candidate_revision"
            )
            and previous.get(
                "source_unified_revision"
            )
            == result.get(
                "source_unified_revision"
            )
            and previous.get(
                "decision"
            )
            == result.get(
                "decision"
            )
            and previous.get(
                "reasons"
            )
            == result.get(
                "reasons"
            )
        ):
            output = dict(
                previous
            )

            output["stored"] = True
            output["duplicate"] = True

            output.pop(
                "created_at",
                None,
            )

            return output

        previous_revision = 0

        if isinstance(
            previous,
            dict,
        ):
            old_revision = (
                previous.get(
                    "revision"
                )
            )

            if (
                isinstance(
                    old_revision,
                    int,
                )
                and not isinstance(
                    old_revision,
                    bool,
                )
                and old_revision >= 1
            ):
                previous_revision = (
                    old_revision
                )

        payload = {
            **result,
            "conversation_id":
                conversation_id,
            "expected_unified_revision":
                expected_unified_revision,
            "revision":
                previous_revision + 1,
            "created_at":
                _now(),
        }

        _atomic_write(
            gate_path,
            payload,
        )

    output = dict(
        payload
    )

    output["stored"] = True
    output["duplicate"] = False
    output.pop(
        "created_at",
        None,
    )

    return output