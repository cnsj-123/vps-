from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    # 2. Telemetry shape — malformed telemetry
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
    # 3. Current user must be excluded from carried Context
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
    # 4. Carried-forward source freshness
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
    # 5. Retrieval observation availability
    #
    # "Unavailable" means the retrieval observation telemetry was
    # not attached at all. It does NOT mean "no memories matched":
    # an empty result is a valid observation, so this gate never
    # turns memory_count == 0 into a deny on its own.
    # --------------------------------------------------

    retrieval_quality = (
        unified_telemetry.get(
            "retrieval_quality"
        )
    )

    retrieval_observation_available = (
        isinstance(
            retrieval_quality,
            dict,
        )
        and isinstance(
            retrieval_quality.get(
                "outcome"
            ),
            str,
        )
    )

    if not retrieval_observation_available:
        reasons.append(
            "retrieval_observation_unavailable"
        )

    # --------------------------------------------------
    # 6. Section evidence
    # --------------------------------------------------

    section_counts: dict[str, int] = {}

    for field in _EVIDENCE_COUNT_FIELDS:
        value = unified_telemetry.get(
            field
        )

        if _valid_nonnegative_int(
            value
        ):
            section_counts[field] = value

        else:
            section_counts[field] = 0

            if value is not None:
                reasons.append(
                    "malformed_telemetry"
                )

    section_presence = {
        field: (
            unified_telemetry.get(field)
            is True
        )
        for field in _EVIDENCE_BOOL_FIELDS
    }

    usable_context_evidence = (
        any(
            count > 0
            for count in section_counts.values()
        )
        or any(
            section_presence.values()
        )
    )

    if not usable_context_evidence:
        reasons.append(
            "no_usable_context_evidence"
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

    retrieval_candidate_count = (
        unified_telemetry.get(
            "retrieval_candidate_count"
        )
    )

    if not _valid_nonnegative_int(
        retrieval_candidate_count
    ):
        retrieval_candidate_count = None

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
            (
                conversation_candidate.get(
                    "revision"
                )
                if isinstance(
                    conversation_candidate,
                    dict,
                )
                else None
            ),
        "source_unified_revision":
            (
                unified.get(
                    "revision"
                )
                if isinstance(
                    unified,
                    dict,
                )
                else None
            ),
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
            (
                unified_telemetry.get(
                    "estimated_tokens"
                )
                if _valid_nonnegative_int(
                    unified_telemetry.get(
                        "estimated_tokens"
                    )
                )
                else None
            ),
        "token_budget":
            (
                unified_telemetry.get(
                    "token_budget"
                )
                if _valid_nonnegative_int(
                    unified_telemetry.get(
                        "token_budget"
                    )
                )
                else None
            ),
    }


def update_context_confidence_gate(
    conversation_id: str,
) -> dict[str, Any]:
    """Evaluate and persist a privacy-safe confidence shadow report.

    Read-only over the existing Context state. The persisted file
    follows the same directory / atomic-write / monotonic revision
    conventions as the other Context stages and intentionally
    contains no rendered text.
    """

    _validate_conversation_id(
        conversation_id
    )

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
            "decision":
                "deny_shadow",
            "allowed":
                False,
            "reason":
                "conversation_candidate_not_found",
            "reasons": [
                "conversation_candidate_not_found",
            ],
            "conversation_id":
                conversation_id,
        }

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
            "decision":
                "deny_shadow",
            "allowed":
                False,
            "reason":
                "unified_candidate_not_found",
            "reasons": [
                "unified_candidate_not_found",
            ],
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