from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ombrebrain.context.validators.freshness import (
    validate_context_freshness,
)


# Memory Flash — shadow only.
#
# retrieved != surfaced
# surfaced   != noticed
# noticed    != recall_requested
# recall_requested != full_memory_loaded
# full_memory_loaded != used_in_response
#
# This phase implements exactly ONE step of that ladder:
#
#     retrieved -> surfaced_as_flash
#
# A Flash is a cue, not a full memory. It only says "there may be a
# past memory related to the current turn". It never puts a full
# memory into a prompt, it never enters Unified / Preview / Mutation /
# Real Injection, and it never reaches the model in this phase.
#
# No reinforcement occurs in this phase: nothing here writes to any
# memory source file, activation count, importance, strength, weight
# or decay.
#
# It consumes the retrieval evidence this request already produced
# (the persisted Unified candidate). It never re-runs retrieval,
# never embeds, never vector-searches and never calls a model or an
# external API.
#
# Cue source priority (no model-generated summary is ever created):
#   1. the memory title already carried by the Context layer
#      (``metadata.name``);
#   2. otherwise the memory representation already exposed to the
#      Context layer (``content``), reduced to a conservative bounded
#      prefix (trim + whitespace normalize + hard character cap).
#
# A whole raw memory, a conversation transcript, the current user
# message, rendered Context or a system prompt never become a cue.

_VERSION = "memory-flash.v1"
_MODE = "shadow_only"

# The Surfacing Policy contract this artifact may be built from. The
# name is repeated here (rather than imported) because
# memory_surfacing_policy imports memory_flash, not the other way
# round.
_POLICY_VERSION = "memory-surfacing-policy.v1"
_POLICY_MODE = "shadow_only"

_DEFAULT_ROOT = "/app/buckets/.context"

# Conservative first-version budget (see resolve_flash_budget).
_DEFAULT_MAX_ITEMS = 3
_DEFAULT_TOKEN_BUDGET = 96
_DEFAULT_MAX_CHARS = 160

# Hard caps: configuration can never open an unbounded Flash.
_MAX_ITEMS_CAP = 8
_TOKEN_BUDGET_CAP = 256
_MAX_CHARS_CAP = 320

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_COGNITIVE_REQUEST_ID_RE = re.compile(
    r"^ctxreq_[0-9a-f]{32}$"
)

_SPACE_RE = re.compile(r"\s+")

_LOCK = threading.RLock()


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


def _validate_cognitive_request_id(
    cognitive_request_id: str,
) -> None:
    if (
        not isinstance(
            cognitive_request_id,
            str,
        )
        or not _COGNITIVE_REQUEST_ID_RE.fullmatch(
            cognitive_request_id
        )
    ):
        raise ValueError(
            "invalid cognitive_request_id"
        )


def _path(
    kind: str,
    conversation_id: str,
    cognitive_request_id: str,
) -> Path:
    """One artifact file per (conversation, request).

    A per-conversation file would be overwritten by a concurrent
    request from the same conversation, so the request id is part of
    the path.
    """

    _validate_conversation_id(
        conversation_id
    )

    _validate_cognitive_request_id(
        cognitive_request_id
    )

    return (
        _root()
        / kind
        / conversation_id
        / (cognitive_request_id + ".json")
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


# ------------------------------------------------------
# Budget
# ------------------------------------------------------


def _parse_positive_int(
    raw: Any,
    *,
    default: int,
    cap: int,
) -> int:
    """Strict positive-int env parsing.

    Anything that is not a plain positive decimal integer (bool,
    float, negative, "true", "1e3", empty) falls back to the safe
    default. A parsed value is always clamped to ``cap``.
    """

    if raw is None:
        return default

    text = str(raw).strip()

    if not text.isdigit():
        return default

    try:
        value = int(text)
    except (
        TypeError,
        ValueError,
    ):
        return default

    if value < 1:
        return default

    return min(value, cap)


def resolve_flash_budget(
    *,
    max_items: Any = None,
    token_budget: Any = None,
    max_chars: Any = None,
) -> dict[str, int]:
    """Resolve the Flash budget from explicit values or environment.

    Invalid or extreme configuration can never break a request or
    open an unbounded Flash: it degrades to the safe default and is
    always clamped to the hard cap.
    """

    if max_items is None:
        max_items = os.environ.get(
            "OMBRE_MEMORY_FLASH_MAX_ITEMS"
        )

    if token_budget is None:
        token_budget = os.environ.get(
            "OMBRE_MEMORY_FLASH_TOKEN_BUDGET"
        )

    if max_chars is None:
        max_chars = os.environ.get(
            "OMBRE_MEMORY_FLASH_MAX_CHARS"
        )

    return {
        "max_items": _parse_positive_int(
            max_items,
            default=_DEFAULT_MAX_ITEMS,
            cap=_MAX_ITEMS_CAP,
        ),
        "token_budget": _parse_positive_int(
            token_budget,
            default=_DEFAULT_TOKEN_BUDGET,
            cap=_TOKEN_BUDGET_CAP,
        ),
        "max_chars": _parse_positive_int(
            max_chars,
            default=_DEFAULT_MAX_CHARS,
            cap=_MAX_CHARS_CAP,
        ),
    }


# ------------------------------------------------------
# Cue
# ------------------------------------------------------


def normalize_cue_text(
    value: Any,
) -> str:
    """Trim + collapse whitespace. Pure and total."""

    if not isinstance(value, str):
        return ""

    return _SPACE_RE.sub(
        " ",
        value.strip(),
    )


def _cue_source(
    candidate: Any,
) -> str:
    """The already-existing short representation, if any.

    ``metadata.name`` (a title) is preferred; ``content`` is the
    fallback. Nothing is generated and nothing is summarised here.
    """

    if not isinstance(
        candidate,
        dict,
    ):
        return ""

    metadata = candidate.get(
        "metadata"
    )

    if isinstance(
        metadata,
        dict,
    ):
        name = normalize_cue_text(
            metadata.get("name")
        )

        if name:
            return name

    return normalize_cue_text(
        candidate.get("content")
    )


def build_flash_cue(
    candidate: Any,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str | None:
    """Build one bounded cue, or None when nothing is flashable.

    Never returns a full raw memory: the result is always trimmed,
    whitespace-normalized and capped at ``max_chars``.
    """

    source = _cue_source(
        candidate
    )

    if not source:
        return None

    limit = _parse_positive_int(
        max_chars,
        default=_DEFAULT_MAX_CHARS,
        cap=_MAX_CHARS_CAP,
    )

    return source[:limit]


def estimate_cue_tokens(
    cue: str,
) -> int:
    """Same token heuristic the rest of the Context layer uses."""

    if not isinstance(cue, str) or not cue:
        return 0

    return max(
        1,
        (len(cue) + 2) // 3,
    )


def candidate_memory_id(
    candidate: Any,
) -> str | None:
    """Stable memory id already carried by the Context candidate."""

    if not isinstance(
        candidate,
        dict,
    ):
        return None

    memory_id = candidate.get("id")

    if (
        isinstance(
            memory_id,
            str,
        )
        and memory_id.strip()
    ):
        return memory_id.strip()

    return None


# ------------------------------------------------------
# Artifact
# ------------------------------------------------------


def build_memory_flash(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    source_unified_revision: Any,
    source_confidence_revision: Any,
    retrieved_candidate_count: int,
    eligible: Any,
    policy_report: Any,
    budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the privacy-safe Flash artifact for one request.

    ``eligible`` is the Surfacing Policy output (already filtered,
    already in canonical retrieval order). This function only turns
    each eligible memory into a bounded cue and applies the budget:
    there is no scoring, no ranking and no threshold here.
    """

    _validate_conversation_id(
        conversation_id
    )

    _validate_cognitive_request_id(
        cognitive_request_id
    )

    resolved = (
        budget
        if isinstance(
            budget,
            dict,
        )
        else resolve_flash_budget()
    )

    max_items = resolved["max_items"]
    token_budget = resolved["token_budget"]
    max_chars = resolved["max_chars"]

    policy = (
        policy_report
        if isinstance(
            policy_report,
            dict,
        )
        else {}
    )

    eligible_items = (
        eligible
        if isinstance(
            eligible,
            list,
        )
        else []
    )

    flashes: list[dict[str, Any]] = []

    used_tokens = 0
    rank = 0

    for candidate in eligible_items:
        if len(flashes) >= max_items:
            break

        memory_id = candidate_memory_id(
            candidate
        )

        if memory_id is None:
            continue

        cue = build_flash_cue(
            candidate,
            max_chars=max_chars,
        )

        if cue is None:
            continue

        cost = estimate_cue_tokens(
            cue
        )

        if used_tokens + cost > token_budget:
            # Later candidates do not fit; the budget is
            # not exceeded.
            break

        rank += 1

        flash: dict[str, Any] = {
            "memory_id": memory_id,
            "cue": cue,
            "reason": "surfacing_eligible",
            "estimated_tokens": cost,
            "rank": rank,
        }

        metadata = candidate.get(
            "metadata"
        )

        if isinstance(
            metadata,
            dict,
        ):
            source_type = metadata.get(
                "type"
            )

            if (
                isinstance(
                    source_type,
                    str,
                )
                and source_type.strip()
            ):
                flash["source_type"] = (
                    source_type.strip()
                )

        flashes.append(flash)

        used_tokens += cost

    decision = (
        "surfaced"
        if flashes
        else "no_surface"
    )

    reason = (
        "flash_surfaced"
        if flashes
        else (
            policy.get("reason")
            or "no_eligible_candidates"
        )
    )

    return {
        "version":
            _VERSION,
        "mode":
            _MODE,
        "conversation_id":
            conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "source_unified_revision":
            source_unified_revision,
        "source_confidence_revision":
            policy.get(
                "source_confidence_revision"
            ),
        "source_confidence_decision":
            policy.get(
                "confidence_decision"
            ),
        "source_confidence_reason":
            policy.get(
                "confidence_reason"
            ),
        "decision":
            decision,
        "reason":
            reason,
        "retrieved_candidate_count":
            retrieved_candidate_count,
        "eligible_candidate_count":
            len(eligible_items),
        "surfaced_count":
            len(flashes),
        "estimated_tokens":
            used_tokens,
        "token_budget":
            token_budget,
        "max_items":
            max_items,
        "max_chars":
            max_chars,
        "flashes":
            flashes,
    }


def _not_stored(
    reason: str,
) -> dict[str, Any]:
    """Refuse to build a Flash artifact, without raising."""

    return {
        "stored": False,
        "mode": _MODE,
        "decision": "no_surface",
        "reason": reason,
    }


def update_memory_flash(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    policy_report: Any,
    expected_unified_revision: Any,
    budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Persist one Flash artifact per (conversation, request).

    Shadow-only and fail-open at the caller. It refuses to build
    anything (``stored=False``, no artifact file) unless the Surfacing
    Policy report is a well-formed ``memory-surfacing-policy.v1``
    ``shadow_only`` report AND its ``source_unified_revision`` is a
    valid revision that equals ``expected_unified_revision``. That
    binding is checked with the shared
    ``validate_context_freshness()`` helper, so a policy produced for
    a different Unified revision can never be turned into a cue
    artifact for this request.

    Never raises for a refused report; it only raises on an invalid
    conversation / request id, exactly like the rest of this module.
    """

    _validate_conversation_id(
        conversation_id
    )

    _validate_cognitive_request_id(
        cognitive_request_id
    )

    if not isinstance(
        policy_report,
        dict,
    ):
        return _not_stored(
            "invalid_policy_report"
        )

    policy = policy_report

    if (
        policy.get("version")
        != _POLICY_VERSION
        or policy.get("mode")
        != _POLICY_MODE
    ):
        return _not_stored(
            "malformed_policy_report"
        )

    freshness = validate_context_freshness(
        checked_revision=(
            policy.get(
                "source_unified_revision"
            )
        ),
        expected_revision=(
            expected_unified_revision
        ),
        invalid_reason=(
            "invalid_flash_policy_unified_revision"
        ),
        mismatch_reason=(
            "flash_policy_unified_revision_mismatch"
        ),
    )

    if not freshness["valid"]:
        return _not_stored(
            freshness["reason"]
        )

    retrieved_candidate_count = (
        policy.get(
            "retrieved_candidate_count"
        )
    )

    if not _is_nonnegative_int(
        retrieved_candidate_count
    ):
        retrieved_candidate_count = None

    eligible = (
        policy.get("eligible")
        if policy.get("decision")
        == "allow_shadow"
        else []
    )

    artifact = build_memory_flash(
        conversation_id=conversation_id,
        cognitive_request_id=
            cognitive_request_id,
        source_unified_revision=(
            expected_unified_revision
        ),
        source_confidence_revision=(
            policy.get(
                "source_confidence_revision"
            )
        ),
        retrieved_candidate_count=(
            retrieved_candidate_count
        ),
        eligible=eligible,
        policy_report=policy,
        budget=budget,
    )

    artifact["created_at"] = _now()

    path = _path(
        "memory_flash",
        conversation_id,
        cognitive_request_id,
    )

    with _LOCK:
        _atomic_write(
            path,
            artifact,
        )

    output = dict(
        artifact
    )

    output["stored"] = True

    output.pop(
        "created_at",
        None,
    )

    return output


def memory_flash_status(
    *,
    conversation_id: str,
    cognitive_request_id: str,
) -> dict[str, Any]:
    """Read back one Flash artifact (privacy-safe summary only)."""

    state = _read_json(
        _path(
            "memory_flash",
            conversation_id,
            cognitive_request_id,
        )
    )

    if state is None:
        return {
            "exists": False,
        }

    return {
        "exists": True,
        "version": state.get("version"),
        "mode": state.get("mode"),
        "decision": state.get("decision"),
        "reason": state.get("reason"),
        "surfaced_count":
            state.get("surfaced_count"),
        "retrieved_candidate_count":
            state.get(
                "retrieved_candidate_count"
            ),
        "eligible_candidate_count":
            state.get(
                "eligible_candidate_count"
            ),
        "estimated_tokens":
            state.get("estimated_tokens"),
        "token_budget":
            state.get("token_budget"),
        "source_unified_revision":
            state.get(
                "source_unified_revision"
            ),
        "source_confidence_revision":
            state.get(
                "source_confidence_revision"
            ),
    }


def _is_nonnegative_int(
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