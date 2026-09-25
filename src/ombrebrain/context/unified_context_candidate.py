from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ombrebrain.context.retrieval_decision_shadow import (
    build_candidate_evidence,
)
from ombrebrain.context.retrieval_shadow_metrics import (
    record_retrieval_shadow_metrics,
)


_VERSION = "unified-context-candidate.v1"

_DEFAULT_ROOT = "/app/buckets/.context"
_DEFAULT_TOKEN_BUDGET = 1200

_MAX_PLANS = 4
_MAX_MEMORIES = 6

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_SPACE_RE = re.compile(r"\s+")

_LOCK = threading.RLock()

_CONTEXT_SERVICE = None


def bind_context_service(
    service: Any,
) -> None:
    """Bind the already-initialized OB ContextService.

    This does not create or replace any memory/state engine.
    """
    global _CONTEXT_SERVICE
    _CONTEXT_SERVICE = service


def _current_user_query(
    conversation_id: str,
    conversation_candidate: dict[str, Any],
) -> str:
    """Read current user text only for retrieval.

    The query is never copied into the unified candidate.
    """
    compact = _read_json(
        _path(
            "compact",
            conversation_id,
        )
    )

    if not isinstance(
        compact,
        dict,
    ):
        return ""

    if (
        compact.get("source_revision")
        != conversation_candidate.get(
            "source_revision"
        )
    ):
        return ""

    telemetry = (
        conversation_candidate.get(
            "telemetry"
        )
        or {}
    )

    expected_index = telemetry.get(
        "latest_user_source_index"
    )

    messages = compact.get(
        "recent_messages"
    )

    if not isinstance(
        messages,
        list,
    ):
        return ""

    for item in reversed(messages):
        if not isinstance(
            item,
            dict,
        ):
            continue

        if item.get("role") != "user":
            continue

        source_index = item.get(
            "source_index"
        )

        if (
            isinstance(
                expected_index,
                int,
            )
            and source_index
            != expected_index
        ):
            continue

        text = item.get("text")

        if (
            isinstance(
                text,
                str,
            )
            and text.strip()
        ):
            return text.strip()[:3000]

    return ""


async def update_unified_context_candidate_from_runtime(
    conversation_id: str,
) -> dict[str, Any]:
    """Build unified shadow candidate from the live ContextService."""

    _validate_conversation_id(
        conversation_id
    )

    service = _CONTEXT_SERVICE

    if service is None:
        return {
            "stored": False,
            "reason":
                "context_service_unavailable",
            "conversation_id":
                conversation_id,
        }

    conversation_candidate = _read_json(
        _path(
            "context_candidate",
            conversation_id,
        )
    )

    if not isinstance(
        conversation_candidate,
        dict,
    ):
        return {
            "stored": False,
            "reason":
                "conversation_candidate_not_found",
            "conversation_id":
                conversation_id,
        }

    query = _current_user_query(
        conversation_id,
        conversation_candidate,
    )

    context_candidates = await service.get_candidates(
        query=query
    )

    shadow_observation: dict[str, Any] = {}

    result = update_unified_context_candidate(
        conversation_id=
            conversation_id,
        context_candidates=
            context_candidates,
        excluded_texts=(
            (query,)
            if query
            else ()
        ),
        shadow_observation=
            shadow_observation,
    )

    result["retrieval_query_used"] = bool(
        query
    )

    # Request-local, shadow-only Memory observation snapshot.
    #
    # It is returned only to the in-process shadow Memory observers so
    # they consume THIS request's own Unified artifact and retrieval
    # evidence instead of re-reading the shared conversation-level
    # file (which a concurrent request may already have overwritten).
    # It is never persisted, never logged and never part of Unified
    # sections, Preview, Real Injection or a cache key.
    try:
        result["memory_shadow_snapshot"] = (
            build_memory_shadow_snapshot(
                conversation_id=
                    conversation_id,
                unified=(
                    shadow_observation.get(
                        "unified_artifact"
                    )
                ),
                retrieval_memories=(
                    context_candidates.get(
                        "memories"
                    )
                    if isinstance(
                        context_candidates,
                        dict,
                    )
                    else []
                ),
            )
        )

    except Exception:
        # Observation must remain fail-open.
        result["memory_shadow_snapshot"] = None

    # Privacy-safe rolling metrics only.
    # This is observation-only and must never affect retrieval,
    # Unified candidate assembly or upstream forwarding.
    try:
        source_telemetry = (
            context_candidates.get(
                "telemetry"
            )
            if isinstance(
                context_candidates,
                dict,
            )
            else {}
        )

        metrics = (
            record_retrieval_shadow_metrics(
                conversation_id=
                    conversation_id,
                revision=
                    result.get(
                        "revision"
                    ),
                telemetry=(
                    source_telemetry
                    if isinstance(
                        source_telemetry,
                        dict,
                    )
                    else {}
                ),
                retrieval_query_used=
                    bool(query),
            )
        )

    except Exception:
        # Metrics must remain fail-open.
        metrics = {
            "stored": False,
            "reason":
                "metrics_store_failed",
        }

    result[
        "retrieval_metrics"
    ] = metrics

    return result



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

    if not isinstance(
        value,
        dict,
    ):
        return None

    return value


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
        str(temp),
        str(path),
    )


def _canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _estimate_tokens(
    value: Any,
) -> int:
    if value is None:
        return 0

    text = (
        value
        if isinstance(value, str)
        else _canonical_json(value)
    )

    if not text:
        return 0

    return max(
        1,
        (len(text) + 2) // 3,
    )


def _normalize_key(
    text: str,
) -> str:
    return _SPACE_RE.sub(
        " ",
        text.strip(),
    ).casefold()


def _fingerprint(
    value: Any,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            value
        ).encode("utf-8")
    ).hexdigest()


def _clean_conversation_item(
    raw: Any,
) -> dict[str, Any] | None:
    if not isinstance(
        raw,
        dict,
    ):
        return None

    text = raw.get("text")

    if (
        not isinstance(text, str)
        or not text.strip()
    ):
        return None

    item = {
        "text": text.strip(),
    }

    for key in (
        "id",
        "provenance",
        "role",
        "source_index",
    ):
        value = raw.get(key)

        if value is not None:
            item[key] = value

    return item


def _plan_item(
    raw: Any,
) -> tuple[
    dict[str, Any],
    str,
] | None:

    if not isinstance(
        raw,
        dict,
    ):
        return None

    status = str(
        raw.get("status")
        or "active"
    ).lower()

    if status != "active":
        return None

    name = str(
        raw.get("name")
        or ""
    ).strip()

    content = str(
        raw.get("content")
        or ""
    ).strip()

    if not name and not content:
        return None

    output = {
        "id": raw.get("id"),
        "name": name,
        "content": content,
        "status": status,
        "weight": raw.get(
            "weight"
        ),
    }

    why = str(
        raw.get("why_remembered")
        or ""
    ).strip()

    if why:
        output[
            "why_remembered"
        ] = why

    dedup_text = (
        content
        or name
    )

    return output, dedup_text


def _memory_item(
    raw: Any,
) -> tuple[
    dict[str, Any],
    str,
] | None:

    if not isinstance(
        raw,
        dict,
    ):
        return None

    content = str(
        raw.get("content")
        or raw.get("text")
        or ""
    ).strip()

    if not content:
        return None

    output = {
        "id": raw.get("id"),
        "content": content,
    }

    relevance = raw.get(
        "context_relevance"
    )

    if isinstance(
        relevance,
        (int, float),
    ) and not isinstance(
        relevance,
        bool,
    ):
        output[
            "context_relevance"
        ] = round(
            float(relevance),
            4,
        )

    meta = raw.get(
        "metadata"
    )

    if isinstance(
        meta,
        dict,
    ):
        safe_meta = {}

        for key in (
            "type",
            "domain",
            "name",
        ):
            value = meta.get(key)

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):
                safe_meta[key] = (
                    value.strip()
                )

        if safe_meta:
            output[
                "metadata"
            ] = safe_meta

    return output, content


def build_unified_context_candidate(
    *,
    conversation_id: str,
    conversation_candidate: dict[str, Any],
    context_candidates: dict[str, Any],
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
    excluded_texts: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:

    _validate_conversation_id(
        conversation_id
    )

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
        raise ValueError(
            "invalid conversation_candidate"
        )

    if not isinstance(
        context_candidates,
        dict,
    ):
        raise ValueError(
            "invalid context_candidates"
        )

    if (
        not isinstance(
            token_budget,
            int,
        )
        or isinstance(
            token_budget,
            bool,
        )
        or token_budget < 1
    ):
        raise ValueError(
            "invalid token_budget"
        )

    conversation_sections = (
        conversation_candidate.get(
            "sections"
        )
        or {}
    )

    if not isinstance(
        conversation_sections,
        dict,
    ):
        conversation_sections = {}

    used_tokens = 0
    budget_rejected = 0
    dedup_rejected = 0

    section_tokens = {
        "current_task": 0,
        "trusted_facts": 0,
        "constraints": 0,
        "decisions": 0,
        "open_items": 0,
        "state": 0,
        "plans": 0,
        "memories": 0,
        "recent_context": 0,
    }

    section_budget_rejected = {
        key: 0
        for key in section_tokens
    }

    seen = set()
    excluded_keys = set()

    for excluded in excluded_texts:
        if (
            isinstance(excluded, str)
            and excluded.strip()
        ):
            key = _normalize_key(
                excluded
            )
            seen.add(key)
            excluded_keys.add(key)

    excluded_text_rejected = 0

    sections = {
        "current_task": None,
        "trusted_facts": [],
        "constraints": [],
        "decisions": [],
        "open_items": [],
        "state": None,
        "plans": [],
        "memories": [],
        "recent_context": [],
    }

    def admit(
        section: str,
        output: dict[str, Any],
        dedup_text: str | None,
    ) -> bool:
        nonlocal used_tokens
        nonlocal budget_rejected
        nonlocal dedup_rejected
        nonlocal excluded_text_rejected

        key = None

        if (
            isinstance(
                dedup_text,
                str,
            )
            and dedup_text.strip()
        ):
            key = _normalize_key(
                dedup_text
            )

            if key in seen:
                dedup_rejected += 1

                if key in excluded_keys:
                    excluded_text_rejected += 1

                return False

        cost = _estimate_tokens(
            output
        )

        if (
            used_tokens + cost
            > token_budget
        ):
            budget_rejected += 1
            section_budget_rejected[
                section
            ] += 1
            return False

        if section == "current_task":
            sections[
                section
            ] = output

        elif section == "state":
            sections[
                section
            ] = output

        else:
            sections[
                section
            ].append(
                output
            )

        used_tokens += cost

        section_tokens[
            section
        ] += cost

        if key is not None:
            seen.add(key)

        return True

    # ------------------------------------------------------
    # 1. Carried-forward current task
    # ------------------------------------------------------
    task = _clean_conversation_item(
        conversation_sections.get(
            "current_task"
        )
    )

    if task is not None:
        admit(
            "current_task",
            task,
            task["text"],
        )

    # ------------------------------------------------------
    # 2-5. Explicit/high-confidence conversation state
    # ------------------------------------------------------
    for section in (
        "trusted_facts",
        "constraints",
        "decisions",
        "open_items",
    ):
        raw_items = (
            conversation_sections.get(
                section
            )
            or []
        )

        if not isinstance(
            raw_items,
            list,
        ):
            continue

        for raw in raw_items:
            item = (
                _clean_conversation_item(
                    raw
                )
            )

            if item is None:
                continue

            admit(
                section,
                item,
                item["text"],
            )

    # ------------------------------------------------------
    # 6. Persistent State
    # ------------------------------------------------------
    raw_state = context_candidates.get(
        "state"
    )

    if isinstance(
        raw_state,
        dict,
    ):
        state = {}

        relationship = raw_state.get(
            "relationship"
        )

        if isinstance(
            relationship,
            dict,
        ) and relationship:
            safe_relationship = {}

            for key, value in relationship.items():
                if (
                    isinstance(
                        key,
                        str,
                    )
                    and isinstance(
                        value,
                        (int, float),
                    )
                    and not isinstance(
                        value,
                        bool,
                    )
                ):
                    safe_relationship[
                        key
                    ] = float(value)

            if safe_relationship:
                state[
                    "relationship"
                ] = safe_relationship

        current_focus = raw_state.get(
            "current_focus"
        )

        if (
            isinstance(
                current_focus,
                str,
            )
            and current_focus.strip()
        ):
            focus_key = _normalize_key(
                current_focus
            )

            if focus_key in seen:
                dedup_rejected += 1
            else:
                current_focus = (
                    current_focus.strip()
                )

                state[
                    "current_focus"
                ] = current_focus

        last_seen = raw_state.get(
            "last_seen"
        )

        if (
            isinstance(
                last_seen,
                str,
            )
            and last_seen.strip()
        ):
            state[
                "last_seen"
            ] = last_seen.strip()

        if state:
            admitted = admit(
                "state",
                state,
                None,
            )

            if (
                admitted
                and "current_focus"
                in state
            ):
                seen.add(
                    _normalize_key(
                        state[
                            "current_focus"
                        ]
                    )
                )

    # ------------------------------------------------------
    # 7. Active Plans
    # ------------------------------------------------------
    raw_plans = (
        context_candidates.get(
            "plans"
        )
        or []
    )

    if isinstance(
        raw_plans,
        list,
    ):
        plans = [
            item
            for item in raw_plans
            if isinstance(
                item,
                dict,
            )
        ]

        plans.sort(
            key=lambda item: float(
                item.get("weight")
                or 0.0
            ),
            reverse=True,
        )

        for raw in plans[
            :_MAX_PLANS
        ]:
            parsed = _plan_item(
                raw
            )

            if parsed is None:
                continue

            item, dedup_text = parsed

            admit(
                "plans",
                item,
                dedup_text,
            )

    # ------------------------------------------------------
    # 8. Relevant Memories
    # ------------------------------------------------------
    raw_memories = (
        context_candidates.get(
            "memories"
        )
        or []
    )

    if isinstance(
        raw_memories,
        list,
    ):
        memories = [
            item
            for item in raw_memories
            if isinstance(
                item,
                dict,
            )
        ]

        memories.sort(
            key=lambda item: float(
                item.get(
                    "context_relevance"
                )
                or 0.0
            ),
            reverse=True,
        )

        for raw in memories[
            :_MAX_MEMORIES
        ]:
            parsed = _memory_item(
                raw
            )

            if parsed is None:
                continue

            item, dedup_text = parsed

            admit(
                "memories",
                item,
                dedup_text,
            )

    # ------------------------------------------------------
    # 9. Recent conversation context — lowest priority.
    # ------------------------------------------------------
    recent = (
        conversation_sections.get(
            "recent_context"
        )
        or []
    )

    if isinstance(
        recent,
        list,
    ):
        for raw in recent:
            item = (
                _clean_conversation_item(
                    raw
                )
            )

            if item is None:
                continue

            admit(
                "recent_context",
                item,
                item["text"],
            )

    conversation_telemetry = (
        conversation_candidate.get(
            "telemetry"
        )
        or {}
    )

    source_telemetry = (
        context_candidates.get(
            "telemetry"
        )
        or {}
    )

    return {
        "version":
            _VERSION,
        "conversation_id":
            conversation_id,
        "created_at":
            _now(),
        "source_revisions": {
            "conversation_candidate":
                conversation_candidate.get(
                    "revision"
                ),
            "conversation_source":
                conversation_candidate.get(
                    "source_revision"
                ),
            "state":
                context_candidates.get(
                    "state_revision"
                ),
        },
        "sections":
            sections,
        "telemetry": {
            "token_budget":
                token_budget,
            "estimated_tokens":
                used_tokens,
            "truncated":
                budget_rejected > 0,
            "budget_rejected":
                budget_rejected,
            "dedup_rejected":
                dedup_rejected,
            "excluded_text_rejected":
                excluded_text_rejected,
            "section_tokens":
                section_tokens,
            "section_budget_rejected":
                section_budget_rejected,
            "current_user_excluded":
                conversation_telemetry.get(
                    "current_user_excluded"
                )
                is True,
            "has_current_task":
                sections[
                    "current_task"
                ]
                is not None,
            "trusted_fact_count":
                len(
                    sections[
                        "trusted_facts"
                    ]
                ),
            "constraint_count":
                len(
                    sections[
                        "constraints"
                    ]
                ),
            "decision_count":
                len(
                    sections[
                        "decisions"
                    ]
                ),
            "open_item_count":
                len(
                    sections[
                        "open_items"
                    ]
                ),
            "state_included":
                sections[
                    "state"
                ]
                is not None,
            "plan_count":
                len(
                    sections[
                        "plans"
                    ]
                ),
            "memory_count":
                len(
                    sections[
                        "memories"
                    ]
                ),
            "recent_context_count":
                len(
                    sections[
                        "recent_context"
                    ]
                ),
            "retrieval_candidate_count":
                source_telemetry.get(
                    "retrieval_candidate_count",
                    0,
                ),
            "relevance_rejected":
                source_telemetry.get(
                    "relevance_rejected",
                    0,
                ),
            "retrieval_quality":
                dict(
                    source_telemetry.get(
                        "retrieval_quality"
                    )
                    or {}
                ),
            "anti_echo":
                dict(
                    source_telemetry.get(
                        "anti_echo"
                    )
                    or {}
                ),
            "retrieval_dedup":
                dict(
                    source_telemetry.get(
                        "dedup"
                    )
                    or {}
                ),
        },
    }


def update_unified_context_candidate(
    *,
    conversation_id: str,
    context_candidates: dict[str, Any],
    excluded_texts: tuple[str, ...] | list[str] = (),
    shadow_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the Unified candidate and return its privacy-safe summary.

    ``shadow_observation`` is an optional out-parameter for the
    in-process shadow Memory observers: when a dict is passed, the
    full artifact built for THIS request is stored under
    ``"unified_artifact"`` so no observer ever has to re-read the
    shared conversation-level file. It is never persisted, never
    logged and never part of the live rendered Context. The default
    (None) keeps every existing caller byte-for-byte unchanged.
    """

    candidate_path = _path(
        "context_candidate",
        conversation_id,
    )

    unified_path = _path(
        "unified_context_candidate",
        conversation_id,
    )

    with _LOCK:
        conversation_candidate = (
            _read_json(
                candidate_path
            )
        )

        if conversation_candidate is None:
            return {
                "stored": False,
                "reason":
                    "conversation_candidate_not_found",
                "conversation_id":
                    conversation_id,
            }

        result = (
            build_unified_context_candidate(
                conversation_id=
                    conversation_id,
                conversation_candidate=
                    conversation_candidate,
                context_candidates=
                    context_candidates,
                excluded_texts=
                    excluded_texts,
            )
        )

        input_fingerprint = _fingerprint(
            {
                "conversation_candidate":
                    conversation_candidate,
                "context_candidates":
                    context_candidates,
                "excluded_texts":
                    list(excluded_texts),
            }
        )

        previous = _read_json(
            unified_path
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
                "input_fingerprint"
            )
            == input_fingerprint
        ):
            telemetry = (
                previous.get(
                    "telemetry"
                )
                or {}
            )

            if isinstance(
                shadow_observation,
                dict,
            ):
                shadow_observation[
                    "unified_artifact"
                ] = previous

            return {
                "stored": True,
                "duplicate": True,
                "conversation_id":
                    conversation_id,
                "revision":
                    previous.get(
                        "revision"
                    ),
                "estimated_tokens":
                    telemetry.get(
                        "estimated_tokens"
                    ),
                "token_budget":
                    telemetry.get(
                        "token_budget"
                    ),
                "truncated":
                    telemetry.get(
                        "truncated"
                    ),
                "budget_rejected":
                    telemetry.get(
                        "budget_rejected"
                    ),
                "dedup_rejected":
                    telemetry.get(
                        "dedup_rejected"
                    ),
                "excluded_text_rejected":
                    telemetry.get(
                        "excluded_text_rejected"
                    ),
                "has_current_task":
                    telemetry.get(
                        "has_current_task"
                    ),
                "trusted_fact_count":
                    telemetry.get(
                        "trusted_fact_count"
                    ),
                "constraint_count":
                    telemetry.get(
                        "constraint_count"
                    ),
                "decision_count":
                    telemetry.get(
                        "decision_count"
                    ),
                "open_item_count":
                    telemetry.get(
                        "open_item_count"
                    ),
                "state_included":
                    telemetry.get(
                        "state_included"
                    ),
                "plan_count":
                    telemetry.get(
                        "plan_count"
                    ),
                "memory_count":
                    telemetry.get(
                        "memory_count"
                    ),
                "recent_context_count":
                    telemetry.get(
                        "recent_context_count"
                    ),
                "current_user_excluded":
                    telemetry.get(
                        "current_user_excluded"
                    ),
                "retrieval_candidate_count":
                    telemetry.get(
                        "retrieval_candidate_count"
                    ),
                "relevance_rejected":
                    telemetry.get(
                        "relevance_rejected"
                    ),
                "retrieval_quality":
                    dict(
                        telemetry.get(
                            "retrieval_quality"
                        )
                        or {}
                    ),
                "anti_echo":
                    dict(
                        telemetry.get(
                            "anti_echo"
                        )
                        or {}
                    ),
                "retrieval_dedup":
                    dict(
                        telemetry.get(
                            "retrieval_dedup"
                        )
                        or {}
                    ),
            }

        previous_revision = 0

        if isinstance(
            previous,
            dict,
        ):
            value = previous.get(
                "revision"
            )

            if (
                isinstance(
                    value,
                    int,
                )
                and not isinstance(
                    value,
                    bool,
                )
            ):
                previous_revision = value

        result[
            "revision"
        ] = previous_revision + 1

        result[
            "input_fingerprint"
        ] = input_fingerprint

        _atomic_write(
            unified_path,
            result,
        )

        if isinstance(
            shadow_observation,
            dict,
        ):
            shadow_observation[
                "unified_artifact"
            ] = result

    telemetry = (
        result.get(
            "telemetry"
        )
        or {}
    )

    return {
        "stored": True,
        "duplicate": False,
        "conversation_id":
            conversation_id,
        "revision":
            result[
                "revision"
            ],
        "estimated_tokens":
            telemetry.get(
                "estimated_tokens"
            ),
        "token_budget":
            telemetry.get(
                "token_budget"
            ),
        "truncated":
            telemetry.get(
                "truncated"
            ),
        "budget_rejected":
            telemetry.get(
                "budget_rejected"
            ),
        "dedup_rejected":
            telemetry.get(
                "dedup_rejected"
            ),
        "excluded_text_rejected":
            telemetry.get(
                "excluded_text_rejected"
            ),
        "has_current_task":
            telemetry.get(
                "has_current_task"
            ),
        "trusted_fact_count":
            telemetry.get(
                "trusted_fact_count"
            ),
        "constraint_count":
            telemetry.get(
                "constraint_count"
            ),
        "decision_count":
            telemetry.get(
                "decision_count"
            ),
        "open_item_count":
            telemetry.get(
                "open_item_count"
            ),
        "state_included":
            telemetry.get(
                "state_included"
            ),
        "plan_count":
            telemetry.get(
                "plan_count"
            ),
        "memory_count":
            telemetry.get(
                "memory_count"
            ),
        "recent_context_count":
            telemetry.get(
                "recent_context_count"
            ),
        "current_user_excluded":
            telemetry.get(
                "current_user_excluded"
            ),
        "retrieval_candidate_count":
            telemetry.get(
                "retrieval_candidate_count"
            ),
        "relevance_rejected":
            telemetry.get(
                "relevance_rejected"
            ),
        "retrieval_quality":
            dict(
                telemetry.get(
                    "retrieval_quality"
                )
                or {}
            ),
        "anti_echo":
            dict(
                telemetry.get(
                    "anti_echo"
                )
                or {}
            ),
        "retrieval_dedup":
            dict(
                telemetry.get(
                    "retrieval_dedup"
                )
                or {}
            ),
    }


def build_memory_shadow_snapshot(
    *,
    conversation_id: str,
    unified: Any,
    retrieval_memories: Any,
) -> dict[str, Any] | None:
    """Request-local, shadow-only Memory observation snapshot.

    It carries the Unified artifact THIS request just built plus the
    per-candidate conservative.v1 evidence derived from the same
    request's retrieval candidate set. It exists only so the shadow
    Memory Surfacing / Flash / Exposure Ledger observers can consume
    this request's own evidence instead of re-reading the shared,
    conversation-level Unified file (which a concurrent request may
    already have overwritten).

    It is never persisted, never logged, never added to Unified
    sections, Preview, Real Injection, a header or a cache key, and
    it never changes retrieval candidate selection. Returns None when
    no valid Unified artifact was produced for this request.
    """

    if (
        not isinstance(unified, dict)
        or unified.get("version") != _VERSION
        or unified.get("conversation_id")
        != conversation_id
    ):
        return None

    memories = (
        retrieval_memories
        if isinstance(
            retrieval_memories,
            list,
        )
        else []
    )

    return {
        "version":
            "memory-shadow-snapshot.v1",
        "mode":
            "shadow_only",
        "conversation_id":
            conversation_id,
        "revision":
            unified.get("revision"),
        "unified":
            unified,
        "evidence":
            build_candidate_evidence(
                memories
            ),
    }


def unified_context_candidate_status(
    conversation_id: str,
) -> dict[str, Any]:

    state = _read_json(
        _path(
            "unified_context_candidate",
            conversation_id,
        )
    )

    if state is None:
        return {
            "exists": False,
            "conversation_id":
                conversation_id,
        }

    telemetry = (
        state.get(
            "telemetry"
        )
        or {}
    )

    return {
        "exists": True,
        "conversation_id":
            conversation_id,
        "version":
            state.get(
                "version"
            ),
        "revision":
            state.get(
                "revision"
            ),
        "source_revisions":
            dict(
                state.get(
                    "source_revisions"
                )
                or {}
            ),
        "estimated_tokens":
            telemetry.get(
                "estimated_tokens"
            ),
        "token_budget":
            telemetry.get(
                "token_budget"
            ),
        "truncated":
            telemetry.get(
                "truncated"
            ),
        "has_current_task":
            telemetry.get(
                "has_current_task"
            ),
        "trusted_fact_count":
            telemetry.get(
                "trusted_fact_count"
            ),
        "constraint_count":
            telemetry.get(
                "constraint_count"
            ),
        "decision_count":
            telemetry.get(
                "decision_count"
            ),
        "open_item_count":
            telemetry.get(
                "open_item_count"
            ),
        "state_included":
            telemetry.get(
                "state_included"
            ),
        "plan_count":
            telemetry.get(
                "plan_count"
            ),
        "memory_count":
            telemetry.get(
                "memory_count"
            ),
        "recent_context_count":
            telemetry.get(
                "recent_context_count"
            ),
    }
