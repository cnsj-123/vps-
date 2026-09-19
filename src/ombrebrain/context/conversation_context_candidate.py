from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()

_VERSION = "conversation-context-candidate.v1"
_DEFAULT_ROOT = "/app/buckets/.context"

_DEFAULT_TOKEN_BUDGET = 900

_MAX_FACTS = 8
_MAX_CONSTRAINTS = 8
_MAX_DECISIONS = 6
_MAX_OPEN_ITEMS = 6
_MAX_RECENT_MESSAGES = 4

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_SPACE_RE = re.compile(r"\s+")

_CJK_CONTROL_SPACE_RE = re.compile(
    r"(?<=[\u4e00-\u9fff：:])"
    r"\s+"
    r"(?=[\u4e00-\u9fff：:])"
)

_CONTROL_RE = re.compile(
    r"^\s*(?:"
    r"(?:事实|确认事实|撤销事实|删除事实|取消事实|"
    r"替换事实|更新事实)|"
    r"(?:取消|关闭|移除)约束|"
    r"(?:撤销|取消)决定|"
    r"(?:完成|关闭|解决)待办|"
    r"(?:替换|更新)(?:约束|决定)"
    r")\s*[:：]",
    re.IGNORECASE,
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
        path.parent.chmod(
            0o700
        )
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
        temp.chmod(
            0o600
        )
    except OSError:
        pass

    os.replace(
        str(temp),
        str(path),
    )


def _estimate_tokens(
    text: str,
) -> int:
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


def _normalize_control_text(
    text: str,
) -> str:
    text = _CJK_CONTROL_SPACE_RE.sub(
        "",
        text.strip(),
    )

    return _SPACE_RE.sub(
        " ",
        text,
    )


def _is_control_text(
    text: str,
) -> bool:
    for line in text.splitlines():
        if _CONTROL_RE.match(
            _normalize_control_text(
                line
            )
        ):
            return True

    return False


def _source_index(
    item: Any,
) -> int | None:
    if not isinstance(
        item,
        dict,
    ):
        return None

    value = item.get(
        "source_index"
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
        return value

    return None


def _text_item(
    item: Any,
) -> dict[str, Any] | None:
    if not isinstance(
        item,
        dict,
    ):
        return None

    text = item.get(
        "text"
    )

    if (
        not isinstance(
            text,
            str,
        )
        or not text.strip()
    ):
        return None

    return {
        "text":
            text.strip(),
        "source_index":
            _source_index(
                item
            ),
    }


def _recent_messages(
    compact: dict[str, Any],
) -> list[dict[str, Any]]:
    raw = compact.get(
        "recent_messages"
    )

    if not isinstance(
        raw,
        list,
    ):
        return []

    result = []

    for item in raw:
        if not isinstance(
            item,
            dict,
        ):
            continue

        role = item.get(
            "role"
        )

        text = item.get(
            "text"
        )

        if (
            role not in (
                "user",
                "assistant",
            )
            or not isinstance(
                text,
                str,
            )
            or not text.strip()
        ):
            continue

        result.append(
            {
                "role":
                    role,
                "text":
                    text.strip(),
                "source_index":
                    _source_index(
                        item
                    ),
            }
        )

    return result


def _latest_user_source_index(
    messages: list[dict[str, Any]],
) -> int | None:
    for message in reversed(
        messages
    ):
        if message.get(
            "role"
        ) != "user":
            continue

        value = message.get(
            "source_index"
        )

        if isinstance(
            value,
            int,
        ):
            return value

    return None


def _safe_source(
    payload: dict[str, Any] | None,
    *,
    compact_revision: int,
) -> tuple[
    dict[str, Any] | None,
    bool,
    bool,
]:
    if not isinstance(
        payload,
        dict,
    ):
        return None, False, False

    revision = payload.get(
        "source_revision"
    )

    if (
        not isinstance(
            revision,
            int,
        )
        or isinstance(
            revision,
            bool,
        )
    ):
        return None, False, True

    if revision > compact_revision:
        return None, True, False

    if revision < compact_revision:
        return None, False, True

    return payload, False, False


def build_context_candidate(
    *,
    conversation_id: str,
    compact: dict[str, Any],
    semantic_state: dict[str, Any] | None,
    trusted_facts: dict[str, Any] | None,
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
) -> dict[str, Any]:

    _validate_conversation_id(
        conversation_id
    )

    compact_revision = compact.get(
        "source_revision"
    )

    if (
        not isinstance(
            compact_revision,
            int,
        )
        or isinstance(
            compact_revision,
            bool,
        )
        or compact_revision < 1
    ):
        raise ValueError(
            "invalid compact source_revision"
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

    (
        semantic_state,
        semantic_source_ahead,
        semantic_source_stale,
    ) = _safe_source(
        semantic_state,
        compact_revision=
            compact_revision,
    )

    (
        trusted_facts,
        facts_source_ahead,
        facts_source_stale,
    ) = _safe_source(
        trusted_facts,
        compact_revision=
            compact_revision,
    )

    messages = _recent_messages(
        compact
    )

    latest_user_source = (
        _latest_user_source_index(
            messages
        )
    )

    used_tokens = 0
    budget_rejected = 0
    dedup_rejected = 0
    control_rejected = 0

    seen = set()

    sections = {
        "current_task": None,
        "trusted_facts": [],
        "constraints": [],
        "decisions": [],
        "open_items": [],
        "recent_context": [],
    }

    def add_text(
        section: str,
        item: dict[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        nonlocal used_tokens
        nonlocal budget_rejected
        nonlocal dedup_rejected

        text = item.get(
            "text"
        )

        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
        ):
            return False

        text = text.strip()
        key = _normalize_key(
            text
        )

        if key in seen:
            dedup_rejected += 1
            return False

        cost = (
            _estimate_tokens(
                text
            )
            + 2
        )

        if (
            used_tokens + cost
            > token_budget
        ):
            budget_rejected += 1
            return False

        output = {
            "text":
                text,
            "source_index":
                item.get(
                    "source_index"
                ),
        }

        if extra:
            output.update(
                extra
            )

        if section == "current_task":
            sections[
                "current_task"
            ] = output
        else:
            sections[
                section
            ].append(
                output
            )

        seen.add(
            key
        )

        used_tokens += cost
        return True

    current_task_duplicate_excluded = False

    if semantic_state is not None:
        task = _text_item(
            semantic_state.get(
                "current_task"
            )
        )

        if task is not None:
            task_source = task.get(
                "source_index"
            )

            if (
                latest_user_source
                is not None
                and task_source
                == latest_user_source
            ):
                current_task_duplicate_excluded = True
            else:
                add_text(
                    "current_task",
                    task,
                )

    if trusted_facts is not None:
        facts = trusted_facts.get(
            "active_facts"
        )

        if isinstance(
            facts,
            list,
        ):
            candidates = [
                item
                for item in facts
                if isinstance(
                    item,
                    dict,
                )
                and item.get(
                    "status",
                    "active",
                )
                == "active"
            ]

            candidates.sort(
                key=lambda x: (
                    x.get(
                        "last_source_index"
                    )
                    if isinstance(
                        x.get(
                            "last_source_index"
                        ),
                        int,
                    )
                    else -1
                ),
                reverse=True,
            )

            for raw in candidates[
                :_MAX_FACTS
            ]:
                item = _text_item(
                    raw
                )

                if item is None:
                    continue

                add_text(
                    "trusted_facts",
                    item,
                    extra={
                        "id":
                            raw.get("id"),
                        "provenance":
                            raw.get(
                                "provenance"
                            ),
                    },
                )

    semantic_specs = (
        (
            "constraints",
            _MAX_CONSTRAINTS,
        ),
        (
            "decisions",
            _MAX_DECISIONS,
        ),
        (
            "open_items",
            _MAX_OPEN_ITEMS,
        ),
    )

    if semantic_state is not None:
        for section, limit in semantic_specs:
            raw_items = semantic_state.get(
                section
            )

            if not isinstance(
                raw_items,
                list,
            ):
                continue

            candidates = [
                item
                for item in raw_items
                if isinstance(
                    item,
                    dict,
                )
                and item.get(
                    "status",
                    "active",
                )
                == "active"
            ]

            candidates.sort(
                key=lambda x: (
                    x.get(
                        "last_source_index"
                    )
                    if isinstance(
                        x.get(
                            "last_source_index"
                        ),
                        int,
                    )
                    else -1
                ),
                reverse=True,
            )

            for raw in candidates[
                :limit
            ]:
                item = _text_item(
                    raw
                )

                if item is None:
                    continue

                add_text(
                    section,
                    item,
                    extra={
                        "id":
                            raw.get("id"),
                    },
                )

    recent_candidates = []

    for message in messages:
        source_index = message.get(
            "source_index"
        )

        if (
            latest_user_source
            is not None
            and message.get(
                "role"
            )
            == "user"
            and source_index
            == latest_user_source
        ):
            continue

        if _is_control_text(
            message["text"]
        ):
            control_rejected += 1
            continue

        recent_candidates.append(
            message
        )

    selected_recent = []

    for message in reversed(
        recent_candidates
    ):
        if (
            len(selected_recent)
            >= _MAX_RECENT_MESSAGES
        ):
            break

        before = len(
            sections[
                "recent_context"
            ]
        )

        added = add_text(
            "recent_context",
            message,
            extra={
                "role":
                    message.get(
                        "role"
                    ),
            },
        )

        if added:
            selected_recent.append(
                sections[
                    "recent_context"
                ].pop()
            )

    selected_recent.reverse()

    sections[
        "recent_context"
    ] = selected_recent

    return {
        "version":
            _VERSION,
        "conversation_id":
            conversation_id,
        "source_revision":
            compact_revision,
        "created_at":
            _now(),
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
            "control_rejected":
                control_rejected,
            "latest_user_source_index":
                latest_user_source,
            "current_user_excluded":
                latest_user_source
                is not None,
            "current_task_current_user_duplicate_excluded":
                current_task_duplicate_excluded,
            "semantic_source_ahead":
                semantic_source_ahead,
            "semantic_source_stale":
                semantic_source_stale,
            "trusted_facts_source_ahead":
                facts_source_ahead,
            "trusted_facts_source_stale":
                facts_source_stale,
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
            "recent_context_count":
                len(
                    sections[
                        "recent_context"
                    ]
                ),
            "has_current_task":
                sections[
                    "current_task"
                ]
                is not None,
        },
    }


def update_conversation_context_candidate(
    conversation_id: str,
) -> dict[str, Any]:

    compact_path = _path(
        "compact",
        conversation_id,
    )

    semantic_path = _path(
        "semantic_state",
        conversation_id,
    )

    facts_path = _path(
        "trusted_facts",
        conversation_id,
    )

    candidate_path = _path(
        "context_candidate",
        conversation_id,
    )

    with _LOCK:
        compact = _read_json(
            compact_path
        )

        if compact is None:
            return {
                "stored": False,
                "reason":
                    "compact_not_found",
                "conversation_id":
                    conversation_id,
            }

        source_revision = compact.get(
            "source_revision"
        )

        if (
            not isinstance(
                source_revision,
                int,
            )
            or isinstance(
                source_revision,
                bool,
            )
            or source_revision < 1
        ):
            return {
                "stored": False,
                "reason":
                    "invalid_source_revision",
                "conversation_id":
                    conversation_id,
            }

        semantic_state = _read_json(
            semantic_path
        )

        trusted_facts = _read_json(
            facts_path
        )

        previous = _read_json(
            candidate_path
        )

        semantic_revision = (
            semantic_state.get(
                "source_revision"
            )
            if isinstance(
                semantic_state,
                dict,
            )
            else None
        )

        facts_revision = (
            trusted_facts.get(
                "source_revision"
            )
            if isinstance(
                trusted_facts,
                dict,
            )
            else None
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
                "source_revision"
            )
            == source_revision
            and (
                previous.get(
                    "source_revisions"
                )
                or {}
            ).get(
                "semantic_state"
            )
            == semantic_revision
            and (
                previous.get(
                    "source_revisions"
                )
                or {}
            ).get(
                "trusted_facts"
            )
            == facts_revision
        ):
            telemetry = previous.get(
                "telemetry"
            )

            if not isinstance(
                telemetry,
                dict,
            ):
                telemetry = {}

            return {
                "stored": True,
                "duplicate": True,
                "conversation_id":
                    conversation_id,
                "revision":
                    previous.get(
                        "revision",
                        0,
                    ),
                "source_revision":
                    source_revision,
                "estimated_tokens":
                    telemetry.get(
                        "estimated_tokens"
                    ),
                "truncated":
                    telemetry.get(
                        "truncated"
                    ),
            }

        candidate = build_context_candidate(
            conversation_id=
                conversation_id,
            compact=compact,
            semantic_state=
                semantic_state,
            trusted_facts=
                trusted_facts,
        )

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
                previous_revision = (
                    value
                )

        candidate[
            "revision"
        ] = previous_revision + 1

        candidate[
            "source_revisions"
        ] = {
            "compact":
                source_revision,
            "semantic_state":
                semantic_revision,
            "trusted_facts":
                facts_revision,
        }

        _atomic_write(
            candidate_path,
            candidate,
        )

    telemetry = candidate.get(
        "telemetry"
    )

    if not isinstance(
        telemetry,
        dict,
    ):
        telemetry = {}

    return {
        "stored": True,
        "duplicate": False,
        "conversation_id":
            conversation_id,
        "revision":
            candidate["revision"],
        "source_revision":
            source_revision,
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
        "control_rejected":
            telemetry.get(
                "control_rejected"
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
        "recent_context_count":
            telemetry.get(
                "recent_context_count"
            ),
        "current_user_excluded":
            telemetry.get(
                "current_user_excluded"
            ),
    }


def context_candidate_status(
    conversation_id: str,
) -> dict[str, Any]:

    state = _read_json(
        _path(
            "context_candidate",
            conversation_id,
        )
    )

    if state is None:
        return {
            "exists": False,
            "conversation_id":
                conversation_id,
        }

    telemetry = state.get(
        "telemetry"
    )

    if not isinstance(
        telemetry,
        dict,
    ):
        telemetry = {}

    sections = state.get(
        "sections"
    )

    if not isinstance(
        sections,
        dict,
    ):
        sections = {}

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
        "source_revision":
            state.get(
                "source_revision"
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
            sections.get(
                "current_task"
            )
            is not None,
        "trusted_fact_count":
            len(
                sections.get(
                    "trusted_facts"
                )
                or []
            ),
        "constraint_count":
            len(
                sections.get(
                    "constraints"
                )
                or []
            ),
        "decision_count":
            len(
                sections.get(
                    "decisions"
                )
                or []
            ),
        "open_item_count":
            len(
                sections.get(
                    "open_items"
                )
                or []
            ),
        "recent_context_count":
            len(
                sections.get(
                    "recent_context"
                )
                or []
            ),
    }
