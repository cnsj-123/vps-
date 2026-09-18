from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()

_VERSION = "conversation-semantic.v3"
_DEFAULT_ROOT = "/app/buckets/.context"

_MAX_CURRENT_TASK_CHARS = 600
_MAX_ITEM_CHARS = 320
_MAX_ITEMS = 8

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[。！？!?])\s*|\n+"
)

_CONSTRAINT_RE = re.compile(
    r"(?:"
    r"必须|不要|不能|不允许|只能|只保留|只允许|"
    r"要求|保持|务必|禁止|"
    r"\bmust\b|\bdo\s+not\b|\bdon't\b|"
    r"\bshould\s+not\b|\bonly\b|\brequire(?:ment)?\b"
    r")",
    re.IGNORECASE,
)

_DECISION_RE = re.compile(
    r"(?:"
    r"决定|确定|确认|采用|选择|改成|定为|结论|"
    r"\bdecided?\b|\bconfirmed?\b|\bchoose\b|\bchosen\b"
    r")",
    re.IGNORECASE,
)

_OPEN_RE = re.compile(
    r"(?:"
    r"[？?]|下一步|后续|待办|还需要|需要继续|继续做|"
    r"\bTODO\b|\bnext\s+step\b|\bremaining\b|\bopen\s+item\b"
    r")",
    re.IGNORECASE,
)


_ACK_ONLY_RE = re.compile(
    r"^\s*(?:"
    r"好|好的|好了|可以|行|"
    r"嗯+|哦+|收到|明白|知道了|"
    r"继续|继续吧|来|开始|开始吧|欧克克|"
    r"ok(?:ay)?|yes|go|done|sure"
    r")[\s。！!,.，]*$",
    re.IGNORECASE,
)


def _is_ack_or_continuation(
    text: str,
) -> bool:
    if not isinstance(
        text,
        str,
    ):
        return False

    return bool(
        _ACK_ONLY_RE.fullmatch(
            text.strip()
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


def _compact_path(
    conversation_id: str,
) -> Path:
    _validate_conversation_id(
        conversation_id
    )

    return (
        _root()
        / "compact"
        / (conversation_id + ".json")
    )


def _semantic_path(
    conversation_id: str,
) -> Path:
    _validate_conversation_id(
        conversation_id
    )

    return (
        _root()
        / "semantic"
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
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    if not isinstance(
        payload,
        dict,
    ):
        return None

    return payload


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


def _clip(
    text: str,
    limit: int,
) -> str:
    text = text.strip()

    if len(text) <= limit:
        return text

    if limit <= 4:
        return text[:limit]

    return (
        text[: limit - 3]
        + "..."
    )


def _messages_from_compact(
    compact: dict[str, Any],
) -> list[dict[str, Any]]:
    messages: list[
        dict[str, Any]
    ] = []

    older = compact.get(
        "older_context"
    )

    if isinstance(
        older,
        list,
    ):
        for item in older:
            if not isinstance(
                item,
                dict,
            ):
                continue

            role = item.get("role")
            text = item.get(
                "excerpt"
            )

            if (
                role in (
                    "user",
                    "assistant",
                )
                and isinstance(
                    text,
                    str,
                )
                and text.strip()
            ):
                messages.append(
                    {
                        "role": role,
                        "text": text.strip(),
                        "source_index":
                            item.get(
                                "source_index"
                            ),
                    }
                )

    recent = compact.get(
        "recent_messages"
    )

    if isinstance(
        recent,
        list,
    ):
        for item in recent:
            if not isinstance(
                item,
                dict,
            ):
                continue

            role = item.get("role")
            text = item.get(
                "text"
            )

            if (
                role in (
                    "user",
                    "assistant",
                )
                and isinstance(
                    text,
                    str,
                )
                and text.strip()
            ):
                messages.append(
                    {
                        "role": role,
                        "text": text.strip(),
                        "source_index":
                            item.get(
                                "source_index"
                            ),
                    }
                )

    return messages


def _sentences(
    text: str,
) -> list[str]:
    return [
        item.strip()
        for item in _SENTENCE_SPLIT_RE.split(
            text
        )
        if item.strip()
    ]


def _collect(
    messages: list[dict[str, Any]],
    *,
    role: str,
    pattern: re.Pattern[str],
) -> list[dict[str, Any]]:
    found: list[
        dict[str, Any]
    ] = []

    seen: set[str] = set()

    for message in messages:
        if message.get(
            "role"
        ) != role:
            continue

        text = message.get(
            "text"
        )

        if not isinstance(
            text,
            str,
        ):
            continue

        for sentence in _sentences(
            text
        ):
            if not pattern.search(
                sentence
            ):
                continue

            item_text = _clip(
                sentence,
                _MAX_ITEM_CHARS,
            )

            key = item_text.strip()

            if key in seen:
                continue

            seen.add(
                key
            )

            found.append(
                {
                    "text": item_text,
                    "source_index":
                        message.get(
                            "source_index"
                        ),
                }
            )

    return found[
        -_MAX_ITEMS:
    ]


def build_semantic_frame(
    compact: dict[str, Any],
    previous_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:

    conversation_id = compact.get(
        "conversation_id"
    )

    _validate_conversation_id(
        conversation_id
    )

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
        raise ValueError(
            "invalid compact source_revision"
        )

    messages = _messages_from_compact(
        compact
    )

    previous_task = None

    if isinstance(
        previous_frame,
        dict,
    ):
        candidate = previous_frame.get(
            "current_task"
        )

        if (
            isinstance(
                candidate,
                dict,
            )
            and isinstance(
                candidate.get("text"),
                str,
            )
            and candidate.get(
                "text"
            ).strip()
            and not _is_ack_or_continuation(
                candidate["text"]
            )
        ):
            previous_task = {
                "text": candidate[
                    "text"
                ],
                "source_index":
                    candidate.get(
                        "source_index"
                    ),
            }

    user_messages = [
        message
        for message in messages
        if message.get("role")
        == "user"
    ]

    latest_user = (
        user_messages[-1]
        if user_messages
        else None
    )

    current_task = None
    task_carried_forward = False
    latest_user_ack_only = False
    task_recovered_from_history = False

    if latest_user is not None:
        latest_text = latest_user[
            "text"
        ]

        latest_user_ack_only = (
            _is_ack_or_continuation(
                latest_text
            )
        )

        if latest_user_ack_only:
            # Find the newest substantive task still visible in
            # the bounded Compact window.
            history_task = None

            for candidate_message in reversed(
                user_messages[:-1]
            ):
                candidate_text = (
                    candidate_message.get(
                        "text"
                    )
                )

                if (
                    isinstance(
                        candidate_text,
                        str,
                    )
                    and candidate_text.strip()
                    and not _is_ack_or_continuation(
                        candidate_text
                    )
                ):
                    history_task = {
                        "text": _clip(
                            candidate_text,
                            _MAX_CURRENT_TASK_CHARS,
                        ),
                        "source_index":
                            candidate_message.get(
                                "source_index"
                            ),
                    }
                    break

            # Prefer whichever sourced task is newer.
            candidates = [
                item
                for item in (
                    previous_task,
                    history_task,
                )
                if isinstance(
                    item,
                    dict,
                )
            ]

            if candidates:
                def _source_rank(item):
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

                    return -1

                current_task = dict(
                    max(
                        candidates,
                        key=_source_rank,
                    )
                )

                task_carried_forward = True

                if (
                    history_task
                    is not None
                    and current_task.get(
                        "source_index"
                    )
                    == history_task.get(
                        "source_index"
                    )
                    and (
                        previous_task is None
                        or previous_task.get(
                            "source_index"
                        )
                        != history_task.get(
                            "source_index"
                        )
                    )
                ):
                    task_recovered_from_history = True

        else:
            current_task = {
                "text": _clip(
                    latest_text,
                    _MAX_CURRENT_TASK_CHARS,
                ),
                "source_index":
                    latest_user.get(
                        "source_index"
                    ),
            }

    elif previous_task is not None:
        current_task = dict(
            previous_task
        )

        task_carried_forward = True

    constraints = _collect(
        messages,
        role="user",
        pattern=_CONSTRAINT_RE,
    )

    decisions = _collect(
        messages,
        role="assistant",
        pattern=_DECISION_RE,
    )

    open_items = _collect(
        messages,
        role="user",
        pattern=_OPEN_RE,
    )

    return {
        "version": _VERSION,
        "conversation_id":
            conversation_id,
        "source_revision":
            source_revision,
        "created_at": _now(),

        "current_task":
            current_task,

        "decisions":
            decisions,

        "constraints":
            constraints,

        "open_items":
            open_items,

        # Intentionally deferred in deterministic v1.
        # Facts require semantic interpretation and
        # should not be guessed with keyword rules.
        "established_facts": [],

        "telemetry": {
            "source_messages":
                len(messages),
            "has_current_task":
                current_task is not None,
            "current_task_carried_forward":
                task_carried_forward,
            "latest_user_ack_only":
                latest_user_ack_only,
            "task_recovered_from_history":
                task_recovered_from_history,
            "decision_count":
                len(decisions),
            "constraint_count":
                len(constraints),
            "open_item_count":
                len(open_items),
            "fact_count": 0,
            "facts_deferred":
                True,
            "extractor":
                "deterministic-conservative",
        },
    }


def update_conversation_semantic(
    conversation_id: str,
) -> dict[str, Any]:

    source_path = _compact_path(
        conversation_id
    )

    target_path = _semantic_path(
        conversation_id
    )

    with _LOCK:
        compact = _read_json(
            source_path
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

        previous = _read_json(
            target_path
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
                "source_revision":
                    source_revision,
                **telemetry,
            }

        semantic = build_semantic_frame(
            compact,
            previous_frame=(
                previous
                if isinstance(
                    previous,
                    dict,
                )
                else None
            ),
        )

        _atomic_write(
            target_path,
            semantic,
        )

    telemetry = semantic[
        "telemetry"
    ]

    return {
        "stored": True,
        "duplicate": False,
        "conversation_id":
            conversation_id,
        "source_revision":
            semantic[
                "source_revision"
            ],
        **telemetry,
    }


def semantic_status(
    conversation_id: str,
) -> dict[str, Any]:

    path = _semantic_path(
        conversation_id
    )

    with _LOCK:
        payload = _read_json(
            path
        )

    if payload is None:
        return {
            "exists": False,
            "conversation_id":
                conversation_id,
        }

    telemetry = payload.get(
        "telemetry"
    )

    if not isinstance(
        telemetry,
        dict,
    ):
        telemetry = {}

    # Never expose semantic text via status.
    return {
        "exists": True,
        "conversation_id":
            conversation_id,
        "version":
            payload.get(
                "version"
            ),
        "source_revision":
            payload.get(
                "source_revision"
            ),
        "created_at":
            payload.get(
                "created_at"
            ),
        "telemetry":
            telemetry,
    }
