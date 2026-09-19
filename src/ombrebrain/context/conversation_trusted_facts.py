from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()

_VERSION = "conversation-trusted-facts.v1"
_DEFAULT_ROOT = "/app/buckets/.context"

_MAX_ACTIVE_FACTS = 16
_MAX_HISTORY = 32
_MAX_FACT_CHARS = 500

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_SPACE_RE = re.compile(r"\s+")

_CJK_SPACE_RE = re.compile(
    r"(?<=[\u4e00-\u9fff：:])"
    r"\s+"
    r"(?=[\u4e00-\u9fff：:])"
)

_ASSERT_RE = re.compile(
    r"^\s*(?:事实|确认事实)\s*[:：]\s*"
    r"(?P<fact>.+?)\s*$",
    re.IGNORECASE,
)

_REVOKE_RE = re.compile(
    r"^\s*(?:撤销事实|删除事实|取消事实)\s*[:：]\s*"
    r"(?P<fact>.+?)\s*$",
    re.IGNORECASE,
)

_REPLACE_RE = re.compile(
    r"^\s*(?:替换事实|更新事实)\s*[:：]\s*"
    r"(?P<target>.+?)\s*"
    r"(?:=>|->|→)\s*"
    r"(?P<replacement>.+?)\s*$",
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


def _facts_path(
    conversation_id: str,
) -> Path:
    _validate_conversation_id(
        conversation_id
    )

    return (
        _root()
        / "trusted_facts"
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


def _normalize_command_text(
    text: str,
) -> str:
    text = text.strip()

    text = _CJK_SPACE_RE.sub(
        "",
        text,
    )

    return _SPACE_RE.sub(
        " ",
        text,
    )


def _clean_fact_text(
    text: str,
) -> str:
    text = _normalize_command_text(
        text
    )

    text = text.rstrip(
        "。！？!?"
    ).strip()

    if len(text) > _MAX_FACT_CHARS:
        text = text[
            :_MAX_FACT_CHARS
        ]

    return text


def _fact_key(
    text: str,
) -> str:
    return (
        _clean_fact_text(
            text
        )
        .casefold()
    )


def _fact_id(
    text: str,
) -> str:
    raw = _fact_key(
        text
    ).encode(
        "utf-8"
    )

    return (
        "fact_"
        + hashlib.sha256(
            raw
        ).hexdigest()[:16]
    )


def _segments(
    text: str,
) -> list[str]:
    # Deliberately conservative:
    # only whole lines are commands.
    return [
        item.strip()
        for item in text.splitlines()
        if item.strip()
    ]


def _parse_command(
    text: str,
) -> dict[str, Any] | None:
    normalized = (
        _normalize_command_text(
            text
        )
    )

    match = _REPLACE_RE.fullmatch(
        normalized
    )

    if match is not None:
        target = _clean_fact_text(
            match.group(
                "target"
            )
        )

        replacement = (
            _clean_fact_text(
                match.group(
                    "replacement"
                )
            )
        )

        if (
            target
            and replacement
        ):
            return {
                "action":
                    "replace",
                "target":
                    target,
                "replacement":
                    replacement,
            }

        return None

    match = _REVOKE_RE.fullmatch(
        normalized
    )

    if match is not None:
        fact = _clean_fact_text(
            match.group(
                "fact"
            )
        )

        if fact:
            return {
                "action":
                    "revoke",
                "target":
                    fact,
            }

        return None

    match = _ASSERT_RE.fullmatch(
        normalized
    )

    if match is not None:
        fact = _clean_fact_text(
            match.group(
                "fact"
            )
        )

        if fact:
            return {
                "action":
                    "assert",
                "fact":
                    fact,
            }

    return None


def _empty_state(
    conversation_id: str,
) -> dict[str, Any]:
    return {
        "version":
            _VERSION,
        "conversation_id":
            conversation_id,
        "revision": 0,
        "source_revision": 0,
        "last_processed_source_index":
            -1,
        "updated_at": None,
        "active_facts": [],
        "history": [],
        "telemetry": {
            "mode":
                "explicit-user-only",
            "inference_enabled":
                False,
            "history_dropped_total":
                0,
        },
    }


def _recent_user_messages(
    compact: dict[str, Any],
) -> list[dict[str, Any]]:
    recent = compact.get(
        "recent_messages"
    )

    if not isinstance(
        recent,
        list,
    ):
        return []

    result = []

    for item in recent:
        if not isinstance(
            item,
            dict,
        ):
            continue

        if item.get(
            "role"
        ) != "user":
            continue

        text = item.get(
            "text"
        )

        source_index = item.get(
            "source_index"
        )

        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
            or not isinstance(
                source_index,
                int,
            )
            or isinstance(
                source_index,
                bool,
            )
        ):
            continue

        result.append(
            {
                "text":
                    text.strip(),
                "source_index":
                    source_index,
            }
        )

    result.sort(
        key=lambda x:
            x["source_index"]
    )

    return result


def _find_active(
    active: list[dict[str, Any]],
    text: str,
) -> int | None:
    key = _fact_key(
        text
    )

    for index, item in enumerate(
        active
    ):
        candidate = item.get(
            "text"
        )

        if (
            isinstance(
                candidate,
                str,
            )
            and _fact_key(
                candidate
            )
            == key
        ):
            return index

    return None


def _assert_fact(
    *,
    active: list[dict[str, Any]],
    fact: str,
    source_index: int,
    source_revision: int,
) -> tuple[int, int]:
    index = _find_active(
        active,
        fact,
    )

    if index is not None:
        item = active[index]

        item[
            "last_source_index"
        ] = source_index

        item[
            "last_seen_revision"
        ] = source_revision

        item["confirmations"] = (
            int(
                item.get(
                    "confirmations",
                    1,
                )
            )
            + 1
        )

        return 0, 1

    active.append(
        {
            "id":
                _fact_id(
                    fact
                ),
            "status":
                "active",
            "text":
                fact,
            "provenance":
                "user_explicit",
            "first_source_index":
                source_index,
            "last_source_index":
                source_index,
            "first_seen_revision":
                source_revision,
            "last_seen_revision":
                source_revision,
            "confirmations":
                1,
        }
    )

    return 1, 0


def _revoke_fact(
    *,
    active: list[dict[str, Any]],
    history: list[dict[str, Any]],
    target: str,
    source_index: int,
    source_revision: int,
    status: str,
    replacement: str | None = None,
) -> bool:
    index = _find_active(
        active,
        target,
    )

    if index is None:
        return False

    item = active.pop(
        index
    )

    closed = dict(
        item
    )

    closed["status"] = status
    closed[
        "closed_revision"
    ] = source_revision
    closed[
        "close_source_index"
    ] = source_index

    if replacement is not None:
        closed[
            "superseded_by_id"
        ] = _fact_id(
            replacement
        )

    history.append(
        closed
    )

    return True


def update_conversation_trusted_facts(
    conversation_id: str,
) -> dict[str, Any]:

    source_path = _compact_path(
        conversation_id
    )

    target_path = _facts_path(
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

        state = _read_json(
            target_path
        )

        if (
            not isinstance(
                state,
                dict,
            )
            or state.get(
                "version"
            )
            != _VERSION
        ):
            state = _empty_state(
                conversation_id
            )

        previous_source_revision = (
            state.get(
                "source_revision",
                0,
            )
        )

        if (
            isinstance(
                previous_source_revision,
                int,
            )
            and source_revision
            < previous_source_revision
        ):
            return {
                "stored": False,
                "reason":
                    "stale_source_revision",
                "conversation_id":
                    conversation_id,
            }

        if (
            source_revision
            == previous_source_revision
        ):
            return {
                "stored": True,
                "duplicate": True,
                "conversation_id":
                    conversation_id,
                "revision":
                    state.get(
                        "revision",
                        0,
                    ),
                "source_revision":
                    source_revision,
                "fact_count":
                    len(
                        state.get(
                            "active_facts"
                        )
                        or []
                    ),
                "history_count":
                    len(
                        state.get(
                            "history"
                        )
                        or []
                    ),
            }

        active = [
            dict(item)
            for item in (
                state.get(
                    "active_facts"
                )
                or []
            )
            if isinstance(
                item,
                dict,
            )
        ]

        history = [
            dict(item)
            for item in (
                state.get(
                    "history"
                )
                or []
            )
            if isinstance(
                item,
                dict,
            )
        ]

        last_processed = state.get(
            "last_processed_source_index",
            -1,
        )

        if (
            not isinstance(
                last_processed,
                int,
            )
            or isinstance(
                last_processed,
                bool,
            )
        ):
            last_processed = -1

        added = 0
        confirmed = 0
        revoked = 0
        replaced = 0
        unmatched = 0
        commands = 0

        max_seen = last_processed

        for message in _recent_user_messages(
            compact
        ):
            source_index = message[
                "source_index"
            ]

            if source_index <= last_processed:
                continue

            if source_index > max_seen:
                max_seen = source_index

            for segment in _segments(
                message["text"]
            ):
                command = _parse_command(
                    segment
                )

                if command is None:
                    continue

                commands += 1

                action = command[
                    "action"
                ]

                if action == "assert":
                    (
                        added_delta,
                        confirmed_delta,
                    ) = _assert_fact(
                        active=active,
                        fact=command[
                            "fact"
                        ],
                        source_index=
                            source_index,
                        source_revision=
                            source_revision,
                    )

                    added += (
                        added_delta
                    )

                    confirmed += (
                        confirmed_delta
                    )

                    continue

                if action == "revoke":
                    matched = _revoke_fact(
                        active=active,
                        history=history,
                        target=command[
                            "target"
                        ],
                        source_index=
                            source_index,
                        source_revision=
                            source_revision,
                        status=
                            "revoked_explicit",
                    )

                    if matched:
                        revoked += 1
                    else:
                        unmatched += 1

                    continue

                if action == "replace":
                    matched = _revoke_fact(
                        active=active,
                        history=history,
                        target=command[
                            "target"
                        ],
                        source_index=
                            source_index,
                        source_revision=
                            source_revision,
                        status=
                            "superseded_explicit",
                        replacement=
                            command[
                                "replacement"
                            ],
                    )

                    if not matched:
                        unmatched += 1
                        continue

                    replaced += 1

                    (
                        added_delta,
                        confirmed_delta,
                    ) = _assert_fact(
                        active=active,
                        fact=command[
                            "replacement"
                        ],
                        source_index=
                            source_index,
                        source_revision=
                            source_revision,
                    )

                    added += (
                        added_delta
                    )

                    confirmed += (
                        confirmed_delta
                    )

        archived = 0

        while (
            len(active)
            > _MAX_ACTIVE_FACTS
        ):
            oldest = active.pop(
                0
            )

            closed = dict(
                oldest
            )

            closed[
                "status"
            ] = "archived_capacity"

            closed[
                "closed_revision"
            ] = source_revision

            history.append(
                closed
            )

            archived += 1

        history_dropped = 0

        if len(history) > _MAX_HISTORY:
            history_dropped = (
                len(history)
                - _MAX_HISTORY
            )

            history = history[
                -_MAX_HISTORY:
            ]

        previous_telemetry = state.get(
            "telemetry"
        )

        if not isinstance(
            previous_telemetry,
            dict,
        ):
            previous_telemetry = {}

        dropped_total = int(
            previous_telemetry.get(
                "history_dropped_total",
                0,
            )
            or 0
        )

        dropped_total += (
            history_dropped
        )

        state = {
            "version":
                _VERSION,
            "conversation_id":
                conversation_id,
            "revision":
                int(
                    state.get(
                        "revision",
                        0,
                    )
                    or 0
                )
                + 1,
            "source_revision":
                source_revision,
            "last_processed_source_index":
                max_seen,
            "updated_at":
                _now(),
            "active_facts":
                active,
            "history":
                history,
            "telemetry": {
                "mode":
                    "explicit-user-only",
                "inference_enabled":
                    False,
                "commands_this_revision":
                    commands,
                "added_this_revision":
                    added,
                "confirmed_this_revision":
                    confirmed,
                "revoked_this_revision":
                    revoked,
                "replaced_this_revision":
                    replaced,
                "unmatched_this_revision":
                    unmatched,
                "archived_this_revision":
                    archived,
                "history_dropped_total":
                    dropped_total,
            },
        }

        _atomic_write(
            target_path,
            state,
        )

    return {
        "stored": True,
        "duplicate": False,
        "conversation_id":
            conversation_id,
        "revision":
            state["revision"],
        "source_revision":
            source_revision,
        "fact_count":
            len(
                state[
                    "active_facts"
                ]
            ),
        "history_count":
            len(
                state[
                    "history"
                ]
            ),
        "commands_this_revision":
            commands,
        "added_this_revision":
            added,
        "confirmed_this_revision":
            confirmed,
        "revoked_this_revision":
            revoked,
        "replaced_this_revision":
            replaced,
        "unmatched_this_revision":
            unmatched,
        "inference_enabled": False,
    }


def trusted_facts_status(
    conversation_id: str,
) -> dict[str, Any]:

    path = _facts_path(
        conversation_id
    )

    with _LOCK:
        state = _read_json(
            path
        )

    if state is None:
        return {
            "exists": False,
            "conversation_id":
                conversation_id,
        }

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
        "fact_count":
            len(
                state.get(
                    "active_facts"
                )
                or []
            ),
        "history_count":
            len(
                state.get(
                    "history"
                )
                or []
            ),
        "telemetry":
            state.get(
                "telemetry"
            )
            or {},
    }
