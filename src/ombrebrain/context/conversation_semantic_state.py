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

_VERSION = "conversation-semantic-state.v2"
_PREVIOUS_VERSION = "conversation-semantic-state.v1"
_DEFAULT_ROOT = "/app/buckets/.context"

_MAX_ACTIVE_PER_KIND = 16
_MAX_HISTORY = 32

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_SPACE_RE = re.compile(r"\s+")

_CJK_CONTROL_SPACE_RE = re.compile(
    r"(?<=[\u4e00-\u9fff：:])"
    r"\s+"
    r"(?=[\u4e00-\u9fff：:])"
)


def _normalize_lifecycle_text(
    text: str,
) -> str:
    # Preserve original casing/content because captured
    # replacement text may be persisted into semantic state.
    text = text.strip()

    text = _CJK_CONTROL_SPACE_RE.sub(
        "",
        text,
    )

    return _SPACE_RE.sub(
        " ",
        text,
    )


def _lifecycle_compare_key(
    text: str,
) -> str:
    # Matching is case-insensitive, persistence is not.
    return (
        _normalize_lifecycle_text(
            text
        )
        .casefold()
    )



_LIFECYCLE_PATTERNS = (
    (
        "replace",
        "constraint",
        re.compile(
            r"^\s*(?:替换|更新)约束\s*[:：]\s*"
            r"(?P<target>.+?)\s*"
            r"(?:=>|->|→)\s*"
            r"(?P<replacement>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "replace",
        "decision",
        re.compile(
            r"^\s*(?:替换|更新)决定\s*[:：]\s*"
            r"(?P<target>.+?)\s*"
            r"(?:=>|->|→)\s*"
            r"(?P<replacement>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "cancel",
        "constraint",
        re.compile(
            r"^\s*(?:取消|关闭|移除)约束\s*[:：]\s*"
            r"(?P<target>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "revoke",
        "decision",
        re.compile(
            r"^\s*(?:撤销|取消)决定\s*[:：]\s*"
            r"(?P<target>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "resolve",
        "open_item",
        re.compile(
            r"^\s*(?:完成|关闭|解决)待办\s*[:：]\s*"
            r"(?P<target>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
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


def _state_path(
    conversation_id: str,
) -> Path:
    _validate_conversation_id(
        conversation_id
    )

    return (
        _root()
        / "semantic_state"
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


def _normalize_text(
    text: str,
) -> str:
    return _SPACE_RE.sub(
        " ",
        text.strip().lower(),
    )


def _item_id(
    kind: str,
    text: str,
) -> str:
    raw = (
        kind
        + "\0"
        + _normalize_text(text)
    ).encode("utf-8")

    return (
        "sem_"
        + hashlib.sha256(
            raw
        ).hexdigest()[:16]
    )



def _parse_lifecycle_command(
    current_task: Any,
) -> dict[str, Any] | None:

    if not isinstance(
        current_task,
        dict,
    ):
        return None

    text = current_task.get(
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

    source_index = current_task.get(
        "source_index"
    )

    if (
        not isinstance(
            source_index,
            int,
        )
        or isinstance(
            source_index,
            bool,
        )
    ):
        source_index = None

    for (
        action,
        kind,
        pattern,
    ) in _LIFECYCLE_PATTERNS:

        match = pattern.fullmatch(
            _normalize_lifecycle_text(
                text
            )
        )

        if match is None:
            continue

        target = (
            match.group(
                "target"
            )
            .strip()
        )

        if not target:
            return None

        replacement = None

        if action == "replace":
            replacement = (
                match.group(
                    "replacement"
                )
                .strip()
            )

            if not replacement:
                return None

        return {
            "action": action,
            "kind": kind,
            "target_text":
                target,
            "replacement_text":
                replacement,
            "source_index":
                source_index,
        }

    return None


def _prepare_state(
    payload: Any,
    conversation_id: str,
) -> dict[str, Any]:

    if not isinstance(
        payload,
        dict,
    ):
        return _empty_state(
            conversation_id
        )

    version = payload.get(
        "version"
    )

    if version == _VERSION:
        return dict(payload)

    if version == _PREVIOUS_VERSION:
        migrated = dict(
            payload
        )

        migrated["version"] = (
            _VERSION
        )

        for key in (
            "constraints",
            "decisions",
            "open_items",
            "established_facts",
            "history",
        ):
            if not isinstance(
                migrated.get(key),
                list,
            ):
                migrated[key] = []

        return migrated

    return _empty_state(
        conversation_id
    )


def _apply_lifecycle_command(
    *,
    state: dict[str, Any],
    history: list[dict[str, Any]],
    event: dict[str, Any] | None,
    source_revision: int,
) -> dict[str, Any]:

    result = {
        "detected":
            event is not None,
        "matched": False,
        "action":
            event.get("action")
            if isinstance(
                event,
                dict,
            )
            else None,
        "kind":
            event.get("kind")
            if isinstance(
                event,
                dict,
            )
            else None,
        "replacement": None,
    }

    if event is None:
        return result

    kind = event["kind"]

    key_by_kind = {
        "constraint":
            "constraints",
        "decision":
            "decisions",
        "open_item":
            "open_items",
    }

    key = key_by_kind.get(
        kind
    )

    if key is None:
        return result

    active = [
        dict(item)
        for item in (
            state.get(key)
            or []
        )
        if isinstance(
            item,
            dict,
        )
    ]

    target_norm = (
        _lifecycle_compare_key(
            event["target_text"]
        )
    )

    matched_index = None

    for index, item in enumerate(
        active
    ):
        text = item.get("text")

        if (
            isinstance(
                text,
                str,
            )
            and _lifecycle_compare_key(
                text
            )
            == target_norm
        ):
            matched_index = index
            break

    if matched_index is None:
        return result

    closed = active.pop(
        matched_index
    )

    action = event["action"]

    status_by_action = {
        "cancel":
            "cancelled_explicit",
        "revoke":
            "revoked_explicit",
        "resolve":
            "resolved_explicit",
        "replace":
            "superseded_explicit",
    }

    closed["status"] = (
        status_by_action[
            action
        ]
    )

    closed[
        "closed_revision"
    ] = source_revision

    closed[
        "close_source_index"
    ] = event.get(
        "source_index"
    )

    closed[
        "lifecycle_action"
    ] = action

    replacement_text = (
        event.get(
            "replacement_text"
        )
    )

    if (
        action == "replace"
        and isinstance(
            replacement_text,
            str,
        )
        and replacement_text
    ):
        replacement_id = (
            _item_id(
                kind,
                replacement_text,
            )
        )

        closed[
            "superseded_by_id"
        ] = replacement_id

        result[
            "replacement"
        ] = {
            "text":
                replacement_text,
            "source_index":
                event.get(
                    "source_index"
                ),
        }

    history.append(
        closed
    )

    state[key] = active

    result["matched"] = True

    return result


def _filter_lifecycle_source(
    incoming: list[Any],
    event: dict[str, Any] | None,
    lifecycle_result: dict[str, Any],
    history: list[dict[str, Any]],
) -> list[Any]:

    source_index = None
    target_norm = None

    if isinstance(
        event,
        dict,
    ):
        source_index = event.get(
            "source_index"
        )

        target_text = event.get(
            "target_text"
        )

        if isinstance(
            target_text,
            str,
        ):
            target_norm = (
                _lifecycle_compare_key(
                    target_text
                )
            )

    matched = bool(
        lifecycle_result.get(
            "matched"
        )
    )

    result = []

    for item in incoming:
        if not isinstance(
            item,
            dict,
        ):
            result.append(item)
            continue

        item_source_index = (
            item.get(
                "source_index"
            )
        )

        item_text = item.get(
            "text"
        )

        if not isinstance(
            item_text,
            str,
        ):
            result.append(item)
            continue

        # Lifecycle control commands are never semantic content.
        parsed = _parse_lifecycle_command(
            {
                "text": item_text,
                "source_index":
                    item_source_index,
            }
        )

        if parsed is not None:
            continue

        if (
            source_index is not None
            and item_source_index
            == source_index
        ):
            continue

        item_norm = (
            _lifecycle_compare_key(
                item_text
            )
        )

        # Same-turn resurrection guard.
        if (
            matched
            and target_norm is not None
            and item_norm
            == target_norm
        ):
            continue

        # Later-turn bounded-window resurrection guard.
        blocked = False

        for closed in history:
            if not isinstance(
                closed,
                dict,
            ):
                continue

            if closed.get(
                "status"
            ) not in (
                "cancelled_explicit",
                "revoked_explicit",
                "resolved_explicit",
                "superseded_explicit",
            ):
                continue

            closed_text = closed.get(
                "text"
            )

            if not isinstance(
                closed_text,
                str,
            ):
                continue

            if (
                _lifecycle_compare_key(
                    closed_text
                )
                != item_norm
            ):
                continue

            close_source_index = (
                closed.get(
                    "close_source_index"
                )
            )

            if (
                isinstance(
                    close_source_index,
                    int,
                )
                and not isinstance(
                    close_source_index,
                    bool,
                )
                and isinstance(
                    item_source_index,
                    int,
                )
                and not isinstance(
                    item_source_index,
                    bool,
                )
                and item_source_index
                <= close_source_index
            ):
                blocked = True
                break

        if blocked:
            continue

        result.append(item)

    return result


def _purge_closed_echoes(
    *,
    state: dict[str, Any],
    history: list[dict[str, Any]],
) -> int:

    closed_statuses = {
        "cancelled_explicit",
        "revoked_explicit",
        "resolved_explicit",
        "superseded_explicit",
    }

    purged = 0

    for key, kind in (
        ("constraints", "constraint"),
        ("decisions", "decision"),
        ("open_items", "open_item"),
    ):
        active = state.get(key)

        if not isinstance(
            active,
            list,
        ):
            state[key] = []
            continue

        kept = []

        for item in active:
            if not isinstance(
                item,
                dict,
            ):
                kept.append(item)
                continue

            item_text = item.get(
                "text"
            )

            item_source = item.get(
                "first_source_index"
            )

            if (
                not isinstance(
                    item_text,
                    str,
                )
                or not isinstance(
                    item_source,
                    int,
                )
                or isinstance(
                    item_source,
                    bool,
                )
            ):
                kept.append(item)
                continue

            item_norm = (
                _lifecycle_compare_key(
                    item_text
                )
            )

            stale = False

            for closed in history:
                if not isinstance(
                    closed,
                    dict,
                ):
                    continue

                if closed.get(
                    "status"
                ) not in closed_statuses:
                    continue

                if closed.get(
                    "kind"
                ) != kind:
                    continue

                closed_text = closed.get(
                    "text"
                )

                close_source = closed.get(
                    "close_source_index"
                )

                if (
                    not isinstance(
                        closed_text,
                        str,
                    )
                    or not isinstance(
                        close_source,
                        int,
                    )
                    or isinstance(
                        close_source,
                        bool,
                    )
                ):
                    continue

                if (
                    _lifecycle_compare_key(
                        closed_text
                    )
                    != item_norm
                ):
                    continue

                if item_source <= close_source:
                    stale = True
                    break

            if stale:
                purged += 1
            else:
                kept.append(item)

        state[key] = kept

    return purged


def _empty_state(
    conversation_id: str,
) -> dict[str, Any]:
    return {
        "version": _VERSION,
        "conversation_id":
            conversation_id,
        "revision": 0,
        "source_revision": 0,
        "updated_at": None,
        "current_task": None,
        "constraints": [],
        "decisions": [],
        "open_items": [],
        "established_facts": [],
        "history": [],
        "telemetry": {
            "facts_deferred": True,
            "history_dropped_total": 0,
        },
    }


def _clean_source_item(
    item: Any,
) -> dict[str, Any] | None:
    if not isinstance(
        item,
        dict,
    ):
        return None

    text = item.get("text")

    if (
        not isinstance(
            text,
            str,
        )
        or not text.strip()
    ):
        return None

    source_index = item.get(
        "source_index"
    )

    if (
        not isinstance(
            source_index,
            int,
        )
        or isinstance(
            source_index,
            bool,
        )
    ):
        source_index = None

    return {
        "text": text.strip(),
        "source_index":
            source_index,
    }


def _merge_kind(
    *,
    kind: str,
    active: list[Any],
    incoming: list[Any],
    source_revision: int,
    history: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    int,
    int,
    int,
]:
    clean_active = [
        dict(item)
        for item in active
        if isinstance(
            item,
            dict,
        )
        and isinstance(
            item.get("id"),
            str,
        )
        and isinstance(
            item.get("text"),
            str,
        )
    ]

    by_id = {
        item["id"]: item
        for item in clean_active
    }

    added = 0
    refreshed = 0
    archived = 0

    for raw in incoming:
        item = _clean_source_item(
            raw
        )

        if item is None:
            continue

        item_id = _item_id(
            kind,
            item["text"],
        )

        existing = by_id.get(
            item_id
        )

        if existing is not None:
            existing[
                "last_seen_revision"
            ] = source_revision

            existing[
                "last_source_index"
            ] = item[
                "source_index"
            ]

            existing["occurrences"] = (
                int(
                    existing.get(
                        "occurrences",
                        1,
                    )
                )
                + 1
            )

            refreshed += 1
            continue

        created = {
            "id": item_id,
            "kind": kind,
            "status": "active",
            "text": item["text"],
            "first_source_index":
                item["source_index"],
            "last_source_index":
                item["source_index"],
            "first_seen_revision":
                source_revision,
            "last_seen_revision":
                source_revision,
            "occurrences": 1,
        }

        clean_active.append(
            created
        )

        by_id[item_id] = created
        added += 1

    while (
        len(clean_active)
        > _MAX_ACTIVE_PER_KIND
    ):
        oldest = clean_active.pop(0)

        archived_item = dict(
            oldest
        )

        archived_item[
            "status"
        ] = "archived_capacity"

        archived_item[
            "closed_revision"
        ] = source_revision

        history.append(
            archived_item
        )

        archived += 1

    return (
        clean_active,
        added,
        refreshed,
        archived,
    )


def update_semantic_state(
    conversation_id: str,
) -> dict[str, Any]:

    semantic_path = _semantic_path(
        conversation_id
    )

    state_path = _state_path(
        conversation_id
    )

    with _LOCK:
        semantic = _read_json(
            semantic_path
        )

        if semantic is None:
            return {
                "stored": False,
                "reason":
                    "semantic_not_found",
                "conversation_id":
                    conversation_id,
            }

        source_revision = semantic.get(
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

        state = _prepare_state(
            _read_json(
                state_path
            ),
            conversation_id,
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
                "source_revision":
                    source_revision,
                "current_source_revision":
                    previous_source_revision,
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
                "constraint_count":
                    len(
                        state.get(
                            "constraints"
                        )
                        or []
                    ),
                "decision_count":
                    len(
                        state.get(
                            "decisions"
                        )
                        or []
                    ),
                "open_item_count":
                    len(
                        state.get(
                            "open_items"
                        )
                        or []
                    ),
            }

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

        purged_closed_echoes = (
            _purge_closed_echoes(
                state=state,
                history=history,
            )
        )

        semantic_telemetry = (
            semantic.get(
                "telemetry"
            )
        )

        if not isinstance(
            semantic_telemetry,
            dict,
        ):
            semantic_telemetry = {}

        lifecycle_skipped_carried_forward = bool(
            semantic_telemetry.get(
                "current_task_carried_forward"
            )
        )

        if lifecycle_skipped_carried_forward:
            lifecycle_event = None
        else:
            lifecycle_event = (
                _parse_lifecycle_command(
                    semantic.get(
                        "current_task"
                    )
                )
            )

        lifecycle_result = (
            _apply_lifecycle_command(
                state=state,
                history=history,
                event=lifecycle_event,
                source_revision=
                    source_revision,
            )
        )

        totals = {
            "added": 0,
            "refreshed": 0,
            "archived": 0,
        }

        for key, kind in (
            (
                "constraints",
                "constraint",
            ),
            (
                "decisions",
                "decision",
            ),
            (
                "open_items",
                "open_item",
            ),
        ):
            incoming = semantic.get(
                key
            )

            if not isinstance(
                incoming,
                list,
            ):
                incoming = []

            incoming = (
                _filter_lifecycle_source(
                    incoming,
                    lifecycle_event,
                    lifecycle_result,
                    history,
                )
            )

            replacement_item = (
                lifecycle_result.get(
                    "replacement"
                )
            )

            replacement_kind = (
                lifecycle_result.get(
                    "kind"
                )
            )

            if (
                lifecycle_result.get(
                    "matched"
                )
                and isinstance(
                    replacement_item,
                    dict,
                )
                and replacement_kind
                == kind
            ):
                replacement_text = (
                    replacement_item.get(
                        "text"
                    )
                )

                replacement_exists = (
                    isinstance(
                        replacement_text,
                        str,
                    )
                    and any(
                        isinstance(
                            candidate,
                            dict,
                        )
                        and isinstance(
                            candidate.get(
                                "text"
                            ),
                            str,
                        )
                        and _lifecycle_compare_key(
                            candidate["text"]
                        )
                        == _lifecycle_compare_key(
                            replacement_text
                        )
                        for candidate
                        in incoming
                    )
                )

                if not replacement_exists:
                    incoming.append(
                        replacement_item
                    )

            (
                merged,
                added,
                refreshed,
                archived,
            ) = _merge_kind(
                kind=kind,
                active=(
                    state.get(
                        key
                    )
                    or []
                ),
                incoming=incoming,
                source_revision=
                    source_revision,
                history=history,
            )

            state[key] = merged

            totals["added"] += (
                added
            )

            totals["refreshed"] += (
                refreshed
            )

            totals["archived"] += (
                archived
            )

        history_dropped = 0

        if len(history) > _MAX_HISTORY:
            history_dropped = (
                len(history)
                - _MAX_HISTORY
            )

            history = history[
                -_MAX_HISTORY:
            ]

        previous_telemetry = (
            state.get(
                "telemetry"
            )
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

        semantic_current_task = (
            semantic.get(
                "current_task"
            )
        )

        previous_current_task = (
            state.get(
                "current_task"
            )
            if isinstance(
                state.get(
                    "current_task"
                ),
                dict,
            )
            else None
        )

        if lifecycle_event is not None:
            # Lifecycle commands manage semantic state.
            # They are not the user's actual ongoing task.
            current_task = (
                previous_current_task
            )
        elif isinstance(
            semantic_current_task,
            dict,
        ):
            current_task = (
                semantic_current_task
            )
        else:
            current_task = (
                previous_current_task
            )

        state["version"] = (
            _VERSION
        )

        state[
            "conversation_id"
        ] = conversation_id

        state["revision"] = (
            int(
                state.get(
                    "revision",
                    0,
                )
                or 0
            )
            + 1
        )

        state[
            "source_revision"
        ] = source_revision

        state["updated_at"] = (
            _now()
        )

        state["current_task"] = (
            current_task
        )

        # Trusted facts remain intentionally disabled.
        state[
            "established_facts"
        ] = []

        state["history"] = history

        previous_unmatched = int(
            previous_telemetry.get(
                "lifecycle_unmatched_total",
                0,
            )
            or 0
        )

        if (
            lifecycle_result.get(
                "detected"
            )
            and not lifecycle_result.get(
                "matched"
            )
        ):
            previous_unmatched += 1

        state["telemetry"] = {
            "facts_deferred": True,
            "added_this_revision":
                totals["added"],
            "refreshed_this_revision":
                totals["refreshed"],
            "archived_this_revision":
                totals["archived"],
            "history_dropped_total":
                dropped_total,
            "purged_closed_echoes_this_revision":
                purged_closed_echoes,
            "lifecycle_skipped_carried_forward":
                lifecycle_skipped_carried_forward,
            "lifecycle_command_detected":
                lifecycle_result.get(
                    "detected"
                ),
            "lifecycle_action":
                lifecycle_result.get(
                    "action"
                ),
            "lifecycle_kind":
                lifecycle_result.get(
                    "kind"
                ),
            "lifecycle_matched":
                lifecycle_result.get(
                    "matched"
                ),
            "lifecycle_replacement_added":
                (
                    lifecycle_result.get(
                        "matched"
                    )
                    and isinstance(
                        lifecycle_result.get(
                            "replacement"
                        ),
                        dict,
                    )
                ),
            "lifecycle_unmatched_total":
                previous_unmatched,
        }

        _atomic_write(
            state_path,
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
        "constraint_count":
            len(
                state[
                    "constraints"
                ]
            ),
        "decision_count":
            len(
                state[
                    "decisions"
                ]
            ),
        "open_item_count":
            len(
                state[
                    "open_items"
                ]
            ),
        "history_count":
            len(
                state[
                    "history"
                ]
            ),
        "added_this_revision":
            totals["added"],
        "refreshed_this_revision":
            totals["refreshed"],
        "archived_this_revision":
            totals["archived"],
        "purged_closed_echoes_this_revision":
            purged_closed_echoes,
        "lifecycle_skipped_carried_forward":
            lifecycle_skipped_carried_forward,
        "lifecycle_command_detected":
            lifecycle_result.get(
                "detected"
            ),
        "lifecycle_action":
            lifecycle_result.get(
                "action"
            ),
        "lifecycle_kind":
            lifecycle_result.get(
                "kind"
            ),
        "lifecycle_matched":
            lifecycle_result.get(
                "matched"
            ),
        "lifecycle_replacement_added":
            (
                lifecycle_result.get(
                    "matched"
                )
                and isinstance(
                    lifecycle_result.get(
                        "replacement"
                    ),
                    dict,
                )
            ),
        "facts_deferred": True,
    }


def semantic_state_status(
    conversation_id: str,
) -> dict[str, Any]:

    path = _state_path(
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

    # Privacy-safe status: no task/item text.
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
        "has_current_task":
            isinstance(
                state.get(
                    "current_task"
                ),
                dict,
            ),
        "constraint_count":
            len(
                state.get(
                    "constraints"
                )
                or []
            ),
        "decision_count":
            len(
                state.get(
                    "decisions"
                )
                or []
            ),
        "open_item_count":
            len(
                state.get(
                    "open_items"
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
        "fact_count": 0,
        "telemetry":
            state.get(
                "telemetry"
            )
            or {},
    }
