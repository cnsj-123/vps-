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

_VERSION = "conversation-semantic-state.v1"
_DEFAULT_ROOT = "/app/buckets/.context"

_MAX_ACTIVE_PER_KIND = 16
_MAX_HISTORY = 32

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_SPACE_RE = re.compile(r"\s+")


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

        state = _read_json(
            state_path
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

        current_task = semantic.get(
            "current_task"
        )

        if not isinstance(
            current_task,
            dict,
        ):
            current_task = (
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
