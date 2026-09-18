from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ombrebrain.gateway.cache_fingerprint import (
    cache_fingerprint_summary_from_body,
)


_LOCK = threading.RLock()
_VERSION = "conversation-shadow.v1"
_MAX_CONVERSATIONS = 128
_DEFAULT_ROOT = "/app/buckets/.context"


def _root() -> Path:
    value = (
        os.environ.get("OMBRE_CONTEXT_STATE_DIR")
        or _DEFAULT_ROOT
    ).strip()

    return Path(value)


def _index_path() -> Path:
    return _root() / "conversation_shadow.json"


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _empty_index() -> dict[str, Any]:
    return {
        "version": _VERSION,
        "conversations": [],
    }


def _load_unlocked() -> dict[str, Any]:
    path = _index_path()

    if not path.is_file():
        return _empty_index()

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return _empty_index()

    if not isinstance(payload, dict):
        return _empty_index()

    conversations = payload.get("conversations")

    if not isinstance(conversations, list):
        conversations = []

    return {
        "version": _VERSION,
        "conversations": [
            item
            for item in conversations
            if isinstance(item, dict)
        ][-_MAX_CONVERSATIONS:],
    }


def _save_unlocked(payload: dict[str, Any]) -> None:
    root = _root()

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        root.chmod(0o700)
    except OSError:
        pass

    payload = {
        "version": _VERSION,
        "conversations": list(
            payload.get("conversations") or []
        )[-_MAX_CONVERSATIONS:],
    }

    path = _index_path()
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


def observe_conversation_shadow(
    body: bytes,
) -> dict[str, Any]:
    """
    Observe append-only conversation continuity.

    Privacy rule:
    only hashes, counts and locally generated conversation IDs
    are persisted. Prompt/system/user/assistant text is never written.

    No request mutation occurs here.
    """

    fingerprint = (
        cache_fingerprint_summary_from_body(body)
    )

    if fingerprint is None:
        return {
            "observed": False,
            "reason": "unsupported_or_non_json",
        }

    boundary = fingerprint.get(
        "boundary_prefix_sha256"
    )

    parent = fingerprint.get(
        "parent_prefix_sha256"
    )

    if not isinstance(boundary, str):
        return {
            "observed": False,
            "reason": "no_cache_boundary",
        }

    message_count = fingerprint.get(
        "messages_count",
        0,
    )

    if (
        not isinstance(message_count, int)
        or isinstance(message_count, bool)
    ):
        message_count = 0

    now = _now()

    with _LOCK:
        index = _load_unlocked()
        conversations = index["conversations"]

        # Exact same stable boundary means retry/duplicate
        # observation. Do not increment the round counter.
        duplicate = None

        for item in conversations:
            if (
                item.get(
                    "last_boundary_prefix_sha256"
                )
                == boundary
            ):
                duplicate = item
                break

        if duplicate is not None:
            duplicate["last_seen_at"] = now
            duplicate["last_message_count"] = (
                message_count
            )

            _save_unlocked(index)

            return {
                "observed": True,
                "conversation_id": duplicate[
                    "conversation_id"
                ],
                "round": int(
                    duplicate.get("rounds", 1)
                ),
                "new_conversation": False,
                "duplicate": True,
                "continuity": "same_boundary",
                "messages_count": message_count,
                "boundary_prefix_sha256": boundary,
                "parent_prefix_sha256": parent,
            }

        # Normal append-only continuation:
        # previous turn's boundary == this turn's parent.
        matched = None

        if isinstance(parent, str):
            for item in conversations:
                if (
                    item.get(
                        "last_boundary_prefix_sha256"
                    )
                    == parent
                ):
                    matched = item
                    break

        if matched is None:
            conversation_id = (
                "ctx_"
                + uuid.uuid4().hex[:16]
            )

            item = {
                "conversation_id": conversation_id,
                "created_at": now,
                "last_seen_at": now,
                "rounds": 1,
                "last_message_count": message_count,
                "last_boundary_prefix_sha256": (
                    boundary
                ),
                "last_parent_prefix_sha256": (
                    parent
                ),
            }

            conversations.append(item)

            if (
                len(conversations)
                > _MAX_CONVERSATIONS
            ):
                del conversations[
                    : len(conversations)
                    - _MAX_CONVERSATIONS
                ]

            _save_unlocked(index)

            return {
                "observed": True,
                "conversation_id": conversation_id,
                "round": 1,
                "new_conversation": True,
                "duplicate": False,
                "continuity": "new",
                "messages_count": message_count,
                "boundary_prefix_sha256": boundary,
                "parent_prefix_sha256": parent,
            }

        matched["rounds"] = (
            int(
                matched.get(
                    "rounds",
                    1,
                )
            )
            + 1
        )

        matched["last_seen_at"] = now
        matched["last_message_count"] = (
            message_count
        )
        matched[
            "last_parent_prefix_sha256"
        ] = parent
        matched[
            "last_boundary_prefix_sha256"
        ] = boundary

        _save_unlocked(index)

        return {
            "observed": True,
            "conversation_id": matched[
                "conversation_id"
            ],
            "round": matched["rounds"],
            "new_conversation": False,
            "duplicate": False,
            "continuity": "parent_match",
            "messages_count": message_count,
            "boundary_prefix_sha256": boundary,
            "parent_prefix_sha256": parent,
        }


def conversation_shadow_status() -> dict[str, Any]:
    with _LOCK:
        payload = _load_unlocked()

    conversations = payload[
        "conversations"
    ]

    return {
        "version": _VERSION,
        "conversation_count": len(
            conversations
        ),
        "conversations": [
            {
                "conversation_id": item.get(
                    "conversation_id"
                ),
                "rounds": item.get(
                    "rounds",
                    0,
                ),
                "created_at": item.get(
                    "created_at"
                ),
                "last_seen_at": item.get(
                    "last_seen_at"
                ),
                "last_message_count": item.get(
                    "last_message_count",
                    0,
                ),
            }
            for item in conversations
        ],
    }
