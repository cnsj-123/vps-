from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()

_VERSION = "conversation-compact.v1"
_DEFAULT_ROOT = "/app/buckets/.context"

# 最近 6 条完整保留。
_RECENT_MESSAGES = 6

# 再向前最多保留 6 条，但每条只保留确定性 excerpt。
_OLDER_MESSAGES = 6
_OLDER_EXCERPT_CHARS = 180

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
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


def _snapshot_path(
    conversation_id: str,
) -> Path:
    _validate_conversation_id(
        conversation_id
    )

    return (
        _root()
        / "snapshots"
        / (conversation_id + ".json")
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


def _clip_excerpt(
    text: str,
) -> tuple[str, bool]:
    text = text.strip()

    if len(text) <= _OLDER_EXCERPT_CHARS:
        return text, False

    marker = " […] "

    remaining = (
        _OLDER_EXCERPT_CHARS
        - len(marker)
    )

    left = remaining // 2
    right = remaining - left

    return (
        text[:left]
        + marker
        + text[-right:],
        True,
    )


def build_compact_payload(
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic two-layer conversation compression.

    No LLM call and no semantic inference:
      - newest 6 stable messages are preserved exactly;
      - up to 6 preceding messages become bounded excerpts.
    """

    conversation_id = snapshot.get(
        "conversation_id"
    )

    _validate_conversation_id(
        conversation_id
    )

    source_revision = snapshot.get(
        "revision"
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
            "invalid snapshot revision"
        )

    source_messages = snapshot.get(
        "messages"
    )

    if not isinstance(
        source_messages,
        list,
    ):
        source_messages = []

    clean_messages: list[
        dict[str, Any]
    ] = []

    for item in source_messages:
        if not isinstance(
            item,
            dict,
        ):
            continue

        role = item.get("role")
        text = item.get("text")

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

        clean_messages.append(
            {
                "role": role,
                "text": text,
                "source_index": item.get(
                    "source_index"
                ),
            }
        )

    recent = clean_messages[
        -_RECENT_MESSAGES:
    ]

    older_end = max(
        0,
        len(clean_messages)
        - len(recent),
    )

    older_source = clean_messages[
        max(
            0,
            older_end
            - _OLDER_MESSAGES,
        ):
        older_end
    ]

    older_context: list[
        dict[str, Any]
    ] = []

    clipped_count = 0

    for item in older_source:
        excerpt, clipped = (
            _clip_excerpt(
                item["text"]
            )
        )

        if clipped:
            clipped_count += 1

        older_context.append(
            {
                "role": item["role"],
                "excerpt": excerpt,
                "source_index": item.get(
                    "source_index"
                ),
            }
        )

    source_chars = sum(
        len(item["text"])
        for item in clean_messages
    )

    compact_chars = (
        sum(
            len(item["excerpt"])
            for item in older_context
        )
        + sum(
            len(item["text"])
            for item in recent
        )
    )

    if source_chars > 0:
        ratio = (
            float(compact_chars)
            / float(source_chars)
        )
    else:
        ratio = 0.0

    return {
        "version": _VERSION,
        "conversation_id":
            conversation_id,
        "source_revision":
            source_revision,
        "source_updated_at":
            snapshot.get(
                "updated_at"
            ),
        "created_at": _now(),
        "older_context":
            older_context,
        "recent_messages":
            recent,
        "telemetry": {
            "source_messages":
                len(clean_messages),
            "older_messages":
                len(older_context),
            "recent_messages":
                len(recent),
            "older_clipped":
                clipped_count,
            "source_chars":
                source_chars,
            "compact_chars":
                compact_chars,
            "compaction_ratio":
                ratio,
            "compaction_applied":
                compact_chars
                < source_chars,
        },
    }


def update_conversation_compact(
    conversation_id: str,
) -> dict[str, Any]:
    """
    Build/update the deterministic compact shadow from
    the latest bounded conversation snapshot.
    """

    source_path = _snapshot_path(
        conversation_id
    )

    target_path = _compact_path(
        conversation_id
    )

    with _LOCK:
        snapshot = _read_json(
            source_path
        )

        if snapshot is None:
            return {
                "stored": False,
                "reason":
                    "snapshot_not_found",
                "conversation_id":
                    conversation_id,
            }

        source_revision = snapshot.get(
            "revision"
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

        compact = build_compact_payload(
            snapshot
        )

        _atomic_write(
            target_path,
            compact,
        )

    telemetry = compact[
        "telemetry"
    ]

    return {
        "stored": True,
        "duplicate": False,
        "conversation_id":
            conversation_id,
        "source_revision":
            compact[
                "source_revision"
            ],
        **telemetry,
    }


def compact_status(
    conversation_id: str,
) -> dict[str, Any]:
    path = _compact_path(
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

    # No conversation text is exposed here.
    return {
        "exists": True,
        "conversation_id":
            conversation_id,
        "version": payload.get(
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
        "older_count": len(
            payload.get(
                "older_context"
            )
            if isinstance(
                payload.get(
                    "older_context"
                ),
                list,
            )
            else []
        ),
        "recent_count": len(
            payload.get(
                "recent_messages"
            )
            if isinstance(
                payload.get(
                    "recent_messages"
                ),
                list,
            )
            else []
        ),
        "telemetry": telemetry,
    }
