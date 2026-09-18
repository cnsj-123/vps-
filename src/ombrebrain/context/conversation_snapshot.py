from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ombrebrain.gateway.models import SegmentKind
from ombrebrain.gateway.operit_adapter import OperitAdapter


_LOCK = threading.RLock()

_VERSION = "conversation-snapshot.v1"
_DEFAULT_ROOT = "/app/buckets/.context"

_MAX_MESSAGES = 12
_MAX_MESSAGE_CHARS = 3000
_MAX_TOTAL_CHARS = 16000

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


def _snapshot_dir() -> Path:
    return _root() / "snapshots"


def _snapshot_path(
    conversation_id: str,
) -> Path:
    if not isinstance(
        conversation_id,
        str,
    ):
        raise ValueError(
            "conversation_id must be a string"
        )

    if not _CONVERSATION_ID_RE.fullmatch(
        conversation_id
    ):
        raise ValueError(
            "invalid conversation_id"
        )

    return (
        _snapshot_dir()
        / (
            conversation_id
            + ".json"
        )
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


def _load_existing(
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


def _trim_message(
    text: str,
    limit: int,
) -> tuple[str, bool]:
    text = text.strip()

    if len(text) <= limit:
        return text, False

    if limit <= 32:
        return (
            text[:limit],
            True,
        )

    marker = "\n[…]\n"

    left = (
        limit
        - len(marker)
    ) // 2

    right = (
        limit
        - len(marker)
        - left
    )

    return (
        text[:left]
        + marker
        + text[-right:],
        True,
    )


def _stable_messages(
    body: bytes,
) -> tuple[
    list[dict[str, Any]],
    dict[str, int | bool],
] | None:

    try:
        payload = json.loads(
            body
        )
    except Exception:
        return None

    if not OperitAdapter.supports(
        payload
    ):
        return None

    request = OperitAdapter().adapt(
        payload
    )

    canonical_messages = list(
        request.history
    )

    if request.current is not None:
        canonical_messages.append(
            request.current
        )

    candidates: list[
        dict[str, Any]
    ] = []

    excluded_segments = 0
    truncated_messages = 0

    for message in canonical_messages:
        if message.role not in (
            "user",
            "assistant",
        ):
            continue

        stable_parts: list[str] = []

        for block in message.blocks:
            if (
                block.kind
                is SegmentKind.USER_TEXT
                and isinstance(
                    block.text,
                    str,
                )
                and block.text.strip()
            ):
                stable_parts.append(
                    block.text.strip()
                )
            else:
                excluded_segments += 1

        if not stable_parts:
            continue

        text = "\n".join(
            stable_parts
        )

        text, truncated = (
            _trim_message(
                text,
                _MAX_MESSAGE_CHARS,
            )
        )

        if truncated:
            truncated_messages += 1

        candidates.append(
            {
                "role": message.role,
                "text": text,
                "source_index": (
                    message.original_index
                ),
            }
        )

    # Snapshot keeps the newest stable messages.
    candidates = candidates[
        -_MAX_MESSAGES:
    ]

    selected: list[
        dict[str, Any]
    ] = []

    used_chars = 0
    total_budget_truncated = False

    # Work newest -> oldest so recent context wins.
    for item in reversed(
        candidates
    ):
        remaining = (
            _MAX_TOTAL_CHARS
            - used_chars
        )

        if remaining <= 0:
            total_budget_truncated = True
            break

        text = item["text"]

        if len(text) > remaining:
            text, _ = _trim_message(
                text,
                remaining,
            )
            total_budget_truncated = True

        selected.append(
            {
                "role": item["role"],
                "text": text,
                "source_index": (
                    item[
                        "source_index"
                    ]
                ),
            }
        )

        used_chars += len(
            text
        )

    selected.reverse()

    if len(selected) < len(
        candidates
    ):
        total_budget_truncated = True

    telemetry = {
        "stable_candidates": len(
            candidates
        ),
        "included_messages": len(
            selected
        ),
        "excluded_segments": (
            excluded_segments
        ),
        "truncated_messages": (
            truncated_messages
        ),
        "total_chars": (
            used_chars
        ),
        "budget_truncated": (
            total_budget_truncated
        ),
    }

    return (
        selected,
        telemetry,
    )


def update_conversation_snapshot(
    body: bytes,
    *,
    conversation_id: str,
    boundary_prefix_sha256: (
        str | None
    ) = None,
) -> dict[str, Any]:

    path = _snapshot_path(
        conversation_id
    )

    extracted = _stable_messages(
        body
    )

    if extracted is None:
        return {
            "stored": False,
            "reason":
                "unsupported_or_non_json",
        }

    messages, telemetry = (
        extracted
    )

    now = _now()

    with _LOCK:
        previous = _load_existing(
            path
        )

        previous_revision = 0

        if isinstance(
            previous,
            dict,
        ):
            value = previous.get(
                "revision",
                0,
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
                and value >= 0
            ):
                previous_revision = (
                    value
                )

            if (
                isinstance(
                    boundary_prefix_sha256,
                    str,
                )
                and previous.get(
                    "boundary_prefix_sha256"
                )
                == boundary_prefix_sha256
            ):
                return {
                    "stored": True,
                    "duplicate": True,
                    "conversation_id":
                        conversation_id,
                    "revision":
                        previous_revision,
                    **telemetry,
                }

        snapshot = {
            "version": _VERSION,
            "conversation_id":
                conversation_id,
            "revision":
                previous_revision + 1,
            "updated_at": now,
            "boundary_prefix_sha256":
                boundary_prefix_sha256,
            "messages": messages,
            "telemetry": telemetry,
        }

        _atomic_write(
            path,
            snapshot,
        )

    return {
        "stored": True,
        "duplicate": False,
        "conversation_id":
            conversation_id,
        "revision":
            snapshot["revision"],
        **telemetry,
    }


def snapshot_status(
    conversation_id: str,
) -> dict[str, Any]:

    path = _snapshot_path(
        conversation_id
    )

    with _LOCK:
        payload = _load_existing(
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

    # Status intentionally exposes no conversation text.
    return {
        "exists": True,
        "conversation_id":
            conversation_id,
        "version": payload.get(
            "version"
        ),
        "revision": payload.get(
            "revision"
        ),
        "updated_at": payload.get(
            "updated_at"
        ),
        "message_count": len(
            payload.get(
                "messages"
            )
            if isinstance(
                payload.get(
                    "messages"
                ),
                list,
            )
            else []
        ),
        "telemetry": telemetry,
    }
