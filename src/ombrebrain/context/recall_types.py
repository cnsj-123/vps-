from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Recall Control Plane — shared primitives.
#
# This module holds only the small, dependency-free primitives the
# four Recall modules agree on:
#
#   - request-local identity formats (conversation / request /
#     recall / opaque memref);
#   - the on-disk artifact layout for related_recall / memory_usage
#     / recall_surface;
#   - atomic, privacy-neutral JSON IO;
#   - the conservative positive-int budget parser and the token
#     heuristic the rest of the Context layer already uses.
#
# It performs no authorization, no retrieval, no scoring and no
# memory write. Every Recall module may import it; it imports none
# of them, so there is no cycle and no shared mutable recall state
# beyond the filesystem lock.

_DEFAULT_ROOT = "/app/buckets/.context"

CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

COGNITIVE_REQUEST_ID_RE = re.compile(
    r"^ctxreq_[0-9a-f]{32}$"
)

RECALL_ID_RE = re.compile(
    r"^recall_[0-9a-f]{32}$"
)

MEMREF_RE = re.compile(
    r"^memref_[0-9a-f]{32}$"
)

# A request fingerprint is a full sha256 hex digest.
FINGERPRINT_RE = re.compile(
    r"^[0-9a-f]{64}$"
)

_SPACE_RE = re.compile(r"\s+")

# Serializes the artifact writes of the Recall control plane. It is
# only ever held for synchronous filesystem work, never across an
# await, so concurrent requests can never deadlock the event loop.
LOCK = threading.RLock()


def state_root() -> Path:
    return Path(
        (
            os.environ.get(
                "OMBRE_CONTEXT_STATE_DIR"
            )
            or _DEFAULT_ROOT
        ).strip()
    )


def now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def is_valid_conversation_id(
    value: Any,
) -> bool:
    return (
        isinstance(value, str)
        and bool(
            CONVERSATION_ID_RE.fullmatch(
                value
            )
        )
    )


def is_valid_cognitive_request_id(
    value: Any,
) -> bool:
    return (
        isinstance(value, str)
        and bool(
            COGNITIVE_REQUEST_ID_RE.fullmatch(
                value
            )
        )
    )


def validate_conversation_id(
    value: Any,
) -> None:
    if not is_valid_conversation_id(value):
        raise ValueError(
            "invalid conversation_id"
        )


def validate_cognitive_request_id(
    value: Any,
) -> None:
    if not is_valid_cognitive_request_id(
        value
    ):
        raise ValueError(
            "invalid cognitive_request_id"
        )


def is_valid_recall_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(
            RECALL_ID_RE.fullmatch(value)
        )
    )


def is_valid_memref(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(MEMREF_RE.fullmatch(value))
    )


def is_valid_fingerprint(value: Any) -> bool:
    """A full, lowercase sha256 hex digest."""

    return (
        isinstance(value, str)
        and bool(FINGERPRINT_RE.fullmatch(value))
    )


def related_recall_dir(
    conversation_id: str,
    cognitive_request_id: str,
) -> Path:
    validate_conversation_id(conversation_id)

    validate_cognitive_request_id(
        cognitive_request_id
    )

    return (
        state_root()
        / "related_recall"
        / conversation_id
        / cognitive_request_id
    )


def related_recall_path(
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
) -> Path:
    return (
        related_recall_dir(
            conversation_id,
            cognitive_request_id,
        )
        / (recall_id + ".json")
    )


def recall_request_dir(
    conversation_id: str,
    cognitive_request_id: str,
) -> Path:
    validate_conversation_id(conversation_id)

    validate_cognitive_request_id(
        cognitive_request_id
    )

    return (
        state_root()
        / "recall_request"
        / conversation_id
        / cognitive_request_id
    )


def recall_request_path(
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
) -> Path:
    return (
        recall_request_dir(
            conversation_id,
            cognitive_request_id,
        )
        / (recall_id + ".json")
    )


def memory_usage_path(
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
) -> Path:
    validate_conversation_id(conversation_id)

    validate_cognitive_request_id(
        cognitive_request_id
    )

    return (
        state_root()
        / "memory_usage"
        / conversation_id
        / cognitive_request_id
        / (recall_id + ".json")
    )


def recall_surface_path(
    conversation_id: str,
    cognitive_request_id: str,
) -> Path:
    validate_conversation_id(conversation_id)

    validate_cognitive_request_id(
        cognitive_request_id
    )

    return (
        state_root()
        / "recall_surface"
        / conversation_id
        / (cognitive_request_id + ".json")
    )


def read_json(
    path: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None

    return (
        value
        if isinstance(value, dict)
        else None
    )


def atomic_write(
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

    os.replace(temp, path)

    try:
        path.chmod(0o600)
    except OSError:
        pass


def parse_positive_int(
    raw: Any,
    *,
    default: int,
    cap: int,
) -> int:
    """Strict positive-int parsing with a hard cap.

    Anything that is not a plain positive decimal integer degrades
    to the safe default, and a parsed value is always clamped to
    ``cap``, so configuration can never open an unbounded Recall.
    """

    if raw is None:
        return default

    text = str(raw).strip()

    if not text.isdigit():
        return default

    try:
        value = int(text)
    except (TypeError, ValueError):
        return default

    if value < 1:
        return default

    return min(value, cap)


def normalize_whitespace(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    return _SPACE_RE.sub(
        " ",
        value.strip(),
    )


def normalize_key(text: str) -> str:
    """Exact identity of the existing excluded-text mechanism."""

    return _SPACE_RE.sub(
        " ",
        text.strip(),
    ).casefold()


def estimate_tokens(value: Any) -> int:
    if value is None:
        return 0

    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    if not text:
        return 0

    return max(1, (len(text) + 2) // 3)


def bound_text(
    value: Any,
    *,
    max_chars: Any,
) -> str:
    """Whitespace-normalized, hard-capped text. Never wider."""

    text = normalize_whitespace(value)

    if not text:
        return ""

    limit = parse_positive_int(
        max_chars,
        default=len(text),
        cap=len(text),
    )

    return text[:limit]