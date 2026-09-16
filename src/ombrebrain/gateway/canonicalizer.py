from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re
from typing import Any

from .models import (
    CanonicalBlock,
    CanonicalMessage,
    CanonicalRequest,
    SegmentKind,
)


_MEMORY_MARKER = re.compile(
    r"(?:relevant[_ -]?memories(?:\.txt)?|相关记忆|相关记忆文件)",
    re.IGNORECASE,
)

_FINGERTIPS_MARKER = re.compile(
    r"(?:\bfingertips\b|指尖语气)",
    re.IGNORECASE,
)

_PERCEPTION_MARKER = re.compile(
    r"^\s*(?:time|时间|当前时间)\s*[:：]",
    re.IGNORECASE,
)


def _marker_kind(line: str) -> SegmentKind | None:
    stripped = line.strip()

    if not stripped:
        return None

    if _MEMORY_MARKER.search(stripped):
        return SegmentKind.FRONTEND_MEMORY

    if _FINGERTIPS_MARKER.search(stripped):
        return SegmentKind.FINGERTIPS

    if _PERCEPTION_MARKER.search(stripped):
        return SegmentKind.PERCEPTION

    return None


def split_operit_text(
    text: str,
    *,
    source_message_index: int,
    source_block_index: int,
    metadata: dict[str, Any] | None = None,
) -> tuple[CanonicalBlock, ...]:
    """
    Conservatively split an Operit text block into semantic sections.

    Important:
    - Nothing is deleted.
    - Nothing is rewritten.
    - Unrecognized text stays USER_TEXT.
    - Only explicit marker lines start dynamic sections.
    """

    if not text:
        return (
            CanonicalBlock(
                kind=SegmentKind.USER_TEXT,
                source_message_index=source_message_index,
                source_block_index=source_block_index,
                text=text,
                metadata=deepcopy(metadata or {}),
            ),
        )

    lines = text.splitlines(keepends=True)

    chunks: list[tuple[SegmentKind, str]] = []

    current_kind = SegmentKind.USER_TEXT
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines

        if not current_lines:
            return

        chunks.append(
            (
                current_kind,
                "".join(current_lines),
            )
        )

        current_lines = []

    for line in lines:
        marker = _marker_kind(line)

        if marker is not None and marker != current_kind:
            flush()
            current_kind = marker

        current_lines.append(line)

    flush()

    if not chunks:
        chunks = [
            (
                SegmentKind.USER_TEXT,
                text,
            )
        ]

    return tuple(
        CanonicalBlock(
            kind=kind,
            source_message_index=source_message_index,
            source_block_index=source_block_index,
            text=chunk,
            metadata=deepcopy(metadata or {}),
        )
        for kind, chunk in chunks
    )


def _canonicalize_message(
    message: dict[str, Any],
    index: int,
) -> CanonicalMessage:

    role = str(message.get("role") or "")
    content = message.get("content")

    blocks: list[CanonicalBlock] = []

    if isinstance(content, str):
        blocks.extend(
            split_operit_text(
                content,
                source_message_index=index,
                source_block_index=0,
            )
        )

    elif isinstance(content, list):
        for block_index, block in enumerate(content):

            if (
                isinstance(block, dict)
                and block.get("type") == "text"
            ):
                metadata = {
                    key: deepcopy(value)
                    for key, value in block.items()
                    if key not in {
                        "type",
                        "text",
                    }
                }

                blocks.extend(
                    split_operit_text(
                        str(block.get("text") or ""),
                        source_message_index=index,
                        source_block_index=block_index,
                        metadata=metadata,
                    )
                )

            else:
                blocks.append(
                    CanonicalBlock(
                        kind=SegmentKind.UNKNOWN,
                        source_message_index=index,
                        source_block_index=block_index,
                        raw=deepcopy(block),
                    )
                )

    else:
        blocks.append(
            CanonicalBlock(
                kind=SegmentKind.UNKNOWN,
                source_message_index=index,
                source_block_index=0,
                raw=deepcopy(content),
            )
        )

    return CanonicalMessage(
        role=role,
        original_index=index,
        blocks=tuple(blocks),
    )


def canonicalize_anthropic(
    payload: dict[str, Any],
) -> CanonicalRequest:

    messages_raw = payload.get("messages")

    if not isinstance(messages_raw, list):
        raise ValueError("messages must be a list")

    messages: list[CanonicalMessage] = []

    for index, message in enumerate(messages_raw):

        if not isinstance(message, dict):
            raise ValueError(
                f"messages[{index}] must be an object"
            )

        messages.append(
            _canonicalize_message(
                message,
                index,
            )
        )

    current: CanonicalMessage | None = None

    if messages and messages[-1].role == "user":
        current = messages[-1]
        history = tuple(messages[:-1])
    else:
        history = tuple(messages)

    passthrough = {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {
            "system",
            "messages",
        }
    }

    return CanonicalRequest(
        protocol="anthropic_messages",
        system=deepcopy(payload.get("system")),
        history=history,
        current=current,
        passthrough=passthrough,
        source_payload=deepcopy(payload),
    )


def safe_canonical_summary(
    request: CanonicalRequest,
) -> dict[str, Any]:
    """
    Return structure/counts only.

    No user text, system text, memory text or Fingertips text is returned.
    """

    history_counts: Counter[str] = Counter()

    for message in request.history:
        history_counts.update(
            block.kind.value
            for block in message.blocks
        )

    current_counts: Counter[str] = Counter()

    if request.current is not None:
        current_counts.update(
            block.kind.value
            for block in request.current.blocks
        )

    history_user_layout = [
        {
            "index": message.original_index,
            "segments": [
                block.kind.value
                for block in message.blocks
            ],
        }
        for message in request.history
        if message.role == "user"
    ]

    current_layout = None

    if request.current is not None:
        current_layout = {
            "index": request.current.original_index,
            "role": request.current.role,
            "segments": [
                block.kind.value
                for block in request.current.blocks
            ],
        }

    return {
        "protocol": request.protocol,
        "history_messages": len(request.history),
        "current_user_present": request.current is not None,
        "history_segments": dict(
            sorted(history_counts.items())
        ),
        "current_segments": dict(
            sorted(current_counts.items())
        ),
        "history_user_layout": history_user_layout,
        "current_layout": current_layout,
        "passthrough_keys": sorted(
            request.passthrough.keys()
        ),
        "system_present": request.system is not None,
    }
