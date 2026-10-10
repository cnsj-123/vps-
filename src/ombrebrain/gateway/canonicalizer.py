from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any

from .models import (
    CanonicalBlock,
    CanonicalMessage,
    CanonicalRequest,
    SegmentKind,
)


# Operit serializes attachments as:
#
# <attachment id="..." filename="..." type="..." size="...">
#   ...
# </attachment>
#
# Self-closing attachment references are accepted as well.
_ATTACHMENT_RE = re.compile(
    r'''
    <attachment\b
        (?P<attrs>[^>]*)
    >
        (?P<body>.*?)
    </attachment>
    |
    <attachment\b
        (?P<self_attrs>[^>]*)
    />
    ''',
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

_ATTR_RE = re.compile(
    r'\b(?P<name>id|filename|type|size)\s*=\s*"(?P<value>[^"]*)"',
    re.IGNORECASE,
)


# Current Operit message_insert package.
_OPERIT_EXTRA_ID_PREFIX = "message_insert_extra_bundle_"
_OPERIT_EXTRA_FILENAME_PREFIX = "Time:"

# A separate front-end memory attachment. This is the ONLY filename
# that may ever be classified / extracted as Frontend Memory.
_FRONTEND_MEMORY_FILENAME = "relevant_memories.txt"

# Older message_insert attachment forms kept for compatibility.
_LEGACY_EXTRA_FILENAMES = {
    "extra_info_time.txt",
    "extra_info_battery.txt",
    "extra_info_weather.txt",
    "extra_info_location.txt",
    "extra_info_notifications.txt",
}

_LEGACY_EXTRA_ID_PREFIXES = (
    "message_insert_extra_time_",
    "message_insert_extra_battery_",
    "message_insert_extra_weather_",
    "message_insert_extra_location_",
    "message_insert_extra_notifications_",
)


def _attachment_attrs(raw_attrs: str) -> dict[str, str]:
    attrs: dict[str, str] = {}

    for match in _ATTR_RE.finditer(raw_attrs):
        attrs[match.group("name").lower()] = match.group("value")

    return attrs


def _classify_attachment(attrs: dict[str, str]) -> SegmentKind:
    attachment_id = attrs.get("id", "")
    filename = attrs.get("filename", "")

    attachment_id_lower = attachment_id.lower()
    filename_lower = filename.lower()

    # Strong Operit-owned identifiers. These are safer than scanning
    # attachment contents for words such as Time / memory.
    if (
        filename.startswith(_OPERIT_EXTRA_FILENAME_PREFIX)
        or attachment_id.startswith(_OPERIT_EXTRA_ID_PREFIX)
        or filename_lower in _LEGACY_EXTRA_FILENAMES
        or any(
            attachment_id.startswith(prefix)
            for prefix in _LEGACY_EXTRA_ID_PREFIXES
        )
    ):
        return SegmentKind.DYNAMIC_CONTEXT

    # A separate front-end memory attachment, if one exists.
    if filename_lower == _FRONTEND_MEMORY_FILENAME:
        return SegmentKind.FRONTEND_MEMORY

    # Keep this conservative: only attachment metadata may trigger
    # Fingertips classification, never normal user text.
    if (
        "fingertips" in filename_lower
        or "fingertips" in attachment_id_lower
        or "指尖" in filename
        or "指尖" in attachment_id
    ):
        return SegmentKind.FINGERTIPS

    return SegmentKind.UNKNOWN


def split_operit_text(
    text: str,
    *,
    source_message_index: int,
    source_block_index: int,
    metadata: dict[str, Any] | None = None,
) -> tuple[CanonicalBlock, ...]:
    """
    Split only on explicit Operit attachment envelopes.

    Critical safety rule:
    ordinary text is never reclassified merely because it contains words such
    as Time, battery, memory, Fingertips, etc.

    Phase 2B remains observation-only; source_payload is still untouched.
    """

    block_metadata = deepcopy(metadata or {})

    if not text:
        return (
            CanonicalBlock(
                kind=SegmentKind.USER_TEXT,
                source_message_index=source_message_index,
                source_block_index=source_block_index,
                text=text,
                metadata=block_metadata,
            ),
        )

    blocks: list[CanonicalBlock] = []
    cursor = 0

    for match in _ATTACHMENT_RE.finditer(text):
        plain = text[cursor:match.start()]

        # Ignore separator-only whitespace in the canonical view.
        # Nothing is deleted from source_payload.
        if plain.strip():
            blocks.append(
                CanonicalBlock(
                    kind=SegmentKind.USER_TEXT,
                    source_message_index=source_message_index,
                    source_block_index=source_block_index,
                    text=plain,
                    metadata=deepcopy(block_metadata),
                )
            )

        raw_attachment = match.group(0)

        attrs_text = (
            match.group("attrs")
            if match.group("attrs") is not None
            else match.group("self_attrs") or ""
        )

        attrs = _attachment_attrs(attrs_text)
        kind = _classify_attachment(attrs)

        blocks.append(
            CanonicalBlock(
                kind=kind,
                source_message_index=source_message_index,
                source_block_index=source_block_index,
                text=raw_attachment,
                metadata=deepcopy(block_metadata),
            )
        )

        cursor = match.end()

    tail = text[cursor:]

    if tail.strip():
        blocks.append(
            CanonicalBlock(
                kind=SegmentKind.USER_TEXT,
                source_message_index=source_message_index,
                source_block_index=source_block_index,
                text=tail,
                metadata=deepcopy(block_metadata),
            )
        )

    if not blocks:
        blocks.append(
            CanonicalBlock(
                kind=SegmentKind.USER_TEXT,
                source_message_index=source_message_index,
                source_block_index=source_block_index,
                text=text,
                metadata=block_metadata,
            )
        )

    return tuple(blocks)


@dataclass(frozen=True)
class FrontendMemorySegment:
    """One explicitly identified Frontend Memory attachment body.

    Only an ``<attachment filename="relevant_memories.txt">`` envelope
    produces a segment. ``body`` is the exact inner text of the
    attachment (no trimming, no normalization). ``placement`` is the
    canonical position in the request: ``"current"`` for the current
    user message, ``"history"`` otherwise.
    """

    placement: str
    source_message_index: int
    source_block_index: int
    body: str


def _parse_single_attachment(
    raw: Any,
) -> tuple[str, str] | None:
    """Parse exactly one attachment envelope, or return None.

    Fail-safe by construction: anything that is not a single, fully
    matched attachment envelope returns None. The attachment grammar
    is never duplicated -- this reuses the same ``_ATTACHMENT_RE`` /
    ``_ATTR_RE`` the canonical splitter uses.
    """

    if not isinstance(raw, str):
        return None

    match = _ATTACHMENT_RE.fullmatch(raw)

    if match is None:
        return None

    attrs_text = match.group("attrs")

    if attrs_text is None:
        attrs_text = match.group("self_attrs") or ""

    attrs = _attachment_attrs(attrs_text)

    body = match.group("body")

    return attrs.get("filename", ""), (
        body if body is not None else ""
    )


def extract_frontend_memory_segments(
    request: CanonicalRequest,
) -> tuple[FrontendMemorySegment, ...]:
    """Public, fail-safe accessor for canonical Frontend Memory.

    This is the single supported way for another package to read
    Frontend Memory structure from a canonical request. It does NOT
    expose the private attachment grammar and does NOT reclassify
    ordinary text: a segment is returned only when the canonical
    splitter already tagged the block ``SegmentKind.FRONTEND_MEMORY``
    AND the envelope's own filename is exactly
    ``relevant_memories.txt``.

    Malformed / unknown / self-closing / empty attachments are
    skipped, so a broken attachment can never be treated as trusted
    Frontend Memory.
    """

    segments: list[FrontendMemorySegment] = []

    messages = list(request.history)

    if request.current is not None:
        messages.append(request.current)

    for message in messages:
        placement = (
            "current"
            if message is request.current
            else "history"
        )

        for block in message.blocks:
            if (
                block.kind
                is not SegmentKind.FRONTEND_MEMORY
            ):
                continue

            parsed = _parse_single_attachment(
                block.text
            )

            if parsed is None:
                continue

            filename, body = parsed

            if (
                filename.lower()
                != _FRONTEND_MEMORY_FILENAME
            ):
                # Defensive: classification already required this.
                continue

            if not body.strip():
                continue

            segments.append(
                FrontendMemorySegment(
                    placement=placement,
                    source_message_index=(
                        block.source_message_index
                    ),
                    source_block_index=(
                        block.source_block_index
                    ),
                    body=body,
                )
            )

    return tuple(segments)


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
    Structure/counts only.

    No user text, attachment payload, system text, memory text or Fingertips
    text is emitted.
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
