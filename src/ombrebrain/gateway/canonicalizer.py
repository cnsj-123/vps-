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
    """One explicitly identified Frontend Memory attachment.

    Only an ``<attachment filename="relevant_memories.txt">`` envelope
    produces a segment. This is an *attachment-level* view: the
    repository has no proven item-level schema for the inner body, so
    one attachment must never be assumed to be one memory.

    ``body`` is the exact inner text (no trimming, no normalization).
    ``raw_attachment`` is the exact canonical attachment text as it
    appears in the request -- envelope, metadata and body -- so callers
    can measure the real model-visible footprint without reconstructing
    tags. It exists for in-memory measurement only and must never be
    persisted or logged by callers. ``placement`` is the canonical
    position in the request: ``"current"`` for the current user
    message, ``"history"`` otherwise.
    """

    placement: str
    source_message_index: int
    source_block_index: int
    body: str
    raw_attachment: str


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

    It returns one entry per explicit attachment. It does NOT parse,
    split or interpret the inner body: no proven item-level Frontend
    Memory schema exists, so nothing here claims an attachment equals
    one memory.

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

            # ``block.text`` is the exact canonical attachment text;
            # it is only read if it is a real string.
            raw_attachment = block.text

            if not isinstance(raw_attachment, str):
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
                    raw_attachment=raw_attachment,
                )
            )

    return tuple(segments)


# ---------------------------------------------------------------------
# Canonical Frontend attachment control transform (public, pure).
#
# This is the ONE enforcement primitive a future memory authority would
# use to control Frontend Memory attachment occurrences WITHOUT falling
# back to Frontend Flash / Recall / item parsing: it rewrites only the
# explicit ``relevant_memories.txt`` attachment envelopes and leaves
# every other byte of the request untouched.
#
# The low-level actions are mechanical rewrite primitives, NOT high-level
# Memory policy:
#
#     keep_exact : the exact raw attachment envelope survives unchanged.
#     drop       : the exact raw attachment envelope is removed.
#
# v0 production code emits ``keep_exact`` ONLY. ``drop`` exists so the
# transform is a genuine enforcement primitive (and can be unit-tested),
# not so a live DROP policy can be run.
#
# It reuses the SAME attachment grammar / classification / single-envelope
# parser the canonical splitter uses -- there is deliberately no second
# attachment parser.
# ---------------------------------------------------------------------

REWRITE_KEEP_EXACT = "keep_exact"
REWRITE_DROP = "drop"

_KNOWN_REWRITE_ACTIONS = (
    REWRITE_KEEP_EXACT,
    REWRITE_DROP,
)

_REWRITE_PAYLOAD_UNSUPPORTED = "payload_unsupported"
_REWRITE_OCCURRENCE_COUNT_MISMATCH = (
    "occurrence_count_mismatch"
)
_REWRITE_UNKNOWN_ACTION = "unknown_rewrite_action"
_REWRITE_REFUSED = "rewrite_refused"


@dataclass(frozen=True)
class FrontendRewriteResult:
    """The fail-closed outcome of the canonical control transform.

    ``payload`` is the rewritten request (a NEW object; the input object
    is never mutated). It is ``None`` whenever ``ok`` is False: a partial
    rewrite is never returned.
    """

    ok: bool
    reason: str | None
    payload: dict[str, Any] | None
    occurrence_count: int
    kept_count: int
    dropped_count: int


def _frontend_occurrences_in_text(
    text: str,
) -> list[tuple[int, int, str, str]]:
    """(start, end, raw_attachment, body) for every Frontend occurrence.

    Only a genuine ``relevant_memories.txt`` envelope with a non-blank
    body is returned -- exactly the occurrences
    ``extract_frontend_memory_segments`` would count. Malformed /
    self-closing / empty attachments and ordinary user text are never
    treated as controllable Frontend content.
    """

    occurrences: list[tuple[int, int, str, str]] = []

    for match in _ATTACHMENT_RE.finditer(text):
        attrs_text = match.group("attrs")

        if attrs_text is None:
            attrs_text = match.group("self_attrs") or ""

        attrs = _attachment_attrs(attrs_text)

        if (
            _classify_attachment(attrs)
            is not SegmentKind.FRONTEND_MEMORY
        ):
            continue

        body = match.group("body")

        body = body if body is not None else ""

        if not body.strip():
            continue

        occurrences.append(
            (
                match.start(),
                match.end(),
                match.group(0),
                body,
            )
        )

    return occurrences


def _apply_rewrite_actions(
    text: str,
    occurrences: list[tuple[int, int, str, str]],
    actions: list[str],
) -> str:
    """Rebuild ``text`` keeping / dropping exact envelopes.

    Everything that is not a dropped envelope survives verbatim; a kept
    envelope is re-emitted from its exact ``raw_attachment`` slice, never
    reconstructed, so no whitespace / tag normalization can occur.
    """

    pieces: list[str] = []
    cursor = 0

    for (start, end, raw, _body), action in zip(
        occurrences, actions
    ):
        pieces.append(text[cursor:start])

        if action == REWRITE_KEEP_EXACT:
            pieces.append(raw)

        cursor = end

    pieces.append(text[cursor:])

    return "".join(pieces)


def rewrite_frontend_memory_attachments(
    payload: Any,
    decide: Any,
    *,
    expected_occurrences: int | None = None,
) -> FrontendRewriteResult:
    """Pure, fail-closed rewrite of Frontend Memory attachment envelopes.

    ``decide`` is called once per Frontend occurrence, in canonical
    request order, with keyword arguments::

        decide(
            occurrence=<int>,
            placement=<"current" | "history">,
            raw_attachment=<exact envelope>,
            body=<exact inner body>,
        )

    and must return ``keep_exact`` or ``drop``. Any other return value (or
    a refused / raised callback) fails the WHOLE transform.

    The transform never mutates ``payload`` and never returns a partial
    result: on any failure ``ok`` is False, ``reason`` is a closed enum
    and ``payload`` is None. Every non-Frontend field -- system, ordinary
    user text, dynamic-context / Fingertips / unknown attachments, tools,
    params, model, block metadata, non-text blocks, message count and
    roles -- is preserved exactly.
    """

    if not isinstance(payload, dict):
        return FrontendRewriteResult(
            ok=False,
            reason=_REWRITE_PAYLOAD_UNSUPPORTED,
            payload=None,
            occurrence_count=0,
            kept_count=0,
            dropped_count=0,
        )

    messages = payload.get("messages")

    if not isinstance(messages, list) or not all(
        isinstance(message, dict)
        for message in messages
    ):
        return FrontendRewriteResult(
            ok=False,
            reason=_REWRITE_PAYLOAD_UNSUPPORTED,
            payload=None,
            occurrence_count=0,
            kept_count=0,
            dropped_count=0,
        )

    rewritten = deepcopy(payload)

    rewritten_messages = rewritten["messages"]

    last_index = len(rewritten_messages) - 1

    state = {
        "occurrence": 0,
        "kept": 0,
        "dropped": 0,
    }

    failure: dict[str, str] | None = None

    def rewrite_text(
        text: str,
        placement: str,
        *,
        message_index: int,
        block_index: int,
    ) -> str:
        nonlocal failure

        occurrences = _frontend_occurrences_in_text(
            text
        )

        actions: list[str] = []

        for start, end, raw, body in occurrences:
            occurrence = state["occurrence"]

            state["occurrence"] += 1

            try:
                action = decide(
                    occurrence=occurrence,
                    placement=placement,
                    raw_attachment=raw,
                    body=body,
                )
            except Exception:
                failure = {
                    "reason": _REWRITE_REFUSED
                }
                return text

            if action not in _KNOWN_REWRITE_ACTIONS:
                # A refusal (None) and an unknown action are the same
                # fail-closed outcome: no partial rewrite is allowed.
                failure = {
                    "reason": (
                        _REWRITE_UNKNOWN_ACTION
                        if action is not None
                        else _REWRITE_REFUSED
                    )
                }
                return text

            actions.append(action)

        if not actions:
            return text

        state["kept"] += actions.count(
            REWRITE_KEEP_EXACT
        )
        state["dropped"] += actions.count(
            REWRITE_DROP
        )

        return _apply_rewrite_actions(
            text, occurrences, actions
        )

    for index, message in enumerate(
        rewritten_messages
    ):
        if failure is not None:
            break

        placement = (
            "current"
            if (
                index == last_index
                and message.get("role") == "user"
            )
            else "history"
        )

        content = message.get("content")

        if isinstance(content, str):
            message["content"] = rewrite_text(
                content,
                placement,
                message_index=index,
                block_index=0,
            )
            continue

        if not isinstance(content, list):
            continue

        for block_index, block in enumerate(content):
            if failure is not None:
                break

            if not (
                isinstance(block, dict)
                and block.get("type") == "text"
            ):
                continue

            text = block.get("text")

            if not isinstance(text, str):
                continue

            block["text"] = rewrite_text(
                text,
                placement,
                message_index=index,
                block_index=block_index,
            )

    if failure is not None:
        return FrontendRewriteResult(
            ok=False,
            reason=failure["reason"],
            payload=None,
            occurrence_count=state["occurrence"],
            kept_count=0,
            dropped_count=0,
        )

    if (
        expected_occurrences is not None
        and state["occurrence"] != expected_occurrences
    ):
        return FrontendRewriteResult(
            ok=False,
            reason=_REWRITE_OCCURRENCE_COUNT_MISMATCH,
            payload=None,
            occurrence_count=state["occurrence"],
            kept_count=0,
            dropped_count=0,
        )

    return FrontendRewriteResult(
        ok=True,
        reason=None,
        payload=rewritten,
        occurrence_count=state["occurrence"],
        kept_count=state["kept"],
        dropped_count=state["dropped"],
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
