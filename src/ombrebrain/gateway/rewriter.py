from __future__ import annotations

from copy import deepcopy
from typing import Any

from .canonicalizer import (
    _ATTACHMENT_RE,
    _attachment_attrs,
    _classify_attachment,
)
from .models import SegmentKind


_HISTORY_DROP_KINDS = {
    SegmentKind.DYNAMIC_CONTEXT,
    SegmentKind.FINGERTIPS,
}


def _rewrite_historical_text(
    text: str,
) -> tuple[str, dict[str, int]]:
    """
    Remove only strongly identified dynamic attachments from historical
    user text.

    Ordinary text is never interpreted or rewritten.
    Unknown and frontend-memory attachments are preserved verbatim.
    """

    counts = {
        "dynamic_context_removed": 0,
        "fingertips_removed": 0,
    }

    def replace(match):
        attrs_text = (
            match.group("attrs")
            if match.group("attrs") is not None
            else match.group("self_attrs") or ""
        )

        attrs = _attachment_attrs(attrs_text)
        kind = _classify_attachment(attrs)

        if kind == SegmentKind.DYNAMIC_CONTEXT:
            counts["dynamic_context_removed"] += 1
            return ""

        if kind == SegmentKind.FINGERTIPS:
            counts["fingertips_removed"] += 1
            return ""

        return match.group(0)

    rewritten = _ATTACHMENT_RE.sub(
        replace,
        text,
    )

    return rewritten, counts


def rewrite_history_for_cache(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    Produce a cache-stable copy of an Anthropic-compatible request.

    Safety contract:
    - input payload is never mutated
    - assistant messages are untouched
    - current user message is untouched
    - historical ordinary user text is untouched
    - frontend_memory attachments are preserved
    - unknown attachments are preserved
    - only strongly identified historical dynamic_context / fingertips
      attachments are removed
    """

    output = deepcopy(payload)

    report = {
        "history_user_messages_seen": 0,
        "dynamic_context_removed": 0,
        "fingertips_removed": 0,
    }

    messages = output.get("messages")

    if not isinstance(messages, list):
        return output, report

    current_index = None

    if (
        messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "user"
    ):
        current_index = len(messages) - 1

    for message_index, message in enumerate(messages):

        if message_index == current_index:
            continue

        if not isinstance(message, dict):
            continue

        if message.get("role") != "user":
            continue

        report["history_user_messages_seen"] += 1

        content = message.get("content")

        if isinstance(content, str):
            rewritten, counts = _rewrite_historical_text(
                content
            )

            message["content"] = rewritten

            report["dynamic_context_removed"] += (
                counts["dynamic_context_removed"]
            )
            report["fingertips_removed"] += (
                counts["fingertips_removed"]
            )

            continue

        if not isinstance(content, list):
            continue

        for block in content:

            if not isinstance(block, dict):
                continue

            if block.get("type") != "text":
                continue

            text = block.get("text")

            if not isinstance(text, str):
                continue

            rewritten, counts = _rewrite_historical_text(
                text
            )

            block["text"] = rewritten

            report["dynamic_context_removed"] += (
                counts["dynamic_context_removed"]
            )
            report["fingertips_removed"] += (
                counts["fingertips_removed"]
            )

    return output, report
