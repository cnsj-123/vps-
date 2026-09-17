from __future__ import annotations

import json
from typing import Any

from .attachment_ownership import (
    fingertips_signature,
)
from .canonicalizer import (
    _ATTACHMENT_RE,
    _attachment_attrs,
    _classify_attachment,
)
from .models import SegmentKind
from .operit_adapter import OperitAdapter


_adapter = OperitAdapter()


_FEATURES = (
    "id_present",
    "filename_present",
    "id_starts_fingertips",
    "id_contains_fingertips",
    "filename_exact_fingertips_txt",
    "filename_starts_fingertips",
    "id_contains_zhijin",
    "filename_contains_zhijin",
)


def fingertips_ownership_summary_from_body(
    body: bytes,
) -> dict[str, Any] | None:
    """
    Privacy-safe probe.

    Emits counts/booleans only.
    Never emits attachment id, filename or contents.
    """

    try:
        payload = json.loads(body)
    except Exception:
        return None

    if not _adapter.supports(payload):
        return None

    messages = payload.get("messages")

    if not isinstance(messages, list):
        return None

    current_index = None

    if (
        messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "user"
    ):
        current_index = len(messages) - 1

    result: dict[str, Any] = {
        "candidates": 0,
        "history_candidates": 0,
        "current_candidates": 0,
        "features": {
            name: 0
            for name in _FEATURES
        },
    }

    def inspect_text(
        text: str,
        *,
        is_current: bool,
    ) -> None:
        for match in _ATTACHMENT_RE.finditer(text):
            attrs_text = (
                match.group("attrs")
                if match.group("attrs") is not None
                else match.group("self_attrs") or ""
            )

            attrs = _attachment_attrs(attrs_text)

            if (
                _classify_attachment(attrs)
                != SegmentKind.FINGERTIPS
            ):
                continue

            result["candidates"] += 1

            if is_current:
                result["current_candidates"] += 1
            else:
                result["history_candidates"] += 1

            signature = fingertips_signature(attrs)

            for name in _FEATURES:
                if signature[name]:
                    result["features"][name] += 1

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        is_current = index == current_index

        if isinstance(content, str):
            inspect_text(
                content,
                is_current=is_current,
            )
            continue

        if not isinstance(content, list):
            continue

        for block in content:
            if (
                not isinstance(block, dict)
                or block.get("type") != "text"
            ):
                continue

            text = block.get("text")

            if isinstance(text, str):
                inspect_text(
                    text,
                    is_current=is_current,
                )

    return result
