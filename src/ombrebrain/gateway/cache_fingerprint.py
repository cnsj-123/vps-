from __future__ import annotations

import hashlib
import json
from typing import Any


_PROMPT_PARAMETER_KEYS = (
    "tool_choice",
    "thinking",
    "context_management",
    "output_config",
    "output_format",
)


def _digest(value: Any) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")

    return hashlib.sha256(data).hexdigest()[:16]


def _message_prefix_without_cache_control(
    messages: list[Any],
    last_index: int,
) -> list[Any]:
    """
    Return a detached message prefix with only
    content-block cache_control metadata removed.

    Actual message text/content is preserved for hashing
    but is never returned or logged.
    """

    prefix: list[Any] = []

    for message in messages[: last_index + 1]:
        if not isinstance(message, dict):
            prefix.append(message)
            continue

        clean_message = dict(message)
        content = message.get("content")

        if isinstance(content, list):
            clean_content: list[Any] = []

            for block in content:
                if isinstance(block, dict):
                    clean_block = dict(block)
                    clean_block.pop(
                        "cache_control",
                        None,
                    )
                    clean_content.append(clean_block)
                else:
                    clean_content.append(block)

            clean_message["content"] = clean_content

        prefix.append(clean_message)

    return prefix


def _message_prefix_digest(
    messages: Any,
    last_index: int | None,
) -> str | None:
    if (
        not isinstance(messages, list)
        or last_index is None
        or last_index < 0
        or last_index >= len(messages)
    ):
        return None

    return _digest(
        _message_prefix_without_cache_control(
            messages,
            last_index,
        )
    )


def cache_fingerprint_summary_from_body(
    body: bytes,
) -> dict[str, Any] | None:
    """
    Privacy-safe request fingerprint.

    Never returns prompt/tool/system text.
    Only hashes, counts, and structural metadata.
    """

    try:
        payload = json.loads(body)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    tools = payload.get("tools")
    system = payload.get("system")
    messages = payload.get("messages")

    params = {
        key: payload.get(key)
        for key in _PROMPT_PARAMETER_KEYS
        if key in payload
    }

    cache_markers = 0
    boundary_message_index = None

    if isinstance(messages, list):
        for message_index, message in enumerate(
            messages
        ):
            if not isinstance(message, dict):
                continue

            content = message.get("content")

            if not isinstance(content, list):
                continue

            for block in content:
                if (
                    isinstance(block, dict)
                    and isinstance(
                        block.get("cache_control"),
                        dict,
                    )
                ):
                    cache_markers += 1
                    boundary_message_index = (
                        message_index
                    )

    model = payload.get("model")

    parent_boundary_index = None

    if boundary_message_index is not None:
        candidate = boundary_message_index - 2

        if candidate >= 0:
            parent_boundary_index = candidate

    boundary_prefix_sha256 = (
        _message_prefix_digest(
            messages,
            boundary_message_index,
        )
    )

    parent_prefix_sha256 = (
        _message_prefix_digest(
            messages,
            parent_boundary_index,
        )
    )

    return {
        "model_present": isinstance(model, str),
        "model_sha256": (
            _digest(model)
            if isinstance(model, str)
            else None
        ),
        "tools_sha256": _digest(tools),
        "system_sha256": _digest(system),
        "params_sha256": _digest(params),
        "tools_count": (
            len(tools)
            if isinstance(tools, list)
            else 0
        ),
        "messages_count": (
            len(messages)
            if isinstance(messages, list)
            else 0
        ),
        "message_cache_markers": cache_markers,
        "boundary_message_index": (
            boundary_message_index
        ),
        "boundary_prefix_sha256": (
            boundary_prefix_sha256
        ),
        "parent_boundary_index": (
            parent_boundary_index
        ),
        "parent_prefix_sha256": (
            parent_prefix_sha256
        ),
    }
