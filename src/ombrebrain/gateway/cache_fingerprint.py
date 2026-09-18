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
    }
