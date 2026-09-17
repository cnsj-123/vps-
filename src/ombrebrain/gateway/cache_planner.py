from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .operit_adapter import OperitAdapter
from .rewriter import rewrite_history_for_cache


_adapter = OperitAdapter()


def _cache_control_info(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    cache_control = value.get("cache_control")

    if not isinstance(cache_control, dict):
        return None

    result: dict[str, Any] = {
        "type": str(cache_control.get("type") or ""),
        "ttl": str(cache_control.get("ttl") or "5m"),
    }

    return result


def _content_cache_controls(
    content: Any,
) -> list[tuple[int, dict[str, Any]]]:
    found: list[tuple[int, dict[str, Any]]] = []

    if not isinstance(content, list):
        return found

    for block_index, block in enumerate(content):
        info = _cache_control_info(block)

        if info is not None:
            found.append(
                (
                    block_index,
                    info,
                )
            )

    return found


def _find_stable_boundary(
    messages: list[Any],
    current_index: int | None,
) -> dict[str, Any] | None:
    """
    Find the last cacheable historical text position before current user.

    Observation only:
    no content is changed and no cache_control is inserted.
    """

    if current_index is None:
        upper = len(messages)
    else:
        upper = current_index

    for message_index in range(
        upper - 1,
        -1,
        -1,
    ):
        message = messages[message_index]

        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "")
        content = message.get("content")

        if isinstance(content, list):
            for block_index in range(
                len(content) - 1,
                -1,
                -1,
            ):
                block = content[block_index]

                if not isinstance(block, dict):
                    continue

                if block.get("type") != "text":
                    continue

                info = _cache_control_info(block)

                return {
                    "message_index": message_index,
                    "block_index": block_index,
                    "role": role,
                    "candidate_kind": "text_block",
                    "requires_normalization": False,
                    "already_cached": info is not None,
                    "existing_ttl": (
                        info["ttl"]
                        if info is not None
                        else None
                    ),
                }

        if isinstance(content, str):
            return {
                "message_index": message_index,
                "block_index": None,
                "role": role,
                "candidate_kind": "string_content",
                "requires_normalization": True,
                "already_cached": False,
                "existing_ttl": None,
            }

    return None


def cache_plan_summary_from_body(
    body: bytes,
) -> dict[str, Any] | None:
    """
    Build a privacy-safe cache plan summary.

    Important:
    - does not mutate request
    - does not emit user/system/attachment text
    - simulates the Phase 2C historical cleanup before planning
    - does not insert cache_control
    """

    try:
        payload = json.loads(body)
    except Exception:
        return None

    if not _adapter.supports(payload):
        return None

    cleaned, rewrite_report = rewrite_history_for_cache(
        payload
    )

    messages = cleaned.get("messages")

    if not isinstance(messages, list):
        return None

    current_index: int | None = None

    if (
        messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "user"
    ):
        current_index = len(messages) - 1

    explicit_locations: list[dict[str, Any]] = []
    ttl_counts: Counter[str] = Counter()

    system = cleaned.get("system")

    if isinstance(system, list):
        for block_index, info in _content_cache_controls(
            system
        ):
            explicit_locations.append(
                {
                    "scope": "system",
                    "message_index": None,
                    "block_index": block_index,
                    "ttl": info["ttl"],
                }
            )
            ttl_counts[info["ttl"]] += 1

    current_cache_control_blocks = 0
    history_cache_control_blocks = 0

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        controls = _content_cache_controls(
            message.get("content")
        )

        for block_index, info in controls:
            explicit_locations.append(
                {
                    "scope": (
                        "current"
                        if message_index == current_index
                        else "history"
                    ),
                    "message_index": message_index,
                    "block_index": block_index,
                    "role": str(
                        message.get("role") or ""
                    ),
                    "ttl": info["ttl"],
                }
            )

            ttl_counts[info["ttl"]] += 1

            if message_index == current_index:
                current_cache_control_blocks += 1
            else:
                history_cache_control_blocks += 1

    top_level_info = _cache_control_info(cleaned)

    boundary = _find_stable_boundary(
        messages,
        current_index,
    )

    warnings: list[str] = []

    explicit_count = len(explicit_locations)

    if explicit_count >= 4:
        warnings.append(
            "explicit_breakpoint_limit_risk"
        )

    if current_cache_control_blocks:
        warnings.append(
            "current_dynamic_tail_has_cache_control"
        )

    if (
        boundary is not None
        and boundary["requires_normalization"]
    ):
        warnings.append(
            "boundary_requires_string_to_block_normalization"
        )

    if boundary is None:
        warnings.append(
            "no_stable_history_boundary"
        )

    return {
        "protocol": "anthropic_messages",
        "strategy": "explicit_history_boundary",
        "messages_total": len(messages),
        "current_user_index": current_index,
        "stable_history_messages": (
            current_index
            if current_index is not None
            else len(messages)
        ),
        "rewrite_simulation": rewrite_report,
        "boundary": boundary,
        "top_level_auto_cache": (
            top_level_info is not None
        ),
        "top_level_ttl": (
            top_level_info["ttl"]
            if top_level_info is not None
            else None
        ),
        "explicit_breakpoints": explicit_count,
        "history_cache_control_blocks": (
            history_cache_control_blocks
        ),
        "current_cache_control_blocks": (
            current_cache_control_blocks
        ),
        "ttl_counts": dict(sorted(ttl_counts.items())),
        "locations": explicit_locations,
        "warnings": warnings,
    }
