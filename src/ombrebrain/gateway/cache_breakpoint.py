from __future__ import annotations

from copy import deepcopy
from typing import Any

from .cache_planner import _find_stable_boundary


def _explicit_breakpoint_count(
    payload: dict[str, Any],
) -> int:
    count = 0

    tools = payload.get("tools")

    if isinstance(tools, list):
        for tool in tools:
            if (
                isinstance(tool, dict)
                and isinstance(
                    tool.get("cache_control"),
                    dict,
                )
            ):
                count += 1

    system = payload.get("system")

    if isinstance(system, list):
        for block in system:
            if (
                isinstance(block, dict)
                and isinstance(
                    block.get("cache_control"),
                    dict,
                )
            ):
                count += 1

    messages = payload.get("messages")

    if isinstance(messages, list):
        for message in messages:
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
                    count += 1

    return count


def relocate_current_breakpoint(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Move exactly one current-user block cache_control to the last
    stable historical text block.

    Safety:
    - input never mutated
    - system/tools/history content untouched
    - current text untouched
    - only cache_control metadata is moved
    - ambiguous shapes fail open
    """

    output = deepcopy(payload)

    report: dict[str, Any] = {
        "applied": False,
        "reason": "not_applied",
        "current_user_index": None,
        "source_block_index": None,
        "boundary_message_index": None,
        "boundary_block_index": None,
        "moved_ttl": None,
        "explicit_before": _explicit_breakpoint_count(
            output
        ),
        "explicit_after": None,
    }

    messages = output.get("messages")

    if not isinstance(messages, list) or not messages:
        report["reason"] = "no_messages"
        report["explicit_after"] = report["explicit_before"]
        return output, report

    current_index = len(messages) - 1
    current = messages[current_index]

    if (
        not isinstance(current, dict)
        or current.get("role") != "user"
    ):
        report["reason"] = "no_current_user"
        report["explicit_after"] = report["explicit_before"]
        return output, report

    report["current_user_index"] = current_index

    content = current.get("content")

    if not isinstance(content, list):
        report["reason"] = "current_content_not_blocks"
        report["explicit_after"] = report["explicit_before"]
        return output, report

    current_markers: list[
        tuple[int, dict[str, Any]]
    ] = []

    for block_index, block in enumerate(content):
        if not isinstance(block, dict):
            continue

        cache_control = block.get("cache_control")

        if isinstance(cache_control, dict):
            current_markers.append(
                (
                    block_index,
                    deepcopy(cache_control),
                )
            )

    if len(current_markers) != 1:
        report["reason"] = (
            "current_cache_control_count_not_one"
        )
        report["explicit_after"] = report["explicit_before"]
        return output, report

    source_block_index, cache_control = (
        current_markers[0]
    )

    source_block = content[source_block_index]

    if source_block.get("type") != "text":
        report["reason"] = "current_marker_not_text"
        report["explicit_after"] = report["explicit_before"]
        return output, report

    report["source_block_index"] = source_block_index
    report["moved_ttl"] = str(
        cache_control.get("ttl") or "5m"
    )

    boundary = _find_stable_boundary(
        messages,
        current_index,
    )

    if boundary is None:
        report["reason"] = "no_stable_boundary"
        report["explicit_after"] = report["explicit_before"]
        return output, report

    if boundary.get("requires_normalization"):
        report["reason"] = (
            "boundary_requires_normalization"
        )
        report["explicit_after"] = report["explicit_before"]
        return output, report

    boundary_message_index = boundary.get(
        "message_index"
    )
    boundary_block_index = boundary.get(
        "block_index"
    )

    if (
        not isinstance(boundary_message_index, int)
        or not isinstance(boundary_block_index, int)
    ):
        report["reason"] = "invalid_boundary"
        report["explicit_after"] = report["explicit_before"]
        return output, report

    boundary_message = messages[
        boundary_message_index
    ]

    if not isinstance(boundary_message, dict):
        report["reason"] = "invalid_boundary_message"
        report["explicit_after"] = report["explicit_before"]
        return output, report

    boundary_content = boundary_message.get(
        "content"
    )

    if not isinstance(boundary_content, list):
        report["reason"] = "invalid_boundary_content"
        report["explicit_after"] = report["explicit_before"]
        return output, report

    if not (
        0
        <= boundary_block_index
        < len(boundary_content)
    ):
        report["reason"] = "invalid_boundary_block"
        report["explicit_after"] = report["explicit_before"]
        return output, report

    boundary_block = boundary_content[
        boundary_block_index
    ]

    if not isinstance(boundary_block, dict):
        report["reason"] = "invalid_boundary_block"
        report["explicit_after"] = report["explicit_before"]
        return output, report

    existing = boundary_block.get(
        "cache_control"
    )

    if (
        existing is not None
        and existing != cache_control
    ):
        report["reason"] = (
            "boundary_cache_control_conflict"
        )
        report["explicit_after"] = report["explicit_before"]
        return output, report

    # Everything is validated before mutation.
    source_block.pop(
        "cache_control",
        None,
    )

    boundary_block["cache_control"] = deepcopy(
        cache_control
    )

    report.update(
        {
            "applied": True,
            "reason": "moved",
            "boundary_message_index": (
                boundary_message_index
            ),
            "boundary_block_index": (
                boundary_block_index
            ),
            "explicit_after": (
                _explicit_breakpoint_count(output)
            ),
        }
    )

    return output, report
