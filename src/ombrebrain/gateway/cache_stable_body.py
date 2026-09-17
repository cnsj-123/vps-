from __future__ import annotations

import json
from typing import Any

from .cache_breakpoint import (
    relocate_current_breakpoint,
)
from .operit_adapter import OperitAdapter
from .rewriter import rewrite_history_for_cache


_adapter = OperitAdapter()


def rewrite_cache_stable_body(
    body: bytes,
) -> tuple[bytes, dict[str, Any]]:
    """
    Atomically build the cache-stable Operit request.

    Pipeline:
      1. strip confirmed historical dynamic attachments
      2. move the current-user cache breakpoint to the last
         stable historical text block

    Fail-open:
      - invalid JSON -> exact original bytes
      - unsupported request -> exact original bytes
      - breakpoint migration cannot be proven safe ->
        exact original bytes
      - unexpected exception should be handled by the
        gateway and also fail open

    The current user message text/content is not changed.
    Frontend-memory and unknown attachments are preserved.
    """

    base_report: dict[str, Any] = {
        "applied": False,
        "reason": "not_applied",
        "original_bytes": len(body),
        "sent_bytes": len(body),
        "rewrite": None,
        "breakpoint_move": None,
    }

    try:
        payload = json.loads(body)
    except Exception:
        base_report["reason"] = "non_json"
        return body, base_report

    if not _adapter.supports(payload):
        base_report["reason"] = "unsupported"
        return body, base_report

    cleaned, rewrite_report = (
        rewrite_history_for_cache(payload)
    )

    transformed, move_report = (
        relocate_current_breakpoint(cleaned)
    )

    base_report["rewrite"] = rewrite_report
    base_report["breakpoint_move"] = move_report

    # Atomic safety rule:
    # if the breakpoint move is not unambiguously safe,
    # send the exact original request.
    if not move_report.get("applied"):
        base_report["reason"] = (
            "breakpoint_move_not_applied"
        )
        return body, base_report

    transformed_body = json.dumps(
        transformed,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    base_report.update(
        {
            "applied": True,
            "reason": "cache_stable",
            "sent_bytes": len(transformed_body),
        }
    )

    return transformed_body, base_report
