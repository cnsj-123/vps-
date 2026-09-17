from __future__ import annotations

import json
from typing import Any

from .cache_breakpoint import (
    relocate_current_breakpoint,
)
from .operit_adapter import OperitAdapter
from .rewriter import rewrite_history_for_cache


_adapter = OperitAdapter()


def cache_move_shadow_summary_from_body(
    body: bytes,
) -> dict[str, Any] | None:
    """
    Simulate:
      Phase 2C historical cleanup
      +
      Phase 3B current breakpoint relocation

    No prompt text is returned.
    No request mutation is exposed to the caller.
    """

    try:
        payload = json.loads(body)
    except Exception:
        return None

    if not _adapter.supports(payload):
        return None

    cleaned, rewrite_report = (
        rewrite_history_for_cache(payload)
    )

    transformed, move_report = (
        relocate_current_breakpoint(cleaned)
    )

    normalized_original = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    normalized_transformed = json.dumps(
        transformed,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return {
        "rewrite": rewrite_report,
        "breakpoint_move": move_report,
        "normalized_original_bytes": len(
            normalized_original
        ),
        "normalized_transformed_bytes": len(
            normalized_transformed
        ),
    }
