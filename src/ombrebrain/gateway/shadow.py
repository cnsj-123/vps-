from __future__ import annotations

import json
from typing import Any

from .operit_adapter import OperitAdapter
from .rewriter import rewrite_history_for_cache


_adapter = OperitAdapter()


def rewrite_shadow_summary_from_body(
    body: bytes,
) -> dict[str, Any] | None:
    """
    Simulate historical cache rewriting without changing the request body.

    Privacy:
    - no prompt text is returned
    - no attachment content is returned
    - no hashes of user text are returned
    - only counts and byte sizes are returned

    Byte comparison uses normalized JSON on both sides so formatting
    differences in the incoming body do not create fake savings.
    """

    try:
        payload = json.loads(body)
    except Exception:
        return None

    if not _adapter.supports(payload):
        return None

    rewritten, report = rewrite_history_for_cache(
        payload
    )

    normalized_original = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    normalized_rewritten = json.dumps(
        rewritten,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    before = len(normalized_original)
    after = len(normalized_rewritten)

    return {
        **report,
        "raw_body_bytes": len(body),
        "normalized_original_bytes": before,
        "rewritten_bytes": after,
        "bytes_removed": max(0, before - after),
    }
