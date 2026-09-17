from __future__ import annotations

import json
from typing import Any

from .operit_adapter import OperitAdapter
from .rewriter import rewrite_history_for_cache


_adapter = OperitAdapter()


def rewrite_body_for_cache(
    body: bytes,
) -> tuple[bytes, dict[str, Any]]:
    """
    Rewrite an Operit Anthropic-compatible request for cache stability.

    Fail-open contract:
    - invalid JSON -> original bytes
    - unsupported payload -> original bytes
    - no removable historical dynamic context -> original bytes
    - current user message remains untouched
    - unknown/front-end-memory attachments remain untouched
    """

    base_report: dict[str, Any] = {
        "applied": False,
        "history_user_messages_seen": 0,
        "dynamic_context_removed": 0,
        "fingertips_removed": 0,
        "original_bytes": len(body),
        "sent_bytes": len(body),
    }

    try:
        payload = json.loads(body)
    except Exception:
        return body, base_report

    if not _adapter.supports(payload):
        return body, base_report

    rewritten, report = rewrite_history_for_cache(
        payload
    )

    removed = (
        report["dynamic_context_removed"]
        + report["fingertips_removed"]
    )

    merged_report = {
        **base_report,
        **report,
    }

    # Preserve the exact original bytes if there is nothing to remove.
    if removed == 0:
        return body, merged_report

    rewritten_body = json.dumps(
        rewritten,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    merged_report.update(
        {
            "applied": True,
            "sent_bytes": len(rewritten_body),
        }
    )

    return rewritten_body, merged_report
