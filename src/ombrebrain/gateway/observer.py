from __future__ import annotations

import json
from typing import Any

from .operit_adapter import OperitAdapter


_adapter = OperitAdapter()


def canonical_summary_from_body(
    body: bytes,
) -> dict[str, Any] | None:
    """
    Build a privacy-safe canonical summary from a JSON request body.

    Returns None for:
    - non-JSON bodies
    - unsupported request shapes
    - malformed Operit requests

    This function never mutates body or payload.
    """

    try:
        payload = json.loads(body)
    except Exception:
        return None

    if not _adapter.supports(payload):
        return None

    try:
        return _adapter.safe_summary(payload)
    except Exception:
        return None
