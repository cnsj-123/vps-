from __future__ import annotations

from typing import Any

from .canonicalizer import (
    canonicalize_anthropic,
    safe_canonical_summary,
)
from .models import CanonicalRequest


class OperitAdapter:
    """
    Adapter for the Anthropic-compatible Messages request emitted by Operit.
    """

    protocol = "anthropic_messages"

    @staticmethod
    def supports(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False

        messages = payload.get("messages")

        if not isinstance(messages, list):
            return False

        return all(
            isinstance(message, dict)
            for message in messages
        )

    def adapt(
        self,
        payload: dict[str, Any],
    ) -> CanonicalRequest:

        if not self.supports(payload):
            raise ValueError(
                "unsupported Operit request shape"
            )

        return canonicalize_anthropic(payload)

    def safe_summary(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:

        return safe_canonical_summary(
            self.adapt(payload)
        )
