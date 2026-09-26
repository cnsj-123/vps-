from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from ombrebrain.context.recall_types import (
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
)


# Live Recall Transport Context.
#
# One MCP HTTP request that authenticated with a valid live recall
# capability is bound to a trusted ``conversation_id`` +
# ``cognitive_request_id`` for the duration of that request only.
#
# The binding lives in a ``ContextVar`` -- never a module-level
# mutable "current conversation id" -- so two concurrent Provider
# requests can never see each other's binding. The middleware sets it
# per request and always resets it in a ``finally``.
#
# Only the trusted internal handler reads this. The model never
# supplies, sees, or influences these values.

_BINDING_KEY = "ombre_live_recall_transport_binding"

_binding: ContextVar[dict[str, str] | None] = (
    ContextVar(_BINDING_KEY, default=None)
)


def _normalize_binding(
    binding: Any,
) -> dict[str, str] | None:
    """Accept only the exact trusted binding shape."""

    if not isinstance(binding, dict):
        return None

    conversation_id = binding.get("conversation_id")

    cognitive_request_id = binding.get(
        "cognitive_request_id"
    )

    if (
        not is_valid_conversation_id(conversation_id)
        or not is_valid_cognitive_request_id(
            cognitive_request_id
        )
    ):
        return None

    return {
        "conversation_id": conversation_id,
        "cognitive_request_id": (
            cognitive_request_id
        ),
    }


def bind_live_recall_transport(
    binding: Any,
) -> Token | None:
    """Bind the current request. Returns a reset token.

    An invalid binding is refused and nothing is set, so a malformed
    scope can never authorize a Recall.
    """

    normalized = _normalize_binding(binding)

    if normalized is None:
        return None

    return _binding.set(normalized)


def reset_live_recall_transport(
    token: Token | None,
) -> None:
    """Undo a binding. Must be called in a ``finally``."""

    if token is None:
        return

    try:
        _binding.reset(token)
    except (ValueError, RuntimeError):
        # Resetting a token from another context is never fatal for
        # the request; the per-request scope dies with the task.
        pass


def current_live_recall_transport_binding(
) -> dict[str, str] | None:
    """The trusted binding of THIS request, or None."""

    return _normalize_binding(_binding.get())