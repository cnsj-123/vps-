from __future__ import annotations

from typing import Any, Callable, Iterable


# Live Recall transport registry wiring.
#
# ``Recall`` is a provider-bound, transport-only tool. It only exists
# because the Anthropic Messages MCP Connector calls back into the
# HTTP ``/mcp`` connector, where ``LiveRecallMCPView`` can hide it from
# every normal MCP view.
#
# stdio does NOT pass through any HTTP middleware, so a stdio
# ``tools/list`` would expose ``Recall`` directly. Therefore the tool
# is registered ONLY when the active transport is ``streamable-http``:
# under stdio it does not exist in the FastMCP registry at all -- there
# is deliberately no "unavailable" shell.
#
# This module is a tiny, side-effect-free helper so the decision can be
# unit-tested without importing the side-effectful ``server`` module.
# It never creates a second FastMCP instance or a second connector.

LIVE_RECALL_TOOL_NAME = "Recall"

_HTTP_TRANSPORT = "streamable-http"


def live_recall_transport_registration_enabled(
    transport: Any,
) -> bool:
    """True only for the HTTP connector that has the MCP view middleware."""

    return (
        str(transport or "").strip().lower()
        == _HTTP_TRANSPORT
    )


def register_live_recall_transport_tool(
    mcp: Any,
    handler: Callable[..., Any],
    *,
    transport: Any,
) -> bool:
    """Register ``Recall`` on the existing registry, HTTP-only.

    Returns True when the tool was registered by THIS call. Under stdio
    nothing is registered. Registering twice on the same server is a
    no-op, so the registry can never contain two ``Recall`` entries.
    """

    if not live_recall_transport_registration_enabled(
        transport
    ):
        return False

    if not callable(handler):
        raise TypeError(
            "live recall transport handler must be callable"
        )

    if getattr(
        mcp, "_ombre_live_recall_registered", False
    ):
        return False

    mcp.tool(name=LIVE_RECALL_TOOL_NAME)(handler)

    try:
        setattr(
            mcp, "_ombre_live_recall_registered", True
        )
    except Exception:
        # A server object that forbids attributes still got the tool;
        # duplicate protection degrades to at-most-one here.
        pass

    return True


def resolve_strict_tool_names(
    base_names: Iterable[str],
    *,
    transport: Any,
) -> tuple[str, ...]:
    """The strict-argument tool names for the active transport.

    ``Recall`` is appended only for streamable-http, so stdio never
    tries to strict-adapt (or warn about) a tool that does not exist.
    """

    names = list(base_names)

    if (
        live_recall_transport_registration_enabled(
            transport
        )
        and LIVE_RECALL_TOOL_NAME not in names
    ):
        names.append(LIVE_RECALL_TOOL_NAME)

    return tuple(names)


__all__ = [
    "LIVE_RECALL_TOOL_NAME",
    "live_recall_transport_registration_enabled",
    "register_live_recall_transport_tool",
    "resolve_strict_tool_names",
]