from __future__ import annotations

import unittest

from ombrebrain.context.live_recall_transport_registry import (
    LIVE_RECALL_TOOL_NAME,
    live_recall_transport_registration_enabled,
    register_live_recall_transport_tool,
    resolve_strict_tool_names,
)


# stdio never passes through the HTTP MCP view middleware, so the
# internal transport-only ``Recall`` tool must not exist in the FastMCP
# registry at all. These tests are the registration regression: they do
# NOT go through the HTTP middleware, they assert the registry itself.


class FakeMCP:
    """A tiny stand-in for the FastMCP registry surface."""

    def __init__(self):
        self.tools: dict[str, object] = {}
        self.tool_calls: list[str] = []

    def tool(self, *, name):
        self.tool_calls.append(name)

        def decorator(handler):
            # Register exactly like FastMCP: first registration wins.
            self.tools.setdefault(name, handler)
            return handler

        return decorator

    def get_tool(self, name):
        return self.tools.get(name)


async def _handler(memref: str) -> str:
    return "unused"


class RegistrationGateTests(unittest.TestCase):

    def test_only_streamable_http_enables_registration(self):
        self.assertTrue(
            live_recall_transport_registration_enabled(
                "streamable-http"
            )
        )
        self.assertTrue(
            live_recall_transport_registration_enabled(
                "  STREAMABLE-HTTP  "
            )
        )

        for transport in (
            "stdio",
            "sse",
            "websocket",
            "",
            None,
            "streamable-http-extra",
        ):
            self.assertFalse(
                live_recall_transport_registration_enabled(
                    transport
                ),
                transport,
            )

    def test_stdio_never_registers_recall(self):
        mcp = FakeMCP()

        registered = register_live_recall_transport_tool(
            mcp,
            _handler,
            transport="stdio",
        )

        self.assertFalse(registered)
        self.assertEqual(mcp.tool_calls, [])
        self.assertEqual(mcp.tools, {})
        self.assertIsNone(
            mcp.get_tool(LIVE_RECALL_TOOL_NAME)
        )

    def test_streamable_http_registers_recall_exactly_once(self):
        mcp = FakeMCP()

        first = register_live_recall_transport_tool(
            mcp,
            _handler,
            transport="streamable-http",
        )

        second = register_live_recall_transport_tool(
            mcp,
            _handler,
            transport="streamable-http",
        )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(mcp.tool_calls, ["Recall"])
        self.assertEqual(
            list(mcp.tools.keys()), ["Recall"]
        )
        self.assertIs(
            mcp.get_tool("Recall"), _handler
        )

    def test_registration_requires_a_callable_handler(self):
        mcp = FakeMCP()

        with self.assertRaises(TypeError):
            register_live_recall_transport_tool(
                mcp,
                "not-callable",
                transport="streamable-http",
            )

        self.assertEqual(mcp.tools, {})


class StrictToolNamesTests(unittest.TestCase):

    _BASE = ("hold", "grow", "trace", "feel", "I")

    def test_stdio_does_not_append_recall(self):
        names = resolve_strict_tool_names(
            self._BASE, transport="stdio"
        )

        self.assertEqual(names, self._BASE)
        self.assertNotIn("Recall", names)

    def test_streamable_http_appends_recall(self):
        names = resolve_strict_tool_names(
            self._BASE,
            transport="streamable-http",
        )

        self.assertEqual(
            names, self._BASE + ("Recall",)
        )

    def test_recall_is_never_duplicated(self):
        names = resolve_strict_tool_names(
            self._BASE + ("Recall",),
            transport="streamable-http",
        )

        self.assertEqual(
            names.count("Recall"), 1
        )

    def test_order_of_base_names_is_preserved(self):
        names = resolve_strict_tool_names(
            self._BASE,
            transport="streamable-http",
        )

        self.assertEqual(
            list(names)[: len(self._BASE)],
            list(self._BASE),
        )


if __name__ == "__main__":
    unittest.main()