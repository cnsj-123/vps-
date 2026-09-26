"""Live Recall MCP view middleware.

Two MCP views live behind the SAME ``/mcp`` connector:

  - the normal OAuth / static-token view, which must keep seeing the
    frozen public tool set only -- the internal transport tools
    ``Recall`` and ``UseMemory`` are registered in the FastMCP registry
    but are removed from ``tools/list`` and refused on ``tools/call``;
  - the short-lived live transport capability view, which sees ONLY
    the tools its credential really grants:
      * v1 ``obrcap_`` capability -> ``Recall`` only;
      * v2 ``obmtcap_`` capability -> ``Recall`` + ``UseMemory``.

Keeping both views on one connector is deliberate: the repository
retired its second connector (``/mcp-extra``) on purpose, to avoid two
lifecycles, two auth boundaries and two body-limit boundaries.

The v1 capability is NEVER widened: an ``obrcap_`` token can never
list or call ``UseMemory``, even when the Usage Attribution flag is ON.
The allowed tool set is chosen by the server from the trusted
credential KIND (a scope key written by ``MCPAuthMiddleware``), never
from anything the model sends.

This middleware sits on the INNER side of the existing
``MCPRequestBodyLimitMiddleware`` (so the 4MB MCP body limit still
applies before any JSON-RPC parsing here) and on the inner side of
``MCPAuthMiddleware`` (so the trusted capability binding is already on
the scope). It never writes protocol logic into ``server.py``.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from ombrebrain.context.live_recall_transport_context import (
    bind_live_recall_transport,
    reset_live_recall_transport,
)


_Receive = Callable[[], Awaitable[dict]]
_Send = Callable[[dict], Awaitable[None]]


def default_mcp_path_matcher(path: object) -> bool:
    """Match only the single public MCP endpoint, without prefix lookalikes.

    Mirrors ``web.request_limits.is_mcp_endpoint_path``. It is kept
    local so this security middleware has no dependency on the rest of
    the ``web`` package graph; ``build_http_app`` always injects the
    canonical matcher explicitly.
    """

    return str(path or "").rstrip("/") == "/mcp"


LIVE_RECALL_TOOL_NAME = "Recall"

LIVE_MEMORY_USAGE_TOOL_NAME = "UseMemory"

AUTH_KIND_SCOPE_KEY = "ombre_mcp_auth_kind"
BINDING_SCOPE_KEY = "ombre_live_recall_binding"
ALLOWED_TOOLS_SCOPE_KEY = "ombre_mcp_allowed_tools"

CAPABILITY_AUTH_KIND = "live_recall_capability"

MEMORY_CAPABILITY_AUTH_KIND = "live_memory_capability"

# The tools no normal MCP view may ever list or call.
_HIDDEN_TOOL_NAMES = frozenset(
    {
        LIVE_RECALL_TOOL_NAME,
        LIVE_MEMORY_USAGE_TOOL_NAME,
    }
)

# The trusted allowed tool set per credential kind. The v1 capability
# stays Recall-only forever; only v2 may reach UseMemory.
_CAPABILITY_ALLOWED_TOOLS: dict[str, tuple[str, ...]] = {
    CAPABILITY_AUTH_KIND: (LIVE_RECALL_TOOL_NAME,),
    MEMORY_CAPABILITY_AUTH_KIND: (
        LIVE_RECALL_TOOL_NAME,
        LIVE_MEMORY_USAGE_TOOL_NAME,
    ),
}

# The capability view fails closed for every other MCP method:
# resources/*, prompts/*, logging/*, completion/*, ...
_CAPABILITY_ALLOWED_METHODS = frozenset(
    {
        "initialize",
        "notifications/initialized",
        "ping",
        "tools/list",
        "tools/call",
    }
)

_JSON_CONTENT_TYPE = "application/json"

# Generic, privacy-safe JSON-RPC errors. Never a CID, a RID, a token,
# a memref or another tool's argument body.
_ERR_INVALID_REQUEST = -32600
_ERR_METHOD_NOT_FOUND = -32601
_ERR_INTERNAL = -32603


def is_live_recall_capability_scope(
    scope: dict,
) -> bool:
    return (
        scope.get(AUTH_KIND_SCOPE_KEY)
        == CAPABILITY_AUTH_KIND
    )


def is_live_memory_capability_scope(
    scope: dict,
) -> bool:
    return (
        scope.get(AUTH_KIND_SCOPE_KEY)
        == MEMORY_CAPABILITY_AUTH_KIND
    )


def capability_allowed_tools(
    scope: dict,
) -> tuple[str, ...] | None:
    """The trusted allowed tool set of a capability scope.

    Returns None for the normal (non-capability) view. A capability
    scope whose declared tool set contradicts its credential kind
    returns an EMPTY tuple, which fails closed: nothing may be listed
    and nothing may be called. The model can never influence this --
    both values are written by the auth middleware from server-side
    constants.
    """

    kind = scope.get(AUTH_KIND_SCOPE_KEY)

    if not isinstance(kind, str):
        return None

    allowed = _CAPABILITY_ALLOWED_TOOLS.get(kind)

    if allowed is None:
        return None

    declared = scope.get(ALLOWED_TOOLS_SCOPE_KEY)

    if declared is not None and (
        not isinstance(declared, (list, tuple))
        or tuple(declared) != allowed
    ):
        return ()

    return allowed


def _json_rpc_id(value: Any) -> Any:
    if isinstance(value, bool):
        return None

    if isinstance(value, (str, int)):
        return value

    return None


async def _send_json_rpc_error(
    send: _Send,
    rpc_id: Any,
    code: int,
    message: str,
) -> None:
    """A legal MCP JSON-RPC error -- never a 500 traceback."""

    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": _json_rpc_id(rpc_id),
            "error": {
                "code": int(code),
                "message": str(message),
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (
                    b"content-length",
                    str(len(body)).encode("ascii"),
                ),
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        }
    )


def _content_type_of(
    headers: list[tuple[bytes, bytes]],
) -> str:
    for key, value in headers:
        if key.lower() == b"content-type":
            try:
                return value.decode("latin-1").lower()
            except Exception:
                return ""

    return ""


def _filter_tools_list_response(
    raw: bytes,
    *,
    status: int,
    content_type: str,
    allowed_tools: tuple[str, ...] | None,
) -> bytes | None:
    """Filter ``result.tools`` on a tools/list response, or fail closed.

    ``allowed_tools is None`` means the normal view: the hidden
    transport tools are removed and everything else is preserved.
    Otherwise it is a capability view, which must expose EXACTLY its
    granted tool set -- a registry missing one of them, or an empty
    grant, is a hard failure, never a partially filtered list.

    Returns the new response body, or None when the response structure
    cannot be confirmed. Never leaks the full tool list to a capability
    client, and never leaks a hidden transport tool to a normal client.
    """

    if status != 200:
        return None

    if _JSON_CONTENT_TYPE not in content_type:
        return None

    try:
        payload = json.loads(raw)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    result = payload.get("result")

    if not isinstance(result, dict):
        return None

    tools = result.get("tools")

    if not isinstance(tools, list):
        return None

    for tool in tools:
        if (
            not isinstance(tool, dict)
            or not isinstance(tool.get("name"), str)
        ):
            return None

    if allowed_tools is None:
        filtered = [
            tool
            for tool in tools
            if tool.get("name") not in _HIDDEN_TOOL_NAMES
        ]
    else:
        if not allowed_tools:
            return None

        filtered = [
            tool
            for tool in tools
            if tool.get("name") in allowed_tools
        ]

        # The capability view must expose exactly its granted set; a
        # registry without it is a hard failure, never a silent list.
        if (
            len(filtered) != len(allowed_tools)
            or {
                tool.get("name")
                for tool in filtered
            }
            != set(allowed_tools)
        ):
            return None

    new_payload = dict(payload)

    new_result = dict(result)

    new_result["tools"] = filtered

    new_payload["result"] = new_result

    body = json.dumps(
        new_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return body


def _classify(
    body: bytes,
) -> dict[str, Any] | None:
    """Classify one JSON-RPC request body.

    Returns None when the body is not a single well-formed JSON-RPC
    request object, or ``{"kind": "batch"}`` for a JSON array.
    """

    try:
        payload = json.loads(body)
    except Exception:
        return None

    if isinstance(payload, list):
        return {"kind": "batch"}

    if not isinstance(payload, dict):
        return None

    method = payload.get("method")

    if not isinstance(method, str):
        return None

    params = payload.get("params")

    tool_name = None

    if method == "tools/call" and isinstance(
        params, dict
    ):
        candidate = params.get("name")

        if isinstance(candidate, str):
            tool_name = candidate

    return {
        "kind": "request",
        "id": payload.get("id"),
        "method": method,
        "tool_name": tool_name,
    }


def _batch_touches_tools(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except Exception:
        return False

    if not isinstance(payload, list):
        return False

    for item in payload:
        if isinstance(item, dict) and item.get(
            "method"
        ) in ("tools/list", "tools/call"):
            return True

    return False


async def _read_body(
    receive: _Receive,
) -> tuple[bytes, _Receive] | None:
    """Buffer the request body and return a replaying receive."""

    buffered: list[dict] = []

    while True:
        message = await receive()

        if not isinstance(message, dict):
            return None

        if message.get("type") == "http.disconnect":
            return None

        if message.get("type") == "http.request":
            buffered.append(message)

            if not message.get("more_body", False):
                break

    iterator = iter(buffered)

    async def replay_receive() -> dict:
        try:
            return next(iterator)
        except StopIteration:
            return await receive()

    return (
        b"".join(
            message.get("body", b"")
            for message in buffered
        ),
        replay_receive,
    )


class LiveRecallMCPView:
    """Split the single ``/mcp`` connector into its two MCP views."""

    def __init__(
        self,
        app: Any,
        *,
        path_matcher: Callable[[object], bool] = (
            default_mcp_path_matcher
        ),
    ) -> None:
        self.app = app
        self.path_matcher = path_matcher

    async def __call__(
        self,
        scope: dict,
        receive: _Receive,
        send: _Send,
    ) -> None:
        if (
            scope.get("type") != "http"
            or not self.path_matcher(scope.get("path"))
        ):
            await self.app(scope, receive, send)
            return

        capability_view_tools = capability_allowed_tools(
            scope
        )

        method = str(
            scope.get("method", "")
        ).upper()

        if method != "POST":
            # The capability view is a strict POST-only JSON-RPC
            # surface. The normal view keeps its exact behaviour.
            if capability_view_tools is not None:
                await _send_json_rpc_error(
                    send,
                    None,
                    _ERR_INVALID_REQUEST,
                    "Invalid Request",
                )
                return

            await self.app(scope, receive, send)
            return

        buffered = await _read_body(receive)

        if buffered is None:
            return

        body, replay_receive = buffered

        classified = _classify(body)

        # A JSON-RPC batch that touches tools/* cannot be filtered
        # safely, so it fails closed in BOTH views rather than leaking
        # the hidden tool or the full tool list.
        if (
            classified is not None
            and classified.get("kind") == "batch"
        ):
            if _batch_touches_tools(body):
                await _send_json_rpc_error(
                    send,
                    None,
                    _ERR_INVALID_REQUEST,
                    "Invalid Request",
                )
                return

            await self.app(scope, replay_receive, send)
            return

        if capability_view_tools is not None:
            await self._capability_view(
                scope,
                replay_receive,
                send,
                classified,
                capability_view_tools,
            )
            return

        await self._normal_view(
            scope, replay_receive, send, classified
        )

    async def _normal_view(
        self,
        scope: dict,
        receive: _Receive,
        send: _Send,
        classified: dict[str, Any] | None,
    ) -> None:
        if classified is None:
            await self.app(scope, receive, send)
            return

        method = classified.get("method")

        if (
            method == "tools/call"
            and classified.get("tool_name")
            in _HIDDEN_TOOL_NAMES
        ):
            # Security sits before dispatch: guessing the hidden name
            # is not enough.
            await _send_json_rpc_error(
                send,
                classified.get("id"),
                _ERR_METHOD_NOT_FOUND,
                "Method not found",
            )
            return

        if method == "tools/list":
            await self._forward_filtered(
                scope,
                receive,
                send,
                allowed_tools=None,
            )
            return

        await self.app(scope, receive, send)

    async def _capability_view(
        self,
        scope: dict,
        receive: _Receive,
        send: _Send,
        classified: dict[str, Any] | None,
        allowed_tools: tuple[str, ...],
    ) -> None:
        binding = scope.get(BINDING_SCOPE_KEY)

        if classified is None:
            await _send_json_rpc_error(
                send,
                None,
                _ERR_INVALID_REQUEST,
                "Invalid Request",
            )
            return

        method = classified.get("method")

        if method not in _CAPABILITY_ALLOWED_METHODS:
            await _send_json_rpc_error(
                send,
                classified.get("id"),
                _ERR_METHOD_NOT_FOUND,
                "Method not found",
            )
            return

        if method == "tools/call" and (
            classified.get("tool_name")
            not in allowed_tools
        ):
            # An empty grant (inconsistent trusted scope) refuses
            # every call, and a v1 capability can never reach
            # UseMemory.
            await _send_json_rpc_error(
                send,
                classified.get("id"),
                _ERR_METHOD_NOT_FOUND,
                "Method not found",
            )
            return

        token = bind_live_recall_transport(binding)

        try:
            if method == "tools/list":
                await self._forward_filtered(
                    scope,
                    receive,
                    send,
                    allowed_tools=allowed_tools,
                )
                return

            await self.app(scope, receive, send)
        finally:
            reset_live_recall_transport(token)

    async def _forward_filtered(
        self,
        scope: dict,
        receive: _Receive,
        send: _Send,
        *,
        allowed_tools: tuple[str, ...] | None,
    ) -> None:
        captured: dict[str, Any] = {
            "start": None,
            "body": bytearray(),
        }

        async def capture(message: dict) -> None:
            if message.get("type") == "http.response.start":
                captured["start"] = message
            elif (
                message.get("type")
                == "http.response.body"
            ):
                captured["body"] += message.get(
                    "body", b""
                )

        await self.app(scope, receive, capture)

        start = captured["start"]

        if not isinstance(start, dict):
            await _send_json_rpc_error(
                send,
                None,
                _ERR_INTERNAL,
                "Internal error",
            )
            return

        headers = list(start.get("headers", []))

        filtered_body = _filter_tools_list_response(
            bytes(captured["body"]),
            status=int(start.get("status", 200)),
            content_type=_content_type_of(headers),
            allowed_tools=allowed_tools,
        )

        if filtered_body is None:
            # Fail closed: never fall back to the unfiltered list.
            await _send_json_rpc_error(
                send,
                None,
                _ERR_INTERNAL,
                "Internal error",
            )
            return

        new_body = filtered_body

        new_headers = [
            (key, value)
            for key, value in headers
            if key.lower()
            not in (
                b"content-length",
                b"transfer-encoding",
            )
        ]

        new_headers.append(
            (
                b"content-length",
                str(len(new_body)).encode("ascii"),
            )
        )

        await send(
            {
                "type": "http.response.start",
                "status": int(start.get("status", 200)),
                "headers": new_headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": new_body,
                "more_body": False,
            }
        )