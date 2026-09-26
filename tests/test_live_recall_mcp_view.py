from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from _recall_fixtures import (
    BRIDGE_ENV,
    CID,
    CID_B,
    LIVE_EXPOSURE_ENV,
    RECALL_ENABLED_ENV,
    RID,
    RID_B,
    FakeBucketManager,
    FakeRetrievalAdapter,
    bucket,
    memory,
    projected_model_result,
    read_usage,
    seed_live_exposure,
    seed_recall_control_plane,
)

from ombrebrain.context.live_recall_authorization import (
    request_live_recall,
)
from ombrebrain.context.live_recall_transport_capability import (
    live_recall_transport_capability_expired,
    resolve_live_recall_transport_capability,
)
from ombrebrain.context.live_recall_transport_context import (
    current_live_recall_transport_binding,
)
from ombrebrain.context.live_recall_usage_attribution import (
    attribute_live_recall_usage,
    live_recall_usage_attribution_enabled,
    render_live_recall_usage_ack,
)
from ombrebrain.context.live_recall_usage_surface import (
    build_live_recall_usage_surface,
    compose_live_recall_result_with_usage_refs,
)
from ombrebrain.context.live_memory_transport_capability import (
    live_memory_transport_capability_expired,
    resolve_live_memory_transport_capability,
)


# The repository's ``web`` package eagerly imports the whole Dashboard
# graph (which needs optional runtime deps), so this test loads the two
# security modules directly, exactly like the Gateway tests load
# ``src/web/gateway.py`` by path.
_ROOT = Path(__file__).resolve().parents[1] / "src"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(
        name, path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")

    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


if "web" not in sys.modules:
    _web_pkg = types.ModuleType("web")
    _web_pkg.__path__ = [str(_ROOT / "web")]
    sys.modules["web"] = _web_pkg

request_limits = _load(
    "web.request_limits",
    _ROOT / "web" / "request_limits.py",
)

view_module = _load(
    "web.live_recall_mcp_view",
    _ROOT / "web" / "live_recall_mcp_view.py",
)

server_app = _load(
    "server_app", _ROOT / "server_app.py"
)


GOOD_TOKEN = "normal-mcp-token"

_CAPABILITY_TOKEN = "obrcap_" + "a" * 64

_MEMREF_A = "memref_" + "a" * 32

_MEMREF_B = "memref_" + "b" * 32


def _tools():
    return [
        {
            "name": "hold",
            "description": "write a memory",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "grow",
            "description": "grow a memory",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "trace",
            "description": "edit a memory",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "Recall",
            "description": "internal transport tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memref": {"type": "string"}
                },
            },
        },
        {
            "name": "UseMemory",
            "description": "internal usage attribution tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "userefs": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
            },
        },
    ]


class FakeMCP:
    """A minimal stand-in for the FastMCP ASGI transport.

    It answers the JSON-RPC methods this middleware cares about and
    records which tool handlers were actually executed.
    """

    def __init__(
        self,
        *,
        tools=None,
        recall_handler=None,
        usage_handler=None,
        malformed_tools_list=False,
    ):
        self.tools = tools if tools is not None else _tools()
        self.recall_handler = recall_handler
        self.usage_handler = usage_handler
        self.malformed_tools_list = malformed_tools_list
        self.handler_calls = []
        self.handler_arguments = []
        self.requests = 0

    async def __call__(self, scope, receive, send):
        body = await self._read(receive)

        try:
            payload = json.loads(body)
        except Exception:
            payload = None

        status, response = await self.dispatch(payload)

        await self._respond(send, status, response)

    async def dispatch(self, payload):
        """Run one JSON-RPC request; return (status, payload|None)."""

        self.requests += 1

        if not isinstance(payload, dict):
            return 200, self._error(
                None, -32600, "Invalid Request"
            )

        method = payload.get("method")

        rpc_id = payload.get("id")

        if method == "notifications/initialized":
            return 202, None

        if method == "initialize":
            return 200, {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "serverInfo": {
                        "name": "fake",
                        "version": "1",
                    },
                },
            }

        if method == "ping":
            return 200, {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {},
            }

        if method == "tools/list":
            tools = (
                "not-a-list"
                if self.malformed_tools_list
                else self.tools
            )

            return 200, {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "tools": tools,
                    "nextCursor": None,
                },
            }

        if method == "tools/call":
            params = payload.get("params") or {}

            name = params.get("name")

            self.handler_calls.append(name)

            arguments = params.get("arguments") or {}

            self.handler_arguments.append(arguments)

            if (
                name == "Recall"
                and self.recall_handler is not None
            ):
                text = await self.recall_handler(
                    arguments.get("memref")
                )
            elif (
                name == "UseMemory"
                and self.usage_handler is not None
            ):
                text = await self.usage_handler(
                    arguments.get("userefs")
                )
            else:
                text = "ok:" + str(name)

            return 200, {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [
                        {"type": "text", "text": text}
                    ],
                    "isError": False,
                },
            }

        return 200, self._error(
            rpc_id, -32601, "Method not found"
        )

    @staticmethod
    async def _read(receive):
        chunks = []

        while True:
            message = await receive()

            if message.get("type") == "http.request":
                chunks.append(message.get("body", b""))

                if not message.get(
                    "more_body", False
                ):
                    break
            elif message.get("type") == "http.disconnect":
                break

        return b"".join(chunks)

    @staticmethod
    def _error(rpc_id, code, message):
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": code, "message": message},
        }

    @staticmethod
    async def _respond(send, status, payload):
        if payload is None:
            body = b""
        else:
            body = json.dumps(payload).encode("utf-8")

        headers = [
            (b"content-type", b"application/json"),
            (
                b"content-length",
                str(len(body)).encode("ascii"),
            ),
        ]

        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )


async def _drive(
    app,
    body,
    *,
    method="POST",
    path="/mcp",
    auth_header=None,
    alt_token=None,
    scope_extra=None,
):
    headers = []

    if auth_header is not None:
        headers.append(
            (
                b"authorization",
                ("Bearer " + auth_header).encode(),
            )
        )

    if alt_token is not None:
        headers.append(
            (b"ombre-mcp-token", alt_token.encode())
        )

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }

    if scope_extra:
        scope.update(scope_extra)

    messages = [
        {
            "type": "http.request",
            "body": (
                body
                if isinstance(body, bytes)
                else json.dumps(body).encode("utf-8")
            ),
            "more_body": False,
        }
    ]

    sent = []

    async def receive():
        if messages:
            return messages.pop(0)

        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)

    return sent


def _parse(sent):
    start = next(
        message
        for message in sent
        if message["type"] == "http.response.start"
    )

    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )

    headers = {
        key.decode("latin-1").lower(): value.decode(
            "latin-1"
        )
        for key, value in start["headers"]
    }

    payload = json.loads(body) if body else None

    return start["status"], headers, payload, body


def _run(coro):
    return asyncio.run(coro)


def _tools_list(rpc_id=1):
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "tools/list",
        "params": {},
    }


def _tools_call(name, arguments=None, rpc_id=2):
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments or {},
        },
    }


def _build(
    *,
    auth_required=True,
    token_validator=None,
    resolver=None,
    probe=None,
    memory_resolver=None,
    memory_probe=None,
    auth_mode="token",
    recall_handler=None,
    usage_handler=None,
    malformed_tools_list=False,
    tools=None,
):
    fake = FakeMCP(
        tools=tools,
        recall_handler=recall_handler,
        usage_handler=usage_handler,
        malformed_tools_list=malformed_tools_list,
    )

    view = view_module.LiveRecallMCPView(fake)

    limit = (
        request_limits.MCPRequestBodyLimitMiddleware(
            view, max_bytes=4 * 1024 * 1024
        )
    )

    auth = server_app.MCPAuthMiddleware(
        limit,
        auth_required=auth_required,
        token_validator=(
            token_validator
            if token_validator is not None
            else (lambda token, resource="": False)
        ),
        auth_mode=auth_mode,
        live_recall_capability_resolver=resolver,
        live_recall_capability_expiry_probe=probe,
        live_memory_capability_resolver=memory_resolver,
        live_memory_capability_expiry_probe=memory_probe,
    )

    return auth, fake


def _normal_auth(token_validator=None):
    validator = token_validator or (
        lambda token, resource="": token
        == GOOD_TOKEN
    )

    return _build(token_validator=validator)


class LiveRecallMCPViewTests(unittest.TestCase):

    # --------------------------------------------------
    # Normal MCP view
    # --------------------------------------------------

    def test_normal_view_hides_recall_from_tools_list(self):
        app, _fake = _normal_auth()

        sent = _run(
            _drive(
                app,
                _tools_list(),
                auth_header=GOOD_TOKEN,
            )
        )

        status, headers, payload, raw = _parse(sent)

        self.assertEqual(status, 200)

        names = [
            tool["name"]
            for tool in payload["result"]["tools"]
        ]

        self.assertEqual(
            names, ["hold", "grow", "trace"]
        )
        self.assertNotIn("Recall", names)

        # Other result fields and the JSON-RPC envelope are preserved.
        self.assertIsNone(payload["result"]["nextCursor"])
        self.assertEqual(payload["id"], 1)
        self.assertEqual(payload["jsonrpc"], "2.0")

        # Content-Length is recomputed for the filtered body.
        self.assertEqual(
            int(headers["content-length"]), len(raw)
        )

    def test_normal_token_cannot_call_recall(self):
        app, fake = _normal_auth()

        sent = _run(
            _drive(
                app,
                _tools_call(
                    "Recall", {"memref": _MEMREF_A}
                ),
                auth_header=GOOD_TOKEN,
            )
        )

        _status, _headers, payload, raw = _parse(sent)

        self.assertEqual(
            payload["error"]["code"], -32601
        )
        self.assertEqual(fake.handler_calls, [])
        self.assertNotIn(_MEMREF_A, raw.decode("utf-8"))

    def test_normal_view_malformed_response_fails_closed(self):
        app, fake = _build(
            token_validator=(
                lambda token, resource="": token
                == GOOD_TOKEN
            ),
            malformed_tools_list=True,
        )

        sent = _run(
            _drive(
                app,
                _tools_list(),
                auth_header=GOOD_TOKEN,
            )
        )

        _status, _headers, payload, raw = _parse(sent)

        self.assertIn("error", payload)
        self.assertNotIn("Recall", raw.decode("utf-8"))
        self.assertNotIn("hold", raw.decode("utf-8"))

    def test_normal_view_rejects_batch_tools_requests(self):
        app, fake = _normal_auth()

        sent = _run(
            _drive(
                app,
                [_tools_list()],
                auth_header=GOOD_TOKEN,
            )
        )

        _status, _headers, payload, _raw = _parse(sent)

        self.assertIn("error", payload)
        self.assertEqual(fake.requests, 0)

    def test_anonymous_view_keeps_recall_hidden(self):
        # auth_required=False must still not expose Recall.
        app, fake = _build(auth_required=False)

        list_sent = _run(
            _drive(app, _tools_list())
        )

        _s, _h, listing, _r = _parse(list_sent)

        names = [
            tool["name"]
            for tool in listing["result"]["tools"]
        ]

        self.assertNotIn("Recall", names)

        call_sent = _run(
            _drive(
                app,
                _tools_call("Recall", {"memref": _MEMREF_A}),
            )
        )

        _s2, _h2, error, _r2 = _parse(call_sent)

        self.assertEqual(error["error"]["code"], -32601)
        self.assertEqual(fake.handler_calls, [])

    def test_capability_never_accepted_from_alt_header(self):
        app, _fake = _build(
            token_validator=(
                lambda token, resource="": False
            ),
            resolver=(
                resolve_live_recall_transport_capability
            ),
            auth_mode="token",
        )

        sent = _run(
            _drive(
                app,
                _tools_list(),
                alt_token=_CAPABILITY_TOKEN,
            )
        )

        status, _headers, _payload, _raw = _parse(sent)

        self.assertEqual(status, 401)

    # --------------------------------------------------
    # Capability view
    # --------------------------------------------------

    def _capability_app(self, recall_handler=None):
        return _build(
            token_validator=(
                lambda token, resource="": False
            ),
            resolver=(
                resolve_live_recall_transport_capability
            ),
            probe=(
                live_recall_transport_capability_expired
            ),
            recall_handler=recall_handler,
        )

    def _state_env(self, root, **extra):
        values = {
            "OMBRE_CONTEXT_STATE_DIR": str(root),
            BRIDGE_ENV: "1",
            RECALL_ENABLED_ENV: "1",
            LIVE_EXPOSURE_ENV: "1",
        }

        values.update(extra)

        return patch.dict(
            os.environ, values, clear=False
        )

    def _mint(self, root, conversation_id=CID, rid=RID):
        from ombrebrain.context.live_recall_transport_capability import (
            mint_live_recall_transport_capability,
        )

        token, _report = (
            mint_live_recall_transport_capability(
                conversation_id=conversation_id,
                cognitive_request_id=rid,
            )
        )

        return token

    def _seed(self):
        tmp = tempfile.TemporaryDirectory()

        self.addCleanup(tmp.cleanup)

        memories = [
            memory("mem-1", "alpha text", name="alpha"),
            memory("mem-2", "beta text", name="beta"),
        ]

        memrefs = seed_live_exposure(
            tmp.name, memories
        )

        return tmp.name, memrefs

    def test_capability_view_lists_only_recall(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint(root)

            app, _fake = self._capability_app()

            sent = _run(
                _drive(
                    app,
                    _tools_list(),
                    auth_header=token,
                )
            )

        status, headers, payload, raw = _parse(sent)

        self.assertEqual(status, 200)

        tools = payload["result"]["tools"]

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "Recall")
        self.assertEqual(
            int(headers["content-length"]), len(raw)
        )

        for forbidden in (
            "hold",
            "grow",
            "trace",
            "breath",
        ):
            self.assertNotIn(forbidden, raw.decode("utf-8"))

    def test_capability_view_rejects_write_and_read_tools(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint(root)

            for name in (
                "hold",
                "trace",
                "grow",
                "breath",
                "breath_search",
                "plan",
            ):
                app, fake = self._capability_app()

                sent = _run(
                    _drive(
                        app,
                        _tools_call(name),
                        auth_header=token,
                    )
                )

                _s, _h, payload, _raw = _parse(sent)

                self.assertEqual(
                    payload["error"]["code"], -32601
                )
                self.assertEqual(
                    fake.handler_calls, [], name
                )
                self.assertEqual(fake.requests, 0, name)

    def test_capability_view_rejects_other_methods(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint(root)

            for method in (
                "resources/list",
                "prompts/list",
                "logging/setLevel",
                "completion/complete",
            ):
                app, fake = self._capability_app()

                sent = _run(
                    _drive(
                        app,
                        {
                            "jsonrpc": "2.0",
                            "id": 9,
                            "method": method,
                        },
                        auth_header=token,
                    )
                )

                _s, _h, payload, _raw = _parse(sent)

                self.assertEqual(
                    payload["error"]["code"],
                    -32601,
                    method,
                )
                self.assertEqual(
                    fake.requests, 0, method
                )

    def test_capability_view_allows_initialize_and_ping(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint(root)

            app, fake = self._capability_app()

            init = _run(
                _drive(
                    app,
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {},
                    },
                    auth_header=token,
                )
            )

            ping = _run(
                _drive(
                    app,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "ping",
                    },
                    auth_header=token,
                )
            )

        _s, _h, init_payload, _r = _parse(init)

        self.assertEqual(init_payload["id"], 1)

        _s2, _h2, ping_payload, _r2 = _parse(ping)

        self.assertEqual(ping_payload["id"], 2)
        self.assertEqual(fake.requests, 2)

    def test_capability_view_malformed_body_fails_closed(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint(root)

            app, fake = self._capability_app()

            sent = _run(
                _drive(
                    app,
                    b"not-json",
                    auth_header=token,
                )
            )

        _s, _h, payload, _raw = _parse(sent)

        self.assertEqual(payload["error"]["code"], -32600)
        self.assertEqual(fake.requests, 0)

    def test_capability_view_rejects_non_post(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint(root)

            app, fake = self._capability_app()

            sent = _run(
                _drive(
                    app,
                    b"",
                    method="GET",
                    auth_header=token,
                )
            )

        _s, _h, payload, _raw = _parse(sent)

        self.assertIn("error", payload)
        self.assertEqual(fake.requests, 0)

    def test_expired_capability_is_unauthorized(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint(root)

            # Directly expire the artifact on disk.
            from ombrebrain.context.live_recall_transport_capability import (
                live_recall_transport_capability_path,
            )

            sha = hashlib.sha256(
                token.encode("utf-8")
            ).hexdigest()

            path = (
                live_recall_transport_capability_path(sha)
            )

            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )

            artifact["created_at"] = (
                "2020-01-01T00:00:00Z"
            )
            artifact["expires_at"] = (
                "2020-01-01T00:05:00Z"
            )

            path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )

            app, fake = self._capability_app()

            sent = _run(
                _drive(
                    app,
                    _tools_list(),
                    auth_header=token,
                )
            )

        status, _headers, _payload, _raw = _parse(sent)

        self.assertEqual(status, 401)
        self.assertEqual(fake.requests, 0)

    # --------------------------------------------------
    # End-to-end
    # --------------------------------------------------

    def test_end_to_end_recall_result_is_reachable(self):
        root, memrefs = self._seed()

        with self._state_env(root):
            token = self._mint(root)

            buckets = FakeBucketManager(
                {
                    "mem-1": bucket(
                        "mem-1", "alpha text", name="alpha"
                    )
                }
            )

            retrieval = FakeRetrievalAdapter(
                [memory("mem-2", "beta text", name="beta")]
            )

            seen_binding = {}

            async def recall_handler(memref):
                binding = (
                    current_live_recall_transport_binding()
                )

                seen_binding.update(binding or {})

                report = await request_live_recall(
                    conversation_id=binding[
                        "conversation_id"
                    ],
                    cognitive_request_id=binding[
                        "cognitive_request_id"
                    ],
                    memref=memref,
                    bucket_manager=buckets,
                    retrieval_adapter=retrieval,
                )

                self.assertTrue(report["authorized"])

                return report["rendered"]

            app, fake = self._capability_app(
                recall_handler=recall_handler
            )

            sent = _run(
                _drive(
                    app,
                    _tools_call(
                        "Recall",
                        {"memref": memrefs[0]},
                    ),
                    auth_header=token,
                )
            )

        _status, _headers, payload, _raw = _parse(sent)

        self.assertEqual(fake.handler_calls, ["Recall"])
        self.assertEqual(
            seen_binding,
            {
                "conversation_id": CID,
                "cognitive_request_id": RID,
            },
        )

        text = payload["result"]["content"][0]["text"]

        self.assertIn("OMBRE RECALL DATA", text)

        for forbidden in (
            "mem-1",
            "mem-2",
            CID,
            RID,
            "memref_",
            "recall_",
        ):
            self.assertNotIn(forbidden, text)

        # Usage: recall_requested + memory_loaded, never used; and no
        # reinforcement / lifecycle artifact appears.
        usage_dir = (
            Path(root)
            / "memory_usage"
            / CID
            / RID
        )

        usage_files = sorted(
            usage_dir.glob("*.json")
        )

        self.assertEqual(len(usage_files), 1)

        usage = json.loads(
            usage_files[0].read_text(encoding="utf-8")
        )

        stages = [
            event.get("stage")
            for event in (usage.get("events") or [])
        ]

        self.assertEqual(
            stages.count("recall_requested"), 1
        )
        self.assertGreaterEqual(
            stages.count("memory_loaded"), 1
        )
        self.assertEqual(stages.count("used"), 0)
        self.assertEqual(usage.get("used_count"), 0)

        self.assertFalse(
            (Path(root) / "memory_lifecycle_event").exists()
        )
        self.assertFalse(
            (Path(root) / "memory_reinforcement").exists()
        )

    def test_two_concurrent_capabilities_do_not_cross(self):
        memrefs = {
            "binding_a": _MEMREF_A,
            "binding_b": _MEMREF_B,
        }

        seen = {}

        started = []

        gate = asyncio.Event()

        async def recall_handler(memref):
            binding = (
                current_live_recall_transport_binding()
            )

            seen[memref] = dict(binding or {})

            started.append(memref)

            if len(started) >= 2:
                gate.set()

            await asyncio.wait_for(
                gate.wait(), timeout=5
            )

            return "ok"

        app, _fake = self._build_view_only(
            recall_handler=recall_handler
        )

        async def scenario():
            await asyncio.gather(
                _drive(
                    app,
                    _tools_call(
                        "Recall", {"memref": _MEMREF_A}
                    ),
                    scope_extra={
                        view_module.AUTH_KIND_SCOPE_KEY:
                            view_module.CAPABILITY_AUTH_KIND,
                        view_module.BINDING_SCOPE_KEY: {
                            "conversation_id": CID,
                            "cognitive_request_id": RID,
                        },
                    },
                ),
                _drive(
                    app,
                    _tools_call(
                        "Recall", {"memref": _MEMREF_B}
                    ),
                    scope_extra={
                        view_module.AUTH_KIND_SCOPE_KEY:
                            view_module.CAPABILITY_AUTH_KIND,
                        view_module.BINDING_SCOPE_KEY: {
                            "conversation_id": CID_B,
                            "cognitive_request_id": RID_B,
                        },
                    },
                ),
            )

        _run(scenario())

        self.assertEqual(
            seen[_MEMREF_A]["cognitive_request_id"], RID
        )
        self.assertEqual(
            seen[_MEMREF_A]["conversation_id"], CID
        )
        self.assertEqual(
            seen[_MEMREF_B]["cognitive_request_id"], RID_B
        )
        self.assertEqual(
            seen[_MEMREF_B]["conversation_id"], CID_B
        )

    def _build_view_only(self, *, recall_handler):
        fake = FakeMCP(recall_handler=recall_handler)

        return (
            view_module.LiveRecallMCPView(fake),
            fake,
        )


class MCPAuthPrecedenceTests(unittest.TestCase):
    """Normal-token semantics are evaluated BEFORE the capability fallback."""

    def test_normal_auth_wins_over_capability_fallback(self):
        resolver_calls = []

        def resolver(token):
            resolver_calls.append(token)
            return {
                "conversation_id": CID,
                "cognitive_request_id": RID,
            }

        # A normal token whose SHAPE looks exactly like a capability
        # token. The normal validator accepts it.
        app, fake = _build(
            auth_required=True,
            auth_mode="token",
            token_validator=(
                lambda token, resource="": token
                == _CAPABILITY_TOKEN
            ),
            resolver=resolver,
        )

        listing = _run(
            _drive(
                app,
                _tools_list(),
                auth_header=_CAPABILITY_TOKEN,
            )
        )

        status, _headers, payload, raw = _parse(listing)

        self.assertEqual(status, 200)

        names = [
            tool["name"]
            for tool in payload["result"]["tools"]
        ]

        # Normal public view, Recall hidden.
        self.assertEqual(
            names, ["hold", "grow", "trace"]
        )
        self.assertNotIn("Recall", names)
        self.assertNotIn("Recall", raw.decode("utf-8"))

        # The resolver must not have been consulted at all.
        self.assertEqual(resolver_calls, [])

        # And it is not a capability view: calling Recall is refused.
        call = _run(
            _drive(
                app,
                _tools_call(
                    "Recall", {"memref": _MEMREF_A}
                ),
                auth_header=_CAPABILITY_TOKEN,
            )
        )

        _s, _h, error, _r = _parse(call)

        self.assertEqual(error["error"]["code"], -32601)
        self.assertEqual(fake.handler_calls, [])

    def test_plain_token_never_becomes_capability_when_auth_off(self):
        resolver_calls = []

        def resolver(token):
            resolver_calls.append(token)
            return {
                "conversation_id": CID,
                "cognitive_request_id": RID,
            }

        app, fake = _build(
            auth_required=False,
            resolver=resolver,
        )

        sent = _run(
            _drive(
                app,
                _tools_list(),
                auth_header="plain-normal-token",
            )
        )

        status, _headers, payload, raw = _parse(sent)

        self.assertEqual(status, 200)

        names = [
            tool["name"]
            for tool in payload["result"]["tools"]
        ]

        self.assertNotIn("Recall", names)
        self.assertNotIn("Recall", raw.decode("utf-8"))
        self.assertEqual(resolver_calls, [])
        self.assertEqual(fake.handler_calls, [])

    def test_capability_view_still_reachable_when_auth_off(self):
        tmp = tempfile.TemporaryDirectory()

        self.addCleanup(tmp.cleanup)

        seed_live_exposure(
            tmp.name,
            [
                memory("m-1", "alpha text", name="alpha"),
                memory("m-2", "beta text", name="beta"),
            ],
        )

        fake = FakeMCP()

        with patch.dict(
            os.environ,
            {"OMBRE_CONTEXT_STATE_DIR": tmp.name},
            clear=False,
        ):
            from ombrebrain.context.live_recall_transport_capability import (
                mint_live_recall_transport_capability,
            )

            token, _report = (
                mint_live_recall_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                )
            )

            app, fake = _build(
                auth_required=False,
                resolver=(
                    resolve_live_recall_transport_capability
                ),
                probe=(
                    live_recall_transport_capability_expired
                ),
            )

            sent = _run(
                _drive(
                    app,
                    _tools_list(),
                    auth_header=token,
                )
            )

        status, _headers, payload, _raw = _parse(sent)

        self.assertEqual(status, 200)

        tools = payload["result"]["tools"]

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "Recall")
        self.assertEqual(fake.handler_calls, [])


class _HttpAppHarness:
    """Shared Starlette host + settings for the real middleware stack."""

    def _host(self, fake):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def endpoint(request):
            payload = await request.json()

            status, response = await fake.dispatch(
                payload
            )

            if response is None:
                return JSONResponse({}, status_code=status)

            return JSONResponse(
                response, status_code=status
            )

        class Host:
            def streamable_http_app(self):
                return Starlette(
                    routes=[
                        Route(
                            "/mcp",
                            endpoint,
                            methods=["POST"],
                        )
                    ]
                )

        return Host()

    def _settings(self, max_request_bytes=4 * 1024 * 1024):
        return server_app.HTTPRuntimeSettings(
            auth_required=False,
            max_request_bytes=max_request_bytes,
            auth_mode="oauth",
            public_origin="https://example.com",
        )

    def _build(self, fake, *, settings=None, auth_required=False):
        import logging

        return server_app.build_http_app(
            self._host(fake),
            "streamable-http",
            settings=(
                settings
                if settings is not None
                else self._settings()
            ),
            token_validator=(
                lambda token, resource="": False
            ),
            lifecycle=server_app.RuntimeLifecycle(
                logger=logging.getLogger("test")
            ),
            live_recall_capability_resolver=(
                resolve_live_recall_transport_capability
            ),
            live_recall_capability_expiry_probe=(
                live_recall_transport_capability_expired
            ),
            live_memory_capability_resolver=(
                resolve_live_memory_transport_capability
            ),
            live_memory_capability_expiry_probe=(
                live_memory_transport_capability_expired
            ),
        )


class BuildHttpAppOrderTests(
    _HttpAppHarness, unittest.TestCase
):
    """Drive the REAL ``build_http_app`` middleware stack.

    This is the ordering regression: MCPAuth -> MCPRequestBodyLimit ->
    LiveRecallMCPView -> FastMCP. The body limit must run before the
    view parses JSON-RPC, and the capability view must be reachable
    only through the auth-established binding.
    """

    def test_body_limit_runs_before_the_view(self):
        fake = FakeMCP()

        app = self._build(
            fake, settings=self._settings(256)
        )

        sent = _run(
            _drive(
                app,
                _tools_call(
                    "hold",
                    {"filler": "x" * 4096},
                ),
            )
        )

        status, _headers, _payload, _raw = _parse(sent)

        self.assertEqual(status, 413)
        self.assertEqual(fake.requests, 0)

    def test_normal_view_filters_through_full_stack(self):
        fake = FakeMCP()

        app = self._build(fake)

        sent = _run(
            _drive(app, _tools_list())
        )

        status, _headers, payload, raw = _parse(sent)

        self.assertEqual(status, 200)

        names = [
            tool["name"]
            for tool in payload["result"]["tools"]
        ]

        self.assertEqual(names, ["hold", "grow", "trace"])
        self.assertNotIn("Recall", raw.decode("utf-8"))

    def test_capability_view_through_full_stack(self):
        tmp = tempfile.TemporaryDirectory()

        self.addCleanup(tmp.cleanup)

        seed_live_exposure(
            tmp.name,
            [
                memory("m-1", "alpha text", name="alpha"),
                memory("m-2", "beta text", name="beta"),
            ],
        )

        fake = FakeMCP()

        with patch.dict(
            os.environ,
            {"OMBRE_CONTEXT_STATE_DIR": tmp.name},
            clear=False,
        ):
            from ombrebrain.context.live_recall_transport_capability import (
                mint_live_recall_transport_capability,
            )

            token, _report = (
                mint_live_recall_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                )
            )

            app = self._build(
                fake,
                settings=server_app.HTTPRuntimeSettings(
                    auth_required=True,
                    max_request_bytes=4 * 1024 * 1024,
                    auth_mode="oauth",
                    public_origin="https://example.com",
                ),
            )

            sent = _run(
                _drive(
                    app,
                    _tools_list(),
                    auth_header=token,
                )
            )

        status, _headers, payload, raw = _parse(sent)

        self.assertEqual(status, 200)

        tools = payload["result"]["tools"]

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "Recall")
        self.assertNotIn("hold", raw.decode("utf-8"))


# ------------------------------------------------------
# UseMemory / v2 capability view
# ------------------------------------------------------
#
# Tool visibility and tool reachability are two DIFFERENT facts:
#
#   normal MCP token : Recall hidden, UseMemory hidden
#   v1 obrcap_       : Recall visible, UseMemory hidden
#   v2 obmtcap_      : Recall visible, UseMemory visible
#   stdio            : neither tool exists in the registry at all
#                      (tested in test_live_recall_transport_registry)


_USAGEREF = "useref_" + "a" * 32

_MEMREF = "memref_" + "c" * 32


class MemoryUsageToolVisibilityTests(unittest.TestCase):
    """The normal view and the v1 view can never reach UseMemory."""

    def test_normal_view_hides_and_refuses_use_memory(self):
        app, fake = _normal_auth()

        listing = _run(
            _drive(
                app,
                _tools_list(),
                auth_header=GOOD_TOKEN,
            )
        )

        _status, _headers, payload, raw = _parse(listing)

        names = [
            tool["name"]
            for tool in payload["result"]["tools"]
        ]

        self.assertEqual(
            names, ["hold", "grow", "trace"]
        )

        for hidden in ("Recall", "UseMemory"):
            self.assertNotIn(hidden, names)
            self.assertNotIn(
                hidden, raw.decode("utf-8")
            )

        for name in ("Recall", "UseMemory"):
            call = _run(
                _drive(
                    app,
                    _tools_call(name),
                    auth_header=GOOD_TOKEN,
                )
            )

            _s, _h, error, call_raw = _parse(call)

            self.assertEqual(
                error["error"]["code"], -32601
            )
            self.assertNotIn(
                name, call_raw.decode("utf-8")
            )

        self.assertEqual(fake.handler_calls, [])

    def test_anonymous_view_hides_and_refuses_use_memory(self):
        app, fake = _build(auth_required=False)

        listing = _run(_drive(app, _tools_list()))

        _s, _h, payload, raw = _parse(listing)

        names = [
            tool["name"]
            for tool in payload["result"]["tools"]
        ]

        self.assertNotIn("UseMemory", names)
        self.assertNotIn("Recall", names)

        call = _run(
            _drive(app, _tools_call("UseMemory"))
        )

        _s2, _h2, error, _r2 = _parse(call)

        self.assertEqual(error["error"]["code"], -32601)
        self.assertEqual(fake.handler_calls, [])

    def _capability_app(self):
        return _build(
            token_validator=(
                lambda token, resource="": False
            ),
            resolver=(
                resolve_live_recall_transport_capability
            ),
            probe=(
                live_recall_transport_capability_expired
            ),
            memory_resolver=(
                resolve_live_memory_transport_capability
            ),
            memory_probe=(
                live_memory_transport_capability_expired
            ),
        )

    def _state_env(self, root, **extra):
        values = {
            "OMBRE_CONTEXT_STATE_DIR": str(root),
            BRIDGE_ENV: "1",
            RECALL_ENABLED_ENV: "1",
            LIVE_EXPOSURE_ENV: "1",
        }

        values.update(extra)

        return patch.dict(
            os.environ, values, clear=False
        )

    def _seed(self, memories=None):
        tmp = tempfile.TemporaryDirectory()

        self.addCleanup(tmp.cleanup)

        memrefs = seed_live_exposure(
            tmp.name,
            memories
            if memories is not None
            else [
                memory("mem-1", "alpha text", name="alpha"),
                memory("mem-2", "beta text", name="beta"),
            ],
        )

        return tmp.name, memrefs

    def test_v1_capability_never_reaches_use_memory(self):
        root, memrefs = self._seed()

        with self._state_env(root):
            from ombrebrain.context.live_recall_transport_capability import (
                mint_live_recall_transport_capability,
            )

            token, _report = (
                mint_live_recall_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                )
            )

            app, fake = self._capability_app()

            listing = _run(
                _drive(
                    app,
                    _tools_list(),
                    auth_header=token,
                )
            )

            call = _run(
                _drive(
                    app,
                    _tools_call(
                        "UseMemory",
                        {"userefs": [_USAGEREF]},
                    ),
                    auth_header=token,
                )
            )

        _s, _h, payload, raw = _parse(listing)

        tools = payload["result"]["tools"]

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "Recall")
        self.assertNotIn(
            "UseMemory", raw.decode("utf-8")
        )

        _s2, _h2, error, _r2 = _parse(call)

        self.assertEqual(error["error"]["code"], -32601)
        self.assertEqual(fake.handler_calls, [])
        self.assertEqual(fake.requests, 1)

        # The recalled memref still works through the v1 view.
        self.assertEqual(
            len(memrefs), 2
        )


class MemoryCapabilityViewTests(unittest.TestCase):
    """The v2 obmtcap view sees exactly Recall + UseMemory."""

    def _app(self, recall_handler=None, usage_handler=None):
        return _build(
            token_validator=(
                lambda token, resource="": False
            ),
            resolver=(
                resolve_live_recall_transport_capability
            ),
            probe=(
                live_recall_transport_capability_expired
            ),
            memory_resolver=(
                resolve_live_memory_transport_capability
            ),
            memory_probe=(
                live_memory_transport_capability_expired
            ),
            recall_handler=recall_handler,
            usage_handler=usage_handler,
        )

    def _state_env(self, root, **extra):
        values = {
            "OMBRE_CONTEXT_STATE_DIR": str(root),
            "OMBRE_GATEWAY_CONTEXT_USAGE_ATTRIBUTION": "1",
            BRIDGE_ENV: "1",
            RECALL_ENABLED_ENV: "1",
            LIVE_EXPOSURE_ENV: "1",
        }

        values.update(extra)

        return patch.dict(
            os.environ, values, clear=False
        )

    def _seed(self, memory_ids=("mem-1", "mem-2")):
        tmp = tempfile.TemporaryDirectory()

        self.addCleanup(tmp.cleanup)

        memories = [
            memory(memory_id, "text " + memory_id)
            for memory_id in memory_ids
        ]

        memrefs = seed_live_exposure(
            tmp.name, memories
        )

        return tmp.name, memrefs

    def _mint(self):
        from ombrebrain.context.live_memory_transport_capability import (
            mint_live_memory_transport_capability,
        )

        token, _report = (
            mint_live_memory_transport_capability(
                conversation_id=CID,
                cognitive_request_id=RID,
            )
        )

        return token

    def test_v2_view_lists_exactly_the_two_transport_tools(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint()

            self.assertIsInstance(token, str)

            app, _fake = self._app()

            sent = _run(
                _drive(
                    app,
                    _tools_list(),
                    auth_header=token,
                )
            )

        status, headers, payload, raw = _parse(sent)

        self.assertEqual(status, 200)

        names = {
            tool["name"]
            for tool in payload["result"]["tools"]
        }

        self.assertEqual(
            names, {"Recall", "UseMemory"}
        )

        self.assertEqual(
            int(headers["content-length"]), len(raw)
        )

        for forbidden in (
            "hold",
            "grow",
            "trace",
            "breath",
            "plan",
            "letter",
            "You",
            "Them",
        ):
            self.assertNotIn(
                forbidden, raw.decode("utf-8")
            )

    def test_v2_view_refuses_every_other_tool(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint()

            for name in (
                "hold",
                "grow",
                "trace",
                "breath",
                "breath_search",
                "breath_advanced",
                "plan",
                "dream",
                "anchor",
                "release",
                "pulse",
                "letter_write",
                "letter_read",
                "feel",
                "I",
                "You",
                "Them",
            ):
                app, fake = self._app()

                sent = _run(
                    _drive(
                        app,
                        _tools_call(name),
                        auth_header=token,
                    )
                )

                _s, _h, payload, _raw = _parse(sent)

                self.assertEqual(
                    payload["error"]["code"],
                    -32601,
                    name,
                )
                self.assertEqual(
                    fake.handler_calls, [], name
                )
                self.assertEqual(
                    fake.requests, 0, name
                )

    def test_v2_view_rejects_other_methods_and_non_post(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint()

            for method in (
                "resources/list",
                "prompts/list",
                "logging/setLevel",
                "completion/complete",
                "resources/read",
            ):
                app, fake = self._app()

                sent = _run(
                    _drive(
                        app,
                        {
                            "jsonrpc": "2.0",
                            "id": 9,
                            "method": method,
                        },
                        auth_header=token,
                    )
                )

                _s, _h, payload, _raw = _parse(sent)

                self.assertEqual(
                    payload["error"]["code"],
                    -32601,
                    method,
                )
                self.assertEqual(
                    fake.requests, 0, method
                )

            app, fake = self._app()

            sent = _run(
                _drive(
                    app,
                    b"",
                    method="GET",
                    auth_header=token,
                )
            )

        _s2, _h2, payload2, _r2 = _parse(sent)

        self.assertIn("error", payload2)
        self.assertEqual(fake.requests, 0)

    def test_use_memory_receives_only_userefs_with_trusted_binding(self):
        root, _memrefs = self._seed()

        seen = {}

        async def usage_handler(userefs):
            seen["userefs"] = userefs
            seen["binding"] = dict(
                current_live_recall_transport_binding()
                or {}
            )

            return "ack"

        with self._state_env(root):
            token = self._mint()

            app, fake = self._app(
                usage_handler=usage_handler
            )

            sent = _run(
                _drive(
                    app,
                    _tools_call(
                        "UseMemory",
                        {"userefs": [_USAGEREF]},
                    ),
                    auth_header=token,
                )
            )

        _s, _h, payload, _raw = _parse(sent)

        self.assertEqual(fake.handler_calls, ["UseMemory"])
        self.assertEqual(
            fake.handler_arguments,
            [{"userefs": [_USAGEREF]}],
        )
        self.assertEqual(seen["userefs"], [_USAGEREF])
        self.assertEqual(
            seen["binding"],
            {
                "conversation_id": CID,
                "cognitive_request_id": RID,
            },
        )

        self.assertEqual(
            payload["result"]["content"][0]["text"],
            "ack",
        )

    def test_end_to_end_attribution_through_the_v2_view(self):
        root, memrefs = self._seed()

        seed_recall_control_plane(
            root, ["mem-1", "mem-2"]
        )

        with self._state_env(root):
            token = self._mint()

            surface_report = (
                build_live_recall_usage_surface(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    anchor_ref=memrefs[0],
                    model_result=projected_model_result(
                        ["mem-1", "mem-2"]
                    ),
                )
            )

            self.assertTrue(surface_report["stored"])

            useref = surface_report["usage_refs"][0][
                "useref"
            ]

            async def usage_handler(userefs):
                binding = (
                    current_live_recall_transport_binding()
                )

                report = attribute_live_recall_usage(
                    conversation_id=binding[
                        "conversation_id"
                    ],
                    cognitive_request_id=binding[
                        "cognitive_request_id"
                    ],
                    userefs=userefs,
                )

                return render_live_recall_usage_ack(
                    report
                )

            app, fake = self._app(
                usage_handler=usage_handler
            )

            sent = _run(
                _drive(
                    app,
                    _tools_call(
                        "UseMemory",
                        {"userefs": [useref]},
                    ),
                    auth_header=token,
                )
            )

            usage = read_usage(root)

        _s, _h, payload, _raw = _parse(sent)

        self.assertEqual(fake.handler_calls, ["UseMemory"])

        text = payload["result"]["content"][0]["text"]

        self.assertTrue(
            text.startswith(
                "OMBRE MEMORY USAGE ACK DATA\n"
            )
        )

        ack = json.loads(text.split("\n", 1)[1])

        self.assertEqual(ack["status"], "recorded")
        self.assertEqual(ack["used_count"], 1)
        self.assertEqual(ack["newly_used_count"], 1)

        for forbidden in (
            "useref_",
            "mem-1",
            "recall_",
            CID,
            RID,
            "memref_",
        ):
            self.assertNotIn(forbidden, text)

        self.assertEqual(usage["used_count"], 1)
        self.assertEqual(
            usage["used_memory_ids"], ["mem-1"]
        )
        self.assertEqual(
            [
                event
                for event in usage["events"]
                if event["stage"] == "used"
            ][0]["memory_id"],
            "mem-1",
        )

    def test_recall_alone_never_records_used(self):
        """Recall succeeds; without UseMemory, ``used`` stays zero."""

        root, memrefs = self._seed()

        buckets = FakeBucketManager(
            {
                "mem-1": bucket(
                    "mem-1", "text mem-1"
                )
            }
        )

        retrieval = FakeRetrievalAdapter(
            [memory("mem-2", "text mem-2")]
        )

        async def recall_handler(memref):
            binding = (
                current_live_recall_transport_binding()
            )

            report = await request_live_recall(
                conversation_id=binding[
                    "conversation_id"
                ],
                cognitive_request_id=binding[
                    "cognitive_request_id"
                ],
                memref=memref,
                bucket_manager=buckets,
                retrieval_adapter=retrieval,
            )

            self.assertTrue(report["authorized"])

            if not (
                live_recall_usage_attribution_enabled()
            ):
                return report["rendered"]

            return (
                compose_live_recall_result_with_usage_refs(
                    rendered=report["rendered"],
                    conversation_id=binding[
                        "conversation_id"
                    ],
                    cognitive_request_id=binding[
                        "cognitive_request_id"
                    ],
                    anchor_ref=memref,
                    model_result=report.get(
                        "model_result"
                    ),
                )
            )

        with self._state_env(root):
            token = self._mint()

            app, fake = self._app(
                recall_handler=recall_handler
            )

            sent = _run(
                _drive(
                    app,
                    _tools_call(
                        "Recall",
                        {"memref": memrefs[0]},
                    ),
                    auth_header=token,
                )
            )

            # The real bridge mints the recall id, so the persisted
            # usage artifact is located by directory, not by a guess.
            usage_dir = (
                Path(root)
                / "memory_usage"
                / CID
                / RID
            )

            usage_files = sorted(
                usage_dir.glob("*.json")
            )

            self.assertEqual(len(usage_files), 1)

            usage = json.loads(
                usage_files[0].read_text(
                    encoding="utf-8"
                )
            )

        _s, _h, payload, _raw = _parse(sent)

        self.assertEqual(fake.handler_calls, ["Recall"])

        text = payload["result"]["content"][0]["text"]

        self.assertIn("OMBRE RECALL DATA", text)
        self.assertIn(
            "OMBRE MEMORY USAGE REFS DATA", text
        )

        # The refs cover the exact model projection, nothing else.
        envelope = text.split(
            "OMBRE MEMORY USAGE REFS DATA\n", 1
        )[1]

        refs_payload = json.loads(
            envelope.split("\n", 1)[1]
        )

        self.assertEqual(
            [
                item["rank"]
                for item in refs_payload["usage_refs"]
            ],
            [1, 2],
        )

        # Recall alone is NOT usage.
        self.assertEqual(usage["used_count"], 0)
        self.assertEqual(usage["used_memory_ids"], [])

        stages = [
            event["stage"] for event in usage["events"]
        ]

        self.assertGreaterEqual(
            stages.count("recall_requested"), 1
        )
        self.assertGreaterEqual(
            stages.count("memory_loaded"), 1
        )
        self.assertEqual(stages.count("used"), 0)

        for directory in (
            "memory_lifecycle_events",
            "memory_lifecycle_state",
            "memory_reinforcement",
        ):
            self.assertFalse(
                (Path(root) / directory).exists(),
                directory,
            )

    def test_forged_v2_token_is_unauthorized(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            app, fake = self._app()

            sent = _run(
                _drive(
                    app,
                    _tools_list(),
                    auth_header="obmtcap_" + "a" * 64,
                )
            )

        status, _headers, _payload, _raw = _parse(sent)

        self.assertEqual(status, 401)
        self.assertEqual(fake.requests, 0)

    def test_v2_token_is_never_accepted_from_the_alt_header(self):
        root, _memrefs = self._seed()

        with self._state_env(root):
            token = self._mint()

            app, fake = self._app()

            sent = _run(
                _drive(
                    app,
                    _tools_list(),
                    alt_token=token,
                )
            )

        status, _headers, _payload, _raw = _parse(sent)

        self.assertEqual(status, 401)
        self.assertEqual(fake.requests, 0)

    def test_scope_tool_set_mismatch_fails_closed(self):
        fake = FakeMCP()

        app = view_module.LiveRecallMCPView(fake)

        base_scope = {
            view_module.AUTH_KIND_SCOPE_KEY:
                view_module.MEMORY_CAPABILITY_AUTH_KIND,
            view_module.BINDING_SCOPE_KEY: {
                "conversation_id": CID,
                "cognitive_request_id": RID,
            },
        }

        for declared in (
            ["Recall"],
            ["UseMemory"],
            ["Recall", "UseMemory", "hold"],
            "Recall",
        ):
            listing = _run(
                _drive(
                    app,
                    _tools_list(),
                    scope_extra={
                        **base_scope,
                        view_module.ALLOWED_TOOLS_SCOPE_KEY:
                            declared,
                    },
                )
            )

            _s, _h, payload, raw = _parse(listing)

            self.assertIn(
                "error", payload, str(declared)
            )
            self.assertNotIn(
                "Recall", raw.decode("utf-8")
            )
            self.assertNotIn(
                "UseMemory", raw.decode("utf-8")
            )

            call = _run(
                _drive(
                    app,
                    _tools_call(
                        "Recall", {"memref": _MEMREF}
                    ),
                    scope_extra={
                        **base_scope,
                        view_module.ALLOWED_TOOLS_SCOPE_KEY:
                            declared,
                    },
                )
            )

            _s2, _h2, error, _r2 = _parse(call)

            self.assertEqual(
                error["error"]["code"],
                -32601,
                str(declared),
            )

        self.assertEqual(fake.handler_calls, [])

    def test_consistent_scope_is_accepted(self):
        fake = FakeMCP()

        app = view_module.LiveRecallMCPView(fake)

        sent = _run(
            _drive(
                app,
                _tools_list(),
                scope_extra={
                    view_module.AUTH_KIND_SCOPE_KEY:
                        view_module.MEMORY_CAPABILITY_AUTH_KIND,
                    view_module.BINDING_SCOPE_KEY: {
                        "conversation_id": CID,
                        "cognitive_request_id": RID,
                    },
                    view_module.ALLOWED_TOOLS_SCOPE_KEY: (
                        "Recall",
                        "UseMemory",
                    ),
                },
            )
        )

        _s, _h, payload, _raw = _parse(sent)

        names = {
            tool["name"]
            for tool in payload["result"]["tools"]
        }

        self.assertEqual(
            names, {"Recall", "UseMemory"}
        )


class FullStackMemoryCapabilityTests(
    _HttpAppHarness, unittest.TestCase
):
    """The real middleware stack drives the v2 capability view."""

    def test_v2_view_through_full_stack(self):
        tmp = tempfile.TemporaryDirectory()

        self.addCleanup(tmp.cleanup)

        memrefs = seed_live_exposure(
            tmp.name,
            [
                memory("m-1", "alpha text", name="alpha"),
                memory("m-2", "beta text", name="beta"),
            ],
        )

        fake = FakeMCP()

        with patch.dict(
            os.environ,
            {
                "OMBRE_CONTEXT_STATE_DIR": tmp.name,
                "OMBRE_GATEWAY_CONTEXT_USAGE_ATTRIBUTION": "1",
            },
            clear=False,
        ):
            from ombrebrain.context.live_memory_transport_capability import (
                mint_live_memory_transport_capability,
            )

            token, _report = (
                mint_live_memory_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                )
            )

            app = self._build(
                fake,
                settings=server_app.HTTPRuntimeSettings(
                    auth_required=True,
                    max_request_bytes=4 * 1024 * 1024,
                    auth_mode="oauth",
                    public_origin="https://example.com",
                ),
            )

            listing = _run(
                _drive(
                    app,
                    _tools_list(),
                    auth_header=token,
                )
            )

            denied = _run(
                _drive(
                    app,
                    _tools_call("hold"),
                    auth_header=token,
                )
            )

        _s, _h, payload, raw = _parse(listing)

        names = {
            tool["name"]
            for tool in payload["result"]["tools"]
        }

        self.assertEqual(
            names, {"Recall", "UseMemory"}
        )
        self.assertNotIn("hold", raw.decode("utf-8"))

        _s2, _h2, error, _r2 = _parse(denied)

        self.assertEqual(error["error"]["code"], -32601)
        self.assertEqual(fake.handler_calls, [])

        self.assertEqual(len(memrefs), 2)

    def test_normal_view_through_full_stack_hides_both_tools(self):
        fake = FakeMCP()

        app = self._build(fake)

        sent = _run(_drive(app, _tools_list()))

        _s, _h, payload, raw = _parse(sent)

        names = [
            tool["name"]
            for tool in payload["result"]["tools"]
        ]

        self.assertEqual(
            names, ["hold", "grow", "trace"]
        )

        for hidden in ("Recall", "UseMemory"):
            self.assertNotIn(
                hidden, raw.decode("utf-8")
            )


if __name__ == "__main__":
    unittest.main()