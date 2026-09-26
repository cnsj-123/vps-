from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _recall_fixtures import (
    BRIDGE_ENV,
    CID,
    LIVE_EXPOSURE_ENV,
    RECALL_ENABLED_ENV,
    RID,
    TRANSPORT_ENV,
    TRANSPORT_MCP_URL_ENV,
    TRANSPORT_PROVIDER_ENV,
    USAGE_ATTRIBUTION_ENV,
    anthropic_body,
    anthropic_payload,
    memory,
    seed_live_exposure,
)

from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)
from ombrebrain.context.live_recall_transport_capability import (
    live_recall_transport_capability_path,
)
from ombrebrain.context.live_memory_transport_capability import (
    live_memory_transport_capability_path,
)
from ombrebrain.context.provider_live_recall_transport import (
    is_valid_provider_live_recall_transport_receipt,
    prepare_provider_live_recall_transport,
    provider_live_recall_transport_path,
)


_MCP_URL = "https://example.com/mcp"

_SERVER_NAME = "ombre-live-recall"

_TOKEN_RE = re.compile(r"^obrcap_[0-9a-f]{64}$")

_MEMORY_TOKEN_RE = re.compile(
    r"^obmtcap_[0-9a-f]{64}$"
)

_MEMORIES = [memory("mem-1"), memory("mem-2")]


@contextlib.contextmanager
def transport_env(
    root,
    *,
    on=True,
    provider="anthropic_mcp_connector",
    url=_MCP_URL,
    bridge="1",
    recall="1",
    live="1",
    **extra,
):
    values = {
        "OMBRE_CONTEXT_STATE_DIR": str(root),
        TRANSPORT_ENV: "1" if on else "0",
        TRANSPORT_PROVIDER_ENV: provider,
        TRANSPORT_MCP_URL_ENV: url,
        BRIDGE_ENV: bridge,
        RECALL_ENABLED_ENV: recall,
        LIVE_EXPOSURE_ENV: live,
    }

    values.update(extra)

    with patch.dict(
        os.environ, values, clear=False
    ):
        yield


@contextlib.contextmanager
def state(root):
    with patch.dict(
        os.environ,
        {"OMBRE_CONTEXT_STATE_DIR": str(root)},
        clear=False,
    ):
        yield


def _fresh_root(case):
    tmp = tempfile.TemporaryDirectory()

    case.addCleanup(tmp.cleanup)

    seed_live_exposure(tmp.name, list(_MEMORIES))

    return tmp.name


def _call(body, headers, **kwargs):
    return prepare_provider_live_recall_transport(
        body,
        dict(headers),
        conversation_id=kwargs.pop(
            "conversation_id", CID
        ),
        cognitive_request_id=kwargs.pop(
            "cognitive_request_id", RID
        ),
        **kwargs,
    )


class ProviderLiveRecallTransportTests(
    unittest.TestCase
):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

        self.root = self._tmp.name

        self.addCleanup(self._tmp.cleanup)

        seed_live_exposure(
            self.root, list(_MEMORIES)
        )

    # --------------------------------------------------
    # Default OFF / unsupported
    # --------------------------------------------------

    def test_transport_off_is_byte_exact(self):
        body = anthropic_body()
        headers = {
            "anthropic-beta": "other-beta",
            "x-test": "1",
        }

        with transport_env(self.root, on=False):
            new_body, new_headers, report = _call(
                body, headers
            )

        self.assertEqual(new_body, body)
        self.assertEqual(new_headers, headers)
        self.assertFalse(report["enabled"])
        self.assertFalse(report["applied"])
        self.assertEqual(
            report["reason"], "transport_disabled"
        )

    def test_unsupported_provider_is_ignored(self):
        body = anthropic_body()

        with transport_env(
            self.root, provider="openai_whatever"
        ):
            new_body, new_headers, report = _call(
                body, {}
            )

        self.assertEqual(new_body, body)
        self.assertEqual(report["provider"], "openai_whatever")
        self.assertEqual(
            report["reason"], "unsupported_provider"
        )

    def test_invalid_mcp_url_is_refused(self):
        body = anthropic_body()

        for url in (
            "",
            "http://example.com/mcp",
            "ftp://example.com/mcp",
            "/mcp",
            "example.com/mcp",
            "https://example.com/mcp?a=1",
            "https://example.com/mcp#frag",
            "https://",
        ):
            with transport_env(self.root, url=url):
                new_body, _headers, report = _call(
                    body, {}
                )

            self.assertEqual(new_body, body, url)
            self.assertEqual(
                report["reason"],
                "invalid_mcp_url",
                url,
            )

    def test_prerequisites_must_all_be_on(self):
        body = anthropic_body()

        for field in ("bridge", "recall", "live"):
            with transport_env(
                self.root, **{field: "0"}
            ):
                new_body, _headers, report = _call(
                    body, {}
                )

            self.assertEqual(new_body, body, field)
            self.assertEqual(
                report["reason"],
                "prerequisites_disabled",
                field,
            )

    def test_missing_live_exposure_is_refused(self):
        with tempfile.TemporaryDirectory() as empty:
            with transport_env(empty):
                new_body, _headers, report = _call(
                    anthropic_body(), {}
                )

        self.assertEqual(
            report["reason"],
            "live_exposure_unavailable",
        )

    def test_invalid_request_identity_is_refused(self):
        body = anthropic_body()

        with transport_env(self.root):
            new_body, _headers, report = _call(
                body, {}, conversation_id="nope"
            )

        self.assertEqual(new_body, body)
        self.assertEqual(
            report["reason"],
            "invalid_request_identity",
        )

    # --------------------------------------------------
    # Applied mutation
    # --------------------------------------------------

    def test_valid_exposure_injects_connector(self):
        payload = anthropic_payload()
        body = anthropic_body()
        headers = {"x-test": "1"}

        with transport_env(self.root):
            (
                new_body,
                new_headers,
                report,
            ) = _call(body, headers)

            token = json.loads(new_body)[
                "mcp_servers"
            ][0]["authorization_token"]

            sha = hashlib.sha256(
                token.encode("utf-8")
            ).hexdigest()

            artifact_exists = (
                live_recall_transport_capability_path(
                    sha
                ).is_file()
            )

            receipt_path = (
                provider_live_recall_transport_path(
                    CID, RID
                )
            )

            receipt = json.loads(
                receipt_path.read_text(
                    encoding="utf-8"
                )
            )

        self.assertNotEqual(new_body, body)
        self.assertTrue(report["applied"])
        self.assertEqual(
            report["reason"], "transport_applied"
        )
        self.assertTrue(report["receipt_valid"])
        self.assertEqual(report["tools_delta"], 1)
        self.assertEqual(report["mcp_servers_delta"], 1)
        self.assertTrue(report["beta_added"])

        mutated = json.loads(new_body)

        # Immutable regions.
        for key in (
            "messages",
            "system",
            "model",
            "stream",
        ):
            self.assertEqual(
                mutated.get(key), payload.get(key), key
            )

        # Existing tools keep their exact prefix and order.
        self.assertEqual(
            mutated["tools"][: len(payload["tools"])],
            payload["tools"],
        )

        # Exactly one appended toolset, only Recall enabled.
        self.assertEqual(len(mutated["tools"]), 2)

        toolset = mutated["tools"][-1]

        self.assertEqual(
            toolset["type"], "mcp_toolset"
        )
        self.assertEqual(
            toolset["mcp_server_name"], _SERVER_NAME
        )
        self.assertEqual(
            toolset["default_config"],
            {"enabled": False},
        )
        self.assertEqual(
            toolset["configs"],
            {"Recall": {"enabled": True}},
        )

        # Exactly one appended MCP server carrying the capability.
        self.assertEqual(len(mutated["mcp_servers"]), 1)

        server = mutated["mcp_servers"][0]

        self.assertEqual(server["type"], "url")
        self.assertEqual(server["url"], _MCP_URL)
        self.assertEqual(server["name"], _SERVER_NAME)

        token = server["authorization_token"]

        self.assertRegex(token, _TOKEN_RE)

        self.assertTrue(artifact_exists)

        # Beta header.
        self.assertEqual(
            new_headers["anthropic-beta"],
            "mcp-client-2025-11-20",
        )
        self.assertEqual(new_headers["x-test"], "1")

        self.assertTrue(
            is_valid_provider_live_recall_transport_receipt(
                receipt,
                conversation_id=CID,
                cognitive_request_id=RID,
                selected_body_sha256=hashlib.sha256(
                    new_body
                ).hexdigest(),
            )
        )

    def test_existing_tools_and_servers_are_preserved(self):
        payload = anthropic_payload()

        payload["tools"] = [
            {"name": "t1"},
            {
                "type": "mcp_toolset",
                "mcp_server_name": "other",
                "configs": {"X": {"enabled": True}},
            },
        ]

        payload["mcp_servers"] = [
            {
                "type": "url",
                "url": "https://other.example/mcp",
                "name": "other",
            }
        ]

        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        with transport_env(self.root):
            new_body, _headers, report = _call(
                body, {}
            )

        mutated = json.loads(new_body)

        self.assertTrue(report["applied"])

        self.assertEqual(
            mutated["tools"][:2], payload["tools"]
        )
        self.assertEqual(
            mutated["mcp_servers"][:1],
            payload["mcp_servers"],
        )
        self.assertEqual(len(mutated["tools"]), 3)
        self.assertEqual(
            len(mutated["mcp_servers"]), 2
        )

    def test_no_reordering_of_preexisting_entries(self):
        payload = anthropic_payload()

        payload["tools"] = [
            {"name": name}
            for name in ("a", "b", "c", "d")
        ]

        payload["mcp_servers"] = [
            {
                "type": "url",
                "url": f"https://s{i}.example/mcp",
                "name": f"s{i}",
            }
            for i in range(3)
        ]

        body = json.dumps(payload).encode("utf-8")

        with transport_env(self.root):
            new_body, _headers, _report = _call(
                body, {}
            )

        mutated = json.loads(new_body)

        self.assertEqual(
            [tool["name"] for tool in mutated["tools"][:4]],
            ["a", "b", "c", "d"],
        )
        self.assertEqual(
            [s["name"] for s in mutated["mcp_servers"][:3]],
            ["s0", "s1", "s2"],
        )

    # --------------------------------------------------
    # Collisions
    # --------------------------------------------------

    def test_mcp_server_name_collision_refuses(self):
        payload = anthropic_payload()

        payload["mcp_servers"] = [
            {
                "type": "url",
                "url": "https://x/mcp",
                "name": _SERVER_NAME,
            }
        ]

        body = json.dumps(payload).encode("utf-8")

        with transport_env(self.root):
            new_body, _headers, report = _call(
                body, {}
            )

        self.assertEqual(new_body, body)
        self.assertEqual(
            report["reason"],
            "mcp_server_name_collision",
        )

    def test_existing_toolset_collision_refuses(self):
        payload = anthropic_payload()

        payload["tools"] = [
            {"name": "t1"},
            {
                "type": "mcp_toolset",
                "mcp_server_name": _SERVER_NAME,
            },
        ]

        body = json.dumps(payload).encode("utf-8")

        with transport_env(self.root):
            new_body, _headers, report = _call(
                body, {}
            )

        self.assertEqual(new_body, body)
        self.assertEqual(
            report["reason"],
            "mcp_toolset_collision",
        )

    # --------------------------------------------------
    # Malformed shapes
    # --------------------------------------------------

    def test_non_list_tools_or_servers_refuse(self):
        payload = anthropic_payload()

        payload["tools"] = {"not": "a list"}

        bad_tools = json.dumps(payload).encode("utf-8")

        payload["tools"] = [{"name": "t1"}]
        payload["mcp_servers"] = {"not": "a list"}

        bad_servers = json.dumps(payload).encode(
            "utf-8"
        )

        with transport_env(self.root):
            body1, _h1, report1 = _call(
                bad_tools, {}
            )
            body2, _h2, report2 = _call(
                bad_servers, {}
            )

        self.assertEqual(body1, bad_tools)
        self.assertEqual(
            report1["reason"], "invalid_tools"
        )
        self.assertEqual(body2, bad_servers)
        self.assertEqual(
            report2["reason"], "invalid_mcp_servers"
        )

    def test_non_json_and_non_object_bodies_refuse(self):
        with transport_env(self.root):
            body1, _h1, report1 = _call(
                b"not json", {}
            )
            body2, _h2, report2 = _call(
                b"[1,2,3]", {}
            )
            body3, _h3, report3 = _call(
                b'"a string"', {}
            )

        self.assertEqual(
            report1["reason"], "non_json_body"
        )
        self.assertEqual(
            report2["reason"], "unsupported_request"
        )
        self.assertEqual(
            report3["reason"], "unsupported_request"
        )
        self.assertEqual(body1, b"not json")
        self.assertEqual(body2, b"[1,2,3]")

    # --------------------------------------------------
    # Beta header
    # --------------------------------------------------

    def test_beta_header_variants(self):
        body = anthropic_body()

        # No beta header: exactly one token added.
        with transport_env(_fresh_root(self)):
            _b, headers, report = _call(body, {})

        self.assertEqual(
            headers["anthropic-beta"],
            "mcp-client-2025-11-20",
        )
        self.assertTrue(report["beta_added"])

        # An unrelated beta is preserved and the MCP beta appended.
        with transport_env(_fresh_root(self)):
            _b, headers, _report = _call(
                body, {"anthropic-beta": "some-other-beta"}
            )

        self.assertEqual(
            headers["anthropic-beta"],
            "some-other-beta, mcp-client-2025-11-20",
        )

        # The MCP beta already present is never duplicated.
        with transport_env(_fresh_root(self)):
            _b, headers, report = _call(
                body,
                {
                    "anthropic-beta":
                        "x, mcp-client-2025-11-20"
                },
            )

        self.assertEqual(
            headers["anthropic-beta"],
            "x, mcp-client-2025-11-20",
        )
        self.assertFalse(report["beta_added"])

    def test_beta_header_untouched_when_off(self):
        body = anthropic_body()
        headers = {"anthropic-beta": "keep-me"}

        with transport_env(self.root, on=False):
            _b, new_headers, _report = _call(
                body, headers
            )

        self.assertEqual(new_headers, headers)

    # --------------------------------------------------
    # Privacy
    # --------------------------------------------------

    def test_raw_capability_and_url_are_never_reported(self):
        body = anthropic_body()

        with transport_env(_fresh_root(self)):
            new_body, new_headers, report = (
                _call(body, {})
            )

            receipt_text = (
                provider_live_recall_transport_path(
                    CID, RID
                ).read_text(encoding="utf-8")
            )

        mutated = json.loads(new_body)

        token = mutated["mcp_servers"][0][
            "authorization_token"
        ]

        report_text = json.dumps(report)

        self.assertNotIn(token, report_text)
        self.assertNotIn("obrcap_", report_text)
        self.assertNotIn(_MCP_URL, report_text)

        self.assertNotIn(token, receipt_text)
        self.assertNotIn("obrcap_", receipt_text)
        self.assertNotIn(_MCP_URL, receipt_text)
        self.assertNotIn("memref_", receipt_text)

        # The URL is only ever stored hashed.
        self.assertIn(
            hashlib.sha256(
                _MCP_URL.encode("utf-8")
            ).hexdigest(),
            receipt_text,
        )

        # The raw capability IS in the outbound authorization_token.
        self.assertIn(token, new_body.decode("utf-8"))


class ContextPipelineBindingTests(
    unittest.TestCase
):

    def test_with_binding_runs_pipeline_once(self):
        body = anthropic_body()

        calls = {"count": 0}

        def fake_sources(_body):
            calls["count"] += 1
            return CID

        async def scenario():
            with patch.object(
                coordinator,
                "observe_context_sources",
                side_effect=fake_sources,
            ):
                selection = await (
                    coordinator.run_context_pipeline_with_binding(
                        body
                    )
                )

            return selection

        selection = asyncio.run(scenario())

        self.assertEqual(calls["count"], 1)
        self.assertEqual(selection.body, body)
        self.assertEqual(
            selection.conversation_id, CID
        )
        self.assertRegex(
            selection.cognitive_request_id,
            r"^ctxreq_[0-9a-f]{32}$",
        )

    def test_plain_entry_point_returns_bytes(self):
        body = anthropic_body()

        async def scenario():
            with patch.object(
                coordinator,
                "observe_context_sources",
                return_value=CID,
            ):
                return await (
                    coordinator.run_context_pipeline(
                        body
                    )
                )

        result = asyncio.run(scenario())

        self.assertEqual(result, body)

    def test_pipeline_failure_drops_the_binding(self):
        body = anthropic_body()

        async def scenario():
            with patch.object(
                coordinator,
                "observe_context_sources",
                side_effect=RuntimeError("boom"),
            ):
                return await (
                    coordinator.run_context_pipeline_with_binding(
                        body
                    )
                )

        selection = asyncio.run(scenario())

        self.assertEqual(selection.body, body)
        self.assertIsNone(selection.conversation_id)
        self.assertIsNone(
            selection.cognitive_request_id
        )


class UsageAttributionTransportTests(
    unittest.TestCase
):
    """``OMBRE_GATEWAY_CONTEXT_USAGE_ATTRIBUTION`` is additive.

    OFF keeps the frozen v1 behavior byte-for-byte in shape: an
    ``obrcap_`` capability and a Recall-only toolset. ON keeps the
    SAME server / URL / beta / preserved fields and only changes the
    capability kind (v2) plus the toolset configs (Recall +
    UseMemory).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

        self.root = self._tmp.name

        self.addCleanup(self._tmp.cleanup)

        seed_live_exposure(
            self.root, list(_MEMORIES)
        )

    def _prepare(self, *, usage=None):
        """Run ONE transport preparation on a FRESH state root.

        A request can only own one persisted transport receipt, so
        each comparison call needs its own root -- exactly like two
        different Gateway requests.
        """

        tmp = tempfile.TemporaryDirectory()

        self.addCleanup(tmp.cleanup)

        root = tmp.name

        seed_live_exposure(root, list(_MEMORIES))

        extra = (
            {USAGE_ATTRIBUTION_ENV: usage}
            if usage is not None
            else {}
        )

        with transport_env(root, **extra):
            body, headers, report = _call(
                anthropic_body(), {}
            )

        return body, headers, report, root

    def test_flag_off_is_the_frozen_recall_only_transport(self):
        new_body, headers, report, root = self._prepare(
            usage="0"
        )

        self.assertTrue(report["applied"])

        mutated = json.loads(new_body)

        toolset = mutated["tools"][-1]

        self.assertEqual(
            toolset["configs"],
            {"Recall": {"enabled": True}},
        )
        self.assertEqual(
            toolset["default_config"],
            {"enabled": False},
        )

        token = mutated["mcp_servers"][0][
            "authorization_token"
        ]

        self.assertRegex(token, _TOKEN_RE)

        self.assertNotIn(
            "UseMemory", new_body.decode("utf-8")
        )

        # No v2 capability artifact was ever minted.
        sha = hashlib.sha256(
            token.encode("utf-8")
        ).hexdigest()

        with state(root):
            self.assertFalse(
                live_memory_transport_capability_path(
                    sha
                ).exists()
            )

        self.assertEqual(
            headers["anthropic-beta"],
            "mcp-client-2025-11-20",
        )

    def test_flag_off_with_no_env_is_byte_identical_to_off(self):
        off_body, off_headers, off_report, _root = (
            self._prepare(usage="0")
        )

        unset_body, unset_headers, unset_report, _root2 = (
            self._prepare()
        )

        off = json.loads(off_body)
        unset = json.loads(unset_body)

        off["mcp_servers"][0][
            "authorization_token"
        ] = "token"

        unset["mcp_servers"][0][
            "authorization_token"
        ] = "token"

        self.assertEqual(off, unset)
        self.assertEqual(off_headers, unset_headers)
        self.assertEqual(
            off_report["reason"],
            unset_report["reason"],
        )

    def test_flag_on_only_adds_use_memory(self):
        off_body, _off_headers, _off_report, _root = (
            self._prepare(usage="0")
        )

        on_body, on_headers, on_report, on_root = (
            self._prepare(usage="1")
        )

        self.assertTrue(on_report["applied"])
        self.assertEqual(
            on_report["tools_delta"], 1
        )

        off = json.loads(off_body)

        on = json.loads(on_body)

        # Same keys, same preserved fields.
        self.assertEqual(
            set(on.keys()), set(off.keys())
        )

        for key in (
            "messages",
            "system",
            "model",
            "stream",
        ):
            self.assertEqual(
                on.get(key), off.get(key), key
            )

        # Existing entries untouched, one appended toolset / server.
        self.assertEqual(
            on["tools"][: len(off["tools"]) - 1],
            off["tools"][: len(off["tools"]) - 1],
        )
        self.assertEqual(len(on["tools"]), 2)
        self.assertEqual(len(on["mcp_servers"]), 1)

        toolset = on["tools"][-1]

        self.assertEqual(
            toolset["type"], "mcp_toolset"
        )
        self.assertEqual(
            toolset["mcp_server_name"], _SERVER_NAME
        )
        self.assertEqual(
            toolset["default_config"],
            {"enabled": False},
        )
        self.assertEqual(
            toolset["configs"],
            {
                "Recall": {"enabled": True},
                "UseMemory": {"enabled": True},
            },
        )

        server = on["mcp_servers"][0]

        self.assertEqual(server["type"], "url")
        self.assertEqual(server["url"], _MCP_URL)
        self.assertEqual(server["name"], _SERVER_NAME)

        token = server["authorization_token"]

        self.assertRegex(token, _MEMORY_TOKEN_RE)

        # The v2 credential resolves to the trusted binding, and only
        # its SHA256 is on disk.
        sha = hashlib.sha256(
            token.encode("utf-8")
        ).hexdigest()

        with state(on_root):
            artifact_path = (
                live_memory_transport_capability_path(
                    sha
                )
            )

            self.assertTrue(artifact_path.is_file())

            artifact = json.loads(
                artifact_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                artifact["allowed_tools"],
                ["Recall", "UseMemory"],
            )

            self.assertFalse(
                live_recall_transport_capability_path(
                    sha
                ).exists()
            )

        with state(on_root):
            receipt_path = (
                provider_live_recall_transport_path(
                    CID, RID
                )
            )

            receipt = json.loads(
                receipt_path.read_text(
                    encoding="utf-8"
                )
            )

            receipt_text = receipt_path.read_text(
                encoding="utf-8"
            )

        self.assertTrue(
            is_valid_provider_live_recall_transport_receipt(
                receipt,
                conversation_id=CID,
                cognitive_request_id=RID,
                selected_body_sha256=hashlib.sha256(
                    on_body
                ).hexdigest(),
            )
        )

        # The receipt (and the report) still never carries the raw
        # token, a memref or the MCP URL.
        for text in (
            receipt_text,
            json.dumps(on_report),
        ):
            self.assertNotIn(token, text)
            self.assertNotIn("obmtcap_", text)
            self.assertNotIn("obrcap_", text)
            self.assertNotIn(_MCP_URL, text)
            self.assertNotIn("memref_", text)

        self.assertIn(token, on_body.decode("utf-8"))

        self.assertEqual(
            on_headers["anthropic-beta"],
            "mcp-client-2025-11-20",
        )

    def test_flag_on_does_not_relax_the_prerequisites(self):
        body = anthropic_body()

        for field in ("bridge", "recall", "live"):
            with transport_env(
                self.root,
                **{field: "0"},
                **{USAGE_ATTRIBUTION_ENV: "1"},
            ):
                new_body, _headers, report = _call(
                    body, {}
                )

            self.assertEqual(new_body, body, field)
            self.assertEqual(
                report["reason"],
                "prerequisites_disabled",
                field,
            )

    def test_flag_on_keeps_the_collision_guard(self):
        payload = anthropic_payload()

        payload["mcp_servers"] = [
            {
                "type": "url",
                "url": "https://x/mcp",
                "name": _SERVER_NAME,
            }
        ]

        body = json.dumps(payload).encode("utf-8")

        with transport_env(
            self.root,
            **{USAGE_ATTRIBUTION_ENV: "1"},
        ):
            new_body, _headers, report = _call(
                body, {}
            )

        self.assertEqual(new_body, body)
        self.assertEqual(
            report["reason"],
            "mcp_server_name_collision",
        )

    def test_flag_on_never_changes_the_provider_or_the_url(self):
        _body, _headers, report, _root = self._prepare(
            usage="1"
        )

        self.assertEqual(
            report["provider"],
            "anthropic_mcp_connector",
        )
        self.assertEqual(
            report["mcp_servers_delta"], 1
        )
        self.assertTrue(report["receipt_valid"])


if __name__ == "__main__":
    unittest.main()