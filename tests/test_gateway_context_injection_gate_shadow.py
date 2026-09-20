from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import (
    AsyncMock,
    Mock,
    patch,
)


_GATEWAY_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "web"
    / "gateway.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "gateway_injection_gate_test_target",
    _GATEWAY_PATH,
)

if (
    _SPEC is None
    or _SPEC.loader is None
):
    raise RuntimeError(
        "cannot load gateway.py"
    )

gateway = importlib.util.module_from_spec(
    _SPEC
)

_SPEC.loader.exec_module(
    gateway
)


CID = "ctx_0123456789abcdef"


class GatewayInjectionGateTests(
    unittest.IsolatedAsyncioTestCase
):

    async def test_gate_runs_when_enabled(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        preview = Mock(
            return_value={
                "stored": True,
                "duplicate": False,
                "revision": 2,
                "source_revision": 5,
                "eligible": True,
                "reason": None,
                "estimated_tokens": 300,
                "token_budget": 1000,
                "section_names": [
                    "plans",
                ],
                "render_sha256":
                    "previewhash",
            }
        )

        gate = Mock(
            return_value={
                "stored": True,
                "duplicate": False,
                "revision": 1,
                "mode": "shadow_only",
                "decision":
                    "allow_shadow",
                "allowed": True,
                "reason": None,
                "reasons": [],
                "source_candidate_revision": 8,
                "source_unified_revision": 5,
                "source_preview_revision": 2,
                "estimated_tokens": 300,
                "token_budget": 1000,
                "section_names": [
                    "plans",
                ],
                "render_sha256":
                    "previewhash",
            }
        )

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                    "0",
            },
            clear=False,
        ):
            with patch.object(
                gateway,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                gateway,
                "update_context_injection_preview",
                preview,
            ), patch.object(
                gateway,
                "update_context_injection_gate",
                gate,
            ):
                await (
                    gateway._observe_unified_context_shadow(
                        CID
                    )
                )

        unified.assert_awaited_once_with(
            CID
        )

        preview.assert_called_once_with(
            CID
        )

        gate.assert_called_once_with(
            CID
        )

    async def test_gate_disabled_does_not_run(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        preview = Mock(
            return_value={
                "stored": True,
                "revision": 2,
            }
        )

        gate = Mock()

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                    "0",
            },
            clear=False,
        ):
            with patch.object(
                gateway,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                gateway,
                "update_context_injection_preview",
                preview,
            ), patch.object(
                gateway,
                "update_context_injection_gate",
                gate,
            ):
                await (
                    gateway._observe_unified_context_shadow(
                        CID
                    )
                )

        gate.assert_not_called()

    async def test_preview_failure_prevents_gate(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        preview = Mock(
            side_effect=RuntimeError(
                "synthetic preview failure"
            )
        )

        gate = Mock()

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                    "0",
            },
            clear=False,
        ):
            with patch.object(
                gateway,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                gateway,
                "update_context_injection_preview",
                preview,
            ), patch.object(
                gateway,
                "update_context_injection_gate",
                gate,
            ):
                # Must remain fail-open.
                await (
                    gateway._observe_unified_context_shadow(
                        CID
                    )
                )

        gate.assert_not_called()

    async def test_gate_failure_is_fail_open(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        preview = Mock(
            return_value={
                "stored": True,
                "revision": 2,
            }
        )

        gate = Mock(
            side_effect=RuntimeError(
                "synthetic gate failure"
            )
        )

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                    "0",
            },
            clear=False,
        ):
            with patch.object(
                gateway,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                gateway,
                "update_context_injection_preview",
                preview,
            ), patch.object(
                gateway,
                "update_context_injection_gate",
                gate,
            ):
                # Must not propagate.
                await (
                    gateway._observe_unified_context_shadow(
                        CID
                    )
                )


if __name__ == "__main__":
    unittest.main()
