from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


_GATEWAY_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "web"
    / "gateway.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "gateway_injection_preview_test_target",
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

# Context pipeline orchestration now lives in the context
# coordinator; the gateway only calls into it.
from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)


CID = "ctx_0123456789abcdef"


class GatewayInjectionPreviewTests(
    unittest.IsolatedAsyncioTestCase
):

    async def test_preview_runs_when_enabled(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        preview = unittest.mock.Mock(
            return_value={
                "stored": True,
                "duplicate": False,
                "revision": 1,
                "source_revision": 5,
                "eligible": True,
                "reason": None,
                "estimated_tokens": 120,
                "token_budget": 1000,
                "section_names": [
                    "memories",
                ],
                "render_sha256":
                    "abc123",
            }
        )

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                    "0",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                coordinator,
                "update_context_injection_preview",
                preview,
            ):
                await (
                    coordinator.observe_unified_preview_gate(
                        CID
                    )
                )

        unified.assert_awaited_once_with(
            CID
        )

        preview.assert_called_once_with(
            CID
        )

    async def test_preview_disabled_does_not_run(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        preview = unittest.mock.Mock()

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                    "0",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                coordinator,
                "update_context_injection_preview",
                preview,
            ):
                await (
                    coordinator.observe_unified_preview_gate(
                        CID
                    )
                )

        preview.assert_not_called()

    async def test_preview_failure_is_fail_open(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        preview = unittest.mock.Mock(
            side_effect=RuntimeError(
                "synthetic preview failure"
            )
        )

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
                "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                    "0",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                coordinator,
                "update_context_injection_preview",
                preview,
            ):
                # Must not propagate.
                await (
                    coordinator.observe_unified_preview_gate(
                        CID
                    )
                )


if __name__ == "__main__":
    unittest.main()
