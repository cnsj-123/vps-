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
    "gateway_unified_candidate_test_target",
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


class GatewayUnifiedCandidateTests(
    unittest.IsolatedAsyncioTestCase
):

    async def test_unified_observer_runs_when_enabled(
        self,
    ):
        env = {
            "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                "1",
            "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                "0",
        }

        updater = AsyncMock(
            return_value={
                "stored": True,
                "duplicate": False,
                "revision": 1,
                "estimated_tokens": 100,
                "token_budget": 1200,
                "truncated": False,
                "budget_rejected": 0,
                "dedup_rejected": 0,
                "has_current_task": False,
                "trusted_fact_count": 0,
                "constraint_count": 1,
                "decision_count": 0,
                "open_item_count": 0,
                "state_included": True,
                "plan_count": 1,
                "memory_count": 2,
                "recent_context_count": 3,
                "current_user_excluded": True,
                "retrieval_query_used": True,
            }
        )

        with patch.dict(
            os.environ,
            env,
            clear=False,
        ):
            with patch.object(
                gateway,
                "update_unified_context_candidate_from_runtime",
                updater,
            ):
                await (
                    gateway._observe_unified_context_shadow(
                        CID
                    )
                )

        updater.assert_awaited_once_with(
            CID
        )

    async def test_disabled_does_not_run(
        self,
    ):
        updater = AsyncMock()

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "0",
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
                updater,
            ):
                await (
                    gateway._observe_unified_context_shadow(
                        CID
                    )
                )

        updater.assert_not_awaited()

    async def test_failure_is_fail_open(
        self,
    ):
        updater = AsyncMock(
            side_effect=RuntimeError(
                "synthetic failure"
            )
        )

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                    "1",
            },
            clear=False,
        ):
            with patch.object(
                gateway,
                "update_unified_context_candidate_from_runtime",
                updater,
            ):
                await (
                    gateway._observe_unified_context_shadow(
                        CID
                    )
                )


if __name__ == "__main__":
    unittest.main()
