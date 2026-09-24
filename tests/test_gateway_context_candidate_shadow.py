from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


_GATEWAY_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "web"
    / "gateway.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "gateway_context_candidate_test_target",
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


class GatewayContextCandidateTests(
    unittest.TestCase
):

    def _env(
        self,
        candidate,
    ):
        return {
            "OMBRE_GATEWAY_CONTEXT_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_SNAPSHOT_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_COMPACT_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_SEMANTIC_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_SEMANTIC_STATE_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_TRUSTED_FACTS_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_CANDIDATE_SHADOW":
                "1" if candidate else "0",
            "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                "0",
        }

    def test_candidate_drives_full_dependency_chain(
        self,
    ):
        calls = []

        def mark(
            name,
            result,
        ):
            def fn(*args, **kwargs):
                calls.append(name)
                return dict(result)
            return fn

        with patch.dict(
            os.environ,
            self._env(True),
            clear=False,
        ):
            with patch.object(
                gateway,
                "observe_conversation_shadow",
                side_effect=mark(
                    "shadow",
                    {
                        "observed": True,
                        "conversation_id":
                            CID,
                        "boundary_prefix_sha256":
                            "abc",
                    },
                ),
            ), patch.object(
                gateway,
                "update_conversation_snapshot",
                side_effect=mark(
                    "snapshot",
                    {
                        "stored": True,
                        "revision": 1,
                    },
                ),
            ), patch.object(
                gateway,
                "update_conversation_compact",
                side_effect=mark(
                    "compact",
                    {
                        "stored": True,
                        "source_revision": 1,
                    },
                ),
            ), patch.object(
                gateway,
                "update_conversation_trusted_facts",
                side_effect=mark(
                    "trusted_facts",
                    {
                        "stored": True,
                        "duplicate": False,
                    },
                ),
            ), patch.object(
                gateway,
                "update_conversation_semantic",
                side_effect=mark(
                    "semantic",
                    {
                        "stored": True,
                        "duplicate": False,
                    },
                ),
            ), patch.object(
                gateway,
                "update_semantic_state",
                side_effect=mark(
                    "semantic_state",
                    {
                        "stored": True,
                        "duplicate": False,
                    },
                ),
            ), patch.object(
                gateway,
                "update_conversation_context_candidate",
                side_effect=mark(
                    "candidate",
                    {
                        "stored": True,
                        "duplicate": False,
                        "revision": 1,
                        "source_revision": 1,
                        "estimated_tokens": 20,
                        "token_budget": 900,
                        "truncated": False,
                        "budget_rejected": 0,
                        "dedup_rejected": 0,
                        "control_rejected": 0,
                        "has_current_task": True,
                        "trusted_fact_count": 1,
                        "constraint_count": 1,
                        "decision_count": 0,
                        "open_item_count": 0,
                        "recent_context_count": 2,
                        "current_user_excluded": True,
                    },
                ),
            ):
                gateway._observe_context_shadow(
                    b"{}"
                )

        self.assertEqual(
            calls,
            [
                "shadow",
                "snapshot",
                "compact",
                "trusted_facts",
                "semantic",
                "semantic_state",
                "candidate",
            ],
        )

    def test_candidate_disabled_does_not_run(
        self,
    ):
        env = self._env(
            False
        )

        env[
            "OMBRE_GATEWAY_CONTEXT_SEMANTIC_STATE_SHADOW"
        ] = "1"

        with patch.dict(
            os.environ,
            env,
            clear=False,
        ):
            with patch.object(
                gateway,
                "observe_conversation_shadow",
                return_value={
                    "observed": True,
                    "conversation_id":
                        CID,
                    "boundary_prefix_sha256":
                        "abc",
                },
            ), patch.object(
                gateway,
                "update_conversation_snapshot",
                return_value={
                    "stored": True,
                },
            ), patch.object(
                gateway,
                "update_conversation_compact",
                return_value={
                    "stored": True,
                },
            ), patch.object(
                gateway,
                "update_conversation_semantic",
                return_value={
                    "stored": True,
                },
            ), patch.object(
                gateway,
                "update_semantic_state",
                return_value={
                    "stored": True,
                },
            ), patch.object(
                gateway,
                "update_conversation_context_candidate",
            ) as candidate_mock:
                gateway._observe_context_shadow(
                    b"{}"
                )

        candidate_mock.assert_not_called()

    def test_candidate_failure_is_fail_open(
        self,
    ):
        with patch.dict(
            os.environ,
            self._env(True),
            clear=False,
        ):
            with patch.object(
                gateway,
                "observe_conversation_shadow",
                return_value={
                    "observed": True,
                    "conversation_id":
                        CID,
                    "boundary_prefix_sha256":
                        "abc",
                },
            ), patch.object(
                gateway,
                "update_conversation_snapshot",
                return_value={
                    "stored": True,
                },
            ), patch.object(
                gateway,
                "update_conversation_compact",
                return_value={
                    "stored": True,
                },
            ), patch.object(
                gateway,
                "update_conversation_trusted_facts",
                return_value={
                    "stored": True,
                },
            ), patch.object(
                gateway,
                "update_conversation_semantic",
                return_value={
                    "stored": True,
                },
            ), patch.object(
                gateway,
                "update_semantic_state",
                return_value={
                    "stored": True,
                },
            ), patch.object(
                gateway,
                "update_conversation_context_candidate",
                side_effect=RuntimeError(
                    "synthetic failure"
                ),
            ):
                # Must not propagate.
                gateway._observe_context_shadow(
                    b"{}"
                )


if __name__ == "__main__":
    unittest.main()
