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
    "gateway_trusted_facts_test_target",
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


class GatewayTrustedFactsShadowTests(
    unittest.TestCase
):

    def test_trusted_facts_can_run_without_semantic(
        self,
    ):
        env = {
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
                "1",
            "OMBRE_GATEWAY_CONTEXT_CANDIDATE_SHADOW":
                "0",
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
                    "conversation_id": CID,
                    "boundary_prefix_sha256":
                        "abc",
                },
            ), patch.object(
                gateway,
                "update_conversation_snapshot",
                return_value={
                    "stored": True,
                    "revision": 1,
                },
            ) as snapshot_mock, patch.object(
                gateway,
                "update_conversation_compact",
                return_value={
                    "stored": True,
                    "source_revision": 1,
                },
            ) as compact_mock, patch.object(
                gateway,
                "update_conversation_trusted_facts",
                return_value={
                    "stored": True,
                    "duplicate": False,
                    "revision": 1,
                    "source_revision": 1,
                    "fact_count": 1,
                    "history_count": 0,
                    "commands_this_revision": 1,
                    "added_this_revision": 1,
                    "confirmed_this_revision": 0,
                    "revoked_this_revision": 0,
                    "replaced_this_revision": 0,
                    "unmatched_this_revision": 0,
                    "inference_enabled": False,
                },
            ) as facts_mock, patch.object(
                gateway,
                "update_conversation_semantic",
                side_effect=AssertionError(
                    "semantic must not run"
                ),
            ):
                gateway._observe_context_shadow(
                    b"{}"
                )

        snapshot_mock.assert_called_once()
        compact_mock.assert_called_once_with(
            CID
        )
        facts_mock.assert_called_once_with(
            CID
        )

    def test_disabled_trusted_facts_does_not_run(
        self,
    ):
        env = {
            "OMBRE_GATEWAY_CONTEXT_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_SNAPSHOT_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_COMPACT_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_SEMANTIC_SHADOW":
                "1",
            "OMBRE_GATEWAY_CONTEXT_SEMANTIC_STATE_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_TRUSTED_FACTS_SHADOW":
                "0",
            "OMBRE_GATEWAY_CONTEXT_CANDIDATE_SHADOW":
                "0",
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
                    "conversation_id": CID,
                    "boundary_prefix_sha256":
                        "abc",
                },
            ), patch.object(
                gateway,
                "update_conversation_snapshot",
                return_value={
                    "stored": True,
                    "revision": 1,
                },
            ), patch.object(
                gateway,
                "update_conversation_compact",
                return_value={
                    "stored": True,
                    "source_revision": 1,
                },
            ), patch.object(
                gateway,
                "update_conversation_trusted_facts",
            ) as facts_mock, patch.object(
                gateway,
                "update_conversation_semantic",
                return_value={
                    "stored": True,
                },
            ):
                gateway._observe_context_shadow(
                    b"{}"
                )

        facts_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
