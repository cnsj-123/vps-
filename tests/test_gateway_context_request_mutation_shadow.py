from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import (
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
    "gateway_mutation_shadow_test_target",
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

# The Context pipeline orchestration now lives in the context
# coordinator; the gateway only calls into it. The observer under
# test is resolved from the coordinator module.
from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)


CID = "ctx_0123456789abcdef"
BODY = b'{"messages":[]}'


class GatewayMutationShadowTests(
    unittest.TestCase
):

    def test_mutation_shadow_runs_when_enabled(
        self,
    ):
        builder = Mock(
            return_value=(
                b'{"shadow":"mutated"}',
                {
                    "version":
                        "context-request-mutation-shadow.v1",
                    "mode":
                        "shadow_only",
                    "built":
                        True,
                    "safe_to_mutate":
                        True,
                    "would_inject":
                        True,
                    "upstream_mutated":
                        False,
                    "reason":
                        "shadow_mutation_safe",
                },
            )
        )

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                    "0",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "build_context_request_mutation_shadow_from_runtime",
                builder,
            ):
                result = (
                    coordinator.observe_request_mutation(
                        CID,
                        BODY,
                    )
                )

        self.assertIsNone(
            result
        )

        builder.assert_called_once_with(
            BODY,
            conversation_id=CID,
        )

    def test_disabled_does_not_build(
        self,
    ):
        builder = Mock()

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                    "0",
                "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                    "0",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "build_context_request_mutation_shadow_from_runtime",
                builder,
            ):
                coordinator.observe_request_mutation(
                    CID,
                    BODY,
                )

        builder.assert_not_called()

    def test_failure_is_fail_open(
        self,
    ):
        builder = Mock(
            side_effect=RuntimeError(
                "synthetic mutation failure"
            )
        )

        with patch.dict(
            os.environ,
            {
                "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                    "0",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "build_context_request_mutation_shadow_from_runtime",
                builder,
            ):
                # Must not propagate.
                coordinator.observe_request_mutation(
                    CID,
                    BODY,
                )


if __name__ == "__main__":
    unittest.main()
