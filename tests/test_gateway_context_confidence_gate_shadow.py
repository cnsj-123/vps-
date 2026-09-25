from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
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
    "gateway_confidence_gate_test_target",
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

# Confidence Gate orchestration lives in the Context coordinator;
# the gateway never drives it directly.
from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)


CID = "ctx_0123456789abcdef"

_CONFIDENCE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW"
)

# Every Context flag is pinned explicitly so host variables cannot
# leak into these tests.
_ISOLATED_ENV = {
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
    _CONFIDENCE_ENV:
        "0",
}


def _allow_report() -> dict:
    return {
        "version":
            "context-confidence-gate.v1",
        "mode":
            "shadow_only",
        "decision":
            "allow_shadow",
        "allowed":
            True,
        "reason":
            None,
        "reasons": [],
        "stored":
            True,
        "duplicate":
            False,
        "revision":
            1,
    }


def _deny_report() -> dict:
    return {
        "version":
            "context-confidence-gate.v1",
        "mode":
            "shadow_only",
        "decision":
            "deny_shadow",
        "allowed":
            False,
        "reason":
            "semantic_source_stale",
        "reasons": [
            "semantic_source_stale",
        ],
        "stored":
            True,
        "duplicate":
            False,
        "revision":
            1,
    }


class ConfidenceGateCoordinatorTests(
    unittest.IsolatedAsyncioTestCase,
):

    async def test_confidence_runs_when_enabled(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        confidence = Mock(
            return_value=_allow_report()
        )

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                _CONFIDENCE_ENV:
                    "1",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                coordinator,
                "update_context_confidence_gate",
                confidence,
            ):
                result = await (
                    coordinator
                    .observe_unified_preview_gate(
                        CID
                    )
                )

        unified.assert_awaited_once_with(
            CID
        )

        confidence.assert_called_once_with(
            CID,
            expected_unified_revision=5,
        )

        # No Preview/Gate flags are on, so the chain stops after the
        # Confidence Gate observer.
        self.assertIs(
            result,
            False,
        )

    async def test_confidence_disabled_does_not_run(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        confidence = Mock()

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                    "1",
                _CONFIDENCE_ENV:
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
                "update_context_confidence_gate",
                confidence,
            ):
                await (
                    coordinator
                    .observe_unified_preview_gate(
                        CID
                    )
                )

        confidence.assert_not_called()

    async def test_confidence_failure_is_fail_open(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        confidence = Mock(
            side_effect=RuntimeError(
                "synthetic confidence failure"
            )
        )

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                _CONFIDENCE_ENV:
                    "1",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                coordinator,
                "update_context_confidence_gate",
                confidence,
            ):
                # Must not propagate.
                result = await (
                    coordinator
                    .observe_unified_preview_gate(
                        CID
                    )
                )

        self.assertIs(
            result,
            False,
        )

    async def test_confidence_not_run_when_unified_not_stored(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": False,
                "revision": 5,
            }
        )

        confidence = Mock(
            side_effect=AssertionError(
                "confidence must not read a chain "
                "this request did not refresh"
            )
        )

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                _CONFIDENCE_ENV:
                    "1",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                coordinator,
                "update_context_confidence_gate",
                confidence,
            ):
                await (
                    coordinator
                    .observe_unified_preview_gate(
                        CID
                    )
                )

        confidence.assert_not_called()

    async def test_confidence_deny_does_not_change_downstream(
        self,
    ):
        # The key shadow-only guarantee: a deny_shadow decision must
        # not stop the existing Preview / Injection Gate stages.
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
            return_value={
                "stored": True,
                "revision": 1,
                "decision": "allow_shadow",
            }
        )

        confidence = Mock(
            return_value=_deny_report()
        )

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
                    "1",
                _CONFIDENCE_ENV:
                    "1",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                coordinator,
                "update_context_confidence_gate",
                confidence,
            ), patch.object(
                coordinator,
                "update_context_injection_preview",
                preview,
            ), patch.object(
                coordinator,
                "update_context_injection_gate",
                gate,
            ):
                result = await (
                    coordinator
                    .observe_unified_preview_gate(
                        CID
                    )
                )

        confidence.assert_called_once_with(
            CID,
            expected_unified_revision=5,
        )

        # Even though confidence denied, Preview and Gate still ran
        # and the chain is still reported as refreshed.
        preview.assert_called_once_with(
            CID
        )

        gate.assert_called_once_with(
            CID
        )

        self.assertIs(
            result,
            True,
        )

    async def test_confidence_alone_does_not_run_preview_or_gate(
        self,
    ):
        unified = AsyncMock(
            return_value={
                "stored": True,
                "revision": 5,
            }
        )

        preview = Mock()
        gate = Mock()

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                _CONFIDENCE_ENV:
                    "1",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                unified,
            ), patch.object(
                coordinator,
                "update_context_confidence_gate",
                Mock(
                    return_value=_allow_report()
                ),
            ), patch.object(
                coordinator,
                "update_context_injection_preview",
                preview,
            ), patch.object(
                coordinator,
                "update_context_injection_gate",
                gate,
            ):
                await (
                    coordinator
                    .observe_unified_preview_gate(
                        CID
                    )
                )

        preview.assert_not_called()

        gate.assert_not_called()


class ConfidenceGatePipelineTests(
    unittest.IsolatedAsyncioTestCase,
):

    async def test_pipeline_fail_open_on_confidence_exception(
        self,
    ):
        body = b'{"messages":[]}'

        confidence = Mock(
            side_effect=RuntimeError(
                "secret"
            )
        )

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                _CONFIDENCE_ENV:
                    "1",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "observe_context_sources",
                Mock(return_value=CID),
            ), patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                AsyncMock(
                    return_value={
                        "stored": True,
                        "revision": 1,
                    }
                ),
            ), patch.object(
                coordinator,
                "update_context_confidence_gate",
                confidence,
            ):
                with self.assertLogs(
                    "ombre_brain.gateway",
                    level="INFO",
                ) as captured:
                    selected = await (
                        coordinator
                        .run_context_pipeline(
                            body
                        )
                    )

        self.assertIs(
            selected,
            body,
        )

        logged = "\n".join(
            record.getMessage()
            for record in captured.records
        )

        self.assertIn(
            "[gateway.context_confidence_gate]",
            logged,
        )

        self.assertIn(
            "fail_open=true",
            logged,
        )

        # Only the exception type is logged, never its message.
        self.assertNotIn(
            "secret",
            logged,
        )

    async def test_telemetry_never_logs_context_text(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            for kind, payload in (
                (
                    "context_candidate",
                    {
                        "version":
                            "conversation-context-candidate.v1",
                        "conversation_id":
                            CID,
                        "revision":
                            2,
                        "source_revision":
                            1,
                        "sections": {
                            "secret":
                                "context-secret",
                        },
                        "telemetry": {
                            "current_user_excluded":
                                True,
                            "semantic_source_stale":
                                False,
                            "semantic_source_ahead":
                                False,
                            "trusted_facts_source_stale":
                                False,
                            "trusted_facts_source_ahead":
                                False,
                        },
                    },
                ),
                (
                    "unified_context_candidate",
                    {
                        "version":
                            "unified-context-candidate.v1",
                        "conversation_id":
                            CID,
                        "revision":
                            3,
                        "source_revisions": {
                            "conversation_candidate":
                                2,
                            "conversation_source":
                                1,
                        },
                        "sections": {
                            "secret":
                                "context-secret",
                        },
                        "telemetry": {
                            "current_user_excluded":
                                True,
                            "has_current_task":
                                True,
                            "state_included":
                                False,
                            "trusted_fact_count":
                                1,
                            "constraint_count":
                                0,
                            "decision_count":
                                0,
                            "open_item_count":
                                0,
                            "plan_count":
                                0,
                            "memory_count":
                                0,
                            "recent_context_count":
                                0,
                            "retrieval_candidate_count":
                                0,
                            "estimated_tokens":
                                80,
                            "token_budget":
                                1200,
                            "retrieval_quality": {
                                "outcome":
                                    "included",
                            },
                        },
                    },
                ),
            ):
                path = (
                    Path(root)
                    / kind
                    / (CID + ".json")
                )

                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                path.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            with patch.dict(
                os.environ,
                {
                    **_ISOLATED_ENV,
                    _CONFIDENCE_ENV:
                        "1",
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                with self.assertLogs(
                    "ombre_brain.gateway",
                    level="INFO",
                ) as captured:
                    coordinator.observe_context_confidence(
                        CID,
                        expected_unified_revision=3,
                    )

        logged = "\n".join(
            record.getMessage()
            for record in captured.records
        )

        self.assertNotIn(
            "context-secret",
            logged,
        )

        self.assertIn(
            "[gateway.context_confidence_gate]",
            logged,
        )

        self.assertIn(
            '"decision":"allow_shadow"',
            logged.replace(
                " ",
                "",
            ),
        )

    async def test_confidence_log_uses_explicit_allowlist(
        self,
    ):
        # A report carrying unexpected extra fields must never leak
        # them: the observer logs an explicit allowlist, never
        # **report, and never conversation_id.
        report = {
            "version":
                "context-confidence-gate.v1",
            "mode":
                "shadow_only",
            "decision":
                "deny_shadow",
            "allowed":
                False,
            "reason":
                "semantic_source_stale",
            "reasons": [
                "semantic_source_stale",
            ],
            "stored":
                True,
            "duplicate":
                False,
            "revision":
                1,
            "source_candidate_revision":
                6,
            "source_unified_revision":
                3,
            "current_user_excluded":
                True,
            "retrieval_observation_available":
                True,
            "retrieval_candidate_count":
                3,
            "usable_context_evidence":
                True,
            "has_current_task":
                True,
            "state_included":
                False,
            "trusted_fact_count":
                1,
            "constraint_count":
                0,
            "decision_count":
                0,
            "open_item_count":
                0,
            "plan_count":
                0,
            "memory_count":
                0,
            "recent_context_count":
                0,
            "estimated_tokens":
                80,
            "token_budget":
                1200,
            "expected_unified_revision":
                3,
            "observed_unified_revision":
                3,
            # These must never reach the log.
            "conversation_id":
                CID,
            "dangerous_text":
                "SECRET",
            "query":
                "secret-query",
            "memory_text":
                "secret-memory",
            "current_user_text":
                "secret-user",
        }

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                _CONFIDENCE_ENV:
                    "1",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "update_context_confidence_gate",
                Mock(
                    return_value=report
                ),
            ):
                with self.assertLogs(
                    "ombre_brain.gateway",
                    level="INFO",
                ) as captured:
                    coordinator.observe_context_confidence(
                        CID,
                        expected_unified_revision=3,
                    )

        logged = "\n".join(
            record.getMessage()
            for record in captured.records
        )

        for secret in (
            CID,
            "SECRET",
            "secret-query",
            "secret-memory",
            "secret-user",
            "dangerous_text",
        ):
            with self.subTest(
                secret=secret
            ):
                self.assertNotIn(
                    secret,
                    logged,
                )

        self.assertIn(
            "[gateway.context_confidence_gate]",
            logged,
        )

        self.assertIn(
            '"decision":"deny_shadow"',
            logged.replace(
                " ",
                "",
            ),
        )

    async def test_confidence_deny_is_shadow_decision_only(
        self,
    ):
        # The deny decision must never reach the live pipeline: the
        # coordinator never passes a confidence result into the real
        # injection selector and never uses it as a freshness
        # condition.
        body = b'{"messages":[]}'

        deny = Mock(
            return_value=_deny_report()
        )

        selector = Mock(
            side_effect=AssertionError(
                "confidence must not reach the selector"
            )
        )

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                _CONFIDENCE_ENV:
                    "1",
            },
            clear=False,
        ):
            with patch.object(
                coordinator,
                "observe_context_sources",
                Mock(return_value=CID),
            ), patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                AsyncMock(
                    return_value={
                        "stored": True,
                        "revision": 1,
                    }
                ),
            ), patch.object(
                coordinator,
                "update_context_confidence_gate",
                deny,
            ), patch.object(
                coordinator,
                "select_context_injected_body",
                selector,
            ):
                selected = await (
                    coordinator.run_context_pipeline(
                        body
                    )
                )

        deny.assert_called_once_with(
            CID,
            expected_unified_revision=1,
        )

        # Real injection stays OFF by default, so the selector is
        # not called and the body is unchanged.
        selector.assert_not_called()

        self.assertIs(
            selected,
            body,
        )

    async def _run_real_injection_with_confidence(
        self,
        body,
        *,
        confidence_on,
    ):
        selected_body = body + b" "

        report = {
            "version":
                "context-real-injection.v1",
            "enabled":
                True,
            "applied":
                True,
            "reason":
                "injection_applied",
            "conversation_id":
                CID,
            "original_sha256":
                hashlib.sha256(
                    body
                ).hexdigest(),
            "selected_sha256":
                hashlib.sha256(
                    selected_body
                ).hexdigest(),
        }

        confidence_calls: list = []
        mutation_calls: list = []
        selector_freshness: list = []

        real_select = (
            coordinator.select_real_injection
        )

        def mutation_impl(
            conversation_id,
            forward_body,
        ):
            mutation_calls.append(
                conversation_id
            )

        def selector_impl(
            conversation_id,
            forward_body,
            *,
            context_chain_fresh,
        ):
            selector_freshness.append(
                context_chain_fresh
            )

            return real_select(
                conversation_id,
                forward_body,
                context_chain_fresh=
                    context_chain_fresh,
            )

        confidence = Mock(
            side_effect=lambda *args, **kwargs: (
                confidence_calls.append(args)
                or _deny_report()
            )
        )

        injector = Mock(
            return_value=(
                selected_body,
                report,
            )
        )

        environ = {
            **_ISOLATED_ENV,
            "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                "1",
            _CONFIDENCE_ENV:
                "1" if confidence_on else "0",
        }

        with patch.dict(
            os.environ,
            environ,
            clear=False,
        ):
            with patch.object(
                coordinator,
                "observe_context_sources",
                Mock(return_value=CID),
            ), patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                AsyncMock(
                    return_value={
                        "stored": True,
                        "revision": 5,
                    }
                ),
            ), patch.object(
                coordinator,
                "update_context_injection_preview",
                Mock(
                    return_value={
                        "stored": True,
                        "revision": 1,
                        "source_revision": 5,
                    }
                ),
            ), patch.object(
                coordinator,
                "update_context_injection_gate",
                Mock(
                    return_value={
                        "stored": True,
                        "revision": 1,
                        "decision":
                            "allow_shadow",
                        "allowed": True,
                    }
                ),
            ), patch.object(
                coordinator,
                "update_context_confidence_gate",
                confidence,
            ), patch.object(
                coordinator,
                "observe_request_mutation",
                Mock(side_effect=mutation_impl),
            ), patch.object(
                coordinator,
                "select_real_injection",
                Mock(side_effect=selector_impl),
            ), patch.object(
                coordinator,
                "select_context_injected_body",
                injector,
            ):
                selected = await (
                    coordinator.run_context_pipeline(
                        body
                    )
                )

        return (
            selected,
            confidence_calls,
            mutation_calls,
            selector_freshness,
        )

    async def test_confidence_deny_does_not_block_real_injection(
        self,
    ):
        # Strict shadow-only proof: with Real Injection ON, a fresh
        # chain and an allowing Gate, a Confidence deny must not stop
        # the Mutation observer or the Real Injection selector, and
        # the resulting body must equal the run with Confidence OFF.
        body = b'{"messages":[{"role":"user","content":"hi"}]}'

        (
            selected_with,
            confidence_calls,
            mutation_with,
            freshness_with,
        ) = await (
            self._run_real_injection_with_confidence(
                body,
                confidence_on=True,
            )
        )

        (
            selected_without,
            confidence_off_calls,
            mutation_without,
            freshness_without,
        ) = await (
            self._run_real_injection_with_confidence(
                body,
                confidence_on=False,
            )
        )

        # Confidence ran and denied...
        self.assertEqual(
            len(confidence_calls),
            1,
        )

        self.assertEqual(
            confidence_off_calls,
            [],
        )

        # ...but neither the Mutation observer nor the Real
        # Injection selector was skipped...
        self.assertEqual(
            mutation_with,
            [CID],
        )

        self.assertEqual(
            mutation_without,
            [CID],
        )

        self.assertEqual(
            freshness_with,
            [True],
        )

        self.assertEqual(
            freshness_without,
            [True],
        )

        # ...and the injected body is identical to the run without
        # Confidence.
        self.assertNotEqual(
            selected_with,
            body,
        )

        self.assertEqual(
            selected_with,
            selected_without,
        )


class ConfidenceGateGatewayThinnessTests(
    unittest.TestCase,
):

    def test_gateway_does_not_import_confidence_gate(
        self,
    ):
        source = _GATEWAY_PATH.read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            "context_confidence_gate",
            source,
        )

        self.assertNotIn(
            "confidence",
            source,
        )

    def test_gateway_still_uses_single_context_entry(
        self,
    ):
        self.assertIs(
            gateway.run_context_pipeline,
            coordinator.run_context_pipeline,
        )


if __name__ == "__main__":
    unittest.main()