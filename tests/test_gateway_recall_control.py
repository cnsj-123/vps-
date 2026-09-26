from __future__ import annotations

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

from _recall_fixtures import (
    CID,
    RID,
    memory,
)

from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)
from ombrebrain.context.retrieval_decision_shadow import (
    build_candidate_evidence,
)


_GATEWAY_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "web"
    / "gateway.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "gateway_recall_control_test_target",
    _GATEWAY_PATH,
)

if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load gateway.py")

gateway = importlib.util.module_from_spec(_SPEC)

_SPEC.loader.exec_module(gateway)


_SURFACE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW"
)
_FLASH_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW"
)
_LEDGER_ENV = (
    "OMBRE_GATEWAY_CONTEXT_EXPOSURE_LEDGER_SHADOW"
)
_CONFIDENCE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW"
)

_ALL_FLAGS = (
    "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION",
    "OMBRE_GATEWAY_CONTEXT_RECALL_ENABLED",
    "OMBRE_GATEWAY_CONTEXT_RELATED_RECALL_SHADOW",
    _CONFIDENCE_ENV,
    _FLASH_ENV,
    _LEDGER_ENV,
    _SURFACE_ENV,
)


def _unified_artifact(memories, *, revision=5):
    return {
        "version": "unified-context-candidate.v1",
        "conversation_id": CID,
        "revision": revision,
        "sections": {"memories": memories},
        "telemetry": {
            "current_user_excluded": True,
            "memory_count": len(memories),
        },
    }


def _snapshot(memories, *, revision=5):
    return {
        "version": "memory-shadow-snapshot.v1",
        "mode": "shadow_only",
        "conversation_id": CID,
        "revision": revision,
        "unified": _unified_artifact(
            memories, revision=revision
        ),
        "evidence": build_candidate_evidence(
            memories
        ),
    }


def _allow_confidence(
    revision=1, *, source_revision=5
):
    return {
        "version": "context-confidence-gate.v1",
        "mode": "shadow_only",
        "conversation_id": CID,
        "decision": "allow_shadow",
        "allowed": True,
        "reason": None,
        "reasons": [],
        "stored": True,
        "duplicate": False,
        "revision": revision,
        "source_unified_revision":
            source_revision,
    }


class GatewayThinnessTests(unittest.TestCase):
    """The gateway stays thin: Recall lives in the Context layer."""

    def setUp(self):
        self.source = _GATEWAY_PATH.read_text(
            encoding="utf-8"
        )

    def test_gateway_does_not_import_recall_modules(
        self,
    ):
        for name in (
            "recall_request",
            "related_recall",
            "memory_usage_signal",
            "memory_recall_surface",
            "recall_types",
        ):
            self.assertNotIn(name, self.source)

    def test_gateway_has_no_recall_orchestration(
        self,
    ):
        lowered = self.source.lower()

        for forbidden in (
            "request_related_recall",
            "recall_id",
            "memref",
            "anchor_memory",
            "recall_requested",
            "usage_signal",
        ):
            self.assertNotIn(
                forbidden, lowered
            )

    def test_gateway_still_only_calls_the_pipeline(
        self,
    ):
        self.assertIn(
            "run_context_pipeline", self.source
        )

    def test_streaming_passthrough_is_preserved(
        self,
    ):
        # Raw chunks are still relayed one-to-one; nothing buffers the
        # streaming response and no recall hook sits on the response
        # path.
        self.assertIn(
            "aiter_raw", self.source
        )
        self.assertIn("yield chunk", self.source)

        self.assertNotIn(
            "aiter_bytes", self.source
        )
        self.assertNotIn(
            "aiter_text", self.source
        )


class CoordinatorRecallSurfaceTests(
    unittest.IsolatedAsyncioTestCase,
):
    def _patches(self, memories):
        return (
            patch.object(
                coordinator,
                "observe_context_sources",
                Mock(return_value=CID),
            ),
            patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                AsyncMock(
                    return_value={
                        "stored": True,
                        "revision": 5,
                        "memory_shadow_snapshot":
                            _snapshot(memories),
                    }
                ),
            ),
            patch.object(
                coordinator,
                "update_context_confidence_gate",
                Mock(
                    return_value=(
                        _allow_confidence()
                    )
                ),
            ),
        )

    async def _run(
        self,
        body,
        *,
        root,
        surface,
        flash,
        ledger,
        memories,
    ):
        env = {
            flag: "0" for flag in _ALL_FLAGS
        }

        env["OMBRE_CONTEXT_STATE_DIR"] = root
        env[_CONFIDENCE_ENV] = "1"
        env[_FLASH_ENV] = (
            "1" if flash else "0"
        )
        env[_LEDGER_ENV] = (
            "1" if ledger else "0"
        )
        env[_SURFACE_ENV] = (
            "1" if surface else "0"
        )

        (
            sources_patch,
            unified_patch,
            confidence_patch,
        ) = self._patches(memories)

        with patch.dict(
            os.environ, env, clear=False
        ):
            with sources_patch, unified_patch, (
                confidence_patch
            ):
                return await (
                    coordinator.run_context_pipeline(
                        body
                    )
                )

    async def test_all_off_is_byte_identical(self):
        body = (
            b'{"messages":[{"role":"user",'
            b'"content":"hi"}]}'
        )

        with tempfile.TemporaryDirectory() as root:
            env = {
                flag: "0" for flag in _ALL_FLAGS
            }

            env["OMBRE_CONTEXT_STATE_DIR"] = root

            with patch.dict(
                os.environ, env, clear=False
            ), patch.object(
                coordinator,
                "observe_context_sources",
                Mock(return_value=None),
            ):
                selected = await (
                    coordinator.run_context_pipeline(
                        body
                    )
                )

            generated = sorted(
                str(path.relative_to(root))
                for path in Path(root).rglob(
                    "*.json"
                )
            )

        self.assertEqual(selected, body)
        self.assertEqual(generated, [])

    async def test_surface_flag_does_not_change_body(
        self,
    ):
        body = b'{"messages":[]}'

        memories = [
            memory("mem-1", "alpha note"),
            memory("mem-2", "beta note"),
        ]

        with tempfile.TemporaryDirectory() as root:
            selected_on = await self._run(
                body,
                root=root,
                surface=True,
                flash=True,
                ledger=True,
                memories=memories,
            )

            surface_files = list(
                (
                    Path(root)
                    / "recall_surface"
                    / CID
                ).glob("*.json")
            )

            surface = json.loads(
                surface_files[0].read_text(
                    encoding="utf-8"
                )
            )

        with tempfile.TemporaryDirectory() as root:
            selected_off = await self._run(
                body,
                root=root,
                surface=False,
                flash=True,
                ledger=True,
                memories=memories,
            )

            off_files = list(
                (
                    Path(root)
                    / "recall_surface"
                    / CID
                ).glob("*.json")
            )

        self.assertEqual(selected_on, body)
        self.assertEqual(selected_off, body)
        self.assertEqual(off_files, [])

        self.assertEqual(
            surface["version"],
            "memory-recall-surface.v1",
        )
        self.assertEqual(
            surface["memory_count"], 2
        )

        # the model-facing shape carries no real memory id
        blob = json.dumps(surface["surfaces"])

        self.assertNotIn("mem-1", blob)
        self.assertNotIn("mem-2", blob)

    async def test_surface_is_request_scoped(self):
        body = b'{"messages":[]}'

        with tempfile.TemporaryDirectory() as root:
            await self._run(
                body,
                root=root,
                surface=True,
                flash=True,
                ledger=True,
                memories=[
                    memory("mem-1", "alpha note")
                ],
            )

            surface_files = list(
                (
                    Path(root)
                    / "recall_surface"
                    / CID
                ).glob("*.json")
            )

        # one artifact per (conversation, request)
        self.assertEqual(len(surface_files), 1)
        self.assertTrue(
            surface_files[0].name.startswith(
                "ctxreq_"
            )
        )

    async def test_recall_flags_do_not_trigger_a_recall(
        self,
    ):
        body = b'{"messages":[]}'

        with tempfile.TemporaryDirectory() as root:
            await self._run(
                body,
                root=root,
                surface=True,
                flash=True,
                ledger=True,
                memories=[
                    memory("mem-1", "alpha note")
                ],
            )

            # A Flash alone never creates a recall / usage artifact.
            self.assertFalse(
                (
                    Path(root)
                    / "related_recall"
                ).exists()
            )
            self.assertFalse(
                (
                    Path(root)
                    / "memory_usage"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()