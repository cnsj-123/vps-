from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
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
    "gateway_memory_flash_test_target",
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

from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)


CID = "ctx_0123456789abcdef"

_REQUEST_ID_RE = re.compile(
    r"^ctxreq_[0-9a-f]{32}$"
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
    _CONFIDENCE_ENV,
    _FLASH_ENV,
    _LEDGER_ENV,
)

_ISOLATED_ENV = {
    flag: "0" for flag in _ALL_FLAGS
}


def _allow_confidence(
    revision=1,
):
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
        "reasons":
            [],
        "stored":
            True,
        "duplicate":
            False,
        "revision":
            revision,
    }


def _memory(
    memory_id,
    content="cue text",
    name=None,
):
    item = {
        "id": memory_id,
        "content": content,
    }

    if name is not None:
        item["metadata"] = {"name": name}

    return item


def _unified_artifact(
    memories,
    *,
    revision=5,
):
    return {
        "version":
            "unified-context-candidate.v1",
        "conversation_id": CID,
        "revision": revision,
        "sections": {
            "memories": memories,
        },
        "telemetry": {
            "current_user_excluded": True,
            "memory_count": len(memories),
        },
    }


def _write_unified(
    root,
    memories,
    *,
    revision=5,
):
    path = (
        Path(root)
        / "unified_context_candidate"
        / (CID + ".json")
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            _unified_artifact(
                memories,
                revision=revision,
            )
        ),
        encoding="utf-8",
    )


def _flash_dir(root):
    return (
        Path(root)
        / "memory_flash"
        / CID
    )


def _ledger_dir(root):
    return (
        Path(root)
        / "exposure_ledger"
        / CID
    )


def _read_only_file(directory):
    files = sorted(
        directory.iterdir()
    )

    if not files:
        return None

    return json.loads(
        files[0].read_text(
            encoding="utf-8"
        )
    )


class MemoryFlashPipelineTests(
    unittest.IsolatedAsyncioTestCase,
):

    async def _run(
        self,
        body,
        *,
        root,
        flash_on,
        ledger_on,
        real_injection=False,
        memories=None,
    ):
        if memories is None:
            memories = [
                _memory("mem-1"),
                _memory("mem-2"),
            ]

        _write_unified(root, memories)

        selected_body = body + b" "

        applied = {
            "version":
                "context-real-injection.v1",
            "enabled":
                True,
            "applied":
                True,
            "reason":
                "injection_applied",
            "conversation_id": CID,
            "original_sha256":
                hashlib.sha256(
                    body
                ).hexdigest(),
            "selected_sha256":
                hashlib.sha256(
                    selected_body
                ).hexdigest(),
        }

        environ = {
            **_ISOLATED_ENV,
            _CONFIDENCE_ENV: "1",
            _FLASH_ENV:
                "1" if flash_on else "0",
            _LEDGER_ENV:
                "1" if ledger_on else "0",
            "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                "1" if real_injection else "0",
            "OMBRE_CONTEXT_STATE_DIR": root,
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
                "update_context_confidence_gate",
                Mock(
                    return_value=(
                        _allow_confidence()
                    )
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
                "select_context_injected_body",
                Mock(
                    return_value=(
                        selected_body,
                        applied,
                    )
                ),
            ):
                selected = await (
                    coordinator.run_context_pipeline(
                        body
                    )
                )

        return selected, selected_body

    async def test_flash_on_does_not_change_selected_body(
        self,
    ):
        body = (
            b'{"messages":[{"role":"user",'
            b'"content":"hi"}]}'
        )

        with tempfile.TemporaryDirectory() as off_root:
            (
                selected_off,
                _expected,
            ) = await self._run(
                body,
                root=off_root,
                flash_on=False,
                ledger_on=False,
                real_injection=True,
            )

            self.assertFalse(
                _flash_dir(off_root).exists()
            )

        with tempfile.TemporaryDirectory() as on_root:
            (
                selected_on,
                expected,
            ) = await self._run(
                body,
                root=on_root,
                flash_on=True,
                ledger_on=True,
                real_injection=True,
            )

            flash = _read_only_file(
                _flash_dir(on_root)
            )

            ledger = _read_only_file(
                _ledger_dir(on_root)
            )

        # Real Injection is the only thing that may change the body.
        self.assertNotEqual(
            selected_on,
            body,
        )
        self.assertEqual(
            selected_on,
            expected,
        )
        self.assertEqual(
            selected_off,
            selected_on,
        )

        # ...and the Flash really ran and produced cues.
        self.assertEqual(
            flash["surfaced_count"],
            2,
        )
        self.assertEqual(
            ledger["surfaced_count"],
            2,
        )

    async def test_three_cues_never_change_body(
        self,
    ):
        body = b'{"messages":[]}'

        with tempfile.TemporaryDirectory() as root:
            (
                selected,
                expected,
            ) = await self._run(
                body,
                root=root,
                flash_on=True,
                ledger_on=True,
                real_injection=True,
                memories=[
                    _memory("mem-1"),
                    _memory("mem-2"),
                    _memory("mem-3"),
                ],
            )

            flash = _read_only_file(
                _flash_dir(root)
            )

        self.assertEqual(
            flash["surfaced_count"],
            3,
        )
        self.assertEqual(
            selected,
            expected,
        )
        self.assertNotEqual(
            selected,
            body,
        )

    async def test_ledger_on_flash_off_has_zero_surfaced(
        self,
    ):
        body = b'{"messages":[]}'

        with tempfile.TemporaryDirectory() as root:
            await self._run(
                body,
                root=root,
                flash_on=False,
                ledger_on=True,
            )

            self.assertFalse(
                _flash_dir(root).exists()
            )

            ledger = _read_only_file(
                _ledger_dir(root)
            )

        self.assertEqual(
            ledger["retrieved_count"],
            2,
        )
        self.assertEqual(
            ledger["surfaced_count"],
            0,
        )

    async def test_flash_on_ledger_off_writes_no_ledger(
        self,
    ):
        body = b'{"messages":[]}'

        with tempfile.TemporaryDirectory() as root:
            await self._run(
                body,
                root=root,
                flash_on=True,
                ledger_on=False,
            )

            self.assertTrue(
                _flash_dir(root).exists()
            )
            self.assertFalse(
                _ledger_dir(root).exists()
            )

    async def test_flash_failure_is_fail_open(
        self,
    ):
        body = b'{"messages":[]}'

        with tempfile.TemporaryDirectory() as root:
            _write_unified(
                root,
                [_memory("mem-1")],
            )

            with patch.dict(
                os.environ,
                {
                    **_ISOLATED_ENV,
                    _CONFIDENCE_ENV: "1",
                    _FLASH_ENV: "1",
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
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
                            "revision": 5,
                        }
                    ),
                ), patch.object(
                    coordinator,
                    "update_context_confidence_gate",
                    Mock(
                        return_value=(
                            _allow_confidence()
                        )
                    ),
                ), patch.object(
                    coordinator,
                    "update_memory_flash",
                    Mock(
                        side_effect=RuntimeError(
                            "secret-failure"
                        )
                    ),
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

        self.assertIs(selected, body)

        logged = "\n".join(
            record.getMessage()
            for record in captured.records
        )

        self.assertIn(
            "[gateway.context_memory_flash]",
            logged,
        )
        self.assertIn(
            "fail_open=true",
            logged,
        )
        self.assertNotIn(
            "secret-failure",
            logged,
        )


class CognitiveRequestIdTests(
    unittest.IsolatedAsyncioTestCase,
):

    async def test_id_is_per_request_and_valid(
        self,
    ):
        body = b'{"messages":[]}'

        captured: list = []

        async def fake_gate(
            conversation_id,
            *,
            cognitive_request_id=None,
        ):
            captured.append(
                cognitive_request_id
            )
            return False

        with patch.dict(
            os.environ,
            _ISOLATED_ENV,
            clear=False,
        ):
            with patch.object(
                coordinator,
                "observe_context_sources",
                Mock(return_value=CID),
            ), patch.object(
                coordinator,
                "observe_unified_preview_gate",
                fake_gate,
            ):
                for _ in range(2):
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

        self.assertEqual(
            len(captured),
            2,
        )

        for value in captured:
            self.assertRegex(
                value,
                _REQUEST_ID_RE,
            )

        self.assertNotEqual(
            captured[0],
            captured[1],
        )

    async def test_id_never_reaches_body_or_logs(
        self,
    ):
        body = b'{"messages":[]}'

        ids: list = []

        real_flash = (
            coordinator.update_memory_flash
        )

        def wrap_flash(**kwargs):
            ids.append(
                kwargs.get(
                    "cognitive_request_id"
                )
            )
            return real_flash(**kwargs)

        with tempfile.TemporaryDirectory() as root:
            _write_unified(
                root,
                [_memory("mem-1")],
            )

            with patch.dict(
                os.environ,
                {
                    **_ISOLATED_ENV,
                    _CONFIDENCE_ENV: "1",
                    _FLASH_ENV: "1",
                    _LEDGER_ENV: "1",
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
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
                            "revision": 5,
                        }
                    ),
                ), patch.object(
                    coordinator,
                    "update_context_confidence_gate",
                    Mock(
                        return_value=(
                            _allow_confidence()
                        )
                    ),
                ), patch.object(
                    coordinator,
                    "update_memory_flash",
                    wrap_flash,
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

        self.assertEqual(len(ids), 1)

        request_id = ids[0]

        self.assertRegex(
            request_id,
            _REQUEST_ID_RE,
        )

        self.assertNotIn(
            request_id.encode("utf-8"),
            selected,
        )

        logged = "\n".join(
            record.getMessage()
            for record in captured.records
        )

        self.assertNotIn(
            request_id,
            logged,
        )


class MemoryShadowPrivacyTests(
    unittest.IsolatedAsyncioTestCase,
):

    async def test_logs_and_artifacts_are_privacy_safe(
        self,
    ):
        body = b'{"messages":[]}'

        memory_id = "mem_9f8e7d6c5b4a3210"
        raw = "SECRET_MEMORY_TEXT"
        cue = "SECRET_CUE"

        ids: list = []

        real_flash = (
            coordinator.update_memory_flash
        )

        def wrap_flash(**kwargs):
            ids.append(
                kwargs.get(
                    "cognitive_request_id"
                )
            )
            return real_flash(**kwargs)

        with tempfile.TemporaryDirectory() as root:
            _write_unified(
                root,
                [
                    _memory(
                        memory_id,
                        content=raw,
                        name=cue,
                    )
                ],
            )

            with patch.dict(
                os.environ,
                {
                    **_ISOLATED_ENV,
                    _CONFIDENCE_ENV: "1",
                    _FLASH_ENV: "1",
                    _LEDGER_ENV: "1",
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
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
                            "revision": 5,
                        }
                    ),
                ), patch.object(
                    coordinator,
                    "update_context_confidence_gate",
                    Mock(
                        return_value=(
                            _allow_confidence()
                        )
                    ),
                ), patch.object(
                    coordinator,
                    "update_memory_flash",
                    wrap_flash,
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

            flash_text = (
                _flash_dir(root)
                .joinpath(
                    ids[0] + ".json"
                )
                .read_text(
                    encoding="utf-8"
                )
            )

            ledger_text = (
                _ledger_dir(root)
                .joinpath(
                    ids[0] + ".json"
                )
                .read_text(
                    encoding="utf-8"
                )
            )

        logged = "\n".join(
            record.getMessage()
            for record in captured.records
        )

        # The live request is never touched.
        self.assertIs(selected, body)

        # No secret text / id / request id leaks into the logs.
        for secret in (
            raw,
            cue,
            memory_id,
            ids[0],
        ):
            with self.subTest(
                secret=secret
            ):
                self.assertNotIn(
                    secret,
                    logged,
                )

        # The Flash artifact may carry the bounded cue, never the
        # full raw memory.
        self.assertIn(cue, flash_text)
        self.assertNotIn(raw, flash_text)

        # The Ledger keeps the id only: no cue, no raw memory.
        self.assertIn(memory_id, ledger_text)
        self.assertNotIn(cue, ledger_text)
        self.assertNotIn(raw, ledger_text)

        # The memory-stage log lines exclude the conversation id.
        memory_lines = "\n".join(
            record.getMessage()
            for record in captured.records
            if record.getMessage().startswith(
                (
                    "[gateway.context_memory_flash]",
                    "[gateway.context_exposure_ledger]",
                )
            )
        )

        self.assertIn(
            "[gateway.context_memory_flash]",
            memory_lines,
        )
        self.assertIn(
            "[gateway.context_exposure_ledger]",
            memory_lines,
        )
        self.assertNotIn(
            CID,
            memory_lines,
        )


class MemoryStageOrderTests(
    unittest.IsolatedAsyncioTestCase,
):

    async def test_stage_order_is_unified_confidence_flash_ledger_preview_gate(
        self,
    ):
        body = b'{"messages":[]}'

        order: list = []

        artifact = _unified_artifact(
            [_memory("mem-1")]
        )

        def record(name, result):
            order.append(name)
            return result

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                _CONFIDENCE_ENV: "1",
                _FLASH_ENV: "1",
                _LEDGER_ENV: "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
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
                        "revision": 5,
                    }
                ),
            ), patch.object(
                coordinator,
                "update_context_confidence_gate",
                Mock(
                    side_effect=lambda *a, **k: (
                        record(
                            "confidence",
                            _allow_confidence(),
                        )
                    )
                ),
            ), patch.object(
                coordinator,
                "read_unified_context_candidate",
                Mock(
                    return_value=artifact
                ),
            ), patch.object(
                coordinator,
                "evaluate_surfacing_policy",
                Mock(
                    side_effect=lambda **k: (
                        record(
                            "surfacing",
                            {
                                "decision":
                                    "allow_shadow",
                                "eligible": [],
                            },
                        )
                    )
                ),
            ), patch.object(
                coordinator,
                "update_memory_flash",
                Mock(
                    side_effect=lambda **k: (
                        record(
                            "flash",
                            {
                                "mode":
                                    "shadow_only",
                                "stored": True,
                                "decision":
                                    "no_surface",
                                "reason":
                                    "none",
                                "flashes": [],
                            },
                        )
                    )
                ),
            ), patch.object(
                coordinator,
                "update_exposure_ledger",
                Mock(
                    side_effect=lambda **k: (
                        record(
                            "ledger",
                            {
                                "mode":
                                    "shadow_only",
                                "stored": True,
                                "decision":
                                    "recorded",
                                "reason":
                                    "exposure_recorded",
                                "retrieved_count":
                                    0,
                                "surfaced_count":
                                    0,
                            },
                        )
                    )
                ),
            ), patch.object(
                coordinator,
                "update_context_injection_preview",
                Mock(
                    side_effect=lambda *a, **k: (
                        record(
                            "preview",
                            {
                                "stored": True,
                                "revision": 1,
                            },
                        )
                    )
                ),
            ), patch.object(
                coordinator,
                "update_context_injection_gate",
                Mock(
                    side_effect=lambda *a, **k: (
                        record(
                            "gate",
                            {
                                "stored": True,
                                "revision": 1,
                            },
                        )
                    )
                ),
            ):
                await (
                    coordinator.run_context_pipeline(
                        body
                    )
                )

        self.assertEqual(
            order,
            [
                "confidence",
                "surfacing",
                "flash",
                "ledger",
                "preview",
                "gate",
            ],
        )

    async def test_flags_off_keeps_old_chain(
        self,
    ):
        body = b'{"messages":[]}'

        surfacing = Mock()
        flash = Mock()
        ledger = Mock()

        with patch.dict(
            os.environ,
            {
                **_ISOLATED_ENV,
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
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
                    }
                ),
            ), patch.object(
                coordinator,
                "update_context_injection_gate",
                Mock(
                    return_value={
                        "stored": True,
                        "revision": 1,
                    }
                ),
            ), patch.object(
                coordinator,
                "evaluate_surfacing_policy",
                surfacing,
            ), patch.object(
                coordinator,
                "update_memory_flash",
                flash,
            ), patch.object(
                coordinator,
                "update_exposure_ledger",
                ledger,
            ):
                selected = await (
                    coordinator.run_context_pipeline(
                        body
                    )
                )

        surfacing.assert_not_called()
        flash.assert_not_called()
        ledger.assert_not_called()

        self.assertIs(selected, body)


class MemoryFlashGatewayThinnessTests(
    unittest.TestCase,
):

    def test_gateway_does_not_import_memory_modules(
        self,
    ):
        source = _GATEWAY_PATH.read_text(
            encoding="utf-8"
        )

        for module_name in (
            "memory_flash",
            "memory_surfacing_policy",
            "exposure_ledger",
        ):
            with self.subTest(
                module=module_name
            ):
                self.assertNotIn(
                    module_name,
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