from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from unittest.mock import AsyncMock, patch

from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)
from ombrebrain.context.memory_flash_live_exposure import (
    is_valid_live_exposure_artifact,
    read_live_memory_exposure,
)
from ombrebrain.context.memory_recall_surface import (
    resolve_recall_ref,
)
from ombrebrain.context.unified_context_candidate import (
    bind_context_service,
)

import ombrebrain.context.unified_context_candidate as unified_module
from ombrebrain.maintenance import (
    memory_authority_migration_report as reporter,
)


# ---------------------------------------------------------------------
# REAL coordinator composition proof for Memory Authority Migration v0.
#
# These are NOT patched-body unit tests. They run the REAL request-local
# Context pipeline (the same entry point the Gateway uses) with the REAL
# Real-Injection machinery and the REAL Flash -> Recall Surface -> Live
# Exposure machinery, and then assert the REAL persisted Migration
# artifact the observer produced.
#
# The privacy boundary under test is the MIGRATION EVIDENCE, not the
# live Flash feature: the cue / memref are allowed to reach the model in
# the live body, but must never leak into Migration evidence.
# ---------------------------------------------------------------------

_FLASH_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW"
)
_SURFACE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW"
)
_LIVE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_LIVE_EXPOSURE"
)
_MIGRATION_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_AUTHORITY_MIGRATION_SHADOW"
)

_UPSTREAM_FLAGS = (
    "OMBRE_GATEWAY_CONTEXT_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_SNAPSHOT_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_COMPACT_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_SEMANTIC_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_SEMANTIC_STATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_TRUSTED_FACTS_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_CANDIDATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION",
    "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW",
    _FLASH_ENV,
    "OMBRE_GATEWAY_CONTEXT_EXPOSURE_LEDGER_SHADOW",
    _SURFACE_ENV,
    _LIVE_ENV,
    _MIGRATION_ENV,
)

_ISOLATED_ENV = {
    flag: "0" for flag in _UPSTREAM_FLAGS
}

_NOW = datetime(
    2026,
    9,
    26,
    12,
    0,
    tzinfo=timezone.utc,
)

_SENTINEL_FRONTEND = "FRONTEND_SECRET_LIVE_COMPOSITION"
_SENTINEL_LIVE_CUE = "LIVE_CUE_SECRET_MIGRATION"
_SENTINEL_MEMID = "OB_MEMORY_ID_SECRET_MIGRATION"

_FRONTEND_ATTACHMENT = (
    '<attachment id="front_0" '
    'filename="relevant_memories.txt" '
    'type="text/plain">'
    + _SENTINEL_FRONTEND
    + "</attachment>"
)


def _payload(attachment=False):
    """A real cache-stabilized request.

    The cache boundary sits on the assistant turn BEFORE the current
    user turn, so both Real Injection and Live Exposure have a valid
    boundary to insert after.
    """

    current = [
        {
            "type": "text",
            "text": "what did we decide earlier",
        }
    ]

    if attachment:
        current.append(
            {"type": "text", "text": _FRONTEND_ATTACHMENT}
        )

    return {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "old user"}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "old assistant",
                        "cache_control": {
                            "type": "ephemeral"
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": current,
            },
        ],
    }


def _body(attachment=False):
    return json.dumps(
        _payload(attachment=attachment),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _body_texts(body):
    """Every model-facing text segment value in a serialized body.

    The live body is re-serialized JSON, so an attachment envelope's
    quotes are escaped on the wire. Comparing DECODED text segment
    values proves byte-for-byte envelope preservation without confusing
    JSON escaping for mutation.
    """

    payload = json.loads(body)

    texts = []

    for message in payload.get("messages", []):
        content = message.get("content")

        if not isinstance(content, list):
            continue

        for part in content:
            if (
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
            ):
                texts.append(part["text"])

    return texts


def _memory(
    memory_id,
    content,
    *,
    age_hours=100,
):
    return {
        "id": memory_id,
        "content": content,
        "context_relevance": 0.9,
        "metadata": {
            "importance": 5,
            "activation_count": 3,
            "strength": 0.5,
            "weight": 0.5,
            "score": 0.5,
            "decay": 0.0,
            "archive": False,
            "last_active": (
                _NOW - timedelta(hours=age_hours)
            ).isoformat(),
        },
    }


def _source_candidates(memories):
    return {
        "state": {},
        "state_revision": 4,
        "plans": [],
        "memories": memories,
        "telemetry": {
            "retrieval_candidate_count": len(memories),
            "relevance_rejected": 0,
            "anti_echo": {},
            "dedup": {},
        },
    }


class RealCompositionTestCase(
    unittest.IsolatedAsyncioTestCase,
):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name

        self.old_root = os.environ.get(
            "OMBRE_CONTEXT_STATE_DIR"
        )
        os.environ["OMBRE_CONTEXT_STATE_DIR"] = self.root

        self.old_bound = unified_module._CONTEXT_SERVICE

    def tearDown(self):
        unified_module._CONTEXT_SERVICE = self.old_bound

        if self.old_root is None:
            os.environ.pop(
                "OMBRE_CONTEXT_STATE_DIR", None
            )
        else:
            os.environ[
                "OMBRE_CONTEXT_STATE_DIR"
            ] = self.old_root

        self.temp.cleanup()

    def bind(self, memories):
        service = AsyncMock()
        service.get_candidates.return_value = (
            _source_candidates(memories)
        )
        bind_context_service(service)
        return service

    async def _run(self, forward_body, flags):
        environ = {
            **_ISOLATED_ENV,
            "OMBRE_CONTEXT_STATE_DIR": self.root,
            **flags,
        }

        with patch.dict(
            os.environ, environ, clear=False
        ):
            return await (
                coordinator
                .run_context_pipeline_with_binding(
                    forward_body
                )
            )

    def migration_artifacts(self, cid):
        directory = (
            Path(self.root)
            / "memory_authority_migration"
            / cid
        )

        if not directory.is_dir():
            return []

        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
        )

    def single_migration_artifact(self, cid):
        files = self.migration_artifacts(cid)

        self.assertEqual(len(files), 1)

        return json.loads(
            files[0].read_text(encoding="utf-8")
        )


class RealContextInjectionFrontendTests(
    RealCompositionTestCase,
):
    """REAL Real Injection + Frontend preservation + Migration observer."""

    async def test_real_injection_preserves_frontend_attachment(
        self,
    ):
        forward_body = _body(attachment=True)

        self.bind(
            [_memory("m1", "an old decision")]
        )

        selection = await self._run(
            forward_body,
            {
                "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW":
                    "1",
                "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                    "1",
                _MIGRATION_ENV: "1",
            },
        )

        final_body = selection.body

        # The REAL Real-Injection machinery actually changed the body.
        self.assertNotEqual(final_body, forward_body)

        # Live Exposure is OFF, so the final body IS the selected body.
        self.assertIsNone(
            read_live_memory_exposure(
                conversation_id=selection.conversation_id,
                cognitive_request_id=(
                    selection.cognitive_request_id
                ),
            )
        )

        # The exact Frontend raw attachment envelope survives the
        # Real-Injection stage byte-for-byte (compared decoded, so JSON
        # escaping is not mistaken for mutation).
        self.assertIn(
            _FRONTEND_ATTACHMENT,
            _body_texts(final_body),
        )

        artifact = self.single_migration_artifact(
            selection.conversation_id
        )

        self.assertTrue(artifact["live_authority"] is False)
        self.assertTrue(
            artifact["frontend_final_observation_available"]
        )
        self.assertTrue(
            artifact["frontend_final_presence_verified"]
        )
        self.assertTrue(
            artifact["frontend_direct_mouth_active"]
        )
        self.assertFalse(
            artifact["ob_live_cue_mouth_active"]
        )

        exposure = artifact["current_exposure"]

        self.assertFalse(
            exposure["actual_ob_live_exposure_applied"]
        )
        self.assertEqual(
            exposure["actual_ob_live_exposed_count"], 0
        )
        # pre_live == final: Live Exposure did not run.
        self.assertEqual(
            exposure["pre_live_body_sha256"],
            exposure["final_body_sha256"],
        )
        self.assertFalse(exposure["live_body_changed"])


class RealFlashSurfaceLiveExposureMigrationTests(
    RealCompositionTestCase,
):
    """REAL Flash -> Recall Surface -> Live Exposure -> Migration."""

    _FLAGS = {
        "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW":
            "1",
        _FLASH_ENV: "1",
        _SURFACE_ENV: "1",
        _LIVE_ENV: "1",
        _MIGRATION_ENV: "1",
    }

    async def _dual_mouth_run(self):
        forward_body = _body(attachment=True)

        # The sentinel is the ACTUAL raw OB memory id of the real
        # source candidate, so the downstream privacy assertions are
        # not vacuous. The exact fixture object is retained and bound.
        memory_fixture = _memory(
            _SENTINEL_MEMID,
            _SENTINEL_LIVE_CUE,
        )

        self.bind([memory_fixture])

        selection = await self._run(
            forward_body, dict(self._FLAGS)
        )

        return forward_body, selection, memory_fixture

    async def test_real_dual_mouth_composition(self):
        forward_body, selection, memory_fixture = (
            await self._dual_mouth_run()
        )

        # Positive proof the sentinel really is the source memory id.
        self.assertEqual(
            memory_fixture["id"], _SENTINEL_MEMID
        )

        self.assertNotEqual(
            selection.body, forward_body
        )

        receipt = read_live_memory_exposure(
            conversation_id=selection.conversation_id,
            cognitive_request_id=(
                selection.cognitive_request_id
            ),
        )

        self.assertIsNotNone(receipt)
        self.assertTrue(
            is_valid_live_exposure_artifact(receipt)
        )
        self.assertTrue(receipt["applied"])
        self.assertGreaterEqual(
            receipt["live_exposed_count"], 1
        )

        # The Frontend attachment still exists in the final body.
        self.assertIn(
            _FRONTEND_ATTACHMENT,
            _body_texts(selection.body),
        )

        artifact = self.single_migration_artifact(
            selection.conversation_id
        )

        self.assertTrue(
            artifact["frontend_direct_mouth_active"]
        )
        self.assertTrue(
            artifact["ob_live_cue_mouth_active"]
        )
        self.assertTrue(artifact["dual_mouth_active"])

        exposure = artifact["current_exposure"]

        self.assertTrue(
            exposure["actual_ob_live_exposure_applied"]
        )
        self.assertGreaterEqual(
            exposure["actual_ob_live_exposed_count"], 1
        )
        self.assertTrue(exposure["live_body_changed"])

        metrics = artifact["metrics"]

        self.assertGreaterEqual(
            metrics["actual_ob_live_exposed_count"], 1
        )
        self.assertGreater(
            metrics[
                "actual_model_visible_memory"
                "_estimated_tokens"
            ],
            metrics[
                "frontend_direct_model_visible"
                "_estimated_tokens"
            ],
        )

    async def test_real_migration_evidence_leaks_no_live_cue_or_memref(
        self,
    ):
        forward_body, selection, memory_fixture = (
            await self._dual_mouth_run()
        )

        # POSITIVE PROOF #1: the retained source fixture really carries
        # the sentinel as its raw OB memory id.
        self.assertEqual(
            memory_fixture["id"], _SENTINEL_MEMID
        )

        # POSITIVE PROOF #2: the REAL candidate that the coordinator
        # retrieved is the SAME object, so the sentinel genuinely
        # entered the request-local source path.
        bound = (
            unified_module._CONTEXT_SERVICE
            .get_candidates.return_value
        )

        self.assertIs(
            bound["memories"][0], memory_fixture
        )
        self.assertEqual(
            bound["memories"][0]["id"],
            _SENTINEL_MEMID,
        )

        receipt = read_live_memory_exposure(
            conversation_id=selection.conversation_id,
            cognitive_request_id=(
                selection.cognitive_request_id
            ),
        )

        self.assertIsNotNone(receipt)

        memrefs = receipt["exposed_memrefs"]

        self.assertTrue(memrefs)

        memref = memrefs[0]

        # POSITIVE PROOF #3: the REAL generated memref that reached the
        # live body resolves back to exactly the sentinel raw memory id
        # in THIS request's Recall Surface. This ties the sentinel to
        # the actual live channel before any privacy boundary.
        self.assertEqual(
            resolve_recall_ref(
                conversation_id=(
                    selection.conversation_id
                ),
                cognitive_request_id=(
                    selection.cognitive_request_id
                ),
                memref=memref,
            ),
            _SENTINEL_MEMID,
        )

        # The real cue / memref DO exist in the live model-facing body
        # (that is the intentional existing channel).
        body_blob = "\n".join(_body_texts(selection.body))

        self.assertIn(_SENTINEL_LIVE_CUE, body_blob)
        self.assertIn(memref, body_blob)

        artifact = self.single_migration_artifact(
            selection.conversation_id
        )

        artifact_blob = json.dumps(
            artifact, sort_keys=True
        )

        migration_logger = logging.getLogger(
            "ombrebrain.context."
            "memory_authority_migration_shadow"
        )

        with self.assertLogs(
            migration_logger, level="INFO"
        ) as captured:
            from ombrebrain.context import (
                memory_authority_migration_shadow
                as migration_shadow,
            )

            migration_shadow.log_memory_authority_migration_report(
                {
                    "mode": artifact["mode"],
                    "stored": True,
                    "decision": "compared",
                    "reason": "comparison_stored",
                    "source_unified_revision":
                        artifact[
                            "source_unified_revision"
                        ],
                    "frontend_direct_mouth_active":
                        artifact[
                            "frontend_direct_mouth_active"
                        ],
                    "ob_live_cue_mouth_active":
                        artifact[
                            "ob_live_cue_mouth_active"
                        ],
                    "dual_mouth_active":
                        artifact["dual_mouth_active"],
                    "contract_coverage_status":
                        artifact[
                            "contract_coverage_status"
                        ],
                    "metrics": artifact["metrics"],
                }
            )

        logs = "\n".join(captured.output)

        report = reporter.build_report(root=self.root)
        reporter_blob = json.dumps(
            report, sort_keys=True
        )

        for sentinel in (
            _SENTINEL_LIVE_CUE,
            memref,
            _SENTINEL_MEMID,
            _SENTINEL_FRONTEND,
        ):
            with self.subTest(sentinel=sentinel):
                self.assertNotIn(sentinel, artifact_blob)
                self.assertNotIn(sentinel, logs)
                self.assertNotIn(sentinel, reporter_blob)

        # The reporter still counts the REAL artifact.
        self.assertEqual(
            report["valid_artifact_count"], 1
        )
        self.assertEqual(
            report["request_with_dual_mouth_count"], 1
        )


class RealObservationOnceTests(RealCompositionTestCase):
    """The ORIGINAL forward-body Source Evidence observation and the
    FINAL-body Migration observation are two DIFFERENT passes and each
    canonicalizes exactly ONCE across the real coordinator."""

    async def test_source_and_migration_each_canonicalize_once(
        self,
    ):
        forward_body = _body(attachment=True)

        self.bind(
            [_memory("m1", "an old decision")]
        )

        from ombrebrain.context import (
            memory_source_evidence as source_evidence,
        )
        from ombrebrain.context import (
            memory_authority_migration_shadow
            as migration_shadow,
        )

        real_source = source_evidence._canonicalize
        real_migration = migration_shadow._canonicalize

        counts = {"source": 0, "migration": 0}

        def spy_source(body):
            counts["source"] += 1
            return real_source(body)

        def spy_migration(body):
            counts["migration"] += 1
            return real_migration(body)

        with patch.object(
            source_evidence,
            "_canonicalize",
            side_effect=spy_source,
        ), patch.object(
            migration_shadow,
            "_canonicalize",
            side_effect=spy_migration,
        ):
            selection = await self._run(
                forward_body,
                {
                    "OMBRE_GATEWAY_CONTEXT_"
                    "UNIFIED_CANDIDATE_SHADOW": "1",
                    "OMBRE_GATEWAY_CONTEXT_"
                    "MEMORY_SOURCE_ARBITRATION_SHADOW": "1",
                    "OMBRE_GATEWAY_CONTEXT_"
                    "MEMORY_DELIVERY_SHADOW": "1",
                    _MIGRATION_ENV: "1",
                },
            )

        # Original forward-body Source Evidence: exactly one pass.
        self.assertEqual(counts["source"], 1)

        # Final-body Migration: exactly one DIFFERENT pass.
        self.assertEqual(counts["migration"], 1)

        artifact = self.single_migration_artifact(
            selection.conversation_id
        )

        self.assertTrue(artifact["live_authority"] is False)
        self.assertTrue(
            artifact["frontend_final_presence_verified"]
        )


if __name__ == "__main__":
    unittest.main()
