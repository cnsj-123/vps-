from __future__ import annotations

import ast
import hashlib
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
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)
from ombrebrain.context import (
    frontend_controlled_direct_surface as surface_module,
)
from ombrebrain.context import (
    memory_live_authority_candidate_shadow as candidate_module,
)
from ombrebrain.context import (
    memory_source_evidence as source_evidence,
)
from ombrebrain.context.frontend_controlled_direct_surface import (
    build_frontend_controlled_direct_surface,
)
from ombrebrain.context.memory_authority_migration_shadow import (
    build_memory_authority_migration_artifact,
    observe_frontend_final_presence,
)
from ombrebrain.context.memory_delivery_shadow import (
    build_memory_delivery_plan_artifact,
)
from ombrebrain.context.memory_flash import (
    candidate_memory_id,
)
from ombrebrain.context.memory_flash_live_exposure import (
    is_valid_live_exposure_artifact,
    read_live_memory_exposure,
)
from ombrebrain.context.memory_live_authority_candidate_shadow import (
    is_valid_memory_live_authority_candidate_artifact,
)
from ombrebrain.context.memory_recall_surface import (
    resolve_recall_ref,
)
from ombrebrain.context.memory_source_evidence import (
    MemorySourceEvidenceSnapshot,
    frontend_evidence_units,
    ob_evidence_units,
)
from ombrebrain.context.unified_context_candidate import (
    bind_context_service,
)
from ombrebrain.gateway import (
    OperitAdapter,
    extract_frontend_memory_segments,
)
from ombrebrain.gateway.canonicalizer import (
    REWRITE_DROP,
    REWRITE_KEEP_EXACT,
)

import ombrebrain.context.unified_context_candidate as unified_module


# ---------------------------------------------------------------------
# REAL coordinator composition + isolation proof for the Unified Memory
# Authority Compatibility Candidate Shadow v0.
#
# These run the REAL request-local Context pipeline (the same entry
# point the Gateway uses) with the REAL Real-Injection and the REAL
# Flash -> Recall Surface -> Live Exposure machinery. The candidate must
# EXPLICITLY adopt the CURRENT Frontend direct exposure while the
# returned body stays byte-for-byte identical, and it must never gain
# body authority.
# ---------------------------------------------------------------------

_CANDIDATE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_LIVE_AUTHORITY_CANDIDATE_SHADOW"
)
_MIGRATION_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_AUTHORITY_MIGRATION_SHADOW"
)
_DELIVERY_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_DELIVERY_SHADOW"
)
_ARBITRATION_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_SOURCE_ARBITRATION_SHADOW"
)
_FLASH_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW"
)
_SURFACE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW"
)
_LIVE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_LIVE_EXPOSURE"
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
    _DELIVERY_ENV,
    _ARBITRATION_ENV,
    _CANDIDATE_ENV,
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

_SENTINEL_FRONTEND = "FRONTEND_CONTROL_SECRET"
_SENTINEL_OB = "OB_CONTENT_CONTROL_SECRET"
_SENTINEL_MEMID = "OB_MEMORY_ID_CONTROL_SECRET"
_SENTINEL_LIVE_CUE = "LIVE_CUE_CONTROL_SECRET"

_FRONTEND_ATTACHMENT = (
    '<attachment id="front_0" '
    'filename="relevant_memories.txt" '
    'type="text/plain">'
    + _SENTINEL_FRONTEND
    + "</attachment>"
)

_CANDIDATE_LOGGER = (
    "ombrebrain.context."
    "memory_live_authority_candidate_shadow"
)


def _payload(attachment=False):
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


def _memory(memory_id, content, *, age_hours=100):
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

    def candidate_artifacts(self, cid):
        directory = (
            Path(self.root)
            / "memory_live_authority_candidate"
            / cid
        )

        if not directory.is_dir():
            return []

        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
        )

    def single_candidate_artifact(self, cid):
        files = self.candidate_artifacts(cid)

        self.assertEqual(len(files), 1)

        return json.loads(
            files[0].read_text(encoding="utf-8")
        )


class CandidateOnlyFlagMatrixTests(
    RealCompositionTestCase,
):
    """Candidate ON alone drives only its required shadow evidence."""

    async def test_flag_off_is_a_true_noop(self):
        forward_body = _body(attachment=True)

        self.bind([_memory("m1", "an old decision")])

        selection = await self._run(forward_body, {})

        # Byte-for-byte unchanged.
        self.assertEqual(
            selection.body, forward_body
        )

        # No candidate artifact was written and no candidate directory
        # was ever created. With every flag OFF the pipeline is not even
        # bound, so the binding is None.
        self.assertIsNone(selection.conversation_id)
        self.assertFalse(
            (
                Path(self.root)
                / "memory_live_authority_candidate"
            ).exists()
        )

    async def test_candidate_only_builds_everything_once(self):
        forward_body = _body(attachment=True)

        self.bind([_memory("m1", "an old decision")])

        real_source = (
            source_evidence
            .build_memory_source_evidence_snapshot
        )
        real_surface = (
            coordinator
            .build_frontend_controlled_direct_surface
        )
        real_candidate_build = (
            candidate_module
            .build_memory_live_authority_candidate_artifact
        )

        counts = {
            "source": 0,
            "policy": 0,
            "delivery": 0,
            "migration": 0,
            "surface": 0,
            "candidate": 0,
        }

        def spy_source(*args, **kwargs):
            counts["source"] += 1
            return real_source(*args, **kwargs)

        def spy_policy(*args, **kwargs):
            counts["policy"] += 1
            return real_policy(*args, **kwargs)

        def spy_delivery(*args, **kwargs):
            counts["delivery"] += 1
            return real_delivery(*args, **kwargs)

        def spy_migration(*args, **kwargs):
            counts["migration"] += 1
            return real_migration(*args, **kwargs)

        def spy_surface(*args, **kwargs):
            counts["surface"] += 1
            return real_surface(*args, **kwargs)

        def spy_candidate_build(*args, **kwargs):
            counts["candidate"] += 1
            return real_candidate_build(*args, **kwargs)

        real_policy = coordinator.evaluate_surfacing_policy
        real_delivery = (
            coordinator.update_memory_delivery_plan
        )
        real_migration = (
            coordinator.update_memory_authority_migration
        )

        with patch.object(
            source_evidence,
            "build_memory_source_evidence_snapshot",
            side_effect=spy_source,
        ), patch.object(
            coordinator,
            "evaluate_surfacing_policy",
            side_effect=spy_policy,
        ), patch.object(
            coordinator,
            "update_memory_delivery_plan",
            side_effect=spy_delivery,
        ), patch.object(
            coordinator,
            "update_memory_authority_migration",
            side_effect=spy_migration,
        ), patch.object(
            coordinator,
            "build_frontend_controlled_direct_surface",
            side_effect=spy_surface,
        ), patch.object(
            candidate_module,
            "build_memory_live_authority_candidate_artifact",
            side_effect=spy_candidate_build,
        ):
            selection = await self._run(
                forward_body,
                {_CANDIDATE_ENV: "1"},
            )

        self.assertEqual(counts["source"], 1)
        self.assertEqual(counts["policy"], 1)
        self.assertEqual(counts["delivery"], 1)
        self.assertEqual(counts["migration"], 1)
        self.assertEqual(counts["surface"], 1)
        self.assertEqual(counts["candidate"], 1)

        # No live memory channel ran: the returned body is the exact
        # forward body.
        self.assertEqual(selection.body, forward_body)

        self.assertIsNone(
            read_live_memory_exposure(
                conversation_id=selection.conversation_id,
                cognitive_request_id=(
                    selection.cognitive_request_id
                ),
            )
        )

        artifact = self.single_candidate_artifact(
            selection.conversation_id
        )

        self.assertTrue(
            is_valid_memory_live_authority_candidate_artifact(
                artifact
            )
        )
        self.assertEqual(
            artifact["final_body_sha256"],
            hashlib.sha256(forward_body).hexdigest(),
        )
        self.assertEqual(
            artifact["candidate_body_sha256"],
            hashlib.sha256(forward_body).hexdigest(),
        )
        self.assertTrue(
            artifact["candidate_body_byte_identical"]
        )
        self.assertFalse(artifact["live_cutover_ready"])
        self.assertFalse(artifact["live_authority"])

    async def test_delivery_migration_candidate_share_one_chain(
        self,
    ):
        forward_body = _body(attachment=True)

        self.bind([_memory("m1", "an old decision")])

        counts = {
            "source": 0,
            "delivery": 0,
            "migration": 0,
            "candidate": 0,
        }

        real_source = (
            source_evidence
            .build_memory_source_evidence_snapshot
        )
        real_delivery = (
            coordinator.update_memory_delivery_plan
        )
        real_migration = (
            coordinator.update_memory_authority_migration
        )
        real_candidate_build = (
            candidate_module
            .build_memory_live_authority_candidate_artifact
        )

        def spy_source(*args, **kwargs):
            counts["source"] += 1
            return real_source(*args, **kwargs)

        def spy_delivery(*args, **kwargs):
            counts["delivery"] += 1
            return real_delivery(*args, **kwargs)

        def spy_migration(*args, **kwargs):
            counts["migration"] += 1
            return real_migration(*args, **kwargs)

        def spy_candidate_build(*args, **kwargs):
            counts["candidate"] += 1
            return real_candidate_build(*args, **kwargs)

        with patch.object(
            source_evidence,
            "build_memory_source_evidence_snapshot",
            side_effect=spy_source,
        ), patch.object(
            coordinator,
            "update_memory_delivery_plan",
            side_effect=spy_delivery,
        ), patch.object(
            coordinator,
            "update_memory_authority_migration",
            side_effect=spy_migration,
        ), patch.object(
            candidate_module,
            "build_memory_live_authority_candidate_artifact",
            side_effect=spy_candidate_build,
        ):
            selection = await self._run(
                forward_body,
                {
                    _DELIVERY_ENV: "1",
                    _MIGRATION_ENV: "1",
                    _CANDIDATE_ENV: "1",
                },
            )

        # Exactly one of each -- no duplicated upstream chain.
        self.assertEqual(counts["source"], 1)
        self.assertEqual(counts["delivery"], 1)
        self.assertEqual(counts["migration"], 1)
        self.assertEqual(counts["candidate"], 1)

        self.assertEqual(selection.body, forward_body)

    async def test_arbitration_and_candidate_share_one_evidence(
        self,
    ):
        forward_body = _body(attachment=True)

        self.bind([_memory("m1", "an old decision")])

        counts = {"source": 0, "walk": 0}

        real_source = (
            source_evidence
            .build_memory_source_evidence_snapshot
        )
        real_walk = (
            candidate_module._final_body_control_walk
        )

        def spy_source(*args, **kwargs):
            counts["source"] += 1
            return real_source(*args, **kwargs)

        def spy_walk(*args, **kwargs):
            counts["walk"] += 1
            return real_walk(*args, **kwargs)

        with patch.object(
            source_evidence,
            "build_memory_source_evidence_snapshot",
            side_effect=spy_source,
        ), patch.object(
            candidate_module,
            "_final_body_control_walk",
            side_effect=spy_walk,
        ):
            selection = await self._run(
                forward_body,
                {
                    _ARBITRATION_ENV: "1",
                    _DELIVERY_ENV: "1",
                    _MIGRATION_ENV: "1",
                    _CANDIDATE_ENV: "1",
                },
            )

        # ONE forward-body Source Evidence canonicalization shared by
        # arbitration and delivery...
        self.assertEqual(counts["source"], 1)

        # ...and a SEPARATE final-body control walk.
        self.assertEqual(counts["walk"], 1)

        artifact = self.single_candidate_artifact(
            selection.conversation_id
        )

        self.assertTrue(
            artifact["frontend_exact_envelope_verified_count"]
            if "frontend_exact_envelope_verified_count"
            in artifact
            else artifact["metrics"][
                "frontend_exact_envelope_verified_count"
            ]
            >= 1
        )


class RealInjectionCompositionTests(
    RealCompositionTestCase,
):
    """REAL Real Injection + Frontend preservation + Candidate."""

    async def test_real_injection_preserves_frontend_envelope(
        self,
    ):
        forward_body = _body(attachment=True)

        self.bind([_memory("m1", "an old decision")])

        selection = await self._run(
            forward_body,
            {
                "OMBRE_GATEWAY_CONTEXT_"
                "UNIFIED_CANDIDATE_SHADOW": "1",
                "OMBRE_GATEWAY_CONTEXT_"
                "INJECTION_PREVIEW_SHADOW": "1",
                "OMBRE_GATEWAY_CONTEXT_"
                "INJECTION_GATE_SHADOW": "1",
                "OMBRE_GATEWAY_CONTEXT_"
                "REQUEST_MUTATION_SHADOW": "1",
                "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION":
                    "1",
                _CANDIDATE_ENV: "1",
            },
        )

        final_body = selection.body

        # Real Injection actually changed the body; Live Exposure OFF so
        # final == selected.
        self.assertNotEqual(final_body, forward_body)

        self.assertIsNone(
            read_live_memory_exposure(
                conversation_id=selection.conversation_id,
                cognitive_request_id=(
                    selection.cognitive_request_id
                ),
            )
        )

        self.assertIn(
            _FRONTEND_ATTACHMENT,
            _body_texts(final_body),
        )

        artifact = self.single_candidate_artifact(
            selection.conversation_id
        )

        # The candidate bound the exact Final Frontend envelope and
        # authorized preserve_direct_data, with a byte-identical proof
        # against the ACTUAL final body.
        self.assertEqual(
            artifact["final_body_sha256"],
            hashlib.sha256(final_body).hexdigest(),
        )
        self.assertEqual(
            artifact["candidate_body_sha256"],
            hashlib.sha256(final_body).hexdigest(),
        )
        self.assertEqual(
            artifact["final_body_bytes"], len(final_body)
        )
        self.assertTrue(
            artifact["candidate_body_byte_identical"]
        )
        self.assertEqual(
            artifact["frontend_control_action"],
            "preserve_direct_data",
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics[
                "frontend_controlled_direct_candidate_count"
            ],
            1,
        )
        self.assertEqual(
            metrics[
                "frontend_exact_envelope_verified_count"
            ],
            1,
        )


class RealObLiveExposureCompositionTests(
    RealCompositionTestCase,
):
    """REAL Flash -> Recall Surface -> Live Exposure + Candidate."""

    _FLAGS = {
        "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW":
            "1",
        _FLASH_ENV: "1",
        _SURFACE_ENV: "1",
        _LIVE_ENV: "1",
        _CANDIDATE_ENV: "1",
    }

    async def _dual_mouth_run(self):
        forward_body = _body(attachment=True)

        memory_fixture = _memory(
            _SENTINEL_MEMID,
            _SENTINEL_LIVE_CUE,
        )

        self.bind([memory_fixture])

        selection = await self._run(
            forward_body, dict(self._FLAGS)
        )

        return forward_body, selection, memory_fixture

    async def test_candidate_leaves_live_ob_cue_untouched(
        self,
    ):
        forward_body, selection, memory_fixture = (
            await self._dual_mouth_run()
        )

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

        # Frontend attachment survived the live body.
        self.assertIn(
            _FRONTEND_ATTACHMENT,
            _body_texts(selection.body),
        )

        artifact = self.single_candidate_artifact(
            selection.conversation_id
        )

        # Byte-identical candidate proof against the ACTUAL live body.
        self.assertEqual(
            artifact["final_body_sha256"],
            hashlib.sha256(selection.body).hexdigest(),
        )
        self.assertEqual(
            artifact["candidate_body_sha256"],
            hashlib.sha256(selection.body).hexdigest(),
        )
        self.assertTrue(
            artifact["candidate_body_byte_identical"]
        )
        self.assertEqual(
            artifact["ob_control_strategy"],
            "delegate_existing_ob_control_plane",
        )

        metrics = artifact["metrics"]

        # The OB live channel is delegated, not re-implemented.
        self.assertGreaterEqual(
            metrics["ob_existing_cue_contract_count"]
            + metrics["ob_policy_unknown_count"],
            1,
        )
        self.assertEqual(
            metrics[
                "frontend_controlled_direct_candidate_count"
            ],
            1,
        )

    async def test_candidate_artifact_leaks_no_cue_or_memref(
        self,
    ):
        forward_body, selection, memory_fixture = (
            await self._dual_mouth_run()
        )

        self.assertEqual(
            candidate_memory_id(memory_fixture),
            _SENTINEL_MEMID,
        )

        receipt = read_live_memory_exposure(
            conversation_id=selection.conversation_id,
            cognitive_request_id=(
                selection.cognitive_request_id
            ),
        )

        memrefs = receipt["exposed_memrefs"]

        self.assertTrue(memrefs)

        memref = memrefs[0]

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

        artifact = self.single_candidate_artifact(
            selection.conversation_id
        )

        artifact_blob = json.dumps(
            artifact, sort_keys=True
        )

        logger = logging.getLogger(_CANDIDATE_LOGGER)

        with self.assertLogs(
            logger, level="INFO"
        ) as captured:
            candidate_module \
                .log_memory_live_authority_candidate_report(
                    {
                        "mode": artifact["mode"],
                        "stored": True,
                        "decision": "candidate_observed",
                        "reason": "candidate_stored",
                        "strategy": artifact["strategy"],
                        "source_unified_revision":
                            artifact[
                                "source_unified_revision"
                            ],
                        "compatibility_candidate_ready":
                            artifact[
                                "compatibility_candidate_ready"
                            ],
                        "candidate_coverage_status":
                            artifact[
                                "candidate_coverage_status"
                            ],
                        "metrics": artifact["metrics"],
                    }
                )

        logs = "\n".join(captured.output)

        for sentinel in (
            _SENTINEL_FRONTEND,
            _SENTINEL_OB,
            _SENTINEL_MEMID,
            _SENTINEL_LIVE_CUE,
            memref,
        ):
            with self.subTest(sentinel=sentinel):
                self.assertNotIn(sentinel, artifact_blob)
                self.assertNotIn(sentinel, logs)


class ObserverBodyAuthorityTests(
    RealCompositionTestCase,
):
    """The coordinator ignores any body-like value from the Candidate."""

    async def test_candidate_raise_still_returns_final_body(
        self,
    ):
        forward_body = _body(attachment=True)

        self.bind([_memory("m1", "an old decision")])

        with patch.object(
            coordinator,
            "update_memory_live_authority_candidate",
            side_effect=RuntimeError("boom"),
        ):
            selection = await self._run(
                forward_body, {_CANDIDATE_ENV: "1"}
            )

        self.assertEqual(selection.body, forward_body)

    async def test_candidate_fake_bytes_still_returns_final_body(
        self,
    ):
        forward_body = _body(attachment=True)

        self.bind([_memory("m1", "an old decision")])

        with patch.object(
            coordinator,
            "update_memory_live_authority_candidate",
            return_value=b"FAKE-CANDIDATE-BODY",
        ):
            selection = await self._run(
                forward_body, {_CANDIDATE_ENV: "1"}
            )

        self.assertEqual(selection.body, forward_body)
        self.assertNotIn(
            b"FAKE-CANDIDATE-BODY", selection.body
        )

    async def test_candidate_malformed_dict_still_returns_final_body(
        self,
    ):
        forward_body = _body(attachment=True)

        self.bind([_memory("m1", "an old decision")])

        with patch.object(
            coordinator,
            "update_memory_live_authority_candidate",
            return_value={
                "stored": True,
                "artifact": {"body": "FAKE"},
                "metrics": "not-a-dict",
            },
        ):
            selection = await self._run(
                forward_body, {_CANDIDATE_ENV: "1"}
            )

        self.assertEqual(selection.body, forward_body)


class SameConversationInterleavingTests(
    unittest.TestCase,
):
    """Deterministic A/B interleaving: A never consumes B evidence."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name

        self.old_root = os.environ.get(
            "OMBRE_CONTEXT_STATE_DIR"
        )
        os.environ["OMBRE_CONTEXT_STATE_DIR"] = self.root

    def tearDown(self):
        if self.old_root is None:
            os.environ.pop(
                "OMBRE_CONTEXT_STATE_DIR", None
            )
        else:
            os.environ[
                "OMBRE_CONTEXT_STATE_DIR"
            ] = self.old_root

        self.temp.cleanup()

    def _stack(self, body, *, cid, rid):
        payload = json.loads(body)
        request = OperitAdapter().adapt(payload)

        units = frontend_evidence_units(
            extract_frontend_memory_segments(request)
        )

        snap = MemorySourceEvidenceSnapshot(
            version="memory-source-evidence.v1",
            conversation_id=cid,
            cognitive_request_id=rid,
            source_unified_revision=4,
            frontend_observation_available=True,
            ob_observation_available=True,
            units=tuple(units),
        )

        plan = build_memory_delivery_plan_artifact(
            conversation_id=cid,
            cognitive_request_id=rid,
            source_unified_revision=4,
            source_evidence_snapshot=snap,
            base_policy=None,
        )

        migration = (
            build_memory_authority_migration_artifact(
                conversation_id=cid,
                cognitive_request_id=rid,
                source_unified_revision=4,
                delivery_plan=plan,
                frontend_observation=(
                    observe_frontend_final_presence(
                        body, snap
                    )
                ),
                live_exposure_observation={
                    "applied": False,
                    "exposed_count": 0,
                    "estimated_tokens": 0,
                    "byte_delta": 0,
                },
                pre_live_body=body,
                final_body=body,
            )
        )

        surface = build_frontend_controlled_direct_surface(
            conversation_id=cid,
            cognitive_request_id=rid,
            source_unified_revision=4,
            source_evidence_snapshot=snap,
            delivery_plan=plan,
            migration=migration,
        )

        return snap, plan, migration, surface

    def _attachment(self, body, occ_id):
        return (
            '<attachment id="' + occ_id + '" '
            'filename="relevant_memories.txt" '
            'type="text/plain">'
            + body
            + "</attachment>"
        )

    def _frontend_body(self, body, occ_id):
        text = "q " + self._attachment(body, occ_id)

        return json.dumps(
            {
                "model": "example",
                "messages": [
                    {"role": "user", "content": text}
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def _workspace(self, snap, plan, migration, revision=4):
        return SimpleNamespace(
            memory_source_evidence=snap,
            memory_delivery_plan=plan,
            memory_authority_migration=migration,
            unified={"revision": revision},
            frontend_controlled_direct_surface=None,
            memory_live_authority_candidate=None,
        )

    def _artifact(self, cid, rid):
        path = (
            Path(self.root)
            / "memory_live_authority_candidate"
            / cid
            / (rid + ".json")
        )

        if not path.is_file():
            return None

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def test_request_a_never_consumes_request_b_evidence(
        self,
    ):
        cid = "ctx_0123456789abcdef"
        rid_a = "ctxreq_" + "a" * 32
        rid_b = "ctxreq_" + "b" * 32

        body_a = self._frontend_body(
            "AAA", "occ_a"
        )
        body_b = self._frontend_body(
            "BBB", "occ_b"
        )

        snap_a, plan_a, migration_a, _ = self._stack(
            body_a, cid=cid, rid=rid_a
        )
        snap_b, plan_b, migration_b, _ = self._stack(
            body_b, cid=cid, rid=rid_b
        )

        # Request A reaches its own request-local authority, then
        # PAUSES before the Candidate.
        workspace_a = self._workspace(
            snap_a, plan_a, migration_a
        )

        # Request B for the SAME conversation, but a DIFFERENT request,
        # completes the real pipeline and overwrites every
        # conversation-scoped observation.
        workspace_b = self._workspace(
            snap_b, plan_b, migration_b
        )

        with patch.dict(
            os.environ,
            {_CANDIDATE_ENV: "1"},
            clear=False,
        ):
            coordinator.observe_memory_live_authority_candidate(
                cid,
                cognitive_request_id=rid_b,
                workspace=workspace_b,
                final_body=body_b,
            )

            artifact_b = self._artifact(cid, rid_b)

            # Resume A AFTER B. A must consume ONLY its own
            # request-local workspace authority.
            coordinator.observe_memory_live_authority_candidate(
                cid,
                cognitive_request_id=rid_a,
                workspace=workspace_a,
                final_body=body_a,
            )

            artifact_a = self._artifact(cid, rid_a)

        self.assertIsNotNone(artifact_b)
        self.assertIsNotNone(artifact_a)

        # A's candidate is bound to A's FINAL body, not B's.
        self.assertEqual(
            artifact_a["final_body_sha256"],
            hashlib.sha256(body_a).hexdigest(),
        )
        self.assertNotEqual(
            artifact_a["final_body_sha256"],
            hashlib.sha256(body_b).hexdigest(),
        )

        self.assertEqual(
            artifact_b["final_body_sha256"],
            hashlib.sha256(body_b).hexdigest(),
        )

        self.assertNotEqual(
            artifact_a["cognitive_request_id"],
            artifact_b["cognitive_request_id"],
        )

        # Each artifact is bound to its OWN request, and A's evidence
        # never leaked into B.
        self.assertEqual(
            artifact_a["cognitive_request_id"], rid_a
        )
        self.assertEqual(
            artifact_b["cognitive_request_id"], rid_b
        )


class StaticSafetyTests(unittest.TestCase):

    _FORBIDDEN = (
        "select_context_injected_body",
        "select_live_memory_flash_body",
        "get_candidates",
        "embedding",
        "rerank",
        "touch_many",
        "reinforce",
        "record_usage",
        "archive",
        "restore",
    )

    def _identifiers(self, source_path):
        tree = ast.parse(
            Path(source_path).read_text(
                encoding="utf-8"
            )
        )

        names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)

        return names

    def test_candidate_and_surface_have_no_live_call_paths(
        self,
    ):
        for path in (
            Path(
                "src/ombrebrain/context/"
                "memory_live_authority_candidate_shadow.py"
            ),
            Path(
                "src/ombrebrain/context/"
                "frontend_controlled_direct_surface.py"
            ),
        ):
            identifiers = self._identifiers(path)

            for forbidden in self._FORBIDDEN:
                self.assertNotIn(
                    forbidden,
                    identifiers,
                    "%s in %s" % (forbidden, path),
                )

    def test_candidate_production_never_emits_drop(self):
        body = json.dumps(
            {
                "model": "example",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "q "
                            '<attachment id="front_0" '
                            'filename="relevant_memories.txt" '
                            'type="text/plain">'
                            + _SENTINEL_FRONTEND
                            + "</attachment>"
                        ),
                    }
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8")

        payload = json.loads(body)
        request = OperitAdapter().adapt(payload)
        units = frontend_evidence_units(
            extract_frontend_memory_segments(request)
        )

        snap = MemorySourceEvidenceSnapshot(
            version="memory-source-evidence.v1",
            conversation_id="ctx_0123456789abcdef",
            cognitive_request_id="ctxreq_" + "a" * 32,
            source_unified_revision=4,
            frontend_observation_available=True,
            ob_observation_available=True,
            units=tuple(units),
        )

        plan = build_memory_delivery_plan_artifact(
            conversation_id=snap.conversation_id,
            cognitive_request_id=(
                snap.cognitive_request_id
            ),
            source_unified_revision=4,
            source_evidence_snapshot=snap,
            base_policy=None,
        )

        migration = (
            build_memory_authority_migration_artifact(
                conversation_id=snap.conversation_id,
                cognitive_request_id=(
                    snap.cognitive_request_id
                ),
                source_unified_revision=4,
                delivery_plan=plan,
                frontend_observation=(
                    observe_frontend_final_presence(
                        body, snap
                    )
                ),
                live_exposure_observation={
                    "applied": False,
                    "exposed_count": 0,
                    "estimated_tokens": 0,
                    "byte_delta": 0,
                },
                pre_live_body=body,
                final_body=body,
            )
        )

        surface = build_frontend_controlled_direct_surface(
            conversation_id=snap.conversation_id,
            cognitive_request_id=(
                snap.cognitive_request_id
            ),
            source_unified_revision=4,
            source_evidence_snapshot=snap,
            delivery_plan=plan,
            migration=migration,
        )

        self.assertTrue(surface.ok)

        actions: list[Any] = []

        real_rewrite = (
            candidate_module
            .rewrite_frontend_memory_attachments
        )

        def spy_rewrite(
            payload_, decide, *, expected_occurrences=None
        ):
            def wrapped(**kwargs):
                action = decide(**kwargs)
                actions.append(action)
                return action

            return real_rewrite(
                payload_,
                wrapped,
                expected_occurrences=expected_occurrences,
            )

        with patch.object(
            candidate_module,
            "rewrite_frontend_memory_attachments",
            side_effect=spy_rewrite,
        ):
            ok, _ = (
                candidate_module._final_body_control_walk(
                    body, surface.surface
                )
            )

        self.assertTrue(ok)
        self.assertTrue(actions)
        self.assertEqual(
            set(actions),
            {REWRITE_KEEP_EXACT},
        )
        self.assertNotIn(REWRITE_DROP, actions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    unittest.main()