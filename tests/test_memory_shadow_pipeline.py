from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from unittest.mock import (
    AsyncMock,
    patch,
)

from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)
from ombrebrain.context.conversation_shadow import (
    conversation_shadow_status,
)
from ombrebrain.context.memory_surfacing_policy import (
    evaluate_surfacing_policy,
)
from ombrebrain.context.retrieval_decision_shadow import (
    build_candidate_evidence,
)
from ombrebrain.context.unified_context_candidate import (
    bind_context_service,
    build_unified_context_candidate,
)


CID = "ctx_0123456789abcdef"

_NOW = datetime(
    2026,
    9,
    26,
    12,
    0,
    tzinfo=timezone.utc,
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
    _CONFIDENCE_ENV,
    _FLASH_ENV,
    _LEDGER_ENV,
)

_ISOLATED_ENV = {
    flag: "0" for flag in _UPSTREAM_FLAGS
}


def _ts(hours):
    return (
        _NOW
        - timedelta(hours=hours)
    ).isoformat()


def _body():
    """A request body with a cache boundary, like the real gateway."""

    return json.dumps(
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text":
                                "what did we "
                                "decide earlier",
                            "cache_control": {
                                "type":
                                    "ephemeral",
                            },
                        }
                    ],
                }
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _memory(
    memory_id,
    content="a past decision",
    *,
    age_hours=100,
    name=None,
    last_active=None,
):
    metadata = {
        "importance": 5,
        "activation_count": 3,
        "strength": 0.5,
        "weight": 0.5,
        "score": 0.5,
        "decay": 0.0,
        "archive": False,
        "last_active": (
            last_active
            if last_active is not None
            else _ts(age_hours)
        ),
    }

    if name is not None:
        metadata["name"] = name

    return {
        "id": memory_id,
        "content": content,
        "context_relevance": 0.9,
        "metadata": metadata,
    }


def _conversation_candidate():
    return {
        "version":
            "conversation-context-candidate.v1",
        "conversation_id": CID,
        "revision": 7,
        "source_revision": 20,
        "sections": {
            "current_task": None,
            "trusted_facts": [],
            "constraints": [],
            "decisions": [],
            "open_items": [],
            "recent_context": [],
        },
        "telemetry": {
            "current_user_excluded": True,
        },
    }


def _source_candidates(memories):
    return {
        "state": {},
        "state_revision": 4,
        "plans": [],
        "memories": memories,
        "telemetry": {
            "retrieval_candidate_count":
                len(memories),
            "relevance_rejected": 0,
            "anti_echo": {},
            "dedup": {},
        },
    }


def _read_json_at(root, *parts):
    return json.loads(
        Path(
            root,
            *parts,
        ).read_text(encoding="utf-8")
    )


class RealPipelineTestCase(
    unittest.IsolatedAsyncioTestCase,
):

    def setUp(self):
        self.temp = (
            tempfile.TemporaryDirectory()
        )

        self.root = self.temp.name

        self.old_root = os.environ.get(
            "OMBRE_CONTEXT_STATE_DIR"
        )

        os.environ[
            "OMBRE_CONTEXT_STATE_DIR"
        ] = self.root

        import ombrebrain.context.unified_context_candidate as unified_module

        self.unified_module = (
            unified_module
        )

        self.old_bound = (
            unified_module._CONTEXT_SERVICE
        )

    def tearDown(self):
        self.unified_module._CONTEXT_SERVICE = (
            self.old_bound
        )

        if self.old_root is None:
            os.environ.pop(
                "OMBRE_CONTEXT_STATE_DIR",
                None,
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

    def path(self, *parts):
        return Path(
            self.root,
            *parts,
        )

    def read_json(self, *parts):
        return json.loads(
            self.path(*parts).read_text(
                encoding="utf-8"
            )
        )

    def conversation_id(self):
        conversations = (
            conversation_shadow_status()[
                "conversations"
            ]
        )

        self.assertTrue(conversations)

        return conversations[-1][
            "conversation_id"
        ]


class SourcePipelineDependencyTests(
    RealPipelineTestCase,
):
    """Enabling a downstream flag must drive the upstream chain.

    ``observe_context_sources`` is NOT mocked here, so this proves the
    real dependency chain.
    """

    async def _run(
        self,
        *,
        flash,
        ledger,
        confidence,
    ):
        environ = {
            **_ISOLATED_ENV,
            _CONFIDENCE_ENV:
                "1" if confidence else "0",
            _FLASH_ENV:
                "1" if flash else "0",
            _LEDGER_ENV:
                "1" if ledger else "0",
        }

        with patch.dict(
            os.environ,
            environ,
            clear=False,
        ):
            return await (
                coordinator.run_context_pipeline(
                    _body()
                )
            )

    async def test_memory_only_flag_drives_prerequisites(
        self,
    ):
        self.bind(
            [
                _memory("m1", "past decision one"),
                _memory("m2", "past decision two"),
            ]
        )

        selected = await self._run(
            flash=True,
            ledger=True,
            confidence=True,
        )

        cid = self.conversation_id()

        # The whole upstream chain really ran, with every legacy
        # Context shadow flag OFF.
        for path in (
            self.path("conversation_shadow.json"),
            self.path("snapshots", cid + ".json"),
            self.path("compact", cid + ".json"),
            self.path(
                "context_candidate",
                cid + ".json",
            ),
            self.path(
                "unified_context_candidate",
                cid + ".json",
            ),
        ):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())

        unified = self.read_json(
            "unified_context_candidate",
            cid + ".json",
        )

        self.assertEqual(
            unified["version"],
            "unified-context-candidate.v1",
        )

        # ...and the Memory Flash really surfaced the memories.
        flash_files = list(
            self.path(
                "memory_flash",
                cid,
            ).iterdir()
        )

        self.assertEqual(len(flash_files), 1)

        artifact = json.loads(
            flash_files[0].read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            artifact["surfaced_count"],
            2,
        )
        self.assertEqual(
            selected,
            _body(),
        )

    async def test_ledger_only_flag_drives_prerequisites(
        self,
    ):
        self.bind(
            [
                _memory("m1", "past decision one"),
                _memory("m2", "past decision two"),
            ]
        )

        await self._run(
            flash=False,
            ledger=True,
            confidence=False,
        )

        cid = self.conversation_id()

        for path in (
            self.path("snapshots", cid + ".json"),
            self.path("compact", cid + ".json"),
            self.path(
                "context_candidate",
                cid + ".json",
            ),
            self.path(
                "unified_context_candidate",
                cid + ".json",
            ),
        ):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())

        # Flash OFF: no Flash artifact at all.
        self.assertFalse(
            self.path(
                "memory_flash",
                cid,
            ).exists()
        )

        ledger_files = list(
            self.path(
                "exposure_ledger",
                cid,
            ).iterdir()
        )

        self.assertEqual(len(ledger_files), 1)

        ledger = json.loads(
            ledger_files[0].read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            ledger["retrieved_count"],
            2,
        )
        self.assertEqual(
            ledger["surfaced_count"],
            0,
        )

    async def test_all_flags_off_runs_nothing(
        self,
    ):
        self.bind(
            [_memory("m1")]
        )

        await self._run(
            flash=False,
            ledger=False,
            confidence=False,
        )

        self.assertFalse(
            self.path(
                "unified_context_candidate"
            ).exists()
        )
        self.assertFalse(
            self.path("exposure_ledger").exists()
        )
        self.assertFalse(
            self.path("memory_flash").exists()
        )


class RealDuplicateChainTests(
    unittest.TestCase,
):
    """Real candidate shapes through the real Unified admission."""

    def _policy(self, retrieval):
        unified = build_unified_context_candidate(
            conversation_id=CID,
            conversation_candidate=(
                _conversation_candidate()
            ),
            context_candidates=(
                _source_candidates(retrieval)
            ),
        )

        # ``build_unified_context_candidate`` is the pure builder; the
        # persisted revision is assigned by the update wrapper.
        unified["revision"] = 1

        return unified, evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified,
            confidence_report={
                "version":
                    "context-confidence-gate.v1",
                "mode": "shadow_only",
                "decision": "allow_shadow",
                "allowed": True,
                "reason": None,
                "reasons": [],
                "stored": True,
                "duplicate": False,
                "revision": 1,
            },
            shadow_evidence=(
                build_candidate_evidence(
                    retrieval,
                    now=_NOW,
                )
            ),
        )

    def test_real_duplicate_id_not_surfaced_twice(
        self,
    ):
        unified, report = self._policy(
            [
                _memory(
                    "dup",
                    "first text",
                ),
                _memory(
                    "dup",
                    "second text",
                ),
            ]
        )

        # Admission dedups by content, not by id, so both candidates
        # survive into the Unified observation.
        self.assertEqual(
            len(
                unified["sections"][
                    "memories"
                ]
            ),
            2,
        )

        self.assertEqual(
            report["eligible_candidate_count"],
            1,
        )
        self.assertEqual(
            report["ineligible"][0]["reason"],
            "duplicate_memory_id",
        )

    def test_real_exact_text_duplicate_removed_by_admission(
        self,
    ):
        unified, report = self._policy(
            [
                _memory("a", "same text"),
                _memory("b", "same text"),
            ]
        )

        # The real admission step already removed the exact duplicate.
        self.assertEqual(
            len(
                unified["sections"][
                    "memories"
                ]
            ),
            1,
        )

        self.assertEqual(
            report["eligible_candidate_count"],
            1,
        )

        # ...and the second candidate never got a chance to surface.
        self.assertEqual(
            [
                item["id"]
                for item in report["eligible"]
            ],
            ["a"],
        )

    def test_real_recent_candidate_is_not_surfaced(
        self,
    ):
        _, report = self._policy(
            [
                _memory(
                    "recent",
                    "just said this",
                    age_hours=1,
                )
            ]
        )

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reason"],
            "anti_echo_recent_24h",
        )

    def test_real_normal_candidate_is_surfaced(
        self,
    ):
        _, report = self._policy(
            [_memory("old", "an old memory")]
        )

        self.assertEqual(
            report["decision"],
            "allow_shadow",
        )
        self.assertEqual(
            report["eligible_candidate_count"],
            1,
        )


class LiveIsolationTests(
    RealPipelineTestCase,
):
    """Memory shadow must not change Unified sections or Preview."""

    async def _run_full_pipeline(
        self,
        *,
        memory_on,
        memories,
    ):
        self.bind(memories)

        environ = {
            **_ISOLATED_ENV,
            "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                "1",
            _CONFIDENCE_ENV:
                "1" if memory_on else "0",
            _FLASH_ENV:
                "1" if memory_on else "0",
            _LEDGER_ENV:
                "1" if memory_on else "0",
        }

        with patch.dict(
            os.environ,
            environ,
            clear=False,
        ):
            selected = await (
                coordinator.run_context_pipeline(
                    _body()
                )
            )

        cid = self.conversation_id()

        return cid, selected

    async def test_unified_sections_and_preview_are_identical(
        self,
    ):
        memories = [
            _memory(
                "m1",
                "first memory",
                name="First",
            ),
            _memory("m2", "second memory"),
        ]

        with tempfile.TemporaryDirectory() as off_root:
            os.environ[
                "OMBRE_CONTEXT_STATE_DIR"
            ] = off_root

            (
                off_cid,
                off_selected,
            ) = await self._run_full_pipeline(
                memory_on=False,
                memories=memories,
            )

            off_unified = _read_json_at(
                off_root,
                "unified_context_candidate",
                off_cid + ".json",
            )

            off_preview = _read_json_at(
                off_root,
                "injection_preview",
                off_cid + ".json",
            )

            self.assertFalse(
                Path(
                    off_root,
                    "memory_flash",
                ).exists()
            )

        with tempfile.TemporaryDirectory() as on_root:
            os.environ[
                "OMBRE_CONTEXT_STATE_DIR"
            ] = on_root

            (
                on_cid,
                on_selected,
            ) = await self._run_full_pipeline(
                memory_on=True,
                memories=memories,
            )

            on_unified = _read_json_at(
                on_root,
                "unified_context_candidate",
                on_cid + ".json",
            )

            on_preview = _read_json_at(
                on_root,
                "injection_preview",
                on_cid + ".json",
            )

            # The Memory shadow really ran...
            flash_files = list(
                Path(
                    on_root,
                    "memory_flash",
                    on_cid,
                ).iterdir()
            )

            self.assertEqual(
                len(flash_files),
                1,
            )

            flash = json.loads(
                flash_files[0].read_text(
                    encoding="utf-8"
                )
            )

            on_flash_request_id = (
                flash_files[0].stem
            )

            self.assertEqual(
                flash["surfaced_count"],
                2,
            )

        # ...and the live artifacts are untouched. (conversation_id /
        # input_fingerprint are per-conversation, so they are not part
        # of this comparison; sections and telemetry are.)
        self.assertEqual(
            off_unified["sections"],
            on_unified["sections"],
        )

        self.assertEqual(
            off_unified["telemetry"],
            on_unified["telemetry"],
        )

        self.assertEqual(
            off_unified["source_revisions"],
            on_unified["source_revisions"],
        )

        self.assertEqual(
            off_preview["rendered"],
            on_preview["rendered"],
        )

        self.assertEqual(
            off_preview["render_sha256"],
            on_preview["render_sha256"],
        )

        # No Memory Flash artifact surface ever reaches the rendered
        # Preview: no request id, no cue section, no flash counter.
        for marker in (
            on_flash_request_id,
            "memory_flash",
            "surfaced_as_flash",
            "surfaced_count",
        ):
            with self.subTest(marker=marker):
                self.assertNotIn(
                    marker,
                    on_preview["rendered"],
                )

        self.assertEqual(
            off_selected,
            on_selected,
        )

        self.assertEqual(
            off_selected,
            _body(),
        )

    async def test_source_memory_is_byte_for_byte_unchanged(
        self,
    ):
        bucket_file = self.path(
            "buckets",
            "m1.json",
        )

        bucket_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        original = json.dumps(
            _memory("m1", "an old memory"),
            ensure_ascii=False,
        ).encode("utf-8")

        bucket_file.write_bytes(original)

        memories = [
            _memory("m1", "an old memory")
        ]

        before = copy.deepcopy(memories)

        environ = {
            **_ISOLATED_ENV,
            _CONFIDENCE_ENV: "1",
            _FLASH_ENV: "1",
            _LEDGER_ENV: "1",
        }

        service = self.bind(memories)

        with patch.dict(
            os.environ,
            environ,
            clear=False,
        ):
            selected = await (
                coordinator.run_context_pipeline(
                    _body()
                )
            )

        cid = self.conversation_id()

        # The source bucket is byte-for-byte identical.
        self.assertEqual(
            bucket_file.read_bytes(),
            original,
        )

        # ...and no activation / importance / decay field was touched
        # in the in-memory observation either.
        self.assertEqual(
            memories,
            before,
        )

        served = (
            service.get_candidates
            .return_value["memories"]
        )

        self.assertEqual(
            served,
            before,
        )

        # The observers really ran.
        self.assertTrue(
            self.path(
                "memory_flash",
                cid,
            ).exists()
        )
        self.assertTrue(
            self.path(
                "exposure_ledger",
                cid,
            ).exists()
        )

        self.assertEqual(
            selected,
            _body(),
        )

        # No reinforcement field is ever written by the observers.
        artifact = json.loads(
            next(
                self.path(
                    "memory_flash",
                    cid,
                ).iterdir()
            ).read_text(
                encoding="utf-8"
            )
        )

        serialized = json.dumps(artifact)

        for field in (
            "activation_count",
            "importance",
            "strength",
            "weight",
            "decay",
            "archive",
        ):
            with self.subTest(field=field):
                self.assertNotIn(field, serialized)


if __name__ == "__main__":
    unittest.main()