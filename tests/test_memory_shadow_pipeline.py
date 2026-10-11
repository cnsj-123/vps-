from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import (
    ThreadPoolExecutor,
)
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from unittest.mock import (
    AsyncMock,
    Mock,
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
    update_unified_context_candidate_from_runtime,
)

import ombrebrain.context.unified_context_candidate as unified_module
import ombrebrain.context.memory_source_evidence as evidence_module


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
_DELIVERY_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_DELIVERY_SHADOW"
)
_ARBITRATION_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_SOURCE_ARBITRATION_SHADOW"
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
    _DELIVERY_ENV,
    _ARBITRATION_ENV,
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


def _compact_file(
    *,
    source_revision,
    text,
    source_index=21,
):
    return {
        "conversation_id": CID,
        "source_revision": source_revision,
        "recent_messages": [
            {
                "role": "user",
                "text": text,
                "source_index": source_index,
            }
        ],
    }


def _candidate_file(
    *,
    revision,
    source_revision,
    fact,
    source_index=21,
):
    payload = _conversation_candidate()

    payload["revision"] = revision
    payload["source_revision"] = source_revision
    payload["sections"]["trusted_facts"] = [
        {
            "id": "fact",
            "text": fact,
        }
    ]
    payload["telemetry"][
        "latest_user_source_index"
    ] = source_index

    return payload


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

    def write(self, kind, value):
        path = self.path(
            kind,
            CID + ".json",
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
            ),
            encoding="utf-8",
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

    async def test_memory_observer_flags_drive_prerequisites(
        self,
    ):
        # Flash + Ledger + Confidence on, every legacy Context shadow
        # flag off. (Flash-only is covered separately by
        # FlashOnlyDependencyTests.)
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
                "conversation_id": CID,
                "decision": "allow_shadow",
                "allowed": True,
                "reason": None,
                "reasons": [],
                "stored": True,
                "duplicate": False,
                "revision": 1,
                "source_unified_revision": 1,
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


class CandidateRaceTests(
    RealPipelineTestCase,
):
    """The runtime must use the candidate captured before the await.

    The real producer (``update_unified_context_candidate_from_runtime``)
    is used; it is NOT mocked. Synchronization is event based -- no
    sleep anywhere.
    """

    async def test_captured_candidate_is_not_replaced_by_disk(
        self,
    ):
        self.write(
            "compact",
            _compact_file(
                source_revision=5,
                text="QUERY_A",
            ),
        )
        self.write(
            "context_candidate",
            _candidate_file(
                revision=10,
                source_revision=5,
                fact="A fact",
            ),
        )

        started = asyncio.Event()
        release = asyncio.Event()
        queries: list = []

        bound = self.bind([])

        async def get_candidates(
            query="",
        ):
            queries.append(query)

            # Only request A blocks; the later request B in this test
            # must run freely.
            if len(queries) == 1:
                started.set()
                await release.wait()

            return _source_candidates(
                [_memory("m-retrieved")]
            )

        bound.get_candidates.side_effect = (
            get_candidates
        )

        task_a = asyncio.create_task(
            update_unified_context_candidate_from_runtime(
                CID
            )
        )

        await started.wait()

        # While A is still awaiting retrieval, the shared
        # conversation-level candidate is replaced by "request B".
        self.write(
            "compact",
            _compact_file(
                source_revision=6,
                text="QUERY_B",
            ),
        )
        self.write(
            "context_candidate",
            _candidate_file(
                revision=11,
                source_revision=6,
                fact="B fact",
            ),
        )

        result_b = (
            await update_unified_context_candidate_from_runtime(
                CID
            )
        )

        b_disk = self.read_json(
            "unified_context_candidate",
            CID + ".json",
        )

        self.assertEqual(
            b_disk["source_revisions"][
                "conversation_candidate"
            ],
            11,
        )

        release.set()

        result_a = await task_a

        a_disk = self.read_json(
            "unified_context_candidate",
            CID + ".json",
        )

        # A's query came from the pre-await capture...
        self.assertEqual(
            queries[0],
            "QUERY_A",
        )
        self.assertEqual(
            queries[1],
            "QUERY_B",
        )

        # ...A used candidate 10, never B's 11...
        self.assertEqual(
            a_disk["source_revisions"][
                "conversation_candidate"
            ],
            10,
        )

        facts = [
            item["text"]
            for item in a_disk["sections"][
                "trusted_facts"
            ]
        ]

        self.assertEqual(facts, ["A fact"])

        # ...and the request-local snapshot agrees.
        snapshot = result_a[
            "memory_shadow_snapshot"
        ]

        self.assertEqual(
            snapshot["unified"][
                "source_revisions"
            ][
                "conversation_candidate"
            ],
            10,
        )

        self.assertEqual(
            [
                item["text"]
                for item in snapshot[
                    "unified"
                ]["sections"]["trusted_facts"]
            ],
            ["A fact"],
        )

        # B kept its own candidate.
        self.assertEqual(
            result_b[
                "memory_shadow_snapshot"
            ]["unified"]["source_revisions"][
                "conversation_candidate"
            ],
            11,
        )


class ProducerEvidenceIsolationTests(
    RealPipelineTestCase,
):
    """Two concurrent producers keep their own retrieval evidence."""

    def test_two_producers_keep_their_own_evidence(
        self,
    ):
        self.write(
            "compact",
            _compact_file(
                source_revision=5,
                text="QUERY",
            ),
        )
        self.write(
            "context_candidate",
            _candidate_file(
                revision=10,
                source_revision=5,
                fact="shared fact",
            ),
        )

        memory_a = "mem_A_111111111111"
        memory_b = "mem_B_222222222222"

        local = threading.local()
        barrier = threading.Barrier(2, timeout=30)

        class _Service:

            async def get_candidates(
                self,
                query="",
            ):
                # Both producers are inside retrieval at once.
                barrier.wait()

                return _source_candidates(
                    [
                        _memory(
                            local.memory_id,
                            "cue "
                            + local.memory_id,
                        )
                    ]
                )

        bind_context_service(_Service())

        results: dict = {}

        def worker(name, memory_id):
            local.memory_id = memory_id

            results[name] = asyncio.run(
                update_unified_context_candidate_from_runtime(
                    CID
                )
            )

        with patch.object(
            unified_module,
            "record_retrieval_shadow_metrics",
            Mock(
                return_value={
                    "stored": False,
                }
            ),
        ):
            with ThreadPoolExecutor(
                max_workers=2
            ) as pool:
                futures = [
                    pool.submit(
                        worker,
                        "A",
                        memory_a,
                    ),
                    pool.submit(
                        worker,
                        "B",
                        memory_b,
                    ),
                ]

                for future in futures:
                    future.result(timeout=30)

        for name, own in (
            ("A", memory_a),
            ("B", memory_b),
        ):
            with self.subTest(request=name):
                snapshot = results[name][
                    "memory_shadow_snapshot"
                ]

                self.assertIsNotNone(snapshot)

                self.assertEqual(
                    snapshot["unified"][
                        "source_revisions"
                    ][
                        "conversation_candidate"
                    ],
                    10,
                )

                # Only this request's retrieval evidence.
                self.assertEqual(
                    {
                        entry["memory_id"]
                        for entry in snapshot[
                            "evidence"
                        ]
                    },
                    {own},
                )

                self.assertEqual(
                    [
                        item["id"]
                        for item in snapshot[
                            "unified"
                        ]["sections"]["memories"]
                    ],
                    [own],
                )


class FlashOnlyDependencyTests(
    RealPipelineTestCase,
):
    """MEMORY_FLASH_SHADOW alone must drive the real upstream chain."""

    async def test_flash_only_flag_uses_real_pipeline(
        self,
    ):
        self.bind(
            [
                _memory("m1", "first memory"),
                _memory("m2", "second memory"),
            ]
        )

        environ = {
            **_ISOLATED_ENV,
            _FLASH_ENV: "1",
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

        for path in (
            self.path(
                "snapshots",
                cid + ".json",
            ),
            self.path(
                "compact",
                cid + ".json",
            ),
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

        # Confidence is OFF, so the Flash reports it could not use a
        # confidence observation -- but the artifact exists, is bound
        # to a valid revision and keeps the real retrieved count.
        self.assertEqual(
            artifact["decision"],
            "no_surface",
        )
        self.assertEqual(
            artifact["reason"],
            "confidence_observation_missing",
        )
        self.assertEqual(
            artifact["surfaced_count"],
            0,
        )
        self.assertEqual(
            artifact["source_unified_revision"],
            unified["revision"],
        )
        self.assertEqual(
            artifact["retrieved_candidate_count"],
            2,
        )

        # Ledger is OFF: nothing is written there.
        self.assertFalse(
            self.path(
                "exposure_ledger",
                cid,
            ).exists()
        )

        # The live body is untouched.
        self.assertEqual(
            selected,
            _body(),
        )


class DeliveryFlagMatrixTests(
    RealPipelineTestCase,
):
    """Delivery-only / off / arbitration matrix on the real chain."""

    def _plan_files(self, cid):
        directory = self.path(
            "memory_delivery_plan",
            cid,
        )

        if not directory.is_dir():
            return []

        return list(directory.iterdir())

    async def test_delivery_only_flag_drives_chain(self):
        self.bind(
            [
                _memory("m1", "first memory"),
                _memory("m2", "second memory"),
            ]
        )

        environ = {
            **_ISOLATED_ENV,
            _DELIVERY_ENV: "1",
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

        plan_files = self._plan_files(cid)

        self.assertEqual(len(plan_files), 1)

        artifact = json.loads(
            plan_files[0].read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            artifact["version"],
            "memory-delivery-plan-shadow.v1",
        )
        self.assertEqual(
            artifact["authority"],
            "proposal_only",
        )
        self.assertFalse(
            artifact["live_authority"]
        )
        self.assertEqual(
            artifact["policy_stage"],
            "base_surfacing_pre_lifecycle",
        )

        # Delivery never mutates the forwarded body.
        self.assertEqual(selected, _body())

    async def test_delivery_off_writes_no_plan(self):
        self.bind([_memory("m1", "only memory")])

        with patch.dict(
            os.environ,
            _ISOLATED_ENV,
            clear=False,
        ):
            selected = await (
                coordinator.run_context_pipeline(
                    _body()
                )
            )

        # With every flag OFF the chain does not run at all, so no
        # conversation identity is minted and no delivery plan tree is
        # created anywhere under the state root.
        self.assertFalse(
            self.path(
                "memory_delivery_plan"
            ).exists()
        )

        # Delivery, even when off, never mutates the forwarded body.
        self.assertEqual(selected, _body())

    async def test_arbitration_and_delivery_share_one_evidence(
        self,
    ):
        # With BOTH consumers ON the coordinator builds the source
        # evidence ONCE; both request-scoped artifacts are produced
        # from the same request-local evidence.
        self.bind([_memory("m1", "shared memory")])

        environ = {
            **_ISOLATED_ENV,
            _DELIVERY_ENV: "1",
            _ARBITRATION_ENV: "1",
        }

        with patch.dict(
            os.environ,
            environ,
            clear=False,
        ):
            await coordinator.run_context_pipeline(
                _body()
            )

        cid = self.conversation_id()

        self.assertEqual(
            len(self._plan_files(cid)), 1
        )

        arbitration_files = list(
            self.path(
                "memory_source_arbitration",
                cid,
            ).iterdir()
        )

        self.assertEqual(
            len(arbitration_files), 1
        )

    async def test_adapter_built_exactly_once_and_shared(
        self,
    ):
        # With BOTH consumers ON the shared adapter must be invoked
        # exactly once, and both observers must receive the SAME
        # request-local evidence object.
        self.bind([_memory("m1", "shared memory")])

        environ = {
            **_ISOLATED_ENV,
            _DELIVERY_ENV: "1",
            _ARBITRATION_ENV: "1",
        }

        built = []
        delivery_seen = []
        arbitration_seen = []

        real_build = (
            evidence_module
            .build_memory_source_evidence_snapshot
        )
        real_delivery = (
            coordinator.update_memory_delivery_plan
        )
        real_arbitration = (
            coordinator
            .update_memory_source_arbitration_from_evidence
        )

        def spy_build(*args, **kwargs):
            snapshot = real_build(*args, **kwargs)

            built.append(snapshot)

            return snapshot

        def spy_delivery(*args, **kwargs):
            delivery_seen.append(
                kwargs.get("source_evidence_snapshot")
            )

            return real_delivery(*args, **kwargs)

        def spy_arbitration(*args, **kwargs):
            arbitration_seen.append(
                kwargs.get("source_evidence_snapshot")
            )

            return real_arbitration(*args, **kwargs)

        with patch.dict(
            os.environ,
            environ,
            clear=False,
        ):
            with patch.object(
                evidence_module,
                "build_memory_source_evidence_snapshot",
                side_effect=spy_build,
            ) as build_mock, patch.object(
                coordinator,
                "update_memory_delivery_plan",
                side_effect=spy_delivery,
            ), patch.object(
                coordinator,
                "update_memory_source_arbitration_from_evidence",
                side_effect=spy_arbitration,
            ):
                await coordinator.run_context_pipeline(
                    _body()
                )

        self.assertEqual(build_mock.call_count, 1)
        self.assertEqual(len(built), 1)
        self.assertEqual(len(delivery_seen), 1)
        self.assertEqual(len(arbitration_seen), 1)

        self.assertIs(delivery_seen[0], built[0])
        self.assertIs(arbitration_seen[0], built[0])


class RealPipelineDeliveryInterleavingTests(
    RealPipelineTestCase,
):
    """REAL coordinator A/B interleaving keeps request-local evidence.

    Unlike the lower-level direct-updater test
    (``DeliveryIsolationTests.test_ab_interleaving_keeps_evidence_isolated``,
    which constructs its own snapshot), this acceptance test exercises the
    REAL path:

        run_context_pipeline / real coordinator
            -> real request-local Unified snapshot
                -> real build_memory_source_evidence_snapshot
                    -> real Delivery Plan

    No manually constructed ``MemorySourceEvidenceSnapshot``; no sleeps. The
    pause is a deterministic ``threading.Event`` hooked around the REAL
    ``build_memory_source_evidence_snapshot`` call. Request A is paused AFTER
    it builds its real request-local snapshot and BEFORE Delivery consumes it,
    while request B runs to completion on the SAME conversation and overwrites
    the conversation-scoped latest Unified.
    """

    def test_real_pipeline_ab_interleaving_is_isolated(
        self,
    ):
        memory_a = "mem_A_111111111111"
        memory_b = "mem_B_222222222222"

        local = threading.local()

        class _Service:

            async def get_candidates(
                self,
                query="",
            ):
                return _source_candidates(
                    [
                        _memory(
                            local.memory_id,
                            "cue " + local.memory_id,
                        )
                    ]
                )

        bind_context_service(_Service())

        built: dict = {}
        delivered: dict = {}

        a_built = threading.Event()
        release_a = threading.Event()

        first_lock = threading.Lock()
        state = {"first": True}

        real_build = (
            evidence_module
            .build_memory_source_evidence_snapshot
        )
        real_delivery = (
            coordinator.update_memory_delivery_plan
        )

        def spy_build(*args, **kwargs):
            snapshot = real_build(*args, **kwargs)

            ident = threading.get_ident()

            built[ident] = snapshot

            is_first = False

            with first_lock:
                if state["first"]:
                    state["first"] = False
                    is_first = True

            if is_first:
                # Request A has built its REAL request-local Source
                # Evidence snapshot. Pause before Delivery consumes it.
                a_built.set()
                release_a.wait(30)

            return snapshot

        def spy_delivery(*args, **kwargs):
            delivered[threading.get_ident()] = (
                kwargs.get("source_evidence_snapshot")
            )

            return real_delivery(*args, **kwargs)

        results: dict = {}

        def run(name, memory_id):
            local.memory_id = memory_id

            results[name] = asyncio.run(
                coordinator.run_context_pipeline(
                    _body()
                )
            )

        environ = {
            **_ISOLATED_ENV,
            _DELIVERY_ENV: "1",
        }

        with patch.dict(
            os.environ,
            environ,
            clear=False,
        ):
            with patch.object(
                evidence_module,
                "build_memory_source_evidence_snapshot",
                side_effect=spy_build,
            ), patch.object(
                coordinator,
                "update_memory_delivery_plan",
                side_effect=spy_delivery,
            ):
                thread_a = threading.Thread(
                    target=run,
                    args=("A", memory_a),
                )
                thread_a.start()

                # A is paused inside the REAL adapter, after building its
                # real snapshot and before Delivery consumes it.
                self.assertTrue(
                    a_built.wait(30),
                    "request A never built its source evidence",
                )

                # Request B runs to completion on the SAME conversation.
                # It overwrites the conversation-scoped latest Unified.
                run("B", memory_b)

                a_thread_id = None
                b_thread_id = threading.get_ident()

                for ident in built:
                    if ident != b_thread_id:
                        a_thread_id = ident

                self.assertIsNotNone(a_thread_id)

                cid = (
                    self.conversation_id()
                )

                unified = self.read_json(
                    "unified_context_candidate",
                    cid + ".json",
                )

                # PROOF: B overwrote the conversation-scoped latest
                # Unified while A was still paused.
                self.assertEqual(
                    [
                        item["id"]
                        for item in unified["sections"][
                            "memories"
                        ]
                    ],
                    [memory_b],
                )

                # A is still paused -- it has not been released yet.
                self.assertTrue(thread_a.is_alive())

                release_a.set()
                thread_a.join(30)

                self.assertFalse(thread_a.is_alive())

        snap_a = built[a_thread_id]
        snap_b = built[b_thread_id]

        rid_a = snap_a.cognitive_request_id
        rid_b = snap_b.cognitive_request_id

        self.assertNotEqual(rid_a, rid_b)

        # PROOF: each Delivery consumed the exact SAME object that its own
        # real adapter built (object identity, not equality).
        self.assertIs(delivered[a_thread_id], snap_a)
        self.assertIs(delivered[b_thread_id], snap_b)

        plan_dir = self.path(
            "memory_delivery_plan",
            cid,
        )

        plans = {}

        for path in plan_dir.iterdir():
            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )
            plans[
                artifact["cognitive_request_id"]
            ] = artifact

        plan_a = plans[rid_a]
        plan_b = plans[rid_b]

        digests_a = {
            unit.content_digest
            for unit in snap_a.units
        }
        digests_b = {
            unit.content_digest
            for unit in snap_b.units
        }

        self.assertTrue(digests_a)
        self.assertTrue(digests_b)
        self.assertEqual(digests_a & digests_b, set())

        rows_a = {
            row["content_digest"]
            for row in plan_a["rows"]
        }
        rows_b = {
            row["content_digest"]
            for row in plan_b["rows"]
        }

        # A plan is bound to A and carries ONLY A evidence; zero B.
        self.assertEqual(
            plan_a["cognitive_request_id"], rid_a
        )
        self.assertEqual(rows_a, digests_a)
        self.assertEqual(rows_a & digests_b, set())

        # B plan is bound to B and carries ONLY B evidence; zero A.
        self.assertEqual(
            plan_b["cognitive_request_id"], rid_b
        )
        self.assertEqual(rows_b, digests_b)
        self.assertEqual(rows_b & digests_a, set())

        # The forwarded body was never mutated by the shadow.
        self.assertEqual(results["A"], _body())
        self.assertEqual(results["B"], _body())


if __name__ == "__main__":
    unittest.main()