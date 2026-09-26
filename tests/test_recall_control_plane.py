from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from _recall_fixtures import (
    CID,
    CID_B,
    RID,
    RID_B,
    FakeBucketManager,
    FakeRetrievalAdapter,
    bucket,
    flash_artifact,
    memory,
    recall_env,
    write_flash,
)

from ombrebrain.context.memory_recall_surface import (
    recall_surface_for_model,
    update_recall_surface,
)
from ombrebrain.context.memory_usage_signal import (
    record_usage,
)
from ombrebrain.context.related_recall import (
    read_related_recall,
    request_related_recall,
)


def _recall_dir(root):
    return (
        Path(root)
        / "related_recall"
        / CID
        / RID
    )


def _usage_dir(root):
    return (
        Path(root)
        / "memory_usage"
        / CID
        / RID
    )


class GateRetrieval:
    """Canonical retrieval surface that forces concurrent interleaving.

    Both callers block on an ``asyncio.Event`` until every party has
    started, so the idempotency / isolation paths are exercised for
    real without ever sleeping.
    """

    def __init__(
        self,
        results,
        *,
        parties=2,
    ):
        self.results = list(results)
        self.parties = parties
        self.started = 0
        self.calls = 0
        self.gate = asyncio.Event()

    async def retrieve_with_observation(
        self,
        query,
        *,
        max_results=8,
        **_,
    ):
        self.calls += 1
        self.started += 1

        if self.started >= self.parties:
            self.gate.set()

        await self.gate.wait()

        return (
            self.results[:max_results],
            {
                "candidates": list(
                    self.results
                ),
                "vector_scores": {},
                "telemetry": {},
            },
        )


class RecallControlPlaneTests(
    unittest.IsolatedAsyncioTestCase,
):
    async def test_end_to_end_through_surface_memref(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [
                        memory(
                            "mem-1",
                            "alpha note",
                            name="merge rules",
                        ),
                        memory("mem-2", "beta note"),
                    ]
                ),
            )

            buckets = FakeBucketManager(
                {
                    "mem-1": bucket(
                        "mem-1", "alpha note"
                    )
                }
            )

            retrieval = FakeRetrievalAdapter(
                [memory("mem-2", "beta note")]
            )

            with recall_env(
                root, surface=True
            ):
                update_recall_surface(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    flash_report=flash_artifact(
                        [
                            memory(
                                "mem-1",
                                "alpha note",
                                name="merge rules",
                            ),
                            memory(
                                "mem-2", "beta note"
                            ),
                        ]
                    ),
                )

                model_view = (
                    recall_surface_for_model(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

                memref = model_view[
                    "surfaces"
                ][0]["memref"]

                report = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_ref=memref,
                        current_query=(
                            "what did we decide"
                        ),
                        bucket_manager=buckets,
                        retrieval_adapter=(
                            retrieval
                        ),
                    )
                )

                status = read_related_recall(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=report[
                        "recall_id"
                    ],
                )

        self.assertTrue(report["stored"])
        self.assertEqual(
            report["decision"], "recalled"
        )
        self.assertEqual(
            status["anchor_memory_id"], "mem-1"
        )
        self.assertEqual(
            [
                item["memory_id"]
                for item in status["memories"]
            ],
            ["mem-1", "mem-2"],
        )
        # Exactly one canonical retrieval.
        self.assertEqual(len(retrieval.calls), 1)

    async def test_retrieved_but_not_surfaced_cannot_be_recalled(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            with recall_env(root):
                report = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-2",
                        bucket_manager=(
                            FakeBucketManager()
                        ),
                        retrieval_adapter=(
                            FakeRetrievalAdapter()
                        ),
                    )
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "anchor_not_surfaced",
        )

    async def test_cross_request_flash_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [memory("mem-1")]
                ),
            )

            with recall_env(root):
                report = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID_B,
                        anchor_memory_id="mem-1",
                        bucket_manager=(
                            FakeBucketManager()
                        ),
                        retrieval_adapter=(
                            FakeRetrievalAdapter()
                        ),
                    )
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "flash_not_found",
        )

    async def test_cross_conversation_flash_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [memory("mem-1")],
                    conversation_id=CID_B,
                ),
            )

            with recall_env(root):
                report = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        bucket_manager=(
                            FakeBucketManager()
                        ),
                        retrieval_adapter=(
                            FakeRetrievalAdapter()
                        ),
                    )
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "flash_not_found",
        )

    async def test_default_off_creates_no_artifacts(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            with recall_env(
                root, enabled=False, shadow=False
            ):
                report = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        bucket_manager=(
                            FakeBucketManager(
                                {
                                    "mem-1": bucket(
                                        "mem-1"
                                    )
                                }
                            )
                        ),
                        retrieval_adapter=(
                            FakeRetrievalAdapter()
                        ),
                    )
                )

            self.assertFalse(
                _recall_dir(root).exists()
            )
            self.assertFalse(
                _usage_dir(root).exists()
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "recall_disabled",
        )

    async def test_idempotent_retry_does_not_duplicate(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            buckets = FakeBucketManager(
                {"mem-1": bucket("mem-1")}
            )

            retrieval = FakeRetrievalAdapter(
                [memory("mem-2")]
            )

            with recall_env(root):
                first = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        bucket_manager=buckets,
                        retrieval_adapter=(
                            retrieval
                        ),
                    )
                )

                second = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        bucket_manager=buckets,
                        retrieval_adapter=(
                            retrieval
                        ),
                    )
                )

                files = list(
                    _recall_dir(root).glob(
                        "recall_*.json"
                    )
                )

                usage_files = list(
                    _usage_dir(root).glob(
                        "recall_*.json"
                    )
                )

                usage = json.loads(
                    usage_files[0].read_text(
                        encoding="utf-8"
                    )
                )

        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(
            first["recall_id"],
            second["recall_id"],
        )
        self.assertEqual(len(files), 1)
        self.assertEqual(len(usage_files), 1)

        # recall_requested is never double counted.
        stages = [
            event["stage"]
            for event in usage["events"]
        ]

        self.assertEqual(
            stages.count("recall_requested"), 1
        )

    async def test_concurrent_retry_is_deterministic(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            buckets = FakeBucketManager(
                {"mem-1": bucket("mem-1")}
            )

            retrieval = GateRetrieval(
                [memory("mem-2")], parties=2
            )

            with recall_env(root):
                first, second = await asyncio.gather(
                    request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        bucket_manager=buckets,
                        retrieval_adapter=retrieval,
                    ),
                    request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        bucket_manager=buckets,
                        retrieval_adapter=retrieval,
                    ),
                )

                files = list(
                    _recall_dir(root).glob(
                        "recall_*.json"
                    )
                )

        self.assertEqual(len(files), 1)
        self.assertEqual(
            first["recall_id"],
            second["recall_id"],
        )

    async def test_same_conversation_requests_are_isolated(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            write_flash(
                root,
                flash_artifact(
                    [memory("mem-2")],
                    cognitive_request_id=RID_B,
                ),
            )

            buckets = FakeBucketManager(
                {
                    "mem-1": bucket("mem-1"),
                    "mem-2": bucket("mem-2"),
                }
            )

            retrieval = GateRetrieval(
                [memory("mem-3")], parties=2
            )

            with recall_env(root):
                report_a, report_b = (
                    await asyncio.gather(
                        request_related_recall(
                            conversation_id=CID,
                            cognitive_request_id=RID,
                            anchor_memory_id=(
                                "mem-1"
                            ),
                            bucket_manager=buckets,
                            retrieval_adapter=(
                                retrieval
                            ),
                        ),
                        request_related_recall(
                            conversation_id=CID,
                            cognitive_request_id=(
                                RID_B
                            ),
                            anchor_memory_id=(
                                "mem-2"
                            ),
                            bucket_manager=buckets,
                            retrieval_adapter=(
                                retrieval
                            ),
                        ),
                    )
                )

                artifact_a = read_related_recall(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=report_a[
                        "recall_id"
                    ],
                )

                artifact_b = read_related_recall(
                    conversation_id=CID,
                    cognitive_request_id=RID_B,
                    recall_id=report_b[
                        "recall_id"
                    ],
                )

        self.assertNotEqual(
            report_a["recall_id"],
            report_b["recall_id"],
        )
        self.assertEqual(
            artifact_a["anchor_memory_id"],
            "mem-1",
        )
        self.assertEqual(
            artifact_b["anchor_memory_id"],
            "mem-2",
        )

        # B's recall can never read A's flash identity.
        a_dir = (
            Path(root)
            / "related_recall"
            / CID
            / RID
        )

        b_dir = (
            Path(root)
            / "related_recall"
            / CID
            / RID_B
        )

        self.assertNotEqual(a_dir, b_dir)

    async def test_same_request_multiple_recalls_do_not_overwrite(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [
                        memory("mem-1"),
                        memory("mem-2"),
                    ]
                ),
            )

            buckets = FakeBucketManager(
                {
                    "mem-1": bucket("mem-1"),
                    "mem-2": bucket("mem-2"),
                }
            )

            retrieval = FakeRetrievalAdapter(
                [memory("mem-3")]
            )

            with recall_env(root):
                report_a = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        bucket_manager=buckets,
                        retrieval_adapter=(
                            retrieval
                        ),
                    )
                )

                report_b = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-2",
                        bucket_manager=buckets,
                        retrieval_adapter=(
                            retrieval
                        ),
                    )
                )

                files = sorted(
                    path.name
                    for path in _recall_dir(
                        root
                    ).glob("recall_*.json")
                )

        self.assertNotEqual(
            report_a["recall_id"],
            report_b["recall_id"],
        )
        self.assertEqual(len(files), 2)

    async def test_usage_cannot_cross_recalls(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [
                        memory("mem-1"),
                        memory("mem-2"),
                    ]
                ),
            )

            buckets = FakeBucketManager(
                {
                    "mem-1": bucket("mem-1"),
                    "mem-2": bucket("mem-2"),
                }
            )

            retrieval = FakeRetrievalAdapter(
                [memory("mem-3")]
            )

            with recall_env(root):
                report_a = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        bucket_manager=buckets,
                        retrieval_adapter=(
                            retrieval
                        ),
                    )
                )

                report_b = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-2",
                        bucket_manager=buckets,
                        retrieval_adapter=(
                            retrieval
                        ),
                    )
                )

                # Usage for B's recall may only name B's loaded ids.
                cross = record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=report_a[
                        "recall_id"
                    ],
                    used_memory_ids=[
                        "mem-2"
                    ],
                )

                same = record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=report_a[
                        "recall_id"
                    ],
                    used_memory_ids=[
                        "mem-1"
                    ],
                )

        # mem-2 belongs to B's recall; in A's recall it was not loaded.
        self.assertEqual(
            cross["rejected"][0]["reason"],
            "memory_not_loaded",
        )
        self.assertEqual(
            same["used_memory_ids"], ["mem-1"]
        )
        self.assertNotIn(
            "mem-2", same["used_memory_ids"]
        )

    async def test_no_reinforcement_source_is_byte_identical(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            source = (
                Path(root)
                / "buckets"
                / "mem-1.json"
            )

            source.parent.mkdir(
                parents=True, exist_ok=True
            )

            source.write_text(
                json.dumps(
                    {
                        "id": "mem-1",
                        "content": (
                            "activation sensitive"
                        ),
                        "metadata": {
                            "activation_count": 3,
                            "last_active":
                                "2024-01-01T00:00:00Z",
                            "importance": 7,
                            "strength": 0.4,
                            "weight": 0.5,
                            "score": 0.9,
                            "decay": 0.1,
                            "archive": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            before = source.read_bytes()

            anchor_object = json.loads(
                before.decode("utf-8")
            )

            anchor_snapshot = json.loads(
                before.decode("utf-8")
            )

            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            buckets = FakeBucketManager(
                {"mem-1": anchor_object}
            )

            retrieval = FakeRetrievalAdapter(
                [memory("mem-2")]
            )

            with recall_env(root):
                report = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        bucket_manager=buckets,
                        retrieval_adapter=(
                            retrieval
                        ),
                    )
                )

                record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=report[
                        "recall_id"
                    ],
                    used_memory_ids=["mem-1"],
                )

            after = source.read_bytes()

        self.assertEqual(before, after)

        # the source object handed to the control plane is untouched
        self.assertEqual(
            anchor_object, anchor_snapshot
        )
        for field in (
            "activation_count",
            "last_active",
            "importance",
            "strength",
            "weight",
            "score",
            "decay",
            "archive",
        ):
            self.assertEqual(
                anchor_object["metadata"][field],
                anchor_snapshot["metadata"][
                    field
                ],
            )

    async def test_retrieval_failure_is_fail_open(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            buckets = FakeBucketManager(
                {"mem-1": bucket("mem-1")}
            )

            retrieval = FakeRetrievalAdapter(
                [],
                raises=RuntimeError("boom"),
            )

            with recall_env(root):
                report = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        bucket_manager=buckets,
                        retrieval_adapter=(
                            retrieval
                        ),
                    )
                )

        # Anchor-only recall, still no exception, still structured.
        self.assertTrue(report["stored"])
        self.assertEqual(
            report["included_count"], 1
        )

    async def test_unknown_memref_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            with recall_env(root):
                report = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_ref=(
                            "memref_" + "9" * 32
                        ),
                        bucket_manager=(
                            FakeBucketManager()
                        ),
                        retrieval_adapter=(
                            FakeRetrievalAdapter()
                        ),
                    )
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "recall_ref_not_found",
        )

    async def test_full_scope_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            with recall_env(root):
                report = (
                    await request_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        anchor_memory_id="mem-1",
                        requested_scope="full",
                        bucket_manager=(
                            FakeBucketManager()
                        ),
                        retrieval_adapter=(
                            FakeRetrievalAdapter()
                        ),
                    )
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "invalid_requested_scope",
        )


if __name__ == "__main__":
    unittest.main()