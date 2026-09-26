from __future__ import annotations

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
    memory,
    recall_artifact,
    recall_env,
    recall_request,
    write_recall,
)

from ombrebrain.context.recall_request import (
    recall_request_fingerprint,
)
from ombrebrain.context.related_recall import (
    build_recall_query,
    build_related_recall,
    is_valid_related_recall_artifact,
    read_related_recall,
    resolve_related_recall_budget,
)


RECALL_ID = "recall_" + "7" * 32


class BudgetTests(unittest.TestCase):
    def test_defaults_are_conservative(self):
        budget = resolve_related_recall_budget()

        self.assertEqual(
            budget["max_related_memories"], 4
        )
        self.assertEqual(
            budget["token_budget"], 512
        )
        self.assertEqual(
            budget["max_chars_per_memory"],
            1200,
        )

    def test_extreme_values_are_clamped(self):
        budget = resolve_related_recall_budget(
            max_related_memories="9999",
            token_budget="99999",
            max_chars_per_memory="99999",
        )

        self.assertEqual(
            budget["max_related_memories"], 8
        )
        self.assertEqual(
            budget["token_budget"], 1200
        )
        self.assertEqual(
            budget["max_chars_per_memory"],
            3000,
        )

    def test_invalid_values_degrade_to_default(
        self,
    ):
        budget = resolve_related_recall_budget(
            max_related_memories="true",
            token_budget="-3",
            max_chars_per_memory="",
        )

        self.assertEqual(
            budget["max_related_memories"], 4
        )
        self.assertEqual(
            budget["token_budget"], 512
        )
        self.assertEqual(
            budget["max_chars_per_memory"],
            1200,
        )


class RecallQueryTests(unittest.TestCase):
    def test_query_is_bounded_and_deterministic(
        self,
    ):
        query = build_recall_query(
            anchor_representation="a" * 5000,
            current_query="b" * 5000,
            anchor_chars=100,
            user_chars=50,
        )

        self.assertEqual(
            query,
            build_recall_query(
                anchor_representation="a" * 5000,
                current_query="b" * 5000,
                anchor_chars=100,
                user_chars=50,
            ),
        )

        anchor_part, user_part = query.split(
            "\n"
        )

        self.assertEqual(len(anchor_part), 100)
        self.assertEqual(len(user_part), 50)

    def test_query_never_includes_store_or_prompt(
        self,
    ):
        query = build_recall_query(
            anchor_representation="anchor",
            current_query="question",
        )

        self.assertEqual(
            query, "anchor\nquestion"
        )


class RelatedRecallTests(
    unittest.IsolatedAsyncioTestCase,
):
    async def _build(
        self,
        *,
        anchor_buckets,
        retrieval_results,
        current_query="current question",
        max_related=4,
        token_budget=512,
        max_chars=1200,
        raises=None,
        anchor="mem-1",
        recall_id=None,
    ):
        buckets = FakeBucketManager(
            anchor_buckets
        )

        retrieval = FakeRetrievalAdapter(
            retrieval_results,
            raises=raises,
        )

        artifact = await build_related_recall(
            recall_request=recall_request(
                anchor
            ),
            recall_id=recall_id,
            current_query=current_query,
            bucket_manager=buckets,
            retrieval_adapter=retrieval,
            budget={
                "max_related_memories": max_related,
                "token_budget": token_budget,
                "max_chars_per_memory": max_chars,
            },
        )

        return artifact, buckets, retrieval

    async def test_anchor_included_first(self):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket("mem-1")
                },
                retrieval_results=[
                    memory("mem-2"),
                    memory("mem-3"),
                ],
            )
        )

        ids = [
            item["memory_id"]
            for item in artifact["memories"]
        ]

        self.assertEqual(
            ids, ["mem-1", "mem-2", "mem-3"]
        )
        self.assertEqual(
            artifact["memories"][0]["reason"],
            "anchor_memory",
        )
        self.assertEqual(
            [
                item["rank"]
                for item in artifact["memories"]
            ],
            [1, 2, 3],
        )

    async def test_related_candidates_follow_canonical_order(
        self,
    ):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket("mem-1")
                },
                retrieval_results=[
                    memory("mem-3"),
                    memory("mem-2"),
                    memory("mem-5"),
                ],
            )
        )

        self.assertEqual(
            [
                item["memory_id"]
                for item in artifact["memories"]
            ],
            ["mem-1", "mem-3", "mem-2", "mem-5"],
        )

    async def test_duplicate_ids_are_removed(self):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket("mem-1")
                },
                retrieval_results=[
                    memory("mem-1"),
                    memory("mem-2"),
                    memory(
                        "mem-2",
                        content="different body",
                    ),
                ],
            )
        )

        ids = [
            item["memory_id"]
            for item in artifact["memories"]
        ]

        self.assertEqual(ids, ["mem-1", "mem-2"])
        self.assertGreaterEqual(
            artifact["duplicate_rejected"], 1
        )

    async def test_exact_duplicate_content_is_removed(
        self,
    ):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket(
                        "mem-1",
                        content="anchor body",
                    )
                },
                retrieval_results=[
                    memory(
                        "mem-2",
                        content="same body",
                    ),
                    memory(
                        "mem-3",
                        content="same body",
                    ),
                ],
            )
        )

        ids = [
            item["memory_id"]
            for item in artifact["memories"]
        ]

        self.assertEqual(
            ids, ["mem-1", "mem-2"]
        )
        self.assertGreaterEqual(
            artifact["duplicate_rejected"], 1
        )

    async def test_current_user_echo_is_excluded(
        self,
    ):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket("mem-1")
                },
                retrieval_results=[
                    memory(
                        "mem-2",
                        content="please recall this",
                    ),
                    memory("mem-3"),
                ],
                current_query=(
                    "Please   recall this"
                ),
            )
        )

        ids = [
            item["memory_id"]
            for item in artifact["memories"]
        ]

        self.assertNotIn("mem-2", ids)
        self.assertIn("mem-3", ids)
        self.assertEqual(
            artifact["echo_rejected"], 1
        )

    async def test_max_item_cap_is_enforced(self):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket("mem-1")
                },
                retrieval_results=[
                    memory("mem-2"),
                    memory("mem-3"),
                    memory("mem-4"),
                    memory("mem-5"),
                ],
                max_related=2,
            )
        )

        self.assertEqual(
            artifact["included_count"], 2
        )

    async def test_token_budget_is_enforced(self):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket(
                        "mem-1",
                        content="a" * 400,
                    )
                },
                retrieval_results=[
                    memory(
                        "mem-2",
                        content="b" * 400,
                    )
                ],
                token_budget=6,
            )
        )

        self.assertLessEqual(
            artifact["estimated_tokens"],
            artifact["token_budget"],
        )
        self.assertGreaterEqual(
            artifact["budget_rejected"], 1
        )

    async def test_per_memory_char_cap_is_enforced(
        self,
    ):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket(
                        "mem-1",
                        content="a" * 5000,
                    )
                },
                retrieval_results=[
                    memory(
                        "mem-2",
                        content="b" * 5000,
                    )
                ],
                max_chars=50,
            )
        )

        for item in artifact["memories"]:
            self.assertLessEqual(
                len(item["content"]), 50
            )

    async def test_anchor_is_counted_in_budget(self):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket(
                        "mem-1",
                        content="a" * 90,
                    )
                },
                retrieval_results=[
                    memory("mem-2")
                ],
            )
        )

        anchor_cost = artifact["memories"][0][
            "content"
        ]

        self.assertGreaterEqual(
            artifact["estimated_tokens"],
            max(
                1,
                (len(anchor_cost) + 2) // 3,
            ),
        )

    async def test_anchor_unavailable_is_a_refusal(
        self,
    ):
        artifact, buckets, retrieval = (
            await self._build(
                anchor_buckets={},
                retrieval_results=[
                    memory("mem-2")
                ],
            )
        )

        self.assertFalse(artifact["stored"])
        self.assertEqual(
            artifact["reason"],
            "anchor_memory_unavailable",
        )

        # No substitution and no retrieval once the anchor is gone.
        self.assertEqual(retrieval.calls, [])
        self.assertEqual(
            buckets.get_calls, ["mem-1"]
        )

    async def test_canonical_retrieval_called_once(
        self,
    ):
        _artifact, buckets, retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket("mem-1")
                },
                retrieval_results=[
                    memory("mem-2")
                ],
                max_related=4,
            )
        )

        self.assertEqual(len(retrieval.calls), 1)
        self.assertEqual(
            retrieval.calls[0][1], 4
        )
        self.assertEqual(
            buckets.get_calls, ["mem-1"]
        )

    async def test_retrieval_failure_is_fail_open(
        self,
    ):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket("mem-1")
                },
                retrieval_results=[],
                raises=RuntimeError("boom"),
            )
        )

        # The anchor still loads; only the neighborhood is empty.
        self.assertNotEqual(
            artifact.get("stored"), False
        )
        self.assertEqual(
            artifact["included_count"], 1
        )
        self.assertEqual(
            artifact["memories"][0][
                "memory_id"
            ],
            "mem-1",
        )

    async def test_artifact_contract(self):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket("mem-1")
                },
                retrieval_results=[
                    memory("mem-2")
                ],
            )
        )

        self.assertEqual(
            artifact["version"],
            "related-memory-recall.v1",
        )
        self.assertEqual(
            artifact["mode"], "shadow_only"
        )
        self.assertEqual(
            artifact["conversation_id"], CID
        )
        self.assertEqual(
            artifact["cognitive_request_id"],
            RID,
        )
        self.assertRegex(
            artifact["recall_id"],
            r"^recall_[0-9a-f]{32}$",
        )
        self.assertEqual(
            artifact["anchor_memory_id"],
            "mem-1",
        )
        self.assertEqual(
            artifact["retrieved_count"], 1
        )
        self.assertEqual(
            artifact["included_count"], 2
        )

    async def test_malformed_recall_request_is_refused(
        self,
    ):
        artifact = await build_related_recall(
            recall_request={"version": "nope"},
            bucket_manager=FakeBucketManager(),
            retrieval_adapter=(
                FakeRetrievalAdapter()
            ),
        )

        self.assertFalse(artifact["stored"])
        self.assertEqual(
            artifact["reason"],
            "invalid_recall_request",
        )

    async def test_pre_minted_recall_id_is_used(self):
        artifact, _buckets, _retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket("mem-1")
                },
                retrieval_results=[
                    memory("mem-2")
                ],
                recall_id=RECALL_ID,
            )
        )

        self.assertEqual(
            artifact["recall_id"], RECALL_ID
        )

    async def test_invalid_recall_id_is_refused(self):
        artifact, _buckets, retrieval = (
            await self._build(
                anchor_buckets={
                    "mem-1": bucket("mem-1")
                },
                retrieval_results=[],
                recall_id="not-a-recall-id",
            )
        )

        self.assertFalse(artifact["stored"])
        self.assertEqual(
            artifact["reason"],
            "invalid_recall_id",
        )

        # Nothing is even loaded for a malformed id.
        self.assertEqual(retrieval.calls, [])


class RelatedRecallValidatorTests(
    unittest.TestCase,
):
    def _artifact(self):
        return recall_artifact(
            ["mem-1", "mem-2"],
            recall_id=RECALL_ID,
        )

    def _expected_fingerprint(self):
        return recall_request_fingerprint(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_memory_id="mem-1",
            requested_scope="related",
        )

    def _valid(self, artifact):
        return is_valid_related_recall_artifact(
            artifact,
            conversation_id=CID,
            cognitive_request_id=RID,
            recall_id=RECALL_ID,
            request_fingerprint=(
                self._expected_fingerprint()
            ),
        )

    def test_validator_accepts_valid_identity(self):
        self.assertTrue(
            self._valid(self._artifact())
        )

    def test_validator_recomputes_fingerprint(self):
        artifact = self._artifact()

        # The artifact must carry the fingerprint recomputed from its
        # own identity -- not one it merely asserts about itself.
        self.assertEqual(
            artifact["request_fingerprint"],
            self._expected_fingerprint(),
        )
        self.assertTrue(self._valid(artifact))

        # Any other valid 64-hex fingerprint is refused.
        tampered = self._artifact()
        tampered["request_fingerprint"] = "b" * 64

        self.assertFalse(self._valid(tampered))

        # A non-hex fingerprint is refused too.
        malformed = self._artifact()
        malformed["request_fingerprint"] = "z" * 64

        self.assertFalse(self._valid(malformed))

    def test_validator_rejects_tampered_identity(self):
        cases = (
            {"conversation_id": "ctx_" + "9" * 16},
            {"cognitive_request_id": "ctxreq_" + "9" * 32},
            {"recall_id": "recall_" + "8" * 32},
            {"mode": "live"},
            {"version": "related-memory-recall.v2"},
            {"anchor_memory_id": ""},
            {"request_fingerprint": "b" * 64},
            {"memories": "nope"},
            {"source_flash_request_id": RID_B},
            {"source_flash_unified_revision": None},
            {"requested_scope": "full"},
            {"included_count": 99},
        )

        for changes in cases:
            with self.subTest(changes=changes):
                artifact = self._artifact()
                artifact.update(changes)

                self.assertFalse(
                    self._valid(artifact)
                )

    def test_validator_rejects_bad_memory_structure(
        self,
    ):
        cases = (
            {"memory_id": ""},
            {"memory_id": 17},
            {"content": ""},
            {"content": 17},
            {"content": "x" * 5000},
            {"reason": "guessed"},
            {"reason": None},
            {"rank": 0},
            {"rank": True},
            {"rank": 5},
        )

        for changes in cases:
            with self.subTest(changes=changes):
                artifact = self._artifact()
                artifact["memories"][1].update(
                    changes
                )

                self.assertFalse(
                    self._valid(artifact)
                )

    def test_validator_requires_anchor_first(self):
        # The first memory must be the anchor.
        wrong_id = self._artifact()
        wrong_id["memories"][0][
            "memory_id"
        ] = "mem-3"

        self.assertFalse(self._valid(wrong_id))

        # ...with reason anchor_memory...
        wrong_reason = self._artifact()
        wrong_reason["memories"][0][
            "reason"
        ] = "related_memory"

        self.assertFalse(
            self._valid(wrong_reason)
        )

        # ...and rank 1.
        wrong_rank = self._artifact()
        wrong_rank["memories"][0]["rank"] = 2
        wrong_rank["memories"][1]["rank"] = 1

        self.assertFalse(
            self._valid(wrong_rank)
        )

    def test_validator_rejects_duplicate_memories(
        self,
    ):
        # A duplicate memory id.
        duplicate_id = self._artifact()
        duplicate_id["memories"][1][
            "memory_id"
        ] = "mem-1"
        duplicate_id["memories"][1][
            "content"
        ] = "other body"

        self.assertFalse(
            self._valid(duplicate_id)
        )

        # Normalized-equal content.
        duplicate_text = self._artifact()
        duplicate_text["memories"][1][
            "content"
        ] = (
            "  "
            + duplicate_text["memories"][0][
                "content"
            ].upper()
            + "  "
        )

        self.assertFalse(
            self._valid(duplicate_text)
        )


class RelatedRecallCorruptionTests(
    unittest.TestCase,
):
    """A tampered persisted recall must read as missing."""

    def _path(self, root):
        return (
            Path(root)
            / "related_recall"
            / CID
            / RID
            / (RECALL_ID + ".json")
        )

    def _seed(self, root):
        write_recall(
            root,
            recall_artifact(
                ["mem-1", "mem-2", "mem-3"],
                recall_id=RECALL_ID,
            ),
        )

    def _read(self, root):
        with recall_env(root):
            return read_related_recall(
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )

    def _tamper(self, root, **changes):
        path = self._path(root)

        artifact = json.loads(
            path.read_text(encoding="utf-8")
        )

        artifact.update(changes)

        path.write_text(
            json.dumps(artifact),
            encoding="utf-8",
        )

    def test_valid_artifact_reads(self):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root)

            artifact = self._read(root)

        self.assertIsInstance(artifact, dict)
        self.assertEqual(
            artifact["anchor_memory_id"], "mem-1"
        )

    def test_tampered_anchor_with_stale_fingerprint(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root)

            # Move the anchor but keep the original fingerprint: the
            # recomputed fingerprint no longer matches.
            self._tamper(
                root, anchor_memory_id="mem-2"
            )

            artifact = self._read(root)

        self.assertIsNone(artifact)

    def test_tampered_source_identity(self):
        cases = (
            {"source_flash_request_id": RID_B},
            {"source_flash_unified_revision": None},
            {"source_flash_unified_revision": 0},
            {"requested_scope": "full"},
            {"request_fingerprint": "b" * 64},
            {"included_count": 99},
            {"conversation_id": CID_B},
            {"cognitive_request_id": RID_B},
            {"recall_id": "recall_" + "8" * 32},
            {"mode": "live"},
        )

        for changes in cases:
            with self.subTest(changes=changes):
                with tempfile.TemporaryDirectory() as (
                    root
                ):
                    self._seed(root)

                    self._tamper(root, **changes)

                    artifact = self._read(root)

                self.assertIsNone(artifact)

    def test_tampered_memory_list_is_unusable(
        self,
    ):
        for changes in (
            {"memory_id": "mem-1"},
            {"content": ""},
            {"reason": "guessed"},
            {"rank": 7},
        ):
            with self.subTest(changes=changes):
                with tempfile.TemporaryDirectory() as (
                    root
                ):
                    self._seed(root)

                    path = self._path(root)

                    artifact = json.loads(
                        path.read_text(
                            encoding="utf-8"
                        )
                    )

                    artifact["memories"][1].update(
                        changes
                    )

                    path.write_text(
                        json.dumps(artifact),
                        encoding="utf-8",
                    )

                    result = self._read(root)

                self.assertIsNone(result)

    def test_first_memory_must_be_anchor(self):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root)

            path = self._path(root)

            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )

            artifact["memories"][0][
                "memory_id"
            ] = "mem-3"

            path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )

            result = self._read(root)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()