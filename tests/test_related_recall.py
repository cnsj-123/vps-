from __future__ import annotations

import unittest

from _recall_fixtures import (
    CID,
    RID,
    FakeBucketManager,
    FakeRetrievalAdapter,
    bucket,
    memory,
    recall_request,
)

from ombrebrain.context.related_recall import (
    build_recall_query,
    build_related_recall,
    is_valid_related_recall_artifact,
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
        return {
            "version": "related-memory-recall.v1",
            "mode": "shadow_only",
            "conversation_id": CID,
            "cognitive_request_id": RID,
            "recall_id": RECALL_ID,
            "anchor_memory_id": "mem-1",
            "request_fingerprint": "a" * 64,
            "memories": [],
        }

    def test_validator_accepts_valid_identity(self):
        self.assertTrue(
            is_valid_related_recall_artifact(
                self._artifact(),
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
                request_fingerprint="a" * 64,
            )
        )

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
        )

        for changes in cases:
            with self.subTest(changes=changes):
                artifact = self._artifact()
                artifact.update(changes)

                self.assertFalse(
                    is_valid_related_recall_artifact(
                        artifact,
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        recall_id=RECALL_ID,
                        request_fingerprint="a" * 64,
                    )
                )


if __name__ == "__main__":
    unittest.main()