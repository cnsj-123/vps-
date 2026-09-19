from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from ombrebrain.context.retrieval import (
    ContextRetrievalAdapter,
)


class FakeEngine:

    enabled = True

    def __init__(self, pairs):
        self.pairs = pairs

    async def search_similar_strict(
        self,
        query,
        top_k,
    ):
        return list(self.pairs)


class DisabledEngine:

    enabled = False


class ContextRetrievalQualityTests(
    unittest.IsolatedAsyncioTestCase
):

    async def test_empty_query_is_observed(
        self,
    ):
        bucket_mgr = AsyncMock()

        adapter = ContextRetrievalAdapter(
            bucket_mgr=bucket_mgr,
            embedding_engine=
                FakeEngine([]),
        )

        result = await adapter.retrieve(
            ""
        )

        self.assertEqual(
            result,
            [],
        )

        self.assertEqual(
            adapter.last_telemetry[
                "outcome"
            ],
            "empty_query",
        )

        bucket_mgr.search.assert_not_awaited()

    async def test_included_and_rejected_are_counted(
        self,
    ):
        bucket_mgr = AsyncMock()

        bucket_mgr.search.return_value = [
            {
                "id": "good",
                "content": "good memory",
                "metadata": {},
            },
            {
                "id": "low",
                "content": "low memory",
                "metadata": {},
            },
        ]

        adapter = ContextRetrievalAdapter(
            bucket_mgr=bucket_mgr,
            embedding_engine=
                FakeEngine(
                    [
                        ("good", 0.90),
                        ("low", 0.40),
                    ]
                ),
        )

        result = await adapter.retrieve(
            "query"
        )

        self.assertEqual(
            len(result),
            1,
        )

        t = adapter.last_telemetry

        self.assertTrue(
            t["embedding_enabled"]
        )

        self.assertEqual(
            t["semantic_score_count"],
            2,
        )

        self.assertEqual(
            t["raw_match_count"],
            2,
        )

        self.assertEqual(
            t[
                "processed_candidate_count"
            ],
            2,
        )

        self.assertEqual(
            t["relevance_rejected"],
            1,
        )

        self.assertEqual(
            t["included_count"],
            1,
        )

        self.assertEqual(
            t["outcome"],
            "included",
        )

        self.assertEqual(
            t["max_semantic_score"],
            0.9,
        )

        self.assertEqual(
            t["semantic_recall_floor"],
            0.55,
        )

        self.assertEqual(
            t[
                "context_relevance_threshold"
            ],
            0.65,
        )

        self.assertGreaterEqual(
            t[
                "min_context_relevance"
            ],
            t[
                "context_relevance_threshold"
            ],
        )

        self.assertGreaterEqual(
            t[
                "max_context_relevance"
            ],
            t[
                "min_context_relevance"
            ],
        )

        self.assertEqual(
            t["filter_mode"],
            "observe_only",
        )

    async def test_no_search_matches_is_distinguished(
        self,
    ):
        bucket_mgr = AsyncMock()

        bucket_mgr.search.return_value = []

        adapter = ContextRetrievalAdapter(
            bucket_mgr=bucket_mgr,
            embedding_engine=
                FakeEngine(
                    [("other", 0.9)]
                ),
        )

        result = await adapter.retrieve(
            "query"
        )

        self.assertEqual(
            result,
            [],
        )

        self.assertEqual(
            adapter.last_telemetry[
                "outcome"
            ],
            "no_search_matches",
        )

        self.assertEqual(
            adapter.last_telemetry[
                "raw_match_count"
            ],
            0,
        )

    async def test_no_semantic_scores_is_distinguished(
        self,
    ):
        bucket_mgr = AsyncMock()

        bucket_mgr.search.return_value = [
            {
                "id": "m1",
                "content": "memory",
                "metadata": {},
            }
        ]

        adapter = ContextRetrievalAdapter(
            bucket_mgr=bucket_mgr,
            embedding_engine=
                FakeEngine([]),
        )

        result = await adapter.retrieve(
            "query"
        )

        self.assertEqual(
            result,
            [],
        )

        self.assertEqual(
            adapter.last_telemetry[
                "outcome"
            ],
            "no_semantic_scores",
        )

        self.assertEqual(
            adapter.last_telemetry[
                "relevance_rejected"
            ],
            1,
        )

    async def test_embedding_disabled_is_distinguished(
        self,
    ):
        bucket_mgr = AsyncMock()

        bucket_mgr.search.return_value = [
            {
                "id": "m1",
                "content": "memory",
                "metadata": {},
            }
        ]

        adapter = ContextRetrievalAdapter(
            bucket_mgr=bucket_mgr,
            embedding_engine=
                DisabledEngine(),
        )

        result = await adapter.retrieve(
            "query"
        )

        self.assertEqual(
            result,
            [],
        )

        self.assertEqual(
            adapter.last_telemetry[
                "outcome"
            ],
            "embedding_disabled",
        )

        self.assertFalse(
            adapter.last_telemetry[
                "embedding_enabled"
            ]
        )

    async def test_below_threshold_is_distinguished(
        self,
    ):
        bucket_mgr = AsyncMock()

        bucket_mgr.search.return_value = [
            {
                "id": "m1",
                "content": "memory",
                "metadata": {},
            }
        ]

        adapter = ContextRetrievalAdapter(
            bucket_mgr=bucket_mgr,
            embedding_engine=
                FakeEngine(
                    [("m1", 0.40)]
                ),
        )

        result = await adapter.retrieve(
            "query"
        )

        self.assertEqual(
            result,
            [],
        )

        self.assertEqual(
            adapter.last_telemetry[
                "outcome"
            ],
            "below_relevance_threshold",
        )


    async def test_semantic_floor_maps_to_context_threshold(
        self,
    ):
        bucket_mgr = AsyncMock()

        bucket_mgr.search.return_value = [
            {
                "id": "edge",
                "content": "edge memory",
                "metadata": {},
            }
        ]

        adapter = ContextRetrievalAdapter(
            bucket_mgr=bucket_mgr,
            embedding_engine=
                FakeEngine(
                    [
                        (
                            "edge",
                            0.5528,
                        )
                    ]
                ),
        )

        result = await adapter.retrieve(
            "query"
        )

        self.assertEqual(
            len(result),
            1,
        )

        t = adapter.last_telemetry

        self.assertEqual(
            t["max_semantic_score"],
            0.5528,
        )

        self.assertEqual(
            t[
                "semantic_recall_floor"
            ],
            0.55,
        )

        self.assertEqual(
            t[
                "context_relevance_threshold"
            ],
            0.65,
        )

        self.assertGreaterEqual(
            t[
                "min_context_relevance"
            ],
            0.65,
        )

        self.assertEqual(
            t["outcome"],
            "included",
        )

        self.assertEqual(
            t["filter_mode"],
            "observe_only",
        )


if __name__ == "__main__":
    unittest.main()
