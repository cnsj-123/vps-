from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from unittest.mock import AsyncMock

from ombrebrain.context.retrieval import (
    ContextRetrievalAdapter,
)
from ombrebrain.context.retrieval_decision_shadow import (
    RetrievalDecisionShadowObserver,
)


class FakeEngine:

    enabled = True

    def __init__(
        self,
        pairs,
    ):
        self.pairs = pairs

    async def search_similar_strict(
        self,
        query,
        top_k,
    ):
        return list(
            self.pairs
        )


class RetrievalDecisionShadowTests(
    unittest.IsolatedAsyncioTestCase
):

    def test_shadow_policy_is_deterministic(
        self,
    ):
        now = datetime(
            2026,
            9,
            19,
            12,
            0,
            tzinfo=timezone.utc,
        )

        def ts(hours):
            return (
                now
                - timedelta(
                    hours=hours
                )
            ).isoformat()

        items = [
            # Drop: recent <24h.
            {
                "id": "recent",
                "content": "recent text",
                "metadata": {
                    "last_active":
                        ts(2),
                },
            },

            # Keep: old baseline.
            {
                "id": "base",
                "content": "same text",
                "metadata": {
                    "last_active":
                        ts(100),
                },
            },

            # Drop: duplicate id.
            {
                "id": "base",
                "content": "other text",
                "metadata": {
                    "last_active":
                        ts(100),
                },
            },

            # Drop: exact text duplicate.
            {
                "id": "other",
                "content": "same text",
                "metadata": {
                    "last_active":
                        ts(100),
                },
            },

            # Keep: 24-72h.
            {
                "id": "middle",
                "content": "middle text",
                "metadata": {
                    "last_active":
                        ts(48),
                },
            },

            # Keep: missing timestamp.
            {
                "id": "missing",
                "content": "missing text",
                "metadata": {},
            },
        ]

        observer = (
            RetrievalDecisionShadowObserver()
        )

        result = observer.observe(
            items,
            now=now,
        ).to_dict()

        self.assertEqual(
            result["mode"],
            "observe_only",
        )

        self.assertEqual(
            result["policy"],
            "conservative.v1",
        )

        self.assertEqual(
            result["input_count"],
            6,
        )

        self.assertEqual(
            result["would_keep"],
            3,
        )

        self.assertEqual(
            result["would_drop_total"],
            3,
        )

        self.assertEqual(
            result[
                "would_drop_recent_24h"
            ],
            1,
        )

        self.assertEqual(
            result[
                "would_drop_duplicate_id"
            ],
            1,
        )

        self.assertEqual(
            result[
                "would_drop_exact_text_duplicate"
            ],
            1,
        )

        self.assertEqual(
            result[
                "would_keep_recent_24_72h"
            ],
            1,
        )

        self.assertEqual(
            result[
                "would_keep_missing_last_active"
            ],
            1,
        )

        self.assertEqual(
            result["input_count"],
            (
                result["would_keep"]
                + result[
                    "would_drop_total"
                ]
            ),
        )

    async def test_shadow_never_filters_real_results(
        self,
    ):
        now = datetime.now(
            timezone.utc
        )

        recent = (
            now
            - timedelta(hours=1)
        ).isoformat()

        old = (
            now
            - timedelta(days=5)
        ).isoformat()

        bucket_mgr = AsyncMock()

        bucket_mgr.search.return_value = [
            {
                "id": "recent",
                "content": "recent memory",
                "metadata": {
                    "last_active":
                        recent,
                },
            },
            {
                "id": "dup",
                "content": "first duplicate id",
                "metadata": {
                    "last_active":
                        old,
                },
            },
            {
                "id": "dup",
                "content": "second duplicate id",
                "metadata": {
                    "last_active":
                        old,
                },
            },
        ]

        adapter = ContextRetrievalAdapter(
            bucket_mgr=
                bucket_mgr,
            embedding_engine=
                FakeEngine(
                    [
                        ("recent", 0.90),
                        ("dup", 0.90),
                    ]
                ),
        )

        result = await adapter.retrieve(
            "query",
            max_results=8,
        )

        # Critical invariant:
        # shadow decisions do not modify retrieval.
        self.assertEqual(
            len(result),
            3,
        )

        telemetry = (
            adapter.last_telemetry
        )

        self.assertEqual(
            telemetry[
                "included_count"
            ],
            3,
        )

        shadow = telemetry[
            "decision_shadow"
        ]

        self.assertEqual(
            shadow["input_count"],
            3,
        )

        self.assertEqual(
            shadow[
                "would_drop_recent_24h"
            ],
            1,
        )

        self.assertEqual(
            shadow[
                "would_drop_duplicate_id"
            ],
            1,
        )

        self.assertEqual(
            shadow["would_keep"],
            1,
        )

        self.assertEqual(
            shadow["would_drop_total"],
            2,
        )

        self.assertEqual(
            telemetry["filter_mode"],
            "observe_only",
        )


if __name__ == "__main__":
    unittest.main()
