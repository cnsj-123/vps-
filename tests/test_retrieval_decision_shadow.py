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


class PerCandidateObservationTests(
    unittest.TestCase
):
    """The per-candidate API of the same conservative.v1 policy."""

    def _items(self):
        now = datetime(
            2026,
            9,
            26,
            12,
            0,
            tzinfo=timezone.utc,
        )

        def ts(hours):
            return (
                now
                - timedelta(hours=hours)
            ).isoformat()

        return now, [
            {
                "id": "recent",
                "content": "recent text",
                "metadata": {
                    "last_active": ts(2),
                },
            },
            {
                "id": "base",
                "content": "same text",
                "metadata": {
                    "last_active": ts(100),
                },
            },
            {
                "id": "base",
                "content": "other text",
                "metadata": {
                    "last_active": ts(100),
                },
            },
            {
                "id": "other",
                "content": "same text",
                "metadata": {
                    "last_active": ts(100),
                },
            },
            {
                "id": "middle",
                "content": "middle text",
                "metadata": {
                    "last_active": ts(48),
                },
            },
            {
                "id": "missing",
                "content": "missing text",
                "metadata": {},
            },
        ]

    def test_per_candidate_decisions(self):
        now, items = self._items()

        observer = (
            RetrievalDecisionShadowObserver()
        )

        decisions = (
            observer.observe_candidates(
                items,
                now=now,
            )
        )

        self.assertEqual(
            [
                (
                    decision.memory_id,
                    decision.would_keep,
                    decision.reason,
                )
                for decision in decisions
            ],
            [
                ("recent", False, "recent_24h"),
                ("base", True, "keep"),
                ("base", False, "duplicate_id"),
                (
                    "other",
                    False,
                    "exact_text_duplicate",
                ),
                ("middle", True, "keep"),
                ("missing", True, "keep"),
            ],
        )

        # ``to_dict`` carries identity / order / decision only.
        self.assertEqual(
            set(decisions[0].to_dict()),
            {
                "memory_id",
                "candidate_fingerprint",
                "index",
                "would_keep",
                "reason",
            },
        )

    def test_fingerprint_is_deterministic_and_content_bound(
        self,
    ):
        from ombrebrain.context.retrieval_decision_shadow import (
            candidate_fingerprint,
        )

        first = {
            "id": "m1",
            "content": "same text",
        }

        second = {
            "id": "m1",
            "content": "other text",
        }

        self.assertEqual(
            candidate_fingerprint(first),
            candidate_fingerprint(
                dict(first)
            ),
        )

        # Same id, different content -> different identity.
        self.assertNotEqual(
            candidate_fingerprint(first),
            candidate_fingerprint(second),
        )

        # Normalization is stable across whitespace.
        self.assertEqual(
            candidate_fingerprint(
                {
                    "id": " m1 ",
                    "content": "  same text ",
                }
            ),
            candidate_fingerprint(first),
        )

        # No raw text ever appears in the fingerprint.
        self.assertNotIn(
            "same text",
            candidate_fingerprint(first),
        )

    def test_aggregate_is_derived_from_per_candidate(
        self,
    ):
        now, items = self._items()

        observer = (
            RetrievalDecisionShadowObserver()
        )

        aggregate = observer.observe(
            items,
            now=now,
        )

        decisions = (
            observer.observe_candidates(
                items,
                now=now,
            )
        )

        self.assertEqual(
            aggregate.would_keep,
            sum(
                1
                for decision in decisions
                if decision.would_keep
            ),
        )

        self.assertEqual(
            aggregate.would_drop_total,
            sum(
                1
                for decision in decisions
                if not decision.would_keep
            ),
        )

        self.assertEqual(
            aggregate.would_drop_recent_24h,
            1,
        )
        self.assertEqual(
            aggregate.would_drop_duplicate_id,
            1,
        )
        self.assertEqual(
            aggregate.would_drop_exact_text_duplicate,
            1,
        )
        self.assertEqual(
            aggregate.would_keep_recent_24_72h,
            1,
        )
        self.assertEqual(
            aggregate.would_keep_missing_last_active,
            1,
        )

    def test_input_is_not_mutated(self):
        now, items = self._items()
        snapshot = repr(items)

        (
            RetrievalDecisionShadowObserver()
            .observe_candidates(
                items,
                now=now,
            )
        )

        self.assertEqual(repr(items), snapshot)

    def test_non_dict_items_fail_open(self):
        decisions = (
            RetrievalDecisionShadowObserver()
            .observe_candidates(
                ["not-a-dict", None],
                now=datetime.now(
                    timezone.utc
                ),
            )
        )

        self.assertEqual(len(decisions), 2)

        for decision in decisions:
            self.assertTrue(decision.would_keep)
            self.assertEqual(decision.reason, "keep")


if __name__ == "__main__":
    unittest.main()
