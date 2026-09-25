from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from ombrebrain.context.retrieval import (
    RetrievalCandidate,
    RetrievalReranker,
    RetrievalScorer,
    RetrievalScorerWeights,
    RetrievalScoringContext,
)


NOW = datetime(
    2026,
    9,
    24,
    12,
    0,
    0,
    tzinfo=timezone.utc,
)


def candidate(
    id,
    *,
    semantic=0.5,
    hours_old=None,
    importance=0.5,
):
    return RetrievalCandidate(
        id=id,
        semantic_score=semantic,
        timestamp=(
            NOW - timedelta(hours=hours_old)
            if hours_old is not None
            else None
        ),
        importance=importance,
    )


def ctx(max_semantic=None):
    return RetrievalScoringContext(
        now=NOW,
        max_semantic_score=(
            max_semantic
            if max_semantic is not None
            else 0.0
        ),
    )


class OrderingTests(
    unittest.TestCase
):

    def test_sorted_by_score_descending(self):
        # semantic: 0.9 > 0.7 > 0.5, all else neutral.
        reranker = RetrievalReranker()
        context = ctx(max_semantic=0.9)

        result = reranker.rerank(
            [
                candidate(
                    "low",
                    semantic=0.5,
                ),
                candidate(
                    "high",
                    semantic=0.9,
                ),
                candidate(
                    "mid",
                    semantic=0.7,
                ),
            ],
            context,
        )

        self.assertEqual(
            [c.id for c in result],
            ["high", "mid", "low"],
        )

    def test_recency_can_reorder_semantic_order(
        self,
    ):
        # Same semantic tier, but one memory is fresh and
        # the other ancient: recency must lift the fresh one.
        reranker = RetrievalReranker()
        context = ctx(max_semantic=0.8)

        result = reranker.rerank(
            [
                candidate(
                    "ancient",
                    semantic=0.8,
                    hours_old=2000,
                ),
                candidate(
                    "fresh",
                    semantic=0.8,
                    hours_old=1,
                ),
            ],
            context,
        )

        self.assertEqual(
            [c.id for c in result],
            ["fresh", "ancient"],
        )

    def test_equal_scores_keep_input_order(self):
        # Stable sort: identical inputs keep relative order.
        reranker = RetrievalReranker()
        context = ctx(max_semantic=0.6)

        result = reranker.rerank(
            [
                candidate(
                    "a",
                    semantic=0.6,
                ),
                candidate(
                    "b",
                    semantic=0.6,
                ),
                candidate(
                    "c",
                    semantic=0.6,
                ),
            ],
            context,
        )

        self.assertEqual(
            [c.id for c in result],
            ["a", "b", "c"],
        )

    def test_input_list_is_not_mutated(self):
        original = [
            candidate(
                "low",
                semantic=0.2,
            ),
            candidate(
                "high",
                semantic=0.9,
            ),
        ]

        snapshot = list(original)

        RetrievalReranker().rerank(
            original,
            ctx(max_semantic=0.9),
        )

        self.assertEqual(
            original,
            snapshot,
        )

    def test_empty_input_returns_empty(self):
        result = RetrievalReranker().rerank(
            [],
            ctx(),
        )

        self.assertEqual(result, [])

    def test_raw_buckets_are_normalized(self):
        reranker = RetrievalReranker()

        result = reranker.rerank(
            [
                {
                    "id": "raw-low",
                    "context_relevance": 0.2,
                    "importance": 1,
                    "metadata": {},
                },
                {
                    "id": "raw-high",
                    "context_relevance": 0.9,
                    "importance": 10,
                    "metadata": {},
                },
            ],
            ctx(max_semantic=0.9),
        )

        self.assertEqual(
            [c.id for c in result],
            ["raw-high", "raw-low"],
        )

    def test_rank_returns_scores(self):
        reranker = RetrievalReranker()

        scored = reranker.rank(
            [
                candidate(
                    "a",
                    semantic=0.9,
                ),
                candidate(
                    "b",
                    semantic=0.1,
                ),
            ],
            ctx(max_semantic=0.9),
        )

        ids = [
            c.id for c, _s in scored
        ]
        scores = [
            s for _c, s in scored
        ]

        self.assertEqual(
            ids,
            ["a", "b"],
        )
        self.assertEqual(
            scores,
            sorted(
                scores,
                reverse=True,
            ),
        )

        self.assertIsInstance(
            scored[0][1],
            float,
        )


class TopKTests(
    unittest.TestCase
):

    def test_top_k_truncates(self):
        reranker = RetrievalReranker(
            top_k=2,
        )

        result = reranker.rerank(
            [
                candidate(
                    "a",
                    semantic=0.9,
                ),
                candidate(
                    "b",
                    semantic=0.5,
                ),
                candidate(
                    "c",
                    semantic=0.7,
                ),
            ],
            ctx(max_semantic=0.9),
        )

        self.assertEqual(
            [c.id for c in result],
            ["a", "c"],
        )

    def test_top_k_one(self):
        reranker = RetrievalReranker(
            top_k=1,
        )

        result = reranker.rerank(
            [
                candidate(
                    "a",
                    semantic=0.3,
                ),
                candidate(
                    "b",
                    semantic=0.9,
                ),
            ],
            ctx(max_semantic=0.9),
        )

        self.assertEqual(
            [c.id for c in result],
            ["b"],
        )

    def test_invalid_top_k_is_ignored(self):
        for bad in (0, -3, "x", 2.5):
            reranker = RetrievalReranker(
                top_k=bad,
            )

            result = reranker.rerank(
                [
                    candidate(
                        "a",
                        semantic=0.2,
                    ),
                    candidate(
                        "b",
                        semantic=0.9,
                    ),
                ],
                ctx(max_semantic=0.9),
            )

            self.assertEqual(
                len(result),
                2,
                bad,
            )


class ThresholdTests(
    unittest.TestCase
):

    @staticmethod
    def semantic_only_scorer():
        return RetrievalScorer(
            RetrievalScorerWeights(
                semantic=1.0,
                recency=0.0,
                importance=0.0,
                context_match=0.0,
            )
        )

    def test_score_threshold_filters(self):
        reranker = RetrievalReranker(
            self.semantic_only_scorer(),
            score_threshold=0.5,
        )

        result = reranker.rerank(
            [
                candidate(
                    "keep-high",
                    semantic=0.9,
                ),
                candidate(
                    "edge",
                    semantic=0.5,
                ),
                candidate(
                    "drop",
                    semantic=0.3,
                ),
            ],
            ctx(max_semantic=0.0),
        )

        # >= threshold is kept.
        self.assertEqual(
            [c.id for c in result],
            ["keep-high", "edge"],
        )

    def test_threshold_plus_top_k(self):
        reranker = RetrievalReranker(
            self.semantic_only_scorer(),
            top_k=1,
            score_threshold=0.4,
        )

        result = reranker.rerank(
            [
                candidate(
                    "a",
                    semantic=0.9,
                ),
                candidate(
                    "b",
                    semantic=0.6,
                ),
                candidate(
                    "c",
                    semantic=0.1,
                ),
            ],
            ctx(max_semantic=0.0),
        )

        self.assertEqual(
            [c.id for c in result],
            ["a"],
        )

    def test_threshold_that_drops_everything(self):
        reranker = RetrievalReranker(
            score_threshold=0.99,
        )

        result = reranker.rerank(
            [
                candidate(
                    "a",
                    semantic=0.5,
                ),
            ],
            ctx(max_semantic=0.5),
        )

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
