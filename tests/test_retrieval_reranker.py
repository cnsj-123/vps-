from __future__ import annotations

import inspect
import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from ombrebrain.context.retrieval import (
    ShadowCandidate,
    ShadowRanker,
    ShadowScorer,
    ShadowScoringWeights,
    ShadowScoringContext,
    rank_candidates,
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
    return ShadowCandidate(
        id=id,
        semantic_similarity=semantic,
        timestamp=(
            NOW - timedelta(hours=hours_old)
            if hours_old is not None
            else None
        ),
        importance=importance,
    )


def ctx(max_semantic=None):
    return ShadowScoringContext(
        now=NOW,
        max_semantic_similarity=(
            max_semantic
            if max_semantic is not None
            else 0.0
        ),
    )


def ids(ranked):
    return [c.id for c, _score in ranked]


class OrderingTests(
    unittest.TestCase
):

    def test_sorted_by_score_descending(self):
        # semantic: 0.9 > 0.7 > 0.5, all else neutral.
        ranker = ShadowRanker()
        context = ctx(max_semantic=0.9)

        result = ranker.rank(
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
            ids(result),
            ["high", "mid", "low"],
        )

    def test_recency_can_reorder_semantic_order(
        self,
    ):
        # Same semantic tier, but one memory is fresh and
        # the other ancient: recency must lift the fresh one.
        ranker = ShadowRanker()
        context = ctx(max_semantic=0.8)

        result = ranker.rank(
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
            ids(result),
            ["fresh", "ancient"],
        )

    def test_equal_scores_keep_input_order(self):
        # Stable sort: identical inputs keep relative order.
        ranker = ShadowRanker()
        context = ctx(max_semantic=0.6)

        result = ranker.rank(
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
            ids(result),
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

        ShadowRanker().rank(
            original,
            ctx(max_semantic=0.9),
        )

        self.assertEqual(
            original,
            snapshot,
        )

    def test_empty_input_returns_empty(self):
        result = ShadowRanker().rank(
            [],
            ctx(),
        )

        self.assertEqual(result, [])

    def test_raw_buckets_are_normalized(self):
        ranker = ShadowRanker()

        result = ranker.rank(
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

        # Legacy calibrated context_relevance is not the raw
        # semantic signal, so importance alone orders these.
        self.assertEqual(
            ids(result),
            ["raw-high", "raw-low"],
        )

    def test_rank_returns_scores(self):
        ranker = ShadowRanker()

        scored = ranker.rank(
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

        scores = [
            s for _c, s in scored
        ]

        self.assertEqual(
            ids(scored),
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


class NoSelectionTests(
    unittest.TestCase
):
    """The ranker only ranks. Selection is not its job."""

    def test_rank_keeps_every_candidate(self):
        # A zero-scoring candidate must NOT be dropped: there is
        # no threshold and no admission decision here.
        result = rank_candidates(
            [
                candidate(
                    "high",
                    semantic=0.9,
                ),
                candidate(
                    "zero",
                    semantic=0.0,
                    importance=0.0,
                ),
                candidate(
                    "mid",
                    semantic=0.5,
                ),
            ],
            context=ctx(max_semantic=0.9),
        )

        self.assertEqual(
            sorted(ids(result)),
            ["high", "mid", "zero"],
        )
        self.assertEqual(len(result), 3)

    def test_rank_never_truncates(self):
        payload = [
            candidate(
                f"c{index}",
                semantic=0.5,
            )
            for index in range(12)
        ]

        result = rank_candidates(
            payload,
            context=ctx(max_semantic=0.5),
        )

        self.assertEqual(len(result), 12)
        self.assertEqual(
            {c.id for c, _s in result},
            {c.id for c in payload},
        )

    def test_ranker_exposes_no_selection_parameters(
        self,
    ):
        forbidden = (
            "top_k",
            "score_threshold",
            "threshold",
            "limit",
        )

        for callable_ in (
            ShadowRanker.rank,
            rank_candidates,
        ):
            parameters = inspect.signature(
                callable_
            ).parameters

            for name in forbidden:
                self.assertNotIn(
                    name,
                    parameters,
                    f"{callable_.__name__}"
                    f" must not select via {name}",
                )

    def test_ranker_constructor_exposes_no_selector(
        self,
    ):
        parameters = inspect.signature(
            ShadowRanker.__init__
        ).parameters

        self.assertNotIn(
            "top_k",
            parameters,
        )
        self.assertNotIn(
            "score_threshold",
            parameters,
        )

    def test_custom_scorer_is_used(self):
        scorer = ShadowScorer(
            ShadowScoringWeights(
                semantic=1.0,
                recency=0.0,
                importance=0.0,
                relative_semantic=0.0,
            )
        )

        result = ShadowRanker(
            scorer
        ).rank(
            [
                candidate(
                    "a",
                    semantic=0.3,
                ),
                candidate(
                    "b",
                    semantic=0.8,
                ),
            ],
            ctx(max_semantic=0.8),
        )

        self.assertEqual(
            ids(result),
            ["b", "a"],
        )


if __name__ == "__main__":
    unittest.main()
