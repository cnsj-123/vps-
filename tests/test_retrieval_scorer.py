from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from ombrebrain.context.retrieval import (
    RetrievalCandidate,
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
    *,
    semantic=0.8,
    hours_old=None,
    importance=0.5,
):
    timestamp = (
        NOW - timedelta(hours=hours_old)
        if hours_old is not None
        else None
    )

    return RetrievalCandidate(
        id="c",
        semantic_score=semantic,
        timestamp=timestamp,
        importance=importance,
    )


def context(
    *,
    max_semantic=0.8,
    now=NOW,
):
    return RetrievalScoringContext(
        now=now,
        max_semantic_score=max_semantic,
    )


class DeterministicScoreTests(
    unittest.TestCase
):

    def test_same_inputs_same_score(self):
        scorer = RetrievalScorer()
        c = candidate(
            semantic=0.9,
            hours_old=10,
            importance=0.8,
        )
        ctx = context(
            max_semantic=0.9
        )

        first = scorer.score(c, ctx)
        second = scorer.score(c, ctx)

        self.assertEqual(first, second)

        # Repeat with fresh instances.
        self.assertEqual(
            first,
            RetrievalScorer().score(
                candidate(
                    semantic=0.9,
                    hours_old=10,
                    importance=0.8,
                ),
                context(
                    max_semantic=0.9
                ),
            ),
        )

    def test_default_formula_weights(self):
        scorer = RetrievalScorer()

        # Fully neutral except semantic.
        c = candidate(
            semantic=1.0,
            hours_old=None,
            importance=0.0,
        )

        # context_match = 1.0 / 1.0 = 1.0
        # recency (missing timestamp) = 0.5
        expected = (
            0.5 * 1.0
            + 0.2 * 0.5
            + 0.2 * 0.0
            + 0.1 * 1.0
        )

        self.assertAlmostEqual(
            scorer.score(
                c,
                context(max_semantic=1.0),
            ),
            expected,
            places=9,
        )

    def test_score_is_bounded_0_1(self):
        scorer = RetrievalScorer()
        ctx = context(max_semantic=1.0)

        self.assertEqual(
            scorer.score(
                candidate(
                    semantic=0.0,
                    importance=0.0,
                    hours_old=None,
                ),
                ctx,
            ),
            0.2 * 0.5,
        )

        top = scorer.score(
            candidate(
                semantic=1.0,
                hours_old=0,
                importance=1.0,
            ),
            ctx,
        )

        self.assertLessEqual(top, 1.0)
        self.assertGreaterEqual(top, 0.0)

    def test_custom_weights_are_respected(
        self,
    ):
        scorer = RetrievalScorer(
            RetrievalScorerWeights(
                semantic=1.0,
                recency=0.0,
                importance=0.0,
                context_match=0.0,
            )
        )

        self.assertAlmostEqual(
            scorer.score(
                candidate(semantic=0.42),
                context(max_semantic=0.9),
            ),
            0.42,
            places=9,
        )

    def test_weights_to_dict(self):
        weights = (
            RetrievalScorerWeights()
        )

        self.assertEqual(
            weights.to_dict(),
            {
                "semantic": 0.5,
                "recency": 0.2,
                "importance": 0.2,
                "context_match": 0.1,
            },
        )

    def test_score_accepts_raw_bucket(self):
        scorer = RetrievalScorer()

        # score() normalizes raw dicts on the fly.
        score = scorer.score(
            {
                "id": "raw",
                "context_relevance": 1.0,
                "importance": 1,
                "metadata": {},
            },
            context(max_semantic=1.0),
        )

        self.assertGreater(score, 0.0)


class BoundaryValueTests(
    unittest.TestCase
):

    def test_recency_boundaries(self):
        scorer = RetrievalScorer()
        ctx = context()

        # age 0 -> 1.0
        self.assertEqual(
            scorer.recency_score(
                candidate(hours_old=0),
                ctx,
            ),
            1.0,
        )

        # half-life (default 72h) -> 0.5
        self.assertAlmostEqual(
            scorer.recency_score(
                candidate(hours_old=72),
                ctx,
            ),
            0.5,
            places=9,
        )

        # missing timestamp -> neutral 0.5
        self.assertEqual(
            scorer.recency_score(
                candidate(hours_old=None),
                ctx,
            ),
            0.5,
        )

        # far future age -> decays toward 0
        self.assertLess(
            scorer.recency_score(
                candidate(hours_old=720),
                ctx,
            ),
            0.01,
        )

        # future timestamp clamps to age 0
        future = RetrievalCandidate(
            id="f",
            timestamp=NOW
            + timedelta(hours=5),
        )

        self.assertEqual(
            scorer.recency_score(
                future,
                ctx,
            ),
            1.0,
        )

    def test_custom_half_life(self):
        scorer = RetrievalScorer(
            recency_half_life_hours=24.0,
        )

        self.assertAlmostEqual(
            scorer.recency_score(
                candidate(hours_old=24),
                context(),
            ),
            0.5,
            places=9,
        )

    def test_importance_boundaries(self):
        scorer = RetrievalScorer()

        self.assertEqual(
            scorer.importance_score(
                candidate(importance=0.0)
            ),
            0.0,
        )
        self.assertEqual(
            scorer.importance_score(
                candidate(importance=1.0)
            ),
            1.0,
        )

    def test_context_match_boundaries(self):
        scorer = RetrievalScorer()

        # No best score -> 0.0
        self.assertEqual(
            scorer.context_match_score(
                candidate(semantic=0.9),
                context(max_semantic=0.0),
            ),
            0.0,
        )

        # Best candidate -> 1.0
        self.assertEqual(
            scorer.context_match_score(
                candidate(semantic=0.9),
                context(max_semantic=0.9),
            ),
            1.0,
        )

        # Half of best -> 0.5
        self.assertAlmostEqual(
            scorer.context_match_score(
                candidate(semantic=0.45),
                context(max_semantic=0.9),
            ),
            0.5,
            places=9,
        )

    def test_semantic_boundary_scores(self):
        scorer = RetrievalScorer(
            RetrievalScorerWeights(
                semantic=1.0,
                recency=0.0,
                importance=0.0,
                context_match=0.0,
            )
        )

        ctx = context(max_semantic=0.0)

        self.assertEqual(
            scorer.score(
                candidate(semantic=0.0),
                ctx,
            ),
            0.0,
        )
        self.assertEqual(
            scorer.score(
                candidate(semantic=1.0),
                ctx,
            ),
            1.0,
        )

    def test_naive_context_now_default(self):
        # A naive datetime is treated as UTC and never raises.
        scorer = RetrievalScorer()

        naive_ctx = (
            RetrievalScoringContext(
                now=NOW.replace(
                    tzinfo=None
                ),
                max_semantic_score=1.0,
            )
        )

        score = scorer.score(
            candidate(hours_old=1),
            naive_ctx,
        )

        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
