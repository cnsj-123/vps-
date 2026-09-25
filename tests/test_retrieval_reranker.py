from __future__ import annotations

import inspect
import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from ombrebrain.context.retrieval.candidate import (
    project_candidate,
)
from ombrebrain.context.retrieval.reranker import (
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
    importance=5,
):
    metadata = {
        "importance": importance,
    }

    if hours_old is not None:
        metadata["last_active"] = (
            NOW - timedelta(hours=hours_old)
        ).isoformat()

    return project_candidate(
        {
            "id": id,
            "content": "secret-content",
            "metadata": metadata,
        },
        semantic_similarity=semantic,
    )


def ids(ranked):
    return [
        item.bucket.get("id")
        for item, _score in ranked
    ]


class OrderingTests(
    unittest.TestCase
):

    def test_sorted_by_score_descending(self):
        # semantic: 0.9 > 0.7 > 0.5, all else neutral.
        result = rank_candidates(
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
            now=NOW,
            max_semantic_similarity=0.9,
        )

        self.assertEqual(
            ids(result),
            ["high", "mid", "low"],
        )

    def test_recency_can_reorder_semantic_order(
        self,
    ):
        # Same semantic tier, but one memory is fresh and the other
        # ancient: recency must lift the fresh one.
        result = rank_candidates(
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
            now=NOW,
            max_semantic_similarity=0.8,
        )

        self.assertEqual(
            ids(result),
            ["fresh", "ancient"],
        )

    def test_equal_scores_keep_input_order(self):
        # Stable sort: identical inputs keep relative order.
        result = rank_candidates(
            [
                candidate("a", semantic=0.6),
                candidate("b", semantic=0.6),
                candidate("c", semantic=0.6),
            ],
            now=NOW,
            max_semantic_similarity=0.6,
        )

        self.assertEqual(
            ids(result),
            ["a", "b", "c"],
        )

    def test_input_is_not_mutated(self):
        original = [
            candidate("low", semantic=0.2),
            candidate("high", semantic=0.9),
        ]

        snapshot = list(original)

        rank_candidates(
            original,
            now=NOW,
            max_semantic_similarity=0.9,
        )

        self.assertEqual(original, snapshot)

    def test_empty_input_returns_empty(self):
        result = rank_candidates(
            [],
            now=NOW,
            max_semantic_similarity=0.0,
        )

        self.assertEqual(result, [])

    def test_rank_returns_canonical_candidates_and_scores(
        self,
    ):
        from ombrebrain.retrieval import (
            RetrievalCandidate,
        )

        scored = rank_candidates(
            [
                candidate("a", semantic=0.9),
                candidate("b", semantic=0.1),
            ],
            now=NOW,
            max_semantic_similarity=0.9,
        )

        scores = [value for _item, value in scored]

        self.assertEqual(ids(scored), ["a", "b"])
        self.assertEqual(
            scores,
            sorted(scores, reverse=True),
        )
        self.assertIsInstance(
            scored[0][0],
            RetrievalCandidate,
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
        # A zero-scoring candidate must NOT be dropped: there is no
        # threshold and no admission decision here.
        result = rank_candidates(
            [
                candidate("high", semantic=0.9),
                candidate(
                    "zero",
                    semantic=0.0,
                    importance=1,
                ),
                candidate("mid", semantic=0.5),
            ],
            now=NOW,
            max_semantic_similarity=0.9,
        )

        self.assertEqual(
            sorted(ids(result)),
            ["high", "mid", "zero"],
        )
        self.assertEqual(len(result), 3)

    def test_rank_never_truncates(self):
        payload = [
            candidate(f"c{index}", semantic=0.5)
            for index in range(12)
        ]

        result = rank_candidates(
            payload,
            now=NOW,
            max_semantic_similarity=0.5,
        )

        self.assertEqual(len(result), 12)
        self.assertEqual(
            {item.bucket.get("id") for item, _s in result},
            {item.bucket.get("id") for item in payload},
        )

    def test_rank_exposes_no_selection_parameters(
        self,
    ):
        forbidden = (
            "top_k",
            "score_threshold",
            "threshold",
            "limit",
        )

        parameters = inspect.signature(
            rank_candidates
        ).parameters

        for name in forbidden:
            self.assertNotIn(
                name,
                parameters,
                f"rank_candidates must not"
                f" select via {name}",
            )

    def test_shadow_ranker_class_is_gone(self):
        import ombrebrain.context.retrieval.reranker as module

        self.assertFalse(
            hasattr(module, "ShadowRanker")
        )


if __name__ == "__main__":
    unittest.main()
