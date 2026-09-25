from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from ombrebrain.context.retrieval import (
    scorer as scorer_module,
)
from ombrebrain.context.retrieval.candidate import (
    project_candidate,
)
from ombrebrain.context.retrieval.scorer import (
    importance_score,
    recency_score,
    relative_semantic_score,
    score_candidate,
    semantic_similarity,
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


def bucket(
    *,
    id="c",
    importance=5,
    hours_old=None,
):
    """Real Ombre-Brain bucket shape (importance in metadata)."""

    metadata = {
        "importance": importance,
    }

    if hours_old is not None:
        metadata["last_active"] = (
            NOW - timedelta(hours=hours_old)
        ).isoformat()

    return {
        "id": id,
        "content": "secret-content",
        "metadata": metadata,
    }


def candidate(
    *,
    semantic=0.8,
    importance=5,
    hours_old=None,
):
    return project_candidate(
        bucket(
            importance=importance,
            hours_old=hours_old,
        ),
        semantic_similarity=semantic,
    )


class DeterministicScoreTests(
    unittest.TestCase
):

    def test_same_inputs_same_score(self):
        first = score_candidate(
            candidate(
                semantic=0.9,
                hours_old=10,
                importance=8,
            ),
            now=NOW,
            max_semantic_similarity=0.9,
        )

        second = score_candidate(
            candidate(
                semantic=0.9,
                hours_old=10,
                importance=8,
            ),
            now=NOW,
            max_semantic_similarity=0.9,
        )

        self.assertEqual(first, second)

    def test_default_formula_weights(self):
        # Fully neutral except semantic.
        value = score_candidate(
            candidate(
                semantic=1.0,
                importance=1,
            ),
            now=NOW,
            max_semantic_similarity=1.0,
        )

        # 0.5 * 1.0
        # + 0.2 * 0.5 (missing timestamp is neutral)
        # + 0.2 * 0.0 (importance 1 -> 0.0)
        # + 0.1 * 1.0 (relative semantic 1.0/1.0)
        expected = (
            0.5 * 1.0
            + 0.2 * 0.5
            + 0.2 * 0.0
            + 0.1 * 1.0
        )

        self.assertAlmostEqual(
            value,
            expected,
            places=9,
        )

    def test_fixed_weights_are_unchanged(
        self,
    ):
        # The shadow formula weights are frozen this round.
        self.assertEqual(
            scorer_module._SEMANTIC_WEIGHT,
            0.5,
        )
        self.assertEqual(
            scorer_module._RECENCY_WEIGHT,
            0.2,
        )
        self.assertEqual(
            scorer_module._IMPORTANCE_WEIGHT,
            0.2,
        )
        self.assertEqual(
            scorer_module
            ._RELATIVE_SEMANTIC_WEIGHT,
            0.1,
        )

    def test_score_is_bounded_0_1(self):
        lowest = score_candidate(
            candidate(
                semantic=0.0,
                importance=1,
            ),
            now=NOW,
            max_semantic_similarity=1.0,
        )

        self.assertEqual(
            lowest,
            0.2 * 0.5,
        )

        highest = score_candidate(
            candidate(
                semantic=1.0,
                importance=10,
                hours_old=0,
            ),
            now=NOW,
            max_semantic_similarity=1.0,
        )

        self.assertLessEqual(highest, 1.0)
        self.assertGreaterEqual(highest, 0.0)

    def test_score_does_not_mutate_candidate(
        self,
    ):
        item = candidate(hours_old=3)

        before = item

        score_candidate(
            item,
            now=NOW,
            max_semantic_similarity=0.8,
        )

        self.assertIs(item, before)
        self.assertEqual(
            item.features.semantic_similarity,
            0.8,
        )

    def test_malformed_candidate_never_crashes(
        self,
    ):
        for bad in (
            None,
            5,
            "x",
            [],
            {"id": 1},
        ):
            value = score_candidate(
                bad,
                now=NOW,
                max_semantic_similarity=0.9,
            )

            self.assertGreaterEqual(
                value, 0.0, bad
            )
            self.assertLessEqual(
                value, 1.0, bad
            )

    def test_semantic_similarity_reads_features(
        self,
    ):
        item = candidate(semantic=0.42)

        self.assertEqual(
            semantic_similarity(item),
            0.42,
        )
        self.assertEqual(
            semantic_similarity("not-a-candidate"),
            0.0,
        )


class BoundaryValueTests(
    unittest.TestCase
):

    def test_recency_boundaries(self):
        # age 0 -> 1.0
        self.assertEqual(
            recency_score(
                candidate(hours_old=0),
                now=NOW,
            ),
            1.0,
        )

        # half-life (default 72h) -> 0.5
        self.assertAlmostEqual(
            recency_score(
                candidate(hours_old=72),
                now=NOW,
            ),
            0.5,
            places=9,
        )

        # missing timestamp -> neutral 0.5
        self.assertEqual(
            recency_score(
                candidate(hours_old=None),
                now=NOW,
            ),
            0.5,
        )

        # far past -> decays toward 0
        self.assertLess(
            recency_score(
                candidate(hours_old=720),
                now=NOW,
            ),
            0.01,
        )

        # future timestamp clamps to age 0
        future = project_candidate(
            {
                "id": "f",
                "metadata": {
                    "last_active": (
                        NOW
                        + timedelta(hours=5)
                    ).isoformat(),
                },
            }
        )

        self.assertEqual(
            recency_score(
                future,
                now=NOW,
            ),
            1.0,
        )

    def test_custom_half_life(self):
        self.assertAlmostEqual(
            recency_score(
                candidate(hours_old=24),
                now=NOW,
                recency_half_life_hours=24.0,
            ),
            0.5,
            places=9,
        )

    def test_importance_boundaries(self):
        self.assertEqual(
            importance_score(
                candidate(importance=1)
            ),
            0.0,
        )
        self.assertEqual(
            importance_score(
                candidate(importance=10)
            ),
            1.0,
        )

    def test_relative_semantic_boundaries(self):
        # No best score -> 0.0
        self.assertEqual(
            relative_semantic_score(
                candidate(semantic=0.9),
                max_semantic_similarity=0.0,
            ),
            0.0,
        )

        # Best candidate -> 1.0
        self.assertEqual(
            relative_semantic_score(
                candidate(semantic=0.9),
                max_semantic_similarity=0.9,
            ),
            1.0,
        )

        # Half of best -> 0.5
        self.assertAlmostEqual(
            relative_semantic_score(
                candidate(semantic=0.45),
                max_semantic_similarity=0.9,
            ),
            0.5,
            places=9,
        )

    def test_relative_semantic_is_derived_from_semantic_only(
        self,
    ):
        # Two candidates with the same raw similarity get the same
        # relative score: it is not an independent signal.
        first = relative_semantic_score(
            candidate(
                semantic=0.45,
                importance=1,
            ),
            max_semantic_similarity=0.9,
        )
        second = relative_semantic_score(
            candidate(
                semantic=0.45,
                importance=10,
            ),
            max_semantic_similarity=0.9,
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(
            first,
            0.5,
            places=9,
        )

    def test_naive_now_is_treated_as_utc(
        self,
    ):
        value = score_candidate(
            candidate(hours_old=1),
            now=NOW.replace(tzinfo=None),
            max_semantic_similarity=0.8,
        )

        self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()
