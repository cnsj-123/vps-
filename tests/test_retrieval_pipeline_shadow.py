from __future__ import annotations

import json
import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from unittest.mock import AsyncMock

from ombrebrain.context.retrieval import (
    ContextRetrievalAdapter,
    RetrievalQualityShadow,
    ShadowScorer,
    ShadowScoringWeights,
    ShadowScoringContext,
    normalize_candidate,
    rank_candidates,
    score_candidate,
)
from ombrebrain.context.retrieval.quality import (
    PRIVACY_SAFE_FIELDS,
)
from ombrebrain.context.service import (
    ContextService,
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
    id,
    *,
    relevance=0.8,
    importance=5,
    hours_old=None,
):
    metadata = {}

    if hours_old is not None:
        metadata["last_active"] = (
            NOW - timedelta(hours=hours_old)
        ).isoformat()

    return {
        "id": id,
        "content": f"secret-{id}",
        "context_relevance": relevance,
        "importance": importance,
        "metadata": metadata,
    }


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


class FakeState:
    def to_dict(self):
        return {}


class FakeStateService:
    def get(self):
        return FakeState(), 1


def build_old_adapter(matches, pairs):
    bucket_mgr = AsyncMock()
    bucket_mgr.list_all.return_value = []
    bucket_mgr.search.return_value = list(
        matches
    )

    return ContextRetrievalAdapter(
        bucket_mgr=bucket_mgr,
        embedding_engine=FakeEngine(pairs),
    )


def build_service(matches, pairs):
    bucket_mgr = AsyncMock()
    bucket_mgr.list_all.return_value = []
    bucket_mgr.search.return_value = list(
        matches
    )
    bucket_mgr.embedding_engine = (
        FakeEngine(pairs)
    )

    return ContextService(
        state_service=FakeStateService(),
        bucket_mgr=bucket_mgr,
    )


# A set where the shadow WOULD reorder: the stale memory is
# slightly ahead of a fresh, important one in the old order.
MATCHES = [
    bucket(
        "stale",
        relevance=0.66,
        importance=1,
        hours_old=2000,
    ),
    bucket(
        "fresh",
        relevance=0.90,
        importance=10,
        hours_old=0,
    ),
]

PAIRS = [
    ("stale", 0.66),
    ("fresh", 0.90),
]

VECTOR_SCORES = dict(PAIRS)


def views():
    return [
        normalize_candidate(
            item,
            semantic_similarity=(
                VECTOR_SCORES.get(
                    item["id"]
                )
            ),
        )
        for item in MATCHES
    ]


class CandidatePoolSplitTests(
    unittest.IsolatedAsyncioTestCase
):
    """B. the pool feeds both legacy selection and the shadow."""

    async def test_pool_is_pre_selection(self):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        pool = (
            await adapter.acquire_candidate_pool(
                "secret-query"
            )
        )

        # The pool is the raw search result, before any
        # threshold or cap.
        self.assertEqual(
            [c["id"] for c in pool["candidates"]],
            ["stale", "fresh"],
        )
        self.assertEqual(
            pool["vector_scores"],
            VECTOR_SCORES,
        )

    async def test_legacy_selection_and_shadow_share_one_pool(
        self,
    ):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        pool = (
            await adapter.acquire_candidate_pool(
                "secret-query"
            )
        )

        # Legacy live selection consumes the pool...
        selection = (
            adapter.select_legacy_results(pool)
        )

        self.assertEqual(
            [
                item["id"]
                for item
                in selection["results"]
            ],
            ["stale", "fresh"],
        )

        # ...and the shadow observes the very same pool.
        event = (
            RetrievalQualityShadow()
            .observe(
                pool["candidates"],
                vector_scores=(
                    pool["vector_scores"]
                ),
                legacy_selected=(
                    selection["results"]
                ),
                now=NOW,
            )
        )

        self.assertEqual(
            event["candidate_count"],
            2,
        )
        self.assertEqual(
            event["legacy_selected_count"],
            2,
        )

        # Selection never mutated the pool.
        self.assertEqual(
            [c["id"] for c in pool["candidates"]],
            ["stale", "fresh"],
        )

    async def test_retrieve_publishes_the_pool(self):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        await adapter.retrieve(
            "secret-query"
        )

        self.assertEqual(
            [
                item["id"]
                for item in adapter.last_candidates
            ],
            ["stale", "fresh"],
        )
        self.assertEqual(
            adapter.last_pool_vector_scores,
            VECTOR_SCORES,
        )

    async def test_shadow_failure_does_not_change_live_result(
        self,
    ):
        service = build_service(
            MATCHES, PAIRS
        )

        def explode(candidates, **kwargs):
            raise RuntimeError(
                "synthetic shadow failure"
            )

        service.retrieval_shadow_v2.observe = (
            explode
        )

        payload = await service.get_candidates(
            "secret-query"
        )

        self.assertEqual(
            [
                item["id"]
                for item in payload["memories"]
            ],
            ["stale", "fresh"],
        )


class LegacyOutputRegressionTests(
    unittest.IsolatedAsyncioTestCase
):
    """B. legacy output is unchanged by the pool split."""

    async def test_context_relevance_is_unchanged(
        self,
    ):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        result = await adapter.retrieve(
            "secret-query"
        )

        # Exact legacy calibration, frozen on purpose.
        self.assertEqual(
            [
                (
                    item["id"],
                    item["context_relevance"],
                )
                for item in result
            ],
            [
                ("stale", 0.7356),
                ("fresh", 0.9222),
            ],
        )

    async def test_relevance_threshold_still_applies(
        self,
    ):
        adapter = build_old_adapter(
            [
                bucket("good"),
                bucket("low"),
            ],
            [
                ("good", 0.90),
                ("low", 0.40),
            ],
        )

        result = await adapter.retrieve(
            "secret-query"
        )

        self.assertEqual(
            [item["id"] for item in result],
            ["good"],
        )

    async def test_max_results_cap_is_preserved(
        self,
    ):
        adapter = build_old_adapter(
            [
                bucket(f"m{index}")
                for index in range(12)
            ],
            [
                (f"m{index}", 0.9)
                for index in range(12)
            ],
        )

        result = await adapter.retrieve(
            "secret-query",
            max_results=3,
        )

        self.assertEqual(
            [item["id"] for item in result],
            ["m0", "m1", "m2"],
        )


class OldRetrievalUnchangedTests(
    unittest.IsolatedAsyncioTestCase
):
    """1. old retrieval is unchanged."""

    async def test_old_adapter_has_no_shadow_hook(
        self,
    ):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        # The Retrieval v2 shadow no longer lives inside the
        # old adapter: it is a separate pipeline observed at
        # the service layer.
        self.assertFalse(
            hasattr(
                adapter,
                "quality_v2_shadow",
            )
        )

        await adapter.retrieve(
            "secret-query"
        )

        self.assertNotIn(
            "retrieval_quality_v2",
            adapter.last_telemetry,
        )

    async def test_old_result_matches_service_result(
        self,
    ):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        old_result = await adapter.retrieve(
            "secret-query"
        )

        service = build_service(
            MATCHES, PAIRS
        )

        payload = await service.get_candidates(
            "secret-query"
        )

        # The service returns exactly the old retrieval
        # result, in the old order.
        self.assertEqual(
            [
                item["id"]
                for item in payload["memories"]
            ],
            [
                item["id"]
                for item in old_result
            ],
        )

        self.assertEqual(
            [
                item["id"]
                for item in payload["memories"]
            ],
            ["stale", "fresh"],
        )


class ShadowIsolationTests(
    unittest.TestCase
):
    """2. shadow retrieval does not modify output."""

    def test_shadow_reorders_but_output_stays(
        self,
    ):
        shadow = RetrievalQualityShadow()

        event = shadow.observe(
            MATCHES,
            vector_scores=VECTOR_SCORES,
            legacy_selected=MATCHES,
            now=NOW,
        )

        # The shadow does reorder...
        self.assertTrue(
            event["order_changed"]
        )

        reranked = rank_candidates(
            views(),
            context=ShadowScoringContext(
                now=NOW,
                max_semantic_similarity=0.9,
            ),
        )

        self.assertEqual(
            [c.id for c, _s in reranked],
            ["fresh", "stale"],
        )

        # ...but the input list itself is untouched.
        self.assertEqual(
            [item["id"] for item in MATCHES],
            ["stale", "fresh"],
        )

    def test_rank_does_not_mutate_inputs(self):
        snapshot = json.loads(
            json.dumps(MATCHES)
        )

        original = list(MATCHES)

        rank_candidates(
            MATCHES,
            context=ShadowScoringContext(
                now=NOW,
                max_semantic_similarity=0.9,
            ),
        )

        self.assertEqual(
            MATCHES,
            original,
        )
        self.assertEqual(
            json.loads(
                json.dumps(MATCHES)
            ),
            snapshot,
        )

    def test_observe_does_not_mutate_inputs(self):
        original = list(MATCHES)

        RetrievalQualityShadow().observe(
            MATCHES,
            vector_scores=VECTOR_SCORES,
            now=NOW,
        )

        self.assertEqual(MATCHES, original)

    def test_normalize_is_pure(self):
        candidate = normalize_candidate(
            MATCHES[0]
        )

        # The source dict is unchanged.
        self.assertIn(
            "content",
            MATCHES[0],
        )

        # The candidate exposes no storage fields.
        self.assertFalse(
            hasattr(candidate, "content")
        )


class QualityPrivacyTests(
    unittest.TestCase
):
    """3. quality event privacy."""

    def test_event_exposes_privacy_safe_fields(
        self,
    ):
        event = RetrievalQualityShadow().observe(
            MATCHES,
            vector_scores=VECTOR_SCORES,
            now=NOW,
        )

        for field in PRIVACY_SAFE_FIELDS:
            self.assertIn(
                field,
                event,
                field,
            )

    def test_event_never_leaks_identity_or_text(
        self,
    ):
        event = RetrievalQualityShadow().observe(
            MATCHES,
            vector_scores=VECTOR_SCORES,
            now=NOW,
        )

        serialized = json.dumps(
            event,
            ensure_ascii=False,
        )

        for forbidden in (
            "secret-stale",
            "secret-fresh",
            "stale",
            "fresh",
            "content",
            "last_active",
            "conversation_id",
            "query",
        ):
            self.assertNotIn(
                forbidden,
                serialized,
                forbidden,
            )

    def test_event_is_json_serializable(self):
        event = RetrievalQualityShadow().observe(
            MATCHES,
            vector_scores=VECTOR_SCORES,
            now=NOW,
        )

        payload = json.loads(
            json.dumps(event)
        )

        self.assertIs(
            payload["shadow_only"],
            True,
        )


class ScoreDeterminismTests(
    unittest.TestCase
):
    """5. score deterministic."""

    def test_score_is_repeatable(self):
        context = ShadowScoringContext(
            now=NOW,
            max_semantic_similarity=0.9,
        )

        candidate = views()[1]

        first = score_candidate(
            candidate,
            context,
        )

        for _ in range(5):
            self.assertEqual(
                score_candidate(
                    candidate,
                    context,
                ),
                first,
            )

        # A fresh scorer instance produces the same value.
        self.assertEqual(
            ShadowScorer().score(
                candidate,
                context,
            ),
            first,
        )

    def test_malformed_candidate_never_crashes(
        self,
    ):
        context = ShadowScoringContext(
            now=NOW,
            max_semantic_similarity=0.9,
        )

        for bad in (
            None,
            5,
            "x",
            [],
            {"id": 1},
        ):
            value = score_candidate(
                bad,
                context,
            )

            self.assertGreaterEqual(
                value,
                0.0,
                bad,
            )
            self.assertLessEqual(
                value,
                1.0,
                bad,
            )

    def test_relative_semantic_is_not_an_independent_signal(
        self,
    ):
        from ombrebrain.context.retrieval.scorer import (
            relative_semantic_score,
        )

        context = ShadowScoringContext(
            now=NOW,
            max_semantic_similarity=0.9,
        )

        # relative_semantic is purely derived from the raw
        # semantic similarity, so two candidates with the same
        # similarity get the same relative score.
        first = relative_semantic_score(
            normalize_candidate(
                MATCHES[0],
                semantic_similarity=0.45,
            ),
            context,
        )
        second = relative_semantic_score(
            normalize_candidate(
                MATCHES[1],
                semantic_similarity=0.45,
            ),
            context,
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(
            first,
            0.5,
            places=9,
        )

    def test_score_does_not_mutate_candidate(
        self,
    ):
        candidate = views()[1]

        before = candidate.to_dict()

        score_candidate(
            candidate,
            ShadowScoringContext(
                now=NOW,
                max_semantic_similarity=0.9,
            ),
        )

        self.assertEqual(
            candidate.to_dict(),
            before,
        )

    def test_score_is_bounded(self):
        for item in MATCHES:
            value = score_candidate(
                normalize_candidate(item),
                ShadowScoringContext(
                    now=NOW,
                    max_semantic_similarity=0.9,
                ),
            )

            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_weights_default_formula(self):
        self.assertEqual(
            ShadowScoringWeights().to_dict(),
            {
                "semantic": 0.5,
                "recency": 0.2,
                "importance": 0.2,
                "relative_semantic": 0.1,
            },
        )


class RankDeterminismTests(
    unittest.TestCase
):
    """4. rank deterministic and selection-free."""

    def test_rank_is_repeatable(self):
        context = ShadowScoringContext(
            now=NOW,
            max_semantic_similarity=0.9,
        )

        first = rank_candidates(
            views(),
            context=context,
        )

        for _ in range(5):
            self.assertEqual(
                [
                    c.id
                    for c, _s
                    in rank_candidates(
                        views(),
                        context=context,
                    )
                ],
                [c.id for c, _s in first],
            )

    def test_rank_is_order_independent_for_scores(
        self,
    ):
        # The shadow score depends only on candidate
        # content, not on input position.
        context = ShadowScoringContext(
            now=NOW,
            max_semantic_similarity=0.9,
        )

        forward = rank_candidates(
            views(),
            context=context,
        )

        backward = rank_candidates(
            list(reversed(views())),
            context=context,
        )

        self.assertEqual(
            [c.id for c, _s in forward],
            [c.id for c, _s in backward],
        )

    def test_equal_scores_are_stable(self):
        tied = [
            bucket(
                "a",
                relevance=0.7,
                importance=5,
                hours_old=10,
            ),
            bucket(
                "b",
                relevance=0.7,
                importance=5,
                hours_old=10,
            ),
            bucket(
                "c",
                relevance=0.7,
                importance=5,
                hours_old=10,
            ),
        ]

        result = rank_candidates(
            tied,
            context=ShadowScoringContext(
                now=NOW,
                max_semantic_similarity=0.7,
            ),
        )

        self.assertEqual(
            [c.id for c, _s in result],
            ["a", "b", "c"],
        )

    def test_rank_never_truncates(self):
        # There is no top_k here any more: every candidate
        # is ranked and returned.
        result = rank_candidates(
            MATCHES,
            context=ShadowScoringContext(
                now=NOW,
                max_semantic_similarity=0.9,
            ),
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            sorted(c.id for c, _s in result),
            ["fresh", "stale"],
        )


if __name__ == "__main__":
    unittest.main()
