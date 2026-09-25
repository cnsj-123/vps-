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
    RetrievalScorer,
    RetrievalScorerWeights,
    RetrievalScoringContext,
    normalize_candidate,
    rerank_candidates,
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

    def test_shadow_would_reorder_but_output_stays(
        self,
    ):
        shadow = RetrievalQualityShadow()

        event = shadow.observe(
            MATCHES,
            now=NOW,
        )

        # The shadow does reorder...
        self.assertTrue(
            event["would_change"]
        )

        reranked = shadow.reranked(MATCHES)

        self.assertEqual(
            [c.id for c in reranked],
            ["fresh", "stale"],
        )

        # ...but the input list itself is untouched.
        self.assertEqual(
            [item["id"] for item in MATCHES],
            ["stale", "fresh"],
        )

    def test_rerank_does_not_mutate_inputs(self):
        snapshot = json.loads(
            json.dumps(MATCHES)
        )

        original = list(MATCHES)

        rerank_candidates(
            MATCHES,
            context=RetrievalScoringContext(
                now=NOW,
                max_semantic_score=0.9,
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
        context = RetrievalScoringContext(
            now=NOW,
            max_semantic_score=0.9,
        )

        candidate = normalize_candidate(
            MATCHES[1]
        )

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
            RetrievalScorer().score(
                candidate,
                context,
            ),
            first,
        )

    def test_score_does_not_mutate_candidate(
        self,
    ):
        candidate = normalize_candidate(
            MATCHES[1]
        )

        before = candidate.to_dict()

        score_candidate(
            candidate,
            RetrievalScoringContext(
                now=NOW,
                max_semantic_score=0.9,
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
                RetrievalScoringContext(
                    now=NOW,
                    max_semantic_score=0.9,
                ),
            )

            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_weights_default_formula(self):
        self.assertEqual(
            RetrievalScorerWeights().to_dict(),
            {
                "semantic": 0.5,
                "recency": 0.2,
                "importance": 0.2,
                "context_match": 0.1,
            },
        )


class RerankDeterminismTests(
    unittest.TestCase
):
    """4. rerank deterministic."""

    def test_rerank_is_repeatable(self):
        context = RetrievalScoringContext(
            now=NOW,
            max_semantic_score=0.9,
        )

        first = rerank_candidates(
            MATCHES,
            context=context,
        )

        for _ in range(5):
            self.assertEqual(
                [
                    c.id
                    for c in rerank_candidates(
                        MATCHES,
                        context=context,
                    )
                ],
                [c.id for c in first],
            )

    def test_rerank_is_order_independent_for_scores(
        self,
    ):
        # The shadow score depends only on candidate
        # content, not on input position.
        context = RetrievalScoringContext(
            now=NOW,
            max_semantic_score=0.9,
        )

        forward = rerank_candidates(
            MATCHES,
            context=context,
        )

        backward = rerank_candidates(
            list(reversed(MATCHES)),
            context=context,
        )

        self.assertEqual(
            [c.id for c in forward],
            [c.id for c in backward],
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

        result = rerank_candidates(
            tied,
            context=RetrievalScoringContext(
                now=NOW,
                max_semantic_score=0.7,
            ),
        )

        self.assertEqual(
            [c.id for c in result],
            ["a", "b", "c"],
        )

    def test_top_k_truncates_after_sort(self):
        result = rerank_candidates(
            MATCHES,
            top_k=1,
            context=RetrievalScoringContext(
                now=NOW,
                max_semantic_score=0.9,
            ),
        )

        self.assertEqual(
            [c.id for c in result],
            ["fresh"],
        )


if __name__ == "__main__":
    unittest.main()
