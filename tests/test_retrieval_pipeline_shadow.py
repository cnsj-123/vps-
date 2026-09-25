from __future__ import annotations

import asyncio
import json
import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from ombrebrain.retrieval import (
    RetrievalCandidate,
)

from ombrebrain.context import (
    retrieval as retrieval_package,
)
from ombrebrain.context.retrieval import (
    ContextRetrievalAdapter,
    RetrievalQualityShadow,
)
from ombrebrain.context.retrieval.candidate import (
    project_candidate,
)
from ombrebrain.context.retrieval.quality import (
    PRIVACY_SAFE_FIELDS,
)
from ombrebrain.context.retrieval.reranker import (
    rank_candidates,
)
from ombrebrain.context.retrieval.scorer import (
    score_candidate,
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

REMOVED_DOMAIN_TYPES = (
    "ShadowCandidate",
    "ShadowScorer",
    "ShadowScoringWeights",
    "ShadowScoringContext",
    "ShadowRanker",
)


def bucket(
    id,
    *,
    relevance=0.8,
    importance=5,
    hours_old=None,
):
    """Real Ombre-Brain memory bucket shape."""

    metadata = {
        "importance": importance,
    }

    if hours_old is not None:
        metadata["last_active"] = (
            NOW - timedelta(hours=hours_old)
        ).isoformat()

    return {
        "id": id,
        "content": f"secret-{id}",
        "context_relevance": relevance,
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
    bucket_mgr = FakeBucketManager(
        {"one": matches},
        pairs,
    )

    return ContextRetrievalAdapter(
        bucket_mgr=bucket_mgr,
        embedding_engine=FakeEngine(pairs),
    )


class FakeBucketManager:
    """Async bucket manager that returns a fixed pool."""

    def __init__(self, pools, pairs=None):
        self.pools = pools
        self.embedding_engine = FakeEngine(
            pairs or {}
        )

    async def list_all(self, **kwargs):
        return []

    async def search(self, query, **kwargs):
        return list(
            self.pools.get(query, [])
        )


def build_service(matches, pairs):
    bucket_mgr = FakeBucketManager(
        {"secret-query": matches},
        pairs,
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
        importance=1,
        hours_old=2000,
    ),
    bucket(
        "fresh",
        importance=10,
        hours_old=0,
    ),
]

PAIRS = [
    ("stale", 0.66),
    ("fresh", 0.90),
]

VECTOR_SCORES = dict(PAIRS)


class CandidatePoolSplitTests(
    unittest.IsolatedAsyncioTestCase
):
    """The pool feeds both legacy selection and the shadow."""

    async def test_pool_is_pre_selection(self):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        pool = (
            await adapter.acquire_candidate_pool(
                "one"
            )
        )

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
                "one"
            )
        )

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
        self.assertEqual(
            [c["id"] for c in pool["candidates"]],
            ["stale", "fresh"],
        )

    async def test_observation_is_returned_locally(
        self,
    ):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        results, observation = (
            await adapter.retrieve_with_observation(
                "one"
            )
        )

        self.assertEqual(
            [item["id"] for item in results],
            ["stale", "fresh"],
        )
        self.assertEqual(
            [
                c["id"]
                for c in observation[
                    "candidates"
                ]
            ],
            ["stale", "fresh"],
        )
        self.assertEqual(
            observation["vector_scores"],
            VECTOR_SCORES,
        )
        self.assertEqual(
            observation["telemetry"][
                "included_count"
            ],
            2,
        )

    async def test_no_shared_pool_state_exists(
        self,
    ):
        # The cross-request channel is gone: there is no shared
        # candidate pool field left to race on.
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        self.assertFalse(
            hasattr(adapter, "last_candidates")
        )
        self.assertFalse(
            hasattr(
                adapter,
                "last_pool_vector_scores",
            )
        )

    async def test_retrieve_compat_api_returns_live_only(
        self,
    ):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        results = await adapter.retrieve("one")

        self.assertEqual(
            [item["id"] for item in results],
            ["stale", "fresh"],
        )

    async def test_shadow_failure_does_not_change_live_result(
        self,
    ):
        service = build_service(
            MATCHES, PAIRS
        )

        def explode(items, **kwargs):
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
    """Legacy live output is frozen."""

    async def test_context_relevance_is_unchanged(
        self,
    ):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        result = await adapter.retrieve("one")

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

        result = await adapter.retrieve("one")

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
            "one",
            max_results=3,
        )

        self.assertEqual(
            [item["id"] for item in result],
            ["m0", "m1", "m2"],
        )


class OldRetrievalUnchangedTests(
    unittest.IsolatedAsyncioTestCase
):
    """Old retrieval carries no shadow hook."""

    async def test_old_adapter_has_no_shadow_hook(
        self,
    ):
        adapter = build_old_adapter(
            MATCHES, PAIRS
        )

        self.assertFalse(
            hasattr(
                adapter,
                "quality_v2_shadow",
            )
        )

        await adapter.retrieve("one")

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

        old_result = await adapter.retrieve("one")

        service = build_service(
            MATCHES, PAIRS
        )

        payload = await service.get_candidates(
            "secret-query"
        )

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


class ConcurrencyTests(
    unittest.IsolatedAsyncioTestCase
):
    """P0-2: concurrent requests must never mix pools.

    Two queries are run truly concurrently on one shared adapter.
    Both are parked inside ``bucket_mgr.search`` at the same time
    (controlled by per-query events), then released out of order, so
    the two retrieval pipelines interleave across await points.

    Each shadow observation must be built from exactly its own
    request's pool / vector map / live result.
    """

    async def test_concurrent_queries_do_not_mix_pools(
        self,
    ):
        pool_a = [
            bucket("a1", importance=1),
            bucket("a2", importance=10),
        ]
        pool_b = [
            bucket("b1", importance=2),
            bucket("b2", importance=9),
            bucket("b3", importance=4),
        ]

        pairs_a = {
            "a1": 0.61,
            "a2": 0.92,
        }
        pairs_b = {
            "b1": 0.58,
            "b2": 0.88,
            "b3": 0.71,
        }

        entered = {
            "query-A": asyncio.Event(),
            "query-B": asyncio.Event(),
        }
        release = {
            "query-A": asyncio.Event(),
            "query-B": asyncio.Event(),
        }

        class GatedEngine:
            enabled = True

            async def search_similar_strict(
                self,
                query,
                top_k,
            ):
                pairs = (
                    pairs_a
                    if query == "query-A"
                    else pairs_b
                )
                return list(pairs.items())

        class GatedBucketManager:
            embedding_engine = GatedEngine()

            async def list_all(self, **kwargs):
                return []

            async def search(self, query, **kwargs):
                entered[query].set()

                await release[query].wait()

                return list(
                    pool_a
                    if query == "query-A"
                    else pool_b
                )

        service = ContextService(
            state_service=FakeStateService(),
            bucket_mgr=GatedBucketManager(),
        )

        observed: list[dict] = []

        inner_observe = (
            service.retrieval_shadow_v2
            .observe
        )

        def recording_observe(
            candidates,
            **kwargs,
        ):
            candidate_ids = sorted(
                item.get("id")
                for item in candidates
            )
            score_ids = sorted(
                kwargs.get("vector_scores")
                or {}
            )
            selected_ids = sorted(
                item.get("id")
                for item in
                kwargs.get("legacy_selected")
                or ()
            )

            # Every observation must be internally consistent:
            # the pool and the vector map describe the same set.
            self.assertEqual(
                candidate_ids,
                score_ids,
            )

            observed.append(
                {
                    "candidates":
                        candidate_ids,
                    "scores": score_ids,
                    "legacy": selected_ids,
                }
            )

            return inner_observe(
                candidates,
                **kwargs,
            )

        service.retrieval_shadow_v2.observe = (
            recording_observe
        )

        task_a = asyncio.create_task(
            service.get_candidates("query-A")
        )
        await entered["query-A"].wait()

        task_b = asyncio.create_task(
            service.get_candidates("query-B")
        )
        await entered["query-B"].wait()

        # Both retrievals are now in flight simultaneously.
        # Release B first and let it finish while A is still parked.
        release["query-B"].set()
        result_b = await task_b

        release["query-A"].set()
        result_a = await task_a

        self.assertEqual(
            [
                item["id"]
                for item in result_a["memories"]
            ],
            ["a1", "a2"],
        )
        self.assertEqual(
            [
                item["id"]
                for item in result_b["memories"]
            ],
            ["b1", "b2", "b3"],
        )

        # Exactly two observations, each from one request, with no
        # A/B mixing at all.
        self.assertEqual(len(observed), 2)

        self.assertEqual(
            sorted(
                (
                    tuple(item["candidates"]),
                    tuple(item["scores"]),
                    tuple(item["legacy"]),
                )
                for item in observed
            ),
            sorted(
                [
                    (
                        ("a1", "a2"),
                        ("a1", "a2"),
                        ("a1", "a2"),
                    ),
                    (
                        ("b1", "b2", "b3"),
                        ("b1", "b2", "b3"),
                        ("b1", "b2", "b3"),
                    ),
                ]
            ),
        )

        # And each request's own telemetry reflects its own pool.
        self.assertEqual(
            result_a["telemetry"][
                "retrieval_quality"
            ]["retrieval_quality_v2"][
                "candidate_count"
            ],
            2,
        )
        self.assertEqual(
            result_b["telemetry"][
                "retrieval_quality"
            ]["retrieval_quality_v2"][
                "candidate_count"
            ],
            3,
        )


class ShadowIsolationTests(
    unittest.TestCase
):
    """The shadow never modifies the live inputs."""

    def test_observe_reports_reorder_but_keeps_inputs(
        self,
    ):
        shadow = RetrievalQualityShadow()

        original = list(MATCHES)

        event = shadow.observe(
            MATCHES,
            vector_scores=VECTOR_SCORES,
            legacy_selected=MATCHES,
            now=NOW,
        )

        self.assertTrue(
            event["order_changed"]
        )

        self.assertEqual(MATCHES, original)

    def test_observe_does_not_mutate_inputs(self):
        original = list(MATCHES)

        RetrievalQualityShadow().observe(
            MATCHES,
            vector_scores=VECTOR_SCORES,
            now=NOW,
        )

        self.assertEqual(MATCHES, original)

    def test_rank_does_not_mutate_inputs(self):
        snapshot = json.loads(
            json.dumps(MATCHES)
        )

        original = list(MATCHES)

        rank_candidates(
            [
                project_candidate(
                    item,
                    semantic_similarity=(
                        VECTOR_SCORES.get(
                            item["id"]
                        )
                    ),
                )
                for item in MATCHES
            ],
            now=NOW,
            max_semantic_similarity=0.90,
        )

        self.assertEqual(MATCHES, original)
        self.assertEqual(
            json.loads(
                json.dumps(MATCHES)
            ),
            snapshot,
        )

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

    def test_score_is_deterministic_and_bounded(
        self,
    ):
        views = [
            project_candidate(
                item,
                semantic_similarity=(
                    VECTOR_SCORES.get(
                        item["id"]
                    )
                ),
            )
            for item in MATCHES
        ]

        for item in views:
            first = score_candidate(
                item,
                now=NOW,
                max_semantic_similarity=0.90,
            )

            self.assertEqual(
                first,
                score_candidate(
                    item,
                    now=NOW,
                    max_semantic_similarity=0.90,
                ),
            )
            self.assertGreaterEqual(first, 0.0)
            self.assertLessEqual(first, 1.0)

    def test_rank_never_truncates_or_filters(
        self,
    ):
        views = [
            project_candidate(
                item,
                semantic_similarity=(
                    VECTOR_SCORES.get(
                        item["id"]
                    )
                ),
            )
            for item in MATCHES
        ]

        result = rank_candidates(
            views,
            now=NOW,
            max_semantic_similarity=0.90,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            sorted(
                item.bucket.get("id")
                for item, _score in result
            ),
            ["fresh", "stale"],
        )


class ArchitectureTests(
    unittest.TestCase
):
    """P0-3: no second retrieval domain is exposed."""

    def test_context_retrieval_does_not_export_shadow_domain_types(
        self,
    ):
        for name in REMOVED_DOMAIN_TYPES:
            self.assertFalse(
                hasattr(
                    retrieval_package,
                    name,
                ),
                name,
            )

    def test_shadow_domain_types_are_gone_from_submodules(
        self,
    ):
        import ombrebrain.context.retrieval.candidate as candidate_module
        import ombrebrain.context.retrieval.reranker as reranker_module
        import ombrebrain.context.retrieval.scorer as scorer_module

        for module in (
            candidate_module,
            scorer_module,
            reranker_module,
        ):
            for name in REMOVED_DOMAIN_TYPES:
                self.assertFalse(
                    hasattr(module, name),
                    f"{module.__name__}.{name}",
                )

    def test_package_public_api_is_minimal(
        self,
    ):
        self.assertEqual(
            sorted(retrieval_package.__all__),
            [
                "ContextRetrievalAdapter",
                "RetrievalQualityShadow",
            ],
        )

    def test_project_candidate_returns_canonical_candidate(
        self,
    ):
        candidate = project_candidate(
            bucket("m"),
            semantic_similarity=0.5,
        )

        self.assertIsInstance(
            candidate,
            RetrievalCandidate,
        )
        self.assertIs(
            type(candidate),
            retrieval_package_module_type(),
        )


def retrieval_package_module_type():
    from ombrebrain.retrieval import (
        RetrievalCandidate as Canonical,
    )

    return Canonical


if __name__ == "__main__":
    unittest.main()
