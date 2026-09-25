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
    RetrievalQualityShadow,
)
from ombrebrain.context.retrieval.quality import (
    LOG_TAG,
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

CID = "ctx_0123456789abcdef"


def bucket(
    id,
    *,
    relevance=0.8,
    importance=5,
    hours_old=None,
):
    last_active = (
        (
            NOW - timedelta(hours=hours_old)
        ).isoformat()
        if hours_old is not None
        else None
    )

    metadata = (
        {"last_active": last_active}
        if last_active
        else {}
    )

    return {
        "id": id,
        "content": f"secret-{id}",
        "context_relevance": relevance,
        "importance": importance,
        "metadata": metadata,
    }


class ReportShapeTests(
    unittest.TestCase
):

    def test_report_fields(self):
        report = (
            RetrievalQualityShadow()
            .observe(
                [
                    bucket("a"),
                    bucket("b"),
                ],
                now=NOW,
            )
        )

        self.assertEqual(
            report["version"],
            "retrieval-quality-shadow.v2",
        )
        self.assertEqual(
            report["mode"],
            "shadow_only",
        )
        self.assertEqual(
            report["candidate_count"],
            2,
        )
        self.assertEqual(
            report["ranked_count"],
            2,
        )
        self.assertTrue(
            report["shadow_only"]
        )

        self.assertEqual(
            set(
                report[
                    "score_distribution"
                ]
            ),
            {
                "0.0-0.2",
                "0.2-0.4",
                "0.4-0.6",
                "0.6-0.8",
                "0.8-1.0",
            },
        )

        self.assertIsInstance(
            report["top_score"],
            float,
        )
        self.assertIsInstance(
            report[
                "average_score"
            ],
            float,
        )

    def test_distribution_counts(self):
        # semantic-only scorer for exact bucket control
        from ombrebrain.context.retrieval import (
            ShadowScorer,
            ShadowScoringWeights,
        )

        # semantic_similarity comes from the raw vector score
        # map, not from the legacy calibrated relevance.
        shadow = RetrievalQualityShadow(
            scorer=ShadowScorer(
                ShadowScoringWeights(
                    semantic=1.0,
                    recency=0.0,
                    importance=0.0,
                    relative_semantic=0.0,
                )
            ),
        )

        report = shadow.observe(
            [
                bucket(
                    "a",
                    relevance=0.10,
                ),
                bucket(
                    "b",
                    relevance=0.35,
                ),
                bucket(
                    "c",
                    relevance=0.55,
                ),
                bucket(
                    "d",
                    relevance=0.75,
                ),
                bucket(
                    "e",
                    relevance=0.95,
                ),
            ],
            vector_scores={
                "a": 0.10,
                "b": 0.35,
                "c": 0.55,
                "d": 0.75,
                "e": 0.95,
            },
            now=NOW,
        )

        self.assertEqual(
            report[
                "score_distribution"
            ],
            {
                "0.0-0.2": 1,
                "0.2-0.4": 1,
                "0.4-0.6": 1,
                "0.6-0.8": 1,
                "0.8-1.0": 1,
            },
        )

    def test_empty_candidates(self):
        report = (
            RetrievalQualityShadow()
            .observe(
                [],
                now=NOW,
            )
        )

        self.assertEqual(
            report["candidate_count"],
            0,
        )
        self.assertEqual(
            report["ranked_count"],
            0,
        )
        self.assertIsNone(
            report["top_score"]
        )
        self.assertIsNone(
            report[
                "average_score"
            ]
        )
        self.assertFalse(
            report["order_changed"]
        )

    def test_order_changed_detects_reorder(
        self,
    ):
        # Fresh, important memory ranked below a stale one
        # by the legacy order -> the shadow ranking differs.
        report = (
            RetrievalQualityShadow()
            .observe(
                [
                    bucket(
                        "stale",
                        relevance=0.66,
                        importance=1,
                        hours_old=2000,
                    ),
                    bucket(
                        "fresh",
                        relevance=0.68,
                        importance=10,
                        hours_old=0,
                    ),
                ],
                vector_scores={
                    "stale": 0.66,
                    "fresh": 0.68,
                },
                # Legacy live result, in the legacy order.
                legacy_selected=[
                    bucket("stale"),
                    bucket("fresh"),
                ],
                now=NOW,
            )
        )

        self.assertTrue(
            report["order_changed"]
        )

        delta = report[
            "ranking_delta"
        ]

        self.assertEqual(
            delta["overlap_count"],
            2,
        )
        self.assertEqual(
            delta["reordered_count"],
            2,
        )

        # Ranking-only: no add/drop claim exists any more.
        self.assertNotIn(
            "added_count",
            delta,
        )
        self.assertNotIn(
            "dropped_count",
            delta,
        )

    def test_order_changed_is_false_without_legacy_result(
        self,
    ):
        # With nothing to compare against the shadow makes no
        # ordering claim at all.
        report = (
            RetrievalQualityShadow()
            .observe(
                [
                    bucket(
                        "stale",
                        relevance=0.66,
                        hours_old=2000,
                    ),
                    bucket(
                        "fresh",
                        relevance=0.90,
                        hours_old=0,
                    ),
                ],
                vector_scores={
                    "stale": 0.66,
                    "fresh": 0.90,
                },
                now=NOW,
            )
        )

        self.assertFalse(
            report["order_changed"]
        )
        self.assertEqual(
            report["legacy_selected_count"],
            0,
        )

    def test_no_change_when_order_matches(
        self,
    ):
        report = (
            RetrievalQualityShadow()
            .observe(
                [
                    bucket(
                        "best",
                        relevance=0.9,
                        importance=10,
                        hours_old=0,
                    ),
                    bucket(
                        "second",
                        relevance=0.7,
                        importance=1,
                        hours_old=2000,
                    ),
                ],
                vector_scores={
                    "best": 0.9,
                    "second": 0.7,
                },
                legacy_selected=[
                    bucket("best"),
                    bucket("second"),
                ],
                now=NOW,
            )
        )

        self.assertFalse(
            report["order_changed"]
        )

        self.assertEqual(
            report["ranking_delta"][
                "reordered_count"
            ],
            0,
        )

    def test_observe_failed_report(self):
        report = (
            RetrievalQualityShadow
            .observe_failed()
        )

        self.assertTrue(
            report["shadow_only"]
        )
        self.assertFalse(
            report["order_changed"]
        )
        self.assertEqual(
            report["candidate_count"],
            0,
        )
        self.assertEqual(
            report["reason"],
            "observe_failed",
        )


class PrivacyTests(
    unittest.TestCase
):

    def test_report_never_contains_secrets(
        self,
    ):
        report = (
            RetrievalQualityShadow()
            .observe(
                [
                    bucket(
                        "bucket-alpha",
                        hours_old=3,
                    ),
                    bucket(
                        "bucket-beta",
                        hours_old=100,
                    ),
                ],
                now=NOW,
            )
        )

        serialized = json.dumps(
            report,
            ensure_ascii=False,
        )

        for forbidden in (
            "secret-bucket-alpha",
            "secret-bucket-beta",
            "bucket-alpha",
            "bucket-beta",
            "content",
            "last_active",
            "conversation_id",
            CID,
            "query",
        ):
            self.assertNotIn(
                forbidden,
                serialized,
                forbidden,
            )

    def test_report_is_json_serializable(
        self,
    ):
        report = (
            RetrievalQualityShadow()
            .observe(
                [
                    bucket(
                        "a",
                        hours_old=1,
                    ),
                    bucket(
                        "b",
                        hours_old=50,
                    ),
                ],
                now=NOW,
            )
        )

        # Round-trip must not raise and must keep the fields.
        payload = json.loads(
            json.dumps(
                report,
                ensure_ascii=False,
            )
        )

        self.assertEqual(
            payload["shadow_only"],
            True,
        )
        self.assertEqual(
            payload["candidate_count"],
            2,
        )

    def test_log_line_is_privacy_safe(self):
        shadow = (
            RetrievalQualityShadow()
        )

        report = shadow.observe(
            [
                bucket(
                    "leaky-id",
                    hours_old=2,
                )
            ],
            now=NOW,
        )

        with self.assertLogs(
            "ombre_brain.gateway",
            level="INFO",
        ) as captured:
            shadow.emit(report)

        message = "\n".join(
            record.getMessage()
            for record in (
                captured.records
            )
        )

        self.assertIn(
            LOG_TAG,
            message,
        )

        # The JSON payload is parseable...
        payload = json.loads(
            message.split(
                LOG_TAG,
                1,
            )[1].strip()
        )

        # ...and contains the required fields.
        for key in (
            "candidate_count",
            "ranked_count",
            "legacy_selected_count",
            "score_distribution",
            "top_score",
            "average_score",
            "order_changed",
            "shadow_only",
        ):
            self.assertIn(
                key,
                payload,
            )

        # ...and no secrets.
        self.assertNotIn(
            "leaky-id",
            message,
        )
        self.assertNotIn(
            "secret-leaky-id",
            message,
        )


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


def build_service(
    matches,
    pairs,
):
    """ContextService with a stubbed old-retrieval stack."""

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


class ServiceHookTests(
    unittest.IsolatedAsyncioTestCase
):
    """The shadow runs at the service layer, next to old retrieval.

    The integration point is ContextService.get_candidates():
    old retrieval runs first, the shadow observes its result and
    the returned result is never modified.
    """

    async def test_service_reports_quality_v2(
        self,
    ):
        service = build_service(
            matches=[
                bucket(
                    "stale",
                    relevance=0.66,
                    hours_old=2000,
                ),
                bucket(
                    "fresh",
                    relevance=0.90,
                    hours_old=0,
                ),
            ],
            pairs=[
                ("stale", 0.66),
                ("fresh", 0.90),
            ],
        )

        result = await service.get_candidates(
            "secret-query"
        )

        # Old retrieval result and its order are untouched.
        self.assertEqual(
            [
                item["id"]
                for item in result[
                    "memories"
                ]
            ],
            ["stale", "fresh"],
        )

        event = result["telemetry"][
            "retrieval_quality"
        ][
            "retrieval_quality_v2"
        ]

        self.assertTrue(
            event["shadow_only"]
        )
        self.assertEqual(
            event["candidate_count"],
            2,
        )

    async def test_service_emits_gateway_log(
        self,
    ):
        service = build_service(
            matches=[
                bucket(
                    "fresh",
                    relevance=0.90,
                    hours_old=0,
                )
            ],
            pairs=[("fresh", 0.90)],
        )

        with self.assertLogs(
            "ombre_brain.gateway",
            level="INFO",
        ) as captured:
            await service.get_candidates(
                "secret-query"
            )

        logged = "\n".join(
            record.getMessage()
            for record in captured.records
        )

        self.assertIn(
            LOG_TAG,
            logged,
        )
        self.assertNotIn(
            "secret-query",
            logged,
        )
        self.assertNotIn(
            "secret-fresh",
            logged,
        )

    async def test_empty_query_has_no_shadow_event(
        self,
    ):
        service = build_service(
            matches=[],
            pairs=[],
        )

        result = await service.get_candidates("")

        self.assertEqual(
            result["telemetry"][
                "retrieval_quality"
            ][
                "retrieval_quality_v2"
            ],
            {},
        )

    async def test_shadow_failure_is_fail_open(
        self,
    ):
        service = build_service(
            matches=[
                bucket(
                    "ok",
                    relevance=0.9,
                    hours_old=0,
                )
            ],
            pairs=[("ok", 0.9)],
        )

        def explode(items, **kwargs):
            raise RuntimeError(
                "synthetic shadow failure"
            )

        service.retrieval_shadow_v2.observe = (
            explode
        )

        # Old retrieval still succeeds and returns its result.
        result = await service.get_candidates(
            "q"
        )

        self.assertEqual(
            len(result["memories"]),
            1,
        )

        event = result["telemetry"][
            "retrieval_quality"
        ][
            "retrieval_quality_v2"
        ]

        self.assertTrue(
            event["shadow_only"]
        )
        self.assertEqual(
            event["reason"],
            "observe_failed",
        )


if __name__ == "__main__":
    unittest.main()
