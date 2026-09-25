from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timezone,
)

from ombrebrain.retrieval import (
    RetrievalCandidate,
    RetrievalFeatures,
)

from ombrebrain.context.retrieval.candidate import (
    bucket_metadata,
    parse_timestamp,
    project_candidate,
    read_active_timestamp,
    read_importance,
)
from ombrebrain.context.retrieval.scorer import (
    importance_score,
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
    id="m1",
    relevance=0.8,
    importance=5,
    last_active=None,
    created=None,
    metadata=None,
    **extra,
):
    """Real Ombre-Brain memory bucket shape.

    ``importance`` / ``last_active`` / ``created`` live inside
    ``metadata``; ``context_relevance`` is the legacy live-path field
    added by the selector, not a raw OB bucket field.
    """

    meta = dict(metadata or {})

    if importance is not None:
        meta["importance"] = importance

    if last_active is not None:
        meta["last_active"] = last_active

    if created is not None:
        meta["created"] = created

    raw = {
        "id": id,
        "content": "secret-content",
        "context_relevance": relevance,
        "metadata": meta,
    }

    raw.update(extra)

    return raw


class ProjectionTests(
    unittest.TestCase
):

    def test_projects_into_canonical_domain(
        self,
    ):
        raw = bucket(
            id="b-1",
            last_active=NOW.isoformat(),
        )

        candidate = project_candidate(
            raw,
            semantic_similarity=0.9,
        )

        self.assertIsInstance(
            candidate,
            RetrievalCandidate,
        )
        self.assertIsInstance(
            candidate.features,
            RetrievalFeatures,
        )
        self.assertEqual(
            candidate.features.semantic_similarity,
            0.9,
        )
        self.assertEqual(
            candidate.source,
            "context_retrieval_v2_shadow",
        )

    def test_bucket_is_held_not_copied(
        self,
    ):
        raw = bucket()

        candidate = project_candidate(
            raw,
            semantic_similarity=0.5,
        )

        # The bucket is referenced, not copied: no content is
        # extracted at this internal boundary.
        self.assertIs(
            candidate.bucket,
            raw,
        )

    def test_semantic_similarity_is_clamped(
        self,
    ):
        for raw_value, expected in (
            (1.5, 1.0),
            (-0.3, 0.0),
            ("0.7", 0.7),
            (None, 0.0),
            ("abc", 0.0),
        ):
            candidate = project_candidate(
                bucket(),
                semantic_similarity=raw_value,
            )

            self.assertEqual(
                candidate.features.semantic_similarity,
                expected,
                raw_value,
            )

    def test_legacy_calibrated_relevance_is_ignored(
        self,
    ):
        # context_relevance is the legacy calibrated value, not the
        # raw embedding similarity, so it must never become
        # semantic_similarity.
        candidate = project_candidate(
            bucket(relevance=0.99)
        )

        self.assertEqual(
            candidate.features.semantic_similarity,
            0.0,
        )

    def test_malformed_input_never_raises(
        self,
    ):
        for bad in (
            None,
            5,
            "x",
            [],
            {"id": 123},
        ):
            candidate = project_candidate(bad)

            self.assertIsInstance(
                candidate,
                RetrievalCandidate,
                bad,
            )
            self.assertEqual(
                candidate.features.semantic_similarity,
                0.0,
                bad,
            )


class RealBucketFieldTests(
    unittest.TestCase
):
    """P0-1: the real OB bucket shape is the only shape read."""

    def test_reads_importance_from_real_bucket_metadata(
        self,
    ):
        candidate = project_candidate(
            bucket(importance=10)
        )

        self.assertEqual(
            read_importance(candidate),
            10,
        )
        # 10 -> 1.0
        self.assertEqual(
            importance_score(candidate),
            1.0,
        )

    def test_importance_bounds(self):
        cases = {
            1: 0.0,
            5: 4 / 9,
            10: 1.0,
            0: 0.0,
            99: 1.0,
            "7": 2 / 3,
        }

        for raw_value, expected in (
            cases.items()
        ):
            candidate = project_candidate(
                bucket(importance=raw_value)
            )

            self.assertAlmostEqual(
                importance_score(candidate),
                expected,
                places=6,
                msg=repr(raw_value),
            )

    def test_missing_importance_uses_default_5(
        self,
    ):
        candidate = project_candidate(
            bucket(importance=None)
        )

        self.assertIsNone(
            read_importance(candidate)
        )
        # Default 5 -> 4/9
        self.assertAlmostEqual(
            importance_score(candidate),
            4 / 9,
            places=6,
        )

    def test_top_level_importance_does_not_override_metadata_importance(
        self,
    ):
        # Real OB metadata is the source of truth.
        raw = {
            "id": "m",
            "content": "secret-content",
            "importance": 10,
            "metadata": {
                "importance": 2,
            },
        }

        candidate = project_candidate(raw)

        self.assertEqual(
            read_importance(candidate),
            2,
        )
        self.assertAlmostEqual(
            importance_score(candidate),
            (2 - 1) / 9,
            places=9,
        )

    def test_last_active_has_priority_over_created(
        self,
    ):
        candidate = project_candidate(
            bucket(
                last_active=(
                    "2026-09-23T00:00:00Z"
                ),
                created=(
                    "2020-01-01T00:00:00Z"
                ),
            )
        )

        self.assertEqual(
            read_active_timestamp(candidate),
            datetime(
                2026,
                9,
                23,
                tzinfo=timezone.utc,
            ),
        )

    def test_created_fallback_uses_metadata_created(
        self,
    ):
        # No last_active at all.
        candidate = project_candidate(
            bucket(
                created=(
                    "2026-09-20T08:30:00+02:00"
                ),
            )
        )

        self.assertEqual(
            read_active_timestamp(candidate),
            datetime(
                2026,
                9,
                20,
                6,
                30,
                tzinfo=timezone.utc,
            ),
        )

    def test_invalid_last_active_falls_back_to_created(
        self,
    ):
        for bad in (
            "not-a-date",
            "",
            12345,
        ):
            candidate = project_candidate(
                bucket(
                    last_active=bad,
                    created=(
                        "2026-09-20T00:00:00Z"
                    ),
                )
            )

            self.assertEqual(
                read_active_timestamp(
                    candidate
                ),
                datetime(
                    2026,
                    9,
                    20,
                    tzinfo=timezone.utc,
                ),
                bad,
            )

    def test_top_level_created_at_is_not_used(
        self,
    ):
        candidate = project_candidate(
            {
                "id": "m",
                "content": "secret-content",
                "created_at": (
                    "2020-01-01T00:00:00Z"
                ),
                "metadata": {
                    "created":
                        "2026-09-20T00:00:00Z",
                },
            }
        )

        self.assertEqual(
            read_active_timestamp(candidate),
            datetime(
                2026,
                9,
                20,
                tzinfo=timezone.utc,
            ),
        )

    def test_missing_timestamps_are_none(
        self,
    ):
        candidate = project_candidate(
            bucket()
        )

        self.assertIsNone(
            read_active_timestamp(candidate)
        )


class MetadataReaderTests(
    unittest.TestCase
):

    def test_bucket_metadata_is_total(self):
        for bad in (
            None,
            5,
            "x",
            [],
            {"id": "m"},
            {"metadata": "nope"},
        ):
            self.assertEqual(
                bucket_metadata(bad),
                {},
                bad,
            )

    def test_bucket_metadata_accepts_candidate(
        self,
    ):
        raw = bucket(importance=7)

        self.assertEqual(
            bucket_metadata(
                project_candidate(raw)
            ),
            raw["metadata"],
        )

    def test_parse_timestamp_totality(self):
        self.assertIsNone(
            parse_timestamp(None)
        )
        self.assertIsNone(
            parse_timestamp("")
        )
        self.assertIsNone(
            parse_timestamp("nope")
        )

        self.assertEqual(
            parse_timestamp(
                "2026-09-20T00:00:00Z"
            ),
            datetime(
                2026,
                9,
                20,
                tzinfo=timezone.utc,
            ),
        )

        # Naive datetimes are treated as UTC.
        self.assertEqual(
            parse_timestamp(
                "2026-09-20T00:00:00"
            ),
            datetime(
                2026,
                9,
                20,
                tzinfo=timezone.utc,
            ),
        )


if __name__ == "__main__":
    unittest.main()
