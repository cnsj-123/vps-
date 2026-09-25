from __future__ import annotations

import json
import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from ombrebrain.context.retrieval import (
    ShadowCandidate,
)
from ombrebrain.context.retrieval.candidate import (
    ShadowCandidate as DirectCandidate,
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
    created_at=None,
    metadata=None,
):
    meta = dict(metadata or {})

    if last_active is not None:
        meta["last_active"] = last_active

    return {
        "id": id,
        "content": "secret-content",
        "context_relevance": relevance,
        "importance": importance,
        "created_at": created_at,
        "metadata": meta,
    }


class NormalizationTests(
    unittest.TestCase
):

    def test_fields_are_normalized(
        self,
    ):
        candidate = (
            ShadowCandidate
            .from_bucket(
                bucket(
                    id="b-1",
                    # Legacy calibrated relevance is a
                    # different concept and must be ignored.
                    relevance=0.123,
                    importance=10,
                    last_active=(
                        NOW.isoformat()
                    ),
                ),
                semantic_similarity=0.9,
            )
        )

        self.assertEqual(
            candidate.id,
            "b-1",
        )
        self.assertEqual(
            candidate.semantic_similarity,
            0.9,
        )
        self.assertEqual(
            candidate.timestamp,
            NOW,
        )

        # importance 10 -> 1.0
        self.assertEqual(
            candidate.importance,
            1.0,
        )

    def test_importance_bounds(self):
        # importance 1..10 normalizes to 0..1 via (n-1)/9.
        cases = {
            1: 0.0,
            5: 4 / 9,
            10: 1.0,
            0: 0.0,
            99: 1.0,
            "7": 2 / 3,
            None: 4 / 9,
        }

        for raw, expected in cases.items():
            candidate = (
                ShadowCandidate
                .from_bucket(
                    bucket(
                        importance=raw
                    )
                )
            )

            self.assertAlmostEqual(
                candidate.importance,
                expected,
                places=6,
                msg=repr(raw),
            )

    def test_timestamp_precedence_and_parsing(
        self,
    ):
        # last_active wins over created_at
        candidate = (
            ShadowCandidate
            .from_bucket(
                bucket(
                    last_active=(
                        "2026-09-23T00:00:00Z"
                    ),
                    created_at=(
                        "2020-01-01T00:00:00Z"
                    ),
                )
            )
        )

        self.assertEqual(
            candidate.timestamp,
            datetime(
                2026,
                9,
                23,
                tzinfo=timezone.utc,
            ),
        )

        # falls back to created_at
        fallback = (
            ShadowCandidate
            .from_bucket(
                bucket(
                    created_at=(
                        "2026-09-20T08:30:00+02:00"
                    ),
                )
            )
        )

        self.assertEqual(
            fallback.timestamp,
            datetime(
                2026,
                9,
                20,
                6,
                30,
                tzinfo=timezone.utc,
            ),
        )

    def test_invalid_timestamps_are_none(self):
        for bad in (
            "not-a-date",
            "",
            12345,
        ):
            candidate = (
                ShadowCandidate
                .from_bucket(
                    bucket(
                        last_active=bad
                    )
                )
            )

            self.assertIsNone(
                candidate.timestamp,
                bad,
            )

    def test_semantic_similarity_is_clamped(self):
        for raw, expected in (
            (1.5, 1.0),
            (-0.3, 0.0),
            ("0.7", 0.7),
            (None, 0.0),
            ("abc", 0.0),
        ):
            candidate = (
                ShadowCandidate
                .from_bucket(
                    bucket(
                        relevance=0.8
                    ),
                    semantic_similarity=raw,
                )
            )

            self.assertEqual(
                candidate.semantic_similarity,
                expected,
                raw,
            )

    def test_legacy_calibrated_relevance_is_ignored(
        self,
    ):
        # The legacy live path exposes a *calibrated*
        # context_relevance. It is not the raw embedding
        # similarity, so the shadow view must never pick it
        # up as ``semantic_similarity``.
        candidate = (
            ShadowCandidate
            .from_bucket(
                bucket(relevance=0.99)
            )
        )

        self.assertEqual(
            candidate.semantic_similarity,
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
            candidate = (
                ShadowCandidate
                .from_bucket(bad)
            )

            self.assertIsInstance(
                candidate,
                ShadowCandidate,
                bad,
            )

    def test_storage_structure_is_not_exposed(
        self,
    ):
        raw = bucket(
            metadata={
                "type": "fact",
                "domain": "work",
                "name": "alpha",
                # Non-whitelisted keys must be dropped.
                "secret_extra": "leak",
                "last_active": (
                    NOW.isoformat()
                ),
            }
        )

        candidate = (
            ShadowCandidate
            .from_bucket(raw)
        )

        self.assertEqual(
            candidate.metadata,
            {
                "type": "fact",
                "domain": "work",
                "name": "alpha",
            },
        )

        # No content / created_at / raw bucket accessors.
        serialized = json.dumps(
            candidate.to_dict()
        )

        self.assertNotIn(
            "secret-content",
            serialized,
        )
        self.assertNotIn(
            "secret_extra",
            serialized,
        )
        self.assertNotIn(
            "last_active",
            serialized,
        )


class SerializationTests(
    unittest.TestCase
):

    def test_round_trip(self):
        original = (
            ShadowCandidate
            .from_bucket(
                bucket(
                    id="rt-1",
                    relevance=0.75,
                    importance=8,
                    last_active=(
                        NOW.isoformat()
                    ),
                    metadata={
                        "type": "fact"
                    },
                ),
                semantic_similarity=0.75,
            )
        )

        restored = (
            ShadowCandidate
            .from_dict(
                original.to_dict()
            )
        )

        self.assertEqual(
            restored.id,
            original.id,
        )
        self.assertEqual(
            restored.semantic_similarity,
            original.semantic_similarity,
        )
        self.assertEqual(
            restored.timestamp,
            original.timestamp,
        )
        self.assertAlmostEqual(
            restored.importance,
            original.importance,
            places=6,
        )
        self.assertEqual(
            restored.metadata,
            original.metadata,
        )

        # Full round-trip: serializing the restored candidate
        # yields the exact same payload.
        self.assertEqual(
            restored.to_dict(),
            original.to_dict(),
        )

    def test_to_dict_is_json_serializable(
        self,
    ):
        candidate = (
            ShadowCandidate
            .from_bucket(
                bucket(
                    last_active=(
                        NOW.isoformat()
                    ),
                )
            )
        )

        payload = json.loads(
            json.dumps(
                candidate.to_dict()
            )
        )

        self.assertEqual(
            set(payload),
            {
                "id",
                "semantic_similarity",
                "timestamp",
                "importance",
                "metadata",
            },
        )

    def test_from_dict_malformed_is_safe(
        self,
    ):
        for bad in (
            None,
            "x",
            7,
            [],
        ):
            candidate = (
                ShadowCandidate
                .from_dict(bad)
            )

            self.assertEqual(
                candidate.id,
                "",
                bad,
            )

    def test_future_timestamp_is_kept(self):
        future = NOW + timedelta(
            days=30
        )

        candidate = (
            ShadowCandidate
            .from_bucket(
                bucket(
                    last_active=(
                        future
                        .isoformat()
                    ),
                )
            )
        )

        self.assertEqual(
            candidate.timestamp,
            future,
        )


class PackageExportTests(
    unittest.TestCase
):

    def test_candidate_is_exported_once(self):
        self.assertIs(
            DirectCandidate,
            ShadowCandidate,
        )


if __name__ == "__main__":
    unittest.main()
