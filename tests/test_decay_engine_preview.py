from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from decay_engine import (
    DecayEngine,
    _auto_resolve_decision,
    _decay_cycle_skip_reason,
)


# Behaviour-preservation + guardrails for the read-only Legacy lifecycle
# preview added to DecayEngine.
#
# The preview is the ONLY Legacy surface the lifecycle comparison shadow
# may use. These tests freeze three things:
#
#   1. the extracted skip / auto-resolve predicates behave exactly like
#      the inline checks they replaced;
#   2. preview_decay_cycle_decision() is side-effect-free: no bucket
#      mutation, no embedding work, no input mutation;
#   3. run_decay_cycle() still skips, auto-resolves, scores and archives
#      in the same order, and still runs the embedding self-heal phase.


def _iso_days_ago(days: float) -> str:
    moment = datetime.now(timezone.utc) - timedelta(
        days=days
    )
    return (
        moment.isoformat()
        .replace("+00:00", "Z")
    )


class FakeBucketManager:
    """Minimal BucketManager stand-in. Records every mutation call."""

    def __init__(self, buckets):
        self._buckets = list(buckets)
        self.updates = []
        self.archives = []
        self.deletes = []
        self.touches = []
        self.list_all_calls = 0

    async def list_all(self, *, include_archive=False):
        self.list_all_calls += 1
        return [
            {
                "id": b["id"],
                "metadata": dict(b.get("metadata", {})),
                "content": b.get("content", ""),
            }
            for b in self._buckets
        ]

    async def update(self, memory_id, **kwargs):
        self.updates.append((memory_id, kwargs))
        return True

    async def archive(self, memory_id):
        self.archives.append(memory_id)
        return True

    async def delete(self, memory_id):
        self.deletes.append(memory_id)
        return True

    async def touch(self, memory_id, **kwargs):
        self.touches.append((memory_id, kwargs))
        return True


class FakeEmbeddingOutbox:
    def __init__(self, queued=3):
        self.running = True
        self._queued = queued
        self.reconcile_calls = 0

    async def reconcile(self, *, buckets, include_archive=False):
        self.reconcile_calls += 1
        return self._queued


def engine(bucket_mgr=None, *, threshold=0.3):
    return DecayEngine(
        {"decay": {"threshold": threshold}},
        bucket_mgr or FakeBucketManager([]),
    )


def dynamic_metadata(**overrides):
    meta = {
        "importance": 5,
        "activation_count": 1,
        "last_active": _iso_days_ago(5),
        "resolved": False,
    }
    meta.update(overrides)
    return meta


class SkipPredicateTests(unittest.TestCase):
    def test_non_decaying_types_are_skipped(self):
        for value in (
            "permanent",
            "feel",
            "i",
            "plan",
            "letter",
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    _decay_cycle_skip_reason(
                        {"type": value}
                    ),
                    "type_" + value,
                )

    def test_pinned_and_protected_are_skipped(self):
        self.assertEqual(
            _decay_cycle_skip_reason({"pinned": True}),
            "pinned",
        )
        self.assertEqual(
            _decay_cycle_skip_reason({"protected": True}),
            "protected",
        )

    def test_anchor_uses_parse_bool_semantics(self):
        for truthy in ("1", "true", "yes", "on", True, 1):
            with self.subTest(truthy=truthy):
                self.assertEqual(
                    _decay_cycle_skip_reason(
                        {"anchor": truthy}
                    ),
                    "anchor",
                )
        for falsy in ("0", "false", "no", "off", False, 0, None):
            with self.subTest(falsy=falsy):
                self.assertIsNone(
                    _decay_cycle_skip_reason(
                        {"anchor": falsy}
                    )
                )

    def test_normal_candidate_is_not_skipped(self):
        self.assertIsNone(
            _decay_cycle_skip_reason(
                dynamic_metadata()
            )
        )

    def test_preview_reports_skip_reason_without_scoring(self):
        report = engine().preview_decay_cycle_decision(
            {"type": "plan", "importance": 1}
        )
        self.assertFalse(report["applicable"])
        self.assertEqual(report["skip_reason"], "type_plan")
        self.assertIsNone(report["score_current"])
        self.assertFalse(
            report["nominal_archive_candidate"]
        )

    def test_preview_malformed_metadata(self):
        report = engine().preview_decay_cycle_decision(
            None
        )
        self.assertFalse(report["applicable"])
        self.assertEqual(
            report["skip_reason"], "malformed_metadata"
        )


class AutoResolveCounterfactualTests(unittest.TestCase):
    def test_a_eligible_is_auto_resolve_candidate(self):
        meta = dynamic_metadata(
            importance=3,
            last_active=_iso_days_ago(40),
        )
        report = engine().preview_decay_cycle_decision(meta)

        self.assertTrue(report["applicable"])
        self.assertTrue(
            report["auto_resolve_candidate"]
        )
        self.assertIsNotNone(report["score_current"])
        self.assertIsNotNone(
            report["score_if_auto_resolve_succeeds"]
        )
        # resolved=True accelerates decay, so the counterfactual score
        # is strictly lower and drives the nominal archive decision.
        self.assertLess(
            report[
                "score_if_auto_resolve_succeeds"
            ],
            report["score_current"],
        )
        self.assertEqual(
            report["nominal_archive_candidate"],
            report[
                "archive_candidate_after_auto_resolve_if_successful"
            ],
        )
        # No write happened: the input is untouched.
        self.assertFalse(meta.get("resolved"))

    def test_b_already_resolved_is_not_a_candidate(self):
        meta = dynamic_metadata(
            importance=3,
            last_active=_iso_days_ago(40),
            resolved=True,
        )
        report = engine().preview_decay_cycle_decision(meta)
        self.assertFalse(
            report["auto_resolve_candidate"]
        )
        self.assertTrue(report["current_resolved"])
        self.assertIsNone(
            report["score_if_auto_resolve_succeeds"]
        )
        self.assertEqual(
            report["nominal_archive_candidate"],
            report["archive_candidate_current"],
        )

    def test_c_importance_too_high_is_not_a_candidate(self):
        report = engine().preview_decay_cycle_decision(
            dynamic_metadata(
                importance=8,
                last_active=_iso_days_ago(40),
            )
        )
        self.assertFalse(
            report["auto_resolve_candidate"]
        )

    def test_d_age_too_recent_is_not_a_candidate(self):
        report = engine().preview_decay_cycle_decision(
            dynamic_metadata(
                importance=3,
                last_active=_iso_days_ago(1),
            )
        )
        self.assertFalse(
            report["auto_resolve_candidate"]
        )

    def test_auto_resolve_decision_helper_matches_live_rules(self):
        eligible, imp, days = _auto_resolve_decision(
            dynamic_metadata(
                importance=2,
                last_active=_iso_days_ago(45),
            )
        )
        self.assertTrue(eligible)
        self.assertEqual(imp, 2)
        self.assertGreater(days, 30)


class PreviewIsSideEffectFreeTests(unittest.TestCase):
    def test_preview_never_calls_bucket_manager(self):
        mgr = FakeBucketManager([])
        eng = engine(mgr)

        eng.preview_decay_cycle_decision(
            dynamic_metadata(
                importance=3,
                last_active=_iso_days_ago(40),
            )
        )

        self.assertEqual(mgr.updates, [])
        self.assertEqual(mgr.archives, [])
        self.assertEqual(mgr.deletes, [])
        self.assertEqual(mgr.touches, [])
        self.assertEqual(mgr.list_all_calls, 0)

    def test_preview_does_not_mutate_input(self):
        meta = dynamic_metadata(
            importance=3,
            last_active=_iso_days_ago(40),
        )
        before = dict(meta)
        engine().preview_decay_cycle_decision(meta)
        self.assertEqual(meta, before)

    def test_preview_never_touches_embeddings(self):
        mgr = FakeBucketManager([])
        mgr.embedding_outbox = FakeEmbeddingOutbox()
        mgr.embedding_engine = None
        engine(mgr).preview_decay_cycle_decision(
            dynamic_metadata()
        )
        self.assertEqual(
            mgr.embedding_outbox.reconcile_calls, 0
        )


class LiveCycleBehaviourPreservedTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_live_cycle_skips_resolves_archives_once(self):
        buckets = [
            {
                "id": "skip-permanent",
                "metadata": {
                    "type": "permanent",
                    "importance": 1,
                },
            },
            {
                "id": "auto-resolve",
                "metadata": {
                    "importance": 2,
                    "last_active": _iso_days_ago(
                        40
                    ),
                    "resolved": False,
                },
            },
            {
                "id": "fresh",
                "metadata": {
                    "importance": 10,
                    "last_active": _iso_days_ago(
                        0
                    ),
                    "resolved": False,
                },
            },
        ]
        mgr = FakeBucketManager(buckets)
        mgr.embedding_outbox = FakeEmbeddingOutbox(
            queued=2
        )
        eng = engine(mgr)

        result = await eng.run_decay_cycle()

        # The permanent bucket is never scored or mutated.
        self.assertNotIn(
            "skip-permanent",
            [m for m, _ in mgr.updates],
        )
        self.assertNotIn(
            "skip-permanent", mgr.archives
        )
        # Exactly the eligible bucket is auto-resolved then archived.
        self.assertEqual(
            [m for m, _ in mgr.updates],
            ["auto-resolve"],
        )
        self.assertEqual(
            mgr.updates[0][1], {"resolved": True}
        )
        self.assertEqual(
            mgr.archives, ["auto-resolve"]
        )
        # A well-scoring fresh bucket is never archived.
        self.assertNotIn("fresh", mgr.archives)
        # Decay never deletes.
        self.assertEqual(mgr.deletes, [])
        # The embedding self-heal phase still runs.
        self.assertEqual(
            mgr.embedding_outbox.reconcile_calls, 1
        )
        self.assertEqual(
            result["backfilled_embeddings"], 2
        )
        self.assertEqual(result["auto_resolved"], 1)
        self.assertEqual(result["archived"], 1)
        self.assertEqual(result["checked"], 2)


if __name__ == "__main__":
    unittest.main()