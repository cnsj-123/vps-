from __future__ import annotations

import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from ombrebrain.context.memory_surfacing_policy import (
    evaluate_surfacing_policy,
)
from ombrebrain.context.retrieval_decision_shadow import (
    build_candidate_evidence,
)


CID = "ctx_0123456789abcdef"

# Deterministic "now" so anti-echo ages are exact, never flaky.
_NOW = datetime(
    2026,
    9,
    26,
    12,
    0,
    tzinfo=timezone.utc,
)

_UNSET = object()


def _ts(hours):
    return (
        _NOW
        - timedelta(hours=hours)
    ).isoformat()


def candidate(
    memory_id="m1",
    content="cue text",
    *,
    age_hours=None,
    name=None,
    **extra,
):
    """A real retrieval / Unified memory candidate shape.

    id / content / context_relevance / metadata only -- never a
    shadow-only flag.
    """

    item = {
        "id": memory_id,
        "content": content,
        "context_relevance": 0.9,
    }

    metadata = {}

    if age_hours is not None:
        metadata["last_active"] = _ts(
            age_hours
        )

    if name is not None:
        metadata["name"] = name

    if metadata:
        item["metadata"] = metadata

    item.update(extra)

    return item


def allow_confidence(
    revision=1,
):
    return {
        "version":
            "context-confidence-gate.v1",
        "mode":
            "shadow_only",
        "decision":
            "allow_shadow",
        "allowed":
            True,
        "reason":
            None,
        "reasons":
            [],
        "stored":
            True,
        "duplicate":
            False,
        "revision":
            revision,
    }


def deny_confidence(
    reason="semantic_source_stale",
    revision=1,
):
    report = allow_confidence(
        revision=revision
    )

    report["decision"] = "deny_shadow"
    report["allowed"] = False
    report["reason"] = reason
    report["reasons"] = [reason]

    return report


def invalid_confidence(
    reason="unified_request_revision_mismatch",
):
    return {
        "version":
            "context-confidence-gate.v1",
        "mode":
            "shadow_only",
        "decision":
            "deny_shadow",
        "allowed":
            False,
        "reason":
            reason,
        "reasons":
            [reason],
        "stored":
            False,
        "duplicate":
            False,
        "revision":
            None,
    }


def unified(
    memories,
    *,
    revision=3,
    conversation_id=CID,
    current_user_excluded=True,
    version="unified-context-candidate.v1",
):
    return {
        "version": version,
        "conversation_id":
            conversation_id,
        "revision": revision,
        "sections": {
            "memories": memories,
        },
        "telemetry": {
            "current_user_excluded":
                current_user_excluded,
        },
    }


def evidence_for(candidates):
    return build_candidate_evidence(
        candidates,
        now=_NOW,
    )


def run_policy(
    *,
    memories,
    retrieval=None,
    confidence=_UNSET,
    evidence=_UNSET,
    unified_payload=None,
    conversation_id=CID,
):
    if evidence is _UNSET:
        evidence = evidence_for(
            retrieval
            if retrieval is not None
            else memories
        )

    if confidence is _UNSET:
        confidence = allow_confidence()

    return evaluate_surfacing_policy(
        conversation_id=conversation_id,
        unified=(
            unified_payload
            if unified_payload is not None
            else unified(memories)
        ),
        confidence_report=confidence,
        shadow_evidence=evidence,
    )


class RealEvidenceTests(
    unittest.TestCase
):
    """Eligibility consumes the real conservative.v1 evidence."""

    def test_old_normal_candidate_is_surfaced(
        self,
    ):
        report = run_policy(
            memories=[
                candidate(
                    "m1",
                    age_hours=100,
                )
            ]
        )

        self.assertEqual(
            report["decision"],
            "allow_shadow",
        )
        self.assertEqual(
            report["reason"],
            "surfacing_eligible",
        )
        self.assertEqual(
            report["eligible_candidate_count"],
            1,
        )

    def test_recent_24h_candidate_is_not_surfaced(
        self,
    ):
        report = run_policy(
            memories=[
                candidate(
                    "m1",
                    age_hours=2,
                )
            ]
        )

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reason"],
            "anti_echo_recent_24h",
        )
        self.assertEqual(
            report["eligible_candidate_count"],
            0,
        )

    def test_recent_24_72h_candidate_is_surfaced(
        self,
    ):
        report = run_policy(
            memories=[
                candidate(
                    "m1",
                    age_hours=48,
                )
            ]
        )

        self.assertEqual(
            report["decision"],
            "allow_shadow",
        )

    def test_missing_last_active_is_surfaced(
        self,
    ):
        report = run_policy(
            memories=[candidate("m1")]
        )

        self.assertEqual(
            report["decision"],
            "allow_shadow",
        )

    def test_evidence_is_request_local_and_ordered(
        self,
    ):
        evidence = evidence_for(
            [
                candidate(
                    "a",
                    age_hours=2,
                ),
                candidate(
                    "b",
                    age_hours=100,
                ),
            ]
        )

        self.assertEqual(
            [
                entry["memory_id"]
                for entry in evidence
            ],
            ["a", "b"],
        )
        self.assertFalse(
            evidence[0]["would_keep"]
        )
        self.assertTrue(
            evidence[1]["would_keep"]
        )

        # Evidence never carries text or a score.
        serialized = repr(evidence)

        self.assertNotIn("cue text", serialized)

    def test_evidence_dedup_reasons_map_through(
        self,
    ):
        # The sidecar vocabulary comes from the shared conservative.v1
        # observer; this locks the reason mapping without inventing a
        # second rule set.
        report = run_policy(
            memories=[
                candidate("m1"),
                candidate("m2"),
            ],
            evidence=[
                {
                    "memory_id": "m1",
                    "index": 0,
                    "would_keep": False,
                    "reason": "duplicate_id",
                },
                {
                    "memory_id": "m2",
                    "index": 1,
                    "would_keep": False,
                    "reason":
                        "exact_text_duplicate",
                },
            ],
        )

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reasons"],
            [
                "retrieval_dedup_duplicate_id",
                "retrieval_dedup_exact_text_duplicate",
            ],
        )

    def test_unknown_evidence_reason_is_conservative(
        self,
    ):
        report = run_policy(
            memories=[candidate("m1")],
            evidence=[
                {
                    "memory_id": "m1",
                    "index": 0,
                    "would_keep": False,
                    "reason": "something_new",
                }
            ],
        )

        self.assertEqual(
            report["reason"],
            "shadow_evidence_rejected",
        )

    def test_missing_evidence_for_candidate(
        self,
    ):
        report = run_policy(
            memories=[candidate("m1")],
            evidence=[],
        )

        self.assertEqual(
            report["reason"],
            "missing_shadow_evidence",
        )

    def test_evidence_unavailable_is_no_surface(
        self,
    ):
        report = run_policy(
            memories=[candidate("m1")],
            evidence=None,
        )

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reason"],
            "shadow_evidence_unavailable",
        )

    def test_fake_candidate_flags_are_ignored(
        self,
    ):
        # The former synthetic contract (anti_echo_rejected /
        # dedup_rejected / duplicate) must have no effect at all: the
        # real evidence decides.
        candidate_item = candidate(
            "m1",
            age_hours=100,
        )

        candidate_item["anti_echo_rejected"] = (
            True
        )
        candidate_item["dedup_rejected"] = True
        candidate_item["duplicate"] = True

        report = run_policy(
            memories=[candidate_item],
            retrieval=[candidate_item],
        )

        self.assertEqual(
            report["decision"],
            "allow_shadow",
        )
        self.assertEqual(
            report["eligible_candidate_count"],
            1,
        )

    def test_fake_contract_is_absent_from_source(
        self,
    ):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ombrebrain"
            / "context"
            / "memory_surfacing_policy.py"
        ).read_text(encoding="utf-8")

        for fake in (
            "anti_echo_rejected",
            "dedup_rejected",
            "candidate.get(\"duplicate\")",
        ):
            with self.subTest(fake=fake):
                self.assertNotIn(fake, source)


class CandidateEligibilityTests(
    unittest.TestCase
):

    def test_missing_memory_id_is_not_surfaced(
        self,
    ):
        report = run_policy(
            memories=[candidate(None)]
        )

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reason"],
            "missing_memory_id",
        )

    def test_duplicate_memory_id_only_one_eligible(
        self,
    ):
        report = run_policy(
            memories=[
                candidate(
                    "dup",
                    "first",
                    age_hours=100,
                ),
                candidate(
                    "dup",
                    "second",
                    age_hours=100,
                ),
            ]
        )

        self.assertEqual(
            report["eligible_candidate_count"],
            1,
        )
        self.assertEqual(
            report["ineligible"][0]["reason"],
            "duplicate_memory_id",
        )

    def test_malformed_candidate_is_not_surfaced(
        self,
    ):
        report = run_policy(
            memories=["not-a-dict"]
        )

        self.assertEqual(
            report["reason"],
            "malformed_candidate",
        )

    def test_missing_flashable_cue_is_not_surfaced(
        self,
    ):
        report = run_policy(
            memories=[
                candidate(
                    "m1",
                    "",
                    age_hours=100,
                )
            ]
        )

        self.assertEqual(
            report["reason"],
            "no_flashable_cue",
        )

    def test_order_follows_canonical_order(
        self,
    ):
        report = run_policy(
            memories=[
                candidate(
                    "a",
                    "cue a",
                    age_hours=100,
                ),
                candidate(
                    "b",
                    "cue b",
                    age_hours=100,
                ),
                candidate(
                    "c",
                    "cue c",
                    age_hours=100,
                ),
            ]
        )

        self.assertEqual(
            [
                item["id"]
                for item in report["eligible"]
            ],
            ["a", "b", "c"],
        )

    def test_no_candidates_is_no_surface(
        self,
    ):
        report = run_policy(memories=[])

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reason"],
            "no_eligible_candidates",
        )


class ConfidenceIntegrationTests(
    unittest.TestCase
):

    def test_confidence_missing_is_no_surface(
        self,
    ):
        report = run_policy(
            memories=[candidate("m1")],
            confidence=None,
        )

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reason"],
            "confidence_observation_missing",
        )

    def test_structural_invalid_confidence_is_rejected(
        self,
    ):
        report = run_policy(
            memories=[candidate("m1")],
            confidence=(
                invalid_confidence()
            ),
        )

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reason"],
            "confidence_observation_invalid",
        )

    def test_persisted_deny_does_not_globally_block(
        self,
    ):
        # Confidence is only a signal: a normally persisted
        # deny_shadow must NOT block an otherwise eligible candidate.
        report = run_policy(
            memories=[candidate("m1")],
            confidence=deny_confidence(),
        )

        self.assertEqual(
            report["decision"],
            "allow_shadow",
        )
        self.assertEqual(
            report["confidence_decision"],
            "deny_shadow",
        )
        self.assertEqual(
            report["confidence_reason"],
            "semantic_source_stale",
        )
        self.assertEqual(
            len(report["eligible"]),
            1,
        )


class UnifiedValidationTests(
    unittest.TestCase
):

    def test_invalid_unified_version(
        self,
    ):
        report = run_policy(
            memories=[candidate("m1")],
            unified_payload=unified(
                [candidate("m1")],
                version="not-the-version",
            ),
        )

        self.assertEqual(
            report["reason"],
            "unified_observation_invalid",
        )

    def test_unified_conversation_mismatch(
        self,
    ):
        report = run_policy(
            memories=[candidate("m1")],
            unified_payload=unified(
                [candidate("m1")],
                conversation_id=(
                    "ctx_ffffffffffffffff"
                ),
            ),
        )

        self.assertEqual(
            report["reason"],
            "unified_conversation_mismatch",
        )

    def test_invalid_unified_revision(
        self,
    ):
        for bad in (
            None,
            True,
            0,
            -1,
            "3",
        ):
            report = run_policy(
                memories=[candidate("m1")],
                unified_payload=unified(
                    [candidate("m1")],
                    revision=bad,
                ),
            )

            self.assertEqual(
                report["reason"],
                "invalid_unified_revision",
                bad,
            )

    def test_current_user_not_excluded(
        self,
    ):
        report = run_policy(
            memories=[candidate("m1")],
            unified_payload=unified(
                [candidate("m1")],
                current_user_excluded=False,
            ),
        )

        self.assertEqual(
            report["reason"],
            "current_user_not_excluded",
        )

    def test_malformed_telemetry(
        self,
    ):
        payload = unified([candidate("m1")])
        payload["telemetry"] = "nope"

        report = run_policy(
            memories=[candidate("m1")],
            unified_payload=payload,
        )

        self.assertEqual(
            report["reason"],
            "malformed_telemetry",
        )

    def test_malformed_sections(
        self,
    ):
        payload = unified([candidate("m1")])
        payload["sections"] = "nope"

        report = run_policy(
            memories=[candidate("m1")],
            unified_payload=payload,
        )

        self.assertEqual(
            report["reason"],
            "malformed_telemetry",
        )

    def test_non_list_memories(
        self,
    ):
        payload = unified([candidate("m1")])
        payload["sections"] = {
            "memories": "nope",
        }

        report = run_policy(
            memories=[candidate("m1")],
            unified_payload=payload,
        )

        self.assertEqual(
            report["reason"],
            "malformed_telemetry",
        )

    def test_invalid_conversation_id(
        self,
    ):
        report = run_policy(
            memories=[candidate("m1")],
            conversation_id="",
        )

        self.assertEqual(
            report["reason"],
            "invalid_conversation_id",
        )

    def test_policy_never_raises_on_garbage(
        self,
    ):
        for value in (
            None,
            123,
            "x",
            [],
        ):
            report = evaluate_surfacing_policy(
                conversation_id=CID,
                unified=value,
                confidence_report=value,
                shadow_evidence=value,
            )

            self.assertEqual(
                report["decision"],
                "no_surface",
            )

    def test_policy_is_read_only_on_candidates(
        self,
    ):
        payload = unified([candidate("m1")])
        snapshot = repr(payload)

        run_policy(
            memories=[candidate("m1")],
            unified_payload=payload,
        )

        self.assertEqual(
            repr(payload),
            snapshot,
        )


if __name__ == "__main__":
    unittest.main()