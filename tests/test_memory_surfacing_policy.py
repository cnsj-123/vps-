from __future__ import annotations

import unittest

from ombrebrain.context.memory_surfacing_policy import (
    evaluate_surfacing_policy,
)


CID = "ctx_0123456789abcdef"


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


def memory(
    memory_id="m1",
    content="some cue",
    name=None,
    **extra,
):
    item = {
        "id": memory_id,
        "content": content,
    }

    if name is not None:
        item["metadata"] = {"name": name}

    item.update(extra)

    return item


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


class MemorySurfacingPolicyTests(
    unittest.TestCase
):

    def test_valid_candidate_is_eligible(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified([memory()]),
            confidence_report=(
                allow_confidence()
            ),
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
            report["retrieved_candidate_count"],
            1,
        )
        self.assertEqual(
            report["eligible_candidate_count"],
            1,
        )
        self.assertEqual(
            len(report["eligible"]),
            1,
        )

    def test_missing_memory_id_is_not_surfaced(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                [memory(memory_id=None)]
            ),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reason"],
            "missing_memory_id",
        )
        self.assertEqual(
            report["eligible_candidate_count"],
            0,
        )

    def test_duplicate_memory_id_only_one_eligible(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                [
                    memory(memory_id="dup"),
                    memory(
                        memory_id="dup",
                        content="second",
                    ),
                ]
            ),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["eligible_candidate_count"],
            1,
        )
        self.assertEqual(
            report["ineligible"][0]["reason"],
            "duplicate_memory_id",
        )

    def test_anti_echo_rejected_is_not_surfaced(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                [
                    memory(
                        anti_echo_rejected=True
                    )
                ]
            ),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reason"],
            "anti_echo_rejected",
        )

    def test_dedup_rejected_is_not_surfaced(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                [memory(dedup_rejected=True)]
            ),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["reason"],
            "retrieval_dedup_rejected",
        )

    def test_duplicate_flag_is_not_surfaced(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                [memory(duplicate=True)]
            ),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["reason"],
            "retrieval_dedup_rejected",
        )

    def test_malformed_candidate_is_not_surfaced(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                ["not-a-dict"]
            ),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["reason"],
            "malformed_candidate",
        )

    def test_missing_flashable_cue_is_not_surfaced(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                [
                    memory(
                        content="",
                        name=None,
                    )
                ]
            ),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["reason"],
            "no_flashable_cue",
        )

    def test_order_follows_canonical_retrieval_order(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                [
                    memory(memory_id="a"),
                    memory(memory_id="b"),
                    memory(memory_id="c"),
                ]
            ),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            [
                item["id"]
                for item in report["eligible"]
            ],
            ["a", "b", "c"],
        )

    def test_confidence_missing_is_no_surface(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified([memory()]),
            confidence_report=None,
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
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified([memory()]),
            confidence_report=(
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
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified([memory()]),
            confidence_report=(
                deny_confidence()
            ),
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

    def test_invalid_unified_version(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                [memory()],
                version="not-the-version",
            ),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["reason"],
            "unified_observation_invalid",
        )

    def test_unified_conversation_mismatch(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                [memory()],
                conversation_id=(
                    "ctx_ffffffffffffffff"
                ),
            ),
            confidence_report=(
                allow_confidence()
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
            report = evaluate_surfacing_policy(
                conversation_id=CID,
                unified=unified(
                    [memory()],
                    revision=bad,
                ),
                confidence_report=(
                    allow_confidence()
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
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified(
                [memory()],
                current_user_excluded=False,
            ),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["reason"],
            "current_user_not_excluded",
        )

    def test_malformed_telemetry(
        self,
    ):
        payload = unified([memory()])
        payload["telemetry"] = "nope"

        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=payload,
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["reason"],
            "malformed_telemetry",
        )

    def test_malformed_sections(
        self,
    ):
        payload = unified([memory()])
        payload["sections"] = "nope"

        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=payload,
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["reason"],
            "malformed_telemetry",
        )

    def test_non_list_memories(
        self,
    ):
        payload = unified([memory()])
        payload["sections"] = {
            "memories": "nope",
        }

        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=payload,
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["reason"],
            "malformed_telemetry",
        )

    def test_invalid_conversation_id(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id="",
            unified=unified([memory()]),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["reason"],
            "invalid_conversation_id",
        )

    def test_no_candidates_is_no_surface(
        self,
    ):
        report = evaluate_surfacing_policy(
            conversation_id=CID,
            unified=unified([]),
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            report["decision"],
            "no_surface",
        )
        self.assertEqual(
            report["reason"],
            "no_eligible_candidates",
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
            )

            self.assertEqual(
                report["decision"],
                "no_surface",
            )

    def test_policy_is_read_only_on_candidates(
        self,
    ):
        payload = unified([memory()])
        snapshot = repr(payload)

        evaluate_surfacing_policy(
            conversation_id=CID,
            unified=payload,
            confidence_report=(
                allow_confidence()
            ),
        )

        self.assertEqual(
            repr(payload),
            snapshot,
        )


if __name__ == "__main__":
    unittest.main()