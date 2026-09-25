from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ombrebrain.context.context_confidence_gate import (
    evaluate_context_confidence,
    update_context_confidence_gate,
)


CID = "ctx_0123456789abcdef"

# Canonical privacy-safe field names the report may expose. Anything
# else (text, ids, query) would break the privacy contract.
_ALLOWED_REPORT_FIELDS = {
    "version",
    "mode",
    "decision",
    "allowed",
    "reason",
    "reasons",
    "source_candidate_revision",
    "source_unified_revision",
    "current_user_excluded",
    "semantic_source_stale",
    "trusted_facts_source_stale",
    "retrieval_observation_available",
    "retrieval_candidate_count",
    "usable_context_evidence",
    "has_current_task",
    "state_included",
    "trusted_fact_count",
    "constraint_count",
    "decision_count",
    "open_item_count",
    "plan_count",
    "memory_count",
    "recent_context_count",
    "estimated_tokens",
    "token_budget",
    "stored",
    "duplicate",
    "revision",
    "conversation_id",
}


def candidate(
    **telemetry_overrides,
) -> dict:
    telemetry = {
        "current_user_excluded":
            True,
        "semantic_source_stale":
            False,
        "semantic_source_ahead":
            False,
        "trusted_facts_source_stale":
            False,
        "trusted_facts_source_ahead":
            False,
    }

    telemetry.update(
        telemetry_overrides
    )

    return {
        "version":
            "conversation-context-candidate.v1",
        "conversation_id":
            CID,
        "revision":
            6,
        "source_revision":
            5,
        "telemetry":
            telemetry,
    }


def unified(
    **telemetry_overrides,
) -> dict:
    telemetry = {
        "current_user_excluded":
            True,
        "estimated_tokens":
            120,
        "token_budget":
            1200,
        "has_current_task":
            True,
        "trusted_fact_count":
            1,
        "constraint_count":
            0,
        "decision_count":
            0,
        "open_item_count":
            0,
        "state_included":
            False,
        "plan_count":
            0,
        "memory_count":
            0,
        "recent_context_count":
            0,
        "retrieval_candidate_count":
            0,
        "relevance_rejected":
            0,
        "retrieval_quality": {
            "outcome":
                "ok",
            "embedding_enabled":
                False,
        },
        "anti_echo": {},
        "retrieval_dedup": {},
    }

    telemetry.update(
        telemetry_overrides
    )

    return {
        "version":
            "unified-context-candidate.v1",
        "conversation_id":
            CID,
        "revision":
            3,
        "source_revisions": {
            "conversation_candidate":
                6,
            "conversation_source":
                5,
            "state":
                2,
        },
        "telemetry":
            telemetry,
    }


_UNSET = object()


def _evaluate(
    *,
    conversation_candidate=_UNSET,
    unified_candidate=_UNSET,
) -> dict:
    return evaluate_context_confidence(
        conversation_id=CID,
        conversation_candidate=(
            candidate()
            if conversation_candidate
            is _UNSET
            else conversation_candidate
        ),
        unified=(
            unified()
            if unified_candidate
            is _UNSET
            else unified_candidate
        ),
    )


class ConfidenceGateDecisionTests(
    unittest.TestCase,
):

    def test_valid_evidence_allows_shadow(
        self,
    ):
        report = _evaluate()

        self.assertEqual(
            report["decision"],
            "allow_shadow",
        )

        self.assertIs(
            report["allowed"],
            True,
        )

        self.assertEqual(
            report["reasons"],
            [],
        )

        self.assertIsNone(
            report["reason"]
        )

        self.assertEqual(
            report["mode"],
            "shadow_only",
        )

        self.assertEqual(
            report["version"],
            "context-confidence-gate.v1",
        )

    def test_malformed_candidate_version_is_deny(
        self,
    ):
        report = _evaluate(
            conversation_candidate={
                "version":
                    "other",
            }
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "invalid_conversation_candidate",
            report["reasons"],
        )

    def test_malformed_unified_version_is_deny(
        self,
    ):
        report = _evaluate(
            unified_candidate={
                "version":
                    "other",
            }
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "invalid_unified_candidate",
            report["reasons"],
        )

    def test_non_dict_input_is_deny(
        self,
    ):
        for value in (
            None,
            "not-a-dict",
            [],
            7,
        ):
            with self.subTest(
                value=value
            ):
                report = _evaluate(
                    conversation_candidate=
                        value
                )

                self.assertEqual(
                    report["decision"],
                    "deny_shadow",
                )

                self.assertIn(
                    "invalid_conversation_candidate",
                    report["reasons"],
                )

    def test_conversation_id_mismatch_is_deny(
        self,
    ):
        mismatched = candidate()

        mismatched["conversation_id"] = (
            "ctx_ffffffffffffffff"
        )

        report = _evaluate(
            conversation_candidate=mismatched
        )

        self.assertIn(
            "conversation_candidate_conversation_mismatch",
            report["reasons"],
        )

    def test_stale_semantic_source_is_deny(
        self,
    ):
        report = _evaluate(
            conversation_candidate=candidate(
                semantic_source_stale=True
            )
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "semantic_source_stale",
            report["reasons"],
        )

    def test_stale_trusted_facts_source_is_deny(
        self,
    ):
        report = _evaluate(
            conversation_candidate=candidate(
                trusted_facts_source_stale=True
            )
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "trusted_facts_source_stale",
            report["reasons"],
        )

    def test_unknown_freshness_flag_is_deny(
        self,
    ):
        report = _evaluate(
            conversation_candidate=candidate(
                semantic_source_stale=None
            )
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "semantic_source_stale_unknown",
            report["reasons"],
        )

    def test_current_user_not_excluded_is_deny(
        self,
    ):
        report = _evaluate(
            conversation_candidate=candidate(
                current_user_excluded=False
            )
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "candidate_current_user_not_excluded",
            report["reasons"],
        )

        self.assertIs(
            report["current_user_excluded"],
            False,
        )

    def test_unified_current_user_not_excluded_is_deny(
        self,
    ):
        report = _evaluate(
            unified_candidate=unified(
                current_user_excluded="yes"
            )
        )

        self.assertIn(
            "unified_current_user_not_excluded",
            report["reasons"],
        )

    def test_retrieval_observation_unavailable_is_deny(
        self,
    ):
        report = _evaluate(
            unified_candidate=unified(
                retrieval_quality={}
            )
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "retrieval_observation_unavailable",
            report["reasons"],
        )

        self.assertIs(
            report[
                "retrieval_observation_available"
            ],
            False,
        )

    def test_malformed_telemetry_is_deny(
        self,
    ):
        report = _evaluate(
            unified_candidate=unified(
                plan_count="many",
                memory_count="lots",
            )
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "malformed_telemetry",
            report["reasons"],
        )

        # One reason code per evidence problem.
        self.assertEqual(
            report["reasons"].count(
                "malformed_telemetry"
            ),
            1,
        )

    def test_malformed_candidate_telemetry_is_deny(
        self,
    ):
        broken = candidate()

        broken["telemetry"] = "not-a-dict"

        report = _evaluate(
            conversation_candidate=broken
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "malformed_candidate_telemetry",
            report["reasons"],
        )

    def test_no_usable_context_evidence_is_deny(
        self,
    ):
        report = _evaluate(
            unified_candidate=unified(
                has_current_task=False,
                trusted_fact_count=0,
                constraint_count=0,
                decision_count=0,
                open_item_count=0,
                state_included=False,
                plan_count=0,
                memory_count=0,
                recent_context_count=0,
            )
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "no_usable_context_evidence",
            report["reasons"],
        )

        self.assertIs(
            report["usable_context_evidence"],
            False,
        )

    # --------------------------------------------------
    # empty memory retrieval is NOT on its own a deny
    # --------------------------------------------------

    def test_zero_memory_with_current_task_allows(
        self,
    ):
        report = _evaluate(
            unified_candidate=unified(
                memory_count=0,
                has_current_task=True,
            )
        )

        self.assertEqual(
            report["reasons"],
            [],
        )

        self.assertIs(
            report["allowed"],
            True,
        )

    def test_zero_memory_with_plans_allows(
        self,
    ):
        report = _evaluate(
            unified_candidate=unified(
                has_current_task=False,
                trusted_fact_count=0,
                memory_count=0,
                plan_count=2,
                retrieval_candidate_count=0,
            )
        )

        self.assertEqual(
            report["reasons"],
            [],
        )

        self.assertEqual(
            report["memory_count"],
            0,
        )

        self.assertEqual(
            report["plan_count"],
            2,
        )

    def test_zero_memory_with_persistent_state_allows(
        self,
    ):
        report = _evaluate(
            unified_candidate=unified(
                has_current_task=False,
                trusted_fact_count=0,
                memory_count=0,
                state_included=True,
            )
        )

        self.assertEqual(
            report["reasons"],
            [],
        )

    def test_report_is_privacy_safe(
        self,
    ):
        report = _evaluate()

        self.assertTrue(
            set(report).issubset(
                _ALLOWED_REPORT_FIELDS
            ),
            sorted(set(report)),
        )

    def test_invalid_conversation_id_raises(
        self,
    ):
        for value in (
            "",
            "not-a-conversation-id",
            123,
            None,
        ):
            with self.subTest(
                value=value
            ):
                with self.assertRaises(
                    ValueError
                ):
                    evaluate_context_confidence(
                        conversation_id=value,
                        conversation_candidate=
                            candidate(),
                        unified=unified(),
                    )


class ConfidenceGatePersistenceTests(
    unittest.TestCase,
):

    def setUp(self):
        self._tmp = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(
        self,
        kind,
        payload,
    ) -> None:
        path = (
            self.root
            / kind
            / (CID + ".json")
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def _update(self) -> dict:
        import os

        from unittest.mock import patch

        with patch.dict(
            os.environ,
            {
                "OMBRE_CONTEXT_STATE_DIR":
                    str(self.root),
            },
            clear=False,
        ):
            return update_context_confidence_gate(
                CID
            )

    def test_missing_candidate_is_deny(
        self,
    ):
        report = self._update()

        self.assertIs(
            report["stored"],
            False,
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertEqual(
            report["reason"],
            "conversation_candidate_not_found",
        )

    def test_missing_unified_is_deny(
        self,
    ):
        self._write(
            "context_candidate",
            candidate(),
        )

        report = self._update()

        self.assertIs(
            report["stored"],
            False,
        )

        self.assertEqual(
            report["reason"],
            "unified_candidate_not_found",
        )

    def test_valid_state_persists_allow(
        self,
    ):
        self._write(
            "context_candidate",
            candidate(),
        )

        self._write(
            "unified_context_candidate",
            unified(),
        )

        report = self._update()

        self.assertIs(
            report["stored"],
            True,
        )

        self.assertIs(
            report["duplicate"],
            False,
        )

        self.assertEqual(
            report["revision"],
            1,
        )

        written = json.loads(
            (
                self.root
                / "confidence_gate"
                / (CID + ".json")
            ).read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            written["decision"],
            "allow_shadow",
        )

        self.assertEqual(
            written["conversation_id"],
            CID,
        )

    def test_duplicate_evidence_does_not_bump_revision(
        self,
    ):
        self._write(
            "context_candidate",
            candidate(),
        )

        self._write(
            "unified_context_candidate",
            unified(),
        )

        first = self._update()
        second = self._update()

        self.assertEqual(
            first["revision"],
            1,
        )

        self.assertIs(
            second["duplicate"],
            True,
        )

        self.assertEqual(
            second["revision"],
            1,
        )

    def test_changed_evidence_bumps_revision(
        self,
    ):
        self._write(
            "context_candidate",
            candidate(),
        )

        self._write(
            "unified_context_candidate",
            unified(),
        )

        first = self._update()

        self.assertEqual(
            first["decision"],
            "allow_shadow",
        )

        # Same candidate revision, but the evidence is now stale.
        self._write(
            "context_candidate",
            {
                **candidate(
                    semantic_source_stale=True
                ),
                "revision":
                    6,
            },
        )

        second = self._update()

        self.assertEqual(
            second["decision"],
            "deny_shadow",
        )

        self.assertIs(
            second["duplicate"],
            False,
        )

        self.assertEqual(
            second["revision"],
            2,
        )

        self.assertEqual(
            second["reason"],
            "semantic_source_stale",
        )


if __name__ == "__main__":
    unittest.main()