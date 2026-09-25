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

# The real legacy retrieval outcome contract. Every one of these is
# a valid observation and must never be denied for its name alone.
_KNOWN_OUTCOMES = (
    "empty_query",
    "included",
    "no_search_matches",
    "embedding_disabled",
    "no_semantic_scores",
    "below_relevance_threshold",
    "no_included_results",
)

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
                "included",
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


def _unified_without(*fields) -> dict:
    """Unified fixture with the given telemetry fields removed."""

    value = unified()

    for field in fields:
        value["telemetry"].pop(
            field,
            None,
        )

    return value


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

    # --------------------------------------------------
    # revision legality (shared freshness validator)
    # --------------------------------------------------

    def test_matching_revision_chain_allows_shadow(
        self,
    ):
        report = _evaluate()

        self.assertEqual(
            report["decision"],
            "allow_shadow",
        )

        self.assertEqual(
            report["reasons"],
            [],
        )

        self.assertEqual(
            report["source_candidate_revision"],
            6,
        )

        self.assertEqual(
            report["source_unified_revision"],
            3,
        )

    def test_candidate_revision_missing_is_deny(
        self,
    ):
        broken = candidate()

        del broken["revision"]

        report = _evaluate(
            conversation_candidate=broken
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "invalid_candidate_revision",
            report["reasons"],
        )

    def test_candidate_revision_malformed_is_deny(
        self,
    ):
        for value in (
            None,
            True,
            False,
            0,
            -1,
            "6",
            6.0,
            [],
            {},
        ):
            with self.subTest(
                revision=value
            ):
                broken = candidate()

                broken["revision"] = value

                report = _evaluate(
                    conversation_candidate=
                        broken
                )

                self.assertEqual(
                    report["decision"],
                    "deny_shadow",
                )

                self.assertIn(
                    "invalid_candidate_revision",
                    report["reasons"],
                )

    def test_candidate_source_revision_malformed_is_deny(
        self,
    ):
        for value in (
            None,
            True,
            0,
            -2,
            "5",
        ):
            with self.subTest(
                source_revision=value
            ):
                broken = candidate()

                broken[
                    "source_revision"
                ] = value

                report = _evaluate(
                    conversation_candidate=
                        broken
                )

                self.assertEqual(
                    report["decision"],
                    "deny_shadow",
                )

                self.assertIn(
                    "invalid_candidate_source_revision",
                    report["reasons"],
                )

    def test_unified_revision_missing_is_deny(
        self,
    ):
        broken = unified()

        del broken["revision"]

        report = _evaluate(
            unified_candidate=broken
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "invalid_unified_revision",
            report["reasons"],
        )

    def test_unified_revision_malformed_is_deny(
        self,
    ):
        for value in (
            None,
            True,
            0,
            -1,
            "3",
        ):
            with self.subTest(
                revision=value
            ):
                broken = unified()

                broken["revision"] = value

                report = _evaluate(
                    unified_candidate=broken
                )

                self.assertEqual(
                    report["decision"],
                    "deny_shadow",
                )

                self.assertIn(
                    "invalid_unified_revision",
                    report["reasons"],
                )

    # --------------------------------------------------
    # Candidate -> Unified source chain
    # --------------------------------------------------

    def test_unified_source_revisions_missing_is_deny(
        self,
    ):
        broken = unified()

        del broken["source_revisions"]

        report = _evaluate(
            unified_candidate=broken
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "invalid_unified_source_revisions",
            report["reasons"],
        )

    def test_unified_source_revisions_non_dict_is_deny(
        self,
    ):
        for value in (
            None,
            ["6", "5"],
            "6",
            8,
        ):
            with self.subTest(
                source_revisions=value
            ):
                broken = unified()

                broken[
                    "source_revisions"
                ] = value

                report = _evaluate(
                    unified_candidate=broken
                )

                self.assertEqual(
                    report["decision"],
                    "deny_shadow",
                )

                self.assertIn(
                    "invalid_unified_source_revisions",
                    report["reasons"],
                )

    def test_unified_source_candidate_entry_malformed_is_deny(
        self,
    ):
        for value in (
            None,
            True,
            0,
            "6",
        ):
            with self.subTest(
                value=value
            ):
                broken = unified()

                broken["source_revisions"] = {
                    "conversation_candidate":
                        value,
                    "conversation_source":
                        5,
                }

                report = _evaluate(
                    unified_candidate=broken
                )

                self.assertIn(
                    "invalid_unified_source_revisions",
                    report["reasons"],
                )

    def test_unified_source_candidate_revision_mismatch_is_deny(
        self,
    ):
        broken = unified()

        broken["source_revisions"] = {
            "conversation_candidate":
                7,
            "conversation_source":
                5,
        }

        report = _evaluate(
            unified_candidate=broken
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "unified_source_candidate_revision_mismatch",
            report["reasons"],
        )

    def test_unified_source_conversation_revision_mismatch_is_deny(
        self,
    ):
        broken = unified()

        broken["source_revisions"] = {
            "conversation_candidate":
                6,
            "conversation_source":
                9,
        }

        report = _evaluate(
            unified_candidate=broken
        )

        self.assertEqual(
            report["decision"],
            "deny_shadow",
        )

        self.assertIn(
            "unified_source_conversation_revision_mismatch",
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

    # --------------------------------------------------
    # strict boolean telemetry
    # --------------------------------------------------

    def test_boolean_telemetry_must_be_real_bool(
        self,
    ):
        cases = (
            ("has_current_task", "yes"),
            ("has_current_task", "true"),
            ("has_current_task", 1),
            ("has_current_task", 0),
            ("has_current_task", []),
            ("has_current_task", {}),
            ("has_current_task", None),
            ("state_included", "true"),
            ("state_included", "yes"),
            ("state_included", 1),
            ("state_included", 0),
            ("state_included", []),
            ("state_included", {}),
            ("state_included", None),
        )

        for field, value in cases:
            with self.subTest(
                field=field,
                value=value,
            ):
                report = _evaluate(
                    unified_candidate=unified(
                        **{field: value}
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

    def test_malformed_boolean_not_masked_by_other_evidence(
        self,
    ):
        # trusted_fact_count=1 must not hide malformed telemetry.
        report = _evaluate(
            unified_candidate=unified(
                has_current_task="yes",
                trusted_fact_count=1,
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

        self.assertIs(
            report["usable_context_evidence"],
            True,
        )

    # --------------------------------------------------
    # retrieval outcome contract
    # --------------------------------------------------

    def test_every_known_retrieval_outcome_allows(
        self,
    ):
        for outcome in _KNOWN_OUTCOMES:
            with self.subTest(
                outcome=outcome
            ):
                report = _evaluate(
                    unified_candidate=unified(
                        retrieval_quality={
                            "outcome":
                                outcome,
                        }
                    )
                )

                self.assertEqual(
                    report["decision"],
                    "allow_shadow",
                )

                self.assertEqual(
                    report["reasons"],
                    [],
                )

                self.assertIs(
                    report[
                        "retrieval_observation_available"
                    ],
                    True,
                )

    def test_unknown_retrieval_outcome_is_deny(
        self,
    ):
        for outcome in (
            "ok",
            "whatever",
            "INCLUDED",
        ):
            with self.subTest(
                outcome=outcome
            ):
                report = _evaluate(
                    unified_candidate=unified(
                        retrieval_quality={
                            "outcome":
                                outcome,
                        }
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

                self.assertIs(
                    report[
                        "retrieval_observation_available"
                    ],
                    False,
                )

    def test_empty_retrieval_outcome_is_deny(
        self,
    ):
        for outcome in ("", "   "):
            with self.subTest(
                outcome=repr(outcome)
            ):
                report = _evaluate(
                    unified_candidate=unified(
                        retrieval_quality={
                            "outcome":
                                outcome,
                        }
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

    def test_non_string_retrieval_outcome_is_deny(
        self,
    ):
        for outcome in (1, 0, [], {}):
            with self.subTest(
                outcome=outcome
            ):
                report = _evaluate(
                    unified_candidate=unified(
                        retrieval_quality={
                            "outcome":
                                outcome,
                        }
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

                self.assertIs(
                    report[
                        "retrieval_observation_available"
                    ],
                    False,
                )

    def test_missing_retrieval_outcome_is_unavailable(
        self,
    ):
        for quality in (
            {
                "embedding_enabled":
                    False,
            },
            {
                "outcome":
                    None,
            },
        ):
            with self.subTest(
                quality=quality
            ):
                report = _evaluate(
                    unified_candidate=unified(
                        retrieval_quality=quality
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

    # --------------------------------------------------
    # telemetry completeness (stable Unified contract)
    # --------------------------------------------------

    def test_missing_evidence_count_field_is_deny(
        self,
    ):
        for field in (
            "trusted_fact_count",
            "constraint_count",
            "decision_count",
            "open_item_count",
            "plan_count",
            "memory_count",
            "recent_context_count",
        ):
            with self.subTest(
                field=field
            ):
                report = _evaluate(
                    unified_candidate=(
                        _unified_without(
                            field
                        )
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

    def test_missing_evidence_bool_field_is_deny(
        self,
    ):
        for field in (
            "has_current_task",
            "state_included",
        ):
            with self.subTest(
                field=field
            ):
                report = _evaluate(
                    unified_candidate=(
                        _unified_without(
                            field
                        )
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

                # Not guessed as False.
                self.assertIsNone(
                    report[field]
                )

    def test_missing_numeric_field_is_deny(
        self,
    ):
        for field in (
            "retrieval_candidate_count",
            "estimated_tokens",
            "token_budget",
        ):
            with self.subTest(
                field=field
            ):
                report = _evaluate(
                    unified_candidate=(
                        _unified_without(
                            field
                        )
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

                self.assertIsNone(
                    report[field]
                )

    def test_malformed_numeric_field_is_deny(
        self,
    ):
        cases = (
            ("retrieval_candidate_count", "3"),
            ("retrieval_candidate_count", True),
            ("retrieval_candidate_count", -1),
            ("retrieval_candidate_count", []),
            ("estimated_tokens", "80"),
            ("estimated_tokens", False),
            ("estimated_tokens", -1),
            ("estimated_tokens", {}),
            ("token_budget", 0),
            ("token_budget", True),
            ("token_budget", -5),
            ("token_budget", "1200"),
        )

        for field, value in cases:
            with self.subTest(
                field=field,
                value=value,
            ):
                report = _evaluate(
                    unified_candidate=unified(
                        **{field: value}
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

                self.assertIsNone(
                    report[field]
                )

    def test_missing_field_not_masked_by_other_evidence(
        self,
    ):
        # trusted_fact_count=1 must not mask a missing plan_count.
        report = _evaluate(
            unified_candidate=(
                _unified_without(
                    "plan_count"
                )
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

        self.assertIs(
            report["usable_context_evidence"],
            True,
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

    def _update(
        self,
        *,
        expected_unified_revision=3,
    ) -> dict:
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
                CID,
                expected_unified_revision=
                    expected_unified_revision,
            )

    def _state_path(self) -> Path:
        return (
            self.root
            / "confidence_gate"
            / (CID + ".json")
        )

    def test_missing_candidate_is_deny(
        self,
    ):
        # Unified exists and matches this request; the Candidate does
        # not.
        self._write(
            "unified_context_candidate",
            unified(),
        )

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

        self.assertFalse(
            self._state_path().is_file()
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

        # The binding is auditable in state.
        self.assertEqual(
            written[
                "expected_unified_revision"
            ],
            3,
        )

        self.assertEqual(
            written[
                "source_unified_revision"
            ],
            3,
        )

        self.assertEqual(
            report[
                "expected_unified_revision"
            ],
            3,
        )

    # --------------------------------------------------
    # per-request Unified revision binding
    # --------------------------------------------------

    def test_expected_revision_malformed_is_deny(
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

        for value in (
            None,
            True,
            False,
            0,
            -1,
            "3",
            3.0,
            [],
            {},
        ):
            with self.subTest(
                expected=value
            ):
                report = self._update(
                    expected_unified_revision=
                        value
                )

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
                    "invalid_expected_unified_revision",
                )

        # Nothing was ever persisted for any malformed expectation.
        self.assertFalse(
            self._state_path().is_file()
        )

    def test_stale_unified_revision_is_deny(
        self,
    ):
        # Request A produced Unified revision 3, but the disk file has
        # already moved on to revision 4.
        self._write(
            "context_candidate",
            candidate(),
        )

        newer = unified()

        newer["revision"] = 4

        self._write(
            "unified_context_candidate",
            newer,
        )

        report = self._update(
            expected_unified_revision=3
        )

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
            "confidence_unified_request_revision_mismatch",
        )

        self.assertEqual(
            report[
                "expected_unified_revision"
            ],
            3,
        )

        self.assertEqual(
            report[
                "observed_unified_revision"
            ],
            4,
        )

        # A mismatched observation is not a valid decision, so no
        # state is written.
        self.assertFalse(
            self._state_path().is_file()
        )

    def test_malformed_observed_unified_revision_is_deny(
        self,
    ):
        self._write(
            "context_candidate",
            candidate(),
        )

        for value in (
            None,
            True,
            0,
            -1,
            "3",
        ):
            with self.subTest(
                observed=value
            ):
                broken = unified()

                broken["revision"] = value

                self._write(
                    "unified_context_candidate",
                    broken,
                )

                report = self._update(
                    expected_unified_revision=3
                )

                self.assertEqual(
                    report["decision"],
                    "deny_shadow",
                )

                self.assertIs(
                    report["stored"],
                    False,
                )

                self.assertEqual(
                    report["reason"],
                    "invalid_observed_unified_revision",
                )

        self.assertFalse(
            self._state_path().is_file()
        )

    def test_cross_request_interleaving_is_not_claimed(
        self,
    ):
        # Regression: request A successfully observes its own
        # revision 3 and persists a decision...
        self._write(
            "context_candidate",
            candidate(),
        )

        self._write(
            "unified_context_candidate",
            unified(),
        )

        first = self._update(
            expected_unified_revision=3
        )

        self.assertIs(
            first["stored"],
            True,
        )

        self.assertEqual(
            first["decision"],
            "allow_shadow",
        )

        before = self._state_path().read_text(
            encoding="utf-8"
        )

        # ...then request B overwrites the disk Unified with
        # revision 4 before A's observer runs.
        newer = unified()

        newer["revision"] = 4

        self._write(
            "unified_context_candidate",
            newer,
        )

        stale = self._update(
            expected_unified_revision=3
        )

        # A must not treat B's Unified as its own evidence.
        self.assertIs(
            stale["stored"],
            False,
        )

        self.assertEqual(
            stale["decision"],
            "deny_shadow",
        )

        self.assertEqual(
            stale["reason"],
            "confidence_unified_request_revision_mismatch",
        )

        # ...and the interleave must not overwrite the existing
        # confidence state.
        self.assertEqual(
            self._state_path().read_text(
                encoding="utf-8"
            ),
            before,
        )

    def test_matching_expected_revision_allows_shadow(
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

        report = self._update(
            expected_unified_revision=3
        )

        self.assertIs(
            report["stored"],
            True,
        )

        self.assertEqual(
            report["decision"],
            "allow_shadow",
        )

        self.assertEqual(
            report["reasons"],
            [],
        )

        self.assertIs(
            report["duplicate"],
            False,
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