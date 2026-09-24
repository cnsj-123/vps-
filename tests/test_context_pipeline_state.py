from __future__ import annotations

import json
import unittest

from ombrebrain.context.pipeline_state import (
    CHAIN_STAGE,
    FRESHNESS_FRESH,
    FRESHNESS_STALE,
    STAGE_CANDIDATE,
    STAGE_GATE,
    STAGE_PREVIEW,
    STAGE_UNIFIED,
    build_context_chain_event,
    from_stage_payload,
)
from ombrebrain.context.validators.freshness import (
    validate_context_freshness,
)


CID = "ctx_0123456789abcdef"


def unified_payload(
    *,
    revision=3,
    conversation_candidate=2,
):
    return {
        "version":
            "unified-context-candidate.v1",
        "conversation_id":
            CID,
        "revision":
            revision,
        "source_revisions": {
            "conversation_candidate":
                conversation_candidate,
            "conversation_source":
                7,
            "state":
                1,
        },
        "sections": {
            "trusted_facts": [
                {"text": "secret-fact"}
            ]
        },
        "telemetry": {
            "estimated_tokens":
                42,
            "token_budget":
                1200,
            "truncated":
                False,
        },
    }


def preview_payload(
    *,
    revision=5,
    source_revision=3,
    eligible=True,
    reason=None,
):
    return {
        "version":
            "context-injection-preview.v1",
        "conversation_id":
            CID,
        "revision":
            revision,
        "source_revision":
            source_revision,
        "eligible":
            eligible,
        "reason":
            reason,
        "estimated_tokens":
            30,
        "token_budget":
            1000,
        "section_names": [
            "trusted_facts"
        ],
        "render_sha256":
            "a" * 64,
        "rendered":
            "OMBRE CONTEXT DATA\nsecret-render",
    }


def gate_payload(
    *,
    revision=6,
    source_preview_revision=5,
    decision="allow_shadow",
):
    return {
        "version":
            "context-injection-gate.v1",
        "mode":
            "shadow_only",
        "conversation_id":
            CID,
        "revision":
            revision,
        "decision":
            decision,
        "allowed":
            decision == "allow_shadow",
        "reason":
            None,
        "reasons": [],
        "source_candidate_revision":
            2,
        "source_unified_revision":
            3,
        "source_preview_revision":
            source_preview_revision,
        "estimated_tokens":
            30,
        "token_budget":
            1000,
        "section_names": [
            "trusted_facts"
        ],
        "render_sha256":
            "a" * 64,
    }


class FreshnessValidatorTests(
    unittest.TestCase
):

    def test_valid_pair(self):
        result = validate_context_freshness(
            checked_revision=5,
            expected_revision=5,
        )

        self.assertEqual(
            result,
            {
                "valid": True,
                "reason": None,
                "checked_revision": 5,
                "expected_revision": 5,
            },
        )

    def test_mismatch(self):
        result = validate_context_freshness(
            checked_revision=4,
            expected_revision=5,
            mismatch_reason="preview_revision_mismatch",
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "preview_revision_mismatch",
        )

    def test_invalid_revision_is_reported_first(
        self,
    ):
        for value in (
            None,
            True,
            0,
            -1,
            "5",
            5.0,
        ):
            result = validate_context_freshness(
                checked_revision=value,
                expected_revision=5,
                invalid_reason=(
                    "invalid_gate_source_revision"
                ),
                mismatch_reason=(
                    "preview_revision_mismatch"
                ),
            )

            self.assertFalse(
                result["valid"],
                value,
            )
            self.assertEqual(
                result["reason"],
                "invalid_gate_source_revision",
                value,
            )

    def test_never_raises(self):
        result = validate_context_freshness(
            checked_revision=None,
            expected_revision=None,
        )

        self.assertFalse(result["valid"])


class PipelineStateTests(
    unittest.TestCase
):

    def test_non_dict_payload_is_safe(self):
        for value in (
            None,
            123,
            "x",
            [],
        ):
            state = from_stage_payload(
                STAGE_UNIFIED,
                value,
            )

            self.assertIsNone(
                state.revision
            )
            self.assertEqual(
                state.sections,
                {},
            )

    def test_unified_normalization(self):
        state = from_stage_payload(
            STAGE_UNIFIED,
            unified_payload(),
        )

        self.assertEqual(
            state.conversation_id,
            CID,
        )
        self.assertEqual(state.revision, 3)

        # source_revision = the Candidate revision.
        self.assertEqual(
            state.source_revision,
            2,
        )
        self.assertEqual(
            state.estimated_tokens,
            42,
        )
        self.assertEqual(
            state.token_budget,
            1200,
        )
        self.assertEqual(
            state.metadata[
                "source_revisions"
            ][
                "conversation_source"
            ],
            7,
        )

    def test_preview_normalization(self):
        state = from_stage_payload(
            STAGE_PREVIEW,
            preview_payload(),
        )

        self.assertEqual(state.revision, 5)
        self.assertEqual(
            state.source_revision,
            3,
        )
        self.assertEqual(
            state.decision,
            "eligible",
        )

    def test_preview_ineligible_decision(self):
        state = from_stage_payload(
            STAGE_PREVIEW,
            preview_payload(
                eligible=False,
                reason="empty_context",
            ),
        )

        self.assertEqual(
            state.decision,
            "ineligible",
        )
        self.assertEqual(
            state.metadata["reason"],
            "empty_context",
        )

    def test_gate_normalization(self):
        state = from_stage_payload(
            STAGE_GATE,
            gate_payload(),
        )

        self.assertEqual(state.revision, 6)
        self.assertEqual(
            state.source_revision,
            5,
        )
        self.assertEqual(
            state.decision,
            "allow_shadow",
        )
        self.assertEqual(
            state.metadata[
                "source_unified_revision"
            ],
            3,
        )


class ContextChainEventTests(
    unittest.TestCase
):

    def test_fresh_chain_event_shape(self):
        event = build_context_chain_event(
            unified=unified_payload(),
            preview=preview_payload(),
            gate=gate_payload(),
        )

        self.assertEqual(
            event["stage"],
            CHAIN_STAGE,
        )
        self.assertEqual(
            event["decision"],
            FRESHNESS_FRESH,
        )
        self.assertIsNone(event["reason"])
        self.assertEqual(event["revision"], 3)
        self.assertEqual(
            event["estimated_tokens"],
            30,
        )

        self.assertTrue(
            event["metrics"][
                "freshness"
            ]["valid"]
        )

        for stage in (
            STAGE_UNIFIED,
            STAGE_PREVIEW,
            STAGE_GATE,
        ):
            self.assertIn(
                stage,
                event["metrics"]["stages"],
            )

    def test_stale_preview_source_is_reported(
        self,
    ):
        event = build_context_chain_event(
            unified=unified_payload(
                revision=4
            ),
            preview=preview_payload(
                source_revision=3
            ),
            gate=gate_payload(),
        )

        self.assertEqual(
            event["decision"],
            FRESHNESS_STALE,
        )
        self.assertEqual(
            event["reason"],
            "preview_source_revision_mismatch",
        )
        self.assertFalse(
            event["metrics"][
                "freshness"
            ]["valid"]
        )

    def test_gate_preview_mismatch_is_reported(
        self,
    ):
        event = build_context_chain_event(
            unified=unified_payload(),
            preview=preview_payload(),
            gate=gate_payload(
                source_preview_revision=4
            ),
        )

        self.assertEqual(
            event["decision"],
            FRESHNESS_STALE,
        )
        self.assertEqual(
            event["reason"],
            "preview_revision_mismatch",
        )

    def test_event_never_leaks_context_text(self):
        event = build_context_chain_event(
            unified=unified_payload(),
            preview=preview_payload(),
            gate=gate_payload(),
        )

        serialized = json.dumps(
            event,
            ensure_ascii=False,
        )

        for secret in (
            "secret-fact",
            "secret-render",
            "OMBRE CONTEXT DATA",
            "<ombre_context_data>",
        ):
            self.assertNotIn(
                secret,
                serialized,
            )


class PipelineParserTests(
    unittest.TestCase
):
    """Round 2: payload -> state parsing coverage."""

    def test_payload_to_state_field_mapping(
        self,
    ):
        payload = {
            "conversation_id":
                CID,
            "revision":
                7,
            "source_revision":
                6,
            "sections": {
                "state": {
                    "current_focus":
                        "secret-focus"
                }
            },
            "estimated_tokens":
                11,
            "token_budget":
                22,
            "telemetry": {
                "estimated_tokens":
                    99,
                "token_budget":
                    88,
            },
        }

        state = from_stage_payload(
            STAGE_CANDIDATE,
            payload,
        )

        self.assertEqual(
            state.stage,
            STAGE_CANDIDATE,
        )
        self.assertEqual(
            state.conversation_id,
            CID,
        )
        self.assertEqual(
            state.revision,
            7,
        )
        self.assertEqual(
            state.source_revision,
            6,
        )
        self.assertEqual(
            state.sections,
            {
                "state": {
                    "current_focus":
                        "secret-focus"
                }
            },
        )

        # Top-level values win over telemetry.
        self.assertEqual(
            state.estimated_tokens,
            11,
        )
        self.assertEqual(
            state.token_budget,
            22,
        )

        # Candidate keeps the generic shape only.
        self.assertIsNone(
            state.decision
        )
        self.assertEqual(
            state.metadata,
            {},
        )

    def test_telemetry_fallback_when_top_level_missing(
        self,
    ):
        state = from_stage_payload(
            STAGE_UNIFIED,
            {
                "telemetry": {
                    "estimated_tokens":
                        42,
                    "token_budget":
                        1200,
                }
            },
        )

        self.assertEqual(
            state.estimated_tokens,
            42,
        )
        self.assertEqual(
            state.token_budget,
            1200,
        )

    def test_unified_top_level_source_revision_wins(
        self,
    ):
        payload = unified_payload()
        payload["source_revision"] = 9

        state = from_stage_payload(
            STAGE_UNIFIED,
            payload,
        )

        self.assertEqual(
            state.source_revision,
            9,
        )

    def test_preview_non_list_section_names_become_empty(
        self,
    ):
        payload = preview_payload()
        payload["section_names"] = (
            "trusted_facts"
        )

        state = from_stage_payload(
            STAGE_PREVIEW,
            payload,
        )

        self.assertEqual(
            state.metadata[
                "section_names"
            ],
            [],
        )

    def test_gate_top_level_source_revision_wins(
        self,
    ):
        payload = gate_payload()
        payload["source_revision"] = 8

        state = from_stage_payload(
            STAGE_GATE,
            payload,
        )

        self.assertEqual(
            state.source_revision,
            8,
        )

    def test_invalid_revision_values_are_dropped(
        self,
    ):
        for bad in (
            True,
            "5",
            -1,
            1.5,
            None,
            [1],
        ):
            state = from_stage_payload(
                STAGE_PREVIEW,
                {
                    "revision": bad,
                    "source_revision": bad,
                },
            )

            self.assertIsNone(
                state.revision,
                bad,
            )
            self.assertIsNone(
                state.source_revision,
                bad,
            )

    def test_zero_revision_is_kept_by_lax_extractor(
        self,
    ):
        # Documents the current behavior: the parser's _as_int
        # accepts any non-negative int, including 0. Strict 1-based
        # validation lives in the freshness validator.
        state = from_stage_payload(
            STAGE_PREVIEW,
            {"revision": 0},
        )

        self.assertEqual(
            state.revision,
            0,
        )

    def test_candidate_stage_has_no_stage_specific_fields(
        self,
    ):
        state = from_stage_payload(
            STAGE_CANDIDATE,
            unified_payload(),
        )

        self.assertIsNone(
            state.decision
        )
        self.assertEqual(
            state.metadata,
            {},
        )

    def test_replace_does_not_mutate_original(
        self,
    ):
        state = from_stage_payload(
            STAGE_PREVIEW,
            preview_payload(),
        )

        updated = state.replace(
            revision=99
        )

        self.assertEqual(
            updated.revision,
            99,
        )
        self.assertEqual(
            state.revision,
            5,
        )

    def test_split_module_import_paths(self):
        # The canonical homes after the Round 2 split...
        from ombrebrain.context import (
            pipeline_events,
            pipeline_parser,
        )
        from ombrebrain.context.pipeline_events import (
            to_event,
        )

        # ...expose the very same objects as the compat imports.
        self.assertIs(
            pipeline_parser.from_stage_payload,
            from_stage_payload,
        )
        self.assertIs(
            pipeline_events.build_context_chain_event,
            build_context_chain_event,
        )

        state = from_stage_payload(
            STAGE_PREVIEW,
            preview_payload(),
        )

        event = to_event(
            state,
            reason="observed",
        )

        self.assertEqual(
            event["stage"],
            STAGE_PREVIEW,
        )
        self.assertEqual(
            event["decision"],
            "eligible",
        )
        self.assertEqual(
            event["revision"],
            5,
        )
        self.assertEqual(
            event["reason"],
            "observed",
        )

    def test_compat_reexports_from_pipeline_state(self):
        import ombrebrain.context.pipeline_state as pipeline_state

        # Old import path keeps working...
        self.assertTrue(
            callable(
                pipeline_state.from_stage_payload
            )
        )
        self.assertTrue(
            callable(
                pipeline_state.build_context_chain_event
            )
        )

        # ...and unknown attributes still fail normally.
        with self.assertRaises(
            AttributeError
        ):
            pipeline_state.does_not_exist


class ChainEventRound2Tests(
    unittest.TestCase
):
    """Round 2: same-revision chain and module-level to_event."""

    def test_same_revision_chain_is_fresh(
        self,
    ):
        event = build_context_chain_event(
            unified=unified_payload(
                revision=2,
                conversation_candidate=1,
            ),
            preview=preview_payload(
                revision=1,
                source_revision=2,
            ),
            gate=gate_payload(
                revision=1,
                source_preview_revision=1,
            ),
        )

        self.assertEqual(
            event["decision"],
            FRESHNESS_FRESH,
        )
        self.assertIsNone(
            event["reason"]
        )

        freshness = event[
            "metrics"
        ]["freshness"]

        self.assertTrue(
            freshness["valid"]
        )
        self.assertEqual(
            freshness[
                "checked_revision"
            ],
            2,
        )
        self.assertEqual(
            freshness[
                "expected_revision"
            ],
            2,
        )

        stages = event["metrics"][
            "stages"
        ]

        self.assertEqual(
            stages[STAGE_PREVIEW][
                "freshness_state"
            ],
            FRESHNESS_FRESH,
        )


if __name__ == "__main__":
    unittest.main()
