from __future__ import annotations

import json
import unittest

from ombrebrain.context.pipeline_state import (
    CHAIN_STAGE,
    FRESHNESS_FRESH,
    FRESHNESS_STALE,
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


if __name__ == "__main__":
    unittest.main()
