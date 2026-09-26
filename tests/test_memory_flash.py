from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from unittest.mock import patch

from ombrebrain.context.memory_flash import (
    build_flash_cue,
    estimate_cue_tokens,
    memory_flash_status,
    normalize_cue_text,
    resolve_flash_budget,
    update_memory_flash,
)
from ombrebrain.context.memory_surfacing_policy import (
    evaluate_surfacing_policy,
)
from ombrebrain.context.retrieval_decision_shadow import (
    build_candidate_evidence,
)


CID = "ctx_0123456789abcdef"
RID = "ctxreq_0123456789abcdef0123456789abcdef"

_NOW = datetime(
    2026,
    9,
    26,
    12,
    0,
    tzinfo=timezone.utc,
)


def _ts(hours):
    return (
        _NOW
        - timedelta(hours=hours)
    ).isoformat()


def candidate(
    memory_id="m1",
    content="cue text",
    *,
    age_hours=100,
    name=None,
):
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

    return item


def allow_confidence(
    source_unified_revision=3,
):
    return {
        "version":
            "context-confidence-gate.v1",
        "mode":
            "shadow_only",
        "conversation_id": CID,
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
            1,
        "source_unified_revision":
            source_unified_revision,
    }


def unified(
    memories,
    *,
    revision=3,
):
    return {
        "version":
            "unified-context-candidate.v1",
        "conversation_id": CID,
        "revision": revision,
        "sections": {
            "memories": memories,
        },
        "telemetry": {
            "current_user_excluded": True,
        },
    }


def policy_for(
    memories,
    *,
    revision=3,
):
    return evaluate_surfacing_policy(
        conversation_id=CID,
        unified=unified(
            memories,
            revision=revision,
        ),
        confidence_report=(
            allow_confidence(
                source_unified_revision=(
                    revision
                )
            )
        ),
        shadow_evidence=(
            build_candidate_evidence(
                memories,
                now=_NOW,
            )
        ),
    )


class FlashBudgetTests(
    unittest.TestCase
):

    def test_defaults(self):
        with patch.dict(
            os.environ,
            {},
            clear=False,
        ):
            for key in (
                "OMBRE_MEMORY_FLASH_MAX_ITEMS",
                "OMBRE_MEMORY_FLASH_TOKEN_BUDGET",
                "OMBRE_MEMORY_FLASH_MAX_CHARS",
            ):
                os.environ.pop(key, None)

            budget = resolve_flash_budget()

        self.assertEqual(
            budget,
            {
                "max_items": 3,
                "token_budget": 96,
                "max_chars": 160,
            },
        )

    def test_env_overrides(self):
        with patch.dict(
            os.environ,
            {
                "OMBRE_MEMORY_FLASH_MAX_ITEMS":
                    "5",
                "OMBRE_MEMORY_FLASH_TOKEN_BUDGET":
                    "120",
                "OMBRE_MEMORY_FLASH_MAX_CHARS":
                    "200",
            },
            clear=False,
        ):
            budget = resolve_flash_budget()

        self.assertEqual(
            budget,
            {
                "max_items": 5,
                "token_budget": 120,
                "max_chars": 200,
            },
        )

    def test_invalid_values_fall_back_to_default(
        self,
    ):
        for bad in (
            "",
            "abc",
            "-1",
            "0",
            "true",
            "1e3",
            "1.5",
            "None",
        ):
            with patch.dict(
                os.environ,
                {
                    "OMBRE_MEMORY_FLASH_MAX_ITEMS":
                        bad,
                },
                clear=False,
            ):
                budget = resolve_flash_budget()

            self.assertEqual(
                budget["max_items"],
                3,
                bad,
            )

    def test_bool_is_not_a_valid_int(self):
        with patch.dict(
            os.environ,
            {},
            clear=False,
        ):
            budget = resolve_flash_budget(
                max_items=True,
                token_budget=False,
                max_chars=True,
            )

        self.assertEqual(
            budget["max_items"],
            3,
        )
        self.assertEqual(
            budget["token_budget"],
            96,
        )
        self.assertEqual(
            budget["max_chars"],
            160,
        )

    def test_hard_caps(self):
        with patch.dict(
            os.environ,
            {
                "OMBRE_MEMORY_FLASH_MAX_ITEMS":
                    "1000",
                "OMBRE_MEMORY_FLASH_TOKEN_BUDGET":
                    "100000",
                "OMBRE_MEMORY_FLASH_MAX_CHARS":
                    "100000",
            },
            clear=False,
        ):
            budget = resolve_flash_budget()

        self.assertEqual(
            budget,
            {
                "max_items": 8,
                "token_budget": 256,
                "max_chars": 320,
            },
        )


class FlashCueTests(
    unittest.TestCase
):

    def test_whitespace_is_normalized(self):
        self.assertEqual(
            normalize_cue_text(
                "  a   b\n\tc  "
            ),
            "a b c",
        )

    def test_cue_is_bounded(self):
        cue = build_flash_cue(
            {
                "id": "m1",
                "content": "x" * 1000,
            },
            max_chars=160,
        )

        self.assertEqual(len(cue), 160)

    def test_title_is_preferred(self):
        cue = build_flash_cue(
            {
                "id": "m1",
                "content": "the body",
                "metadata": {
                    "name": "the title"
                },
            }
        )

        self.assertEqual(cue, "the title")

    def test_no_flashable_cue(self):
        for value in (
            None,
            "x",
            {"id": "m1"},
            {"id": "m1", "content": "   "},
        ):
            self.assertIsNone(
                build_flash_cue(value),
                value,
            )

    def test_token_estimate_matches_context_layer(
        self,
    ):
        self.assertEqual(
            estimate_cue_tokens(""),
            0,
        )
        self.assertEqual(
            estimate_cue_tokens("abc"),
            (3 + 2) // 3,
        )


class FlashPolicyBindingTests(
    unittest.TestCase
):
    """A Flash artifact may only be built from a bound policy."""

    def _flash_path(self, root):
        return (
            Path(root)
            / "memory_flash"
            / CID
            / (RID + ".json")
        )

    def _call(self, root, policy):
        return update_memory_flash(
            conversation_id=CID,
            cognitive_request_id=RID,
            policy_report=policy,
            expected_unified_revision=3,
        )

    def test_valid_policy_is_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                output = self._call(
                    root,
                    policy_for(
                        [candidate("m1")]
                    ),
                )

                self.assertTrue(
                    self._flash_path(
                        root
                    ).is_file()
                )

        self.assertTrue(output["stored"])

    def test_policy_unified_revision_mismatch(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                policy = policy_for(
                    [candidate("m1")],
                    revision=4,
                )

                output = self._call(
                    root,
                    policy,
                )

                self.assertFalse(
                    self._flash_path(
                        root
                    ).exists()
                )

        self.assertFalse(output["stored"])
        self.assertEqual(
            output["reason"],
            "flash_policy_unified_revision_mismatch",
        )

    def test_invalid_policy_source_revision(
        self,
    ):
        policy = policy_for(
            [candidate("m1")]
        )
        policy["source_unified_revision"] = None

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                output = self._call(
                    root,
                    policy,
                )

                self.assertFalse(
                    self._flash_path(
                        root
                    ).exists()
                )

        self.assertFalse(output["stored"])
        self.assertEqual(
            output["reason"],
            "invalid_flash_policy_unified_revision",
        )

    def test_policy_conversation_mismatch(
        self,
    ):
        # Another conversation's policy must never be written into
        # this conversation's Flash path, even with the same revision
        # number.
        policy = policy_for(
            [candidate("m1")]
        )
        policy["conversation_id"] = (
            "ctx_ffffffffffffffff"
        )

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                output = self._call(
                    root,
                    policy,
                )

                self.assertFalse(
                    self._flash_path(
                        root
                    ).exists()
                )

        self.assertFalse(output["stored"])
        self.assertEqual(
            output["reason"],
            "flash_policy_conversation_mismatch",
        )

    def test_malformed_policy_report(self):
        for policy in (
            {
                "version": "wrong",
                "mode": "shadow_only",
                "source_unified_revision": 3,
            },
            {
                "version":
                    "memory-surfacing-policy.v1",
                "mode": "live",
                "source_unified_revision": 3,
            },
            [],
            None,
            "nope",
        ):
            with tempfile.TemporaryDirectory() as root:
                with patch.dict(
                    os.environ,
                    {
                        "OMBRE_CONTEXT_STATE_DIR":
                            root,
                    },
                    clear=False,
                ):
                    output = self._call(
                        root,
                        policy,
                    )

                    self.assertFalse(
                        self._flash_path(
                            root
                        ).exists()
                    )

            self.assertFalse(output["stored"])

            self.assertIn(
                output["reason"],
                (
                    "malformed_policy_report",
                    "invalid_policy_report",
                ),
            )


class MemoryFlashArtifactTests(
    unittest.TestCase
):

    def _write(
        self,
        *,
        memories,
        revision=3,
    ):
        return update_memory_flash(
            conversation_id=CID,
            cognitive_request_id=RID,
            policy_report=policy_for(
                memories,
                revision=revision,
            ),
            expected_unified_revision=revision,
        )

    def test_artifact_is_written_per_request(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                output = self._write(
                    memories=[
                        candidate(
                            "m1",
                            "cue one",
                        ),
                        candidate(
                            "m2",
                            "cue two",
                        ),
                    ]
                )

                path = (
                    Path(root)
                    / "memory_flash"
                    / CID
                    / (RID + ".json")
                )

                self.assertTrue(
                    path.is_file()
                )

        self.assertTrue(output["stored"])
        self.assertEqual(
            output["surfaced_count"],
            2,
        )
        self.assertEqual(
            output["version"],
            "memory-flash.v1",
        )
        self.assertEqual(
            output["mode"],
            "shadow_only",
        )
        self.assertEqual(
            output["source_unified_revision"],
            3,
        )
        self.assertNotIn(
            "created_at",
            output,
        )

    def test_max_items_is_enforced(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                output = update_memory_flash(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    policy_report=policy_for(
                        [
                            candidate(
                                f"m{i}",
                                f"cue {i}",
                            )
                            for i in range(5)
                        ]
                    ),
                    expected_unified_revision=3,
                    budget={
                        "max_items": 3,
                        "token_budget": 96,
                        "max_chars": 160,
                    },
                )

        self.assertEqual(
            output["surfaced_count"],
            3,
        )

    def test_token_budget_stops_later_candidates(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                output = update_memory_flash(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    policy_report=policy_for(
                        [
                            candidate(
                                "m1",
                                "aaa",
                            ),
                            candidate(
                                "m2",
                                "bbb",
                            ),
                            candidate(
                                "m3",
                                "ccc",
                            ),
                        ]
                    ),
                    expected_unified_revision=3,
                    budget={
                        "max_items": 8,
                        "token_budget": 2,
                        "max_chars": 160,
                    },
                )

        # Each 3-char cue costs 1 token; only two fit in 2 tokens.
        self.assertEqual(
            output["surfaced_count"],
            2,
        )
        self.assertLessEqual(
            output["estimated_tokens"],
            2,
        )

    def test_full_raw_memory_is_never_stored(
        self,
    ):
        secret = (
            "SECRET_MEMORY_TEXT" * 100
        )

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                self._write(
                    memories=[
                        candidate(
                            "m1",
                            secret,
                        )
                    ]
                )

                text = (
                    Path(root)
                    / "memory_flash"
                    / CID
                    / (RID + ".json")
                ).read_text(
                    encoding="utf-8"
                )

        self.assertNotIn(secret, text)

        # The bounded cue prefix is present.
        self.assertIn(
            "SECRET_MEMORY_TEXT",
            text,
        )

        self.assertLess(
            len(json.loads(text)["flashes"][0]["cue"]),
            321,
        )

    def test_no_surface_still_writes_one_artifact(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                output = self._write(
                    memories=[
                        candidate(
                            "m1",
                            "",
                        )
                    ]
                )

                status = memory_flash_status(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                )

        self.assertTrue(output["stored"])
        self.assertEqual(
            output["decision"],
            "no_surface",
        )
        self.assertEqual(
            output["surfaced_count"],
            0,
        )
        self.assertEqual(
            output["reason"],
            "no_flashable_cue",
        )
        self.assertTrue(status["exists"])
        self.assertEqual(
            status["decision"],
            "no_surface",
        )

    def test_status_never_exposes_cues(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                self._write(
                    memories=[
                        candidate(
                            "m1",
                            name="SECRET_CUE",
                        )
                    ]
                )

                status = memory_flash_status(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                )

        self.assertNotIn(
            "cue",
            json.dumps(status),
        )
        self.assertNotIn(
            "SECRET_CUE",
            json.dumps(status),
        )

    def test_distinct_requests_never_overwrite(
        self,
    ):
        other = (
            "ctxreq_ffffffffffffffffffffffffffffffff"
        )

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                policy = policy_for(
                    [candidate("m1")]
                )

                update_memory_flash(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    policy_report=policy,
                    expected_unified_revision=3,
                )

                update_memory_flash(
                    conversation_id=CID,
                    cognitive_request_id=other,
                    policy_report=policy,
                    expected_unified_revision=3,
                )

                directory = (
                    Path(root)
                    / "memory_flash"
                    / CID
                )

                files = sorted(
                    p.name
                    for p in directory.iterdir()
                )

        self.assertEqual(
            files,
            sorted(
                [
                    RID + ".json",
                    other + ".json",
                ]
            ),
        )

    def test_invalid_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                with self.assertRaises(
                    ValueError
                ):
                    update_memory_flash(
                        conversation_id="bad",
                        cognitive_request_id=RID,
                        policy_report={},
                        expected_unified_revision=3,
                    )

                with self.assertRaises(
                    ValueError
                ):
                    update_memory_flash(
                        conversation_id=CID,
                        cognitive_request_id="bad",
                        policy_report={},
                        expected_unified_revision=3,
                    )

    def test_source_memory_file_is_untouched(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            memory_file = (
                Path(root)
                / "buckets"
                / "m1.json"
            )

            memory_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            original = (
                b'{"id":"m1","content":"raw"}'
            )

            memory_file.write_bytes(
                original
            )

            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                self._write(
                    memories=[candidate("m1")],
                )

            self.assertEqual(
                memory_file.read_bytes(),
                original,
            )


if __name__ == "__main__":
    unittest.main()