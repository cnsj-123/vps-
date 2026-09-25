from __future__ import annotations

import json
import os
import tempfile
import unittest
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


CID = "ctx_0123456789abcdef"
RID = "ctxreq_0123456789abcdef0123456789abcdef"


def allow_confidence():
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
            1,
    }


def unified(
    memories,
    *,
    revision=3,
    current_user_excluded=True,
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
            "current_user_excluded":
                current_user_excluded,
        },
    }


def memory(
    memory_id,
    content="cue text",
    name=None,
):
    item = {
        "id": memory_id,
        "content": content,
    }

    if name is not None:
        item["metadata"] = {"name": name}

    return item


def policy_for(
    memories,
):
    return evaluate_surfacing_policy(
        conversation_id=CID,
        unified=unified(memories),
        confidence_report=(
            allow_confidence()
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


class MemoryFlashArtifactTests(
    unittest.TestCase
):

    def _write(self, root, *, memories):
        policy = policy_for(memories)

        return update_memory_flash(
            conversation_id=CID,
            cognitive_request_id=RID,
            policy_report=policy,
            expected_unified_revision=3,
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
                    root,
                    memories=[
                        memory("m1"),
                        memory("m2"),
                    ],
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
                            memory(f"m{i}")
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
                            memory(
                                "m1",
                                content="aaa",
                            ),
                            memory(
                                "m2",
                                content="bbb",
                            ),
                            memory(
                                "m3",
                                content="ccc",
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
                    root,
                    memories=[
                        memory(
                            "m1",
                            content=secret,
                        )
                    ],
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
                policy = evaluate_surfacing_policy(
                    conversation_id=CID,
                    unified=unified(
                        [
                            memory(
                                "m1",
                                content="",
                            )
                        ]
                    ),
                    confidence_report=(
                        allow_confidence()
                    ),
                )

                output = update_memory_flash(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    policy_report=policy,
                    expected_unified_revision=3,
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
                    root,
                    memories=[
                        memory(
                            "m1",
                            name="SECRET_CUE",
                        )
                    ],
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
                    [memory("m1")]
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
                    root,
                    memories=[memory("m1")],
                )

            self.assertEqual(
                memory_file.read_bytes(),
                original,
            )


if __name__ == "__main__":
    unittest.main()