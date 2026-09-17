from __future__ import annotations

from copy import deepcopy
import json
import unittest

from ombrebrain.gateway import (
    cache_move_shadow_summary_from_body,
    relocate_current_breakpoint,
)


class GatewayCacheBreakpointTests(unittest.TestCase):

    def _payload(self):
        return {
            "system": [
                {
                    "type": "text",
                    "text": "system-secret",
                    "cache_control": {
                        "type": "ephemeral",
                        "ttl": "1h",
                    },
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "old-user",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "old-assistant",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "current-secret",
                            "cache_control": {
                                "type": "ephemeral",
                                "ttl": "1h",
                            },
                        }
                    ],
                },
            ],
        }

    def test_moves_current_marker_to_previous_assistant(self):
        payload = self._payload()
        before = deepcopy(payload)

        transformed, report = (
            relocate_current_breakpoint(payload)
        )

        self.assertEqual(payload, before)

        self.assertTrue(report["applied"])
        self.assertEqual(report["reason"], "moved")

        self.assertEqual(
            report["boundary_message_index"],
            1,
        )

        self.assertEqual(
            report["boundary_block_index"],
            0,
        )

        self.assertEqual(
            report["moved_ttl"],
            "1h",
        )

        self.assertEqual(
            report["explicit_before"],
            2,
        )

        self.assertEqual(
            report["explicit_after"],
            2,
        )

        current_block = (
            transformed["messages"][2]
            ["content"][0]
        )

        history_block = (
            transformed["messages"][1]
            ["content"][0]
        )

        self.assertNotIn(
            "cache_control",
            current_block,
        )

        self.assertEqual(
            history_block["cache_control"],
            {
                "type": "ephemeral",
                "ttl": "1h",
            },
        )

        # Actual text is untouched.
        self.assertEqual(
            current_block["text"],
            "current-secret",
        )

        # System breakpoint survives.
        self.assertEqual(
            transformed["system"][0]
            ["cache_control"]["ttl"],
            "1h",
        )

    def test_multiple_current_markers_fail_open(self):
        payload = self._payload()

        payload["messages"][2]["content"].append(
            {
                "type": "text",
                "text": "second",
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": "1h",
                },
            }
        )

        before = deepcopy(payload)

        transformed, report = (
            relocate_current_breakpoint(payload)
        )

        self.assertFalse(report["applied"])

        self.assertEqual(
            report["reason"],
            "current_cache_control_count_not_one",
        )

        self.assertEqual(
            transformed,
            before,
        )

    def test_string_boundary_fails_open(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "old-user",
                },
                {
                    "role": "assistant",
                    "content": "old-assistant",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "current",
                            "cache_control": {
                                "type": "ephemeral",
                                "ttl": "1h",
                            },
                        }
                    ],
                },
            ],
        }

        before = deepcopy(payload)

        transformed, report = (
            relocate_current_breakpoint(payload)
        )

        self.assertFalse(report["applied"])

        self.assertEqual(
            report["reason"],
            "boundary_requires_normalization",
        )

        self.assertEqual(transformed, before)

    def test_shadow_does_not_leak_prompt_text(self):
        payload = self._payload()

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        summary = (
            cache_move_shadow_summary_from_body(
                body
            )
        )

        self.assertIsNotNone(summary)
        assert summary is not None

        rendered = json.dumps(
            summary,
            ensure_ascii=False,
        )

        self.assertNotIn(
            "system-secret",
            rendered,
        )

        self.assertNotIn(
            "old-assistant",
            rendered,
        )

        self.assertNotIn(
            "current-secret",
            rendered,
        )

        self.assertTrue(
            summary["breakpoint_move"]["applied"]
        )


if __name__ == "__main__":
    unittest.main()
