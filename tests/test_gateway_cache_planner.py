from __future__ import annotations

import json
import unittest

from ombrebrain.gateway import cache_plan_summary_from_body


def dynamic_attachment(name: str) -> str:
    return (
        '<attachment '
        f'id="message_insert_extra_bundle_{name}" '
        'filename="Time:12:00" '
        'type="text/plain">'
        "dynamic secret"
        "</attachment>"
    )


class GatewayCachePlannerTests(unittest.TestCase):

    def test_recommends_boundary_before_current_user(self) -> None:
        payload = {
            "model": "m",
            "system": [
                {
                    "type": "text",
                    "text": "private system",
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
                            "text": (
                                "historical question"
                                + dynamic_attachment("old")
                            ),
                            "cache_control": {
                                "type": "ephemeral",
                                "ttl": "1h",
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "private answer",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "current question"
                                + dynamic_attachment("current")
                            ),
                            "cache_control": {
                                "type": "ephemeral",
                            },
                        }
                    ],
                },
            ],
        }

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        summary = cache_plan_summary_from_body(body)

        self.assertIsNotNone(summary)
        assert summary is not None

        self.assertEqual(
            summary["strategy"],
            "explicit_history_boundary",
        )

        self.assertEqual(
            summary["current_user_index"],
            2,
        )

        self.assertEqual(
            summary["stable_history_messages"],
            2,
        )

        boundary = summary["boundary"]

        self.assertEqual(
            boundary["message_index"],
            1,
        )

        self.assertEqual(
            boundary["block_index"],
            0,
        )

        self.assertEqual(
            boundary["role"],
            "assistant",
        )

        self.assertFalse(
            boundary["requires_normalization"]
        )

        # Existing markers:
        # system + historical user + current user
        self.assertEqual(
            summary["explicit_breakpoints"],
            3,
        )

        self.assertEqual(
            summary["current_cache_control_blocks"],
            1,
        )

        # Planner simulates historical cleanup.
        self.assertEqual(
            summary["rewrite_simulation"]
            ["dynamic_context_removed"],
            1,
        )

        # No private text may appear in serialized summary.
        rendered = json.dumps(
            summary,
            ensure_ascii=False,
        )

        self.assertNotIn(
            "private system",
            rendered,
        )

        self.assertNotIn(
            "historical question",
            rendered,
        )

        self.assertNotIn(
            "private answer",
            rendered,
        )

        self.assertNotIn(
            "current question",
            rendered,
        )

        self.assertNotIn(
            "dynamic secret",
            rendered,
        )

    def test_string_boundary_reports_normalization(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "old question",
                },
                {
                    "role": "assistant",
                    "content": "old answer",
                },
                {
                    "role": "user",
                    "content": "current question",
                },
            ],
        }

        body = json.dumps(payload).encode("utf-8")

        summary = cache_plan_summary_from_body(body)

        self.assertIsNotNone(summary)
        assert summary is not None

        boundary = summary["boundary"]

        self.assertEqual(
            boundary["message_index"],
            1,
        )

        self.assertEqual(
            boundary["candidate_kind"],
            "string_content",
        )

        self.assertTrue(
            boundary["requires_normalization"]
        )

        self.assertIn(
            "boundary_requires_string_to_block_normalization",
            summary["warnings"],
        )

    def test_non_json_returns_none(self) -> None:
        self.assertIsNone(
            cache_plan_summary_from_body(
                b"not-json"
            )
        )

    def test_unsupported_json_returns_none(self) -> None:
        self.assertIsNone(
            cache_plan_summary_from_body(
                b'{"hello":"world"}'
            )
        )


if __name__ == "__main__":
    unittest.main()
