from __future__ import annotations

import json
import unittest

from ombrebrain.gateway import rewrite_shadow_summary_from_body


class GatewayRewriteShadowTests(unittest.TestCase):

    def test_shadow_reports_removals_only(self) -> None:
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "historical text\n"
                                '<attachment '
                                'id="message_insert_extra_bundle_old" '
                                'filename="Time:12:00" '
                                'type="text/plain">'
                                "dynamic secret"
                                "</attachment>\n"
                                '<attachment '
                                'id="fingertips_old" '
                                'filename="fingertips.txt" '
                                'type="text/plain">'
                                "typing secret"
                                "</attachment>\n"
                                '<attachment '
                                'id="memory_old" '
                                'filename="relevant_memories.txt" '
                                'type="text/plain">'
                                "memory secret"
                                "</attachment>"
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "answer",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "current text\n"
                                '<attachment '
                                'id="message_insert_extra_bundle_current" '
                                'filename="Time:12:01" '
                                'type="text/plain">'
                                "current dynamic"
                                "</attachment>"
                            ),
                        }
                    ],
                },
            ],
        }

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        summary = rewrite_shadow_summary_from_body(body)

        self.assertIsNotNone(summary)
        assert summary is not None

        self.assertEqual(
            summary["history_user_messages_seen"],
            1,
        )

        self.assertEqual(
            summary["dynamic_context_removed"],
            1,
        )

        self.assertEqual(
            summary["fingertips_removed"],
            0,
        )

        self.assertGreater(
            summary["bytes_removed"],
            0,
        )

        self.assertGreater(
            summary["normalized_original_bytes"],
            summary["rewritten_bytes"],
        )

    def test_no_dynamic_content_means_zero_savings(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "normal historical text",
                },
                {
                    "role": "assistant",
                    "content": "answer",
                },
                {
                    "role": "user",
                    "content": "current text",
                },
            ],
        }

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        summary = rewrite_shadow_summary_from_body(body)

        self.assertIsNotNone(summary)
        assert summary is not None

        self.assertEqual(
            summary["dynamic_context_removed"],
            0,
        )

        self.assertEqual(
            summary["fingertips_removed"],
            0,
        )

        self.assertEqual(
            summary["bytes_removed"],
            0,
        )

    def test_non_json_returns_none(self) -> None:
        self.assertIsNone(
            rewrite_shadow_summary_from_body(
                b"not-json"
            )
        )


if __name__ == "__main__":
    unittest.main()
