from __future__ import annotations

import json
import unittest

from ombrebrain.gateway import canonical_summary_from_body


class GatewayCanonicalObserverTests(unittest.TestCase):

    def test_summary_without_text_leak(self) -> None:
        secret = "THIS_PRIVATE_TEXT_MUST_NEVER_APPEAR"

        payload = {
            "model": "example",
            "system": [
                {
                    "type": "text",
                    "text": "private system text",
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                secret
                                + "\nTime: 12:00\n"
                                + "Battery: 80%\n"
                            ),
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
                                "current private text\n"
                                "Fingertips 指尖语气\n"
                                "typing rhythm\n"
                            ),
                        }
                    ],
                },
            ],
        }

        original = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        body_before = bytes(original)

        summary = canonical_summary_from_body(original)

        self.assertIsNotNone(summary)
        assert summary is not None

        rendered = json.dumps(
            summary,
            ensure_ascii=False,
        )

        self.assertNotIn(secret, rendered)
        self.assertNotIn("private system text", rendered)
        self.assertNotIn("private answer", rendered)
        self.assertNotIn("current private text", rendered)
        self.assertNotIn("typing rhythm", rendered)

        # Observer must not alter the request bytes.
        self.assertEqual(original, body_before)

        self.assertEqual(
            summary["history_messages"],
            2,
        )

        self.assertEqual(
            summary["history_segments"],
            {
                "perception": 1,
                "user_text": 2,
            },
        )

        self.assertEqual(
            summary["current_segments"],
            {
                "fingertips": 1,
                "user_text": 1,
            },
        )

    def test_non_json_returns_none(self) -> None:
        self.assertIsNone(
            canonical_summary_from_body(
                b"not-json"
            )
        )

    def test_unsupported_json_returns_none(self) -> None:
        self.assertIsNone(
            canonical_summary_from_body(
                b'{"hello":"world"}'
            )
        )


if __name__ == "__main__":
    unittest.main()
