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
                                + "\n"
                                '<attachment '
                                'id="message_insert_extra_bundle_1" '
                                'filename="Time:12:00" '
                                'type="text/plain">'
                                "private dynamic data"
                                "</attachment>"
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
                                '<attachment '
                                'id="message_insert_extra_bundle_2" '
                                'filename="Time:12:01" '
                                'type="text/plain">'
                                "current private dynamic data"
                                "</attachment>"
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

        before = bytes(original)

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
        self.assertNotIn("private dynamic data", rendered)

        self.assertEqual(original, before)

        self.assertEqual(
            summary["history_user_layout"],
            [
                {
                    "index": 0,
                    "segments": [
                        "user_text",
                        "dynamic_context",
                    ],
                }
            ],
        )

        self.assertEqual(
            summary["current_layout"]["segments"],
            [
                "user_text",
                "dynamic_context",
            ],
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
