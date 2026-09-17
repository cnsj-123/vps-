from __future__ import annotations

import json
import unittest

from ombrebrain.gateway import rewrite_body_for_cache


def dynamic_attachment(name: str) -> str:
    return (
        '<attachment '
        f'id="message_insert_extra_bundle_{name}" '
        'filename="Time:12:00" '
        'type="text/plain">'
        "dynamic-data"
        "</attachment>"
    )


def fingertips_attachment() -> str:
    return (
        '<attachment '
        'id="fingertips_1" '
        'filename="fingertips.txt" '
        'type="text/plain">'
        "typing-data"
        "</attachment>"
    )


def memory_attachment() -> str:
    return (
        '<attachment '
        'id="memory_1" '
        'filename="relevant_memories.txt" '
        'type="text/plain">'
        "memory-data"
        "</attachment>"
    )


def unknown_attachment() -> str:
    return (
        '<attachment '
        'id="document_1" '
        'filename="notes.txt" '
        'type="text/plain">'
        "document-data"
        "</attachment>"
    )


class GatewayBodyRewriterTests(unittest.TestCase):

    def test_rewrites_history_but_preserves_current(self) -> None:
        current_text = (
            "current question\n"
            + dynamic_attachment("current")
            + fingertips_attachment()
        )

        payload = {
            "model": "m",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "historical question\n"
                                + dynamic_attachment("history")
                                + fingertips_attachment()
                                + memory_attachment()
                                + unknown_attachment()
                            ),
                            "cache_control": {
                                "type": "ephemeral",
                            },
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
                            "text": current_text,
                        }
                    ],
                },
            ],
        }

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        rewritten_body, report = rewrite_body_for_cache(
            body
        )

        self.assertTrue(report["applied"])

        self.assertEqual(
            report["dynamic_context_removed"],
            1,
        )

        self.assertEqual(
            report["fingertips_removed"],
            1,
        )

        rewritten = json.loads(rewritten_body)

        historical = (
            rewritten["messages"][0]
            ["content"][0]["text"]
        )

        self.assertIn(
            "historical question",
            historical,
        )

        self.assertNotIn(
            "message_insert_extra_bundle_history",
            historical,
        )

        self.assertNotIn(
            'filename="fingertips.txt"',
            historical,
        )

        # These deliberately survive Phase 2C.
        self.assertIn(
            'filename="relevant_memories.txt"',
            historical,
        )

        self.assertIn(
            'filename="notes.txt"',
            historical,
        )

        # Current message must remain semantically exact.
        self.assertEqual(
            rewritten["messages"][2],
            payload["messages"][2],
        )

        # Existing cache metadata must survive.
        self.assertEqual(
            rewritten["messages"][0]["content"][0]
            ["cache_control"]["type"],
            "ephemeral",
        )

    def test_no_removal_returns_exact_original_bytes(self) -> None:
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": "historical normal text",
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
            indent=2,
        ).encode("utf-8")

        rewritten_body, report = rewrite_body_for_cache(
            body
        )

        self.assertFalse(report["applied"])
        self.assertEqual(
            rewritten_body,
            body,
        )

    def test_non_json_fails_open_exactly(self) -> None:
        body = b"not-json"

        rewritten_body, report = rewrite_body_for_cache(
            body
        )

        self.assertFalse(report["applied"])
        self.assertEqual(
            rewritten_body,
            body,
        )

    def test_unsupported_json_fails_open_exactly(self) -> None:
        body = b'{"hello":"world"}'

        rewritten_body, report = rewrite_body_for_cache(
            body
        )

        self.assertFalse(report["applied"])
        self.assertEqual(
            rewritten_body,
            body,
        )


if __name__ == "__main__":
    unittest.main()
