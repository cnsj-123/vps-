from __future__ import annotations

import json
import unittest

from ombrebrain.gateway import (
    rewrite_cache_stable_body,
)


def dynamic_attachment(name: str) -> str:
    return (
        '<attachment '
        f'id="message_insert_extra_bundle_{name}" '
        'filename="Time:12:00" '
        'type="text/plain">'
        "dynamic-secret"
        "</attachment>"
    )


def fingertips_attachment() -> str:
    return (
        '<attachment '
        'id="fingertips_1" '
        'filename="fingertips.txt" '
        'type="text/plain">'
        "typing-secret"
        "</attachment>"
    )


def frontend_memory_attachment() -> str:
    return (
        '<attachment '
        'id="memory_1" '
        'filename="relevant_memories.txt" '
        'type="text/plain">'
        "memory-secret"
        "</attachment>"
    )


def unknown_attachment() -> str:
    return (
        '<attachment '
        'id="document_1" '
        'filename="notes.txt" '
        'type="text/plain">'
        "document-secret"
        "</attachment>"
    )


class GatewayCacheStableBodyTests(unittest.TestCase):

    def test_atomic_transform(self) -> None:
        current_text = (
            "current-user-text\n"
            + dynamic_attachment("current")
            + fingertips_attachment()
        )

        payload = {
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
                            "text": (
                                "historical-user-text\n"
                                + dynamic_attachment("old")
                                + fingertips_attachment()
                                + frontend_memory_attachment()
                                + unknown_attachment()
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "assistant-secret",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": current_text,
                            "cache_control": {
                                "type": "ephemeral",
                                "ttl": "1h",
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

        transformed_body, report = (
            rewrite_cache_stable_body(body)
        )

        self.assertTrue(report["applied"])
        self.assertEqual(
            report["reason"],
            "cache_stable",
        )

        self.assertEqual(
            report["rewrite"]
            ["dynamic_context_removed"],
            1,
        )

        self.assertEqual(
            report["rewrite"]
            ["fingertips_removed"],
            1,
        )

        self.assertTrue(
            report["breakpoint_move"]["applied"]
        )

        transformed = json.loads(
            transformed_body
        )

        historical = (
            transformed["messages"][0]
            ["content"][0]["text"]
        )

        # Historical dynamic context removed.
        self.assertNotIn(
            "message_insert_extra_bundle_old",
            historical,
        )

        self.assertNotIn(
            'filename="fingertips.txt"',
            historical,
        )

        # Memory and unknown remain.
        self.assertIn(
            'filename="relevant_memories.txt"',
            historical,
        )

        self.assertIn(
            'filename="notes.txt"',
            historical,
        )

        # Current user content is unchanged semantically.
        self.assertEqual(
            transformed["messages"][2]
            ["content"][0]["text"],
            current_text,
        )

        # Current breakpoint removed.
        self.assertNotIn(
            "cache_control",
            transformed["messages"][2]
            ["content"][0],
        )

        # Previous assistant receives it.
        self.assertEqual(
            transformed["messages"][1]
            ["content"][0]["cache_control"],
            {
                "type": "ephemeral",
                "ttl": "1h",
            },
        )

        # System breakpoint remains.
        self.assertEqual(
            transformed["system"][0]
            ["cache_control"]["ttl"],
            "1h",
        )

    def test_breakpoint_failure_is_exact_fail_open(self) -> None:
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

        body = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        transformed_body, report = (
            rewrite_cache_stable_body(body)
        )

        self.assertFalse(report["applied"])

        self.assertEqual(
            report["reason"],
            "breakpoint_move_not_applied",
        )

        self.assertEqual(
            transformed_body,
            body,
        )

    def test_non_json_exact_fail_open(self) -> None:
        body = b"not-json"

        transformed_body, report = (
            rewrite_cache_stable_body(body)
        )

        self.assertFalse(report["applied"])
        self.assertEqual(
            transformed_body,
            body,
        )

    def test_unsupported_exact_fail_open(self) -> None:
        body = b'{"hello":"world"}'

        transformed_body, report = (
            rewrite_cache_stable_body(body)
        )

        self.assertFalse(report["applied"])
        self.assertEqual(
            transformed_body,
            body,
        )


if __name__ == "__main__":
    unittest.main()
