from __future__ import annotations

from copy import deepcopy
import unittest

from ombrebrain.gateway import rewrite_history_for_cache


def dynamic_attachment(identifier: str) -> str:
    return (
        '<attachment '
        f'id="message_insert_extra_bundle_{identifier}" '
        'filename="Time:12:00" '
        'type="text/plain">'
        "private dynamic data"
        "</attachment>"
    )


def fingertips_attachment() -> str:
    return (
        '<attachment '
        'id="fingertips_context_1" '
        'filename="fingertips.txt" '
        'type="text/plain">'
        "private typing data"
        "</attachment>"
    )


def memory_attachment() -> str:
    return (
        '<attachment '
        'id="memory_1" '
        'filename="relevant_memories.txt" '
        'type="text/plain">'
        "private memory"
        "</attachment>"
    )


def unknown_attachment() -> str:
    return (
        '<attachment '
        'id="document_1" '
        'filename="notes.txt" '
        'type="text/plain">'
        "important document"
        "</attachment>"
    )


class GatewayRewriterTests(unittest.TestCase):

    def test_history_dynamic_removed_current_preserved(self) -> None:

        history_text = (
            "historical user text\n"
            + dynamic_attachment("old")
            + "\n"
            + fingertips_attachment()
            + "\n"
            + memory_attachment()
            + "\n"
            + unknown_attachment()
        )

        current_text = (
            "current user text\n"
            + dynamic_attachment("current")
            + "\n"
            + fingertips_attachment()
        )

        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": history_text,
                            "cache_control": {
                                "type": "ephemeral",
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "assistant answer",
                        }
                    ],
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

        source_before = deepcopy(payload)

        rewritten, report = rewrite_history_for_cache(
            payload
        )

        # Source must never be mutated.
        self.assertEqual(
            payload,
            source_before,
        )

        historical = rewritten["messages"][0]["content"][0]["text"]

        self.assertIn(
            "historical user text",
            historical,
        )

        self.assertNotIn(
            "message_insert_extra_bundle_old",
            historical,
        )

        self.assertIn(
            "fingertips_context_1",
            historical,
        )

        # Phase 2C-A deliberately keeps these.
        self.assertIn(
            'filename="relevant_memories.txt"',
            historical,
        )

        self.assertIn(
            'filename="notes.txt"',
            historical,
        )

        # Metadata survives.
        self.assertEqual(
            rewritten["messages"][0]["content"][0]
            ["cache_control"]["type"],
            "ephemeral",
        )

        # Assistant untouched.
        self.assertEqual(
            rewritten["messages"][1],
            payload["messages"][1],
        )

        # Current user untouched exactly.
        self.assertEqual(
            rewritten["messages"][2],
            payload["messages"][2],
        )

        self.assertEqual(
            report["history_user_messages_seen"],
            1,
        )

        self.assertEqual(
            report["dynamic_context_removed"],
            1,
        )

        self.assertEqual(
            report["fingertips_removed"],
            0,
        )

    def test_normal_words_are_never_removed(self) -> None:

        text = (
            "Time: tomorrow at 8\n"
            "Fingertips 指尖语气\n"
            "相关记忆应该怎么设计？"
        )

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": text,
                },
                {
                    "role": "assistant",
                    "content": "ok",
                },
            ],
        }

        rewritten, report = rewrite_history_for_cache(
            payload
        )

        self.assertEqual(
            rewritten["messages"][0]["content"],
            text,
        )

        self.assertEqual(
            report["dynamic_context_removed"],
            0,
        )

        self.assertEqual(
            report["fingertips_removed"],
            0,
        )

    def test_unknown_attachment_is_preserved_exactly(self) -> None:

        attachment = unknown_attachment()

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": attachment,
                },
                {
                    "role": "assistant",
                    "content": "ok",
                },
            ],
        }

        rewritten, report = rewrite_history_for_cache(
            payload
        )

        self.assertEqual(
            rewritten["messages"][0]["content"],
            attachment,
        )

        self.assertEqual(
            report["dynamic_context_removed"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
