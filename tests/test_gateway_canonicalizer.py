from __future__ import annotations

import unittest

from ombrebrain.gateway import OperitAdapter
from ombrebrain.gateway.models import SegmentKind


class GatewayCanonicalizerTests(unittest.TestCase):

    def setUp(self) -> None:
        self.adapter = OperitAdapter()

    def test_operit_attachment_boundary_and_roundtrip(self) -> None:
        payload = {
            "model": "example-model",
            "stream": True,
            "max_tokens": 4096,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024,
            },
            "system": [
                {
                    "type": "text",
                    "text": "stable system",
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "historical question\n"
                                '<attachment '
                                'id="message_insert_extra_bundle_abc" '
                                'filename="Time:12:00" '
                                'type="text/plain">'
                                "【当前时间】12:00\n"
                                "【当前电量】80%\n"
                                "【相关记忆】secret memory"
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
                            "text": "historical answer",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "current question\n"
                                '<attachment '
                                'id="message_insert_extra_bundle_def" '
                                'filename="Time:12:01" '
                                'type="text/plain">'
                                "【当前时间】12:01\n"
                                "【相关记忆】another memory\n"
                                "Fingertips 指尖语气"
                                "</attachment>"
                            ),
                            "cache_control": {
                                "type": "ephemeral",
                                "ttl": "1h",
                            },
                        }
                    ],
                },
            ],
        }

        request = self.adapter.adapt(payload)

        self.assertEqual(
            request.roundtrip_payload(),
            payload,
        )

        self.assertEqual(
            request.history[0].kinds,
            (
                SegmentKind.USER_TEXT,
                SegmentKind.DYNAMIC_CONTEXT,
            ),
        )

        assert request.current is not None

        self.assertEqual(
            request.current.kinds,
            (
                SegmentKind.USER_TEXT,
                SegmentKind.DYNAMIC_CONTEXT,
            ),
        )

    def test_words_in_normal_text_are_never_dynamic(self) -> None:
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Time: tomorrow at 8\n"
                                "相关记忆应该怎么设计？\n"
                                "Fingertips 指尖语气是什么？"
                            ),
                        }
                    ],
                }
            ],
        }

        request = self.adapter.adapt(payload)
        assert request.current is not None

        self.assertEqual(
            request.current.kinds,
            (
                SegmentKind.USER_TEXT,
            ),
        )

    def test_separate_frontend_memory_attachment(self) -> None:
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "hello\n"
                                '<attachment '
                                'id="memory_x" '
                                'filename="relevant_memories.txt" '
                                'type="text/plain">'
                                "private memory"
                                "</attachment>"
                            ),
                        }
                    ],
                }
            ],
        }

        request = self.adapter.adapt(payload)
        assert request.current is not None

        self.assertEqual(
            request.current.kinds,
            (
                SegmentKind.USER_TEXT,
                SegmentKind.FRONTEND_MEMORY,
            ),
        )

    def test_unknown_attachment_stays_unknown(self) -> None:
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "hello\n"
                                '<attachment '
                                'id="document_1" '
                                'filename="notes.txt" '
                                'type="text/plain">'
                                "document"
                                "</attachment>"
                            ),
                        }
                    ],
                }
            ],
        }

        request = self.adapter.adapt(payload)
        assert request.current is not None

        self.assertEqual(
            request.current.kinds,
            (
                SegmentKind.USER_TEXT,
                SegmentKind.UNKNOWN,
            ),
        )


if __name__ == "__main__":
    unittest.main()
