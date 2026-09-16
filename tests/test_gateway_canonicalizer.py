from __future__ import annotations

import unittest

from ombrebrain.gateway import OperitAdapter
from ombrebrain.gateway.models import SegmentKind


class GatewayCanonicalizerTests(unittest.TestCase):

    def setUp(self) -> None:
        self.adapter = OperitAdapter()

    def test_operit_shape_and_roundtrip(self) -> None:

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
                                "historical question\n"
                                "Time: 12:00\n"
                                "Battery: 80%\n"
                                "Weather: sunny\n"
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
                                "relevant_memories.txt\n"
                                "memory candidate\n"
                                "Time: 12:01\n"
                                "Battery: 79%\n"
                                "Fingertips 指尖语气\n"
                                "typing rhythm\n"
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

        self.assertTrue(
            self.adapter.supports(payload)
        )

        request = self.adapter.adapt(payload)

        self.assertEqual(
            request.protocol,
            "anthropic_messages",
        )

        self.assertEqual(
            len(request.history),
            2,
        )

        self.assertIsNotNone(
            request.current
        )

        # Phase 2A must never alter the original payload.
        self.assertEqual(
            request.roundtrip_payload(),
            payload,
        )

        history_user = request.history[0]

        self.assertEqual(
            history_user.kinds,
            (
                SegmentKind.USER_TEXT,
                SegmentKind.PERCEPTION,
            ),
        )

        current = request.current
        assert current is not None

        self.assertEqual(
            current.kinds,
            (
                SegmentKind.USER_TEXT,
                SegmentKind.FRONTEND_MEMORY,
                SegmentKind.PERCEPTION,
                SegmentKind.FINGERTIPS,
            ),
        )

        self.assertEqual(
            current.blocks[-1]
            .metadata["cache_control"]["type"],
            "ephemeral",
        )

        self.assertEqual(
            request.passthrough["thinking"]["type"],
            "enabled",
        )

    def test_safe_summary_leaks_no_text(self) -> None:

        secret = "PRIVATE_TEXT_MUST_NOT_APPEAR"

        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": secret,
                        }
                    ],
                }
            ],
        }

        summary = self.adapter.safe_summary(
            payload
        )

        rendered = repr(summary)

        self.assertNotIn(
            secret,
            rendered,
        )

        self.assertEqual(
            summary["current_segments"],
            {
                "user_text": 1,
            },
        )

    def test_normal_text_is_not_reclassified(self) -> None:

        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "A normal sentence about "
                                "time and memory."
                            ),
                        }
                    ],
                }
            ],
        }

        request = self.adapter.adapt(
            payload
        )

        assert request.current is not None

        self.assertEqual(
            request.current.kinds,
            (
                SegmentKind.USER_TEXT,
            ),
        )


if __name__ == "__main__":
    unittest.main()
