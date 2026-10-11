from __future__ import annotations

import copy
import json
import unittest

from ombrebrain.gateway import OperitAdapter
from ombrebrain.gateway.canonicalizer import (
    REWRITE_DROP,
    REWRITE_KEEP_EXACT,
    rewrite_frontend_memory_attachments,
)
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


def _attachment(filename, body, attachment_id):
    return (
        '<attachment id="'
        + attachment_id
        + '" filename="'
        + filename
        + '" type="text/plain">'
        + body
        + "</attachment>"
    )


_FRONTEND_A = _attachment(
    "relevant_memories.txt", "MEMORY_A_BODY", "front_a"
)
_FRONTEND_B = _attachment(
    "relevant_memories.txt", "MEMORY_B_BODY", "front_b"
)
_DYNAMIC = _attachment(
    'Time:"12:00"',
    "dyn body",
    "message_insert_extra_bundle_x",
)
_UNKNOWN = _attachment(
    "notes.txt", "unknown body", "document_1"
)

_FINGERTIPS = _attachment(
    "fingertips_style.txt", "tone body", "fingertips_1"
)


def _payload():
    return {
        "model": "m",
        "max_tokens": 4096,
        "system": [{"type": "text", "text": "STABLE_SYSTEM"}],
        "messages": [
            {
                "role": "user",
                "content": (
                    "HISTORY_ORDINARY "
                    + _DYNAMIC
                    + " "
                    + _FRONTEND_A
                    + " HISTORY_TAIL"
                ),
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ASSISTANT_TEXT"},
                    {
                        "type": "image",
                        "source": {"data": "NON_TEXT_SECRET"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "CURRENT_ORDINARY"},
                    {
                        "type": "text",
                        "text": (
                            _FRONTEND_B
                            + " "
                            + _UNKNOWN
                            + " "
                            + _FINGERTIPS
                        ),
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "tool_result",
                        "content": "NON_TEXT_TOOL",
                    },
                ],
            },
        ],
    }


class FrontendAttachmentControlTransformTests(
    unittest.TestCase,
):
    """Unit tests for the canonical, fail-closed control transform."""

    def test_keep_exact_preserves_everything(self) -> None:
        payload = _payload()
        original = copy.deepcopy(payload)

        result = rewrite_frontend_memory_attachments(
            payload,
            lambda **_: REWRITE_KEEP_EXACT,
        )

        self.assertTrue(result.ok)
        self.assertIsNone(result.reason)
        self.assertEqual(result.occurrence_count, 2)
        self.assertEqual(result.kept_count, 2)
        self.assertEqual(result.dropped_count, 0)

        # Every action is keep_exact, so the rewritten payload deep-equals
        # the original and the original object is untouched.
        self.assertEqual(result.payload, original)
        self.assertEqual(payload, original)

    def test_drop_removes_only_the_target_attachment(self) -> None:
        payload = _payload()
        original = copy.deepcopy(payload)

        def decide(occurrence, **_):
            return (
                REWRITE_KEEP_EXACT
                if occurrence == 0
                else REWRITE_DROP
            )

        result = rewrite_frontend_memory_attachments(
            payload, decide
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.occurrence_count, 2)
        self.assertEqual(result.kept_count, 1)
        self.assertEqual(result.dropped_count, 1)

        history_text = result.payload["messages"][0][
            "content"
        ]
        current_text = (
            result.payload["messages"][2]["content"][1][
                "text"
            ]
        )

        # Only B disappears; A survives byte-for-byte, and every other
        # attachment / ordinary text / system / non-text block survives.
        self.assertIn(_FRONTEND_A, history_text)
        self.assertNotIn(_FRONTEND_B, history_text)
        self.assertIn(_DYNAMIC, history_text)
        self.assertIn("HISTORY_ORDINARY", history_text)
        self.assertIn("HISTORY_TAIL", history_text)

        self.assertNotIn(_FRONTEND_B, current_text)
        self.assertIn(_UNKNOWN, current_text)
        self.assertIn(_FINGERTIPS, current_text)
        self.assertIn("CURRENT_ORDINARY", result.payload["messages"][2]["content"][0]["text"])

        self.assertEqual(
            result.payload["system"],
            original["system"],
        )
        self.assertEqual(
            result.payload["messages"][2]["content"][2],
            original["messages"][2]["content"][2],
        )
        self.assertEqual(
            result.payload["messages"][1],
            original["messages"][1],
        )

        # The input object is never mutated.
        self.assertEqual(payload, original)

    def test_string_content_form_is_supported(self) -> None:
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "q "
                        + _FRONTEND_A
                        + " "
                        + _FRONTEND_B
                    ),
                }
            ],
        }

        result = rewrite_frontend_memory_attachments(
            payload,
            lambda occurrence, **_: (
                REWRITE_KEEP_EXACT
            ),
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.occurrence_count, 2)

    def test_ordinary_text_with_memory_words_is_never_controlled(
        self,
    ) -> None:
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "memory relevant frontend recall "
                        "relevant_memories.txt"
                    ),
                }
            ],
        }

        result = rewrite_frontend_memory_attachments(
            payload,
            lambda **_: REWRITE_DROP,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.occurrence_count, 0)
        self.assertEqual(
            result.payload, payload
        )

    def test_empty_and_self_closing_are_not_counted(self) -> None:
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        _attachment(
                            "relevant_memories.txt",
                            "   ",
                            "empty_1",
                        )
                        + '<attachment id="sc_1" '
                        'filename="relevant_memories.txt" />'
                    ),
                }
            ],
        }

        result = rewrite_frontend_memory_attachments(
            payload,
            lambda **_: REWRITE_DROP,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.occurrence_count, 0)

    def test_duplicate_identical_attachments_stay_distinct(
        self,
    ) -> None:
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        _FRONTEND_A + " " + _FRONTEND_A
                    ),
                }
            ],
        }

        seen: list[int] = []

        def decide(occurrence, raw_attachment, **_):
            seen.append(occurrence)
            self.assertEqual(
                raw_attachment, _FRONTEND_A
            )
            return REWRITE_KEEP_EXACT

        result = rewrite_frontend_memory_attachments(
            payload, decide
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.occurrence_count, 2)
        self.assertEqual(seen, [0, 1])

    def test_placement_current_vs_history(self) -> None:
        payload = _payload()

        placements: dict[int, str] = {}

        def decide(
            occurrence, placement, raw_attachment, **_
        ):
            placements[occurrence] = placement
            return REWRITE_KEEP_EXACT

        rewrite_frontend_memory_attachments(
            payload, decide
        )

        # A is in the (history) first message; B is in the current
        # (last user) message.
        self.assertEqual(placements[0], "history")
        self.assertEqual(placements[1], "current")

    def test_unknown_action_fails_closed(self) -> None:
        payload = _payload()

        result = rewrite_frontend_memory_attachments(
            payload,
            lambda **_: "summarize",
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.reason, "unknown_rewrite_action"
        )
        self.assertIsNone(result.payload)

    def test_refusal_fails_closed(self) -> None:
        payload = _payload()

        result = rewrite_frontend_memory_attachments(
            payload,
            lambda **_: None,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.reason, "rewrite_refused"
        )
        self.assertIsNone(result.payload)

    def test_raising_callback_fails_closed(self) -> None:
        payload = _payload()

        def decide(**_):
            raise RuntimeError("boom")

        result = rewrite_frontend_memory_attachments(
            payload, decide
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.reason, "rewrite_refused"
        )
        self.assertIsNone(result.payload)

    def test_occurrence_count_mismatch_fails_closed(self) -> None:
        payload = _payload()

        result = rewrite_frontend_memory_attachments(
            payload,
            lambda **_: REWRITE_KEEP_EXACT,
            expected_occurrences=5,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.reason,
            "occurrence_count_mismatch",
        )
        self.assertIsNone(result.payload)

    def test_unsupported_payload_fails_closed(self) -> None:
        for bad in (
            None,
            [],
            {},
            {"messages": "nope"},
            {"messages": [1, 2]},
        ):
            with self.subTest(bad=bad):
                result = (
                    rewrite_frontend_memory_attachments(
                        bad,
                        lambda **_: REWRITE_KEEP_EXACT,
                    )
                )

                self.assertFalse(result.ok)
                self.assertEqual(
                    result.reason,
                    "payload_unsupported",
                )
                self.assertIsNone(result.payload)


if __name__ == "__main__":
    unittest.main()
