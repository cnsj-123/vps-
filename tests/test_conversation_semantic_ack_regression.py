from __future__ import annotations

import unittest

from ombrebrain.context.conversation_semantic import (
    _is_ack_or_continuation,
    build_semantic_frame,
)


CID = "ctx_0123456789abcdef"


def msg(index, role, text):
    return {
        "role": role,
        "text": text,
        "source_index": index,
    }


def compact(messages, revision=5):
    return {
        "version": "conversation-compact.v1",
        "conversation_id": CID,
        "source_revision": revision,
        "older_context": [],
        "recent_messages": messages,
    }


class SemanticAckRegressionTests(unittest.TestCase):

    def test_ack_variants_are_detected(self):
        for value in (
            "好",
            "好的",
            "好了",
            "继续",
            "收到",
            "ok",
            "OK",
            "okay",
            "来",
            "开始吧",
        ):
            with self.subTest(value=value):
                self.assertTrue(
                    _is_ack_or_continuation(value)
                )

    def test_real_task_is_not_ack(self):
        self.assertFalse(
            _is_ack_or_continuation(
                "继续完善 Semantic Context"
            )
        )

    def test_recovers_from_bad_previous_ack_task(self):
        frame = build_semantic_frame(
            compact([
                msg(
                    156,
                    "user",
                    "下一步继续完善 OB 的 Semantic Context，但必须继续保持 Shadow 模式。",
                ),
                msg(
                    157,
                    "assistant",
                    "好的。",
                ),
                msg(
                    158,
                    "user",
                    "好了",
                ),
            ]),
            previous_frame={
                "version": "conversation-semantic.v2",
                "conversation_id": CID,
                "source_revision": 4,
                "current_task": {
                    "text": "好了",
                    "source_index": 158,
                },
            },
        )

        task = frame["current_task"]
        telemetry = frame["telemetry"]

        self.assertEqual(
            task["source_index"],
            156,
        )

        self.assertTrue(
            telemetry["latest_user_ack_only"]
        )

        self.assertTrue(
            telemetry["current_task_carried_forward"]
        )

        self.assertTrue(
            telemetry["task_recovered_from_history"]
        )


if __name__ == "__main__":
    unittest.main()
