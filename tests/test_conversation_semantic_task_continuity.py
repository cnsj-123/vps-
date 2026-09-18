from __future__ import annotations

import unittest

from ombrebrain.context.conversation_semantic import (
    build_semantic_frame,
)


CID = "ctx_0123456789abcdef"


def msg(
    index,
    role,
    text,
):
    return {
        "role": role,
        "text": text,
        "source_index": index,
    }


def compact(
    messages,
    revision=2,
):
    return {
        "version":
            "conversation-compact.v1",
        "conversation_id": CID,
        "source_revision": revision,
        "older_context": [],
        "recent_messages":
            messages,
    }


def previous_task(
    text="继续完善 OB Semantic Context",
    source_index=100,
):
    return {
        "version":
            "conversation-semantic.v1",
        "conversation_id": CID,
        "source_revision": 1,
        "current_task": {
            "text": text,
            "source_index":
                source_index,
        },
    }


class SemanticTaskContinuityTests(
    unittest.TestCase
):

    def test_ack_does_not_replace_task(
        self,
    ):
        frame = build_semantic_frame(
            compact([
                msg(
                    102,
                    "user",
                    "好了",
                ),
            ]),
            previous_frame=previous_task(),
        )

        self.assertEqual(
            frame[
                "current_task"
            ]["text"],
            "继续完善 OB Semantic Context",
        )

        self.assertEqual(
            frame[
                "current_task"
            ]["source_index"],
            100,
        )

        self.assertTrue(
            frame[
                "telemetry"
            ][
                "current_task_carried_forward"
            ]
        )

        self.assertTrue(
            frame[
                "telemetry"
            ][
                "latest_user_ack_only"
            ]
        )

    def test_continue_does_not_replace_task(
        self,
    ):
        frame = build_semantic_frame(
            compact([
                msg(
                    102,
                    "user",
                    "继续",
                ),
            ]),
            previous_frame=previous_task(),
        )

        self.assertEqual(
            frame[
                "current_task"
            ]["source_index"],
            100,
        )

        self.assertTrue(
            frame[
                "telemetry"
            ][
                "current_task_carried_forward"
            ]
        )

    def test_substantive_turn_replaces_task(
        self,
    ):
        frame = build_semantic_frame(
            compact([
                msg(
                    102,
                    "user",
                    "现在给 Semantic 层加入任务连续性。",
                ),
            ]),
            previous_frame=previous_task(),
        )

        self.assertEqual(
            frame[
                "current_task"
            ]["text"],
            "现在给 Semantic 层加入任务连续性。",
        )

        self.assertEqual(
            frame[
                "current_task"
            ]["source_index"],
            102,
        )

        self.assertFalse(
            frame[
                "telemetry"
            ][
                "current_task_carried_forward"
            ]
        )

        self.assertFalse(
            frame[
                "telemetry"
            ][
                "latest_user_ack_only"
            ]
        )

    def test_ack_without_previous_task_creates_no_fake_task(
        self,
    ):
        frame = build_semantic_frame(
            compact([
                msg(
                    1,
                    "user",
                    "ok",
                ),
            ]),
        )

        self.assertIsNone(
            frame[
                "current_task"
            ]
        )

        self.assertTrue(
            frame[
                "telemetry"
            ][
                "latest_user_ack_only"
            ]
        )


if __name__ == "__main__":
    unittest.main()
