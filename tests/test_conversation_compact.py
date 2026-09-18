from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ombrebrain.context.conversation_compact import (
    build_compact_payload,
    compact_status,
    update_conversation_compact,
)


CID = "ctx_0123456789abcdef"


def message(
    index,
    role,
    text,
):
    return {
        "role": role,
        "text": text,
        "source_index": index,
    }


def snapshot(
    messages,
    revision=1,
):
    return {
        "version":
            "conversation-snapshot.v1",
        "conversation_id": CID,
        "revision": revision,
        "updated_at":
            "2026-09-18T00:00:00Z",
        "messages": messages,
    }


class ConversationCompactTests(
    unittest.TestCase
):

    def setUp(self):
        self.temp = (
            tempfile.TemporaryDirectory()
        )

        self.old_root = os.environ.get(
            "OMBRE_CONTEXT_STATE_DIR"
        )

        os.environ[
            "OMBRE_CONTEXT_STATE_DIR"
        ] = self.temp.name

    def tearDown(self):
        if self.old_root is None:
            os.environ.pop(
                "OMBRE_CONTEXT_STATE_DIR",
                None,
            )
        else:
            os.environ[
                "OMBRE_CONTEXT_STATE_DIR"
            ] = self.old_root

        self.temp.cleanup()

    def snapshot_path(self):
        return (
            Path(self.temp.name)
            / "snapshots"
            / (CID + ".json")
        )

    def compact_path(self):
        return (
            Path(self.temp.name)
            / "compact"
            / (CID + ".json")
        )

    def write_snapshot(
        self,
        payload,
    ):
        path = self.snapshot_path()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_recent_six_preserved(
        self,
    ):
        messages = [
            message(
                i,
                "user"
                if i % 2 == 0
                else "assistant",
                "message-%d" % i,
            )
            for i in range(12)
        ]

        compact = (
            build_compact_payload(
                snapshot(messages)
            )
        )

        recent = compact[
            "recent_messages"
        ]

        self.assertEqual(
            len(recent),
            6,
        )

        self.assertEqual(
            [
                x["text"]
                for x in recent
            ],
            [
                "message-6",
                "message-7",
                "message-8",
                "message-9",
                "message-10",
                "message-11",
            ],
        )

    def test_older_messages_are_clipped(
        self,
    ):
        long_text = (
            "A" * 1000
        )

        messages = [
            message(
                i,
                "user",
                long_text,
            )
            for i in range(12)
        ]

        compact = (
            build_compact_payload(
                snapshot(messages)
            )
        )

        older = compact[
            "older_context"
        ]

        self.assertEqual(
            len(older),
            6,
        )

        self.assertTrue(
            all(
                len(x["excerpt"])
                <= 180
                for x in older
            )
        )

        telemetry = compact[
            "telemetry"
        ]

        self.assertEqual(
            telemetry[
                "older_clipped"
            ],
            6,
        )

        self.assertTrue(
            telemetry[
                "compaction_applied"
            ]
        )

        self.assertLess(
            telemetry[
                "compaction_ratio"
            ],
            1.0,
        )

    def test_duplicate_source_revision_is_idempotent(
        self,
    ):
        messages = [
            message(
                0,
                "user",
                "hello",
            ),
            message(
                1,
                "assistant",
                "hi",
            ),
        ]

        self.write_snapshot(
            snapshot(
                messages,
                revision=3,
            )
        )

        first = (
            update_conversation_compact(
                CID
            )
        )

        second = (
            update_conversation_compact(
                CID
            )
        )

        self.assertTrue(
            first["stored"]
        )

        self.assertFalse(
            first["duplicate"]
        )

        self.assertTrue(
            second["duplicate"]
        )

        self.assertEqual(
            second[
                "source_revision"
            ],
            3,
        )

    def test_new_snapshot_revision_updates_compact(
        self,
    ):
        self.write_snapshot(
            snapshot(
                [
                    message(
                        0,
                        "user",
                        "one",
                    )
                ],
                revision=1,
            )
        )

        first = (
            update_conversation_compact(
                CID
            )
        )

        self.write_snapshot(
            snapshot(
                [
                    message(
                        0,
                        "user",
                        "one",
                    ),
                    message(
                        1,
                        "assistant",
                        "two",
                    ),
                ],
                revision=2,
            )
        )

        second = (
            update_conversation_compact(
                CID
            )
        )

        self.assertEqual(
            first[
                "source_revision"
            ],
            1,
        )

        self.assertEqual(
            second[
                "source_revision"
            ],
            2,
        )

        self.assertFalse(
            second["duplicate"]
        )

    def test_status_contains_no_text(
        self,
    ):
        secret = (
            "DO_NOT_EXPOSE_THIS_TEXT"
        )

        self.write_snapshot(
            snapshot(
                [
                    message(
                        0,
                        "user",
                        secret,
                    )
                ],
                revision=1,
            )
        )

        update_conversation_compact(
            CID
        )

        status = compact_status(
            CID
        )

        rendered = json.dumps(
            status,
            ensure_ascii=False,
        )

        self.assertTrue(
            status["exists"]
        )

        self.assertNotIn(
            secret,
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
