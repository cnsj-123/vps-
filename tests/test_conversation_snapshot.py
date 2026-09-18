from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ombrebrain.context.conversation_snapshot import (
    snapshot_status,
    update_conversation_snapshot,
)


CID = "ctx_0123456789abcdef"


def payload(
    messages,
):
    return json.dumps(
        {
            "model": "test-model",
            "messages": messages,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def text_message(
    role,
    text,
):
    return {
        "role": role,
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }


class ConversationSnapshotTests(
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

    def snapshot_file(self):
        return (
            Path(self.temp.name)
            / "snapshots"
            / (
                CID + ".json"
            )
        )

    def test_stores_stable_dialogue(
        self,
    ):
        result = (
            update_conversation_snapshot(
                payload([
                    text_message(
                        "user",
                        "hello",
                    ),
                    text_message(
                        "assistant",
                        "hi there",
                    ),
                    text_message(
                        "user",
                        "next question",
                    ),
                ]),
                conversation_id=CID,
                boundary_prefix_sha256=(
                    "abc"
                ),
            )
        )

        self.assertTrue(
            result["stored"]
        )

        stored = json.loads(
            self.snapshot_file()
            .read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            [
                item["text"]
                for item
                in stored["messages"]
            ],
            [
                "hello",
                "hi there",
                "next question",
            ],
        )

    def test_dynamic_attachment_is_excluded(
        self,
    ):
        dynamic_secret = (
            "DYNAMIC_SECRET_SHOULD_NOT_STORE"
        )

        text = (
            "real user text\n"
            '<attachment '
            'id="message_insert_extra_bundle_123" '
            'filename="Time: now" '
            'type="text/plain" '
            'size="10">'
            + dynamic_secret
            + "</attachment>"
        )

        update_conversation_snapshot(
            payload([
                text_message(
                    "user",
                    text,
                ),
                text_message(
                    "assistant",
                    "stable answer",
                ),
            ]),
            conversation_id=CID,
            boundary_prefix_sha256=(
                "abc"
            ),
        )

        stored_text = (
            self.snapshot_file()
            .read_text(
                encoding="utf-8"
            )
        )

        self.assertIn(
            "real user text",
            stored_text,
        )

        self.assertIn(
            "stable answer",
            stored_text,
        )

        self.assertNotIn(
            dynamic_secret,
            stored_text,
        )

        self.assertNotIn(
            "message_insert_extra_bundle_123",
            stored_text,
        )

    def test_fingertips_is_excluded(
        self,
    ):
        secret = (
            "FINGERTIPS_SECRET"
        )

        text = (
            "normal\n"
            '<attachment '
            'id="random_123" '
            'filename="Fingertips-data.txt" '
            'type="text/plain" '
            'size="20">'
            + secret
            + "</attachment>"
        )

        update_conversation_snapshot(
            payload([
                text_message(
                    "user",
                    text,
                ),
            ]),
            conversation_id=CID,
            boundary_prefix_sha256=(
                "abc"
            ),
        )

        stored_text = (
            self.snapshot_file()
            .read_text(
                encoding="utf-8"
            )
        )

        self.assertIn(
            "normal",
            stored_text,
        )

        self.assertNotIn(
            secret,
            stored_text,
        )

    def test_duplicate_boundary_does_not_increment_revision(
        self,
    ):
        body = payload([
            text_message(
                "user",
                "hello",
            ),
        ])

        first = (
            update_conversation_snapshot(
                body,
                conversation_id=CID,
                boundary_prefix_sha256=(
                    "same"
                ),
            )
        )

        second = (
            update_conversation_snapshot(
                body,
                conversation_id=CID,
                boundary_prefix_sha256=(
                    "same"
                ),
            )
        )

        self.assertEqual(
            first["revision"],
            1,
        )

        self.assertEqual(
            second["revision"],
            1,
        )

        self.assertTrue(
            second["duplicate"]
        )

    def test_status_does_not_return_text(
        self,
    ):
        secret = (
            "PRIVATE_CONVERSATION"
        )

        update_conversation_snapshot(
            payload([
                text_message(
                    "user",
                    secret,
                ),
            ]),
            conversation_id=CID,
            boundary_prefix_sha256=(
                "abc"
            ),
        )

        status = snapshot_status(
            CID
        )

        self.assertTrue(
            status["exists"]
        )

        self.assertNotIn(
            secret,
            json.dumps(
                status,
                ensure_ascii=False,
            ),
        )


if __name__ == "__main__":
    unittest.main()
