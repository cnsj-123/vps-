from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ombrebrain.context.conversation_semantic import (
    build_semantic_frame,
    semantic_status,
    update_conversation_semantic,
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
    revision=1,
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


class ConversationSemanticTests(
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

    def compact_path(self):
        return (
            Path(self.temp.name)
            / "compact"
            / (CID + ".json")
        )

    def semantic_path(self):
        return (
            Path(self.temp.name)
            / "semantic"
            / (CID + ".json")
        )

    def write_compact(
        self,
        payload,
    ):
        path = self.compact_path()

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

    def test_latest_user_becomes_current_task(
        self,
    ):
        frame = build_semantic_frame(
            compact([
                msg(
                    0,
                    "user",
                    "old task",
                ),
                msg(
                    1,
                    "assistant",
                    "done",
                ),
                msg(
                    2,
                    "user",
                    "现在继续做上下文压缩",
                ),
            ])
        )

        self.assertEqual(
            frame[
                "current_task"
            ]["text"],
            "现在继续做上下文压缩",
        )

        self.assertEqual(
            frame[
                "current_task"
            ]["source_index"],
            2,
        )

    def test_explicit_constraints_only(
        self,
    ):
        frame = build_semantic_frame(
            compact([
                msg(
                    0,
                    "user",
                    "必须保持1h缓存，不要改成5m。",
                ),
                msg(
                    1,
                    "user",
                    "今天聊聊天气。",
                ),
            ])
        )

        texts = [
            x["text"]
            for x in frame[
                "constraints"
            ]
        ]

        rendered = " ".join(
            texts
        )

        self.assertIn(
            "必须保持1h缓存",
            rendered,
        )

        self.assertNotIn(
            "天气",
            rendered,
        )

    def test_explicit_decision_from_assistant(
        self,
    ):
        frame = build_semantic_frame(
            compact([
                msg(
                    0,
                    "assistant",
                    "我们决定先使用 Shadow 模式。",
                ),
                msg(
                    1,
                    "assistant",
                    "普通说明文字。",
                ),
            ])
        )

        self.assertEqual(
            len(
                frame[
                    "decisions"
                ]
            ),
            1,
        )

        self.assertIn(
            "决定",
            frame[
                "decisions"
            ][0]["text"],
        )

    def test_questions_become_open_items(
        self,
    ):
        frame = build_semantic_frame(
            compact([
                msg(
                    0,
                    "user",
                    "下一步怎么把它接进 Gateway？",
                ),
                msg(
                    1,
                    "assistant",
                    "later",
                ),
            ])
        )

        self.assertEqual(
            len(
                frame[
                    "open_items"
                ]
            ),
            1,
        )

    def test_facts_are_not_guessed(
        self,
    ):
        frame = build_semantic_frame(
            compact([
                msg(
                    0,
                    "user",
                    "服务器使用Linux。",
                ),
            ])
        )

        self.assertEqual(
            frame[
                "established_facts"
            ],
            [],
        )

        self.assertTrue(
            frame[
                "telemetry"
            ][
                "facts_deferred"
            ]
        )

    def test_duplicate_revision_is_idempotent(
        self,
    ):
        self.write_compact(
            compact(
                [
                    msg(
                        0,
                        "user",
                        "hello",
                    )
                ],
                revision=5,
            )
        )

        first = (
            update_conversation_semantic(
                CID
            )
        )

        second = (
            update_conversation_semantic(
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
            5,
        )

    def test_status_contains_no_semantic_text(
        self,
    ):
        secret = (
            "PRIVATE_SEMANTIC_TEXT"
        )

        self.write_compact(
            compact([
                msg(
                    0,
                    "user",
                    secret,
                )
            ])
        )

        update_conversation_semantic(
            CID
        )

        status = semantic_status(
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
