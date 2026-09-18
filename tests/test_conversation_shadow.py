from __future__ import annotations

import json
import os
import tempfile
import unittest

from ombrebrain.context.conversation_shadow import (
    conversation_shadow_status,
    observe_conversation_shadow,
)


def message(
    role,
    text,
    *,
    cache=False,
):
    block = {
        "type": "text",
        "text": text,
    }

    if cache:
        block["cache_control"] = {
            "type": "ephemeral",
            "ttl": "1h",
        }

    return {
        "role": role,
        "content": [block],
    }


def body(messages):
    return json.dumps(
        {
            "model": "test-model",
            "messages": messages,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class ConversationShadowTests(
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

    def test_append_only_turns_share_identity(
        self,
    ):
        first = body([
            message("user", "u0"),
            message("assistant", "a0"),
            message("user", "u1"),
            message(
                "assistant",
                "a1",
                cache=True,
            ),
            message("user", "u2"),
        ])

        second = body([
            message("user", "u0"),
            message("assistant", "a0"),
            message("user", "u1"),
            message("assistant", "a1"),
            message("user", "u2"),
            message(
                "assistant",
                "a2",
                cache=True,
            ),
            message("user", "u3"),
        ])

        a = observe_conversation_shadow(
            first
        )
        b = observe_conversation_shadow(
            second
        )

        self.assertTrue(
            a["new_conversation"]
        )
        self.assertEqual(
            a["round"],
            1,
        )

        self.assertEqual(
            a["conversation_id"],
            b["conversation_id"],
        )

        self.assertEqual(
            b["continuity"],
            "parent_match",
        )

        self.assertEqual(
            b["round"],
            2,
        )

    def test_duplicate_does_not_increment(
        self,
    ):
        payload = body([
            message("user", "u0"),
            message("assistant", "a0"),
            message("user", "u1"),
            message(
                "assistant",
                "a1",
                cache=True,
            ),
            message("user", "u2"),
        ])

        a = observe_conversation_shadow(
            payload
        )
        b = observe_conversation_shadow(
            payload
        )

        self.assertEqual(
            a["conversation_id"],
            b["conversation_id"],
        )

        self.assertTrue(
            b["duplicate"]
        )

        self.assertEqual(
            b["round"],
            1,
        )

    def test_unrelated_history_is_new_conversation(
        self,
    ):
        a = observe_conversation_shadow(
            body([
                message("user", "alpha"),
                message("assistant", "one"),
                message("user", "next"),
                message(
                    "assistant",
                    "two",
                    cache=True,
                ),
                message("user", "now"),
            ])
        )

        b = observe_conversation_shadow(
            body([
                message("user", "beta"),
                message("assistant", "other"),
                message("user", "next"),
                message(
                    "assistant",
                    "answer",
                    cache=True,
                ),
                message("user", "now"),
            ])
        )

        self.assertNotEqual(
            a["conversation_id"],
            b["conversation_id"],
        )

    def test_store_contains_no_prompt_text(
        self,
    ):
        secret = (
            "THIS_MUST_NOT_BE_STORED"
        )

        observe_conversation_shadow(
            body([
                message(
                    "user",
                    secret,
                ),
                message(
                    "assistant",
                    "private-answer",
                ),
                message(
                    "user",
                    "more-private-text",
                ),
                message(
                    "assistant",
                    "stable",
                    cache=True,
                ),
                message(
                    "user",
                    "current",
                ),
            ])
        )

        files = []

        for root, _, names in os.walk(
            self.temp.name
        ):
            for name in names:
                path = os.path.join(
                    root,
                    name,
                )
                files.append(path)

        self.assertTrue(files)

        stored = "".join(
            open(
                path,
                "r",
                encoding="utf-8",
            ).read()
            for path in files
        )

        self.assertNotIn(
            secret,
            stored,
        )
        self.assertNotIn(
            "private-answer",
            stored,
        )
        self.assertNotIn(
            "more-private-text",
            stored,
        )

        status = (
            conversation_shadow_status()
        )

        self.assertEqual(
            status["conversation_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
