from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ombrebrain.context.conversation_trusted_facts import (
    trusted_facts_status,
    update_conversation_trusted_facts,
)


CID = "ctx_0123456789abcdef"


def msg(
    role,
    text,
    source_index,
):
    return {
        "role": role,
        "text": text,
        "source_index":
            source_index,
    }


def compact(
    revision,
    messages,
):
    return {
        "version":
            "conversation-compact.v1",
        "conversation_id":
            CID,
        "source_revision":
            revision,
        "older_context": [],
        "recent_messages":
            messages,
    }


class TrustedFactsTests(
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

    def state_path(self):
        return (
            Path(self.temp.name)
            / "trusted_facts"
            / (CID + ".json")
        )

    def write_compact(
        self,
        payload,
    ):
        p = self.compact_path()

        p.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        p.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def update(
        self,
        revision,
        messages,
    ):
        self.write_compact(
            compact(
                revision,
                messages,
            )
        )

        return (
            update_conversation_trusted_facts(
                CID
            )
        )

    def load(self):
        return json.loads(
            self.state_path()
            .read_text(
                encoding="utf-8"
            )
        )

    def test_explicit_user_fact_is_added(
        self,
    ):
        result = self.update(
            1,
            [
                msg(
                    "user",
                    "事实：测试 GPU 是 RTX 3090",
                    10,
                )
            ],
        )

        state = self.load()

        self.assertEqual(
            result["fact_count"],
            1,
        )

        self.assertEqual(
            state[
                "active_facts"
            ][0]["text"],
            "测试 GPU 是 RTX 3090",
        )

        self.assertEqual(
            state[
                "active_facts"
            ][0]["provenance"],
            "user_explicit",
        )

    def test_generic_statement_is_ignored(
        self,
    ):
        result = self.update(
            1,
            [
                msg(
                    "user",
                    "我今天使用 RTX 3090 做实验",
                    10,
                )
            ],
        )

        self.assertEqual(
            result["fact_count"],
            0,
        )

        self.assertEqual(
            result[
                "commands_this_revision"
            ],
            0,
        )

    def test_assistant_fact_command_is_ignored(
        self,
    ):
        result = self.update(
            1,
            [
                msg(
                    "assistant",
                    "事实：这是 assistant 自己说的",
                    10,
                )
            ],
        )

        self.assertEqual(
            result["fact_count"],
            0,
        )

    def test_confirmation_refreshes_without_duplicate(
        self,
    ):
        self.update(
            1,
            [
                msg(
                    "user",
                    "事实：系统运行在 Linux",
                    10,
                )
            ],
        )

        result = self.update(
            2,
            [
                msg(
                    "user",
                    "确认事实：系统运行在 Linux",
                    20,
                )
            ],
        )

        state = self.load()

        self.assertEqual(
            len(
                state[
                    "active_facts"
                ]
            ),
            1,
        )

        self.assertEqual(
            state[
                "active_facts"
            ][0]["confirmations"],
            2,
        )

        self.assertEqual(
            result[
                "confirmed_this_revision"
            ],
            1,
        )

    def test_revoke_moves_fact_to_history(
        self,
    ):
        self.update(
            1,
            [
                msg(
                    "user",
                    "事实：当前模型是 Test-A",
                    10,
                )
            ],
        )

        result = self.update(
            2,
            [
                msg(
                    "user",
                    "撤销事实：当前模型是 Test-A",
                    20,
                )
            ],
        )

        state = self.load()

        self.assertEqual(
            len(
                state[
                    "active_facts"
                ]
            ),
            0,
        )

        self.assertEqual(
            len(
                state[
                    "history"
                ]
            ),
            1,
        )

        self.assertEqual(
            state[
                "history"
            ][0]["status"],
            "revoked_explicit",
        )

        self.assertEqual(
            result[
                "revoked_this_revision"
            ],
            1,
        )

    def test_unmatched_revoke_is_safe(
        self,
    ):
        self.update(
            1,
            [
                msg(
                    "user",
                    "事实：A",
                    10,
                )
            ],
        )

        result = self.update(
            2,
            [
                msg(
                    "user",
                    "删除事实：B",
                    20,
                )
            ],
        )

        state = self.load()

        self.assertEqual(
            len(
                state[
                    "active_facts"
                ]
            ),
            1,
        )

        self.assertEqual(
            result[
                "unmatched_this_revision"
            ],
            1,
        )

    def test_replace_fact(
        self,
    ):
        self.update(
            1,
            [
                msg(
                    "user",
                    "事实：当前版本是 V1",
                    10,
                )
            ],
        )

        result = self.update(
            2,
            [
                msg(
                    "user",
                    "替换事实：当前版本是 V1 => 当前版本是 V2",
                    20,
                )
            ],
        )

        state = self.load()

        self.assertEqual(
            len(
                state[
                    "active_facts"
                ]
            ),
            1,
        )

        self.assertEqual(
            state[
                "active_facts"
            ][0]["text"],
            "当前版本是 V2",
        )

        self.assertEqual(
            state[
                "history"
            ][0]["status"],
            "superseded_explicit",
        )

        self.assertEqual(
            result[
                "replaced_this_revision"
            ],
            1,
        )

    def test_reassert_after_revoke_is_allowed(
        self,
    ):
        self.update(
            1,
            [
                msg(
                    "user",
                    "事实：A",
                    10,
                )
            ],
        )

        self.update(
            2,
            [
                msg(
                    "user",
                    "撤销事实：A",
                    20,
                )
            ],
        )

        self.update(
            3,
            [
                msg(
                    "user",
                    "事实：A",
                    30,
                )
            ],
        )

        state = self.load()

        self.assertEqual(
            len(
                state[
                    "active_facts"
                ]
            ),
            1,
        )

        self.assertEqual(
            state[
                "active_facts"
            ][0][
                "first_source_index"
            ],
            30,
        )

    def test_cjk_spaced_command_is_recognized(
        self,
    ):
        result = self.update(
            1,
            [
                msg(
                    "user",
                    "事 实 ： 测 试 模 式 为 Shadow",
                    10,
                )
            ],
        )

        state = self.load()

        self.assertEqual(
            result["fact_count"],
            1,
        )

        self.assertEqual(
            state[
                "active_facts"
            ][0]["text"],
            "测试模式为 Shadow",
        )

    def test_duplicate_revision_is_idempotent(
        self,
    ):
        payload = [
            msg(
                "user",
                "事实：A",
                10,
            )
        ]

        first = self.update(
            1,
            payload,
        )

        second = (
            update_conversation_trusted_facts(
                CID
            )
        )

        self.assertFalse(
            first["duplicate"]
        )

        self.assertTrue(
            second["duplicate"]
        )

        self.assertEqual(
            first["revision"],
            second["revision"],
        )

    def test_status_exposes_no_fact_text(
        self,
    ):
        secret = (
            "PRIVATE_TRUSTED_FACT_VALUE"
        )

        self.update(
            1,
            [
                msg(
                    "user",
                    "事实：" + secret,
                    10,
                )
            ],
        )

        status = trusted_facts_status(
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

        self.assertFalse(
            status[
                "telemetry"
            ][
                "inference_enabled"
            ]
        )


if __name__ == "__main__":
    unittest.main()
