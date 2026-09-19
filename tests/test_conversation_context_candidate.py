from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ombrebrain.context.conversation_context_candidate import (
    build_context_candidate,
    context_candidate_status,
    update_conversation_context_candidate,
)


CID = "ctx_0123456789abcdef"


def compact(
    revision,
    messages,
):
    return {
        "conversation_id":
            CID,
        "source_revision":
            revision,
        "recent_messages":
            messages,
    }


def message(
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


def semantic(
    revision,
    *,
    task=None,
    constraints=None,
    decisions=None,
    open_items=None,
):
    return {
        "version":
            "conversation-semantic-state.v2",
        "conversation_id":
            CID,
        "source_revision":
            revision,
        "current_task":
            task,
        "constraints":
            constraints or [],
        "decisions":
            decisions or [],
        "open_items":
            open_items or [],
    }


def facts(
    revision,
    items,
):
    return {
        "version":
            "conversation-trusted-facts.v1",
        "conversation_id":
            CID,
        "source_revision":
            revision,
        "active_facts":
            items,
    }


class ContextCandidateTests(
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

    def write(
        self,
        kind,
        payload,
    ):
        p = (
            Path(self.temp.name)
            / kind
            / (CID + ".json")
        )

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

    def test_current_user_is_not_duplicated(
        self,
    ):
        c = compact(
            3,
            [
                message(
                    "assistant",
                    "previous answer",
                    20,
                ),
                message(
                    "user",
                    "current request",
                    21,
                ),
            ],
        )

        s = semantic(
            3,
            task={
                "text":
                    "current request",
                "source_index":
                    21,
            },
        )

        result = build_context_candidate(
            conversation_id=CID,
            compact=c,
            semantic_state=s,
            trusted_facts=None,
        )

        self.assertIsNone(
            result[
                "sections"
            ][
                "current_task"
            ]
        )

        rendered = json.dumps(
            result[
                "sections"
            ],
            ensure_ascii=False,
        )

        self.assertNotIn(
            "current request",
            rendered,
        )

        self.assertTrue(
            result[
                "telemetry"
            ][
                "current_user_excluded"
            ]
        )

    def test_carried_task_is_included(
        self,
    ):
        c = compact(
            4,
            [
                message(
                    "user",
                    "继续",
                    30,
                )
            ],
        )

        s = semantic(
            4,
            task={
                "text":
                    "继续完成 Context Assembly",
                "source_index":
                    28,
            },
        )

        result = build_context_candidate(
            conversation_id=CID,
            compact=c,
            semantic_state=s,
            trusted_facts=None,
        )

        self.assertEqual(
            result[
                "sections"
            ][
                "current_task"
            ][
                "text"
            ],
            "继续完成 Context Assembly",
        )

    def test_trusted_fact_is_included(
        self,
    ):
        c = compact(
            5,
            [
                message(
                    "user",
                    "继续",
                    40,
                )
            ],
        )

        f = facts(
            5,
            [
                {
                    "id":
                        "fact_a",
                    "status":
                        "active",
                    "text":
                        "部署模式为 Shadow",
                    "provenance":
                        "user_explicit",
                    "last_source_index":
                        39,
                }
            ],
        )

        result = build_context_candidate(
            conversation_id=CID,
            compact=c,
            semantic_state=None,
            trusted_facts=f,
        )

        self.assertEqual(
            result[
                "sections"
            ][
                "trusted_facts"
            ][0]["text"],
            "部署模式为 Shadow",
        )

    def test_only_active_semantic_items_are_used(
        self,
    ):
        c = compact(
            6,
            [
                message(
                    "user",
                    "继续",
                    50,
                )
            ],
        )

        s = semantic(
            6,
            constraints=[
                {
                    "id":
                        "a",
                    "status":
                        "active",
                    "text":
                        "必须先测试",
                    "last_source_index":
                        45,
                },
                {
                    "id":
                        "b",
                    "status":
                        "cancelled_explicit",
                    "text":
                        "必须保留旧规则",
                    "last_source_index":
                        44,
                },
            ],
        )

        result = build_context_candidate(
            conversation_id=CID,
            compact=c,
            semantic_state=s,
            trusted_facts=None,
        )

        rendered = json.dumps(
            result[
                "sections"
            ],
            ensure_ascii=False,
        )

        self.assertIn(
            "必须先测试",
            rendered,
        )

        self.assertNotIn(
            "必须保留旧规则",
            rendered,
        )

    def test_control_commands_are_not_recent_context(
        self,
    ):
        c = compact(
            7,
            [
                message(
                    "user",
                    "事实：A",
                    60,
                ),
                message(
                    "assistant",
                    "ok",
                    61,
                ),
                message(
                    "user",
                    "继续",
                    62,
                ),
            ],
        )

        result = build_context_candidate(
            conversation_id=CID,
            compact=c,
            semantic_state=None,
            trusted_facts=None,
        )

        rendered = json.dumps(
            result[
                "sections"
            ][
                "recent_context"
            ],
            ensure_ascii=False,
        )

        self.assertNotIn(
            "事实：A",
            rendered,
        )

        self.assertEqual(
            result[
                "telemetry"
            ][
                "control_rejected"
            ],
            1,
        )

    def test_cross_section_exact_dedup(
        self,
    ):
        c = compact(
            8,
            [
                message(
                    "assistant",
                    "部署模式为 Shadow",
                    70,
                ),
                message(
                    "user",
                    "继续",
                    71,
                ),
            ],
        )

        f = facts(
            8,
            [
                {
                    "id":
                        "fact_a",
                    "status":
                        "active",
                    "text":
                        "部署模式为 Shadow",
                    "last_source_index":
                        69,
                }
            ],
        )

        result = build_context_candidate(
            conversation_id=CID,
            compact=c,
            semantic_state=None,
            trusted_facts=f,
        )

        rendered = json.dumps(
            result[
                "sections"
            ],
            ensure_ascii=False,
        )

        self.assertEqual(
            rendered.count(
                "部署模式为 Shadow"
            ),
            1,
        )

        self.assertGreaterEqual(
            result[
                "telemetry"
            ][
                "dedup_rejected"
            ],
            1,
        )

    def test_budget_is_hard_cap(
        self,
    ):
        c = compact(
            9,
            [
                message(
                    "assistant",
                    "A" * 180,
                    80,
                ),
                message(
                    "assistant",
                    "B" * 180,
                    81,
                ),
                message(
                    "user",
                    "继续",
                    82,
                ),
            ],
        )

        f = facts(
            9,
            [
                {
                    "id":
                        "fact_a",
                    "status":
                        "active",
                    "text":
                        "C" * 180,
                    "last_source_index":
                        79,
                }
            ],
        )

        result = build_context_candidate(
            conversation_id=CID,
            compact=c,
            semantic_state=None,
            trusted_facts=f,
            token_budget=70,
        )

        telemetry = result[
            "telemetry"
        ]

        self.assertLessEqual(
            telemetry[
                "estimated_tokens"
            ],
            70,
        )

        self.assertTrue(
            telemetry[
                "truncated"
            ]
        )

        self.assertGreater(
            telemetry[
                "budget_rejected"
            ],
            0,
        )

    def test_source_ahead_is_not_used(
        self,
    ):
        c = compact(
            10,
            [
                message(
                    "user",
                    "继续",
                    90,
                )
            ],
        )

        f = facts(
            11,
            [
                {
                    "id":
                        "future",
                    "status":
                        "active",
                    "text":
                        "未来事实",
                    "last_source_index":
                        99,
                }
            ],
        )

        result = build_context_candidate(
            conversation_id=CID,
            compact=c,
            semantic_state=None,
            trusted_facts=f,
        )

        self.assertEqual(
            result[
                "sections"
            ][
                "trusted_facts"
            ],
            [],
        )

        self.assertTrue(
            result[
                "telemetry"
            ][
                "trusted_facts_source_ahead"
            ]
        )

    def test_update_is_idempotent(
        self,
    ):
        self.write(
            "compact",
            compact(
                12,
                [
                    message(
                        "user",
                        "继续",
                        100,
                    )
                ],
            ),
        )

        self.write(
            "semantic_state",
            semantic(
                12,
                task={
                    "text":
                        "旧任务",
                    "source_index":
                        98,
                },
            ),
        )

        first = (
            update_conversation_context_candidate(
                CID
            )
        )

        second = (
            update_conversation_context_candidate(
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

    def test_status_does_not_expose_text(
        self,
    ):
        secret = (
            "PRIVATE_CONTEXT_CANDIDATE_VALUE"
        )

        self.write(
            "compact",
            compact(
                13,
                [
                    message(
                        "user",
                        "继续",
                        110,
                    )
                ],
            ),
        )

        self.write(
            "trusted_facts",
            facts(
                13,
                [
                    {
                        "id":
                            "secret",
                        "status":
                            "active",
                        "text":
                            secret,
                        "last_source_index":
                            109,
                    }
                ],
            ),
        )

        update_conversation_context_candidate(
            CID
        )

        status = context_candidate_status(
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
