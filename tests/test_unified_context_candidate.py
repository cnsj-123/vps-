from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ombrebrain.context.unified_context_candidate import (
    build_unified_context_candidate,
    unified_context_candidate_status,
    update_unified_context_candidate,
)


CID = "ctx_0123456789abcdef"


def conversation_candidate(
    *,
    task=None,
    facts=None,
    constraints=None,
    decisions=None,
    open_items=None,
    recent=None,
):
    return {
        "version":
            "conversation-context-candidate.v1",
        "conversation_id":
            CID,
        "revision":
            7,
        "source_revision":
            20,
        "sections": {
            "current_task":
                task,
            "trusted_facts":
                facts or [],
            "constraints":
                constraints or [],
            "decisions":
                decisions or [],
            "open_items":
                open_items or [],
            "recent_context":
                recent or [],
        },
        "telemetry": {
            "current_user_excluded":
                True,
        },
    }


def source_candidates(
    *,
    state=None,
    plans=None,
    memories=None,
):
    return {
        "state":
            state or {},
        "state_revision":
            4,
        "plans":
            plans or [],
        "memories":
            memories or [],
        "telemetry": {
            "retrieval_candidate_count":
                len(memories or []),
            "relevance_rejected":
                0,
            "anti_echo":
                {},
            "dedup":
                {},
        },
    }


class UnifiedContextCandidateTests(
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

    def test_unifies_all_sources(
        self,
    ):
        conv = conversation_candidate(
            task={
                "text":
                    "继续统一上下文",
                "source_index":
                    18,
            },
            facts=[
                {
                    "id":
                        "fact_a",
                    "text":
                        "部署模式为 Shadow",
                }
            ],
            constraints=[
                {
                    "id":
                        "constraint_a",
                    "text":
                        "必须先测试",
                }
            ],
            recent=[
                {
                    "role":
                        "assistant",
                    "text":
                        "上一轮回答",
                    "source_index":
                        19,
                }
            ],
        )

        sources = source_candidates(
            state={
                "relationship": {
                    "trust": 0.5,
                },
                "current_focus":
                    "Context Assembly",
                "last_seen":
                    "2026-09-19T00:00:00Z",
            },
            plans=[
                {
                    "id":
                        "plan_a",
                    "name":
                        "Context Roadmap",
                    "content":
                        "先 Shadow 再 Injection",
                    "status":
                        "active",
                    "weight":
                        0.9,
                }
            ],
            memories=[
                {
                    "id":
                        "memory_a",
                    "content":
                        "过去已完成 Gateway Cache",
                    "context_relevance":
                        0.91,
                }
            ],
        )

        result = (
            build_unified_context_candidate(
                conversation_id=CID,
                conversation_candidate=
                    conv,
                context_candidates=
                    sources,
            )
        )

        sections = result[
            "sections"
        ]

        self.assertIsNotNone(
            sections[
                "current_task"
            ]
        )

        self.assertEqual(
            len(
                sections[
                    "trusted_facts"
                ]
            ),
            1,
        )

        self.assertEqual(
            len(
                sections[
                    "constraints"
                ]
            ),
            1,
        )

        self.assertIsNotNone(
            sections[
                "state"
            ]
        )

        self.assertEqual(
            len(
                sections[
                    "plans"
                ]
            ),
            1,
        )

        self.assertEqual(
            len(
                sections[
                    "memories"
                ]
            ),
            1,
        )

        self.assertEqual(
            len(
                sections[
                    "recent_context"
                ]
            ),
            1,
        )

    def test_single_budget_is_hard_cap(
        self,
    ):
        conv = conversation_candidate(
            facts=[
                {
                    "text":
                        "A" * 300,
                }
            ],
            recent=[
                {
                    "role":
                        "assistant",
                    "text":
                        "B" * 300,
                }
            ],
        )

        sources = source_candidates(
            plans=[
                {
                    "id":
                        "p",
                    "name":
                        "large plan",
                    "content":
                        "C" * 300,
                    "status":
                        "active",
                    "weight":
                        1.0,
                }
            ],
            memories=[
                {
                    "id":
                        "m",
                    "content":
                        "D" * 300,
                    "context_relevance":
                        1.0,
                }
            ],
        )

        result = (
            build_unified_context_candidate(
                conversation_id=CID,
                conversation_candidate=
                    conv,
                context_candidates=
                    sources,
                token_budget=150,
            )
        )

        telemetry = result[
            "telemetry"
        ]

        self.assertLessEqual(
            telemetry[
                "estimated_tokens"
            ],
            150,
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

    def test_high_priority_fact_beats_recent_context(
        self,
    ):
        same = "优先保留这条内容"

        conv = conversation_candidate(
            facts=[
                {
                    "id":
                        "fact",
                    "text":
                        same,
                }
            ],
            recent=[
                {
                    "role":
                        "assistant",
                    "text":
                        same,
                }
            ],
        )

        result = (
            build_unified_context_candidate(
                conversation_id=CID,
                conversation_candidate=
                    conv,
                context_candidates=
                    source_candidates(),
            )
        )

        self.assertEqual(
            len(
                result[
                    "sections"
                ][
                    "trusted_facts"
                ]
            ),
            1,
        )

        self.assertEqual(
            result[
                "sections"
            ][
                "recent_context"
            ],
            [],
        )

        self.assertGreaterEqual(
            result[
                "telemetry"
            ][
                "dedup_rejected"
            ],
            1,
        )

    def test_memory_duplicate_of_fact_is_rejected(
        self,
    ):
        same = "统一事实内容"

        conv = conversation_candidate(
            facts=[
                {
                    "text":
                        same,
                }
            ],
        )

        sources = source_candidates(
            memories=[
                {
                    "id":
                        "m",
                    "content":
                        same,
                    "context_relevance":
                        0.99,
                }
            ],
        )

        result = (
            build_unified_context_candidate(
                conversation_id=CID,
                conversation_candidate=
                    conv,
                context_candidates=
                    sources,
            )
        )

        self.assertEqual(
            result[
                "sections"
            ][
                "memories"
            ],
            [],
        )

    def test_state_focus_duplicate_is_removed(
        self,
    ):
        focus = "继续 Context Assembly"

        conv = conversation_candidate(
            task={
                "text":
                    focus,
            }
        )

        sources = source_candidates(
            state={
                "relationship": {
                    "trust": 0.6,
                },
                "current_focus":
                    focus,
            }
        )

        result = (
            build_unified_context_candidate(
                conversation_id=CID,
                conversation_candidate=
                    conv,
                context_candidates=
                    sources,
            )
        )

        state = result[
            "sections"
        ][
            "state"
        ]

        self.assertIsNotNone(
            state
        )

        self.assertNotIn(
            "current_focus",
            state,
        )

        self.assertIn(
            "relationship",
            state,
        )

    def test_update_is_idempotent(
        self,
    ):
        root = Path(
            self.temp.name
        )

        p = (
            root
            / "context_candidate"
            / (CID + ".json")
        )

        p.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        p.write_text(
            json.dumps(
                conversation_candidate(
                    facts=[
                        {
                            "text":
                                "Shadow Only",
                        }
                    ],
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        sources = source_candidates(
            state={
                "current_focus":
                    "Context",
            }
        )

        first = (
            update_unified_context_candidate(
                conversation_id=CID,
                context_candidates=
                    sources,
            )
        )

        second = (
            update_unified_context_candidate(
                conversation_id=CID,
                context_candidates=
                    sources,
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

        # Duplicate fast-path must preserve privacy-safe telemetry.
        self.assertEqual(
            second["token_budget"],
            first["token_budget"],
        )

        self.assertEqual(
            second["memory_count"],
            first["memory_count"],
        )

        self.assertEqual(
            second["current_user_excluded"],
            first["current_user_excluded"],
        )

    def test_status_does_not_expose_text(
        self,
    ):
        secret = (
            "PRIVATE_UNIFIED_CONTEXT_VALUE"
        )

        root = Path(
            self.temp.name
        )

        p = (
            root
            / "context_candidate"
            / (CID + ".json")
        )

        p.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        p.write_text(
            json.dumps(
                conversation_candidate(
                    facts=[
                        {
                            "text":
                                secret,
                        }
                    ],
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        update_unified_context_candidate(
            conversation_id=CID,
            context_candidates=
                source_candidates(),
        )

        status = (
            unified_context_candidate_status(
                CID
            )
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
