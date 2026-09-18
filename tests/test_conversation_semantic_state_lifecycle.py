from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ombrebrain.context.conversation_semantic_state import (
    update_semantic_state,
)


CID = "ctx_0123456789abcdef"


def item(text, source_index):
    return {
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
            "conversation-semantic.v3",
        "conversation_id": CID,
        "source_revision":
            revision,
        "current_task": task,
        "constraints":
            constraints or [],
        "decisions":
            decisions or [],
        "open_items":
            open_items or [],
        "established_facts": [],
    }


class SemanticLifecycleTests(
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

    def semantic_path(self):
        return (
            Path(self.temp.name)
            / "semantic"
            / (CID + ".json")
        )

    def state_path(self):
        return (
            Path(self.temp.name)
            / "semantic_state"
            / (CID + ".json")
        )

    def write_semantic(
        self,
        payload,
    ):
        path = self.semantic_path()

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

    def load_state(self):
        return json.loads(
            self.state_path()
            .read_text(
                encoding="utf-8"
            )
        )

    def update(self, payload):
        self.write_semantic(
            payload
        )
        return update_semantic_state(
            CID
        )

    def test_cancel_constraint_exact(
        self,
    ):
        self.update(
            semantic(
                1,
                task=item(
                    "继续开发 Context",
                    1,
                ),
                constraints=[
                    item(
                        "必须保持 Shadow 模式",
                        1,
                    )
                ],
            )
        )

        result = self.update(
            semantic(
                2,
                task=item(
                    "取消约束：必须保持 Shadow 模式",
                    2,
                ),
            )
        )

        state = self.load_state()

        self.assertEqual(
            len(state["constraints"]),
            0,
        )

        self.assertEqual(
            len(state["history"]),
            1,
        )

        self.assertEqual(
            state["history"][0]["status"],
            "cancelled_explicit",
        )

        self.assertTrue(
            result["lifecycle_matched"]
        )

        # Control command must not replace real task.
        self.assertEqual(
            state["current_task"]["text"],
            "继续开发 Context",
        )

    def test_unmatched_cancel_is_safe(
        self,
    ):
        self.update(
            semantic(
                1,
                constraints=[
                    item(
                        "不要注入模型",
                        1,
                    )
                ],
            )
        )

        result = self.update(
            semantic(
                2,
                task=item(
                    "取消约束：允许注入模型",
                    2,
                ),
            )
        )

        state = self.load_state()

        self.assertEqual(
            len(state["constraints"]),
            1,
        )

        self.assertFalse(
            result["lifecycle_matched"]
        )

        self.assertEqual(
            state["telemetry"][
                "lifecycle_unmatched_total"
            ],
            1,
        )

    def test_replace_constraint(
        self,
    ):
        self.update(
            semantic(
                1,
                constraints=[
                    item(
                        "必须保持 Shadow 模式",
                        1,
                    )
                ],
            )
        )

        result = self.update(
            semantic(
                2,
                task=item(
                    "替换约束：必须保持 Shadow 模式 => "
                    "必须使用 Controlled Injection",
                    2,
                ),
                # Simulate extractor seeing the command
                # itself as a constraint. It must be filtered.
                constraints=[
                    item(
                        "替换约束：必须保持 Shadow 模式 => "
                        "必须使用 Controlled Injection",
                        2,
                    )
                ],
            )
        )

        state = self.load_state()

        self.assertEqual(
            len(state["constraints"]),
            1,
        )

        self.assertEqual(
            state["constraints"][0]["text"],
            "必须使用 Controlled Injection",
        )

        self.assertEqual(
            state["history"][0]["status"],
            "superseded_explicit",
        )

        self.assertTrue(
            result[
                "lifecycle_replacement_added"
            ]
        )

    def test_control_command_filtered_even_with_different_source_index(
        self,
    ):
        self.update(
            semantic(
                1,
                constraints=[
                    item(
                        "必须保留 lifecycle-test",
                        10,
                    )
                ],
            )
        )

        result = self.update(
            semantic(
                2,
                task=item(
                    "取消约束：必须保留 lifecycle-test",
                    20,
                ),
                constraints=[
                    # Simulate real extractor provenance
                    # differing from current_task index.
                    item(
                        "取消约束：必须保留 lifecycle-test",
                        21,
                    )
                ],
            )
        )

        state = self.load_state()

        self.assertTrue(
            result["lifecycle_matched"]
        )

        self.assertEqual(
            len(state["constraints"]),
            0,
        )

        self.assertEqual(
            len(state["history"]),
            1,
        )

        self.assertFalse(
            any(
                "取消约束"
                in str(x.get("text"))
                for x
                in state["constraints"]
                if isinstance(x, dict)
            )
        )

    def test_cancel_does_not_resurrect_visible_window_item(
        self,
    ):
        self.update(
            semantic(
                1,
                constraints=[
                    item(
                        "必须保留 lifecycle-window-test",
                        10,
                    )
                ],
            )
        )

        result = self.update(
            semantic(
                2,
                task=item(
                    "取消约束：必须保留 lifecycle-window-test",
                    20,
                ),
                constraints=[
                    # Old item is still visible in the
                    # bounded Semantic window.
                    item(
                        "必须保留 lifecycle-window-test",
                        10,
                    ),
                    # Real extractor may emit a spaced
                    # form of the control command.
                    item(
                        "取 消 约 束 ： 必 须 保 留 lifecycle-window-test",
                        21,
                    ),
                ],
            )
        )

        state = self.load_state()

        self.assertTrue(
            result[
                "lifecycle_matched"
            ]
        )

        self.assertEqual(
            len(
                state[
                    "constraints"
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
            "cancelled_explicit",
        )

    def test_old_control_command_does_not_reappear_later(
        self,
    ):
        self.update(
            semantic(
                1,
                constraints=[
                    item(
                        "必须保留 later-control-test",
                        10,
                    )
                ],
            )
        )

        self.update(
            semantic(
                2,
                task=item(
                    "取消约束：必须保留 later-control-test",
                    20,
                ),
            )
        )

        self.update(
            semantic(
                3,
                task=item(
                    "继续开发",
                    30,
                ),
                constraints=[
                    item(
                        "取 消 约 束 ： 必 须 保 留 later-control-test",
                        21,
                    )
                ],
            )
        )

        state = self.load_state()

        self.assertFalse(
            any(
                "取消约束"
                in str(
                    x.get("text")
                ).replace(" ", "")
                for x
                in state["constraints"]
                if isinstance(x, dict)
            )
        )

    def test_closed_old_source_does_not_resurrect_later(
        self,
    ):
        self.update(
            semantic(
                1,
                constraints=[
                    item(
                        "必须保留 later-window-test",
                        10,
                    )
                ],
            )
        )

        self.update(
            semantic(
                2,
                task=item(
                    "取消约束：必须保留 later-window-test",
                    20,
                ),
                constraints=[
                    item(
                        "必须保留 later-window-test",
                        10,
                    )
                ],
            )
        )

        self.update(
            semantic(
                3,
                task=item(
                    "继续开发",
                    30,
                ),
                constraints=[
                    # Old source still visible in window.
                    item(
                        "必须保留 later-window-test",
                        10,
                    )
                ],
            )
        )

        state = self.load_state()

        self.assertFalse(
            any(
                x.get("text")
                == "必须保留 later-window-test"
                for x
                in state["constraints"]
                if isinstance(x, dict)
            )
        )

    def test_same_text_can_be_reintroduced_from_new_source(
        self,
    ):
        self.update(
            semantic(
                1,
                constraints=[
                    item(
                        "必须保留 readd-test",
                        10,
                    )
                ],
            )
        )

        self.update(
            semantic(
                2,
                task=item(
                    "取消约束：必须保留 readd-test",
                    20,
                ),
            )
        )

        self.update(
            semantic(
                3,
                task=item(
                    "重新要求必须保留 readd-test",
                    30,
                ),
                constraints=[
                    # Same text, but genuinely stated again
                    # after the cancellation.
                    item(
                        "必须保留 readd-test",
                        30,
                    )
                ],
            )
        )

        state = self.load_state()

        active = [
            x
            for x
            in state["constraints"]
            if (
                isinstance(x, dict)
                and x.get("text")
                == "必须保留 readd-test"
            )
        ]

        self.assertEqual(
            len(active),
            1,
        )

        self.assertEqual(
            active[0][
                "first_source_index"
            ],
            30,
        )

    def test_revoke_decision(
        self,
    ):
        self.update(
            semantic(
                1,
                decisions=[
                    item(
                        "决定继续使用 Shadow",
                        1,
                    )
                ],
            )
        )

        self.update(
            semantic(
                2,
                task=item(
                    "撤销决定：决定继续使用 Shadow",
                    2,
                ),
            )
        )

        state = self.load_state()

        self.assertEqual(
            len(state["decisions"]),
            0,
        )

        self.assertEqual(
            state["history"][0]["status"],
            "revoked_explicit",
        )

    def test_resolve_open_item(
        self,
    ):
        self.update(
            semantic(
                1,
                open_items=[
                    item(
                        "下一步接入 Context Assembly？",
                        1,
                    )
                ],
            )
        )

        self.update(
            semantic(
                2,
                task=item(
                    "完成待办：下一步接入 Context Assembly？",
                    2,
                ),
            )
        )

        state = self.load_state()

        self.assertEqual(
            len(state["open_items"]),
            0,
        )

        self.assertEqual(
            state["history"][0]["status"],
            "resolved_explicit",
        )

    def test_v1_state_migrates_without_loss(
        self,
    ):
        state_path = self.state_path()

        state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state_path.write_text(
            json.dumps(
                {
                    "version":
                        "conversation-semantic-state.v1",
                    "conversation_id":
                        CID,
                    "revision": 1,
                    "source_revision": 1,
                    "current_task":
                        item(
                            "旧任务",
                            1,
                        ),
                    "constraints": [
                        {
                            "id": "sem_old",
                            "kind":
                                "constraint",
                            "status":
                                "active",
                            "text":
                                "必须保留旧状态",
                            "first_source_index":
                                1,
                            "last_source_index":
                                1,
                            "first_seen_revision":
                                1,
                            "last_seen_revision":
                                1,
                            "occurrences":
                                1,
                        }
                    ],
                    "decisions": [],
                    "open_items": [],
                    "established_facts": [],
                    "history": [],
                    "telemetry": {
                        "facts_deferred":
                            True,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.update(
            semantic(
                2,
                task=item(
                    "继续",
                    2,
                ),
            )
        )

        state = self.load_state()

        self.assertEqual(
            state["version"],
            "conversation-semantic-state.v2",
        )

        self.assertEqual(
            len(state["constraints"]),
            1,
        )

        self.assertEqual(
            state["constraints"][0]["text"],
            "必须保留旧状态",
        )


    def test_carried_forward_lifecycle_command_is_not_replayed(
        self,
    ):
        self.update(
            semantic(
                1,
                constraints=[
                    item(
                        "必须保留 carried-test",
                        10,
                    )
                ],
            )
        )

        self.update(
            semantic(
                2,
                task=item(
                    "取消约束：必须保留 carried-test",
                    20,
                ),
            )
        )

        payload = semantic(
            3,
            task=item(
                "取消约束：必须保留 carried-test",
                20,
            ),
            constraints=[
                item(
                    "必须保留 carried-test",
                    10,
                )
            ],
        )

        payload["telemetry"] = {
            "current_task_carried_forward":
                True,
            "latest_user_ack_only":
                True,
        }

        result = self.update(
            payload
        )

        state = self.load_state()

        self.assertFalse(
            result[
                "lifecycle_command_detected"
            ]
        )

        self.assertTrue(
            result[
                "lifecycle_skipped_carried_forward"
            ]
        )

        self.assertEqual(
            len(
                state[
                    "constraints"
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

    def test_preexisting_stale_active_is_self_healed(
        self,
    ):
        self.update(
            semantic(
                1,
                constraints=[
                    item(
                        "必须保留 self-heal-test",
                        10,
                    )
                ],
            )
        )

        self.update(
            semantic(
                2,
                task=item(
                    "取消约束：必须保留 self-heal-test",
                    20,
                ),
            )
        )

        # Simulate stale pollution created by an older runtime:
        # the already-closed constraint somehow exists in
        # active state again.
        state = self.load_state()

        closed = dict(
            state[
                "history"
            ][0]
        )

        polluted = {
            "id":
                closed["id"],
            "kind":
                closed["kind"],
            "status":
                "active",
            "text":
                closed["text"],
            "first_source_index":
                closed[
                    "first_source_index"
                ],
            "last_source_index":
                closed[
                    "last_source_index"
                ],
            "first_seen_revision":
                closed[
                    "first_seen_revision"
                ],
            "last_seen_revision":
                closed[
                    "last_seen_revision"
                ],
            "occurrences":
                closed.get(
                    "occurrences",
                    1,
                ),
        }

        state["constraints"] = [
            polluted
        ]

        self.state_path().write_text(
            json.dumps(
                state,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        result = self.update(
            semantic(
                3,
                task=item(
                    "继续开发",
                    30,
                ),
                constraints=[
                    # Old closed source is still visible
                    # inside the bounded Semantic window.
                    item(
                        "必须保留 self-heal-test",
                        10,
                    )
                ],
            )
        )

        healed = self.load_state()

        self.assertEqual(
            len(
                healed[
                    "constraints"
                ]
            ),
            0,
        )

        self.assertEqual(
            result[
                "purged_closed_echoes_this_revision"
            ],
            1,
        )



if __name__ == "__main__":
    unittest.main()
