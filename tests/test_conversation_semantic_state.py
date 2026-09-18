from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ombrebrain.context.conversation_semantic_state import (
    semantic_state_status,
    update_semantic_state,
)


CID = "ctx_0123456789abcdef"


def item(
    text,
    source_index,
):
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
        "current_task":
            task,
        "constraints":
            constraints or [],
        "decisions":
            decisions or [],
        "open_items":
            open_items or [],
        "established_facts": [],
    }


class SemanticStateTests(
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

    def test_constraint_survives_window_loss(
        self,
    ):
        self.write_semantic(
            semantic(
                1,
                constraints=[
                    item(
                        "必须保持 Shadow 模式",
                        10,
                    )
                ],
            )
        )

        update_semantic_state(
            CID
        )

        # Next semantic frame no longer contains it.
        self.write_semantic(
            semantic(
                2,
                constraints=[],
            )
        )

        result = (
            update_semantic_state(
                CID
            )
        )

        state = self.load_state()

        self.assertEqual(
            result[
                "constraint_count"
            ],
            1,
        )

        self.assertEqual(
            state[
                "constraints"
            ][0]["text"],
            "必须保持 Shadow 模式",
        )

    def test_exact_repeat_refreshes_not_duplicates(
        self,
    ):
        value = item(
            "不要注入模型",
            20,
        )

        self.write_semantic(
            semantic(
                1,
                constraints=[
                    value
                ],
            )
        )

        update_semantic_state(
            CID
        )

        self.write_semantic(
            semantic(
                2,
                constraints=[
                    item(
                        "不要注入模型",
                        30,
                    )
                ],
            )
        )

        result = (
            update_semantic_state(
                CID
            )
        )

        state = self.load_state()

        self.assertEqual(
            len(
                state[
                    "constraints"
                ]
            ),
            1,
        )

        entry = state[
            "constraints"
        ][0]

        self.assertEqual(
            entry[
                "first_source_index"
            ],
            20,
        )

        self.assertEqual(
            entry[
                "last_source_index"
            ],
            30,
        )

        self.assertEqual(
            entry[
                "occurrences"
            ],
            2,
        )

        self.assertEqual(
            result[
                "refreshed_this_revision"
            ],
            1,
        )

    def test_decisions_and_open_items_persist(
        self,
    ):
        self.write_semantic(
            semantic(
                1,
                decisions=[
                    item(
                        "决定继续 Shadow",
                        5,
                    )
                ],
                open_items=[
                    item(
                        "下一步接入 Context Assembly？",
                        6,
                    )
                ],
            )
        )

        update_semantic_state(
            CID
        )

        self.write_semantic(
            semantic(
                2,
            )
        )

        update_semantic_state(
            CID
        )

        state = self.load_state()

        self.assertEqual(
            len(
                state[
                    "decisions"
                ]
            ),
            1,
        )

        self.assertEqual(
            len(
                state[
                    "open_items"
                ]
            ),
            1,
        )

    def test_source_revision_is_idempotent(
        self,
    ):
        self.write_semantic(
            semantic(
                3,
                constraints=[
                    item(
                        "必须保留来源",
                        3,
                    )
                ],
            )
        )

        first = (
            update_semantic_state(
                CID
            )
        )

        second = (
            update_semantic_state(
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

    def test_stale_source_revision_is_rejected(
        self,
    ):
        self.write_semantic(
            semantic(
                5,
            )
        )

        update_semantic_state(
            CID
        )

        self.write_semantic(
            semantic(
                4,
            )
        )

        result = (
            update_semantic_state(
                CID
            )
        )

        self.assertFalse(
            result["stored"]
        )

        self.assertEqual(
            result["reason"],
            "stale_source_revision",
        )

    def test_active_capacity_is_bounded(
        self,
    ):
        for revision in range(
            1,
            21,
        ):
            self.write_semantic(
                semantic(
                    revision,
                    constraints=[
                        item(
                            "constraint-%02d"
                            % revision,
                            revision,
                        )
                    ],
                )
            )

            update_semantic_state(
                CID
            )

        state = self.load_state()

        self.assertEqual(
            len(
                state[
                    "constraints"
                ]
            ),
            16,
        )

        self.assertEqual(
            len(
                state[
                    "history"
                ]
            ),
            4,
        )

        self.assertTrue(
            all(
                item[
                    "status"
                ]
                == "archived_capacity"
                for item
                in state["history"]
            )
        )

    def test_status_exposes_no_text(
        self,
    ):
        secret = (
            "PRIVATE_PERSISTENT_STATE"
        )

        self.write_semantic(
            semantic(
                1,
                constraints=[
                    item(
                        secret,
                        1,
                    )
                ],
            )
        )

        update_semantic_state(
            CID
        )

        status = (
            semantic_state_status(
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

        self.assertEqual(
            status[
                "fact_count"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
