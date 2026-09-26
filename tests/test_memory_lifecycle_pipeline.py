from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from _lifecycle_fixtures import (
    CID,
    M1,
    M2,
    RID,
    T0,
    guarded_bucket_manager,
    hours,
    iso,
    lifecycle_env,
    source_bucket,
    tree,
    tree_without_lifecycle,
    usage_artifact,
    write_usage,
)
from _recall_fixtures import (
    flash_artifact,
    memory,
    recall_artifact,
    recall_request as make_recall_request,
    write_flash,
    write_recall,
)

from ombrebrain.context.memory_flash import (
    read_memory_flash,
)
from ombrebrain.context.memory_lifecycle_event import (
    lifecycle_events_dir,
    memory_key,
    read_lifecycle_events,
)
from ombrebrain.context.memory_lifecycle_state import (
    lifecycle_state_path,
    read_memory_lifecycle_state,
)
from ombrebrain.context.memory_reinforcement import (
    derive_memory_lifecycle_state,
    observe_memory_usage_lifecycle,
)
from ombrebrain.context.memory_usage_signal import (
    read_memory_usage,
)
from ombrebrain.context.related_recall import (
    read_related_recall,
)
from ombrebrain.context.recall_request import (
    read_recall_request,
)


RECALL_A = "recall_" + "0" * 32
RECALL_B = "recall_" + "1" * 32

_REPO_ROOT = (
    Path(__file__).resolve().parents[1]
)

# Modules that must stay completely unaware of the lifecycle shadow.
_UNWIRED_SOURCES = (
    "src/web/gateway.py",
    "src/ombrebrain/context/"
    "context_pipeline_coordinator.py",
    "src/ombrebrain/context/"
    "context_observation_pipeline.py",
    "src/ombrebrain/context/"
    "memory_surfacing_policy.py",
    "src/ombrebrain/context/memory_flash.py",
    "src/ombrebrain/context/retrieval/__init__.py",
    "src/ombrebrain/context/retrieval/scorer.py",
    "src/ombrebrain/context/retrieval/legacy.py",
    "src/ombrebrain/gateway/gateway_runtime.py",
)


def _current_chain(root):
    """The already-frozen recall chain, as a chain of real artifacts."""

    write_flash(
        root,
        flash_artifact(
            [
                memory(
                    M1,
                    "alpha note",
                    name="alpha",
                    last_active=iso(T0),
                ),
                memory(
                    M2,
                    "beta note",
                    name="beta",
                    last_active=iso(T0),
                ),
            ]
        ),
    )

    recall = recall_artifact(
        [M1, M2],
        recall_id=RECALL_A,
    )

    recall["created_at"] = iso(T0)

    write_recall(root, recall)

    request = make_recall_request(
        M1, revision=5
    )

    request["created_at"] = iso(T0)

    path = (
        Path(root)
        / "recall_request"
        / CID
        / RID
        / (RECALL_A + ".json")
    )

    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    path.write_text(
        json.dumps(request), encoding="utf-8"
    )

    write_usage(
        root,
        usage_artifact(
            recall_id=RECALL_A,
            loaded_memory_ids=(M1, M2),
            used_memory_ids=(M1,),
            at=iso(T0),
        ),
    )

    return {
        "flash": read_memory_flash(
            conversation_id=CID,
            cognitive_request_id=RID,
        ),
        "recall": read_related_recall(
            conversation_id=CID,
            cognitive_request_id=RID,
            recall_id=RECALL_A,
        ),
        "request": read_recall_request(
            conversation_id=CID,
            cognitive_request_id=RID,
            recall_id=RECALL_A,
        ),
        "usage": read_memory_usage(
            conversation_id=CID,
            cognitive_request_id=RID,
            recall_id=RECALL_A,
        ),
    }


class LifecyclePipelineTests(
    unittest.IsolatedAsyncioTestCase,
):
    async def test_chain_unaffected_while_lifecycle_runs(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            source_dir = Path(root) / "buckets"
            source_dir.mkdir()

            source_path = source_dir / (M1 + ".md")
            source_path.write_text(
                "---\nactivation_count: 7\n"
                "last_active: 2020-01-01T00:00:00Z\n"
                "importance: 9\n"
                "---\nbody for "
                + M1
                + "\n",
                encoding="utf-8",
            )

            source_bytes = source_path.read_bytes()

            manager = guarded_bucket_manager(
                {
                    M1: source_bucket(M1),
                    M2: source_bucket(M2),
                }
            )

            # Flag OFF: an explicit call must be a complete no-op.
            with lifecycle_env(root, enabled=False):
                before_chain = _current_chain(root)

                before_tree = tree_without_lifecycle(
                    root
                )

                disabled = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=manager,
                        as_of=T0,
                    )
                )

                self.assertEqual(
                    disabled["reason"],
                    "lifecycle_disabled",
                )

                self.assertEqual(
                    tree(root), before_tree
                )

            # Flag ON: the shadow runs and writes ONLY under its own
            # state dirs.
            with lifecycle_env(root):
                report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=manager,
                        as_of=T0 + hours(24),
                    )
                )

                far = await derive_memory_lifecycle_state(
                    M1,
                    bucket_manager=manager,
                    as_of=T0 + timedelta(days=365),
                )

                after_tree = tree_without_lifecycle(
                    root
                )

                after_chain = _current_chain(root)

                state = read_memory_lifecycle_state(
                    M1
                )

            self.assertTrue(report["stored"])
            self.assertEqual(
                report["new_event_count"], 1
            )
            self.assertTrue(far["state"]["exists"])

            # The frozen chain is byte-identical: the lifecycle
            # shadow changed no Retrieval, no Flash, no Recall and no
            # Usage artifact.
            self.assertEqual(
                after_tree, before_tree
            )
            self.assertEqual(
                after_chain, before_chain
            )
            self.assertEqual(
                source_path.read_bytes(),
                source_bytes,
            )

            self.assertEqual(state["use_count"], 1)
            self.assertAlmostEqual(
                state["strength"], 0.32
            )
            self.assertTrue(
                state["low_accessibility"]
            )

    async def test_lifecycle_adds_only_its_own_files(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_A,
                    loaded_memory_ids=(M1,),
                    used_memory_ids=(M1,),
                    at=iso(T0),
                ),
            )

            before = tree(root)

            with lifecycle_env(root):
                await observe_memory_usage_lifecycle(
                    CID,
                    RID,
                    RECALL_A,
                    as_of=T0,
                )

            after = tree(root)

        added = set(after) - set(before)

        self.assertTrue(added)

        for name in added:
            with self.subTest(name=name):
                self.assertTrue(
                    name.startswith(
                        "memory_lifecycle_events/"
                    )
                    or name.startswith(
                        "memory_lifecycle_state/"
                    )
                )

        for name, payload in before.items():
            with self.subTest(name=name):
                self.assertEqual(
                    after[name], payload
                )

    async def test_hundred_reprocessed_usages_one_event(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_A,
                    loaded_memory_ids=(M1,),
                    used_memory_ids=(M1,),
                    at=iso(T0),
                ),
            )

            with lifecycle_env(root):
                for _ in range(100):
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        as_of=T0,
                    )

                store = read_lifecycle_events(M1)

                state = read_memory_lifecycle_state(
                    M1
                )

        self.assertEqual(len(store["events"]), 1)
        self.assertEqual(state["use_count"], 1)
        self.assertAlmostEqual(
            state["strength"], 0.32
        )

    async def test_decay_uses_usage_time_not_observer_time(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            used_at = iso(T0 - hours(100))

            write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_A,
                    loaded_memory_ids=(M1,),
                    used_memory_ids=(M1,),
                    at=used_at,
                ),
            )

            with lifecycle_env(root):
                await observe_memory_usage_lifecycle(
                    CID,
                    RID,
                    RECALL_A,
                    as_of=T0,
                )

                state = read_memory_lifecycle_state(
                    M1
                )

        self.assertEqual(
            state["last_used_at"], used_at
        )
        self.assertEqual(
            state["reference_at"], used_at
        )
        self.assertEqual(
            state["reference_source"], "last_used"
        )

    async def test_far_future_decay_keeps_memory(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            source_dir = Path(root) / "buckets"
            source_dir.mkdir()

            path = source_dir / (M1 + ".md")
            path.write_text(
                "body for " + M1,
                encoding="utf-8",
            )

            payload = path.read_bytes()

            manager = guarded_bucket_manager(
                {M1: source_bucket(M1)}
            )

            write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_A,
                    loaded_memory_ids=(M1,),
                    used_memory_ids=(M1,),
                    at=iso(T0),
                ),
            )

            with lifecycle_env(root):
                await observe_memory_usage_lifecycle(
                    CID,
                    RID,
                    RECALL_A,
                    bucket_manager=manager,
                    as_of=T0,
                )

                far = await derive_memory_lifecycle_state(
                    M1,
                    bucket_manager=manager,
                    as_of=(
                        T0 + timedelta(days=365)
                    ),
                )

                self.assertTrue(
                    far["state"]["exists"]
                )
                self.assertEqual(
                    far["state"]["event_count"], 1
                )
                self.assertEqual(
                    far["state"]["use_count"], 1
                )
                self.assertAlmostEqual(
                    far["state"]["strength"], 0.32
                )
                self.assertTrue(
                    far["state"][
                        "low_accessibility"
                    ]
                )
                self.assertAlmostEqual(
                    far["state"]["accessibility"],
                    0.02,
                    places=3,
                )

            self.assertEqual(
                path.read_bytes(), payload
            )

            for name in (
                "touch",
                "archive",
                "delete",
                "update",
                "create",
            ):
                with self.subTest(name=name):
                    self.assertEqual(
                        getattr(
                            manager, name
                        ).call_count,
                        0,
                    )

    async def test_state_is_not_model_facing(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _current_chain(root)

                await observe_memory_usage_lifecycle(
                    CID,
                    RID,
                    RECALL_A,
                    as_of=T0,
                )

                flash = json.dumps(
                    read_memory_flash(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

                recall = json.dumps(
                    read_related_recall(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        recall_id=RECALL_A,
                    )
                )

                usage = json.dumps(
                    read_memory_usage(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        recall_id=RECALL_A,
                    )
                )

        for name, rendered in (
            ("flash", flash),
            ("recall", recall),
            ("usage", usage),
        ):
            for forbidden in (
                "strength",
                "accessibility",
                "salience",
                "lifecycle",
            ):
                with self.subTest(
                    name=name, forbidden=forbidden
                ):
                    self.assertNotIn(
                        forbidden, rendered
                    )

    async def test_state_rebuildable_after_state_loss(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            manager = guarded_bucket_manager(
                {M1: source_bucket(M1)}
            )

            write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_A,
                    loaded_memory_ids=(M1,),
                    used_memory_ids=(M1,),
                    at=iso(T0),
                ),
            )

            with lifecycle_env(root):
                await observe_memory_usage_lifecycle(
                    CID,
                    RID,
                    RECALL_A,
                    bucket_manager=manager,
                    as_of=T0,
                )

                first = read_memory_lifecycle_state(
                    M1
                )

                path = lifecycle_state_path(
                    memory_key(M1)
                )

                path.unlink()

                await derive_memory_lifecycle_state(
                    M1,
                    bucket_manager=manager,
                    as_of=T0,
                )

                second = read_memory_lifecycle_state(
                    M1
                )

        for field in (
            "use_count",
            "strength",
            "accessibility",
            "salience",
            "last_used_at",
            "as_of",
        ):
            with self.subTest(field=field):
                self.assertEqual(
                    first[field], second[field]
                )


class ShadowCompositionTests(unittest.TestCase):
    """Static invariants of the three new lifecycle modules.

    The checks parse the modules instead of matching raw text, so a
    comment that merely *mentions* a forbidden legacy concept is not
    a failure while a real call to it is.
    """

    def _module_paths(self) -> list[str]:
        return [
            "src/ombrebrain/context/"
            "memory_lifecycle_event.py",
            "src/ombrebrain/context/"
            "memory_lifecycle_state.py",
            "src/ombrebrain/context/"
            "memory_reinforcement.py",
        ]

    def _tree(self, relative: str) -> ast.Module:
        path = _REPO_ROOT / relative

        self.assertTrue(path.is_file(), relative)

        return ast.parse(
            path.read_text(encoding="utf-8")
        )

    def test_lifecycle_modules_are_unwired(self):
        for relative in _UNWIRED_SOURCES:
            path = _REPO_ROOT / relative

            with self.subTest(relative=relative):
                self.assertTrue(
                    path.is_file(), relative
                )

                source = path.read_text(
                    encoding="utf-8"
                )

                self.assertNotIn(
                    "memory_lifecycle", source
                )

                self.assertNotIn(
                    "memory_reinforcement", source
                )

    def test_lifecycle_code_has_no_legacy_mutation(
        self,
    ):
        forbidden_calls = {
            "touch",
            "archive",
            "update",
            "delete",
            "create",
        }

        forbidden_fields = {
            "activation_count",
            "last_active",
            "importance",
        }

        for relative in self._module_paths():
            tree = self._tree(relative)

            with self.subTest(relative=relative):
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(
                            node.func,
                            ast.Attribute,
                        )
                    ):
                        self.assertNotIn(
                            node.func.attr,
                            forbidden_calls,
                        )

                    if (
                        isinstance(node, ast.Constant)
                        and isinstance(
                            node.value, str
                        )
                    ):
                        self.assertNotIn(
                            node.value,
                            forbidden_fields,
                        )

    def test_lifecycle_code_has_no_model_or_retrieval(
        self,
    ):
        for relative in self._module_paths():
            tree = self._tree(relative)

            with self.subTest(relative=relative):
                for node in ast.walk(tree):
                    if isinstance(
                        node, ast.Import
                    ):
                        modules = [
                            alias.name
                            for alias in node.names
                        ]
                    elif isinstance(
                        node, ast.ImportFrom
                    ):
                        modules = [
                            node.module or ""
                        ]
                    else:
                        modules = []

                    for module in modules:
                        lowered = module.lower()

                        for forbidden in (
                            "embedding",
                            "retrieval",
                            "decay_engine",
                            "bucket_manager",
                            "openai",
                            "anthropic",
                        ):
                            self.assertNotIn(
                                forbidden, lowered
                            )

                    if (
                        isinstance(node, ast.Call)
                        and isinstance(
                            node.func,
                            ast.Attribute,
                        )
                    ):
                        self.assertFalse(
                            node.func.attr.startswith(
                                "retrieve"
                            ),
                            node.func.attr,
                        )

    def test_no_background_scheduler_added(self):
        for relative in self._module_paths():
            tree = self._tree(relative)

            with self.subTest(relative=relative):
                for node in ast.walk(tree):
                    if isinstance(
                        node, ast.Import
                    ):
                        modules = [
                            alias.name
                            for alias in node.names
                        ]
                    elif isinstance(
                        node, ast.ImportFrom
                    ):
                        modules = [
                            node.module or ""
                        ]
                    else:
                        modules = []

                    for module in modules:
                        for forbidden in (
                            "asyncio",
                            "threading",
                            "sched",
                            "apscheduler",
                            "crontab",
                            "schedule",
                        ):
                            self.assertNotIn(
                                forbidden,
                                module.lower(),
                            )

                    if (
                        isinstance(node, ast.Call)
                        and isinstance(
                            node.func,
                            ast.Attribute,
                        )
                    ):
                        self.assertNotIn(
                            node.func.attr,
                            {
                                "create_task",
                                "call_later",
                                "call_at",
                            },
                        )

    def test_state_dirs_are_under_context_state(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"OMBRE_CONTEXT_STATE_DIR": root},
                clear=False,
            ):
                events = lifecycle_events_dir(
                    memory_key(M1)
                )
                state = lifecycle_state_path(
                    memory_key(M1)
                )

        self.assertEqual(
            str(events),
            str(
                Path(root)
                / "memory_lifecycle_events"
                / memory_key(M1)
            ),
        )
        self.assertEqual(
            str(state),
            str(
                Path(root)
                / "memory_lifecycle_state"
                / (memory_key(M1) + ".json")
            ),
        )


if __name__ == "__main__":
    unittest.main()