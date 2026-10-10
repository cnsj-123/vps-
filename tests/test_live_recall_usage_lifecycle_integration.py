from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from _lifecycle_fixtures import (
    T0,
    guarded_bucket_manager,
    iso,
    source_bucket,
    usage_artifact,
    write_usage,
)
from _recall_fixtures import (
    CID,
    DEFAULT_RECALL_ID,
    LIFECYCLE_SHADOW_ENV,
    RID,
    RID_B,
    USAGE_ATTRIBUTION_ENV,
    USAGE_LIFECYCLE_INTEGRATION_ENV,
    memory,
    projected_model_result,
    read_usage,
    seed_live_exposure,
    seed_recall_control_plane,
)

from ombrebrain.context import (
    live_recall_usage_lifecycle_integration as integration,
)
from ombrebrain.context import (
    memory_usage_signal,
    memory_reinforcement,
)
from ombrebrain.context.live_recall_usage_attribution import (
    render_live_recall_usage_ack,
)
from ombrebrain.context.live_recall_usage_lifecycle_integration import (
    SAFE_LIFECYCLE_INTEGRATION_REASONS,
    attribute_live_recall_usage_with_lifecycle,
    live_recall_usage_lifecycle_integration_enabled,
)
from ombrebrain.context.live_recall_usage_ref_receipt import (
    read_usage_ref_receipt,
    usage_ref_receipt_path,
    useref_sha256,
)
from ombrebrain.context.live_recall_usage_surface import (
    build_live_recall_usage_surface,
    live_recall_usage_surface_path,
)
from ombrebrain.context.memory_lifecycle_event import (
    read_lifecycle_events,
)
from ombrebrain.context.memory_lifecycle_state import (
    read_memory_lifecycle_state,
    resolve_lifecycle_config,
    strength_for_use_count,
)


_LOADED_IDS = ["mem-1", "mem-2", "mem-3"]

_EXPOSED_IDS = ["mem-1", "mem-2"]

_RECALL_A = "recall_" + "9" * 32

_REPO_ROOT = Path(__file__).resolve().parents[1]

_LIFECYCLE_DIRS = (
    "memory_lifecycle_events",
    "memory_lifecycle_state",
)


def _has_lifecycle_dirs(root) -> bool:
    return any(
        (Path(root) / name).exists()
        for name in _LIFECYCLE_DIRS
    )


def seed_full_chain(
    root,
    *,
    loaded_ids=None,
    exposed_ids=None,
    conversation_id=CID,
    cognitive_request_id=RID,
    recall_id=DEFAULT_RECALL_ID,
):
    """Seed the frozen Recall chain AND build THIS request's surface."""

    loaded = list(
        _LOADED_IDS if loaded_ids is None else loaded_ids
    )

    exposed = list(
        _EXPOSED_IDS
        if exposed_ids is None
        else exposed_ids
    )

    memrefs = seed_live_exposure(
        root,
        [memory(memory_id) for memory_id in exposed],
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
    )

    plane = seed_recall_control_plane(
        root,
        loaded,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    surface = build_live_recall_usage_surface(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        anchor_ref=memrefs[0],
        model_result=projected_model_result(loaded),
    )

    refs = {
        ref["rank"]: ref["useref"]
        for ref in surface["usage_refs"]
    }

    return {
        "memrefs": memrefs,
        "plane": plane,
        "surface": surface,
        "refs": refs,
    }


class LifecycleIntegrationCase(
    unittest.IsolatedAsyncioTestCase,
):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()

        self.addCleanup(tmp.cleanup)

        self.root = tmp.name

        env = patch.dict(
            os.environ,
            {
                "OMBRE_CONTEXT_STATE_DIR": self.root,
                USAGE_ATTRIBUTION_ENV: "1",
                # Both downstream flags default OFF so every test
                # states the gate combination it needs.
                USAGE_LIFECYCLE_INTEGRATION_ENV: "0",
                LIFECYCLE_SHADOW_ENV: "0",
            },
            clear=False,
        )

        env.start()

        self.addCleanup(env.stop)

        self.chain = seed_full_chain(self.root)

        self.refs = self.chain["refs"]

    async def _call(
        self,
        userefs,
        *,
        bucket_manager=None,
        as_of=None,
        **env,
    ):
        with patch.dict(os.environ, env, clear=False):
            return (
                await attribute_live_recall_usage_with_lifecycle(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    userefs=userefs,
                    bucket_manager=bucket_manager,
                    as_of=as_of,
                )
            )


class GateCombinationTests(LifecycleIntegrationCase):
    """The triple gate, in every load-bearing combination."""

    async def test_default_integration_flag_is_off(self):
        with patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop(
                USAGE_LIFECYCLE_INTEGRATION_ENV, None
            )

            self.assertFalse(
                live_recall_usage_lifecycle_integration_enabled()
            )

    async def test_flag_truthy_values(self):
        for value in ("1", "true", "on", "yes"):
            with patch.dict(
                os.environ,
                {
                    USAGE_LIFECYCLE_INTEGRATION_ENV: value
                },
                clear=False,
            ):
                self.assertTrue(
                    live_recall_usage_lifecycle_integration_enabled(),
                    value,
                )

        for value in ("0", "false", "off", "", "no"):
            with patch.dict(
                os.environ,
                {
                    USAGE_LIFECYCLE_INTEGRATION_ENV: value
                },
                clear=False,
            ):
                self.assertFalse(
                    live_recall_usage_lifecycle_integration_enabled(),
                    value,
                )

    async def test_combo_a_all_off_records_nothing(self):
        result = await self._call(
            [self.refs[1]],
            **{
                USAGE_ATTRIBUTION_ENV: "0",
                USAGE_LIFECYCLE_INTEGRATION_ENV: "0",
                LIFECYCLE_SHADOW_ENV: "0",
            },
        )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["reason"],
            "usage_attribution_disabled",
        )
        self.assertEqual(
            result["lifecycle"]["reason"],
            "usage_attribution_disabled",
        )
        self.assertFalse(result["lifecycle"]["observed"])

        usage = read_usage(self.root)

        self.assertEqual(usage["used_count"], 0)
        self.assertFalse(_has_lifecycle_dirs(self.root))

    async def test_combo_b_integration_off_used_but_no_lifecycle(
        self,
    ):
        result = await self._call(
            [self.refs[1]],
            **{
                USAGE_ATTRIBUTION_ENV: "1",
                USAGE_LIFECYCLE_INTEGRATION_ENV: "0",
                LIFECYCLE_SHADOW_ENV: "1",
            },
        )

        self.assertTrue(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["newly_used_count"], 1
        )
        self.assertEqual(
            result["lifecycle"]["reason"],
            "integration_disabled",
        )
        self.assertFalse(result["lifecycle"]["observed"])

        self.assertEqual(
            read_usage(self.root)["used_count"], 1
        )
        self.assertFalse(_has_lifecycle_dirs(self.root))

    async def test_combo_c_shadow_off_used_but_no_lifecycle(
        self,
    ):
        result = await self._call(
            [self.refs[1]],
            **{
                USAGE_ATTRIBUTION_ENV: "1",
                USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
                LIFECYCLE_SHADOW_ENV: "0",
            },
        )

        self.assertTrue(result["usage"]["recorded"])
        self.assertEqual(
            result["lifecycle"]["reason"],
            "lifecycle_disabled",
        )
        self.assertFalse(result["lifecycle"]["observed"])

        self.assertEqual(
            read_usage(self.root)["used_count"], 1
        )
        self.assertFalse(_has_lifecycle_dirs(self.root))

    async def test_combo_d_all_on_reinforces(self):
        result = await self._call(
            [self.refs[1]],
            **{
                USAGE_ATTRIBUTION_ENV: "1",
                USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
                LIFECYCLE_SHADOW_ENV: "1",
            },
        )

        self.assertTrue(result["usage"]["recorded"])
        self.assertTrue(result["lifecycle"]["enabled"])
        self.assertTrue(result["lifecycle"]["observed"])
        self.assertEqual(
            result["lifecycle"]["reason"],
            "lifecycle_observed",
        )
        self.assertEqual(
            result["lifecycle"]["new_event_count"], 1
        )
        self.assertEqual(
            result["lifecycle"]["state_updated_count"], 1
        )

        self.assertEqual(
            len(read_lifecycle_events("mem-1")["events"]),
            1,
        )


class LifecycleEvidenceTests(LifecycleIntegrationCase):
    """``used`` is a necessary but not sufficient reinforcement signal."""

    def _all_on(self):
        return {
            USAGE_ATTRIBUTION_ENV: "1",
            USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
            LIFECYCLE_SHADOW_ENV: "1",
        }

    async def test_lifecycle_event_provenance_is_the_usage_signal(
        self,
    ):
        await self._call([self.refs[1]], **self._all_on())

        usage = read_usage(self.root)

        used_indices = [
            index
            for index, event in enumerate(
                usage["events"]
            )
            if event["stage"] == "used"
            and event["memory_id"] == "mem-1"
        ]

        self.assertEqual(len(used_indices), 1)

        events = read_lifecycle_events("mem-1")[
            "events"
        ]

        self.assertEqual(len(events), 1)

        event = events[0]

        self.assertEqual(event["stage"], "used")
        self.assertEqual(
            event["source_usage_version"],
            "memory-usage-signal.v1",
        )
        self.assertEqual(
            event["source_usage_event_index"],
            used_indices[0],
        )

    async def test_state_strength_matches_the_frozen_formula(
        self,
    ):
        await self._call([self.refs[1]], **self._all_on())

        state = read_memory_lifecycle_state("mem-1")

        self.assertEqual(state["use_count"], 1)
        self.assertAlmostEqual(
            state["strength"], 0.32
        )

    async def test_multiple_used_events_reinforce_each_memory(
        self,
    ):
        result = await self._call(
            [self.refs[1], self.refs[3]],
            **self._all_on(),
        )

        self.assertTrue(result["usage"]["recorded"])
        self.assertEqual(
            result["lifecycle"]["used_event_count"], 2
        )
        self.assertEqual(
            result["lifecycle"]["new_event_count"], 2
        )

        self.assertEqual(
            len(read_lifecycle_events("mem-1")["events"]),
            1,
        )
        self.assertEqual(
            len(read_lifecycle_events("mem-3")["events"]),
            1,
        )

        # The unselected rank 2 is NOT reinforced.
        self.assertEqual(
            read_lifecycle_events("mem-2")["events"], []
        )
        self.assertIsNone(
            read_memory_lifecycle_state("mem-2")
        )

    async def test_loaded_but_unused_is_not_reinforced(self):
        await self._call(
            [self.refs[1], self.refs[3]],
            **self._all_on(),
        )

        self.assertIsNone(
            read_memory_lifecycle_state("mem-2")
        )
        self.assertEqual(
            read_lifecycle_events("mem-2")["events"], []
        )

    async def test_recall_only_never_reinforces(self):
        # ``self.chain`` seeded recall_requested + memory_loaded only:
        # no UseMemory call happened.
        for memory_id in _LOADED_IDS:
            self.assertEqual(
                read_lifecycle_events(memory_id)["events"],
                [],
            )
            self.assertIsNone(
                read_memory_lifecycle_state(memory_id)
            )

        usage = read_usage(self.root)

        self.assertEqual(usage["used_count"], 0)

        self.assertFalse(_has_lifecycle_dirs(self.root))

    async def test_invalid_usememory_never_reinforces(self):
        observer = AsyncMock(return_value={})

        with patch.object(
            integration,
            "process_completed_usage_for_lifecycle",
            observer,
        ):
            result = await self._call(
                ["useref_" + "0" * 32],
                **self._all_on(),
            )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["lifecycle"]["reason"],
            "usage_not_recorded",
        )
        self.assertEqual(observer.await_count, 0)

        self.assertFalse(_has_lifecycle_dirs(self.root))

    async def test_duplicate_usememory_does_not_reinforce_twice(
        self,
    ):
        first = await self._call(
            [self.refs[1]], **self._all_on()
        )

        self.assertTrue(first["usage"]["recorded"])

        second = await self._call(
            [self.refs[1]], **self._all_on()
        )

        self.assertTrue(second["usage"]["recorded"])
        self.assertEqual(
            second["usage"]["newly_used_count"], 0
        )
        self.assertGreaterEqual(
            second["lifecycle"][
                "duplicate_event_count"
            ],
            1,
        )
        self.assertEqual(
            second["lifecycle"]["new_event_count"], 0
        )

        self.assertEqual(
            len(read_lifecycle_events("mem-1")["events"]),
            1,
        )

        state = read_memory_lifecycle_state("mem-1")

        self.assertEqual(state["use_count"], 1)
        self.assertAlmostEqual(
            state["strength"], 0.32
        )


class CatchUpTests(LifecycleIntegrationCase):
    """Duplicate attribution retries a missing lifecycle observation."""

    async def test_flag_off_then_on_catches_up(self):
        off = await self._call(
            [self.refs[1]],
            **{
                USAGE_LIFECYCLE_INTEGRATION_ENV: "0",
                LIFECYCLE_SHADOW_ENV: "1",
            },
        )

        self.assertTrue(off["usage"]["recorded"])

        self.assertFalse(_has_lifecycle_dirs(self.root))

        catch_up = await self._call(
            [self.refs[1]],
            **{
                USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
                LIFECYCLE_SHADOW_ENV: "1",
            },
        )

        self.assertTrue(catch_up["usage"]["recorded"])
        self.assertEqual(
            catch_up["usage"]["newly_used_count"], 0
        )
        self.assertTrue(
            catch_up["lifecycle"]["observed"]
        )
        self.assertEqual(
            catch_up["lifecycle"]["new_event_count"], 1
        )

        self.assertEqual(
            len(read_lifecycle_events("mem-1")["events"]),
            1,
        )

        again = await self._call(
            [self.refs[1]],
            **{
                USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
                LIFECYCLE_SHADOW_ENV: "1",
            },
        )

        self.assertEqual(
            again["lifecycle"]["new_event_count"], 0
        )
        self.assertGreaterEqual(
            again["lifecycle"]["duplicate_event_count"],
            1,
        )

        self.assertEqual(
            len(read_lifecycle_events("mem-1")["events"]),
            1,
        )

    async def test_multi_use_strength_progression(self):
        second_id = "recall_" + "2" * 32

        second_plane = seed_recall_control_plane(
            self.root,
            ["mem-2", "mem-1"],
            recall_id=second_id,
        )

        second_surface = build_live_recall_usage_surface(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_ref=self.chain["memrefs"][1],
            model_result=projected_model_result(
                ["mem-2", "mem-1"]
            ),
        )

        self.assertTrue(second_surface["stored"])

        first_ref = self.refs[1]

        second_ref = second_surface["usage_refs"][1][
            "useref"
        ]

        env = {
            USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
            LIFECYCLE_SHADOW_ENV: "1",
        }

        await self._call([first_ref], **env)

        await self._call([second_ref], **env)

        self.assertTrue(second_plane["recall_id"])

        events = read_lifecycle_events("mem-1")["events"]

        self.assertEqual(len(events), 2)

        state = read_memory_lifecycle_state("mem-1")

        self.assertEqual(state["use_count"], 2)
        self.assertAlmostEqual(
            state["strength"],
            strength_for_use_count(
                2,
                config=resolve_lifecycle_config(),
            ),
        )

    async def test_per_recall_catch_up_only(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR": root,
                    USAGE_ATTRIBUTION_ENV: "1",
                    USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
                    LIFECYCLE_SHADOW_ENV: "1",
                },
                clear=False,
            ):
                chain = seed_full_chain(
                    root,
                    loaded_ids=["mem-1"],
                    exposed_ids=["mem-1"],
                )

                # Recall A holds an OLD, never-observed used event for
                # a DIFFERENT memory.
                write_usage(
                    root,
                    usage_artifact(
                        recall_id=_RECALL_A,
                        anchor_memory_id="mem-8",
                        loaded_memory_ids=("mem-8",),
                        used_memory_ids=("mem-8",),
                        at=iso(T0),
                    ),
                )

                result = (
                    await attribute_live_recall_usage_with_lifecycle(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        userefs=[chain["refs"][1]],
                    )
                )

                self.assertTrue(
                    result["usage"]["recorded"]
                )
                self.assertTrue(
                    result["lifecycle"]["observed"]
                )

                self.assertEqual(
                    len(
                        read_lifecycle_events("mem-1")[
                            "events"
                        ]
                    ),
                    1,
                )

                # Recall A is NEVER scanned.
                self.assertEqual(
                    read_lifecycle_events("mem-8")[
                        "events"
                    ],
                    [],
                )
                self.assertIsNone(
                    read_memory_lifecycle_state("mem-8")
                )


class RuntimeKillSwitchTests(LifecycleIntegrationCase):
    async def test_integration_runtime_kill_switch(self):
        first = await self._call(
            [self.refs[1]],
            **{
                USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
                LIFECYCLE_SHADOW_ENV: "1",
            },
        )

        self.assertTrue(first["lifecycle"]["observed"])

        second = await self._call(
            [self.refs[2]],
            **{
                USAGE_LIFECYCLE_INTEGRATION_ENV: "0",
                LIFECYCLE_SHADOW_ENV: "1",
            },
        )

        self.assertTrue(second["usage"]["recorded"])
        self.assertEqual(
            second["lifecycle"]["reason"],
            "integration_disabled",
        )

        self.assertEqual(
            read_lifecycle_events("mem-2")["events"], []
        )
        self.assertEqual(
            len(read_lifecycle_events("mem-1")["events"]),
            1,
        )

    async def test_shadow_runtime_kill_switch(self):
        first = await self._call(
            [self.refs[1]],
            **{
                USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
                LIFECYCLE_SHADOW_ENV: "1",
            },
        )

        self.assertTrue(first["lifecycle"]["observed"])

        second = await self._call(
            [self.refs[2]],
            **{
                USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
                LIFECYCLE_SHADOW_ENV: "0",
            },
        )

        self.assertTrue(second["usage"]["recorded"])
        self.assertEqual(
            second["lifecycle"]["reason"],
            "lifecycle_disabled",
        )

        self.assertEqual(
            read_lifecycle_events("mem-2")["events"], []
        )


class FailureIsolationTests(LifecycleIntegrationCase):
    def _all_on(self):
        return {
            USAGE_ATTRIBUTION_ENV: "1",
            USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
            LIFECYCLE_SHADOW_ENV: "1",
        }

    async def test_observer_exception_keeps_usage_recorded(self):
        with patch.object(
            integration,
            "process_completed_usage_for_lifecycle",
            AsyncMock(
                side_effect=RuntimeError("boom")
            ),
        ):
            result = await self._call(
                [self.refs[1]], **self._all_on()
            )

        self.assertTrue(result["usage"]["recorded"])
        self.assertFalse(result["lifecycle"]["observed"])
        self.assertEqual(
            result["lifecycle"]["reason"],
            "lifecycle_unavailable",
        )

        self.assertEqual(
            read_usage(self.root)["used_count"], 1
        )

    async def test_state_write_failure_keeps_event_and_ack(self):
        with patch.object(
            memory_reinforcement,
            "persist_lifecycle_state",
            side_effect=RuntimeError("boom"),
        ):
            result = await self._call(
                [self.refs[1]], **self._all_on()
            )

        self.assertTrue(result["usage"]["recorded"])
        self.assertTrue(result["lifecycle"]["observed"])
        self.assertEqual(
            result["lifecycle"]["state_updated_count"], 0
        )

        self.assertEqual(
            len(read_lifecycle_events("mem-1")["events"]),
            1,
        )
        self.assertIsNone(
            read_memory_lifecycle_state("mem-1")
        )

    async def test_corrupt_surface_before_attribution_records_nothing(
        self,
    ):
        path = live_recall_usage_surface_path(CID, RID)

        artifact = json.loads(
            path.read_text(encoding="utf-8")
        )

        artifact["mappings"][0]["rank"] = 0

        path.write_text(
            json.dumps(artifact), encoding="utf-8"
        )

        result = await self._call(
            [self.refs[1]], **self._all_on()
        )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["reason"],
            "usage_surface_unavailable",
        )

        self.assertEqual(
            read_usage(self.root)["used_count"], 0
        )
        self.assertFalse(_has_lifecycle_dirs(self.root))

    async def test_surface_tamper_after_attribution_blocks_lifecycle(
        self,
    ):
        real = integration.attribute_live_recall_usage

        def tampering(**kwargs):
            report = real(**kwargs)

            path = live_recall_usage_surface_path(CID, RID)

            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )

            # Structurally valid, provenance-invalid.
            artifact["mappings"][0]["memory_id"] = "mem-9"

            path.write_text(
                json.dumps(artifact), encoding="utf-8"
            )

            return report

        with patch.object(
            integration,
            "attribute_live_recall_usage",
            tampering,
        ):
            result = await self._call(
                [self.refs[1]], **self._all_on()
            )

        # ``used`` is a historical fact and stays.
        self.assertTrue(result["usage"]["recorded"])
        self.assertFalse(result["lifecycle"]["observed"])
        self.assertEqual(
            result["lifecycle"]["reason"],
            "usage_provenance_invalid",
        )

        self.assertFalse(_has_lifecycle_dirs(self.root))

    async def test_canonical_memory_stays_read_only(self):
        manager = guarded_bucket_manager(
            {"mem-1": source_bucket("mem-1")}
        )

        result = await self._call(
            [self.refs[1]],
            bucket_manager=manager,
            **self._all_on(),
        )

        self.assertTrue(result["usage"]["recorded"])

        for name in (
            "touch",
            "archive",
            "delete",
            "update",
            "create",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(manager, name).call_count,
                    0,
                )


class ReportAndAckTests(LifecycleIntegrationCase):
    def _all_on(self):
        return {
            USAGE_ATTRIBUTION_ENV: "1",
            USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
            LIFECYCLE_SHADOW_ENV: "1",
        }

    async def test_report_is_privacy_safe(self):
        result = await self._call(
            [self.refs[1], self.refs[3]], **self._all_on()
        )

        rendered = json.dumps(result)

        for forbidden in (
            CID,
            RID,
            DEFAULT_RECALL_ID,
            "useref_",
            "mem-",
            "memory_id",
            "recall_id",
            "memref",
            "cue",
            "query",
            "token",
        ):
            self.assertNotIn(forbidden, rendered)

        self.assertIn(
            result["lifecycle"]["reason"],
            SAFE_LIFECYCLE_INTEGRATION_REASONS,
        )

    async def test_ack_is_identical_with_and_without_lifecycle(
        self,
    ):
        ack_off = await self._ack(False, False)
        ack_on = await self._ack(True, True)

        self.assertEqual(ack_off, ack_on)

        payload = json.loads(
            ack_on.split("\n", 1)[1]
        )

        self.assertEqual(payload["status"], "recorded")

        for forbidden in (
            "lifecycle",
            "reinforced",
            "strength",
            "accessibility",
            "salience",
            "use_count",
            "memory_id",
            "recall_id",
            CID,
            RID,
        ):
            self.assertNotIn(forbidden, ack_on)

    async def _ack(self, integration_on, shadow_on):
        with tempfile.TemporaryDirectory() as root:
            env = {
                "OMBRE_CONTEXT_STATE_DIR": root,
                USAGE_ATTRIBUTION_ENV: "1",
                USAGE_LIFECYCLE_INTEGRATION_ENV: (
                    "1" if integration_on else "0"
                ),
                LIFECYCLE_SHADOW_ENV: (
                    "1" if shadow_on else "0"
                ),
            }

            with patch.dict(
                os.environ, env, clear=False
            ):
                chain = seed_full_chain(root)

                result = (
                    await attribute_live_recall_usage_with_lifecycle(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        userefs=[chain["refs"][1]],
                    )
                )

                return render_live_recall_usage_ack(
                    result["usage"]
                )


def _record_usage_spy(calls):
    """Patch ``record_usage`` while capturing its calls."""

    original = memory_usage_signal.record_usage

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    return patch.object(
        memory_usage_signal, "record_usage", spy
    )


class UserefBindingIntegrityTests(
    LifecycleIntegrationCase,
):
    """The opaque useref ORIGINAL binding is receipt-anchored.

    A Usage Surface can never be its own authority: swapping two
    legal userefs, deleting a receipt, corrupting a receipt or
    editing a receipt must all refuse -- and must produce NO used
    event and NO lifecycle reinforcement.
    """

    def _all_on(self):
        return {
            USAGE_ATTRIBUTION_ENV: "1",
            USAGE_LIFECYCLE_INTEGRATION_ENV: "1",
            LIFECYCLE_SHADOW_ENV: "1",
        }

    def _surface_file(self):
        return live_recall_usage_surface_path(CID, RID)

    def _load_surface(self):
        return json.loads(
            self._surface_file().read_text(
                encoding="utf-8"
            )
        )

    def _write_surface(self, artifact):
        self._surface_file().write_text(
            json.dumps(artifact), encoding="utf-8"
        )

    def _receipt_path(self, useref):
        return usage_ref_receipt_path(
            CID, RID, useref_sha256(useref)
        )

    def _swap_surface_userefs(self):
        artifact = self._load_surface()

        mappings = artifact["mappings"]

        mappings[0]["useref"], mappings[1]["useref"] = (
            mappings[1]["useref"],
            mappings[0]["useref"],
        )

        self._write_surface(artifact)

    def _tamper_receipt(self, useref, **changes):
        path = self._receipt_path(useref)

        artifact = json.loads(
            path.read_text(encoding="utf-8")
        )

        artifact.update(changes)

        path.write_text(
            json.dumps(artifact), encoding="utf-8"
        )

    async def test_surface_useref_swap_is_refused(self):
        self._swap_surface_userefs()

        calls = []

        with _record_usage_spy(calls):
            result = await self._call(
                [self.refs[1]], **self._all_on()
            )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["reason"],
            "usage_ref_binding_invalid",
        )
        self.assertEqual(calls, [])

    async def test_surface_useref_swap_records_no_used(self):
        self._swap_surface_userefs()

        await self._call(
            [self.refs[1]], **self._all_on()
        )

        usage = read_usage(self.root)

        self.assertEqual(usage["used_count"], 0)
        self.assertEqual(usage["used_memory_ids"], [])
        self.assertEqual(
            [
                event
                for event in usage["events"]
                if event["stage"] == "used"
            ],
            [],
        )

    async def test_surface_useref_swap_creates_no_lifecycle(self):
        self._swap_surface_userefs()

        result = await self._call(
            [self.refs[1]], **self._all_on()
        )

        self.assertFalse(result["usage"]["recorded"])
        self.assertFalse(
            result["lifecycle"]["observed"]
        )
        self.assertEqual(
            result["lifecycle"]["new_event_count"], 0
        )

        self.assertEqual(
            read_lifecycle_events("mem-1")["events"], []
        )
        self.assertFalse(_has_lifecycle_dirs(self.root))

    async def test_missing_issuance_receipt_is_refused(self):
        path = self._receipt_path(self.refs[1])

        self.assertTrue(path.is_file())

        path.unlink()

        calls = []

        with _record_usage_spy(calls):
            result = await self._call(
                [self.refs[1]], **self._all_on()
            )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["reason"],
            "usage_ref_binding_invalid",
        )
        self.assertEqual(calls, [])
        self.assertEqual(
            read_usage(self.root)["used_count"], 0
        )
        self.assertFalse(_has_lifecycle_dirs(self.root))

    async def test_corrupt_issuance_receipt_is_refused(self):
        self._receipt_path(self.refs[1]).write_text(
            "{not-json", encoding="utf-8"
        )

        result = await self._call(
            [self.refs[1]], **self._all_on()
        )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["reason"],
            "usage_ref_binding_invalid",
        )
        self.assertEqual(
            read_usage(self.root)["used_count"], 0
        )

    async def test_receipt_filename_hash_mismatch_is_refused(
        self,
    ):
        # Copy the receipt of useref_A onto the path of useref_B.
        source = self._receipt_path(self.refs[1])

        target = self._receipt_path(self.refs[2])

        target.write_text(
            source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        self.assertIsNone(
            read_usage_ref_receipt(
                conversation_id=CID,
                cognitive_request_id=RID,
                ref_sha256=useref_sha256(self.refs[2]),
            )
        )

        result = await self._call(
            [self.refs[2]], **self._all_on()
        )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["reason"],
            "usage_ref_binding_invalid",
        )
        self.assertEqual(
            read_usage(self.root)["used_count"], 0
        )

    async def test_cross_request_receipt_is_refused(self):
        source = self._receipt_path(self.refs[1])

        other = usage_ref_receipt_path(
            CID, RID_B, useref_sha256(self.refs[1])
        )

        other.parent.mkdir(
            parents=True, exist_ok=True
        )

        other.write_text(
            source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        # The artifact still names request A, so it can never resolve
        # under request B.
        self.assertIsNone(
            read_usage_ref_receipt(
                conversation_id=CID,
                cognitive_request_id=RID_B,
                ref_sha256=useref_sha256(
                    self.refs[1]
                ),
            )
        )

    async def test_receipt_memory_id_tamper_is_refused(self):
        self._tamper_receipt(
            self.refs[1], memory_id="mem-2"
        )

        calls = []

        with _record_usage_spy(calls):
            result = await self._call(
                [self.refs[1]], **self._all_on()
            )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["reason"],
            "usage_ref_binding_invalid",
        )
        self.assertEqual(calls, [])
        self.assertFalse(_has_lifecycle_dirs(self.root))

    async def test_receipt_rank_tamper_is_refused(self):
        self._tamper_receipt(self.refs[1], rank=2)

        result = await self._call(
            [self.refs[1]], **self._all_on()
        )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["reason"],
            "usage_ref_binding_invalid",
        )

    async def test_receipt_kind_tamper_is_refused(self):
        self._tamper_receipt(
            self.refs[1], kind="related"
        )

        result = await self._call(
            [self.refs[1]], **self._all_on()
        )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["reason"],
            "usage_ref_binding_invalid",
        )

    async def test_receipt_recall_id_tamper_is_refused(self):
        self._tamper_receipt(
            self.refs[1],
            recall_id="recall_" + "2" * 32,
        )

        result = await self._call(
            [self.refs[1]], **self._all_on()
        )

        self.assertFalse(result["usage"]["recorded"])
        self.assertEqual(
            result["usage"]["reason"],
            "usage_ref_binding_invalid",
        )

    async def test_duplicate_recall_reuses_same_useref_and_receipt(
        self,
    ):
        first_refs = {
            ref["rank"]: ref["useref"]
            for ref in self.chain["surface"][
                "usage_refs"
            ]
        }

        second = build_live_recall_usage_surface(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_ref=self.chain["memrefs"][0],
            model_result=projected_model_result(
                _LOADED_IDS
            ),
        )

        self.assertTrue(second["stored"])
        self.assertEqual(second["reused_count"], 3)

        for ref in second["usage_refs"]:
            self.assertEqual(
                ref["useref"],
                first_refs[ref["rank"]],
            )

        for useref in first_refs.values():
            self.assertTrue(
                self._receipt_path(useref).is_file()
            )

    async def test_persistence_failure_never_exposes_useref(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR": root,
                    USAGE_ATTRIBUTION_ENV: "1",
                    USAGE_LIFECYCLE_INTEGRATION_ENV: "0",
                    LIFECYCLE_SHADOW_ENV: "0",
                },
                clear=False,
            ):
                memrefs = seed_live_exposure(
                    root,
                    [
                        memory("mem-1"),
                        memory("mem-2"),
                    ],
                )

                seed_recall_control_plane(
                    root,
                    list(_LOADED_IDS),
                )

                with patch(
                    "ombrebrain.context."
                    "live_recall_usage_ref_receipt."
                    "atomic_write",
                    side_effect=OSError("boom"),
                ):
                    report = (
                        build_live_recall_usage_surface(
                            conversation_id=CID,
                            cognitive_request_id=RID,
                            anchor_ref=memrefs[0],
                            model_result=(
                                projected_model_result(
                                    list(_LOADED_IDS)
                                )
                            ),
                        )
                    )

                surface_exists = (
                    live_recall_usage_surface_path(
                        CID, RID
                    ).exists()
                )

        self.assertFalse(report["stored"])
        self.assertEqual(report["usage_refs"], [])
        self.assertEqual(report["mapped_count"], 0)
        self.assertFalse(surface_exists)


class StaticForbiddenChecksTests(unittest.TestCase):
    """Static guards on the new integration module and its wiring."""

    _MODULE = (
        _REPO_ROOT
        / "src"
        / "ombrebrain"
        / "context"
        / "live_recall_usage_lifecycle_integration.py"
    )

    _RECEIPT_MODULE = (
        _REPO_ROOT
        / "src"
        / "ombrebrain"
        / "context"
        / "live_recall_usage_ref_receipt.py"
    )

    _FORBIDDEN_ATTRS = frozenset(
        {
            "touch",
            "archive",
            "update",
            "delete",
            "create",
            "save",
            "write",
            "glob",
            "rglob",
        }
    )

    _FORBIDDEN_SUBSTRINGS = (
        "embedding",
        "similarity",
        "cosine",
        "judge",
        "assistant",
        "answer",
        "retrieval",
    )

    def _modules(self):
        return (self._MODULE, self._RECEIPT_MODULE)

    def _tree(self, path):
        return ast.parse(
            path.read_text(encoding="utf-8")
        )

    def test_module_exists(self):
        for path in self._modules():
            with self.subTest(path=str(path)):
                self.assertTrue(path.is_file())

    def test_no_canonical_mutation_or_scan(self):
        for path in self._modules():
            tree = self._tree(path)

            with self.subTest(path=str(path)):
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(
                            node.func, ast.Attribute
                        )
                    ):
                        self.assertNotIn(
                            node.func.attr.lower(),
                            self._FORBIDDEN_ATTRS,
                        )

    def test_no_inference_or_retrieval_coupling(self):
        for path in self._modules():
            tree = self._tree(path)

            identifiers: set[str] = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    identifiers.add(node.id.lower())
                elif isinstance(node, ast.Attribute):
                    identifiers.add(node.attr.lower())
                elif isinstance(node, ast.alias):
                    identifiers.add(node.name.lower())
                elif (
                    isinstance(node, ast.ImportFrom)
                    and isinstance(node.module, str)
                ):
                    identifiers.add(node.module.lower())

            for forbidden in self._FORBIDDEN_SUBSTRINGS:
                with self.subTest(
                    path=str(path),
                    forbidden=forbidden,
                ):
                    self.assertFalse(
                        {
                            identifier
                            for identifier in identifiers
                            if forbidden in identifier
                        },
                        forbidden,
                    )

    def test_no_glob_scan(self):
        for path in self._modules():
            source = path.read_text(encoding="utf-8")

            with self.subTest(path=str(path)):
                self.assertNotIn(".glob(", source)
                self.assertNotIn(".rglob(", source)

    def test_receipt_path_uses_sha256_and_no_raw_useref(
        self,
    ):
        receipt = (
            _REPO_ROOT
            / "src"
            / "ombrebrain"
            / "context"
            / "live_recall_usage_ref_receipt.py"
        ).read_text(encoding="utf-8")

        # The raw handle is never a path component and is never
        # stored: the file name is the digest of the handle.
        self.assertIn(
            "useref.encode(", receipt
        )
        self.assertNotIn(
            '"useref":', receipt
        )

    def test_server_wires_the_coordinator_and_returns_only_usage(
        self,
    ):
        server = (
            _REPO_ROOT / "src" / "server.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "attribute_live_recall_usage_with_lifecycle(",
            server,
        )
        self.assertIn(
            'result["usage"]', server
        )
        self.assertIn(
            "render_live_recall_usage_ack(",
            server,
        )

        # The lifecycle report must never reach the model.
        self.assertNotIn('result["lifecycle"]', server)


if __name__ == "__main__":
    unittest.main()