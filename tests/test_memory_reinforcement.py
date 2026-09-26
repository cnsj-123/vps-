from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
import unittest
from concurrent.futures import (
    ThreadPoolExecutor,
)
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from _lifecycle_fixtures import (
    CID,
    FORBIDDEN_METHODS,
    LIFECYCLE_ENV,
    M1,
    M2,
    M3,
    RID,
    T0,
    guarded_bucket_manager,
    hours,
    iso,
    lifecycle_env,
    source_bucket,
    usage_artifact,
    write_usage,
)

from ombrebrain.context.memory_lifecycle_event import (
    build_lifecycle_event,
    lifecycle_events_dir,
    memory_key,
    persist_lifecycle_event,
    read_lifecycle_events,
)
from ombrebrain.context.memory_lifecycle_state import (
    lifecycle_state_path,
    read_memory_lifecycle_state,
)
from ombrebrain.context.memory_reinforcement import (
    derive_memory_lifecycle_state,
    memory_lifecycle_shadow_enabled,
    observe_memory_usage_lifecycle,
    process_completed_usage_for_lifecycle,
)


RECALL_A = "recall_" + "1" * 32
RECALL_B = "recall_" + "2" * 32
RID_B = "ctxreq_" + "b" * 32

ALLOWED_REPORT_KEYS = {
    "version",
    "mode",
    "stored",
    "decision",
    "reason",
    "used_event_count",
    "new_event_count",
    "duplicate_event_count",
    "invalid_event_count",
    "state_updated_count",
    "exists_count",
    "low_accessibility_count",
}


class GateBucketManager:
    """Canonical ``get`` that forces concurrent interleaving.

    Every party blocks until all parties have started, so idempotency
    and isolation are exercised for real without ever sleeping.
    """

    def __init__(self, buckets, *, parties=2):
        self.buckets = dict(buckets)
        self.parties = parties
        self.started = 0
        self.gate = asyncio.Event()

    async def get(self, bucket_id):
        self.started += 1

        if self.started >= self.parties:
            self.gate.set()

        await self.gate.wait()

        return self.buckets.get(bucket_id)


class RaisingBucketManager:
    async def get(self, bucket_id):
        raise RuntimeError("canonical read failed")


class FlagTests(unittest.TestCase):
    def test_flag_defaults_off(self):
        with patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop(LIFECYCLE_ENV, None)

            self.assertFalse(
                memory_lifecycle_shadow_enabled()
            )

    def test_flag_accepts_truthy_values(self):
        for value in ("1", "true", "YES", " on "):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ,
                    {LIFECYCLE_ENV: value},
                    clear=False,
                ):
                    self.assertTrue(
                        memory_lifecycle_shadow_enabled()
                    )


class ObserverApiTests(
    unittest.IsolatedAsyncioTestCase,
):
    async def test_disabled_flag_creates_nothing(
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

            with lifecycle_env(root, enabled=False):
                report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=(
                            guarded_bucket_manager(
                                {
                                    M1:
                                        source_bucket(
                                            M1
                                        )
                                }
                            )
                        ),
                        as_of=T0,
                    )
                )

                self.assertFalse(report["stored"])
                self.assertEqual(
                    report["reason"],
                    "lifecycle_disabled",
                )
                self.assertEqual(
                    report["new_event_count"], 0
                )

                self.assertFalse(
                    lifecycle_events_dir(
                        memory_key(M1)
                    ).exists()
                )
                self.assertFalse(
                    lifecycle_state_path(
                        memory_key(M1)
                    ).exists()
                )

    async def test_api_never_accepts_raw_usage(
        self,
    ):
        parameters = inspect.signature(
            observe_memory_usage_lifecycle
        ).parameters

        for forbidden in (
            "usage",
            "used",
            "used_memory_ids",
            "loaded_memory_ids",
            "memory_id",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(
                    forbidden, parameters
                )

        with self.assertRaises(TypeError):
            await observe_memory_usage_lifecycle(
                CID,
                RID,
                RECALL_A,
                usage={
                    "memory_id": M1,
                    "used": True,
                },
            )

    async def test_corrupt_usage_writes_no_event(
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

            path = (
                Path(root)
                / "memory_usage"
                / CID
                / RID
                / (RECALL_A + ".json")
            )

            path.write_text(
                json.dumps(
                    {
                        "version": (
                            "memory-usage-signal.v1"
                        ),
                        "mode": "shadow_only",
                        "used_memory_ids": [M1],
                    }
                ),
                encoding="utf-8",
            )

            with lifecycle_env(root):
                report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        as_of=T0,
                    )
                )

                self.assertFalse(report["stored"])
                self.assertEqual(
                    report["reason"],
                    "usage_not_found_or_invalid",
                )
                self.assertFalse(
                    lifecycle_events_dir(
                        memory_key(M1)
                    ).exists()
                )

    async def test_invalid_event_time_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            artifact = usage_artifact(
                recall_id=RECALL_A,
                loaded_memory_ids=(M1,),
                used_memory_ids=(M1,),
                at=iso(T0),
            )

            for event in artifact["events"]:
                if event["stage"] == "used":
                    event["at"] = "yesterday"

            write_usage(root, artifact)

            with lifecycle_env(root):
                report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        as_of=T0,
                    )
                )

                self.assertFalse(report["stored"])
                self.assertEqual(
                    report["reason"],
                    "invalid_usage_event_time",
                )
                self.assertEqual(
                    report["invalid_event_count"], 1
                )
                self.assertFalse(
                    lifecycle_events_dir(
                        memory_key(M1)
                    ).exists()
                )

    async def test_naive_event_time_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            artifact = usage_artifact(
                recall_id=RECALL_A,
                loaded_memory_ids=(M1,),
                used_memory_ids=(M1,),
                at="2026-09-26T06:44:31",
            )

            write_usage(root, artifact)

            with lifecycle_env(root):
                report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        as_of=T0,
                    )
                )

                self.assertEqual(
                    report["reason"],
                    "invalid_usage_event_time",
                )

    async def test_multiple_used_in_one_recall(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_A,
                    loaded_memory_ids=(M1, M2, M3),
                    used_memory_ids=(M1, M3),
                    at=iso(T0),
                ),
            )

            with lifecycle_env(root):
                report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=(
                            guarded_bucket_manager(
                                {
                                    M1:
                                        source_bucket(
                                            M1
                                        ),
                                    M3:
                                        source_bucket(
                                            M3
                                        ),
                                }
                            )
                        ),
                        as_of=T0,
                    )
                )

                m1 = read_memory_lifecycle_state(
                    M1
                )
                m3 = read_memory_lifecycle_state(
                    M3
                )

                self.assertEqual(
                    read_lifecycle_events(M2)[
                        "events"
                    ],
                    [],
                )

        self.assertEqual(
            report["new_event_count"], 2
        )
        self.assertEqual(m1["use_count"], 1)
        self.assertEqual(m3["use_count"], 1)

    async def test_duplicate_usage_does_not_restrengthen(
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
                await observe_memory_usage_lifecycle(
                    CID,
                    RID,
                    RECALL_A,
                    as_of=T0,
                )

                first = read_memory_lifecycle_state(
                    M1
                )

                second_report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        as_of=T0,
                    )
                )

                second = read_memory_lifecycle_state(
                    M1
                )

        self.assertEqual(
            second_report["new_event_count"], 0
        )
        self.assertEqual(
            second_report[
                "duplicate_event_count"
            ],
            1,
        )
        self.assertEqual(first["use_count"], 1)
        self.assertEqual(second["use_count"], 1)
        self.assertAlmostEqual(
            first["strength"],
            second["strength"],
        )

    async def test_missing_canonical_is_not_deleted(
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
                report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=(
                            guarded_bucket_manager(
                                {}
                            )
                        ),
                        as_of=T0,
                    )
                )

                state = read_memory_lifecycle_state(
                    M1
                )

                store = read_lifecycle_events(M1)

        self.assertTrue(report["stored"])
        self.assertEqual(report["exists_count"], 0)
        self.assertEqual(len(store["events"]), 1)
        self.assertEqual(state["use_count"], 1)
        self.assertFalse(state["exists"])

    async def test_raising_canonical_read_is_fail_open(
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
                report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=(
                            RaisingBucketManager()
                        ),
                        as_of=T0,
                    )
                )

                state = read_memory_lifecycle_state(
                    M1
                )

        self.assertTrue(report["stored"])
        self.assertEqual(report["exists_count"], 0)
        self.assertFalse(state["exists"])
        self.assertTrue(
            state["source_exists_checked"]
        )

    async def test_no_bucket_manager_still_derives(
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
                report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        as_of=T0,
                    )
                )

                state = read_memory_lifecycle_state(
                    M1
                )

        self.assertTrue(report["stored"])
        self.assertFalse(state["exists"])
        self.assertFalse(
            state["source_exists_checked"]
        )

    async def test_coordinator_hook_delegates(
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
                report = (
                    await process_completed_usage_for_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        as_of=T0,
                    )
                )

        self.assertTrue(report["stored"])
        self.assertEqual(
            report["new_event_count"], 1
        )


class NoLegacyMutationTests(
    unittest.IsolatedAsyncioTestCase,
):
    def _source_files(self, root):
        directory = Path(root) / "buckets"
        directory.mkdir()

        payloads = {}

        for memory_id in (M1, M2):
            path = directory / (
                memory_id + ".md"
            )

            path.write_text(
                "---\n"
                "id: "
                + memory_id
                + "\nactivation_count: 7\n"
                "last_active: "
                "2020-01-01T00:00:00Z\n"
                "importance: 9\n"
                "type: dynamic\n"
                "score: 0.87\n"
                "---\n"
                "body for "
                + memory_id
                + "\n",
                encoding="utf-8",
            )

            payloads[memory_id] = path

        return directory, payloads

    async def test_only_get_is_ever_called(self):
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

            manager = guarded_bucket_manager(
                {M1: source_bucket(M1)}
            )

            with lifecycle_env(root):
                await observe_memory_usage_lifecycle(
                    CID,
                    RID,
                    RECALL_A,
                    bucket_manager=manager,
                    as_of=T0,
                )

                await derive_memory_lifecycle_state(
                    M1,
                    bucket_manager=manager,
                    as_of=T0,
                )

        self.assertTrue(manager.get_calls)

        for name in FORBIDDEN_METHODS:
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(
                        manager, name
                    ).call_count,
                    0,
                )

    async def test_source_memory_is_byte_identical(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            directory, files = self._source_files(
                root
            )

            before = {
                path: path.read_bytes()
                for path in files.values()
            }

            listing = sorted(
                item.name
                for item in directory.iterdir()
            )

            buckets = {
                M1: source_bucket(M1),
                M2: source_bucket(M2),
            }

            pristine = deepcopy(buckets)

            write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_A,
                    loaded_memory_ids=(M1, M2),
                    used_memory_ids=(M1,),
                    at=iso(T0),
                ),
            )

            with lifecycle_env(root):
                await observe_memory_usage_lifecycle(
                    CID,
                    RID,
                    RECALL_A,
                    bucket_manager=(
                        guarded_bucket_manager(
                            buckets
                        )
                    ),
                    as_of=T0 + hours(48),
                )

                await derive_memory_lifecycle_state(
                    M1,
                    bucket_manager=(
                        guarded_bucket_manager(
                            buckets
                        )
                    ),
                    as_of=T0 + timedelta(days=30),
                )

            for path, payload in before.items():
                with self.subTest(path=path.name):
                    self.assertEqual(
                        path.read_bytes(), payload
                    )

            self.assertEqual(
                sorted(
                    item.name
                    for item in directory.iterdir()
                ),
                listing,
            )

            self.assertEqual(buckets, pristine)

    async def test_canonical_object_not_mutated(self):
        with tempfile.TemporaryDirectory() as root:
            buckets = {
                M1: source_bucket(M1)
            }

            pristine = deepcopy(buckets)

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
                    bucket_manager=(
                        guarded_bucket_manager(
                            buckets
                        )
                    ),
                    as_of=T0,
                )

        self.assertEqual(buckets, pristine)
        self.assertEqual(
            buckets[M1]["metadata"][
                "activation_count"
            ],
            7,
        )
        self.assertEqual(
            buckets[M1]["metadata"]["importance"],
            9,
        )
        self.assertEqual(
            buckets[M1]["metadata"]["last_active"],
            "2020-01-01T00:00:00Z",
        )


class ConcurrencyTests(
    unittest.IsolatedAsyncioTestCase,
):
    async def test_same_usage_concurrent_is_one_event(
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

            manager = GateBucketManager(
                {M1: source_bucket(M1)},
                parties=2,
            )

            with lifecycle_env(root):
                first, second = await asyncio.gather(
                    observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=manager,
                        as_of=T0,
                    ),
                    observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=manager,
                        as_of=T0,
                    ),
                )

                store = read_lifecycle_events(M1)

                state = read_memory_lifecycle_state(
                    M1
                )

        self.assertEqual(len(store["events"]), 1)
        self.assertEqual(
            first["new_event_count"]
            + second["new_event_count"],
            1,
        )
        self.assertEqual(state["use_count"], 1)

    async def test_two_recalls_concurrent_accumulate(
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

            write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_B,
                    cognitive_request_id=RID_B,
                    loaded_memory_ids=(M1,),
                    used_memory_ids=(M1,),
                    at=iso(T0 + hours(1)),
                ),
            )

            manager = GateBucketManager(
                {M1: source_bucket(M1)},
                parties=2,
            )

            with lifecycle_env(root):
                await asyncio.gather(
                    observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=manager,
                        as_of=T0 + hours(1),
                    ),
                    observe_memory_usage_lifecycle(
                        CID,
                        RID_B,
                        RECALL_B,
                        bucket_manager=manager,
                        as_of=T0 + hours(1),
                    ),
                )

                store = read_lifecycle_events(M1)

                state = read_memory_lifecycle_state(
                    M1
                )

        self.assertEqual(len(store["events"]), 2)
        self.assertEqual(state["use_count"], 2)
        self.assertAlmostEqual(
            state["strength"], 0.422, places=3
        )

    async def test_different_memories_stay_isolated(
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

            write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_B,
                    cognitive_request_id=RID_B,
                    loaded_memory_ids=(M2,),
                    used_memory_ids=(M2,),
                    at=iso(T0),
                ),
            )

            with lifecycle_env(root):
                await asyncio.gather(
                    observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=(
                            guarded_bucket_manager(
                                {
                                    M1:
                                        source_bucket(
                                            M1
                                        )
                                }
                            )
                        ),
                        as_of=T0,
                    ),
                    observe_memory_usage_lifecycle(
                        CID,
                        RID_B,
                        RECALL_B,
                        bucket_manager=(
                            guarded_bucket_manager(
                                {
                                    M2:
                                        source_bucket(
                                            M2
                                        )
                                }
                            )
                        ),
                        as_of=T0,
                    ),
                )

                m1 = read_memory_lifecycle_state(
                    M1
                )
                m2 = read_memory_lifecycle_state(
                    M2
                )

        self.assertEqual(m1["memory_id"], M1)
        self.assertEqual(m2["memory_id"], M2)
        self.assertEqual(m1["use_count"], 1)
        self.assertEqual(m2["use_count"], 1)
        self.assertEqual(len(m1["memory_key"]), 64)
        self.assertNotEqual(
            m1["memory_key"], m2["memory_key"]
        )

    async def test_threaded_persist_is_idempotent(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                event = build_lifecycle_event(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_A,
                    memory_id=M1,
                    source_usage_event_at=iso(T0),
                    source_usage_event_index=2,
                )

                with ThreadPoolExecutor(
                    max_workers=4
                ) as pool:
                    results = list(
                        pool.map(
                            lambda _: (
                                persist_lifecycle_event(
                                    event
                                )
                            ),
                            range(8),
                        )
                    )

                store = read_lifecycle_events(M1)

        self.assertEqual(len(store["events"]), 1)
        self.assertEqual(
            sum(
                1
                for result in results
                if result["stored"]
                and not result["duplicate"]
            ),
            1,
        )


class ReportPrivacyTests(
    unittest.IsolatedAsyncioTestCase,
):
    async def test_report_has_only_allowed_keys(
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
                report = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        bucket_manager=(
                            guarded_bucket_manager(
                                {
                                    M1:
                                        source_bucket(
                                            M1
                                        )
                                }
                            )
                        ),
                        as_of=T0,
                    )
                )

        self.assertEqual(
            set(report), ALLOWED_REPORT_KEYS
        )

        rendered = json.dumps(report)

        for secret in (
            M1,
            CID,
            RID,
            RECALL_A,
            memory_key(M1),
            "body for",
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()