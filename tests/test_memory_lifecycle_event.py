from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from concurrent.futures import (
    ThreadPoolExecutor,
)
from datetime import timedelta
from pathlib import Path

from _lifecycle_fixtures import (
    CID,
    M1,
    M2,
    M3,
    RID,
    T0,
    guarded_bucket_manager,
    iso,
    lifecycle_env,
    source_bucket,
    usage_artifact,
    write_usage,
)

from ombrebrain.context.memory_lifecycle_event import (
    LIFECYCLE_EVENT_VERSION,
    STAGE_USED,
    build_lifecycle_event,
    derive_lifecycle_event_id,
    is_valid_lifecycle_event,
    lifecycle_event_path,
    lifecycle_events_dir,
    memory_key,
    read_lifecycle_events,
    record_lifecycle_event_from_usage,
    # The low-level storage primitive. It is deliberately NOT the
    # authorization boundary: tests use it to seed and to exercise
    # atomicity / idempotency directly.
    _persist_verified_lifecycle_event,
)
from ombrebrain.context.memory_reinforcement import (
    derive_memory_lifecycle_state,
    observe_memory_usage_lifecycle,
)


RECALL_A = "recall_" + "1" * 32
RECALL_B = "recall_" + "2" * 32
RECALL_C = "recall_" + "3" * 32


def _event(
    memory_id: str = M1,
    *,
    index: int = 2,
    at: str | None = None,
    conversation_id: str = CID,
    cognitive_request_id: str = RID,
    recall_id: str = RECALL_A,
) -> dict:
    return build_lifecycle_event(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
        memory_id=memory_id,
        source_usage_event_at=(
            at or iso(T0)
        ),
        source_usage_event_index=index,
    )


class LifecycleEventPrimitiveTests(
    unittest.TestCase,
):
    def test_memory_key_is_digest_not_raw_id(
        self,
    ):
        key = memory_key("../../etc/passwd")

        self.assertIsNotNone(key)
        self.assertEqual(len(key), 64)
        self.assertEqual(
            key, key.lower()
        )
        self.assertNotIn(
            "etc", key
        )

    def test_memory_key_rejects_non_ids(self):
        for value in (None, "", "  ", 7, b"M1"):
            with self.subTest(value=value):
                self.assertIsNone(
                    memory_key(value)
                )

    def test_event_id_is_deterministic(self):
        first = derive_lifecycle_event_id(
            conversation_id=CID,
            cognitive_request_id=RID,
            recall_id=RECALL_A,
            memory_id=M1,
            stage=STAGE_USED,
            source_usage_event_at=iso(T0),
            source_usage_event_index=2,
        )

        second = derive_lifecycle_event_id(
            conversation_id=CID,
            cognitive_request_id=RID,
            recall_id=RECALL_A,
            memory_id=M1,
            stage=STAGE_USED,
            source_usage_event_at=iso(T0),
            source_usage_event_index=2,
        )

        self.assertEqual(first, second)
        self.assertTrue(
            first.startswith("mlcevt_")
        )
        self.assertEqual(len(first), 7 + 64)

    def test_event_id_changes_with_provenance(
        self,
    ):
        base = dict(
            conversation_id=CID,
            cognitive_request_id=RID,
            recall_id=RECALL_A,
            memory_id=M1,
            stage=STAGE_USED,
            source_usage_event_at=iso(T0),
            source_usage_event_index=2,
        )

        reference = (
            derive_lifecycle_event_id(**base)
        )

        for field, value in (
            ("memory_id", M2),
            ("recall_id", RECALL_B),
            ("cognitive_request_id",
             "ctxreq_" + "b" * 32),
            ("source_usage_event_index", 3),
            (
                "source_usage_event_at",
                iso(T0 + timedelta(seconds=1)),
            ),
            ("stage", "memory_loaded"),
        ):
            with self.subTest(field=field):
                variant = dict(base)
                variant[field] = value

                self.assertNotEqual(
                    derive_lifecycle_event_id(
                        **variant
                    ),
                    reference,
                )

    def test_event_path_uses_key_and_id(self):
        event = _event()

        path = lifecycle_event_path(
            event["memory_key"],
            event["event_id"],
        )

        self.assertIsNotNone(path)
        self.assertEqual(
            path.name,
            event["event_id"] + ".json",
        )
        self.assertEqual(
            path.parent.name,
            event["memory_key"],
        )
        self.assertNotIn(
            M1, str(path)
        )

    def test_built_event_is_used_only(self):
        event = _event()

        self.assertEqual(
            event["version"],
            LIFECYCLE_EVENT_VERSION,
        )
        self.assertEqual(event["mode"], "shadow_only")
        self.assertEqual(event["stage"], STAGE_USED)
        self.assertTrue(
            is_valid_lifecycle_event(event)
        )

        self.assertEqual(
            event["source_usage_version"],
            "memory-usage-signal.v1",
        )

    def test_persist_is_idempotent_and_immutable(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                event = _event()

                first = _persist_verified_lifecycle_event(
                    event
                )

                path = lifecycle_event_path(
                    event["memory_key"],
                    event["event_id"],
                )

                payload = path.read_bytes()

                second = _persist_verified_lifecycle_event(
                    _event()
                )

                self.assertTrue(first["stored"])
                self.assertFalse(
                    first["duplicate"]
                )

                self.assertTrue(second["stored"])
                self.assertTrue(
                    second["duplicate"]
                )
                self.assertEqual(
                    path.read_bytes(), payload
                )

    def test_persist_fails_closed_on_corrupt_file(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                event = _event()

                path = lifecycle_event_path(
                    event["memory_key"],
                    event["event_id"],
                )

                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                path.write_text(
                    json.dumps(
                        {
                            "version": "bogus",
                            "event_id":
                                event["event_id"],
                        }
                    ),
                    encoding="utf-8",
                )

                result = _persist_verified_lifecycle_event(
                    event
                )

                self.assertFalse(result["stored"])
                self.assertEqual(
                    result["reason"],
                    (
                        "existing_lifecycle_event"
                        "_invalid"
                    ),
                )

                # The corrupt file is never deleted or overwritten.
                self.assertTrue(path.is_file())
                self.assertEqual(
                    json.loads(
                        path.read_text(
                            encoding="utf-8"
                        )
                    )["version"],
                    "bogus",
                )


class LifecycleEventProvenanceTests(
    unittest.TestCase,
):
    def _persist(self, root, event):
        with lifecycle_env(root):
            result = _persist_verified_lifecycle_event(
                event
            )

        return result

    def _tampered(
        self,
    ) -> list[tuple[str, dict]]:
        cases: list[tuple[str, dict]] = []

        def add(name, mutator):
            event = _event()
            mutator(event)
            cases.append((name, event))

        def wrong_event_id(event):
            event["event_id"] = (
                "mlcevt_" + "0" * 64
            )

        def wrong_memory_key(event):
            event["memory_key"] = "0" * 64

        def wrong_memory_id(event):
            event["memory_id"] = M2

        def wrong_cid(event):
            event["conversation_id"] = (
                "ctx_fedcba9876543210"
            )

        def wrong_rid(event):
            event["cognitive_request_id"] = (
                "ctxreq_" + "b" * 32
            )

        def wrong_recall(event):
            event["recall_id"] = RECALL_B

        def wrong_stage(event):
            event["stage"] = "memory_loaded"

        def wrong_time(event):
            event["source_usage_event_at"] = (
                "not-a-timestamp"
            )

        def naive_time(event):
            event[
                "source_usage_event_at"
            ] = "2026-09-26T06:44:31"

        def wrong_version(event):
            event["version"] = "nope"

        def wrong_mode(event):
            event["mode"] = "live"

        def wrong_index(event):
            event["source_usage_event_index"] = (
                -1
            )

        def wrong_source(event):
            event[
                "source_usage_version"
            ] = "memory-usage-signal.v2"

        for name, mutator in (
            ("event_id", wrong_event_id),
            ("memory_key", wrong_memory_key),
            ("memory_id", wrong_memory_id),
            ("conversation_id", wrong_cid),
            ("cognitive_request_id", wrong_rid),
            ("recall_id", wrong_recall),
            ("stage", wrong_stage),
            ("source_time", wrong_time),
            ("naive_time", naive_time),
            ("version", wrong_version),
            ("mode", wrong_mode),
            ("index", wrong_index),
            ("source_version", wrong_source),
        ):
            add(name, mutator)

        return cases

    def test_tampered_events_are_invalid(self):
        for name, event in self._tampered():
            with self.subTest(name=name):
                self.assertFalse(
                    is_valid_lifecycle_event(
                        event
                    )
                )

    def test_tampered_events_never_counted_or_deleted(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                good = _event()
                self._persist(root, good)

                directory = lifecycle_events_dir(
                    good["memory_key"]
                )

                for name, event in self._tampered():
                    (
                        directory
                        / (
                            name
                            + "_"
                            + event["event_id"]
                            + ".json"
                        )
                    ).write_text(
                        json.dumps(event),
                        encoding="utf-8",
                    )

                store = read_lifecycle_events(M1)

                self.assertEqual(
                    len(store["events"]), 1
                )
                self.assertEqual(
                    store["events"][0]["event_id"],
                    good["event_id"],
                )
                self.assertEqual(
                    store["invalid_event_count"],
                    len(self._tampered()),
                )

                # Nothing was removed.
                self.assertEqual(
                    len(
                        list(
                            directory.glob("*.json")
                        )
                    ),
                    len(self._tampered()) + 1,
                )

    def test_filename_must_match_event_id(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                event = _event()

                directory = lifecycle_events_dir(
                    event["memory_key"]
                )

                directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                (
                    directory
                    / (
                        "mlcevt_"
                        + "9" * 64
                        + ".json"
                    )
                ).write_text(
                    json.dumps(event),
                    encoding="utf-8",
                )

                store = read_lifecycle_events(M1)

                self.assertEqual(
                    store["events"], []
                )
                self.assertEqual(
                    store["invalid_event_count"], 1
                )

    def test_read_only_scans_own_memory_dir(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _persist_verified_lifecycle_event(
                    _event(M1)
                )
                _persist_verified_lifecycle_event(
                    _event(
                        M2, recall_id=RECALL_B
                    )
                )

                m1 = read_lifecycle_events(M1)

                self.assertEqual(
                    len(m1["events"]), 1
                )
                self.assertEqual(
                    m1["events"][0]["memory_id"],
                    M1,
                )

                self.assertEqual(
                    read_lifecycle_events(
                        M3
                    )["events"],
                    [],
                )

    def test_read_events_never_returns_other_memory(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                event = _event(M1)

                # A file for M1's key that names M2 is invalid.
                tampered = dict(event)
                tampered["memory_id"] = M2
                tampered["memory_key"] = (
                    memory_key(M2)
                )

                directory = lifecycle_events_dir(
                    memory_key(M1)
                )

                directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                (
                    directory
                    / (event["event_id"] + ".json")
                ).write_text(
                    json.dumps(tampered),
                    encoding="utf-8",
                )

                store = read_lifecycle_events(M1)

                self.assertEqual(
                    store["events"], []
                )
                self.assertEqual(
                    store["invalid_event_count"], 1
                )


class LifecycleEventObserverTests(
    unittest.IsolatedAsyncioTestCase,
):
    async def test_persisted_used_creates_one_event(
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

                self.assertEqual(
                    report["used_event_count"], 1
                )
                self.assertEqual(
                    report["new_event_count"], 1
                )
                self.assertEqual(
                    report["invalid_event_count"], 0
                )

                store = read_lifecycle_events(M1)

        self.assertEqual(len(store["events"]), 1)
        self.assertEqual(
            store["events"][0]["stage"], "used"
        )

    async def test_same_usage_twice_is_one_event(
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
                first = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        as_of=T0,
                    )
                )

                second = (
                    await observe_memory_usage_lifecycle(
                        CID,
                        RID,
                        RECALL_A,
                        as_of=T0,
                    )
                )

                store = read_lifecycle_events(M1)

        self.assertEqual(
            first["new_event_count"], 1
        )
        self.assertEqual(
            second["new_event_count"], 0
        )
        self.assertEqual(
            second["duplicate_event_count"], 1
        )
        self.assertEqual(len(store["events"]), 1)

    async def test_loaded_but_unused_creates_no_event(
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
                        as_of=T0,
                    )
                )

                self.assertEqual(
                    report["new_event_count"], 2
                )

                self.assertEqual(
                    read_lifecycle_events(M2)[
                        "events"
                    ],
                    [],
                )

                self.assertEqual(
                    len(
                        read_lifecycle_events(M1)[
                            "events"
                        ]
                    ),
                    1,
                )

                self.assertEqual(
                    len(
                        read_lifecycle_events(M3)[
                            "events"
                        ]
                    ),
                    1,
                )

    async def test_non_used_stages_create_no_event(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_A,
                    loaded_memory_ids=(M1,),
                    used_memory_ids=(),
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

                events_dir = lifecycle_events_dir(
                    memory_key(M1)
                )

                self.assertEqual(
                    report["new_event_count"], 0
                )
                self.assertEqual(
                    report["used_event_count"], 0
                )
                self.assertEqual(
                    report["reason"], "no_used_events"
                )
                self.assertFalse(
                    events_dir.exists()
                )

    async def test_invalid_usage_creates_no_event(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            # A stage the Usage contract does not allow makes the
            # whole artifact invalid, so nothing may be reinforced.
            artifact = usage_artifact(
                recall_id=RECALL_A,
                loaded_memory_ids=(M1,),
                used_memory_ids=(M1,),
                at=iso(T0),
            )

            artifact["events"].append(
                {
                    "stage": "surfaced_as_flash",
                    "memory_id": M2,
                    "at": iso(T0),
                }
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
                    "usage_not_found_or_invalid",
                )
                self.assertFalse(
                    report["stored"]
                )
                self.assertFalse(
                    lifecycle_events_dir(
                        memory_key(M1)
                    ).exists()
                )

    async def test_missing_usage_creates_no_event(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
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
                    "usage_not_found_or_invalid",
                )
                self.assertFalse(
                    report["stored"]
                )

    async def test_no_event_for_other_recall_id(
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
                        RECALL_B,
                        as_of=T0,
                    )
                )

        self.assertEqual(
            report["reason"],
            "usage_not_found_or_invalid",
        )


class LifecycleEventProvenanceGateTests(
    unittest.IsolatedAsyncioTestCase,
):
    """Persistence requires a real, persisted Usage.used event.

    ``record_lifecycle_event_from_usage`` is the only authorized write
    path. A structurally valid artifact is NOT sufficient, so a
    "phantom" reinforcement receipt cannot be manufactured.
    """

    def test_api_cannot_be_told_provenance(self):
        parameters = inspect.signature(
            record_lifecycle_event_from_usage
        ).parameters

        self.assertEqual(
            list(parameters),
            [
                "conversation_id",
                "cognitive_request_id",
                "recall_id",
                "source_usage_event_index",
            ],
        )

        for forbidden in (
            "memory_id",
            "stage",
            "source_usage_event_at",
            "at",
            "usage",
            "event",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(
                    forbidden, parameters
                )

        with self.assertRaises(TypeError):
            record_lifecycle_event_from_usage(
                CID,
                RID,
                RECALL_A,
                2,
                memory_id=M2,
                source_usage_event_at=iso(T0),
            )

    async def test_phantom_reinforcement_is_refused(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                # No Usage artifact exists at all.
                result = (
                    record_lifecycle_event_from_usage(
                        CID,
                        RID,
                        RECALL_A,
                        0,
                    )
                )

                self.assertFalse(result["stored"])
                self.assertEqual(
                    result["reason"],
                    "usage_not_found_or_invalid",
                )
                self.assertIsNone(result["event_id"])

                self.assertEqual(
                    read_lifecycle_events(M1),
                    {
                        "events": [],
                        "event_file_count": 0,
                        "invalid_event_count": 0,
                    },
                )

                self.assertFalse(
                    lifecycle_events_dir(
                        memory_key(M1)
                    ).exists()
                )

                report = (
                    await derive_memory_lifecycle_state(
                        M1,
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
            report["state"]["use_count"], 0
        )
        self.assertAlmostEqual(
            report["state"]["strength"], 0.20
        )

    def test_corrupt_usage_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path = (
                Path(root)
                / "memory_usage"
                / CID
                / RID
                / (RECALL_A + ".json")
            )

            path.parent.mkdir(
                parents=True, exist_ok=True
            )

            path.write_text(
                json.dumps(
                    {
                        "version": (
                            "memory-usage-signal.v1"
                        ),
                        "mode": "shadow_only",
                    }
                ),
                encoding="utf-8",
            )

            with lifecycle_env(root):
                result = (
                    record_lifecycle_event_from_usage(
                        CID,
                        RID,
                        RECALL_A,
                        0,
                    )
                )

                self.assertFalse(result["stored"])
                self.assertEqual(
                    result["reason"],
                    "usage_not_found_or_invalid",
                )
                self.assertFalse(
                    lifecycle_events_dir(
                        memory_key(M1)
                    ).exists()
                )

    def test_index_must_be_a_strict_int(self):
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
                for index in (
                    -1,
                    99,
                    "2",
                    True,
                    False,
                    None,
                    1.5,
                ):
                    with self.subTest(index=index):
                        result = (
                            record_lifecycle_event_from_usage(
                                CID,
                                RID,
                                RECALL_A,
                                index,
                            )
                        )

                        self.assertFalse(
                            result["stored"]
                        )
                        self.assertEqual(
                            result["reason"],
                            "invalid_usage_event_index",
                        )

                self.assertEqual(
                    read_lifecycle_events(M1)[
                        "events"
                    ],
                    [],
                )

    def test_only_used_slots_are_reinforcement(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            artifact = usage_artifact(
                recall_id=RECALL_A,
                loaded_memory_ids=(M1, M2),
                used_memory_ids=(M2,),
                at=iso(T0),
            )

            # 0 recall_requested, 1 memory_loaded(M1),
            # 2 memory_loaded(M2), 3 used(M2)
            stages = [
                event["stage"]
                for event in artifact["events"]
            ]

            self.assertEqual(
                stages,
                [
                    "recall_requested",
                    "memory_loaded",
                    "memory_loaded",
                    "used",
                ],
            )

            write_usage(root, artifact)

            with lifecycle_env(root):
                for index in (0, 1, 2):
                    with self.subTest(index=index):
                        result = (
                            record_lifecycle_event_from_usage(
                                CID,
                                RID,
                                RECALL_A,
                                index,
                            )
                        )

                        self.assertFalse(
                            result["stored"]
                        )
                        self.assertEqual(
                            result["reason"],
                            "source_event_not_used",
                        )

                accepted = (
                    record_lifecycle_event_from_usage(
                        CID, RID, RECALL_A, 3
                    )
                )

                store = read_lifecycle_events(M2)

                self.assertTrue(accepted["stored"])
                self.assertFalse(
                    accepted["duplicate"]
                )
                self.assertEqual(
                    accepted["memory_id"], M2
                )
                self.assertEqual(
                    accepted[
                        "source_usage_event_index"
                    ],
                    3,
                )

        self.assertEqual(len(store["events"]), 1)
        self.assertEqual(
            store["events"][0]["memory_id"], M2
        )
        self.assertEqual(
            store["events"][0]["stage"], "used"
        )
        self.assertEqual(
            store["events"][0][
                "source_usage_event_at"
            ],
            iso(T0),
        )

    def test_provenance_comes_from_the_artifact(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            # The persisted usage says used=M1 at T0. The API cannot be
            # told a different memory or a different time, so a forged
            # receipt is not representable.
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
                result = (
                    record_lifecycle_event_from_usage(
                        CID, RID, RECALL_A, 3
                    )
                )

                store = read_lifecycle_events(M1)

        self.assertTrue(result["stored"])
        self.assertEqual(result["memory_id"], M1)
        self.assertEqual(
            result["memory_key"], memory_key(M1)
        )

        self.assertEqual(len(store["events"]), 1)

        event = store["events"][0]

        self.assertEqual(event["memory_id"], M1)
        self.assertEqual(event["stage"], "used")
        self.assertEqual(
            event["source_usage_event_at"],
            iso(T0),
        )
        self.assertEqual(
            event["source_usage_event_index"], 3
        )
        self.assertEqual(
            event["conversation_id"], CID
        )
        self.assertEqual(
            event["cognitive_request_id"], RID
        )
        self.assertEqual(
            event["recall_id"], RECALL_A
        )
        self.assertEqual(
            event["source_usage_version"],
            "memory-usage-signal.v1",
        )

        # No event was created for the loaded-but-unused M2.
        self.assertEqual(
            read_lifecycle_events(M2)["events"],
            [],
        )

    def test_hundred_reprocessings_one_receipt(
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
                results = [
                    record_lifecycle_event_from_usage(
                        CID, RID, RECALL_A, 2
                    )
                    for _ in range(100)
                ]

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
        self.assertEqual(
            sum(
                1
                for result in results
                if result["duplicate"]
            ),
            99,
        )

    def test_concurrent_duplicate_is_one_event(self):
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
                with ThreadPoolExecutor(
                    max_workers=8
                ) as pool:
                    results = list(
                        pool.map(
                            lambda _: (
                                record_lifecycle_event_from_usage(
                                    CID,
                                    RID,
                                    RECALL_A,
                                    2,
                                )
                            ),
                            range(16),
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

    def test_reader_does_not_need_the_usage_artifact(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            usage_path = write_usage(
                root,
                usage_artifact(
                    recall_id=RECALL_A,
                    loaded_memory_ids=(M1,),
                    used_memory_ids=(M1,),
                    at=iso(T0),
                ),
            )

            with lifecycle_env(root):
                record_lifecycle_event_from_usage(
                    CID, RID, RECALL_A, 2
                )

                # Authorization happened at write time; the receipt is
                # now history's own source of truth.
                usage_path.unlink()

                store = read_lifecycle_events(M1)

        self.assertEqual(len(store["events"]), 1)
        self.assertEqual(
            store["events"][0]["memory_id"], M1
        )


if __name__ == "__main__":
    unittest.main()