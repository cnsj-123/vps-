from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _recall_fixtures import (
    CID,
    CID_B,
    RID,
    RID_B,
    recall_artifact,
    recall_env,
    recall_request as make_recall_request,
    write_recall,
)

from ombrebrain.context.memory_usage_signal import (
    is_valid_usage_artifact,
    memory_usage_status,
    read_memory_usage,
    record_memory_loaded,
    record_recall_requested,
    record_usage,
)
from ombrebrain.context.recall_request import (
    record_recall_request,
)


RECALL_ID = "recall_" + "1" * 32


def _usage_path(root):
    return (
        Path(root)
        / "memory_usage"
        / CID
        / RID
        / (RECALL_ID + ".json")
    )


def _request_path(root):
    return (
        Path(root)
        / "recall_request"
        / CID
        / RID
        / (RECALL_ID + ".json")
    )


def _seed_recall_request(
    root,
    anchor="M1",
    *,
    recall_id=RECALL_ID,
):
    with recall_env(root):
        return record_recall_request(
            recall_request=make_recall_request(
                anchor
            ),
            recall_id=recall_id,
        )


def _valid_usage(**overrides):
    usage = {
        "version": "memory-usage-signal.v1",
        "mode": "shadow_only",
        "conversation_id": CID,
        "cognitive_request_id": RID,
        "recall_id": RECALL_ID,
        "anchor_memory_id": "M1",
        "loaded_memory_ids": ["M1"],
        "used_memory_ids": [],
        "loaded_count": 1,
        "used_count": 0,
        "events": [
            {
                "stage": "recall_requested",
                "memory_id": "M1",
                "at": "t0",
            },
            {
                "stage": "memory_loaded",
                "memory_id": "M1",
                "at": "t1",
            },
        ],
    }

    usage.update(overrides)

    return usage


class RecordRecallRequestedTests(
    unittest.TestCase,
):
    def test_requested_only_state(self):
        with tempfile.TemporaryDirectory() as root:
            _seed_recall_request(root, "M1")

            with recall_env(root):
                report = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertTrue(report["stored"])
        self.assertEqual(
            usage["loaded_memory_ids"], []
        )
        self.assertEqual(
            usage["used_memory_ids"], []
        )
        self.assertEqual(usage["loaded_count"], 0)
        self.assertEqual(usage["used_count"], 0)
        self.assertEqual(
            [
                event["stage"]
                for event in usage["events"]
            ],
            ["recall_requested"],
        )
        # The anchor comes from the persisted Recall Request, never
        # from the caller.
        self.assertEqual(
            usage["anchor_memory_id"], "M1"
        )
        self.assertEqual(
            usage["events"][0]["memory_id"], "M1"
        )

    def test_requested_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            _seed_recall_request(root, "M1")

            with recall_env(root):
                first = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

                second = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(
            len(usage["events"]), 1
        )

    def test_invalid_inputs_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            _seed_recall_request(root, "M1")

            with recall_env(root):
                bad_binding = (
                    record_recall_requested(
                        conversation_id="nope",
                        cognitive_request_id=RID,
                        recall_id=RECALL_ID,
                    )
                )

                bad_id = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id="not-a-recall-id",
                )

                # A recall id with no persisted Recall Request can
                # never produce a recall_requested event.
                not_found = (
                    record_recall_requested(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        recall_id=(
                            "recall_" + "9" * 32
                        ),
                    )
                )

        self.assertEqual(
            bad_binding["reason"],
            "invalid_request_binding",
        )
        self.assertEqual(
            bad_id["reason"], "invalid_recall_id"
        )
        self.assertEqual(
            not_found["reason"],
            "recall_request_not_found",
        )

    def test_corrupt_existing_artifact_fails_closed(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            _seed_recall_request(root, "M1")

            with recall_env(root):
                record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

                path = _usage_path(root)

                corrupted = json.loads(
                    path.read_text(encoding="utf-8")
                )

                corrupted["conversation_id"] = CID_B

                path.write_text(
                    json.dumps(corrupted),
                    encoding="utf-8",
                )

                report = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "usage_artifact_invalid",
        )

    def test_phantom_requested_is_impossible(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                report = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

            usage_files = list(
                (
                    Path(root) / "memory_usage"
                ).rglob("*.json")
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "recall_request_not_found",
        )
        self.assertEqual(usage_files, [])

    def test_corrupt_recall_request_yields_no_usage(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            _seed_recall_request(root, "M1")

            path = _request_path(root)

            corrupted = json.loads(
                path.read_text(encoding="utf-8")
            )

            corrupted["mode"] = "live"

            path.write_text(
                json.dumps(corrupted),
                encoding="utf-8",
            )

            with recall_env(root):
                report = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

            usage_files = list(
                (
                    Path(root) / "memory_usage"
                ).rglob("*.json")
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "recall_request_not_found",
        )
        self.assertEqual(usage_files, [])


class RecordMemoryLoadedTests(
    unittest.TestCase,
):
    def _requested(self, root, anchor="M1"):
        _seed_recall_request(root, anchor)

        with recall_env(root):
            record_recall_requested(
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )

    def test_loaded_only_from_persisted_artifact(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            self._requested(root)

            write_recall(
                root,
                recall_artifact(
                    ["M1", "M2", "M3"],
                    recall_id=RECALL_ID,
                ),
            )

            with recall_env(root):
                report = record_memory_loaded(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertTrue(report["stored"])
        self.assertEqual(
            usage["loaded_memory_ids"],
            ["M1", "M2", "M3"],
        )
        # loaded is never used
        self.assertEqual(
            usage["used_memory_ids"], []
        )
        self.assertEqual(
            [
                event["stage"]
                for event in usage["events"]
            ],
            [
                "recall_requested",
                "memory_loaded",
                "memory_loaded",
                "memory_loaded",
            ],
        )

    def test_loaded_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            self._requested(root)

            write_recall(
                root,
                recall_artifact(
                    ["M1", "M2"],
                    recall_id=RECALL_ID,
                ),
            )

            with recall_env(root):
                record_memory_loaded(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

                second = record_memory_loaded(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertEqual(
            second["newly_loaded_count"], 0
        )
        self.assertEqual(
            len(usage["events"]), 3
        )

    def test_loaded_requires_requested_state(self):
        with tempfile.TemporaryDirectory() as root:
            # A persisted recall result exists, but no usage state.
            write_recall(
                root,
                recall_artifact(
                    ["M1"],
                    recall_id=RECALL_ID,
                ),
            )

            with recall_env(root):
                report = record_memory_loaded(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"], "usage_not_found"
        )

    def test_malformed_recall_artifact_is_refused(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            self._requested(root)

            # A persisted artifact that is malformed on disk must
            # never contribute loaded memories.
            write_recall(
                root,
                {
                    "version": "nope",
                    "conversation_id": CID,
                    "cognitive_request_id": RID,
                    "recall_id": RECALL_ID,
                },
            )

            with recall_env(root):
                report = record_memory_loaded(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "related_recall_not_found",
        )

    def test_synthetic_loaded_is_impossible(self):
        with tempfile.TemporaryDirectory() as root:
            # A real Recall Request + requested-only usage state, but
            # NO persisted related-recall artifact on disk.
            self._requested(root)

            # A synthetic recall dict cannot be handed in at all: the
            # API only accepts the request binding and re-reads disk.
            synthetic = {
                "version": "related-memory-recall.v1",
                "mode": "shadow_only",
                "conversation_id": CID,
                "cognitive_request_id": RID,
                "recall_id": RECALL_ID,
                "anchor_memory_id": "M1",
                "memories": [
                    {
                        "memory_id": "M999",
                        "content": "synthetic",
                        "reason": "anchor_memory",
                        "rank": 1,
                    }
                ],
            }

            with recall_env(root):
                with self.assertRaises(TypeError):
                    record_memory_loaded(
                        recall_artifact=synthetic
                    )

                report = record_memory_loaded(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "related_recall_not_found",
        )
        self.assertEqual(usage["loaded_memory_ids"], [])
        self.assertEqual(usage["used_memory_ids"], [])
        self.assertNotIn(
            "M999", usage["loaded_memory_ids"]
        )
        # The synthetic artifact is never trusted.
        self.assertNotIn(
            "M999",
            json.dumps(usage),
        )
        self.assertEqual(
            synthetic["memories"][0]["memory_id"],
            "M999",
        )

    def test_tampered_persisted_recall_cannot_load(
        self,
    ):
        for changes in (
            {"anchor_memory_id": "M2"},
            {"request_fingerprint": "0" * 64},
            {"requested_scope": "full"},
            {"source_flash_unified_revision": None},
            {"source_flash_request_id": RID_B},
        ):
            with self.subTest(changes=changes):
                with tempfile.TemporaryDirectory() as (
                    root
                ):
                    self._requested(root)

                    artifact = recall_artifact(
                        ["M1", "M2"],
                        recall_id=RECALL_ID,
                    )

                    # Keep the original fingerprint even when the
                    # anchor moves: the fingerprint no longer matches
                    # the recomputed one.
                    artifact.update(changes)

                    write_recall(root, artifact)

                    with recall_env(root):
                        report = (
                            record_memory_loaded(
                                conversation_id=CID,
                                cognitive_request_id=RID,
                                recall_id=RECALL_ID,
                            )
                        )

                        usage = read_memory_usage(
                            conversation_id=CID,
                            cognitive_request_id=RID,
                            recall_id=RECALL_ID,
                        )

                self.assertFalse(
                    report["stored"]
                )
                self.assertEqual(
                    report["reason"],
                    "related_recall_not_found",
                )
                self.assertEqual(
                    usage["loaded_memory_ids"], []
                )
                self.assertEqual(
                    usage["used_memory_ids"], []
                )


class RecordUsageTests(unittest.TestCase):
    def _seed(self, root, memory_ids):
        anchor = (
            memory_ids[0] if memory_ids else "M1"
        )

        _seed_recall_request(root, anchor)

        write_recall(
            root,
            recall_artifact(
                memory_ids,
                recall_id=RECALL_ID,
            ),
        )

        with recall_env(root):
            record_recall_requested(
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )

            record_memory_loaded(
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )

    def test_requested_loaded_used_state(self):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root, ["M1", "M2"])

            with recall_env(root):
                report = record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    used_memory_ids=["M1"],
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertTrue(report["stored"])
        self.assertEqual(
            usage["loaded_memory_ids"],
            ["M1", "M2"],
        )
        self.assertEqual(
            usage["used_memory_ids"], ["M1"]
        )
        self.assertEqual(
            [
                event["stage"]
                for event in usage["events"]
            ],
            [
                "recall_requested",
                "memory_loaded",
                "memory_loaded",
                "used",
            ],
        )

    def test_used_must_be_subset_of_loaded(self):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root, ["M1", "M2"])

            with recall_env(root):
                report = record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    used_memory_ids=["M9"],
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertEqual(
            usage["used_memory_ids"], []
        )
        self.assertEqual(
            report["rejected"],
            [
                {
                    "memory_id": "M9",
                    "reason": "memory_not_loaded",
                }
            ],
        )

    def test_invalid_used_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root, ["M1"])

            with recall_env(root):
                report = record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    used_memory_ids=[None, 17, ""],
                )

        reasons = {
            item["reason"]
            for item in report["rejected"]
        }

        self.assertEqual(
            reasons, {"invalid_memory_id"}
        )

    def test_usage_events_appended_once(self):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root, ["M1", "M2"])

            with recall_env(root):
                record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    used_memory_ids=["M1"],
                )

                record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    used_memory_ids=["M1"],
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        used_events = [
            event
            for event in usage["events"]
            if event["stage"] == "used"
        ]

        self.assertEqual(len(used_events), 1)
        self.assertEqual(
            usage["used_memory_ids"], ["M1"]
        )

    def test_usage_is_bound_to_the_exact_recall(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root, ["M1"])

            with recall_env(root):
                unknown = record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id="recall_" + "9" * 32,
                    used_memory_ids=["M1"],
                )

                wrong_request = record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID_B,
                    recall_id=RECALL_ID,
                    used_memory_ids=["M1"],
                )

                wrong_conversation = (
                    record_usage(
                        conversation_id=CID_B,
                        cognitive_request_id=RID,
                        recall_id=RECALL_ID,
                        used_memory_ids=["M1"],
                    )
                )

        self.assertEqual(
            unknown["reason"], "recall_not_found"
        )
        self.assertEqual(
            wrong_request["reason"],
            "recall_not_found",
        )
        self.assertEqual(
            wrong_conversation["reason"],
            "recall_not_found",
        )

    def test_invalid_binding_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                report = record_usage(
                    conversation_id="nope",
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    used_memory_ids=["M1"],
                )

        self.assertEqual(
            report["reason"],
            "invalid_request_binding",
        )

    def test_corrupt_usage_artifact_fails_closed(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root, ["M1", "M2"])

            path = _usage_path(root)

            corrupted = json.loads(
                path.read_text(encoding="utf-8")
            )

            # used must stay a subset of loaded: this artifact claims
            # a used memory that was never loaded.
            corrupted["used_memory_ids"] = ["M9"]
            corrupted["used_count"] = 1

            path.write_text(
                json.dumps(corrupted),
                encoding="utf-8",
            )

            with recall_env(root):
                report = record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    used_memory_ids=["M1"],
                )

                read_back = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

                status = memory_usage_status(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "usage_artifact_invalid",
        )
        self.assertIsNone(read_back)
        self.assertFalse(status["exists"])

    def test_status_is_privacy_safe(self):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root, ["M1", "M2"])

            with recall_env(root):
                status = memory_usage_status(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertTrue(status["exists"])
        self.assertEqual(
            status["loaded_count"], 2
        )
        self.assertEqual(
            status["used_count"], 0
        )
        self.assertEqual(
            status["event_count"], 3
        )
        self.assertNotIn(
            "loaded_memory_ids", status
        )


class UsageArtifactValidatorTests(
    unittest.TestCase,
):
    def test_validator_rejects_tampered_identity(self):
        usage = _valid_usage()

        self.assertTrue(
            is_valid_usage_artifact(
                usage,
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )
        )

        for field, value in (
            ("conversation_id", CID_B),
            ("cognitive_request_id", RID_B),
            ("recall_id", "recall_" + "2" * 32),
            ("mode", "live"),
            ("version", "memory-usage-signal.v2"),
        ):
            with self.subTest(field=field):
                tampered = dict(usage)
                tampered[field] = value

                self.assertFalse(
                    is_valid_usage_artifact(
                        tampered,
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        recall_id=RECALL_ID,
                    )
                )

    def test_validator_rejects_bad_anchor_binding(
        self,
    ):
        # Missing / empty anchor.
        for value in (None, "", "  ", 17):
            with self.subTest(anchor=value):
                self.assertFalse(
                    is_valid_usage_artifact(
                        _valid_usage(
                            anchor_memory_id=value
                        ),
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        recall_id=RECALL_ID,
                    )
                )

        # The single recall_requested event must name the anchor.
        mismatched = _valid_usage()
        mismatched["events"][0]["memory_id"] = "M2"

        self.assertFalse(
            is_valid_usage_artifact(
                mismatched,
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )
        )

        # Exactly one recall_requested event; two is corrupt.
        doubled = _valid_usage()
        doubled["events"].append(
            dict(doubled["events"][0])
        )

        self.assertFalse(
            is_valid_usage_artifact(
                doubled,
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )
        )

        # No recall_requested event at all.
        missing = _valid_usage()
        missing["events"] = [
            event
            for event in missing["events"]
            if event["stage"] != "recall_requested"
        ]

        self.assertFalse(
            is_valid_usage_artifact(
                missing,
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )
        )

    def test_validator_rejects_event_state_mismatch(
        self,
    ):
        # loaded array claims a memory with no memory_loaded event.
        loaded_mismatch = _valid_usage(
            loaded_memory_ids=["M1", "M2"],
            loaded_count=2,
        )

        self.assertFalse(
            is_valid_usage_artifact(
                loaded_mismatch,
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )
        )

        # A memory_loaded event with no matching array entry.
        event_only = _valid_usage()
        event_only["events"].append(
            {
                "stage": "memory_loaded",
                "memory_id": "M2",
                "at": "t2",
            }
        )

        self.assertFalse(
            is_valid_usage_artifact(
                event_only,
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )
        )

        # used array and used events must agree.
        used_mismatch = _valid_usage(
            used_memory_ids=["M1"],
            used_count=1,
        )

        self.assertFalse(
            is_valid_usage_artifact(
                used_mismatch,
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )
        )

        # A duplicate event id is corrupt even if the set matches.
        duplicated = _valid_usage()
        duplicated["events"].append(
            {
                "stage": "memory_loaded",
                "memory_id": "M1",
                "at": "t3",
            }
        )

        self.assertFalse(
            is_valid_usage_artifact(
                duplicated,
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
            )
        )


if __name__ == "__main__":
    unittest.main()