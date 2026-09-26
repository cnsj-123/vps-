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


RECALL_ID = "recall_" + "1" * 32


def _usage_path(root):
    return (
        Path(root)
        / "memory_usage"
        / CID
        / RID
        / (RECALL_ID + ".json")
    )


class RecordRecallRequestedTests(
    unittest.TestCase,
):
    def test_requested_only_state(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                report = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    anchor_memory_id="M1",
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
        self.assertEqual(
            usage["events"][0]["memory_id"], "M1"
        )

    def test_requested_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                first = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    anchor_memory_id="M1",
                )

                second = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    anchor_memory_id="M1",
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
            with recall_env(root):
                bad_binding = (
                    record_recall_requested(
                        conversation_id="nope",
                        cognitive_request_id=RID,
                        recall_id=RECALL_ID,
                        anchor_memory_id="M1",
                    )
                )

                bad_id = record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id="not-a-recall-id",
                    anchor_memory_id="M1",
                )

                bad_anchor = (
                    record_recall_requested(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        recall_id=RECALL_ID,
                        anchor_memory_id="",
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
            bad_anchor["reason"],
            "invalid_anchor_memory_id",
        )

    def test_corrupt_existing_artifact_fails_closed(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                record_recall_requested(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    anchor_memory_id="M1",
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
                    anchor_memory_id="M1",
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "usage_artifact_invalid",
        )


class RecordMemoryLoadedTests(
    unittest.TestCase,
):
    def _requested(self, root):
        with recall_env(root):
            record_recall_requested(
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=RECALL_ID,
                anchor_memory_id="M1",
            )

    def test_loaded_only_from_persisted_artifact(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            self._requested(root)

            artifact = recall_artifact(
                ["M1", "M2", "M3"],
                recall_id=RECALL_ID,
            )

            with recall_env(root):
                report = record_memory_loaded(
                    recall_artifact=artifact
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

            artifact = recall_artifact(
                ["M1", "M2"],
                recall_id=RECALL_ID,
            )

            with recall_env(root):
                record_memory_loaded(
                    recall_artifact=artifact
                )

                second = record_memory_loaded(
                    recall_artifact=artifact
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
            with recall_env(root):
                report = record_memory_loaded(
                    recall_artifact=recall_artifact(
                        ["M1"],
                        recall_id=RECALL_ID,
                    )
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"], "usage_not_found"
        )

    def test_malformed_recall_artifact_is_refused(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                report = record_memory_loaded(
                    recall_artifact={
                        "version": "nope"
                    }
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "malformed_recall_artifact",
        )


class RecordUsageTests(unittest.TestCase):
    def _seed(self, root, memory_ids):
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
                anchor_memory_id=(
                    memory_ids[0]
                    if memory_ids
                    else "M1"
                ),
            )

            record_memory_loaded(
                recall_artifact=recall_artifact(
                    memory_ids,
                    recall_id=RECALL_ID,
                )
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
        usage = {
            "version": "memory-usage-signal.v1",
            "mode": "shadow_only",
            "conversation_id": CID,
            "cognitive_request_id": RID,
            "recall_id": RECALL_ID,
            "loaded_memory_ids": ["M1"],
            "used_memory_ids": [],
            "loaded_count": 1,
            "used_count": 0,
            "events": [],
        }

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


if __name__ == "__main__":
    unittest.main()