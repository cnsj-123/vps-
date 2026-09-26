from __future__ import annotations

import tempfile
import unittest

from _recall_fixtures import (
    CID,
    RID,
    CID_B,
    RID_B,
    recall_artifact,
    recall_env,
    write_recall,
)

from ombrebrain.context.memory_usage_signal import (
    memory_usage_status,
    read_memory_usage,
    record_recall,
    record_usage,
)


RECALL_ID = "recall_" + "1" * 32


class RecordRecallTests(unittest.TestCase):
    def test_loaded_is_not_used(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                report = record_recall(
                    recall_artifact=recall_artifact(
                        ["M1", "M2", "M3"],
                        recall_id=RECALL_ID,
                    )
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
        # Being returned is not being used.
        self.assertEqual(
            usage["used_memory_ids"], []
        )
        self.assertEqual(usage["used_count"], 0)

    def test_event_stages_and_timestamps(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                record_recall(
                    recall_artifact=recall_artifact(
                        ["M1", "M2"],
                        recall_id=RECALL_ID,
                    )
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        stages = [
            event["stage"]
            for event in usage["events"]
        ]

        self.assertEqual(
            stages,
            [
                "recall_requested",
                "memory_loaded",
                "memory_loaded",
            ],
        )
        self.assertEqual(
            usage["events"][0]["memory_id"], "M1"
        )

        for event in usage["events"]:
            self.assertIn("at", event)

        self.assertIn(
            "requested_at", usage
        )

    def test_record_recall_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                first = record_recall(
                    recall_artifact=recall_artifact(
                        ["M1"],
                        recall_id=RECALL_ID,
                    )
                )

                second = record_recall(
                    recall_artifact=recall_artifact(
                        ["M1"],
                        recall_id=RECALL_ID,
                    )
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])

        # no double counting of recall_requested
        self.assertEqual(
            len(usage["events"]), 2
        )

    def test_malformed_recall_artifact_is_refused(
        self,
    ):
        report = record_recall(
            recall_artifact={"version": "nope"}
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
            record_recall(
                recall_artifact=recall_artifact(
                    memory_ids,
                    recall_id=RECALL_ID,
                )
            )

    def test_explicit_usage_records_subset(self):
        with tempfile.TemporaryDirectory() as root:
            self._seed(root, ["M1", "M2", "M3"])

            with recall_env(root):
                report = record_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                    used_memory_ids=["M1", "M3"],
                )

                usage = read_memory_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=RECALL_ID,
                )

        self.assertTrue(report["stored"])
        self.assertEqual(
            usage["used_memory_ids"],
            ["M1", "M3"],
        )
        # M2 was loaded but never declared used.
        self.assertNotIn(
            "M2", usage["used_memory_ids"]
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
            unknown["reason"],
            "recall_not_found",
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


if __name__ == "__main__":
    unittest.main()