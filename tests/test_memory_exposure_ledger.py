from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ombrebrain.context.exposure_ledger import (
    build_exposure_ledger,
    exposure_ledger_status,
    retrieved_memory_ids,
    update_exposure_ledger,
)


CID = "ctx_0123456789abcdef"
RID = "ctxreq_0123456789abcdef0123456789abcdef"


def unified(
    memory_ids,
    *,
    revision=3,
    conversation_id=CID,
):
    memories = []

    for value in memory_ids:
        if value is None:
            memories.append({"content": "x"})
        else:
            memories.append(
                {
                    "id": value,
                    "content": "cue " + value,
                }
            )

    return {
        "version":
            "unified-context-candidate.v1",
        "conversation_id":
            conversation_id,
        "revision": revision,
        "sections": {
            "memories": memories,
        },
        "telemetry": {
            "current_user_excluded": True,
        },
    }


class RetrievedIdsTests(
    unittest.TestCase
):

    def test_retrieved_ids_follow_order_and_dedup(
        self,
    ):
        ids = retrieved_memory_ids(
            unified(["a", "b", "a", None, "c"])
        )

        self.assertEqual(
            ids,
            ["a", "b", "c"],
        )

    def test_non_dict_is_safe(self):
        for value in (
            None,
            5,
            "x",
            [],
        ):
            self.assertEqual(
                retrieved_memory_ids(value),
                [],
            )


class BuildLedgerTests(
    unittest.TestCase
):

    def test_surfaced_is_subset_of_retrieved(
        self,
    ):
        ledger = build_exposure_ledger(
            conversation_id=CID,
            cognitive_request_id=RID,
            source_unified_revision=3,
            retrieved_ids=["a", "b"],
            surfaced_ids=[
                "b",
                "not-retrieved",
                "b",
            ],
        )

        self.assertEqual(
            ledger["retrieved_memory_ids"],
            ["a", "b"],
        )
        self.assertEqual(
            ledger["surfaced_memory_ids"],
            ["b"],
        )
        self.assertEqual(
            ledger["retrieved_count"],
            2,
        )
        self.assertEqual(
            ledger["surfaced_count"],
            1,
        )

    def test_events_are_idempotent(
        self,
    ):
        ledger = build_exposure_ledger(
            conversation_id=CID,
            cognitive_request_id=RID,
            source_unified_revision=3,
            retrieved_ids=["a", "a", "b"],
            surfaced_ids=["a", "a"],
        )

        pairs: list[
            tuple[str, str]
        ] = []

        for event in ledger["events"]:
            pair = (
                event["memory_id"],
                event["stage"],
            )

            self.assertNotIn(
                pair,
                pairs,
            )
            pairs.append(pair)

        self.assertEqual(
            pairs,
            [
                ("a", "retrieved"),
                ("b", "retrieved"),
                ("a", "surfaced_as_flash"),
            ],
        )

    def test_only_two_stages_exist(self):
        ledger = build_exposure_ledger(
            conversation_id=CID,
            cognitive_request_id=RID,
            source_unified_revision=3,
            retrieved_ids=["a"],
            surfaced_ids=["a"],
        )

        for event in ledger["events"]:
            self.assertIn(
                event["stage"],
                ("retrieved", "surfaced_as_flash"),
            )

        serialized = json.dumps(ledger)

        for forbidden in (
            "noticed",
            "recall_requested",
            "full_memory_loaded",
            "used_in_response",
        ):
            self.assertNotIn(
                forbidden,
                serialized,
            )

    def test_ledger_carries_no_text(self):
        ledger = build_exposure_ledger(
            conversation_id=CID,
            cognitive_request_id=RID,
            source_unified_revision=3,
            retrieved_ids=["a"],
            surfaced_ids=["a"],
        )

        serialized = json.dumps(ledger)

        for forbidden in (
            "cue",
            "content",
            "raw",
            "query",
        ):
            self.assertNotIn(
                forbidden,
                serialized,
            )


class UpdateLedgerTests(
    unittest.TestCase
):

    def test_invalid_expected_revision(self):
        result = update_exposure_ledger(
            conversation_id=CID,
            cognitive_request_id=RID,
            unified=unified(["a"]),
            surfaced_ids=[],
            expected_unified_revision=0,
        )

        self.assertFalse(result["stored"])
        self.assertEqual(
            result["reason"],
            "invalid_expected_unified_revision",
        )

    def test_non_dict_unified(self):
        result = update_exposure_ledger(
            conversation_id=CID,
            cognitive_request_id=RID,
            unified="nope",
            surfaced_ids=[],
            expected_unified_revision=3,
        )

        self.assertFalse(result["stored"])
        self.assertEqual(
            result["reason"],
            "unified_observation_invalid",
        )

    def test_revision_mismatch_writes_nothing(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                result = update_exposure_ledger(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    unified=unified(
                        ["a"],
                        revision=4,
                    ),
                    surfaced_ids=[],
                    expected_unified_revision=3,
                )

                path = (
                    Path(root)
                    / "exposure_ledger"
                    / CID
                    / (RID + ".json")
                )

                self.assertFalse(
                    path.exists()
                )

        self.assertFalse(result["stored"])
        self.assertEqual(
            result["reason"],
            "ledger_unified_request_revision_mismatch",
        )

    def test_valid_ledger_is_written(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                result = update_exposure_ledger(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    unified=unified(
                        ["a", "b"]
                    ),
                    surfaced_ids=["a"],
                    expected_unified_revision=3,
                )

                path = (
                    Path(root)
                    / "exposure_ledger"
                    / CID
                    / (RID + ".json")
                )

                stored = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

        self.assertTrue(result["stored"])
        self.assertEqual(
            stored["retrieved_memory_ids"],
            ["a", "b"],
        )
        self.assertEqual(
            stored["surfaced_memory_ids"],
            ["a"],
        )
        self.assertEqual(
            stored["version"],
            "memory-exposure-ledger.v1",
        )
        self.assertEqual(
            stored["mode"],
            "shadow_only",
        )

    def test_ledger_never_stores_cue_or_raw(
        self,
    ):
        secret = "SECRET_MEMORY_TEXT"
        cue = "SECRET_CUE"

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                payload = unified(["m1"])
                payload["sections"][
                    "memories"
                ][0]["content"] = secret
                payload["sections"][
                    "memories"
                ][0]["metadata"] = {
                    "name": cue
                }

                update_exposure_ledger(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    unified=payload,
                    surfaced_ids=["m1"],
                    expected_unified_revision=3,
                )

                text = (
                    Path(root)
                    / "exposure_ledger"
                    / CID
                    / (RID + ".json")
                ).read_text(
                    encoding="utf-8"
                )

        self.assertNotIn(secret, text)
        self.assertNotIn(cue, text)

        # The memory id (the pointer a future Recall needs) is kept.
        self.assertIn("m1", text)

    def test_repeat_same_request_is_idempotent(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                kwargs = {
                    "conversation_id": CID,
                    "cognitive_request_id":
                        RID,
                    "unified": unified(
                        ["a", "b"]
                    ),
                    "surfaced_ids": ["a"],
                    "expected_unified_revision":
                        3,
                }

                first = (
                    update_exposure_ledger(
                        **kwargs
                    )
                )

                second = (
                    update_exposure_ledger(
                        **kwargs
                    )
                )

                path = (
                    Path(root)
                    / "exposure_ledger"
                    / CID
                    / (RID + ".json")
                )

                stored = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

        self.assertEqual(
            first["events"],
            second["events"],
        )
        self.assertEqual(
            len(stored["events"]),
            3,
        )

    def test_distinct_requests_never_overwrite(
        self,
    ):
        other = (
            "ctxreq_ffffffffffffffffffffffffffffffff"
        )

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                for request_id in (RID, other):
                    update_exposure_ledger(
                        conversation_id=CID,
                        cognitive_request_id=(
                            request_id
                        ),
                        unified=unified(["a"]),
                        surfaced_ids=[],
                        expected_unified_revision=3,
                    )

                directory = (
                    Path(root)
                    / "exposure_ledger"
                    / CID
                )

                files = sorted(
                    p.name
                    for p in directory.iterdir()
                )

        self.assertEqual(
            files,
            sorted(
                [
                    RID + ".json",
                    other + ".json",
                ]
            ),
        )

    def test_status_is_privacy_safe(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                update_exposure_ledger(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    unified=unified(["a"]),
                    surfaced_ids=["a"],
                    expected_unified_revision=3,
                )

                status = exposure_ledger_status(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                )

        self.assertTrue(status["exists"])
        self.assertEqual(
            status["retrieved_count"],
            1,
        )
        self.assertEqual(
            status["surfaced_count"],
            1,
        )
        self.assertNotIn(
            "memory_ids",
            json.dumps(status),
        )

    def test_missing_status(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                status = exposure_ledger_status(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                )

        self.assertEqual(
            status,
            {"exists": False},
        )

    def test_source_memory_file_unchanged(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            memory_file = (
                Path(root)
                / "buckets"
                / "m1.json"
            )

            memory_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            original = (
                b'{"id":"m1","content":"raw"}'
            )

            memory_file.write_bytes(original)

            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root,
                },
                clear=False,
            ):
                update_exposure_ledger(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    unified=unified(["m1"]),
                    surfaced_ids=["m1"],
                    expected_unified_revision=3,
                )

            self.assertEqual(
                memory_file.read_bytes(),
                original,
            )


if __name__ == "__main__":
    unittest.main()