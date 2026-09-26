from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from _recall_fixtures import (
    CID,
    CID_B,
    RID,
    RID_B,
    flash_artifact,
    memory,
    recall_env,
    recall_request,
    write_flash,
)

from ombrebrain.context.recall_request import (
    authorize_recall_request,
    find_recall_request_by_fingerprint,
    is_valid_recall_request_artifact,
    new_recall_id,
    read_recall_request,
    recall_request_fingerprint,
    recall_request_status,
    record_recall_request,
)


_RECALL_ID_RE = re.compile(
    r"^recall_[0-9a-f]{32}$"
)


class RecallIdTests(unittest.TestCase):
    def test_recall_id_format(self):
        recall_id = new_recall_id()

        self.assertRegex(
            recall_id, _RECALL_ID_RE
        )

    def test_recall_ids_are_unique(self):
        ids = {new_recall_id() for _ in range(500)}

        self.assertEqual(len(ids), 500)

    def test_fingerprint_is_deterministic_and_specific(
        self,
    ):
        first = recall_request_fingerprint(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_memory_id="mem-1",
            requested_scope="related",
        )

        self.assertEqual(
            first,
            recall_request_fingerprint(
                conversation_id=CID,
                cognitive_request_id=RID,
                anchor_memory_id="mem-1",
                requested_scope="related",
            ),
        )

        other_anchor = recall_request_fingerprint(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_memory_id="mem-2",
            requested_scope="related",
        )

        self.assertNotEqual(first, other_anchor)

        other_scope = recall_request_fingerprint(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_memory_id="mem-1",
            requested_scope="full",
        )

        self.assertNotEqual(first, other_scope)


class RecallAuthorizationTests(unittest.TestCase):
    def _authorize(
        self,
        root,
        *,
        conversation_id=CID,
        cognitive_request_id=RID,
        anchor="mem-1",
        revision=None,
        scope="related",
        enabled=True,
    ):
        return authorize_recall_request(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            anchor_memory_id=anchor,
            requested_scope=scope,
            source_flash_unified_revision=(
                revision
            ),
        )

    def test_surfaced_anchor_is_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [
                        memory("mem-1"),
                        memory("mem-2"),
                    ]
                ),
            )

            with recall_env(root):
                result = self._authorize(
                    root, anchor="mem-1"
                )

        self.assertTrue(result["valid"])
        self.assertEqual(
            result["recall_request"][
                "anchor_memory_id"
            ],
            "mem-1",
        )
        self.assertEqual(
            result["recall_request"][
                "version"
            ],
            "memory-recall-request.v1",
        )
        self.assertEqual(
            result["recall_request"][
                "requested_scope"
            ],
            "related",
        )

    def test_retrieved_but_not_surfaced_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            # mem-2 was retrieved (it exists) but the Flash only
            # surfaced mem-1.
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            with recall_env(root):
                result = self._authorize(
                    root, anchor="mem-2"
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "anchor_not_surfaced",
        )

    def test_nonexistent_anchor_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            with recall_env(root):
                result = self._authorize(
                    root, anchor="mem-9"
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "anchor_not_surfaced",
        )

    def test_other_request_flash_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [memory("mem-3")],
                    cognitive_request_id=RID_B,
                ),
            )

            with recall_env(root):
                result = self._authorize(
                    root, anchor="mem-3"
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "flash_not_found",
        )

    def test_other_conversation_flash_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [memory("mem-4")],
                    conversation_id=CID_B,
                ),
            )

            with recall_env(root):
                result = self._authorize(
                    root, anchor="mem-4"
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "flash_not_found",
        )

    def test_wrong_cognitive_request_id_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            with recall_env(root):
                result = self._authorize(
                    root,
                    cognitive_request_id=RID_B,
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "flash_not_found",
        )

    def test_wrong_source_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [memory("mem-1")],
                    revision=5,
                ),
            )

            with recall_env(root):
                result = self._authorize(
                    root,
                    anchor="mem-1",
                    revision=6,
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "source_revision_mismatch",
        )

    def test_optional_revision_defaults_to_flash(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [memory("mem-1")],
                    revision=7,
                ),
            )

            with recall_env(root):
                result = self._authorize(
                    root, anchor="mem-1"
                )

        self.assertTrue(result["valid"])
        self.assertEqual(
            result["recall_request"][
                "source_flash_unified_revision"
            ],
            7,
        )

    def test_malformed_flash_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            artifact = flash_artifact(
                [memory("mem-1")]
            )
            artifact["version"] = "memory-flash.v2"
            write_flash(root, artifact)

            with recall_env(root):
                result = self._authorize(
                    root, anchor="mem-1"
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "flash_identity_mismatch",
        )

    def test_flash_without_surfaced_decision_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact(
                    [memory("mem-1")],
                    decision="no_surface",
                ),
            )

            with recall_env(root):
                result = self._authorize(
                    root, anchor="mem-1"
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "flash_not_found",
        )

    def test_missing_flash_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                result = self._authorize(
                    root, anchor="mem-1"
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "flash_not_found",
        )

    def test_full_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            with recall_env(root):
                result = self._authorize(
                    root,
                    anchor="mem-1",
                    scope="full",
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "invalid_requested_scope",
        )

    def test_recall_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            with recall_env(
                root,
                enabled=False,
                shadow=False,
            ):
                result = self._authorize(
                    root, anchor="mem-1"
                )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "recall_disabled",
        )

    def test_invalid_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_flash(
                root,
                flash_artifact([memory("mem-1")]),
            )

            with recall_env(root):
                bad_conversation = (
                    self._authorize(
                        root,
                        conversation_id="nope",
                        anchor="mem-1",
                    )
                )

                bad_anchor = self._authorize(
                    root, anchor=""
                )

        self.assertEqual(
            bad_conversation["reason"],
            "invalid_request",
        )
        self.assertEqual(
            bad_anchor["reason"],
            "invalid_request",
        )


class RecallRequestPersistenceTests(
    unittest.TestCase,
):
    def _path(self, root, recall_id):
        return (
            Path(root)
            / "recall_request"
            / CID
            / RID
            / (recall_id + ".json")
        )

    def test_authorized_request_is_persisted(self):
        request = recall_request("mem-1")

        recall_id = new_recall_id()

        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                report = record_recall_request(
                    recall_request=request,
                    recall_id=recall_id,
                )

                artifact = read_recall_request(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=recall_id,
                )

                status = recall_request_status(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=recall_id,
                )

        self.assertTrue(report["stored"])
        self.assertFalse(report["duplicate"])
        self.assertEqual(
            report["recall_id"], recall_id
        )
        self.assertEqual(
            artifact["status"], "requested"
        )
        self.assertEqual(
            artifact["anchor_memory_id"], "mem-1"
        )
        self.assertEqual(
            artifact["requested_scope"], "related"
        )
        self.assertEqual(
            artifact["source_flash_unified_revision"],
            5,
        )
        self.assertTrue(status["exists"])
        self.assertEqual(
            status["status"], "requested"
        )

    def test_record_is_idempotent_by_fingerprint(
        self,
    ):
        request = recall_request("mem-1")

        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                first = record_recall_request(
                    recall_request=request,
                    recall_id=new_recall_id(),
                )

                second = record_recall_request(
                    recall_request=request,
                    recall_id=new_recall_id(),
                )

                files = list(
                    (
                        Path(root)
                        / "recall_request"
                        / CID
                        / RID
                    ).glob("recall_*.json")
                )

        self.assertEqual(
            first["recall_id"],
            second["recall_id"],
        )
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(files), 1)

    def test_find_by_fingerprint(self):
        request = recall_request("mem-1")

        fingerprint = recall_request_fingerprint(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_memory_id="mem-1",
            requested_scope="related",
        )

        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                record_recall_request(
                    recall_request=request,
                    recall_id=new_recall_id(),
                )

                found = (
                    find_recall_request_by_fingerprint(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        fingerprint=fingerprint,
                    )
                )

                unknown = (
                    find_recall_request_by_fingerprint(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        fingerprint="0" * 64,
                    )
                )

        self.assertIsInstance(found, dict)
        self.assertIsNone(unknown)

    def test_invalid_inputs_are_refused(self):
        request = recall_request("mem-1")

        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                bad_id = record_recall_request(
                    recall_request=request,
                    recall_id="not-a-recall-id",
                )

                bad_request = record_recall_request(
                    recall_request={"version": "nope"},
                    recall_id=new_recall_id(),
                )

        self.assertEqual(
            bad_id["reason"], "invalid_recall_id"
        )
        self.assertEqual(
            bad_request["reason"],
            "invalid_recall_request",
        )

    def test_corrupt_request_identity_is_rejected(
        self,
    ):
        request = recall_request("mem-1")
        recall_id = new_recall_id()

        fingerprint = recall_request_fingerprint(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_memory_id="mem-1",
            requested_scope="related",
        )

        for changes in (
            {"conversation_id": CID_B},
            {"cognitive_request_id": RID_B},
            {"mode": "live"},
            {"version": "memory-recall-request.v2"},
            {"source_flash_unified_revision": None},
            {"requested_scope": "full"},
            {"anchor_memory_id": ""},
            {"recall_id": "nope"},
            {"request_fingerprint": ""},
            {"request_fingerprint": "z" * 64},
            {"request_fingerprint": "A" * 64},
        ):
            with self.subTest(changes=changes):
                with tempfile.TemporaryDirectory() as (
                    root
                ):
                    with recall_env(root):
                        record_recall_request(
                            recall_request=request,
                            recall_id=recall_id,
                        )

                    path = self._path(
                        root, recall_id
                    )

                    artifact = json.loads(
                        path.read_text(
                            encoding="utf-8"
                        )
                    )

                    artifact.update(changes)

                    path.write_text(
                        json.dumps(artifact),
                        encoding="utf-8",
                    )

                    with recall_env(root):
                        read_back = read_recall_request(
                            conversation_id=CID,
                            cognitive_request_id=RID,
                            recall_id=recall_id,
                        )

                        found = (
                            find_recall_request_by_fingerprint(
                                conversation_id=CID,
                                cognitive_request_id=RID,
                                fingerprint=fingerprint,
                            )
                        )

                        self.assertFalse(
                            is_valid_recall_request_artifact(
                                artifact,
                                conversation_id=CID,
                                cognitive_request_id=RID,
                            )
                        )

                self.assertIsNone(read_back)
                self.assertIsNone(found)

    def test_stale_fingerprint_after_anchor_tamper(
        self,
    ):
        request = recall_request("mem-1")
        recall_id = new_recall_id()

        fingerprint_m1 = recall_request_fingerprint(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_memory_id="mem-1",
            requested_scope="related",
        )

        fingerprint_m2 = recall_request_fingerprint(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_memory_id="mem-2",
            requested_scope="related",
        )

        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                record_recall_request(
                    recall_request=request,
                    recall_id=recall_id,
                )

            path = self._path(root, recall_id)

            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )

            # Move the anchor but keep the original fingerprint: the
            # recomputed fingerprint no longer matches.
            artifact["anchor_memory_id"] = "mem-2"

            path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )

            with recall_env(root):
                self.assertFalse(
                    is_valid_recall_request_artifact(
                        artifact,
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

                read_back = read_recall_request(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=recall_id,
                )

                found_stale = (
                    find_recall_request_by_fingerprint(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        fingerprint=fingerprint_m1,
                    )
                )

                found_current = (
                    find_recall_request_by_fingerprint(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        fingerprint=fingerprint_m2,
                    )
                )

        self.assertIsNone(read_back)
        self.assertIsNone(found_stale)
        self.assertIsNone(found_current)

    def test_foreign_valid_fingerprint_is_rejected(
        self,
    ):
        request = recall_request("mem-1")
        recall_id = new_recall_id()

        correct = recall_request_fingerprint(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_memory_id="mem-1",
            requested_scope="related",
        )

        foreign = "b" * 64

        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                record_recall_request(
                    recall_request=request,
                    recall_id=recall_id,
                )

            path = self._path(root, recall_id)

            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )

            artifact["request_fingerprint"] = foreign

            path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )

            with recall_env(root):
                self.assertFalse(
                    is_valid_recall_request_artifact(
                        artifact,
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

                read_back = read_recall_request(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=recall_id,
                )

                found_foreign = (
                    find_recall_request_by_fingerprint(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        fingerprint=foreign,
                    )
                )

                found_correct = (
                    find_recall_request_by_fingerprint(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        fingerprint=correct,
                    )
                )

        self.assertIsNone(read_back)
        self.assertIsNone(found_foreign)
        self.assertIsNone(found_correct)

    def test_source_flash_request_id_is_bound(self):
        request = recall_request("mem-1")
        recall_id = new_recall_id()

        with tempfile.TemporaryDirectory() as root:
            with recall_env(root):
                record_recall_request(
                    recall_request=request,
                    recall_id=recall_id,
                )

            path = self._path(root, recall_id)

            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )

            # The v1 Flash is request scoped: a Recall Request bound
            # to another request's Flash is not trusted.
            artifact["source_flash_request_id"] = RID_B

            path.write_text(
                json.dumps(artifact),
                encoding="utf-8",
            )

            with recall_env(root):
                self.assertFalse(
                    is_valid_recall_request_artifact(
                        artifact,
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

                read_back = read_recall_request(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    recall_id=recall_id,
                )

        self.assertIsNone(read_back)

    def test_status_must_be_requested(self):
        request = recall_request("mem-1")
        recall_id = new_recall_id()

        for value in (
            "loaded",
            "used",
            "complete",
            None,
            "",
        ):
            with self.subTest(status=value):
                with tempfile.TemporaryDirectory() as (
                    root
                ):
                    with recall_env(root):
                        record_recall_request(
                            recall_request=request,
                            recall_id=recall_id,
                        )

                    path = self._path(
                        root, recall_id
                    )

                    artifact = json.loads(
                        path.read_text(
                            encoding="utf-8"
                        )
                    )

                    artifact["status"] = value

                    path.write_text(
                        json.dumps(artifact),
                        encoding="utf-8",
                    )

                    with recall_env(root):
                        self.assertFalse(
                            is_valid_recall_request_artifact(
                                artifact,
                                conversation_id=CID,
                                cognitive_request_id=RID,
                            )
                        )

                        read_back = (
                            read_recall_request(
                                conversation_id=CID,
                                cognitive_request_id=RID,
                                recall_id=recall_id,
                            )
                        )

                self.assertIsNone(read_back)


if __name__ == "__main__":
    unittest.main()