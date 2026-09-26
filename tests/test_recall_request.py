from __future__ import annotations

import re
import tempfile
import unittest

from _recall_fixtures import (
    CID,
    CID_B,
    RID,
    RID_B,
    flash_artifact,
    memory,
    recall_env,
    write_flash,
)

from ombrebrain.context.recall_request import (
    authorize_recall_request,
    new_recall_id,
    recall_request_fingerprint,
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


if __name__ == "__main__":
    unittest.main()