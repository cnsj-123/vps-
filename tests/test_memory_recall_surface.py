from __future__ import annotations

import json
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
)

from ombrebrain.context.memory_recall_surface import (
    build_recall_surface,
    new_memref,
    recall_surface_for_model,
    recall_surface_status,
    resolve_recall_ref,
    update_recall_surface,
)


_MEMREF_RE = re.compile(r"^memref_[0-9a-f]{32}$")

_REAL_IDS = ["mem-1", "mem-2", "mem-3"]


class MemrefTests(unittest.TestCase):
    def test_memref_format(self):
        self.assertRegex(
            new_memref(), _MEMREF_RE
        )

    def test_memrefs_are_unique(self):
        ids = {new_memref() for _ in range(500)}

        self.assertEqual(len(ids), 500)


class RecallSurfaceTests(unittest.TestCase):
    def _flash(self):
        return flash_artifact(
            [
                memory(
                    "mem-1",
                    "alpha note",
                    name="merge rules",
                ),
                memory("mem-2", "beta note"),
                memory("mem-3", "gamma note"),
            ]
        )

    def test_surface_is_built_and_resolves(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root, surface=True):
                report = update_recall_surface(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    flash_report=self._flash(),
                )

                model_view = (
                    recall_surface_for_model(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

                memrefs = [
                    surface["memref"]
                    for surface in model_view[
                        "surfaces"
                    ]
                ]

                resolved = [
                    resolve_recall_ref(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        memref=memref,
                    )
                    for memref in memrefs
                ]

        self.assertTrue(report["stored"])
        self.assertEqual(report["memory_count"], 3)
        self.assertEqual(
            resolved, _REAL_IDS
        )

    def test_model_view_never_leaks_real_ids(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root, surface=True):
                update_recall_surface(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    flash_report=self._flash(),
                )

                model_view = (
                    recall_surface_for_model(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

        blob = json.dumps(model_view)

        for real_id in _REAL_IDS:
            self.assertNotIn(real_id, blob)

        self.assertNotIn("memory_id", blob)
        self.assertNotIn("mapping", blob)

        for surface in model_view["surfaces"]:
            self.assertRegex(
                surface["memref"], _MEMREF_RE
            )

    def test_surface_is_request_scoped(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root, surface=True):
                update_recall_surface(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    flash_report=self._flash(),
                )

                model_view = (
                    recall_surface_for_model(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

                memref = model_view[
                    "surfaces"
                ][0]["memref"]

                # Another request in the same conversation.
                other_request = resolve_recall_ref(
                    conversation_id=CID,
                    cognitive_request_id=RID_B,
                    memref=memref,
                )

                # Another conversation.
                other_conversation = (
                    resolve_recall_ref(
                        conversation_id=CID_B,
                        cognitive_request_id=RID,
                        memref=memref,
                    )
                )

                guessed = resolve_recall_ref(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    memref=new_memref(),
                )

                malformed = resolve_recall_ref(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    memref="mem-1",
                )

        self.assertIsNone(other_request)
        self.assertIsNone(other_conversation)
        self.assertIsNone(guessed)
        self.assertIsNone(malformed)

    def test_surface_is_shadow_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(
                root, surface=False
            ):
                report = update_recall_surface(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    flash_report=self._flash(),
                )

                model_view = (
                    recall_surface_for_model(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

                status = recall_surface_status(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "surface_disabled",
        )
        self.assertEqual(
            model_view["surfaces"], []
        )
        self.assertFalse(status["exists"])

    def test_identity_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with recall_env(root, surface=True):
                wrong_request = (
                    update_recall_surface(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        flash_report=flash_artifact(
                            [memory("mem-1")],
                            cognitive_request_id=(
                                RID_B
                            ),
                        ),
                    )
                )

                malformed = build_recall_surface(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    flash_report={
                        "version": "nope"
                    },
                )

        self.assertFalse(wrong_request["stored"])
        self.assertEqual(
            wrong_request["reason"],
            "flash_identity_mismatch",
        )
        self.assertEqual(
            malformed["reason"],
            "malformed_flash_report",
        )

    def test_cue_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            flash = flash_artifact(
                [memory("mem-1")]
            )

            flash["flashes"][0]["cue"] = (
                "x" * 5000
            )

            with recall_env(root, surface=True):
                update_recall_surface(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    flash_report=flash,
                )

                model_view = (
                    recall_surface_for_model(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

        self.assertLessEqual(
            len(
                model_view["surfaces"][0][
                    "cue"
                ]
            ),
            320,
        )


if __name__ == "__main__":
    unittest.main()