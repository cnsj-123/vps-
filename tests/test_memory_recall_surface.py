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


class RecallSurfaceSourceGatingTests(
    unittest.TestCase,
):
    """Only a real surfaced Flash with a legal source revision may
    become a model-facing surface."""

    def _build(self, root, flash):
        with recall_env(root, surface=True):
            return build_recall_surface(
                conversation_id=CID,
                cognitive_request_id=RID,
                flash_report=flash,
            )

    def test_no_surface_with_stale_flashes_is_refused(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            flash = flash_artifact(
                [
                    memory("mem-1", "alpha"),
                    memory("mem-2", "beta"),
                ],
                decision="no_surface",
            )

            # A malformed artifact may still carry a stale flashes
            # list. It must never become a cue source.
            self.assertTrue(flash["flashes"])

            artifact = self._build(root, flash)

            surface_files = list(
                (
                    Path(root)
                    / "recall_surface"
                    / CID
                ).glob("*.json")
            )

        self.assertFalse(artifact["stored"])
        self.assertEqual(
            artifact["reason"],
            "flash_not_surfaced",
        )
        self.assertNotIn("mapping", artifact)
        self.assertNotIn("surfaces", artifact)
        self.assertEqual(surface_files, [])

    def test_no_surface_is_never_persisted(self):
        with tempfile.TemporaryDirectory() as root:
            flash = flash_artifact(
                [memory("mem-1")],
                decision="no_surface",
            )

            with recall_env(root, surface=True):
                report = update_recall_surface(
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

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "flash_not_surfaced",
        )
        self.assertEqual(
            model_view["surfaces"], []
        )

    def test_invalid_source_revision_is_refused(
        self,
    ):
        for value in (
            None,
            0,
            -1,
            True,
            "3",
            2.0,
        ):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as (
                    root
                ):
                    flash = flash_artifact(
                        [memory("mem-1")]
                    )

                    flash[
                        "source_unified_revision"
                    ] = value

                    artifact = self._build(
                        root, flash
                    )

                    surface_files = list(
                        (
                            Path(root)
                            / "recall_surface"
                            / CID
                        ).glob("*.json")
                    )

                self.assertFalse(
                    artifact["stored"]
                )
                self.assertEqual(
                    artifact["reason"],
                    (
                        "invalid_flash_source_"
                        "unified_revision"
                    ),
                )
                self.assertEqual(
                    surface_files, []
                )


class RecallSurfaceCorruptionTests(
    unittest.TestCase,
):
    """A tampered persisted surface must never resolve a memref or
    reach the model."""

    def _seed(self, root):
        with recall_env(root, surface=True):
            update_recall_surface(
                conversation_id=CID,
                cognitive_request_id=RID,
                flash_report=flash_artifact(
                    [
                        memory(
                            "mem-1",
                            "alpha",
                            name="merge rules",
                        ),
                        memory("mem-2", "beta"),
                    ]
                ),
            )

            model_view = recall_surface_for_model(
                conversation_id=CID,
                cognitive_request_id=RID,
            )

        memrefs = [
            surface["memref"]
            for surface in model_view["surfaces"]
        ]

        return memrefs

    def _tamper(self, root, **changes):
        path = (
            Path(root)
            / "recall_surface"
            / CID
            / (RID + ".json")
        )

        artifact = json.loads(
            path.read_text(encoding="utf-8")
        )

        artifact.update(changes)

        path.write_text(
            json.dumps(artifact),
            encoding="utf-8",
        )

    def _assert_unusable(self, root, memref):
        with recall_env(root, surface=True):
            model_view = recall_surface_for_model(
                conversation_id=CID,
                cognitive_request_id=RID,
            )

            resolved = resolve_recall_ref(
                conversation_id=CID,
                cognitive_request_id=RID,
                memref=memref,
            )

            status = recall_surface_status(
                conversation_id=CID,
                cognitive_request_id=RID,
            )

        self.assertEqual(
            model_view["surfaces"], []
        )
        self.assertIsNone(resolved)
        self.assertFalse(status["exists"])

    def test_tampered_conversation_id(self):
        with tempfile.TemporaryDirectory() as root:
            memrefs = self._seed(root)

            self._tamper(
                root, conversation_id=CID_B
            )

            self._assert_unusable(root, memrefs[0])

    def test_tampered_cognitive_request_id(self):
        with tempfile.TemporaryDirectory() as root:
            memrefs = self._seed(root)

            self._tamper(root, cognitive_request_id=RID_B)

            self._assert_unusable(root, memrefs[0])

    def test_tampered_mode(self):
        with tempfile.TemporaryDirectory() as root:
            memrefs = self._seed(root)

            self._tamper(root, mode="live")

            self._assert_unusable(root, memrefs[0])

    def test_tampered_source_revision(self):
        with tempfile.TemporaryDirectory() as root:
            memrefs = self._seed(root)

            self._tamper(
                root, source_flash_unified_revision=None
            )

            self._assert_unusable(root, memrefs[0])

    def test_tampered_mapping(self):
        with tempfile.TemporaryDirectory() as root:
            memrefs = self._seed(root)

            self._tamper(
                root, mapping={"memref_bad": "mem-1"}
            )

            self._assert_unusable(root, memrefs[0])

    def test_repeat_update_keeps_memrefs(self):
        with tempfile.TemporaryDirectory() as root:
            memrefs = self._seed(root)

            flash = flash_artifact(
                [
                    memory(
                        "mem-1",
                        "alpha",
                        name="merge rules",
                    ),
                    memory("mem-2", "beta"),
                ]
            )

            with recall_env(root, surface=True):
                second = update_recall_surface(
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

        self.assertTrue(second["duplicate"])
        self.assertEqual(
            second["decision"],
            "surface_reused",
        )
        self.assertEqual(
            [
                surface["memref"]
                for surface in model_view["surfaces"]
            ],
            memrefs,
        )

    def test_new_flash_revision_rebuilds_memrefs(self):
        with tempfile.TemporaryDirectory() as root:
            memrefs = self._seed(root)

            flash = flash_artifact(
                [
                    memory(
                        "mem-1",
                        "alpha",
                        name="merge rules",
                    ),
                    memory("mem-2", "beta"),
                ],
                revision=6,
            )

            with recall_env(root, surface=True):
                second = update_recall_surface(
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

        self.assertFalse(second["duplicate"])
        self.assertNotEqual(
            [
                surface["memref"]
                for surface in model_view["surfaces"]
            ],
            memrefs,
        )


if __name__ == "__main__":
    unittest.main()