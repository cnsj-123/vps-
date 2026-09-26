from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _recall_fixtures import (
    CID,
    CID_B,
    DEFAULT_RECALL_ID,
    RID,
    RID_B,
    USAGE_REFS_MAX_ENV,
    USAGE_REFS_TOKEN_BUDGET_ENV,
    memory,
    projected_model_result,
    seed_live_exposure,
    seed_recall_control_plane,
)

from ombrebrain.context.live_recall_usage_surface import (
    SAFE_BUILD_REASONS,
    append_live_recall_usage_refs,
    build_live_recall_usage_surface,
    compose_live_recall_result_with_usage_refs,
    is_valid_live_recall_usage_surface_artifact,
    is_valid_useref,
    live_recall_usage_surface_path,
    live_recall_usage_surface_status,
    new_useref,
    read_live_recall_usage_surface,
    render_live_recall_usage_refs,
    resolve_usage_refs_max_refs,
    resolve_usage_refs_token_budget,
)


_LOADED_IDS = ["mem-1", "mem-2", "mem-3", "mem-4"]

_EXPOSED_IDS = ["mem-1", "mem-2"]

_UNSET = object()


def _exposed_memories():
    return [memory(memory_id) for memory_id in _EXPOSED_IDS]


class UsageSurfaceCase(unittest.TestCase):
    """Shared seeding: Flash + Surface + Live Exposure + Recall plane."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()

        self.addCleanup(tmp.cleanup)

        self.root = tmp.name

        env = patch.dict(
            os.environ,
            {"OMBRE_CONTEXT_STATE_DIR": self.root},
            clear=False,
        )

        env.start()

        self.addCleanup(env.stop)

        self.memrefs = seed_live_exposure(
            self.root, _exposed_memories()
        )

        self.plane = seed_recall_control_plane(
            self.root,
            list(_LOADED_IDS),
            loaded_memory_ids=list(_LOADED_IDS),
        )

        self.anchor_ref = self.memrefs[0]

    def _build(self, model_result=_UNSET, **kwargs):
        return build_live_recall_usage_surface(
            conversation_id=kwargs.pop(
                "conversation_id", CID
            ),
            cognitive_request_id=kwargs.pop(
                "cognitive_request_id", RID
            ),
            anchor_ref=kwargs.pop(
                "anchor_ref", self.anchor_ref
            ),
            model_result=(
                projected_model_result(
                    ["mem-1", "mem-2", "mem-3"]
                )
                if model_result is _UNSET
                else model_result
            ),
            **kwargs,
        )

    def _surface(self, **kwargs):
        return read_live_recall_usage_surface(
            conversation_id=kwargs.pop(
                "conversation_id", CID
            ),
            cognitive_request_id=kwargs.pop(
                "cognitive_request_id", RID
            ),
        )

    def _surface_text(self, **kwargs):
        return live_recall_usage_surface_path(
            kwargs.pop("conversation_id", CID),
            kwargs.pop(
                "cognitive_request_id", RID
            ),
        ).read_text(encoding="utf-8")


class BuildUsageSurfaceTests(UsageSurfaceCase):

    def test_valid_projection_builds_opaque_refs(self):
        report = self._build()

        self.assertTrue(report["stored"])
        self.assertEqual(
            report["decision"], "surface_built"
        )
        self.assertEqual(
            report["reason"], "usage_surface_built"
        )
        self.assertEqual(report["mapped_count"], 3)

        refs = report["usage_refs"]

        self.assertEqual(
            [ref["rank"] for ref in refs], [1, 2, 3]
        )

        for ref in refs:
            self.assertTrue(is_valid_useref(ref["useref"]))

        artifact = self._surface()

        self.assertIsInstance(artifact, dict)

        self.assertEqual(
            artifact["version"],
            "memory-live-recall-usage-surface.v1",
        )
        self.assertEqual(
            artifact["mode"], "explicit_model_attribution"
        )
        self.assertEqual(artifact["conversation_id"], CID)
        self.assertEqual(
            artifact["cognitive_request_id"], RID
        )
        self.assertEqual(len(artifact["mappings"]), 3)

        for mapping in artifact["mappings"]:
            self.assertEqual(
                set(mapping.keys()),
                {
                    "useref",
                    "recall_id",
                    "memory_id",
                    "rank",
                    "kind",
                    "source_request_fingerprint",
                    "source_model_result_sha256",
                },
            )
            self.assertEqual(
                mapping["recall_id"], DEFAULT_RECALL_ID
            )
            self.assertEqual(
                mapping["source_request_fingerprint"],
                self.plane["request_fingerprint"],
            )

        self.assertEqual(
            artifact["mappings"][0]["memory_id"], "mem-1"
        )
        self.assertEqual(
            artifact["mappings"][0]["kind"], "anchor"
        )
        self.assertEqual(
            artifact["mappings"][2]["memory_id"], "mem-3"
        )
        self.assertEqual(
            artifact["mappings"][2]["kind"], "related"
        )

    def test_surface_never_stores_content_or_identity_text(self):
        self._build()

        text = self._surface_text()

        for forbidden in (
            "content",
            "cue",
            "recalled body",
            "query",
            "assistant",
            "user_message",
            "memref_",
        ):
            self.assertNotIn(forbidden, text)

        # Internal provenance IS allowed: recall id + memory id.
        self.assertIn(DEFAULT_RECALL_ID, text)
        self.assertIn("mem-1", text)

    def test_model_never_sees_internal_fields(self):
        report = self._build()

        text = json.dumps(report)

        for forbidden in (
            "recall_",
            "mem-1",
            CID,
            RID,
            "memref_",
            "memory_id",
            DEFAULT_RECALL_ID,
        ):
            self.assertNotIn(forbidden, text)

        self.assertIn("useref_", text)

    def test_loaded_but_not_projected_memory_gets_no_ref(self):
        # Related Recall loaded 4 items; the bounded projection only
        # output 2. Ranks 3-4 were never seen by the model, so they can
        # never obtain a useref.
        report = self._build(
            projected_model_result(["mem-1", "mem-2"])
        )

        self.assertTrue(report["stored"])
        self.assertEqual(report["mapped_count"], 2)

        # Refs can never outnumber the model projection.
        self.assertLessEqual(
            report["mapped_count"],
            projected_model_result(
                ["mem-1", "mem-2"]
            )["memory_count"],
        )

        artifact = self._surface()

        self.assertEqual(len(artifact["mappings"]), 2)

        self.assertEqual(
            sorted(
                mapping["rank"]
                for mapping in artifact["mappings"]
            ),
            [1, 2],
        )

        for mapping in artifact["mappings"]:
            self.assertNotEqual(
                mapping["memory_id"], "mem-3"
            )
            self.assertNotEqual(
                mapping["memory_id"], "mem-4"
            )

    def test_duplicate_recall_reuses_the_same_userefs(self):
        first = self._build()

        second = self._build()

        self.assertTrue(second["stored"])
        self.assertEqual(
            second["decision"], "surface_reused"
        )
        self.assertEqual(
            second["reason"],
            "duplicate_usage_surface",
        )
        self.assertEqual(second["reused_count"], 3)

        self.assertEqual(
            [ref["useref"] for ref in first["usage_refs"]],
            [ref["useref"] for ref in second["usage_refs"]],
        )

        artifact = self._surface()

        self.assertEqual(len(artifact["mappings"]), 3)
        self.assertEqual(
            len(
                {
                    mapping["useref"]
                    for mapping in artifact["mappings"]
                }
            ),
            3,
        )

    def test_same_request_two_recalls_get_separate_refs(self):
        first = self._build()

        # A second Recall in the SAME request: another anchor memref,
        # hence another recall id -- even for the same raw memory the
        # refs must be distinct.
        second_plane = seed_recall_control_plane(
            self.root,
            ["mem-2", "mem-3"],
            recall_id="recall_" + "2" * 32,
        )

        second = build_live_recall_usage_surface(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_ref=self.memrefs[1],
            model_result=projected_model_result(
                ["mem-2", "mem-3"]
            ),
        )

        self.assertTrue(second["stored"])

        first_refs = {
            ref["useref"] for ref in first["usage_refs"]
        }

        second_refs = {
            ref["useref"] for ref in second["usage_refs"]
        }

        self.assertFalse(first_refs & second_refs)

        artifact = self._surface()

        recall_ids = {
            mapping["recall_id"]
            for mapping in artifact["mappings"]
        }

        self.assertEqual(
            recall_ids,
            {
                DEFAULT_RECALL_ID,
                second_plane["recall_id"],
            },
        )

        # The same raw memory (mem-2) appears under BOTH recalls with
        # different userefs: used must belong to the real artifact.
        mem2_refs = [
            mapping["useref"]
            for mapping in artifact["mappings"]
            if mapping["memory_id"] == "mem-2"
        ]

        self.assertEqual(len(mem2_refs), 2)
        self.assertNotEqual(mem2_refs[0], mem2_refs[1])

    def test_each_mapping_is_bound_to_one_model_result_digest(self):
        self._build(
            projected_model_result(["mem-1", "mem-2"])
        )

        artifact = self._surface()

        digests = {
            mapping["source_model_result_sha256"]
            for mapping in artifact["mappings"]
        }

        self.assertEqual(len(digests), 1)

        digest = next(iter(digests))

        self.assertEqual(len(digest), 64)


class RefusalTests(UsageSurfaceCase):

    def _assert_refused(self, report, reason):
        self.assertFalse(report["stored"])
        self.assertEqual(report["decision"], "refused")
        self.assertEqual(report["usage_refs"], [])
        self.assertEqual(report["mapped_count"], 0)
        self.assertEqual(report["reason"], reason)
        self.assertIn(report["reason"], SAFE_BUILD_REASONS)

    def test_invalid_request_identity_is_refused(self):
        for kwargs in (
            {"conversation_id": "nope"},
            {"cognitive_request_id": "nope"},
            {"conversation_id": None},
            {"cognitive_request_id": None},
        ):
            report = self._build(**kwargs)

            self._assert_refused(
                report, "invalid_request_identity"
            )

        self.assertFalse(
            live_recall_usage_surface_path(CID, RID).exists()
        )

    def test_invalid_model_result_is_refused(self):
        for model_result in (
            None,
            {},
            {"status": "recalled"},
            projected_model_result([]),
            {
                "version": "memory-live-recall-result.v1",
                "mode": "trusted_live_bridge",
                "status": "refused",
                "reason": "recall_unavailable",
                "memory_count": 0,
                "memories": [],
            },
        ):
            report = self._build(model_result)

            self._assert_refused(
                report, "invalid_model_result"
            )

        self.assertFalse(
            live_recall_usage_surface_path(CID, RID).exists()
        )

    def test_unknown_anchor_ref_is_refused(self):
        report = self._build(
            anchor_ref="memref_" + "0" * 32
        )

        self._assert_refused(
            report, "recall_surface_binding_invalid"
        )

    def test_cross_conversation_memref_is_refused(self):
        # A real memref belonging to another conversation.
        other_root = tempfile.TemporaryDirectory()

        self.addCleanup(other_root.cleanup)

        other_memrefs = seed_live_exposure(
            other_root.name,
            _exposed_memories(),
            conversation_id=CID_B,
            cognitive_request_id=RID_B,
        )

        with patch.dict(
            os.environ,
            {"OMBRE_CONTEXT_STATE_DIR": self.root},
            clear=False,
        ):
            report = self._build(
                anchor_ref=other_memrefs[0]
            )

        self._assert_refused(
            report, "recall_surface_binding_invalid"
        )

    def test_memref_without_persisted_recall_is_refused(self):
        # mem-2 is live-exposed and has a surface binding, but no
        # Recall Request was ever persisted for it as an anchor.
        report = self._build(
            anchor_ref=self.memrefs[1]
        )

        self._assert_refused(
            report, "recall_request_not_found"
        )

    def test_kind_never_maps_by_rank_alone(self):
        # The Related Recall artifact has "related_memory" at rank 2;
        # a model result claiming "anchor" there must be refused.
        model_result = projected_model_result(
            ["mem-1", "mem-2"],
            kinds=["anchor", "anchor"],
        )

        report = self._build(model_result)

        self._assert_refused(report, "kind_mismatch")

    def test_memory_outside_loaded_set_is_refused(self):
        # A Usage Signal whose loaded set lacks mem-2.
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"OMBRE_CONTEXT_STATE_DIR": root},
                clear=False,
            ):
                memrefs = seed_live_exposure(
                    root, _exposed_memories()
                )

                seed_recall_control_plane(
                    root,
                    list(_LOADED_IDS),
                    loaded_memory_ids=["mem-1"],
                )

                report = build_live_recall_usage_surface(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    anchor_ref=memrefs[0],
                    model_result=projected_model_result(
                        ["mem-1", "mem-2"]
                    ),
                )

        self._assert_refused(
            report, "memory_not_loaded"
        )

    def test_missing_usage_signal_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"OMBRE_CONTEXT_STATE_DIR": root},
                clear=False,
            ):
                memrefs = seed_live_exposure(
                    root, _exposed_memories()
                )

                plane = seed_recall_control_plane(
                    root, list(_LOADED_IDS)
                )

                Path(
                    root,
                    "memory_usage",
                    CID,
                    RID,
                    plane["recall_id"] + ".json",
                ).unlink()

                report = build_live_recall_usage_surface(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    anchor_ref=memrefs[0],
                    model_result=projected_model_result(
                        ["mem-1"]
                    ),
                )

        self._assert_refused(
            report, "usage_signal_not_found"
        )

    def test_corrupt_surface_is_never_overwritten(self):
        self._build()

        path = live_recall_usage_surface_path(CID, RID)

        artifact = json.loads(
            path.read_text(encoding="utf-8")
        )

        # Tamper: duplicate rank for one recall.
        artifact["mappings"].append(
            dict(artifact["mappings"][0])
        )

        artifact["mappings"][-1]["useref"] = new_useref()

        path.write_text(
            json.dumps(artifact), encoding="utf-8"
        )

        before = path.read_text(encoding="utf-8")

        report = self._build()

        self._assert_refused(
            report, "usage_surface_invalid"
        )

        self.assertEqual(
            path.read_text(encoding="utf-8"), before
        )

    def test_persistence_failure_returns_no_refs(self):
        with patch(
            "ombrebrain.context.live_recall_usage_surface.atomic_write",
            side_effect=OSError("boom"),
        ):
            report = self._build()

        self._assert_refused(
            report,
            "usage_surface_persistence_failed",
        )

        self.assertFalse(
            live_recall_usage_surface_path(CID, RID).exists()
        )

    def test_corrupt_related_recall_is_refused(self):
        path = (
            Path(self.root)
            / "related_recall"
            / CID
            / RID
            / (DEFAULT_RECALL_ID + ".json")
        )

        artifact = json.loads(
            path.read_text(encoding="utf-8")
        )

        artifact["anchor_memory_id"] = "mem-9"

        path.write_text(
            json.dumps(artifact), encoding="utf-8"
        )

        report = self._build()

        self._assert_refused(
            report, "related_recall_not_found"
        )


class SurfaceValidatorTests(unittest.TestCase):

    def _artifact(self):
        return {
            "version": (
                "memory-live-recall-usage-surface.v1"
            ),
            "mode": "explicit_model_attribution",
            "conversation_id": CID,
            "cognitive_request_id": RID,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "mappings": [
                {
                    "useref": new_useref(),
                    "recall_id": DEFAULT_RECALL_ID,
                    "memory_id": "mem-1",
                    "rank": 1,
                    "kind": "anchor",
                    "source_request_fingerprint":
                        "a" * 64,
                    "source_model_result_sha256":
                        "b" * 64,
                }
            ],
        }

    def _assert_invalid(self, artifact):
        self.assertFalse(
            is_valid_live_recall_usage_surface_artifact(
                artifact,
                conversation_id=CID,
                cognitive_request_id=RID,
            )
        )

    def test_valid_artifact(self):
        self.assertTrue(
            is_valid_live_recall_usage_surface_artifact(
                self._artifact(),
                conversation_id=CID,
                cognitive_request_id=RID,
            )
        )

    def test_identity_and_contract_are_enforced(self):
        for mutate in (
            lambda a: a.update({"version": "x"}),
            lambda a: a.update({"mode": "x"}),
            lambda a: a.update(
                {"conversation_id": CID_B}
            ),
            lambda a: a.update(
                {"cognitive_request_id": RID_B}
            ),
            lambda a: a.update({"created_at": ""}),
            lambda a: a.update({"updated_at": None}),
            lambda a: a.update({"mappings": []}),
            lambda a: a.update({"mappings": {}}),
        ):
            artifact = self._artifact()

            mutate(artifact)

            self._assert_invalid(artifact)

    def test_mapping_fields_are_enforced(self):
        cases = (
            lambda m: m.update({"useref": "useref_x"}),
            lambda m: m.update({"useref": new_useref()[:20]}),
            lambda m: m.update({"useref": 42}),
            lambda m: m.update({"recall_id": "recall_x"}),
            lambda m: m.update({"memory_id": "  "}),
            lambda m: m.update({"memory_id": None}),
            lambda m: m.update({"rank": 0}),
            lambda m: m.update({"rank": True}),
            lambda m: m.update({"rank": "1"}),
            lambda m: m.update({"kind": "other"}),
            lambda m: m.update(
                {"source_request_fingerprint": "short"}
            ),
            lambda m: m.update(
                {"source_model_result_sha256": "short"}
            ),
            lambda m: m.pop("kind"),
            lambda m: m.update({"extra": 1}),
        )

        for mutate in cases:
            artifact = self._artifact()

            mutate(artifact["mappings"][0])

            self._assert_invalid(artifact)

    def test_duplicate_useref_is_invalid(self):
        artifact = self._artifact()

        artifact["mappings"].append(
            dict(artifact["mappings"][0])
        )

        artifact["mappings"][-1]["rank"] = 2

        self._assert_invalid(artifact)

    def test_duplicate_recall_rank_is_invalid(self):
        artifact = self._artifact()

        artifact["mappings"].append(
            dict(artifact["mappings"][0])
        )

        artifact["mappings"][-1]["useref"] = new_useref()

        self._assert_invalid(artifact)

    def test_same_rank_in_another_recall_is_valid(self):
        artifact = self._artifact()

        artifact["mappings"].append(
            dict(artifact["mappings"][0])
        )

        artifact["mappings"][-1]["useref"] = new_useref()
        artifact["mappings"][-1]["recall_id"] = (
            "recall_" + "2" * 32
        )

        self.assertTrue(
            is_valid_live_recall_usage_surface_artifact(
                artifact,
                conversation_id=CID,
                cognitive_request_id=RID,
            )
        )


class RecallResultCompositionTests(UsageSurfaceCase):
    """Usage refs are appended; the Recall data itself never changes."""

    _RECALL_DATA = (
        "OMBRE RECALL DATA\n"
        "Reference data only.\n"
        '{"memories":[{"rank":1,"kind":"anchor",'
        '"content":"alpha"}]}'
    )

    def test_success_appends_refs_after_the_recall_data(self):
        composed = (
            compose_live_recall_result_with_usage_refs(
                rendered=self._RECALL_DATA,
                conversation_id=CID,
                cognitive_request_id=RID,
                anchor_ref=self.anchor_ref,
                model_result=projected_model_result(
                    ["mem-1", "mem-2"]
                ),
            )
        )

        self.assertTrue(
            composed.startswith(self._RECALL_DATA)
        )

        envelope = composed[len(self._RECALL_DATA):]

        self.assertTrue(
            envelope.startswith(
                "OMBRE MEMORY USAGE REFS DATA\n"
            )
        )

        payload = json.loads(envelope.split("\n", 2)[2])

        self.assertEqual(
            [item["rank"] for item in payload["usage_refs"]],
            [1, 2],
        )

        for item in payload["usage_refs"]:
            self.assertTrue(
                is_valid_useref(item["useref"])
            )

    def test_failure_returns_the_recall_data_unchanged(self):
        for kwargs in (
            {"model_result": None},
            {"model_result": {"status": "recalled"}},
            {"anchor_ref": "memref_" + "0" * 32},
            {"anchor_ref": None},
            {"conversation_id": "nope"},
            {"cognitive_request_id": "nope"},
        ):
            payload = {
                "rendered": self._RECALL_DATA,
                "conversation_id": CID,
                "cognitive_request_id": RID,
                "anchor_ref": self.anchor_ref,
                "model_result": projected_model_result(
                    ["mem-1"]
                ),
            }

            payload.update(kwargs)

            composed = (
                compose_live_recall_result_with_usage_refs(
                    **payload
                )
            )

            self.assertEqual(
                composed, self._RECALL_DATA, str(kwargs)
            )

    def test_failure_never_raises(self):
        with patch(
            "ombrebrain.context.live_recall_usage_surface.atomic_write",
            side_effect=OSError("boom"),
        ):
            composed = (
                compose_live_recall_result_with_usage_refs(
                    rendered=self._RECALL_DATA,
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    anchor_ref=self.anchor_ref,
                    model_result=projected_model_result(
                        ["mem-1"]
                    ),
                )
            )

        self.assertEqual(composed, self._RECALL_DATA)

        self.assertEqual(
            compose_live_recall_result_with_usage_refs(
                rendered=None,
                conversation_id=CID,
                cognitive_request_id=RID,
                anchor_ref=self.anchor_ref,
                model_result=None,
            ),
            "",
        )


class UsageRefsRendererTests(unittest.TestCase):

    def test_append_never_modifies_the_recall_data(self):
        recall_data = "OMBRE RECALL DATA\n{}"

        refs = [
            {"rank": 1, "useref": new_useref()},
        ]

        combined = append_live_recall_usage_refs(
            recall_data, refs
        )

        self.assertTrue(combined.startswith(recall_data))
        self.assertTrue(
            combined[len(recall_data):].startswith(
                "OMBRE MEMORY USAGE REFS DATA\n"
            )
        )

    def test_append_without_refs_returns_the_recall_data(self):
        recall_data = "OMBRE RECALL DATA\n{}"

        for refs in (None, [], "junk", [{"rank": 0}]):
            self.assertEqual(
                append_live_recall_usage_refs(
                    recall_data, refs
                ),
                recall_data,
                str(refs),
            )

        # A non-string Recall result degrades to just the envelope
        # (never a crash, never a fabricated Recall result).
        envelope_only = append_live_recall_usage_refs(
            None, [{"rank": 1, "useref": new_useref()}]
        )

        self.assertTrue(
            envelope_only.startswith(
                "OMBRE MEMORY USAGE REFS DATA\n"
            )
        )

    def test_renders_data_only_envelope(self):
        refs = [
            {"rank": rank, "useref": new_useref()}
            for rank in range(1, 4)
        ]

        rendered = render_live_recall_usage_refs(refs)

        self.assertTrue(
            rendered.startswith(
                "OMBRE MEMORY USAGE REFS DATA\n"
            )
        )

        payload = json.loads(
            rendered.split("\n", 2)[2]
        )

        self.assertEqual(
            [item["rank"] for item in payload["usage_refs"]],
            [1, 2, 3],
        )

        for forbidden in (
            "recall_id",
            "memory_id",
            "conversation_id",
            "cognitive_request_id",
            "memref_",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_malformed_refs_are_never_rendered(self):
        for refs in (
            None,
            "useref_x",
            [{"rank": 1}],
            [{"rank": 0, "useref": new_useref()}],
            [{"rank": "1", "useref": new_useref()}],
            [{"rank": 1, "useref": "useref_short"}],
            [
                {
                    "rank": 1,
                    "useref": "useref_" + "a" * 32,
                },
                {
                    "rank": 2,
                    "useref": "useref_" + "a" * 32,
                },
            ],
        ):
            self.assertEqual(
                render_live_recall_usage_refs(refs), ""
            )

    def test_tail_is_dropped_but_refs_are_never_truncated(self):
        with patch.dict(
            os.environ,
            {USAGE_REFS_TOKEN_BUDGET_ENV: "70"},
            clear=False,
        ):
            rendered = render_live_recall_usage_refs(
                [
                    {"rank": rank, "useref": new_useref()}
                    for rank in range(1, 6)
                ]
            )

        self.assertNotEqual(rendered, "")

        payload = json.loads(
            rendered.split("\n", 2)[2]
        )

        refs = payload["usage_refs"]

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["rank"], 1)
        self.assertTrue(is_valid_useref(refs[0]["useref"]))
        self.assertEqual(len(refs[0]["useref"]), 39)

    def test_budget_too_small_for_even_rank_one(self):
        with patch.dict(
            os.environ,
            {USAGE_REFS_TOKEN_BUDGET_ENV: "10"},
            clear=False,
        ):
            rendered = render_live_recall_usage_refs(
                [{"rank": 1, "useref": new_useref()}]
            )

        self.assertEqual(rendered, "")

    def test_refs_are_capped(self):
        with patch.dict(
            os.environ,
            {USAGE_REFS_MAX_ENV: "3"},
            clear=False,
        ):
            rendered = render_live_recall_usage_refs(
                [
                    {"rank": rank, "useref": new_useref()}
                    for rank in range(1, 6)
                ]
            )

        payload = json.loads(
            rendered.split("\n", 2)[2]
        )

        self.assertEqual(
            [item["rank"] for item in payload["usage_refs"]],
            [1, 2, 3],
        )

    def test_default_renders_max_five_refs(self):
        rendered = render_live_recall_usage_refs(
            [
                {"rank": rank, "useref": new_useref()}
                for rank in range(1, 9)
            ]
        )

        payload = json.loads(
            rendered.split("\n", 2)[2]
        )

        self.assertEqual(len(payload["usage_refs"]), 5)

        self.assertLessEqual(
            resolve_usage_refs_max_refs(), 8
        )
        self.assertLessEqual(
            resolve_usage_refs_token_budget(), 192
        )


class UsageRefsLimitsTests(unittest.TestCase):

    def test_defaults_and_hard_caps(self):
        for value in (None, "", "0", "-1", "abc", "3.5"):
            with patch.dict(
                os.environ,
                (
                    {USAGE_REFS_MAX_ENV: value}
                    if value is not None
                    else {}
                ),
                clear=False,
            ):
                self.assertGreaterEqual(
                    resolve_usage_refs_max_refs(), 1
                )
                self.assertLessEqual(
                    resolve_usage_refs_max_refs(), 8
                )

        with patch.dict(
            os.environ,
            {USAGE_REFS_MAX_ENV: "99"},
            clear=False,
        ):
            self.assertEqual(
                resolve_usage_refs_max_refs(), 8
            )

        with patch.dict(
            os.environ,
            {USAGE_REFS_TOKEN_BUDGET_ENV: "9999"},
            clear=False,
        ):
            self.assertEqual(
                resolve_usage_refs_token_budget(), 192
            )


class UsageSurfaceStatusTests(UsageSurfaceCase):

    def test_status_is_privacy_safe(self):
        self.assertEqual(
            live_recall_usage_surface_status(
                conversation_id=CID,
                cognitive_request_id=RID,
            ),
            {"exists": False},
        )

        self._build()

        status = live_recall_usage_surface_status(
            conversation_id=CID,
            cognitive_request_id=RID,
        )

        self.assertTrue(status["exists"])
        self.assertEqual(status["mapping_count"], 3)
        self.assertEqual(status["recall_count"], 1)

        self.assertNotIn(
            "useref", json.dumps(status)
        )

    def test_cross_conversation_read_is_refused(self):
        self._build()

        self.assertIsNone(
            read_live_recall_usage_surface(
                conversation_id=CID_B,
                cognitive_request_id=RID,
            )
        )

        self.assertIsNone(
            read_live_recall_usage_surface(
                conversation_id=CID,
                cognitive_request_id=RID_B,
            )
        )


if __name__ == "__main__":
    unittest.main()