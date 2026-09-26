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
    LIFECYCLE_SHADOW_ENV,
    RID,
    RID_B,
    USAGE_ATTRIBUTION_ENV,
    USAGE_REFS_MAX_ENV,
    memory,
    projected_model_result,
    read_usage,
    seed_live_exposure,
    seed_recall_control_plane,
)

from ombrebrain.context import memory_usage_signal
from ombrebrain.context.live_recall_usage_attribution import (
    SAFE_ATTRIBUTION_REASONS,
    attribute_live_recall_usage,
    live_recall_usage_attribution_enabled,
    render_live_recall_usage_ack,
)
from ombrebrain.context.live_recall_usage_surface import (
    build_live_recall_usage_surface,
    live_recall_usage_surface_path,
)


_LOADED_IDS = ["mem-1", "mem-2", "mem-3"]

_EXPOSED_IDS = ["mem-1", "mem-2"]


def _exposed_memories(memory_ids=None):
    return [
        memory(memory_id)
        for memory_id in (
            memory_ids
            if memory_ids is not None
            else _EXPOSED_IDS
        )
    ]


class AttributionCase(unittest.TestCase):
    """Seed a full chain and build the Usage Surface for THIS request."""

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
            self.root, list(_LOADED_IDS)
        )

        self.surface_report = (
            build_live_recall_usage_surface(
                conversation_id=CID,
                cognitive_request_id=RID,
                anchor_ref=self.memrefs[0],
                model_result=projected_model_result(
                    list(_LOADED_IDS)
                ),
            )
        )

        self.assertTrue(self.surface_report["stored"])

        self.refs = {
            ref["rank"]: ref["useref"]
            for ref in self.surface_report["usage_refs"]
        }

    def _attribute(self, userefs, **kwargs):
        return attribute_live_recall_usage(
            conversation_id=kwargs.pop(
                "conversation_id", CID
            ),
            cognitive_request_id=kwargs.pop(
                "cognitive_request_id", RID
            ),
            userefs=userefs,
        )

    def _usage(self, **kwargs):
        return read_usage(
            self.root,
            conversation_id=kwargs.pop(
                "conversation_id", CID
            ),
            cognitive_request_id=kwargs.pop(
                "cognitive_request_id", RID
            ),
            recall_id=kwargs.pop(
                "recall_id", DEFAULT_RECALL_ID
            ),
        )


class ValidAttributionTests(AttributionCase):

    def test_explicit_usage_records_only_the_named_items(self):
        report = self._attribute(
            [self.refs[1], self.refs[3]]
        )

        self.assertTrue(report["recorded"])
        self.assertEqual(
            report["reason"], "usage_recorded"
        )
        self.assertEqual(report["requested_ref_count"], 2)
        self.assertEqual(report["accepted_ref_count"], 2)
        self.assertEqual(report["newly_used_count"], 2)
        self.assertEqual(report["duplicate_count"], 0)

        # The internal report never leaks an id.
        text = json.dumps(report)

        for forbidden in (
            "useref_",
            "mem-1",
            "mem-3",
            DEFAULT_RECALL_ID,
            CID,
            RID,
            "memory_id",
            "recall_id",
        ):
            self.assertNotIn(forbidden, text)

        usage = self._usage()

        self.assertEqual(usage["loaded_count"], 3)
        self.assertEqual(usage["used_count"], 2)
        self.assertEqual(
            usage["used_memory_ids"], ["mem-1", "mem-3"]
        )

        used_events = [
            event["memory_id"]
            for event in usage["events"]
            if event["stage"] == "used"
        ]

        self.assertEqual(used_events, ["mem-1", "mem-3"])

    def test_ignored_memory_stays_loaded_only(self):
        self._attribute([self.refs[1]])

        usage = self._usage()

        self.assertEqual(usage["used_memory_ids"], ["mem-1"])
        self.assertEqual(usage["loaded_count"], 3)
        self.assertIn("mem-2", usage["loaded_memory_ids"])
        self.assertNotIn("mem-2", usage["used_memory_ids"])

        for event in usage["events"]:
            if event["memory_id"] == "mem-2":
                self.assertEqual(
                    event["stage"], "memory_loaded"
                )

    def test_used_is_attributed_in_surface_rank_order(self):
        report = self._attribute(
            [self.refs[3], self.refs[1]]
        )

        self.assertTrue(report["recorded"])

        usage = self._usage()

        self.assertEqual(
            usage["used_memory_ids"], ["mem-1", "mem-3"]
        )

    def test_duplicate_refs_inside_one_call_are_folded(self):
        report = self._attribute(
            [self.refs[1], self.refs[1], self.refs[2]]
        )

        self.assertTrue(report["recorded"])
        self.assertEqual(report["requested_ref_count"], 3)
        self.assertEqual(report["accepted_ref_count"], 2)
        self.assertEqual(report["newly_used_count"], 2)

        usage = self._usage()

        self.assertEqual(usage["used_count"], 2)

    def test_repeated_attribution_is_idempotent(self):
        first = self._attribute(
            [self.refs[1], self.refs[3]]
        )

        self.assertTrue(first["recorded"])

        usage_after_first = self._usage()

        used_events_after_first = [
            event
            for event in usage_after_first["events"]
            if event["stage"] == "used"
        ]

        second = self._attribute(
            [self.refs[1], self.refs[3]]
        )

        self.assertTrue(second["recorded"])
        self.assertEqual(
            second["reason"], "duplicate_usage_refs"
        )
        self.assertEqual(second["newly_used_count"], 0)
        self.assertEqual(second["duplicate_count"], 2)
        self.assertEqual(second["accepted_ref_count"], 2)

        usage_after_second = self._usage()

        self.assertEqual(
            usage_after_second["used_count"], 2
        )
        self.assertEqual(
            usage_after_second["used_memory_ids"],
            usage_after_first["used_memory_ids"],
        )
        self.assertEqual(
            len(
                [
                    event
                    for event in usage_after_second[
                        "events"
                    ]
                    if event["stage"] == "used"
                ]
            ),
            len(used_events_after_first),
        )

    def test_attribution_calls_the_frozen_record_usage(self):
        calls = []

        original = memory_usage_signal.record_usage

        def spy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        with patch.object(
            memory_usage_signal, "record_usage", spy
        ):
            report = self._attribute([self.refs[2]])

        self.assertTrue(report["recorded"])
        self.assertEqual(len(calls), 1)

        call = calls[0]

        self.assertEqual(
            set(call.keys()),
            {
                "conversation_id",
                "cognitive_request_id",
                "recall_id",
                "used_memory_ids",
            },
        )
        self.assertEqual(call["conversation_id"], CID)
        self.assertEqual(
            call["cognitive_request_id"], RID
        )
        self.assertEqual(
            call["recall_id"], DEFAULT_RECALL_ID
        )
        self.assertEqual(
            call["used_memory_ids"], ["mem-2"]
        )

    def test_second_recall_of_one_request_attributed_separately(self):
        second_plane = seed_recall_control_plane(
            self.root,
            ["mem-2", "mem-3"],
            recall_id="recall_" + "2" * 32,
        )

        second_surface = build_live_recall_usage_surface(
            conversation_id=CID,
            cognitive_request_id=RID,
            anchor_ref=self.memrefs[1],
            model_result=projected_model_result(
                ["mem-2", "mem-3"]
            ),
        )

        self.assertTrue(second_surface["stored"])

        first_ref = self.refs[1]

        second_ref = second_surface["usage_refs"][0][
            "useref"
        ]

        report = self._attribute(
            [first_ref, second_ref]
        )

        self.assertFalse(report["recorded"])
        self.assertEqual(
            report["reason"], "mixed_recall_refs"
        )

        self.assertEqual(
            self._usage()["used_count"], 0
        )

        self.assertEqual(
            self._usage(
                recall_id=second_plane["recall_id"]
            )["used_count"],
            0,
        )

        # Each call on its own works.
        first_report = self._attribute([first_ref])

        self.assertTrue(first_report["recorded"])

        second_report = self._attribute([second_ref])

        self.assertTrue(second_report["recorded"])

        self.assertEqual(
            self._usage()["used_memory_ids"], ["mem-1"]
        )

        self.assertEqual(
            self._usage(
                recall_id=second_plane["recall_id"]
            )["used_memory_ids"],
            ["mem-2"],
        )


class RefusalTests(AttributionCase):

    def _assert_refused(self, report, reason):
        self.assertFalse(report["recorded"])
        self.assertEqual(report["reason"], reason)
        self.assertIn(
            report["reason"], SAFE_ATTRIBUTION_REASONS
        )

    def test_fake_ref_is_refused(self):
        report = self._attribute(
            ["useref_" + "0" * 32]
        )

        self._assert_refused(
            report, "invalid_usage_refs"
        )
        self.assertEqual(report["requested_ref_count"], 1)
        self.assertEqual(report["accepted_ref_count"], 0)

        self.assertEqual(
            self._usage()["used_count"], 0
        )

    def test_malformed_refs_are_refused(self):
        for userefs in (
            [],
            (),
            "useref_abc",
            [123],
            [None],
            [""],
            ["useref_x"],
            ["useref_" + "A" * 32],
            {"useref": "x"},
        ):
            report = self._attribute(userefs)

            self.assertFalse(report["recorded"])
            self.assertIn(
                report["reason"],
                {
                    "invalid_usage_refs",
                    "too_many_refs",
                },
            )

        self.assertEqual(
            self._usage()["used_count"], 0
        )

    def test_cross_request_ref_is_refused(self):
        # Request B lives in the SAME conversation and the SAME state
        # root; only the exact request-scoped surface separates them.
        memrefs = seed_live_exposure(
            self.root,
            _exposed_memories(),
            cognitive_request_id=RID_B,
        )

        seed_recall_control_plane(
            self.root,
            list(_LOADED_IDS),
            cognitive_request_id=RID_B,
        )

        other_surface = build_live_recall_usage_surface(
            conversation_id=CID,
            cognitive_request_id=RID_B,
            anchor_ref=memrefs[0],
            model_result=projected_model_result(
                list(_LOADED_IDS)
            ),
        )

        self.assertTrue(other_surface["stored"])

        other_ref = other_surface["usage_refs"][0][
            "useref"
        ]

        calls = []

        original = memory_usage_signal.record_usage

        def spy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        with patch.object(
            memory_usage_signal, "record_usage", spy
        ):
            report = self._attribute([other_ref])

        self._assert_refused(
            report, "invalid_usage_refs"
        )
        self.assertEqual(calls, [])

        usage = self._usage()

        self.assertEqual(usage["used_count"], 0)
        self.assertEqual(
            [
                event
                for event in usage["events"]
                if event["stage"] == "used"
            ],
            [],
        )

        # Request B's usage artifact is untouched as well.
        other_usage = read_usage(
            self.root,
            conversation_id=CID,
            cognitive_request_id=RID_B,
        )

        self.assertEqual(other_usage["used_count"], 0)

    def test_cross_conversation_ref_is_refused(self):
        memrefs = seed_live_exposure(
            self.root,
            _exposed_memories(),
            conversation_id=CID_B,
            cognitive_request_id=RID_B,
        )

        seed_recall_control_plane(
            self.root,
            list(_LOADED_IDS),
            conversation_id=CID_B,
            cognitive_request_id=RID_B,
        )

        other_surface = build_live_recall_usage_surface(
            conversation_id=CID_B,
            cognitive_request_id=RID_B,
            anchor_ref=memrefs[0],
            model_result=projected_model_result(
                list(_LOADED_IDS)
            ),
        )

        self.assertTrue(other_surface["stored"])

        other_ref = other_surface["usage_refs"][0][
            "useref"
        ]

        report = self._attribute([other_ref])

        self._assert_refused(
            report, "invalid_usage_refs"
        )

        self.assertEqual(
            self._usage()["used_count"], 0
        )

        other_usage = read_usage(
            self.root,
            conversation_id=CID_B,
            cognitive_request_id=RID_B,
        )

        self.assertEqual(other_usage["used_count"], 0)

    def test_usage_before_recall_is_refused_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"OMBRE_CONTEXT_STATE_DIR": root},
                clear=False,
            ):
                report = attribute_live_recall_usage(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    userefs=["useref_" + "a" * 32],
                )

                self._assert_refused(
                    report, "usage_surface_unavailable"
                )

                for directory in (
                    "memory_usage",
                    "recall_request",
                    "related_recall",
                    "live_recall_usage_surface",
                    "memory_lifecycle_events",
                    "memory_lifecycle_state",
                ):
                    self.assertFalse(
                        (Path(root) / directory).exists(),
                        directory,
                    )

    def test_too_many_refs_are_refused(self):
        with patch.dict(
            os.environ,
            {USAGE_REFS_MAX_ENV: "2"},
            clear=False,
        ):
            report = self._attribute(
                [self.refs[1], self.refs[2], self.refs[3]]
            )

        self._assert_refused(report, "too_many_refs")
        self.assertEqual(report["requested_ref_count"], 3)

        self.assertEqual(
            self._usage()["used_count"], 0
        )

    def test_corrupt_surface_is_refused(self):
        path = live_recall_usage_surface_path(CID, RID)

        artifact = json.loads(
            path.read_text(encoding="utf-8")
        )

        artifact["mappings"] = artifact["mappings"][:1]

        artifact["mappings"][0]["rank"] = 0

        path.write_text(
            json.dumps(artifact), encoding="utf-8"
        )

        report = self._attribute([self.refs[1]])

        self._assert_refused(
            report, "usage_surface_unavailable"
        )

        self.assertEqual(
            self._usage()["used_count"], 0
        )

    def test_invalid_request_binding_is_refused(self):
        for kwargs in (
            {"conversation_id": "nope"},
            {"cognitive_request_id": "nope"},
            {"conversation_id": None},
            {"cognitive_request_id": None},
        ):
            report = self._attribute(
                [self.refs[1]], **kwargs
            )

            self._assert_refused(
                report, "invalid_usage_refs"
            )

        self.assertEqual(
            self._usage()["used_count"], 0
        )

    def test_record_usage_failure_is_reported(self):
        with patch.object(
            memory_usage_signal,
            "record_usage",
            return_value={
                "stored": False,
                "reason": "usage_not_found",
            },
        ):
            report = self._attribute([self.refs[1]])

        self._assert_refused(
            report, "usage_record_failed"
        )

        self.assertEqual(
            report["accepted_ref_count"], 1
        )


class AckRenderingTests(unittest.TestCase):

    def test_recorded_ack_is_data_only(self):
        rendered = render_live_recall_usage_ack(
            {
                "version": "x",
                "mode": "y",
                "recorded": True,
                "reason": "usage_recorded",
                "requested_ref_count": 3,
                "accepted_ref_count": 2,
                "newly_used_count": 2,
                "duplicate_count": 0,
            }
        )

        self.assertTrue(
            rendered.startswith(
                "OMBRE MEMORY USAGE ACK DATA\n"
            )
        )

        payload = json.loads(
            rendered.split("\n", 1)[1]
        )

        self.assertEqual(
            payload,
            {
                "status": "recorded",
                "used_count": 2,
                "newly_used_count": 2,
                "duplicate": False,
            },
        )

        for forbidden in (
            "useref",
            "memory_id",
            "recall_id",
            "ctx_",
            "ctxreq_",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_duplicate_ack_flags_repeat(self):
        rendered = render_live_recall_usage_ack(
            {
                "recorded": True,
                "accepted_ref_count": 2,
                "newly_used_count": 0,
                "duplicate_count": 2,
            }
        )

        payload = json.loads(
            rendered.split("\n", 1)[1]
        )

        self.assertEqual(
            payload["status"], "recorded"
        )
        self.assertTrue(payload["duplicate"])
        self.assertEqual(payload["newly_used_count"], 0)

    def test_refused_ack_uses_a_safe_reason(self):
        for reason in sorted(SAFE_ATTRIBUTION_REASONS):
            rendered = render_live_recall_usage_ack(
                {
                    "recorded": False,
                    "reason": reason,
                }
            )

            payload = json.loads(
                rendered.split("\n", 1)[1]
            )

            self.assertEqual(
                payload["status"], "refused"
            )
            self.assertEqual(payload["reason"], reason)
            self.assertNotIn("useref", rendered)

    def test_unknown_reason_degrades_to_invalid_refs(self):
        for report in (
            None,
            {},
            {"recorded": False, "reason": "raw_boom"},
            "not-a-dict",
        ):
            rendered = render_live_recall_usage_ack(report)

            payload = json.loads(
                rendered.split("\n", 1)[1]
            )

            self.assertEqual(
                payload,
                {
                    "status": "refused",
                    "reason": "invalid_usage_refs",
                },
            )


class AttributionFlagTests(unittest.TestCase):

    def test_flag_default_is_off(self):
        with patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop(
                USAGE_ATTRIBUTION_ENV, None
            )

            self.assertFalse(
                live_recall_usage_attribution_enabled()
            )

    def test_flag_truthy_values(self):
        for value in ("1", "true", "TRUE", "on", "yes", " on "):
            with patch.dict(
                os.environ,
                {USAGE_ATTRIBUTION_ENV: value},
                clear=False,
            ):
                self.assertTrue(
                    live_recall_usage_attribution_enabled(),
                    value,
                )

        for value in ("0", "false", "off", "", "no"):
            with patch.dict(
                os.environ,
                {USAGE_ATTRIBUTION_ENV: value},
                clear=False,
            ):
                self.assertFalse(
                    live_recall_usage_attribution_enabled(),
                    value,
                )


class NoLifecycleIntegrationTests(AttributionCase):

    def test_used_is_recorded_but_no_lifecycle_artifact_appears(self):
        lifecycle_calls = []

        from ombrebrain.context import memory_lifecycle_event

        original = (
            memory_lifecycle_event.record_lifecycle_event_from_usage
        )

        def spy(*args, **kwargs):
            lifecycle_calls.append((args, kwargs))
            return original(*args, **kwargs)

        with patch.dict(
            os.environ,
            {LIFECYCLE_SHADOW_ENV: "1"},
            clear=False,
        ), patch.object(
            memory_lifecycle_event,
            "record_lifecycle_event_from_usage",
            spy,
        ):
            report = self._attribute(
                [self.refs[1], self.refs[2]]
            )

        self.assertTrue(report["recorded"])

        usage = self._usage()

        self.assertEqual(usage["used_count"], 2)

        # Usage attributed, and NOTHING downstream consumed it.
        self.assertEqual(lifecycle_calls, [])

        for directory in (
            "memory_lifecycle_events",
            "memory_lifecycle_state",
            "memory_reinforcement",
        ):
            self.assertFalse(
                (Path(self.root) / directory).exists(),
                directory,
            )


class StaticForbiddenChecksTests(unittest.TestCase):
    """Static guards on the three new attribution modules.

    ``used`` may only come from an explicit tool signal, so the new
    code must not be able to infer it: no answer-text scan, no
    embedding similarity, no LLM judge, no BucketManager, no retrieval,
    no filesystem glob and no lifecycle mutation.

    The check parses the real source and inspects IDENTIFIERS (not
    prose), so a docstring that explains why a shortcut is forbidden
    does not trip it.
    """

    _MODULES = (
        "live_recall_usage_surface",
        "live_recall_usage_attribution",
        "live_memory_transport_capability",
    )

    _FORBIDDEN_IDENTIFIERS = (
        "glob",
        "rglob",
        "bucketmanager",
        "bucket_manager",
        "retrieve",
        "retrieval",
        "embedding",
        "similarity",
        "cosine",
        "judge",
        "lifecycle",
        "reinforce",
        "reinforcement",
        "decay",
        "assistant",
        "answer",
    )

    def _identifiers(self, module_name):
        import ast
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ombrebrain"
            / "context"
            / (module_name + ".py")
        )

        source = path.read_text(encoding="utf-8")

        tree = ast.parse(source)

        found: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                found.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                found.add(node.attr.lower())
            elif isinstance(node, ast.alias):
                found.add(node.name.lower())
            elif (
                isinstance(node, ast.ImportFrom)
                and isinstance(node.module, str)
            ):
                found.add(node.module.lower())

        return found, source

    def test_no_inference_or_lifecycle_coupling(self):
        for module_name in self._MODULES:
            identifiers, _source = self._identifiers(
                module_name
            )

            for forbidden in self._FORBIDDEN_IDENTIFIERS:
                with self.subTest(
                    module=module_name,
                    forbidden=forbidden,
                ):
                    self.assertFalse(
                        {
                            identifier
                            for identifier in identifiers
                            if forbidden in identifier
                        },
                        forbidden,
                    )

    def test_no_recall_id_index_or_recent_file_guess(self):
        for module_name in self._MODULES:
            identifiers, source = self._identifiers(
                module_name
            )

            self.assertNotIn(
                ".rglob(", source, module_name
            )
            self.assertNotIn(
                ".glob(", source, module_name
            )
            self.assertNotIn(
                "sorted(", source, module_name
            )

            # No "pick the newest file" identifier either.
            self.assertFalse(
                {
                    identifier
                    for identifier in identifiers
                    if "recent" in identifier
                    or "newest" in identifier
                },
                module_name,
            )

    def test_surface_never_writes_usage(self):
        identifiers, _source = self._identifiers(
            "live_recall_usage_surface"
        )

        # The surface may READ the loaded set, but the write APIs of
        # the frozen Usage Signal must never appear in it.
        for forbidden in (
            "record_usage",
            "record_memory_loaded",
            "record_recall_requested",
        ):
            self.assertFalse(
                {
                    identifier
                    for identifier in identifiers
                    if forbidden in identifier
                },
                forbidden,
            )

    def test_attribution_delegates_the_write_to_usage_signal(self):
        _identifiers, source = self._identifiers(
            "live_recall_usage_attribution"
        )

        self.assertIn(
            "memory_usage_signal.record_usage(", source
        )

        # No direct write of the memory_usage artifact.
        for forbidden in (
            "atomic_write",
            "memory_usage_path",
            "open(",
        ):
            self.assertNotIn(forbidden, source)

    def test_provider_transport_gateway_is_untouched(self):
        from pathlib import Path

        gateway = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "web"
            / "gateway.py"
        ).read_text(encoding="utf-8")

        # The raw streaming passthrough stays exactly as frozen.
        self.assertIn(
            "async for chunk in upstream_response.aiter_raw():",
            gateway,
        )

        for forbidden in (
            "use_memory",
            "useref",
            "usage_refs",
            "memory_usage_signal",
        ):
            self.assertNotIn(
                forbidden, gateway.lower()
            )


if __name__ == "__main__":
    unittest.main()