from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from ombrebrain.context import (
    memory_delivery_shadow as delivery,
)
from ombrebrain.context import (
    memory_source_evidence as evidence,
)
from ombrebrain.context.memory_delivery_shadow import (
    MAX_DELIVERY_ROWS,
    build_memory_delivery_plan_artifact,
    is_valid_memory_delivery_plan_artifact,
    update_memory_delivery_plan,
)
from ombrebrain.context.memory_source_evidence import (
    MemorySourceEvidenceSnapshot,
    frontend_evidence_units,
    ob_evidence_units,
)
from ombrebrain.gateway import (
    FrontendMemorySegment,
    OperitAdapter,
    extract_frontend_memory_segments,
)


CID = "ctx_0123456789abcdef"
RID = "ctxreq_" + "a" * 32

_SENTINEL_FRONTEND = "FRONTEND-SECRET-BODY"
_SENTINEL_OB = "OB-SECRET-CONTENT"


def _attachment(filename, body, attachment_id="x"):
    return (
        '<attachment id="'
        + attachment_id
        + '" filename="'
        + filename
        + '" type="text/plain">'
        + body
        + "</attachment>"
    )


def _body(messages):
    return json.dumps(
        {"model": "example", "messages": messages}
    ).encode("utf-8")


def _frontend_current_body(*bodies):
    text = "q " + " ".join(
        _attachment(
            "relevant_memories.txt",
            body,
            "front_%d" % index,
        )
        for index, body in enumerate(bodies)
    )

    return _body(
        [{"role": "user", "content": text}]
    )


def _frontend_units(*bodies):
    payload = json.loads(
        _frontend_current_body(*bodies)
    )
    request = OperitAdapter().adapt(payload)
    return frontend_evidence_units(
        extract_frontend_memory_segments(request)
    )


def _memory(memory_id, content):
    return {"id": memory_id, "content": content}


def _ob_units(memories):
    return ob_evidence_units(memories)


def _snap(
    units,
    *,
    frontend_available=True,
    ob_available=True,
    revision=5,
):
    return MemorySourceEvidenceSnapshot(
        version=evidence._VERSION,
        conversation_id=CID,
        cognitive_request_id=RID,
        source_unified_revision=revision,
        frontend_observation_available=(
            frontend_available
        ),
        ob_observation_available=ob_available,
        units=tuple(units),
    )


def _policy(
    *,
    decision="allow_shadow",
    eligible=None,
    conversation_id=CID,
    revision=5,
    version="memory-surfacing-policy.v1",
    mode="shadow_only",
):
    return {
        "version": version,
        "mode": mode,
        "conversation_id": conversation_id,
        "source_unified_revision": revision,
        "decision": decision,
        "eligible": (
            []
            if eligible is None
            and decision == "no_surface"
            else eligible
        ),
    }


def _build(units, **kwargs):
    policy = kwargs.pop("policy", None)
    snap = kwargs.pop("snapshot", None)

    if snap is None:
        snap = _snap(units, **kwargs)

    return build_memory_delivery_plan_artifact(
        conversation_id=CID,
        cognitive_request_id=RID,
        source_unified_revision=5,
        source_evidence_snapshot=snap,
        base_policy=policy,
    )


def _delivery_dir(root):
    return Path(root) / "memory_delivery_plan" / CID


class DeliveryBuilderTests(unittest.TestCase):

    def test_empty_pool(self):
        artifact = _build([])

        self.assertEqual(artifact["rows"], [])
        self.assertEqual(
            artifact["metrics"]["unit_count"], 0
        )
        self.assertFalse(
            artifact["base_policy_available"]
        )

    def test_frontend_row_is_direct_data(self):
        artifact = _build(
            _frontend_units("MEM-A")
        )

        self.assertEqual(len(artifact["rows"]), 1)

        row = artifact["rows"][0]

        self.assertEqual(row["source"], "frontend")
        self.assertEqual(
            row["immediate_delivery_mode"],
            "direct_data",
        )
        self.assertEqual(
            row["full_content_access_mode"],
            "already_present",
        )
        self.assertIsNone(
            row["policy_binding_available"]
        )
        self.assertIsNone(
            row["base_surfacing_eligible"]
        )
        self.assertIsNone(row["base_eligible_rank"])
        self.assertTrue(row["currently_model_visible"])
        self.assertFalse(row["retrieved"])

    def test_ob_eligible_is_flash_cue(self):
        artifact = _build(
            _ob_units([_memory("m1", "OB-A")]),
            policy=_policy(
                eligible=[_memory("m1", "OB-A")]
            ),
        )

        row = artifact["rows"][0]

        self.assertEqual(
            row["immediate_delivery_mode"],
            "flash_cue",
        )
        self.assertEqual(
            row["full_content_access_mode"],
            "recall_required",
        )
        self.assertTrue(
            row["policy_binding_available"]
        )
        self.assertTrue(
            row["base_surfacing_eligible"]
        )
        self.assertEqual(row["base_eligible_rank"], 0)
        self.assertFalse(row["currently_model_visible"])
        self.assertTrue(row["retrieved"])

    def test_ob_ineligible_is_drop(self):
        artifact = _build(
            _ob_units([_memory("m1", "OB-A")]),
            policy=_policy(eligible=[]),
        )

        row = artifact["rows"][0]

        self.assertEqual(
            row["immediate_delivery_mode"], "drop"
        )
        self.assertEqual(
            row["full_content_access_mode"], "none"
        )
        self.assertFalse(
            row["base_surfacing_eligible"]
        )
        self.assertIsNone(row["base_eligible_rank"])

    def test_ob_no_surface_is_drop(self):
        artifact = _build(
            _ob_units([_memory("m1", "OB-A")]),
            policy=_policy(decision="no_surface"),
        )

        row = artifact["rows"][0]

        self.assertEqual(
            row["immediate_delivery_mode"], "drop"
        )
        self.assertTrue(
            artifact["base_policy_available"]
        )

    def test_missing_policy_is_unknown(self):
        artifact = _build(
            _ob_units([_memory("m1", "OB-A")]),
            policy=None,
        )

        row = artifact["rows"][0]

        self.assertEqual(
            row["immediate_delivery_mode"], "unknown"
        )
        self.assertEqual(
            row["full_content_access_mode"], "unknown"
        )
        self.assertFalse(
            row["policy_binding_available"]
        )
        self.assertIsNone(
            row["base_surfacing_eligible"]
        )
        self.assertFalse(
            artifact["base_policy_available"]
        )

    def test_foreign_policy_is_unknown(self):
        artifact = _build(
            _ob_units([_memory("m1", "OB-A")]),
            policy=_policy(
                eligible=[_memory("m1", "OB-A")],
                conversation_id="ctx_ffffffffffffffff",
            ),
        )

        self.assertEqual(
            artifact["rows"][0][
                "immediate_delivery_mode"
            ],
            "unknown",
        )

    def test_wrong_revision_policy_is_unknown(self):
        artifact = _build(
            _ob_units([_memory("m1", "OB-A")]),
            policy=_policy(
                eligible=[_memory("m1", "OB-A")],
                revision=99,
            ),
        )

        self.assertEqual(
            artifact["rows"][0][
                "immediate_delivery_mode"
            ],
            "unknown",
        )

    def test_unknown_decision_is_unknown(self):
        artifact = _build(
            _ob_units([_memory("m1", "OB-A")]),
            policy=_policy(decision="banana"),
        )

        self.assertEqual(
            artifact["rows"][0][
                "immediate_delivery_mode"
            ],
            "unknown",
        )

    def test_malformed_eligible_shape_is_unknown(self):
        base = _policy(eligible=[_memory("m1", "OB-A")])
        base["eligible"] = "not-a-list"

        artifact = _build(
            _ob_units([_memory("m1", "OB-A")]),
            policy=base,
        )

        self.assertEqual(
            artifact["rows"][0][
                "immediate_delivery_mode"
            ],
            "unknown",
        )

    def test_no_surface_with_nonempty_eligible_is_unknown(
        self,
    ):
        base = _policy(decision="no_surface")
        base["eligible"] = [_memory("m1", "OB-A")]

        artifact = _build(
            _ob_units([_memory("m1", "OB-A")]),
            policy=base,
        )

        self.assertEqual(
            artifact["rows"][0][
                "immediate_delivery_mode"
            ],
            "unknown",
        )

    def test_duplicate_ob_candidates_occurrence_safe(
        self,
    ):
        artifact = _build(
            _ob_units(
                [
                    _memory("m1", "DUP"),
                    _memory("m2", "DUP"),
                ]
            ),
            policy=_policy(
                eligible=[_memory("m1", "DUP")]
            ),
        )

        rows = artifact["rows"]

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["immediate_delivery_mode"],
            "flash_cue",
        )
        self.assertEqual(
            rows[1]["immediate_delivery_mode"],
            "drop",
        )
        # Distinct evidence occurrences, distinct refs.
        self.assertNotEqual(
            rows[0]["evidence_ref"],
            rows[1]["evidence_ref"],
        )

    def test_frontend_first_then_ob(self):
        artifact = _build(
            _frontend_units("MEM-A")
            + _ob_units([_memory("m1", "OB-A")]),
            policy=_policy(
                eligible=[_memory("m1", "OB-A")]
            ),
        )

        self.assertEqual(
            [
                row["source"]
                for row in artifact["rows"]
            ],
            ["frontend", "ob"],
        )

    def test_metrics_are_complete_and_consistent(self):
        artifact = _build(
            _frontend_units("MEM-A")
            + _ob_units(
                [
                    _memory("m1", "OB-A"),
                    _memory("m2", "OB-B"),
                ]
            ),
            policy=_policy(
                eligible=[_memory("m1", "OB-A")]
            ),
        )

        metrics = artifact["metrics"]

        self.assertEqual(metrics["unit_count"], 3)
        self.assertEqual(
            metrics["frontend_unit_count"], 1
        )
        self.assertEqual(metrics["ob_unit_count"], 2)
        self.assertEqual(
            metrics["direct_data_count"], 1
        )
        self.assertEqual(
            metrics["flash_cue_count"], 1
        )
        self.assertEqual(metrics["drop_count"], 1)
        self.assertEqual(metrics["unknown_count"], 0)
        self.assertEqual(
            metrics["frontend_direct_data_count"], 1
        )
        self.assertEqual(
            metrics["ob_flash_cue_count"], 1
        )
        self.assertEqual(metrics["ob_drop_count"], 1)
        self.assertEqual(
            metrics["base_policy_available_ob_count"],
            2,
        )
        self.assertEqual(
            metrics[
                "base_policy_unavailable_ob_count"
            ],
            0,
        )
        self.assertEqual(
            metrics["current_model_visible_count"], 1
        )

    def test_has_no_live_authority(self):
        artifact = _build([])

        self.assertEqual(
            artifact["authority"], "proposal_only"
        )
        self.assertFalse(artifact["live_authority"])
        self.assertEqual(
            artifact["policy_stage"],
            "base_surfacing_pre_lifecycle",
        )
        self.assertFalse(artifact["flash_required"])
        self.assertFalse(
            artifact["lifecycle_required"]
        )

    def test_truncation(self):
        memories = [
            _memory("m%d" % i, "OB-%d" % i)
            for i in range(MAX_DELIVERY_ROWS + 6)
        ]

        artifact = _build(
            _ob_units(memories),
            policy=_policy(eligible=memories),
        )

        self.assertTrue(artifact["rows_truncated"])
        self.assertEqual(
            artifact["rows_truncated_count"], 6
        )
        self.assertEqual(
            len(artifact["rows"]),
            MAX_DELIVERY_ROWS,
        )
        self.assertEqual(
            artifact["metrics"]["unit_count"],
            MAX_DELIVERY_ROWS + 6,
        )
        self.assertTrue(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )


class DeliveryValidatorTests(unittest.TestCase):

    def _valid(self):
        return _build(
            _frontend_units("MEM-A")
            + _ob_units([_memory("m1", "OB-A")]),
            policy=_policy(
                eligible=[_memory("m1", "OB-A")]
            ),
        )

    def test_valid_artifact(self):
        self.assertTrue(
            is_valid_memory_delivery_plan_artifact(
                self._valid()
            )
        )

    def test_rejects_bad_top_level(self):
        mutations = {
            "version": "memory-delivery-plan-shadow.v2",
            "mode": "live",
            "authority": "live_authority",
            "policy_stage": "post_lifecycle",
            "flash_required": True,
            "lifecycle_required": True,
            "live_authority": True,
            "created_at": "not-a-date",
        }

        for key, value in mutations.items():
            with self.subTest(key=key):
                artifact = self._valid()
                artifact[key] = value
                self.assertFalse(
                    is_valid_memory_delivery_plan_artifact(
                        artifact
                    )
                )

    def test_rejects_forged_metric(self):
        artifact = self._valid()
        artifact["metrics"]["flash_cue_count"] = 99

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

    def test_rejects_malformed_row(self):
        artifact = self._valid()
        artifact["rows"][0][
            "immediate_delivery_mode"
        ] = "banana"

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

    def test_rejects_frontend_as_flash_cue(self):
        artifact = self._valid()
        artifact["rows"][0][
            "immediate_delivery_mode"
        ] = "flash_cue"

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

    def test_rejects_ob_as_direct_data(self):
        artifact = self._valid()
        artifact["rows"][1][
            "immediate_delivery_mode"
        ] = "direct_data"

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

    def test_rejects_ob_with_null_binding(self):
        artifact = self._valid()
        artifact["rows"][1][
            "policy_binding_available"
        ] = None

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

    def test_rejects_frontend_with_eligibility(self):
        artifact = self._valid()
        artifact["rows"][0][
            "base_surfacing_eligible"
        ] = False

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

    def test_rejects_forged_frontend_row_when_unavailable(
        self,
    ):
        artifact = _build(
            _ob_units([_memory("m1", "OB-A")]),
            frontend_available=False,
            policy=_policy(
                eligible=[_memory("m1", "OB-A")]
            ),
        )

        self.assertTrue(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

        forged = _frontend_units("MEM-A")[0]

        artifact["rows"].append(
            {
                "evidence_ref": "a" * 64,
                "source": "frontend",
                "kind": "frontend_attachment",
                "granularity": "attachment",
                "source_scope": "current",
                "source_index": 0,
                "source_occurrence": 0,
                "content_digest": forged.content_digest,
                "normalized_digest":
                    forged.normalized_digest,
                "content_chars": forged.content_chars,
                "content_estimated_tokens":
                    forged.content_estimated_tokens,
                "currently_model_visible": True,
                "retrieved": False,
                "policy_binding_available": None,
                "base_surfacing_eligible": None,
                "base_eligible_rank": None,
                "immediate_delivery_mode":
                    "direct_data",
                "full_content_access_mode":
                    "already_present",
                "control_basis":
                    "existing_frontend_direct_baseline",
                "reason":
                    "frontend_attachment_already_model_visible",
            }
        )

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

    def test_rejects_ob_row_when_unavailable(self):
        artifact = _build(
            _frontend_units("MEM-A"),
            ob_available=False,
        )

        self.assertTrue(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

        artifact["rows"].append(
            dict(artifact["rows"][0])
        )  # a frontend row copy is fine

        # Now forge an OB row while OB is unavailable.
        ob_row = _build(
            _ob_units([_memory("m1", "OB-A")]),
            policy=_policy(
                eligible=[_memory("m1", "OB-A")]
            ),
        )["rows"][0]

        artifact["rows"].append(ob_row)

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

    def test_rejects_inconsistent_binding_fact(self):
        artifact = self._valid()
        # Artifact says policy available but OB row says otherwise.
        artifact["rows"][1][
            "policy_binding_available"
        ] = False

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

    def test_rejects_bad_truncation_bookkeeping(self):
        artifact = self._valid()

        artifact["rows_truncated"] = True
        artifact["rows_truncated_count"] = 0

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact
            )
        )

        artifact2 = self._valid()
        artifact2["rows_truncated_count"] = 5
        artifact2["rows_truncated"] = True

        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                artifact2
            )
        )

    def test_rejects_non_snapshot(self):
        self.assertFalse(
            is_valid_memory_delivery_plan_artifact(
                {"version": delivery.VERSION}
            )
        )


class DeliveryPrivacyTests(unittest.TestCase):

    def test_no_raw_text_in_artifact(self):
        artifact = _build(
            _frontend_units(_SENTINEL_FRONTEND)
            + _ob_units(
                [_memory("m1", _SENTINEL_OB)]
            ),
            policy=_policy(
                eligible=[_memory("m1", _SENTINEL_OB)]
            ),
        )

        serialized = json.dumps(artifact)

        self.assertNotIn(
            _SENTINEL_FRONTEND, serialized
        )
        self.assertNotIn(_SENTINEL_OB, serialized)
        self.assertNotIn("m1", serialized)
        self.assertNotIn(
            "raw_attachment", serialized
        )


class DeliveryPersistenceTests(unittest.TestCase):

    def _snapshot(self):
        return _snap(
            _frontend_units("MEM-A")
            + _ob_units([_memory("m1", "OB-A")])
        )

    def test_persists_bound_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"OMBRE_CONTEXT_STATE_DIR": root},
                clear=False,
            ):
                report = update_memory_delivery_plan(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    source_evidence_snapshot=(
                        self._snapshot()
                    ),
                    expected_unified_revision=5,
                    base_policy=_policy(
                        eligible=[
                            _memory("m1", "OB-A")
                        ]
                    ),
                )

            self.assertTrue(report["stored"])
            self.assertTrue(
                report["plan"]["live_authority"]
                is False
            )

            files = list(
                _delivery_dir(root).iterdir()
            )

        self.assertEqual(len(files), 1)

    def test_foreign_evidence_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"OMBRE_CONTEXT_STATE_DIR": root},
                clear=False,
            ):
                report = update_memory_delivery_plan(
                    conversation_id=CID,
                    cognitive_request_id=(
                        "ctxreq_" + "b" * 32
                    ),
                    source_evidence_snapshot=(
                        self._snapshot()
                    ),
                    expected_unified_revision=5,
                    base_policy=None,
                )

            self.assertFalse(report["stored"])
            self.assertEqual(
                report["reason"],
                "evidence_not_bound",
            )
            self.assertFalse(
                _delivery_dir(root).exists()
            )

    def test_invalid_bindings_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"OMBRE_CONTEXT_STATE_DIR": root},
                clear=False,
            ):
                bad_conversation = (
                    update_memory_delivery_plan(
                        conversation_id="nope",
                        cognitive_request_id=RID,
                        source_evidence_snapshot=(
                            self._snapshot()
                        ),
                        expected_unified_revision=5,
                        base_policy=None,
                    )
                )

                bad_request = (
                    update_memory_delivery_plan(
                        conversation_id=CID,
                        cognitive_request_id="nope",
                        source_evidence_snapshot=(
                            self._snapshot()
                        ),
                        expected_unified_revision=5,
                        base_policy=None,
                    )
                )

                bad_revision = (
                    update_memory_delivery_plan(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        source_evidence_snapshot=(
                            self._snapshot()
                        ),
                        expected_unified_revision=0,
                        base_policy=None,
                    )
                )

        self.assertEqual(
            bad_conversation["reason"],
            "invalid_conversation_id",
        )
        self.assertEqual(
            bad_request["reason"],
            "invalid_cognitive_request_id",
        )
        self.assertEqual(
            bad_revision["reason"],
            "invalid_expected_unified_revision",
        )

    def test_store_failure_is_fail_open(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"OMBRE_CONTEXT_STATE_DIR": root},
                clear=False,
            ):
                with patch.object(
                    delivery,
                    "atomic_write",
                    side_effect=OSError("disk"),
                ):
                    report = (
                        update_memory_delivery_plan(
                            conversation_id=CID,
                            cognitive_request_id=RID,
                            source_evidence_snapshot=(
                                self._snapshot()
                            ),
                            expected_unified_revision=5,
                            base_policy=None,
                        )
                    )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"], "persist_failed"
        )

    def test_same_conversation_requests_do_not_mix(
        self,
    ):
        snap_a = _snap(
            _frontend_units("FRONT-A")
            + _ob_units([_memory("a1", "OB-A")])
        )
        snap_b = _snap(
            _frontend_units("FRONT-B")
            + _ob_units(
                [
                    _memory("b1", "OB-B"),
                    _memory("b2", "OB-B2"),
                ]
            )
        )

        rid_a = "ctxreq_" + "4" * 32
        rid_b = "ctxreq_" + "5" * 32

        # Each snapshot is bound to its own request id, so rebuild
        # them with the right cognitive_request_id.
        def _bound(snap, rid):
            return MemorySourceEvidenceSnapshot(
                version=snap.version,
                conversation_id=snap.conversation_id,
                cognitive_request_id=rid,
                source_unified_revision=(
                    snap.source_unified_revision
                ),
                frontend_observation_available=(
                    snap.frontend_observation_available
                ),
                ob_observation_available=(
                    snap.ob_observation_available
                ),
                units=snap.units,
            )

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"OMBRE_CONTEXT_STATE_DIR": root},
                clear=False,
            ):
                def _run(rid, snap):
                    return update_memory_delivery_plan(
                        conversation_id=CID,
                        cognitive_request_id=rid,
                        source_evidence_snapshot=snap,
                        expected_unified_revision=5,
                        base_policy=None,
                    )

                with ThreadPoolExecutor(
                    max_workers=2
                ) as pool:
                    future_a = pool.submit(
                        _run,
                        rid_a,
                        _bound(snap_a, rid_a),
                    )
                    future_b = pool.submit(
                        _run,
                        rid_b,
                        _bound(snap_b, rid_b),
                    )
                    future_a.result()
                    future_b.result()

            artifacts = [
                json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
                for path in sorted(
                    _delivery_dir(root).iterdir()
                )
            ]

        self.assertEqual(len(artifacts), 2)

        by_request = {
            artifact["cognitive_request_id"]:
                artifact
            for artifact in artifacts
        }

        self.assertEqual(
            by_request[rid_a]["metrics"][
                "ob_unit_count"
            ],
            1,
        )
        self.assertEqual(
            by_request[rid_b]["metrics"][
                "ob_unit_count"
            ],
            2,
        )


if __name__ == "__main__":
    unittest.main()
