from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from ombrebrain.context import (
    memory_authority_migration_shadow as migration,
)
from ombrebrain.context import (
    memory_delivery_shadow as delivery,
)
from ombrebrain.context import (
    memory_source_evidence as evidence,
)
from ombrebrain.context.memory_authority_migration_shadow import (
    COVERAGE_FRONTEND_GAP,
    COVERAGE_OB_CUE,
    COVERAGE_OB_UNKNOWN,
    COVERAGE_OB_WITHHOLD,
    REASON_EVIDENCE_MISMATCH,
    REASON_GAPS_PRESENT,
    REASON_NO_RENDERER,
    REASON_SOURCE_UNAVAILABLE,
    REQ_FRONTEND_CHANNEL,
    REQ_NONE,
    REQ_POLICY_RESOLUTION,
    STATUS_BLOCKED,
    STATUS_COVERED,
    build_memory_authority_migration_artifact,
    delivery_plan_bound_to_request,
    is_valid_memory_authority_migration_artifact,
    observe_frontend_final_presence,
    update_memory_authority_migration,
)
from ombrebrain.context.memory_delivery_shadow import (
    build_memory_delivery_plan_artifact,
)
from ombrebrain.context.memory_source_evidence import (
    MemorySourceEvidenceSnapshot,
    frontend_evidence_units,
    ob_evidence_units,
)
from ombrebrain.gateway import (
    OperitAdapter,
    extract_frontend_memory_segments,
)


CID = "ctx_0123456789abcdef"
RID = "ctxreq_" + "a" * 32
REV = 5

_SENTINEL_FRONTEND = "FRONTEND_SECRET_MIGRATION"
_SENTINEL_OB = "OB_CONTENT_SECRET_MIGRATION"
_SENTINEL_MEMID = "OB_MEMORY_ID_SECRET_MIGRATION"


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


def _frontend_body(*bodies):
    text = "q " + " ".join(
        _attachment(
            "relevant_memories.txt",
            body,
            "front_%d" % index,
        )
        for index, body in enumerate(bodies)
    )

    return _body([{"role": "user", "content": text}])


def _frontend_body_with_lead(lead, *bodies):
    """Same Frontend current message, but with a leading text block.

    The Frontend attachment stays in the CURRENT (last) message, so its
    placement and canonical envelope are unchanged. The extra leading
    block only changes the raw bytes, which is how a real OB Live
    Exposure mutation looks to the observer.
    """

    text = "q " + " ".join(
        _attachment(
            "relevant_memories.txt",
            body,
            "front_%d" % index,
        )
        for index, body in enumerate(bodies)
    )

    return _body(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": lead},
                    {"type": "text", "text": text},
                ],
            }
        ]
    )


def _frontend_units_from(body):
    payload = json.loads(body)
    request = OperitAdapter().adapt(payload)
    return frontend_evidence_units(
        extract_frontend_memory_segments(request)
    )


def _memory(memory_id, content):
    return {"id": memory_id, "content": content}


def _ob_units(memories):
    return ob_evidence_units(memories)


def _snap(units, *, rid=RID, revision=REV):
    return MemorySourceEvidenceSnapshot(
        version=evidence._VERSION,
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        frontend_observation_available=True,
        ob_observation_available=True,
        units=tuple(units),
    )


def _snap_avail(
    units,
    *,
    frontend_available=True,
    ob_available=True,
    rid=RID,
    revision=REV,
):
    """A snapshot with EXPLICIT source availability bits.

    ``AVAILABLE + zero units`` and ``UNAVAILABLE`` are different states
    and must stay distinguishable.
    """

    return MemorySourceEvidenceSnapshot(
        version=evidence._VERSION,
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        frontend_observation_available=(
            frontend_available
        ),
        ob_observation_available=ob_available,
        units=tuple(units),
    )


def _policy(*, decision="allow_shadow", eligible=None):
    return {
        "version": "memory-surfacing-policy.v1",
        "mode": "shadow_only",
        "conversation_id": CID,
        "source_unified_revision": REV,
        "decision": decision,
        "eligible": (
            []
            if eligible is None
            and decision == "no_surface"
            else eligible
        ),
    }


def _plan(snap, *, policy=None, rid=RID, revision=REV):
    return build_memory_delivery_plan_artifact(
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        source_evidence_snapshot=snap,
        base_policy=policy,
    )


def _receipt(pre_live, final, *, count=1, tokens=10):
    return {
        "applied": True,
        "original_sha256": hashlib.sha256(
            pre_live
        ).hexdigest(),
        "selected_sha256": hashlib.sha256(
            final
        ).hexdigest(),
        "original_bytes": len(pre_live),
        "selected_bytes": len(final),
        "byte_delta": len(final) - len(pre_live),
        "live_exposed_count": count,
        "estimated_tokens": tokens,
    }


def _migration_dir(root):
    return (
        Path(root)
        / "memory_authority_migration"
        / CID
    )


def _run(
    root,
    *,
    snap,
    plan,
    pre_live,
    final,
    rid=RID,
    revision=REV,
    receipt=None,
    cid=CID,
):
    patches = []

    if receipt is not None:
        patches.append(
            patch.object(
                migration,
                "read_live_memory_exposure",
                return_value=receipt,
            )
        )

    with patch.dict(
        os.environ,
        {"OMBRE_CONTEXT_STATE_DIR": str(root)},
        clear=False,
    ):
        for item in patches:
            item.start()
        try:
            return update_memory_authority_migration(
                conversation_id=cid,
                cognitive_request_id=rid,
                source_evidence_snapshot=snap,
                delivery_plan=plan,
                source_unified_revision=revision,
                pre_live_body=pre_live,
                final_body=final,
            )
        finally:
            for item in reversed(patches):
                item.stop()


class FrontendObservationTests(unittest.TestCase):

    def test_exact_multiset_is_verified(self):
        body = _frontend_body("MEM-A")

        available, verified = (
            observe_frontend_final_presence(
                body, _snap(_frontend_units_from(body))
            )
        )

        self.assertTrue(available)
        self.assertTrue(verified)

    def test_duplicate_identical_attachments_stay_distinct(
        self,
    ):
        body = _frontend_body("SAME", "SAME")

        snap = _snap(_frontend_units_from(body))

        self.assertEqual(
            sum(
                1
                for unit in snap.units
                if unit.source == evidence.SOURCE_FRONTEND
            ),
            2,
        )

        self.assertEqual(
            observe_frontend_final_presence(body, snap),
            (True, True),
        )

    def test_one_disappearing_duplicate_is_not_verified(
        self,
    ):
        full = _frontend_body("SAME", "SAME")
        one = _frontend_body("SAME")

        snap = _snap(_frontend_units_from(full))

        # The snapshot still expects TWO occurrences, but the final
        # body carries only ONE, so the multiset disagrees.
        self.assertEqual(
            observe_frontend_final_presence(one, snap),
            (True, False),
        )

    def test_mismatched_frontend_body_is_not_verified(
        self,
    ):
        snap = _snap(
            _frontend_units_from(_frontend_body("MEM-A"))
        )
        other = _frontend_body("MEM-B")

        self.assertEqual(
            observe_frontend_final_presence(other, snap),
            (True, False),
        )

    def test_uncanonicalizable_final_body_is_unavailable(
        self,
    ):
        snap = _snap(
            _frontend_units_from(_frontend_body("MEM-A"))
        )

        self.assertEqual(
            observe_frontend_final_presence(
                b"not-json", snap
            ),
            (False, False),
        )


class CoverageMappingTests(unittest.TestCase):

    def test_frontend_only_is_a_direct_bypass_gap(self):
        body = _frontend_body("MEM-A")

        snap = _snap(_frontend_units_from(body))

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=body,
                final=body,
            )

        self.assertTrue(report["stored"])

        artifact = report["artifact"]

        self.assertTrue(
            artifact["frontend_direct_mouth_active"]
        )
        self.assertFalse(
            artifact["ob_live_cue_mouth_active"]
        )
        self.assertFalse(artifact["dual_mouth_active"])

        self.assertGreater(
            artifact["metrics"]["contract_gap_unit_count"],
            0,
        )
        self.assertGreater(
            artifact["metrics"][
                "frontend_replacement_required_count"
            ],
            0,
        )
        self.assertEqual(
            artifact["rows"][0]["authority_coverage"],
            COVERAGE_FRONTEND_GAP,
        )
        self.assertEqual(
            artifact["rows"][0]["replacement_requirement"],
            REQ_FRONTEND_CHANNEL,
        )
        self.assertEqual(
            artifact["contract_coverage_status"],
            STATUS_BLOCKED,
        )
        self.assertFalse(artifact["live_cutover_ready"])
        self.assertEqual(
            artifact["live_cutover_reason"],
            REASON_GAPS_PRESENT,
        )

    def test_ob_controlled_only_is_covered(self):
        eligible = _memory("m1", "OB-A")

        snap = _snap(
            _ob_units(
                [
                    eligible,
                    _memory("m2", "OB-B"),
                ]
            )
        )

        plan = _plan(snap, policy=_policy(eligible=[eligible]))

        empty = _frontend_body()

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=empty,
                final=empty,
            )

        self.assertTrue(report["stored"])

        artifact = report["artifact"]
        metrics = artifact["metrics"]

        self.assertEqual(
            metrics["contract_gap_unit_count"], 0
        )
        self.assertEqual(
            artifact["contract_coverage_status"],
            STATUS_COVERED,
        )
        # Covered, but v0 still never claims a live cutover.
        self.assertFalse(artifact["live_cutover_ready"])
        self.assertEqual(
            artifact["live_cutover_reason"],
            REASON_NO_RENDERER,
        )
        self.assertEqual(
            metrics["ob_controlled_cue_contract_count"],
            1,
        )
        self.assertEqual(
            metrics[
                "ob_controlled_withhold_contract_count"
            ],
            1,
        )
        coverages = [
            row["authority_coverage"]
            for row in artifact["rows"]
        ]
        self.assertIn(COVERAGE_OB_CUE, coverages)
        self.assertIn(COVERAGE_OB_WITHHOLD, coverages)

    def test_ob_unknown_only_is_a_gap(self):
        snap = _snap(
            _ob_units([_memory("m1", "OB-A")])
        )

        # No policy: the base policy is unavailable, so every OB unit
        # is UKNOWN rather than being treated as a controlled withhold.
        plan = _plan(snap, policy=None)

        empty = _frontend_body()

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=empty,
                final=empty,
            )

        artifact = report["artifact"]
        metrics = artifact["metrics"]

        self.assertGreater(
            metrics["ob_policy_unknown_count"], 0
        )
        self.assertGreater(
            metrics["contract_gap_unit_count"], 0
        )
        self.assertEqual(
            artifact["contract_coverage_status"],
            STATUS_BLOCKED,
        )
        self.assertEqual(
            artifact["rows"][0]["authority_coverage"],
            COVERAGE_OB_UNKNOWN,
        )
        self.assertEqual(
            artifact["rows"][0][
                "replacement_requirement"
            ],
            REQ_POLICY_RESOLUTION,
        )

    def test_both_mouths_active(self):
        pre_live = _frontend_body("MEM-A")
        final = _frontend_body_with_lead(
            "memory flash cue", "MEM-A"
        )

        snap = _snap(_frontend_units_from(final))

        plan = _plan(snap, policy=None)

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=pre_live,
                final=final,
                receipt=_receipt(
                    pre_live, final, count=1, tokens=7
                ),
            )

        self.assertTrue(report["stored"])

        artifact = report["artifact"]

        self.assertTrue(
            artifact["frontend_direct_mouth_active"]
        )
        self.assertTrue(
            artifact["ob_live_cue_mouth_active"]
        )
        self.assertTrue(artifact["dual_mouth_active"])

        exposure = artifact["current_exposure"]

        self.assertTrue(
            exposure["actual_ob_live_exposure_applied"]
        )
        self.assertEqual(
            exposure["actual_ob_live_exposed_count"], 1
        )
        self.assertEqual(
            exposure[
                "actual_ob_live_exposure_estimated_tokens"
            ],
            7,
        )

    def test_delivery_flash_cue_but_live_exposure_off(self):
        eligible = _memory("m1", "OB-A")

        snap = _snap(_ob_units([eligible]))

        plan = _plan(snap, policy=_policy(eligible=[eligible]))

        empty = _frontend_body()

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=empty,
                final=empty,
            )

        artifact = report["artifact"]
        exposure = artifact["current_exposure"]

        # The CONTRACT says a cue would be used...
        self.assertEqual(
            artifact["rows"][0]["authority_coverage"],
            COVERAGE_OB_CUE,
        )
        # ...but the ACTUAL request applied no Live Exposure.
        self.assertFalse(
            exposure["actual_ob_live_exposure_applied"]
        )
        self.assertEqual(
            exposure["actual_ob_live_exposed_count"], 0
        )
        self.assertFalse(
            artifact["ob_live_cue_mouth_active"]
        )

    def test_exact_overlap_does_nothing(self):
        # Frontend body text and an OB candidate content are IDENTICAL.
        shared = "SHARED-MEMORY-TEXT"

        body = _frontend_body(shared)

        snap = _snap(
            _frontend_units_from(body)
            + _ob_units(
                [_memory("m1", shared)]
            )
        )

        plan = _plan(snap, policy=None)

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=body,
                final=body,
            )

        artifact = report["artifact"]
        metrics = artifact["metrics"]

        # No dedup, no gap reduction, no identity transfer: the
        # Frontend gap and the OB unknown gap both remain.
        self.assertEqual(
            metrics["frontend_direct_bypass_count"], 1
        )
        self.assertEqual(
            metrics["ob_policy_unknown_count"], 1
        )
        self.assertEqual(
            metrics["contract_gap_unit_count"], 2
        )

        coverages = sorted(
            row["authority_coverage"]
            for row in artifact["rows"]
        )
        self.assertEqual(
            coverages,
            sorted(
                [COVERAGE_FRONTEND_GAP, COVERAGE_OB_UNKNOWN]
            ),
        )


class MetricAlgebraTests(unittest.TestCase):

    def _artifact(self):
        eligible = _memory("m1", "OB-A")

        body = _frontend_body("FRONTEND-BODY")

        snap = _snap(
            _frontend_units_from(body)
            + _ob_units(
                [
                    eligible,
                    _memory("m2", "OB-B"),
                ]
            )
        )

        plan = _plan(snap, policy=_policy(eligible=[eligible]))

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=body,
                final=body,
            )

        return report["artifact"]

    def test_full_scope_algebra_holds(self):
        metrics = self._artifact()["metrics"]

        self.assertEqual(
            metrics["unit_count"],
            metrics["contract_covered_unit_count"]
            + metrics["contract_gap_unit_count"],
        )
        self.assertEqual(
            metrics["contract_covered_unit_count"],
            metrics["ob_controlled_cue_contract_count"]
            + metrics[
                "ob_controlled_withhold_contract_count"
            ],
        )
        self.assertEqual(
            metrics["contract_gap_unit_count"],
            metrics["frontend_direct_bypass_count"]
            + metrics["ob_policy_unknown_count"],
        )

    def test_counterfactual_equals_current_frontend_footprint(
        self,
    ):
        metrics = self._artifact()["metrics"]

        self.assertEqual(
            metrics[
                "counterfactual_loss_if_direct_mouth_"
                "removed_without_replacement_chars"
            ],
            metrics["frontend_direct_model_visible_chars"],
        )
        self.assertEqual(
            metrics[
                "counterfactual_loss_if_direct_mouth_"
                "removed_without_replacement_estimated_tokens"
            ],
            metrics[
                "frontend_direct_model_visible"
                "_estimated_tokens"
            ],
        )
        self.assertEqual(
            metrics["frontend_replacement_required_count"],
            metrics["frontend_direct_bypass_count"],
        )

    def test_actual_model_visible_is_frontend_plus_live(self):
        pre_live = _frontend_body("MEM-A")
        final = _frontend_body_with_lead(
            "cue", "MEM-A"
        )

        snap = _snap(_frontend_units_from(final))

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=pre_live,
                final=final,
                receipt=_receipt(
                    pre_live, final, count=2, tokens=13
                ),
            )

        metrics = report["artifact"]["metrics"]

        self.assertEqual(
            metrics[
                "actual_model_visible_memory_estimated_tokens"
            ],
            metrics[
                "frontend_direct_model_visible"
                "_estimated_tokens"
            ]
            + 13,
        )
        self.assertNotEqual(
            metrics[
                "counterfactual_loss_if_direct_mouth_"
                "removed_without_replacement_estimated_tokens"
            ],
            13,
        )


class LiveExposureReceiptTests(unittest.TestCase):

    def _setup(self):
        pre_live = _frontend_body("MEM-A")
        final = _frontend_body_with_lead("cue", "MEM-A")
        snap = _snap(_frontend_units_from(final))
        return pre_live, final, snap

    def test_missing_receipt_is_no_record(self):
        pre_live, final, snap = self._setup()

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=pre_live,
                final=final,
                receipt=None,
            )

            self.assertFalse(report["stored"])
            self.assertEqual(
                report["reason"],
                "live_exposure_receipt_mismatch",
            )
            self.assertFalse(
                _migration_dir(root).exists()
            )

    def test_wrong_original_hash_is_no_record(self):
        pre_live, final, snap = self._setup()

        bad = _receipt(pre_live, final)
        bad["original_sha256"] = "0" * 64

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=pre_live,
                final=final,
                receipt=bad,
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "live_exposure_receipt_mismatch",
        )

    def test_wrong_selected_hash_is_no_record(self):
        pre_live, final, snap = self._setup()

        bad = _receipt(pre_live, final)
        bad["selected_sha256"] = "0" * 64

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=pre_live,
                final=final,
                receipt=bad,
            )

        self.assertFalse(report["stored"])

    def test_wrong_byte_delta_is_no_record(self):
        pre_live, final, snap = self._setup()

        bad = _receipt(pre_live, final)
        bad["byte_delta"] = bad["byte_delta"] + 1

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=pre_live,
                final=final,
                receipt=bad,
            )

        self.assertFalse(report["stored"])

    def test_not_applied_receipt_is_no_record(self):
        pre_live, final, snap = self._setup()

        bad = _receipt(pre_live, final)
        bad["applied"] = False

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=pre_live,
                final=final,
                receipt=bad,
            )

        self.assertFalse(report["stored"])


class BindingTests(unittest.TestCase):

    def test_unbound_evidence_is_refused(self):
        body = _frontend_body("MEM-A")

        foreign = _snap(
            _frontend_units_from(body),
            rid="ctxreq_" + "b" * 32,
        )

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=foreign,
                plan=_plan(foreign),
                pre_live=body,
                final=body,
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"], "evidence_not_bound"
        )

    def test_delivery_plan_not_bound_is_refused(self):
        body = _frontend_body("MEM-A")

        snap = _snap(_frontend_units_from(body))

        foreign_plan = _plan(snap, rid="ctxreq_" + "c" * 32)

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=foreign_plan,
                pre_live=body,
                final=body,
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"], "delivery_plan_not_bound"
        )

    def test_non_bytes_body_is_refused(self):
        body = _frontend_body("MEM-A")

        snap = _snap(_frontend_units_from(body))

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=None,
                final=body,
            )

        self.assertFalse(report["stored"])
        self.assertEqual(report["reason"], "invalid_body")

    def test_delivery_plan_bound_helper_rejects_live_authority(
        self,
    ):
        body = _frontend_body("MEM-A")

        snap = _snap(_frontend_units_from(body))

        plan = _plan(snap)

        self.assertTrue(
            delivery_plan_bound_to_request(
                plan,
                conversation_id=CID,
                cognitive_request_id=RID,
                source_unified_revision=REV,
            )
        )

        forged = dict(plan)
        forged["live_authority"] = True

        self.assertFalse(
            delivery_plan_bound_to_request(
                forged,
                conversation_id=CID,
                cognitive_request_id=RID,
                source_unified_revision=REV,
            )
        )


class ValidatorForgeryTests(unittest.TestCase):

    def _valid(self):
        body = _frontend_body("MEM-A")

        snap = _snap(_frontend_units_from(body))

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=body,
                final=body,
            )

        return report["artifact"]

    def test_valid_artifact_is_accepted(self):
        self.assertTrue(
            is_valid_memory_authority_migration_artifact(
                self._valid()
            )
        )

    def test_live_cutover_ready_true_is_rejected(self):
        artifact = self._valid()
        artifact["live_cutover_ready"] = True

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_forged_coverage_label_is_rejected(self):
        artifact = self._valid()
        # A Frontend row relabelled as a controlled OB cue.
        artifact["rows"][0]["authority_coverage"] = (
            COVERAGE_OB_CUE
        )

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_forged_replacement_requirement_is_rejected(
        self,
    ):
        artifact = self._valid()
        artifact["rows"][0][
            "replacement_requirement"
        ] = REQ_NONE

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_forged_gap_count_breaks_algebra(self):
        artifact = self._valid()
        artifact["metrics"]["contract_gap_unit_count"] = (
            99
        )

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_forged_live_cutover_reason_is_rejected(self):
        artifact = self._valid()
        artifact["live_cutover_reason"] = "nonsense"

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_foreign_version_is_rejected(self):
        artifact = self._valid()
        artifact["version"] = (
            "memory-authority-migration-shadow.v2"
        )

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_non_dict_is_rejected(self):
        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                None
            )
        )
        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                []
            )
        )


class PersistenceTests(unittest.TestCase):

    def _setup(self):
        body = _frontend_body("MEM-A")
        snap = _snap(_frontend_units_from(body))
        return snap, body, _plan(snap)

    def test_persists_and_is_deterministic_duplicate(self):
        snap, body, plan = self._setup()

        with tempfile.TemporaryDirectory() as root:
            first = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=body,
                final=body,
            )
            second = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=body,
                final=body,
            )

            path = _migration_dir(root) / (
                RID + ".json"
            )

            self.assertTrue(path.is_file())
            written = json.loads(
                path.read_text(encoding="utf-8")
            )

        self.assertTrue(first["stored"])
        self.assertTrue(second["stored"])
        self.assertFalse(written["live_authority"])
        self.assertFalse(written["live_cutover_ready"])

    def test_corrupt_existing_artifact_is_refused(self):
        snap, body, plan = self._setup()

        with tempfile.TemporaryDirectory() as root:
            path = _migration_dir(root) / (
                RID + ".json"
            )
            path.parent.mkdir(
                parents=True, exist_ok=True
            )
            path.write_text("{not json", encoding="utf-8")

            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=body,
                final=body,
            )

            # The corrupt file is never silently overwritten.
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "{not json",
            )

        self.assertFalse(report["stored"])

    def test_same_conversation_requests_do_not_mix(self):
        rid_a = "ctxreq_" + "4" * 32
        rid_b = "ctxreq_" + "5" * 32

        body_a = _frontend_body("FRONT-A")
        body_b = _frontend_body("FRONT-B")

        snap_a = _snap(
            _frontend_units_from(body_a), rid=rid_a
        )
        snap_b = _snap(
            _frontend_units_from(body_b), rid=rid_b
        )

        plan_a = _plan(snap_a, rid=rid_a)
        plan_b = _plan(snap_b, rid=rid_b)

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"OMBRE_CONTEXT_STATE_DIR": root},
                clear=False,
            ):
                def _do(rid, snap, plan, body):
                    return (
                        update_memory_authority_migration(
                            conversation_id=CID,
                            cognitive_request_id=rid,
                            source_evidence_snapshot=snap,
                            delivery_plan=plan,
                            source_unified_revision=REV,
                            pre_live_body=body,
                            final_body=body,
                        )
                    )

                with ThreadPoolExecutor(
                    max_workers=2
                ) as pool:
                    fa = pool.submit(
                        _do,
                        rid_a,
                        snap_a,
                        plan_a,
                        body_a,
                    )
                    fb = pool.submit(
                        _do,
                        rid_b,
                        snap_b,
                        plan_b,
                        body_b,
                    )
                    ra = fa.result()
                    rb = fb.result()

            artifacts = [
                json.loads(
                    path.read_text(encoding="utf-8")
                )
                for path in sorted(
                    _migration_dir(root).iterdir()
                )
            ]

        self.assertTrue(ra["stored"])
        self.assertTrue(rb["stored"])
        self.assertEqual(len(artifacts), 2)

        by_request = {
            artifact["cognitive_request_id"]: artifact
            for artifact in artifacts
        }

        self.assertEqual(
            by_request[rid_a]["source_unified_revision"],
            REV,
        )
        self.assertIn(rid_a, by_request)
        self.assertIn(rid_b, by_request)


class PrivacyTests(unittest.TestCase):

    def test_artifact_and_logs_carry_no_sentinels(self):
        body = _frontend_body(_SENTINEL_FRONTEND)

        snap = _snap(
            _frontend_units_from(body)
            + _ob_units(
                [
                    _memory(
                        _SENTINEL_MEMID,
                        _SENTINEL_OB,
                    )
                ]
            )
        )

        plan = _plan(snap, policy=None)

        logger = logging.getLogger(
            "ombrebrain.context."
            "memory_authority_migration_shadow"
        )

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=body,
                final=body,
            )

            with self.assertLogs(
                logger, level="INFO"
            ) as captured:
                migration.log_memory_authority_migration_report(
                    report
                )

        blob = json.dumps(report, sort_keys=True)
        logs = "\n".join(captured.output)

        # Raw content sentinels never appear anywhere.
        for sentinel in (
            _SENTINEL_FRONTEND,
            _SENTINEL_OB,
            _SENTINEL_MEMID,
        ):
            self.assertNotIn(sentinel, blob)
            self.assertNotIn(sentinel, logs)

        # Identifiers are bound into the request-scoped artifact but are
        # never allowed in telemetry.
        for identifier in (CID, RID):
            self.assertNotIn(identifier, logs)

        self.assertNotIn("ctx_", logs)
        self.assertNotIn("ctxreq_", logs)


class SourceAvailabilityTests(unittest.TestCase):
    """AVAILABLE + zero units stays valid EMPTY evidence; UNAVAILABLE
    never silently becomes EMPTY / COVERED, and the Delivery Plan must
    describe exactly the same source occurrence pool."""

    def test_available_zero_frontend_is_valid_empty(self):
        empty = _frontend_body()

        snap = _snap_avail([])

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=empty,
                final=empty,
            )

        self.assertTrue(report["stored"])

        artifact = report["artifact"]

        self.assertTrue(
            artifact[
                "frontend_final_observation_available"
            ]
        )
        self.assertTrue(
            artifact["frontend_final_presence_verified"]
        )
        self.assertEqual(
            artifact["metrics"]["unit_count"], 0
        )
        self.assertEqual(
            artifact["contract_coverage_status"],
            STATUS_COVERED,
        )

    def test_unavailable_frontend_is_no_record(self):
        snap = _snap_avail(
            [], frontend_available=False
        )

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=_frontend_body(),
                final=_frontend_body(),
            )

            self.assertFalse(
                _migration_dir(root).exists()
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            REASON_SOURCE_UNAVAILABLE,
        )

    def test_unavailable_ob_is_never_covered(self):
        snap = _snap_avail([], ob_available=False)

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=_frontend_body(),
                final=_frontend_body(),
            )

            self.assertFalse(
                _migration_dir(root).exists()
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            REASON_SOURCE_UNAVAILABLE,
        )
        # UNAVAILABLE must never become a covered readiness claim.
        self.assertNotEqual(
            report.get("contract_coverage_status"),
            STATUS_COVERED,
        )

    def test_snapshot_and_delivery_availability_disagree(
        self,
    ):
        empty = _frontend_body()

        snap = _snap_avail([])

        # Same request identity, but the Delivery Plan observed the OB
        # plane as UNAVAILABLE: the two availability views disagree.
        plan = _plan(
            _snap_avail([], ob_available=False)
        )

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=empty,
                final=empty,
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            REASON_SOURCE_UNAVAILABLE,
        )

    def test_source_pool_count_mismatch_is_refused(self):
        m1 = _memory("m1", "OB-A")
        m2 = _memory("m2", "OB-B")

        snap = _snap_avail(_ob_units([m1, m2]))

        # Two OB units in the snapshot, one in the Delivery Plan.
        plan = _plan(_snap_avail(_ob_units([m1])))

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=_frontend_body(),
                final=_frontend_body(),
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            REASON_EVIDENCE_MISMATCH,
        )

    def test_total_unit_count_mismatch_is_refused(self):
        body = _frontend_body("MEM-A")

        frontend_units = _frontend_units_from(body)

        m1 = _memory("m1", "OB-A")

        snap = _snap_avail(
            frontend_units + _ob_units([m1])
        )

        # The Frontend split agrees but the full total does not.
        plan = _plan(_snap_avail(frontend_units))

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=plan,
                pre_live=body,
                final=body,
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            REASON_EVIDENCE_MISMATCH,
        )


class ObservationOnceTests(unittest.TestCase):
    """Real request facts are observed EXACTLY ONCE; the builder is pure
    over the precomputed observations."""

    def _spy_run(
        self,
        root,
        *,
        snap,
        plan,
        pre_live,
        final,
        receipt=None,
    ):
        real_canonicalize = migration._canonicalize

        counter = {"canonicalize": 0}

        def spy_canonicalize(body):
            counter["canonicalize"] += 1
            return real_canonicalize(body)

        with patch.dict(
            os.environ,
            {"OMBRE_CONTEXT_STATE_DIR": str(root)},
            clear=False,
        ), patch.object(
            migration,
            "_canonicalize",
            side_effect=spy_canonicalize,
        ), patch.object(
            migration,
            "read_live_memory_exposure",
            return_value=receipt,
        ) as read_mock:
            report = update_memory_authority_migration(
                conversation_id=CID,
                cognitive_request_id=RID,
                source_evidence_snapshot=snap,
                delivery_plan=plan,
                source_unified_revision=REV,
                pre_live_body=pre_live,
                final_body=final,
            )

        return report, counter["canonicalize"], read_mock

    def test_final_canonicalized_once_and_no_receipt_when_unchanged(
        self,
    ):
        body = _frontend_body("MEM-A")

        snap = _snap(_frontend_units_from(body))

        with tempfile.TemporaryDirectory() as root:
            report, canonicalize, read_mock = (
                self._spy_run(
                    root,
                    snap=snap,
                    plan=_plan(snap),
                    pre_live=body,
                    final=body,
                )
            )

        self.assertTrue(report["stored"])
        self.assertEqual(canonicalize, 1)
        self.assertEqual(read_mock.call_count, 0)

    def test_receipt_read_once_when_body_changed(self):
        pre_live = _frontend_body("MEM-A")
        final = _frontend_body_with_lead(
            "cue", "MEM-A"
        )

        snap = _snap(_frontend_units_from(final))

        with tempfile.TemporaryDirectory() as root:
            report, canonicalize, read_mock = (
                self._spy_run(
                    root,
                    snap=snap,
                    plan=_plan(snap),
                    pre_live=pre_live,
                    final=final,
                    receipt=_receipt(pre_live, final),
                )
            )

        self.assertTrue(report["stored"])
        self.assertEqual(canonicalize, 1)
        self.assertEqual(read_mock.call_count, 1)

    def test_builder_is_pure_over_precomputed_observations(
        self,
    ):
        body = _frontend_body("MEM-A")

        snap = _snap(_frontend_units_from(body))

        plan = _plan(snap)

        frontend_observation = (
            observe_frontend_final_presence(body, snap)
        )

        live_observation = (
            migration.observe_actual_ob_live_exposure(
                conversation_id=CID,
                cognitive_request_id=RID,
                pre_live_body=body,
                final_body=body,
            )
        )

        live_ok, live_fact = live_observation

        self.assertTrue(live_ok)

        with patch.object(
            migration,
            "_canonicalize",
            side_effect=AssertionError(
                "builder canonicalized the body"
            ),
        ), patch.object(
            migration,
            "read_live_memory_exposure",
            side_effect=AssertionError(
                "builder read a receipt"
            ),
        ):
            artifact = (
                build_memory_authority_migration_artifact(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    source_unified_revision=REV,
                    delivery_plan=plan,
                    frontend_observation=(
                        frontend_observation
                    ),
                    live_exposure_observation=(
                        live_fact
                    ),
                    pre_live_body=body,
                    final_body=body,
                )
            )

        self.assertTrue(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )


class DuplicatedFactForgeryTests(unittest.TestCase):
    """A duplicated exposure fact that disagrees invalidates the WHOLE
    artifact."""

    def _valid(self):
        body = _frontend_body("MEM-A")

        snap = _snap(_frontend_units_from(body))

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=body,
                final=body,
            )

        return report["artifact"]

    def test_metric_live_exposed_count_disagreement_is_rejected(
        self,
    ):
        artifact = self._valid()

        artifact["metrics"][
            "actual_ob_live_exposed_count"
        ] += 1

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_current_exposure_live_count_disagreement_is_rejected(
        self,
    ):
        artifact = self._valid()

        artifact["current_exposure"][
            "actual_ob_live_exposed_count"
        ] += 1

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_metric_frontend_token_disagreement_is_rejected(
        self,
    ):
        artifact = self._valid()

        artifact["metrics"][
            "frontend_direct_model_visible"
            "_estimated_tokens"
        ] += 1

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_forged_dual_mouth_active_is_rejected(self):
        artifact = self._valid()

        self.assertFalse(
            artifact["ob_live_cue_mouth_active"]
        )

        artifact["dual_mouth_active"] = True

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )


class TruncationInvariantTests(unittest.TestCase):
    """Migration rows mirror ONLY persisted Delivery rows, so the full
    ``unit_count`` is EXACTLY persisted rows + source truncation."""

    def _artifact(self, root, count):
        units = _ob_units(
            [
                _memory(
                    "m%d" % index,
                    "OB-CONTENT-%d" % index,
                )
                for index in range(count)
            ]
        )

        snap = _snap(units)

        empty = _frontend_body()

        report = _run(
            root,
            snap=snap,
            plan=_plan(snap, policy=None),
            pre_live=empty,
            final=empty,
        )

        self.assertTrue(report["stored"])

        return report["artifact"]

    def test_exact_untruncated_identity(self):
        with tempfile.TemporaryDirectory() as root:
            artifact = self._artifact(root, 64)

        self.assertFalse(
            artifact["source_rows_truncated"]
        )
        self.assertEqual(
            artifact["source_rows_truncated_count"], 0
        )
        self.assertEqual(len(artifact["rows"]), 64)
        self.assertEqual(
            artifact["metrics"]["unit_count"], 64
        )
        self.assertTrue(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_exact_truncated_identity_64_70_6(self):
        with tempfile.TemporaryDirectory() as root:
            artifact = self._artifact(root, 70)

        self.assertTrue(
            artifact["source_rows_truncated"]
        )
        self.assertEqual(
            artifact["source_rows_truncated_count"], 6
        )
        self.assertEqual(len(artifact["rows"]), 64)
        self.assertEqual(
            artifact["metrics"]["unit_count"], 70
        )
        self.assertTrue(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_truncated_count_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            artifact = self._artifact(root, 70)

        artifact["source_rows_truncated_count"] = 5

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_claimed_total_beyond_truncation_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            artifact = self._artifact(root, 70)

        artifact["metrics"]["unit_count"] = 71

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_truncation_flag_contradiction_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            artifact = self._artifact(root, 70)

        artifact["source_rows_truncated"] = False

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )


class RowContractForgeryTests(unittest.TestCase):
    """The migration row is closed against the upstream Delivery
    contract; no field may be weakened after copying."""

    def _valid(self):
        body = _frontend_body("MEM-A")

        snap = _snap(_frontend_units_from(body))

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap),
                pre_live=body,
                final=body,
            )

        return report["artifact"]

    def _assert_rejected(self, mutate):
        artifact = self._valid()
        mutate(artifact["rows"][0])

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_wrong_kind_is_rejected(self):
        self._assert_rejected(
            lambda row: row.__setitem__(
                "kind", "ob_candidate"
            )
        )

    def test_wrong_full_access_mode_is_rejected(self):
        self._assert_rejected(
            lambda row: row.__setitem__(
                "current_full_content_access_mode",
                "none",
            )
        )

    def test_wrong_scope_is_rejected(self):
        self._assert_rejected(
            lambda row: row.__setitem__(
                "source_scope", "retrieval"
            )
        )

    def test_wrong_visibility_is_rejected(self):
        self._assert_rejected(
            lambda row: row.__setitem__(
                "currently_model_visible", False
            )
        )

    def test_wrong_delivery_mode_is_rejected(self):
        self._assert_rejected(
            lambda row: row.__setitem__(
                "current_delivery_mode", "drop"
            )
        )


class EvidenceRefForgeryTests(unittest.TestCase):
    """A valid-looking 64-hex string must not become trusted occurrence
    provenance: the ref is recomputed from the row's own fields."""

    def _two_row(self):
        body = _frontend_body("MEM-A")

        snap = _snap(
            _frontend_units_from(body)
            + _ob_units([_memory("m1", "OB-A")])
        )

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                snap=snap,
                plan=_plan(snap, policy=None),
                pre_live=body,
                final=body,
            )

        artifact = report["artifact"]

        self.assertEqual(len(artifact["rows"]), 2)

        return artifact

    def test_random_evidence_ref_is_rejected(self):
        artifact = self._two_row()

        artifact["rows"][0]["evidence_ref"] = "a" * 64

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_swapped_evidence_refs_are_rejected(self):
        artifact = self._two_row()

        self.assertNotEqual(
            artifact["rows"][0]["evidence_ref"],
            artifact["rows"][1]["evidence_ref"],
        )

        (
            artifact["rows"][0]["evidence_ref"],
            artifact["rows"][1]["evidence_ref"],
        ) = (
            artifact["rows"][1]["evidence_ref"],
            artifact["rows"][0]["evidence_ref"],
        )

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )

    def test_changed_content_digest_without_ref_is_rejected(
        self,
    ):
        artifact = self._two_row()

        artifact["rows"][0]["content_digest"] = "b" * 64

        self.assertFalse(
            is_valid_memory_authority_migration_artifact(
                artifact
            )
        )


class StaticSafetyTests(unittest.TestCase):

    _FORBIDDEN = {
        "select_context_injected_body",
        "select_live_memory_flash_body",
        "get_candidates",
        "embedding",
        "rerank",
        "touch",
        "touch_many",
        "archive",
        "delete",
        "restore",
        "reinforce",
        "record_usage",
    }

    def _called_names(self, path):
        tree = ast.parse(
            Path(path).read_text(encoding="utf-8")
        )

        names = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func = node.func

            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)

        return names

    def test_module_calls_no_forbidden_symbol(self):
        repo_root = Path(__file__).resolve().parents[1]

        for path in (
            migration.__file__,
            repo_root
            / "src"
            / "ombrebrain"
            / "maintenance"
            / "memory_authority_migration_report.py",
        ):
            called = self._called_names(path)

            self.assertEqual(
                called & self._FORBIDDEN,
                set(),
                "%s calls a forbidden symbol" % path,
            )


if __name__ == "__main__":
    unittest.main()
