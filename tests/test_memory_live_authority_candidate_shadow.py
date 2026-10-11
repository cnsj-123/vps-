from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ombrebrain.context import (
    memory_source_evidence as evidence,
)
from ombrebrain.context.frontend_controlled_direct_surface import (
    FrontendSurfaceResult,
    build_frontend_controlled_direct_surface,
    surface_bound_to_request,
)
from ombrebrain.context.memory_authority_migration_shadow import (
    build_memory_authority_migration_artifact,
    observe_frontend_final_presence,
)
from ombrebrain.context.memory_delivery_shadow import (
    build_memory_delivery_plan_artifact,
    evidence_ref_for_fields,
)
from ombrebrain.context.memory_live_authority_candidate_shadow import (
    CANDIDATE_COVERAGE_FRONTEND,
    CANDIDATE_COVERAGE_OB_CUE,
    CANDIDATE_COVERAGE_OB_UNKNOWN,
    CANDIDATE_COVERAGE_OB_WITHHOLD,
    CANDIDATE_HANDLING_FRONTEND,
    CANDIDATE_HANDLING_OB_CUE,
    CANDIDATE_HANDLING_OB_UNKNOWN,
    CANDIDATE_HANDLING_OB_WITHHOLD,
    MODE,
    OB_CONTROL_STRATEGY,
    REASON_CANDIDATE_INCOMPLETE,
    REASON_EXACT_ENVELOPE_UNAVAILABLE,
    REASON_NO_LIVE_SELECTOR,
    STATUS_BLOCKED,
    STATUS_COMPATIBILITY_COVERED,
    STRATEGY,
    VERSION,
    _final_body_control_walk,
    build_memory_live_authority_candidate_artifact,
    is_valid_memory_live_authority_candidate_artifact,
    log_memory_live_authority_candidate_report,
    read_memory_live_authority_candidate,
    update_memory_live_authority_candidate,
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
RID_OTHER = "ctxreq_" + "b" * 32
REV = 5

_FRONTEND_SECRET = "FRONTEND_CONTROL_SECRET"
_OB_SECRET = "OB_CONTENT_CONTROL_SECRET"
_MEMID_SECRET = "OB_MEMORY_ID_CONTROL_SECRET"
_CUE_SECRET = "LIVE_CUE_CONTROL_SECRET"

_LOG_NAME = (
    "ombrebrain.context.memory_live_authority_candidate_shadow"
)


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


def _frontend_body(*bodies, ids=None):
    if ids is None:
        ids = ["front_%d" % index for index in range(len(bodies))]

    text = "q " + " ".join(
        _attachment(
            "relevant_memories.txt", body, ids[index]
        )
        for index, body in enumerate(bodies)
    )

    return _body([{"role": "user", "content": text}])


def _frontend_units(body):
    payload = json.loads(body)
    request = OperitAdapter().adapt(payload)
    return frontend_evidence_units(
        extract_frontend_memory_segments(request)
    )


def _memory(memory_id, content):
    return {"id": memory_id, "content": content}


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


def _live_obs():
    return {
        "applied": False,
        "exposed_count": 0,
        "estimated_tokens": 0,
        "byte_delta": 0,
    }


def _migration(
    snap,
    plan,
    *,
    body,
    rid=RID,
    revision=REV,
    frontend_observation=None,
):
    if frontend_observation is None:
        frontend_observation = (
            observe_frontend_final_presence(body, snap)
        )

    return build_memory_authority_migration_artifact(
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        delivery_plan=plan,
        frontend_observation=frontend_observation,
        live_exposure_observation=_live_obs(),
        pre_live_body=body,
        final_body=body,
    )


def _surface(snap, plan, migration, *, rid=RID, revision=REV):
    return build_frontend_controlled_direct_surface(
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        source_evidence_snapshot=snap,
        delivery_plan=plan,
        migration=migration,
    )


def _stack(body, *, policy=None):
    snap = _snap(_frontend_units(body))
    plan = _plan(snap, policy=policy)
    migration = _migration(snap, plan, body=body)
    return snap, plan, migration


def _run(
    root,
    *,
    plan,
    migration,
    surface,
    final_body,
    rid=RID,
    revision=REV,
    cid=CID,
):
    with patch.dict(
        os.environ,
        {"OMBRE_CONTEXT_STATE_DIR": str(root)},
        clear=False,
    ):
        return update_memory_live_authority_candidate(
            conversation_id=cid,
            cognitive_request_id=rid,
            delivery_plan=plan,
            migration=migration,
            surface=surface,
            source_unified_revision=revision,
            final_body=final_body,
        )


def _read(root, *, cid=CID, rid=RID):
    with patch.dict(
        os.environ,
        {"OMBRE_CONTEXT_STATE_DIR": str(root)},
        clear=False,
    ):
        return read_memory_live_authority_candidate(
            conversation_id=cid,
            cognitive_request_id=rid,
        )


def _artifact_path(root, *, cid=CID, rid=RID):
    return (
        Path(root)
        / "memory_live_authority_candidate"
        / cid
        / (rid + ".json")
    )


def _observation(body, surface):
    ok, observation = _final_body_control_walk(
        body, surface.surface
    )

    assert ok, "control walk failed"

    return observation


def _build_artifact(units, body, *, policy=None):
    """Build one valid artifact directly from the pure builder.

    Exercises the REAL control walk so the privacy-safe per-row
    footprint observation is the exact one the production path uses.
    """

    snap = _snap(units)

    plan = _plan(snap, policy=policy)

    migration = _migration(snap, plan, body=body)

    surface = _surface(snap, plan, migration)

    assert surface.ok

    return build_memory_live_authority_candidate_artifact(
        conversation_id=CID,
        cognitive_request_id=RID,
        source_unified_revision=REV,
        delivery_plan=plan,
        migration=migration,
        control_observation=_observation(
            body, surface
        ),
        final_body=body,
    )


def _frontend_only_artifact():
    body = _frontend_body(_FRONTEND_SECRET)

    return _build_artifact(
        _frontend_units(body), body
    )


def _ob_unknown_artifact():
    body = _frontend_body(_FRONTEND_SECRET)

    units = (
        _frontend_units(body)
        + ob_evidence_units(
            [_memory(_MEMID_SECRET, _OB_SECRET)]
        )
    )

    return _build_artifact(units, body, policy=None)


def _clone(artifact):
    return json.loads(json.dumps(artifact))


class CleanFrontendCandidateTests(unittest.TestCase):

    def test_frontend_only_is_compatibility_covered(self):
        body = _frontend_body(_FRONTEND_SECRET)

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        self.assertTrue(surface.ok)

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

            path = _artifact_path(root)

            self.assertTrue(path.is_file())

        self.assertTrue(report["stored"])

        artifact = report["artifact"]

        self.assertTrue(
            is_valid_memory_live_authority_candidate_artifact(
                artifact
            )
        )

        self.assertEqual(artifact["version"], VERSION)
        self.assertEqual(artifact["mode"], MODE)
        self.assertEqual(artifact["strategy"], STRATEGY)
        self.assertEqual(
            artifact["ob_control_strategy"],
            OB_CONTROL_STRATEGY,
        )

        # v0 is never live authority and never applies the candidate.
        self.assertFalse(artifact["live_authority"])
        self.assertFalse(artifact["candidate_applied"])
        self.assertFalse(artifact["live_cutover_ready"])

        # Byte-identical candidate proof.
        self.assertTrue(
            artifact["candidate_body_byte_identical"]
        )
        self.assertEqual(
            artifact["final_body_sha256"],
            artifact["candidate_body_sha256"],
        )
        self.assertEqual(
            artifact["final_body_bytes"],
            artifact["candidate_body_bytes"],
        )
        self.assertEqual(
            artifact["final_body_sha256"],
            hashlib.sha256(body).hexdigest(),
        )
        self.assertEqual(
            artifact["final_body_bytes"], len(body)
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics[
                "frontend_controlled_direct_candidate_count"
            ],
            1,
        )
        self.assertEqual(
            metrics["adopted_frontend_gap_count"], 1
        )
        self.assertEqual(
            metrics["remaining_gap_unit_count"], 0
        )
        self.assertEqual(
            metrics["frontend_exact_envelope_verified_count"],
            1,
        )
        self.assertEqual(
            metrics["source_migration_gap_unit_count"], 1
        )

        self.assertTrue(
            artifact["compatibility_candidate_ready"]
        )
        self.assertEqual(
            artifact["candidate_coverage_status"],
            STATUS_COMPATIBILITY_COVERED,
        )
        self.assertEqual(
            artifact["live_cutover_reason"],
            REASON_NO_LIVE_SELECTOR,
        )

        self.assertEqual(len(artifact["rows"]), 1)

        row = artifact["rows"][0]

        self.assertEqual(
            row["candidate_coverage"],
            CANDIDATE_COVERAGE_FRONTEND,
        )
        self.assertEqual(
            row["candidate_handling"],
            CANDIDATE_HANDLING_FRONTEND,
        )

        # Privacy: no raw content leaves the artifact. Request-binding
        # identifiers (conversation / request id) are allowed in the
        # request-scoped artifact; raw memory content is not.
        blob = json.dumps(artifact)

        for secret in (
            _FRONTEND_SECRET,
            _OB_SECRET,
            _MEMID_SECRET,
            _CUE_SECRET,
        ):
            self.assertNotIn(secret, blob)

    def test_ob_unknown_stays_a_gap_and_blocks(self):
        body = _frontend_body(_FRONTEND_SECRET)

        snap = _snap(
            _frontend_units(body)
            + ob_evidence_units(
                [_memory(_MEMID_SECRET, _OB_SECRET)]
            )
        )

        # No base policy -> every OB unit is UNKNOWN, never a withhold.
        plan = _plan(snap, policy=None)

        migration = _migration(snap, plan, body=body)

        surface = _surface(snap, plan, migration)

        self.assertTrue(surface.ok)

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

        artifact = report["artifact"]

        self.assertTrue(
            is_valid_memory_live_authority_candidate_artifact(
                artifact
            )
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics["ob_policy_unknown_count"], 1
        )
        self.assertEqual(
            metrics["remaining_gap_unit_count"], 1
        )
        self.assertEqual(
            metrics["candidate_gap_unit_count"], 1
        )
        self.assertEqual(
            metrics["adopted_frontend_gap_count"], 1
        )

        self.assertFalse(
            artifact["compatibility_candidate_ready"]
        )
        self.assertEqual(
            artifact["candidate_coverage_status"],
            STATUS_BLOCKED,
        )
        self.assertEqual(
            artifact["live_cutover_reason"],
            REASON_CANDIDATE_INCOMPLETE,
        )

        coverages = [
            row["candidate_coverage"]
            for row in artifact["rows"]
        ]

        self.assertIn(
            CANDIDATE_COVERAGE_FRONTEND, coverages
        )
        self.assertIn(
            CANDIDATE_COVERAGE_OB_UNKNOWN, coverages
        )

        handlings = [
            row["candidate_handling"]
            for row in artifact["rows"]
        ]

        self.assertIn(
            CANDIDATE_HANDLING_FRONTEND, handlings
        )
        self.assertIn(
            CANDIDATE_HANDLING_OB_UNKNOWN, handlings
        )

    def test_ob_controlled_cue_and_withhold_are_delegated(
        self,
    ):
        body = _frontend_body(_FRONTEND_SECRET)

        eligible = _memory("m1", _OB_SECRET)

        snap = _snap(
            _frontend_units(body)
            + ob_evidence_units(
                [eligible, _memory("m2", "OB-B")]
            )
        )

        plan = _plan(
            snap, policy=_policy(eligible=[eligible])
        )

        migration = _migration(snap, plan, body=body)

        surface = _surface(snap, plan, migration)

        self.assertTrue(surface.ok)

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

        artifact = report["artifact"]

        self.assertTrue(
            is_valid_memory_live_authority_candidate_artifact(
                artifact
            )
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics["ob_existing_cue_contract_count"], 1
        )
        self.assertEqual(
            metrics["ob_existing_withhold_contract_count"],
            1,
        )
        self.assertEqual(
            metrics["ob_policy_unknown_count"], 0
        )
        self.assertEqual(
            metrics["remaining_gap_unit_count"], 0
        )
        self.assertTrue(
            artifact["compatibility_candidate_ready"]
        )

        pairs = {
            (
                row["candidate_coverage"],
                row["candidate_handling"],
            )
            for row in artifact["rows"]
        }

        self.assertIn(
            (
                CANDIDATE_COVERAGE_OB_CUE,
                CANDIDATE_HANDLING_OB_CUE,
            ),
            pairs,
        )
        self.assertIn(
            (
                CANDIDATE_COVERAGE_OB_WITHHOLD,
                CANDIDATE_HANDLING_OB_WITHHOLD,
            ),
            pairs,
        )


class DuplicateAttachmentTests(unittest.TestCase):

    def test_duplicate_attachments_stay_distinct(self):
        body = _frontend_body("SAME", "SAME")

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        self.assertTrue(surface.ok)
        self.assertEqual(len(surface.surface.units), 2)

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

        artifact = report["artifact"]

        self.assertEqual(
            artifact["metrics"][
                "frontend_controlled_direct_candidate_count"
            ],
            2,
        )
        self.assertEqual(
            artifact["metrics"][
                "frontend_exact_envelope_verified_count"
            ],
            2,
        )

        refs = [
            row["evidence_ref"]
            for row in artifact["rows"]
        ]

        self.assertEqual(len(refs), 2)
        self.assertEqual(len(set(refs)), 2)


class ExactEnvelopeBindingTests(unittest.TestCase):

    def test_same_body_different_metadata_is_refused(self):
        body = _frontend_body("X")

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        self.assertTrue(surface.ok)

        # Same body, different attachment metadata, identical char
        # footprint: the weaker migration-style observation would not
        # notice, but the exact-envelope binding must.
        mismatch = _frontend_body("X", ids=["1234567"])

        self.assertEqual(len(mismatch), len(body))
        self.assertNotEqual(mismatch, body)

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=mismatch,
            )

            self.assertFalse(
                _artifact_path(root).is_file()
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            REASON_EXACT_ENVELOPE_UNAVAILABLE,
        )


class SurfaceRefusalTests(unittest.TestCase):

    def test_unbuilt_surface_is_refused(self):
        body = _frontend_body("X")

        snap, plan, migration = _stack(body)

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                plan=plan,
                migration=migration,
                surface=None,
                final_body=body,
            )

        self.assertFalse(report["stored"])
        self.assertEqual(report["reason"], "surface_not_built")

    def test_refused_surface_reason_is_propagated(self):
        body = _frontend_body("X")

        snap, plan, migration = _stack(body)

        refused = FrontendSurfaceResult(
            ok=False,
            reason="frontend_control_rows_incomplete",
            surface=None,
        )

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                plan=plan,
                migration=migration,
                surface=refused,
                final_body=body,
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"],
            "frontend_control_rows_incomplete",
        )

    def test_foreign_surface_is_refused(self):
        body = _frontend_body("X")

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        self.assertTrue(surface.ok)

        foreign_surface = dataclasses.replace(
            surface.surface,
            cognitive_request_id=RID_OTHER,
        )

        foreign = FrontendSurfaceResult(
            ok=True,
            reason=None,
            surface=foreign_surface,
        )

        self.assertFalse(
            surface_bound_to_request(
                foreign.surface,
                conversation_id=CID,
                cognitive_request_id=RID,
                source_unified_revision=REV,
            )
        )

        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                plan=plan,
                migration=migration,
                surface=foreign,
                final_body=body,
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"], "surface_not_bound"
        )


class PersistenceTests(unittest.TestCase):

    def test_identical_rerun_is_idempotent(self):
        body = _frontend_body("X")

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        with tempfile.TemporaryDirectory() as root:
            first = _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

            second = _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

            stored = _read(root)

        self.assertTrue(first["stored"])
        self.assertTrue(second["stored"])
        self.assertEqual(
            stored["final_body_sha256"],
            hashlib.sha256(body).hexdigest(),
        )

    def test_conflicting_rerun_is_refused(self):
        first_body = _frontend_body("X")

        snap, plan, migration = _stack(first_body)

        surface = _surface(snap, plan, migration)

        second_body = _frontend_body("Y")

        snap2, plan2, migration2 = _stack(second_body)

        surface2 = _surface(snap2, plan2, migration2)

        with tempfile.TemporaryDirectory() as root:
            first = _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=first_body,
            )

            second = _run(
                root,
                plan=plan2,
                migration=migration2,
                surface=surface2,
                final_body=second_body,
            )

        self.assertTrue(first["stored"])
        self.assertFalse(second["stored"])
        self.assertEqual(
            second["reason"],
            "existing_artifact_mismatch",
        )


class BuilderAndValidatorTests(unittest.TestCase):

    def _valid_artifact(self):
        body = _frontend_body("X")

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        # The real observation path is exercised against an ISOLATED
        # temp state root so this helper never depends on the process
        # default state dir being writable (CI runs unprivileged).
        with tempfile.TemporaryDirectory() as root:
            report = _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

        self.assertTrue(
            report["stored"], report.get("reason")
        )

        # The pure builder path (no persistence) is validated directly,
        # using the REAL privacy-safe control observation.
        return build_memory_live_authority_candidate_artifact(
            conversation_id=CID,
            cognitive_request_id=RID,
            source_unified_revision=REV,
            delivery_plan=plan,
            migration=migration,
            control_observation=_observation(
                body, surface
            ),
            final_body=body,
        )

    def test_pure_builder_matches_update(self):
        built = self._valid_artifact()

        self.assertTrue(
            is_valid_memory_live_authority_candidate_artifact(
                built
            )
        )
        self.assertEqual(built["version"], VERSION)

    def test_adversarial_mutations_are_rejected(self):
        base = self._valid_artifact()

        def mutate(**changes):
            clone = json.loads(json.dumps(base))

            for key, value in changes.items():
                clone[key] = value

            return clone

        # 1. live authority
        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                mutate(live_authority=True)
            )
        )

        # 2. candidate applied
        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                mutate(candidate_applied=True)
            )
        )

        # 3. live cutover ready
        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                mutate(live_cutover_ready=True)
            )
        )

        # 9. byte identity flag
        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                mutate(candidate_body_byte_identical=False)
            )
        )

        # 10. hashes disagree
        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                mutate(
                    candidate_body_sha256="0" * 64
                )
            )
        )

        # 11. hashes equal but byte counts disagree
        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                mutate(
                    candidate_body_bytes=(
                        base["final_body_bytes"] + 1
                    )
                )
            )
        )

        # 15. unknown strategy
        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                mutate(strategy="something_else.v1")
            )
        )

        # unknown ob control strategy
        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                mutate(ob_control_strategy="own_ob_rendering")
            )
        )

    def test_frontend_row_label_forgery_is_rejected(self):
        base = self._valid_artifact()

        clone = json.loads(json.dumps(base))

        clone["rows"][0][
            "candidate_handling"
        ] = CANDIDATE_HANDLING_OB_CUE

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

        clone = json.loads(json.dumps(base))

        clone["rows"][0][
            "candidate_coverage"
        ] = CANDIDATE_COVERAGE_OB_UNKNOWN

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

    def test_forged_evidence_ref_is_rejected(self):
        base = self._valid_artifact()

        clone = json.loads(json.dumps(base))

        clone["rows"][0]["evidence_ref"] = "a" * 64

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

    def test_broken_metric_algebra_is_rejected(self):
        base = self._valid_artifact()

        # candidate_gap falsified
        clone = json.loads(json.dumps(base))

        clone["metrics"]["candidate_gap_unit_count"] += 1

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

        # remaining_gap falsified
        clone = json.loads(json.dumps(base))

        clone["metrics"]["remaining_gap_unit_count"] += 1

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

    def test_ready_true_with_gaps_is_rejected(self):
        body = _frontend_body(_FRONTEND_SECRET)

        snap = _snap(
            _frontend_units(body)
            + ob_evidence_units([_memory("m1", "OB")])
        )

        plan = _plan(snap, policy=None)

        migration = _migration(snap, plan, body=body)

        surface = _surface(snap, plan, migration)

        artifact = (
            build_memory_live_authority_candidate_artifact(
                conversation_id=CID,
                cognitive_request_id=RID,
                source_unified_revision=REV,
                delivery_plan=plan,
                migration=migration,
                control_observation=_observation(
                    body, surface
                ),
                final_body=body,
            )
        )

        self.assertFalse(
            artifact["compatibility_candidate_ready"]
        )

        # Force a ready claim that the remaining OB gap contradicts.
        artifact["compatibility_candidate_ready"] = True
        artifact["candidate_coverage_status"] = (
            STATUS_COMPATIBILITY_COVERED
        )
        artifact["live_cutover_reason"] = (
            REASON_NO_LIVE_SELECTOR
        )

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                artifact
            )
        )


class PrivacyAndReportTests(unittest.TestCase):

    def _report(self):
        body = _frontend_body(_FRONTEND_SECRET)

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        with tempfile.TemporaryDirectory() as root:
            return _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

    def test_report_never_carries_a_body(self):
        report = self._report()

        self.assertNotIn("body", report)
        self.assertNotIn("candidate_body", report)
        self.assertNotIn("final_body", report)

    def test_log_never_carries_private_fields(self):
        report = self._report()

        with self.assertLogs(
            _LOG_NAME, level="INFO"
        ) as captured:
            log_memory_live_authority_candidate_report(
                report
            )

        joined = "\n".join(captured.output)

        for secret in (
            _FRONTEND_SECRET,
            _OB_SECRET,
            _MEMID_SECRET,
            _CUE_SECRET,
        ):
            self.assertNotIn(secret, joined)

        self.assertNotIn(CID, joined)
        self.assertNotIn(RID, joined)
        self.assertNotIn(
            report["artifact"]["final_body_sha256"],
            joined,
        )

    def test_evidence_ref_matches_recomputation(self):
        report = self._report()

        row = report["artifact"]["rows"][0]

        self.assertEqual(
            row["evidence_ref"],
            evidence_ref_for_fields(
                row["source"],
                row["kind"],
                row["source_occurrence"],
                row["content_digest"],
            ),
        )


class ClosedSchemaForgeryTests(unittest.TestCase):
    """The candidate artifact and every row use CLOSED key schemas."""

    def test_frontend_control_action_forgery_is_rejected(self):
        base = _frontend_only_artifact()

        for label in ("drop", "flash_cue", "summarize"):
            with self.subTest(label=label):
                clone = _clone(base)
                clone["frontend_control_action"] = label

                self.assertFalse(
                    is_valid_memory_live_authority_candidate_artifact(
                        clone
                    )
                )

        clone = _clone(base)
        del clone["frontend_control_action"]

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

    def test_unknown_top_level_field_is_rejected(self):
        base = _frontend_only_artifact()

        for field, value in (
            ("raw_attachment", _FRONTEND_SECRET),
            ("candidate_body", "PRIVATE BODY"),
            ("memref", "memref_" + "0" * 32),
            ("future_field", 1),
        ):
            with self.subTest(field=field):
                clone = _clone(base)
                clone[field] = value

                self.assertFalse(
                    is_valid_memory_live_authority_candidate_artifact(
                        clone
                    )
                )

    def test_unknown_row_field_is_rejected(self):
        base = _frontend_only_artifact()

        for field, value in (
            ("cue", _CUE_SECRET),
            ("memory_id", _MEMID_SECRET),
            ("raw_attachment", _FRONTEND_SECRET),
            ("candidate_fingerprint", "abc"),
        ):
            with self.subTest(field=field):
                clone = _clone(base)
                clone["rows"][0][field] = value

                self.assertFalse(
                    is_valid_memory_live_authority_candidate_artifact(
                        clone
                    )
                )


class ReplacementRequirementForgeryTests(
    unittest.TestCase
):

    def test_frontend_requirement_none_is_rejected(self):
        base = _frontend_only_artifact()

        clone = _clone(base)
        clone["rows"][0][
            "replacement_requirement"
        ] = "none"

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

    def test_ob_unknown_requirement_none_is_rejected(self):
        base = _ob_unknown_artifact()

        clone = _clone(base)

        row = next(
            row
            for row in clone["rows"]
            if row["candidate_coverage"]
            == CANDIDATE_COVERAGE_OB_UNKNOWN
        )

        row["replacement_requirement"] = "none"

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

    def test_ob_cue_requirement_policy_resolution_is_rejected(
        self,
    ):
        body = _frontend_body(_FRONTEND_SECRET)

        eligible = _memory("m1", _OB_SECRET)

        units = (
            _frontend_units(body)
            + ob_evidence_units([eligible])
        )

        base = _build_artifact(
            units,
            body,
            policy=_policy(eligible=[eligible]),
        )

        clone = _clone(base)

        row = next(
            row
            for row in clone["rows"]
            if row["candidate_coverage"]
            == CANDIDATE_COVERAGE_OB_CUE
        )

        row["replacement_requirement"] = (
            "policy_resolution_required"
        )

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )


def _rebalance_frontend_duplicate(clone):
    """Rebalance every full-scope alias after duplicating one
    Frontend row, so a rejection can ONLY come from duplicate
    occurrence / evidence provenance.
    """

    metrics = clone["metrics"]

    frontend_records = [
        row
        for row in clone["rows"]
        if row["source"] == "frontend"
    ]

    frontend_count = len(frontend_records)

    metrics["unit_count"] = len(clone["rows"])
    metrics[
        "frontend_controlled_direct_candidate_count"
    ] = frontend_count
    metrics["candidate_covered_unit_count"] = (
        frontend_count
    )
    metrics["candidate_gap_unit_count"] = 0
    metrics["remaining_gap_unit_count"] = 0
    metrics["ob_policy_unknown_count"] = 0
    metrics["ob_existing_cue_contract_count"] = 0
    metrics[
        "ob_existing_withhold_contract_count"
    ] = 0
    metrics["adopted_frontend_gap_count"] = (
        frontend_count
    )
    metrics["frontend_exact_envelope_verified_count"] = (
        frontend_count
    )
    metrics["source_migration_gap_unit_count"] = (
        frontend_count
    )
    metrics[
        "frontend_controlled_direct_model_visible_chars"
    ] = sum(
        row["model_visible_chars"]
        for row in frontend_records
    )
    metrics[
        "frontend_controlled_direct_model_visible"
        "_estimated_tokens"
    ] = sum(
        row["model_visible_estimated_tokens"]
        for row in frontend_records
    )


class OccurrenceUniquenessTests(unittest.TestCase):

    def test_duplicated_row_with_balanced_algebra_is_rejected(
        self,
    ):
        # A base artifact is ready and balances everywhere.
        base = _frontend_only_artifact()

        self.assertTrue(
            is_valid_memory_live_authority_candidate_artifact(
                base
            )
        )

        clone = _clone(base)

        duplicate = _clone(clone["rows"][0])

        clone["rows"].append(duplicate)

        _rebalance_frontend_duplicate(clone)

        # The rejection comes from duplicate occurrence / evidence
        # provenance, NOT from unbalanced arithmetic.
        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

    def test_duplicate_evidence_ref_changed_fields_is_rejected(
        self,
    ):
        base = _frontend_only_artifact()

        clone = _clone(base)

        duplicate = _clone(clone["rows"][0])

        # Same evidence_ref / occurrence, different footprint: the ref
        # still recomputes, so only uniqueness can reject it.
        duplicate["model_visible_chars"] += 1
        duplicate[
            "model_visible_estimated_tokens"
        ] += 1

        clone["rows"].append(duplicate)

        _rebalance_frontend_duplicate(clone)

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )


class FootprintForgeryTests(unittest.TestCase):

    def test_aggregate_token_metric_only_is_rejected(self):
        base = _frontend_only_artifact()

        clone = _clone(base)

        clone["metrics"][
            "frontend_controlled_direct_model_visible"
            "_estimated_tokens"
        ] += 1

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

    def test_aggregate_char_metric_only_is_rejected(self):
        base = _frontend_only_artifact()

        clone = _clone(base)

        clone["metrics"][
            "frontend_controlled_direct_model_visible_chars"
        ] += 1

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

    def test_row_token_without_aggregate_is_rejected(self):
        base = _frontend_only_artifact()

        clone = _clone(base)

        clone["rows"][0][
            "model_visible_estimated_tokens"
        ] += 1

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )

    def test_ob_row_nonzero_footprint_is_rejected(self):
        base = _ob_unknown_artifact()

        clone = _clone(base)

        row = next(
            row
            for row in clone["rows"]
            if row["candidate_coverage"]
            == CANDIDATE_COVERAGE_OB_UNKNOWN
        )

        row["model_visible_chars"] = 5
        row["model_visible_estimated_tokens"] = 5

        self.assertFalse(
            is_valid_memory_live_authority_candidate_artifact(
                clone
            )
        )


class ReadHelperTests(unittest.TestCase):

    def test_malformed_file_returns_none(self):
        body = _frontend_body(_FRONTEND_SECRET)

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        with tempfile.TemporaryDirectory() as root:
            _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

            _artifact_path(root).write_text(
                "{not json", encoding="utf-8"
            )

            self.assertIsNone(_read(root))

    def test_valid_artifact_is_returned(self):
        body = _frontend_body(_FRONTEND_SECRET)

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        with tempfile.TemporaryDirectory() as root:
            _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

            artifact = _read(root)

        self.assertIsNotNone(artifact)
        self.assertEqual(
            artifact["cognitive_request_id"], RID
        )

    def test_wrong_request_path_returns_none(self):
        body = _frontend_body(_FRONTEND_SECRET)

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        with tempfile.TemporaryDirectory() as root:
            _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

            # Copy the valid artifact under a DIFFERENT request path.
            target = _artifact_path(root, rid=RID_OTHER)

            target.write_text(
                _artifact_path(root).read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )

            self.assertIsNone(
                _read(root, rid=RID_OTHER)
            )

    def test_wrong_conversation_path_returns_none(self):
        body = _frontend_body(_FRONTEND_SECRET)

        snap, plan, migration = _stack(body)

        surface = _surface(snap, plan, migration)

        other_cid = "ctx_0123456789abcdee"

        with tempfile.TemporaryDirectory() as root:
            _run(
                root,
                plan=plan,
                migration=migration,
                surface=surface,
                final_body=body,
            )

            # Copy the valid artifact under a DIFFERENT conversation
            # directory.
            target = _artifact_path(root, cid=other_cid)

            target.parent.mkdir(parents=True, exist_ok=True)

            target.write_text(
                _artifact_path(root).read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )

            self.assertIsNone(
                _read(root, cid=other_cid)
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    unittest.main()