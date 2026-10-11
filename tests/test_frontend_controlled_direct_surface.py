from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from ombrebrain.context import (
    memory_delivery_shadow as delivery,
)
from ombrebrain.context import (
    memory_source_evidence as evidence,
)
from ombrebrain.context.frontend_controlled_direct_surface import (
    ACTION_PRESERVE_DIRECT_DATA,
    REASON_DELIVERY_NOT_BOUND,
    REASON_ENVELOPE_UNAVAILABLE,
    REASON_EVIDENCE_NOT_BOUND,
    REASON_INVALID_REVISION,
    REASON_MIGRATION_NOT_BOUND,
    REASON_ROWS_INCOMPLETE,
    REASON_ROW_BINDING_UNAVAILABLE,
    REASON_SOURCE_UNAVAILABLE,
    VERSION,
    build_frontend_controlled_direct_surface,
    migration_bound_to_request,
    surface_bound_to_request,
)
from ombrebrain.context.memory_authority_migration_shadow import (
    build_memory_authority_migration_artifact,
)
from ombrebrain.context.memory_delivery_shadow import (
    build_memory_delivery_plan_artifact,
)
from ombrebrain.context.memory_source_evidence import (
    MemorySourceEvidenceSnapshot,
    MemorySourceEvidenceUnit,
    frontend_evidence_units,
)
from ombrebrain.gateway import (
    OperitAdapter,
    extract_frontend_memory_segments,
)


CID = "ctx_0123456789abcdef"
RID = "ctxreq_" + "a" * 32
REV = 5

_SENTINEL = "FRONTEND_CONTROL_SECRET"


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


def _frontend_body(*bodies):
    text = "q " + " ".join(
        _attachment(
            "relevant_memories.txt",
            body,
            "front_%d" % index,
        )
        for index, body in enumerate(bodies)
    )

    return json.dumps(
        {"model": "example", "messages": [
            {"role": "user", "content": text}
        ]}
    ).encode("utf-8")


def _frontend_units_from(body):
    payload = json.loads(body)
    request = OperitAdapter().adapt(payload)
    return frontend_evidence_units(
        extract_frontend_memory_segments(request)
    )


def _snap(units, *, rid=RID, revision=REV, frontend=True, ob=True):
    return MemorySourceEvidenceSnapshot(
        version=evidence._VERSION,
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        frontend_observation_available=frontend,
        ob_observation_available=ob,
        units=tuple(units),
    )


def _plan(snap, *, rid=RID, revision=REV):
    return build_memory_delivery_plan_artifact(
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        source_evidence_snapshot=snap,
        base_policy=None,
    )


def _migration(plan, body, *, rid=RID, revision=REV):
    return build_memory_authority_migration_artifact(
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        delivery_plan=plan,
        frontend_observation=(True, True),
        live_exposure_observation={
            "applied": False,
            "exposed_count": 0,
            "estimated_tokens": 0,
            "byte_delta": 0,
        },
        pre_live_body=body,
        final_body=body,
    )


def _triple(*bodies, rid=RID, revision=REV):
    body = _frontend_body(*bodies)
    snap = _snap(
        _frontend_units_from(body),
        rid=rid,
        revision=revision,
    )
    plan = _plan(snap, rid=rid, revision=revision)
    migration = _migration(
        plan, body, rid=rid, revision=revision
    )
    return snap, plan, migration, body


def _build(snap, plan, migration, *, cid=CID, rid=RID, revision=REV):
    return build_frontend_controlled_direct_surface(
        conversation_id=cid,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        source_evidence_snapshot=snap,
        delivery_plan=plan,
        migration=migration,
    )


class FrontendControlledDirectSurfaceTests(
    unittest.TestCase,
):

    def test_valid_surface_adopts_direct_data(self) -> None:
        snap, plan, migration, _ = _triple(
            _SENTINEL + "_one"
        )

        result = _build(snap, plan, migration)

        self.assertTrue(result.ok)
        self.assertIsNone(result.reason)

        surface = result.surface

        self.assertIsNotNone(surface)
        self.assertEqual(surface.version, VERSION)
        self.assertEqual(len(surface.units), 1)

        unit = surface.units[0]

        self.assertEqual(
            unit.control_action,
            ACTION_PRESERVE_DIRECT_DATA,
        )
        self.assertEqual(unit.source_scope, "current")
        self.assertEqual(
            unit.native_raw_attachment,
            _attachment(
                "relevant_memories.txt",
                _SENTINEL + "_one",
                "front_0",
            ),
        )
        self.assertEqual(
            unit.evidence_ref,
            delivery.evidence_ref_for_fields(
                "frontend",
                "frontend_attachment",
                unit.source_occurrence,
                unit.content_digest,
            ),
        )

    def test_duplicate_identical_attachments_stay_distinct(
        self,
    ) -> None:
        snap, plan, migration, _ = _triple(
            "SAME", "SAME"
        )

        result = _build(snap, plan, migration)

        self.assertTrue(result.ok)
        self.assertEqual(len(result.surface.units), 2)

        refs = {
            unit.evidence_ref
            for unit in result.surface.units
        }

        self.assertEqual(len(refs), 2)

    def test_native_envelope_is_not_repr_visible(self) -> None:
        snap, plan, migration, _ = _triple(
            _SENTINEL + "_repr"
        )

        unit = _build(
            snap, plan, migration
        ).surface.units[0]

        self.assertNotIn(
            _SENTINEL, repr(unit)
        )

    def test_unbound_evidence_is_refused(self) -> None:
        snap, plan, migration, _ = _triple("x")

        result = _build(
            snap,
            plan,
            migration,
            rid="ctxreq_" + "b" * 32,
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.surface)

    def test_unavailable_ob_plane_is_refused(self) -> None:
        body = _frontend_body("x")
        units = _frontend_units_from(body)

        snap = _snap(units, ob=False)
        plan = _plan(_snap(units, frontend=True))
        migration = _migration(plan, body)

        result = _build(snap, plan, migration)

        self.assertFalse(result.ok)
        self.assertEqual(
            result.reason, REASON_SOURCE_UNAVAILABLE
        )
        self.assertIsNone(result.surface)

    def test_unbound_delivery_is_refused(self) -> None:
        snap, plan, migration, _ = _triple("x")

        tampered_plan = dict(plan)
        tampered_plan["source_unified_revision"] = REV + 1

        result = _build(snap, tampered_plan, migration)

        self.assertFalse(result.ok)
        self.assertEqual(
            result.reason, REASON_DELIVERY_NOT_BOUND
        )
        self.assertIsNone(result.surface)

    def test_unbound_migration_at_build_is_refused(self) -> None:
        snap, plan, migration, _ = _triple("x")

        tampered = dict(migration)
        tampered["source_unified_revision"] = REV + 1

        result = _build(snap, plan, tampered)

        self.assertFalse(result.ok)
        self.assertEqual(
            result.reason, REASON_MIGRATION_NOT_BOUND
        )
        self.assertIsNone(result.surface)

    def test_unbound_migration_is_refused(self) -> None:
        snap, plan, migration, _ = _triple("x")

        self.assertTrue(
            migration_bound_to_request(
                migration,
                conversation_id=CID,
                cognitive_request_id=RID,
                source_unified_revision=REV,
            )
        )

        tampered = dict(migration)
        tampered["live_cutover_ready"] = True

        self.assertFalse(
            migration_bound_to_request(
                tampered,
                conversation_id=CID,
                cognitive_request_id=RID,
                source_unified_revision=REV,
            )
        )

    def test_invalid_revision_is_refused(self) -> None:
        snap, plan, migration, _ = _triple("x")

        result = build_frontend_controlled_direct_surface(
            conversation_id=CID,
            cognitive_request_id=RID,
            source_unified_revision=0,
            source_evidence_snapshot=snap,
            delivery_plan=plan,
            migration=migration,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.reason, REASON_INVALID_REVISION
        )

    def test_truncated_frontend_rows_are_refused(self) -> None:
        # Force the persisted Delivery cap to 1 so a two-Frontend
        # request loses one Frontend row to truncation. The Candidate
        # must never authorize a hidden Frontend occurrence from an
        # aggregate count.
        with patch.object(
            delivery, "MAX_DELIVERY_ROWS", 1
        ):
            snap, plan, migration, _ = _triple(
                "a", "b"
            )

        self.assertTrue(plan["rows_truncated"])
        self.assertEqual(
            plan["metrics"]["frontend_unit_count"], 2
        )
        self.assertEqual(len(plan["rows"]), 1)

        result = _build(snap, plan, migration)

        self.assertFalse(result.ok)
        self.assertIsNone(result.surface)
        self.assertEqual(
            result.reason, REASON_ROWS_INCOMPLETE
        )

    def test_cross_request_row_binding_is_refused(self) -> None:
        snap_a, plan, migration, _ = _triple("aaa")

        body_b = _frontend_body("bbb")
        snap_b = _snap(_frontend_units_from(body_b))

        result = _build(snap_b, plan, migration)

        self.assertFalse(result.ok)
        self.assertIsNone(result.surface)
        self.assertEqual(
            result.reason,
            REASON_ROW_BINDING_UNAVAILABLE,
        )

    def test_missing_native_envelope_is_refused(self) -> None:
        body = _frontend_body("x")

        # A hand-made unit with no native payload: the plan still
        # validates (it only reads control fields), but the surface
        # cannot prove an exact controllable envelope.
        unit = MemorySourceEvidenceUnit(
            source="frontend",
            kind="frontend_attachment",
            granularity="attachment",
            source_index=0,
            source_scope="current",
            content_digest="b" * 64,
            normalized_digest="a" * 64,
            content_chars=1,
            content_estimated_tokens=1,
            model_visible_chars=5,
            model_visible_estimated_tokens=2,
            currently_model_visible=True,
            retrieved=False,
            request_occurrence=0,
            native_payload=None,
        )

        snap = _snap([unit])
        plan = _plan(snap)
        migration = _migration(plan, body)

        result = _build(snap, plan, migration)

        self.assertFalse(result.ok)
        self.assertIsNone(result.surface)
        self.assertEqual(
            result.reason, REASON_ENVELOPE_UNAVAILABLE
        )

    def test_surface_bound_helper(self) -> None:
        snap, plan, migration, _ = _triple("x")

        surface = _build(
            snap, plan, migration
        ).surface

        self.assertTrue(
            surface_bound_to_request(
                surface,
                conversation_id=CID,
                cognitive_request_id=RID,
                source_unified_revision=REV,
            )
        )
        self.assertFalse(
            surface_bound_to_request(
                surface,
                conversation_id=CID,
                cognitive_request_id=RID,
                source_unified_revision=REV + 1,
            )
        )

    def test_builder_requires_a_real_source_evidence_object(
        self,
    ) -> None:
        # An arbitrary dict is never accepted as THIS request's
        # evidence.
        result = build_frontend_controlled_direct_surface(
            conversation_id=CID,
            cognitive_request_id=RID,
            source_unified_revision=REV,
            source_evidence_snapshot={},
            delivery_plan={},
            migration={},
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.reason, REASON_EVIDENCE_NOT_BOUND
        )


if __name__ == "__main__":
    unittest.main()