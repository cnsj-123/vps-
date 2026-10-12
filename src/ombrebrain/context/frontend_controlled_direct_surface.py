from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ombrebrain.context.memory_authority_migration_shadow import (
    COVERAGE_FRONTEND_GAP,
    REQ_FRONTEND_CHANNEL,
    delivery_plan_bound_to_request,
    is_valid_memory_authority_migration_artifact,
)
from ombrebrain.context.memory_delivery_shadow import (
    evidence_ref_for_fields,
)
from ombrebrain.context.memory_source_evidence import (
    KIND_FRONTEND_ATTACHMENT,
    SOURCE_FRONTEND,
    is_bound_to_request,
)
from ombrebrain.context.validators.freshness import (
    is_valid_revision,
)
from ombrebrain.context.recall_types import (
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
)


# Frontend Controlled Direct Surface v0 -- IN-MEMORY ONLY.
#
# The first controlled representation of the CURRENT Frontend Memory
# mouth that does NOT require converting Frontend to Flash:
#
#     one current Frontend attachment occurrence
#         -> one controller-owned compatibility action
#             -> preserve_direct_data
#
# Contract: ``frontend-controlled-direct-surface.v1``.
#
# The surface answers ONE question, and only that question:
#
#     "for THIS request, which exact Frontend attachment occurrences
#      already exist as model-visible direct data, and can each of them
#      be EXPLICITLY adopted by unified control WITHOUT changing a
#      single byte the model sees?"
#
# It is deliberately NOT a memory database, NOT a universal memory
# schema and NOT a Frontend identity system:
#
#   - granularity stays ATTACHMENT-LEVEL: one ``relevant_memories.txt``
#     attachment is ONE occurrence. The inner body is never parsed and
#     no item-level identity / memref / Recall is ever invented;
#   - the closed v0 action enum is ``preserve_direct_data`` ONLY. No
#     flash / summarize / recall / merge / dedup / drop high-level
#     action exists here;
#   - it is built ONLY from THIS request's already-produced
#     request-local authority (``workspace.memory_source_evidence`` /
#     ``workspace.memory_delivery_plan`` / ``workspace
#     .memory_authority_migration``). It never reads a persisted
#     "latest", never rebuilds evidence, never reruns the Surfacing
#     Policy and never reruns retrieval / Migration;
#   - it is IN-MEMORY ONLY: it carries the exact raw Frontend
#     attachment envelope (metadata + body) so a downstream stage can
#     prove exact-envelope binding. It is NEVER persisted, NEVER
#     logged, NEVER serialized and NEVER model-visible.
#
# Strict binding. Every one of the three upstream planes must be valid
# and request-local, must agree on the exact conversation / request /
# Unified revision, and must report BOTH source planes observation-
# available. An ``unavailable`` plane is never turned into "zero
# units". Every Frontend Source Evidence unit must have EXACTLY ONE
# persisted Frontend Delivery row and EXACTLY ONE persisted Frontend
# Migration row, matched by the ONE shared ``evidence_ref_for_fields``
# formula (never by body text and never by an arbitrary 64-hex string).
#
# Frontend row completeness. The persisted row cap may truncate rows,
# so the surface refuses to be built whenever the number of persisted
# Frontend Delivery / Migration rows does not equal the FULL-scope
# Frontend counts. Hidden Frontend rows are never fabricated from an
# aggregate count; a truncated Frontend row set is a refusal reason
# ``frontend_control_rows_incomplete``.
#
# This is a compatibility MIGRATION LANE, not the migration and not a
# policy freeze: it proves only that unified control COULD adopt the
# CURRENT Frontend direct exposure without changing what the model
# sees.

VERSION = "frontend-controlled-direct-surface.v1"

# The CLOSED high-level Frontend control action enum for v0. Only a
# compatibility preserve-direct-data action exists; the migration lane
# does NOT invent Flash / Recall / summarize / merge / dedup / drop.
ACTION_PRESERVE_DIRECT_DATA = "preserve_direct_data"

# The source Migration artifact contract this surface strictly binds to.
_MIGRATION_VERSION = (
    "memory-authority-migration-shadow.v1"
)
_MIGRATION_MODE = "shadow_only"
_MIGRATION_AUTHORITY = "comparison_only"

# The STRICT Delivery Frontend row contract a bound occurrence must
# satisfy: a direct-data baseline whose whole content is already in the
# request. It mirrors the Delivery Plan's own Frontend row contract, so
# a foreign row can never be adopted as controlled direct data.
_ROW_IMMEDIATE_DIRECT_DATA = "direct_data"
_ROW_FULL_ALREADY_PRESENT = "already_present"

# Closed no-surface reasons. A caller distinguishes "the surface was not
# built" from "the surface is empty" by ``ok`` + ``reason``, never by an
# ambiguous zero-unit surface.
REASON_INVALID_CONVERSATION = (
    "invalid_conversation_id"
)
REASON_INVALID_REQUEST = (
    "invalid_cognitive_request_id"
)
REASON_INVALID_REVISION = (
    "invalid_source_unified_revision"
)
REASON_EVIDENCE_NOT_BOUND = "evidence_not_bound"
REASON_SOURCE_UNAVAILABLE = (
    "source_observation_unavailable"
)
REASON_DELIVERY_NOT_BOUND = (
    "delivery_plan_not_bound"
)
REASON_MIGRATION_NOT_BOUND = (
    "migration_not_bound"
)
REASON_ROWS_INCOMPLETE = (
    "frontend_control_rows_incomplete"
)
REASON_ROW_BINDING_UNAVAILABLE = (
    "frontend_control_row_binding_unavailable"
)
REASON_ENVELOPE_UNAVAILABLE = (
    "frontend_native_envelope_unavailable"
)
REASON_BUILD_FAILED = "surface_build_failed"

KNOWN_REASONS = (
    REASON_INVALID_CONVERSATION,
    REASON_INVALID_REQUEST,
    REASON_INVALID_REVISION,
    REASON_EVIDENCE_NOT_BOUND,
    REASON_SOURCE_UNAVAILABLE,
    REASON_DELIVERY_NOT_BOUND,
    REASON_MIGRATION_NOT_BOUND,
    REASON_ROWS_INCOMPLETE,
    REASON_ROW_BINDING_UNAVAILABLE,
    REASON_ENVELOPE_UNAVAILABLE,
    REASON_BUILD_FAILED,
)


@dataclass(frozen=True)
class FrontendControlledDirectUnit:
    """One controlled Frontend occurrence. IN-MEMORY ONLY.

    ``native_raw_attachment`` is the exact raw envelope (metadata +
    body) of the occurrence, retained so a downstream stage can prove
    exact-envelope binding against the actual final body. It is
    excluded from ``repr`` / ``compare`` and is never persisted, never
    logged and never copied into a privacy-safe artifact.
    """

    evidence_ref: str
    source_occurrence: int
    source_scope: str
    content_digest: str
    model_visible_chars: int
    model_visible_estimated_tokens: int
    control_action: str = ACTION_PRESERVE_DIRECT_DATA
    native_raw_attachment: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class FrontendControlledDirectSurface:
    """Request-local, in-memory-only controlled direct surface.

    Deliberately not serializable: it carries exact raw Frontend
    envelopes. It is never logged, never persisted and never
    model-visible.
    """

    version: str
    conversation_id: str
    cognitive_request_id: str
    source_unified_revision: Any
    units: tuple[FrontendControlledDirectUnit, ...]


@dataclass(frozen=True)
class FrontendSurfaceResult:
    """The fail-closed outcome of building the surface."""

    ok: bool
    reason: str | None
    surface: FrontendControlledDirectSurface | None


def _refuse(reason: str) -> FrontendSurfaceResult:
    return FrontendSurfaceResult(
        ok=False,
        reason=reason,
        surface=None,
    )


def migration_bound_to_request(
    migration: Any,
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    source_unified_revision: Any,
) -> bool:
    """STRICT request-local Migration binding.

    Anything other than the exact shadow-only, comparison-only,
    never-live-cutover Migration artifact bound to THIS conversation,
    THIS request and THIS source Unified revision is refused. A
    comparison that did not actually verify the final Frontend
    presence is never a control basis, so both ``frontend_final
    _observation_available`` and ``frontend_final_presence_verified``
    must be True.
    """

    if not isinstance(migration, dict):
        return False

    if (
        migration.get("version") != _MIGRATION_VERSION
        or migration.get("mode") != _MIGRATION_MODE
        or migration.get("authority")
        != _MIGRATION_AUTHORITY
        or migration.get("live_authority") is not False
        or migration.get("live_cutover_ready")
        is not False
    ):
        return False

    if (
        migration.get("conversation_id")
        != conversation_id
    ):
        return False

    if (
        migration.get("cognitive_request_id")
        != cognitive_request_id
    ):
        return False

    if (
        migration.get("source_unified_revision")
        != source_unified_revision
    ):
        return False

    if (
        migration.get(
            "frontend_final_observation_available"
        )
        is not True
    ):
        return False

    if (
        migration.get(
            "frontend_final_presence_verified"
        )
        is not True
    ):
        return False

    return is_valid_memory_authority_migration_artifact(
        migration
    )


def _units(
    source_evidence_snapshot: Any,
) -> tuple[Any, ...]:
    return tuple(
        getattr(
            source_evidence_snapshot,
            "units",
            (),
        )
        or ()
    )


def _frontend_rows(
    rows: Any,
    source_key: str = "source",
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []

    return [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get(source_key) == SOURCE_FRONTEND
    ]


def _exactly_one_match(
    rows: list[dict[str, Any]],
    evidence_ref: str,
) -> dict[str, Any] | None:
    """The ONE persisted row whose evidence_ref equals ``evidence_ref``.

    Duplicate identical attachments have DISTINCT evidence refs (they
    differ by occurrence), so this is occurrence-safe. A missing ref
    and a duplicated ref are both a binding failure -- never a collapse
    and never a first-match pick.
    """

    matches = [
        row
        for row in rows
        if row.get("evidence_ref") == evidence_ref
    ]

    if len(matches) != 1:
        return None

    return matches[0]


def build_frontend_controlled_direct_surface(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    source_unified_revision: Any,
    source_evidence_snapshot: Any,
    delivery_plan: Any,
    migration: Any,
) -> FrontendSurfaceResult:
    """Build the in-memory controlled direct surface, or refuse it.

    Never raises on expected input. It consumes ONLY the three
    request-local planes and never rebuilds evidence, never reruns the
    Surfacing Policy / Migration and never reads a persisted "latest".
    """

    if not is_valid_conversation_id(
        conversation_id
    ):
        return _refuse(REASON_INVALID_CONVERSATION)

    if not is_valid_cognitive_request_id(
        cognitive_request_id
    ):
        return _refuse(REASON_INVALID_REQUEST)

    if not is_valid_revision(
        source_unified_revision
    ):
        return _refuse(REASON_INVALID_REVISION)

    if not is_bound_to_request(
        source_evidence_snapshot,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        expected_unified_revision=(
            source_unified_revision
        ),
    ):
        return _refuse(REASON_EVIDENCE_NOT_BOUND)

    # Both source planes must be truthfully observable. AVAILABLE +
    # zero units is valid EMPTY evidence; UNAVAILABLE is NOT.
    if (
        getattr(
            source_evidence_snapshot,
            "frontend_observation_available",
            None,
        )
        is not True
        or getattr(
            source_evidence_snapshot,
            "ob_observation_available",
            None,
        )
        is not True
    ):
        return _refuse(REASON_SOURCE_UNAVAILABLE)

    if not delivery_plan_bound_to_request(
        delivery_plan,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        source_unified_revision=(
            source_unified_revision
        ),
    ):
        return _refuse(REASON_DELIVERY_NOT_BOUND)

    if not migration_bound_to_request(
        migration,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        source_unified_revision=(
            source_unified_revision
        ),
    ):
        return _refuse(REASON_MIGRATION_NOT_BOUND)

    try:
        return _build(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            source_unified_revision=(
                source_unified_revision
            ),
            source_evidence_snapshot=(
                source_evidence_snapshot
            ),
            delivery_plan=delivery_plan,
            migration=migration,
        )
    except Exception:
        # Fail-closed: a fault is a missing surface, never a partial
        # one.
        return _refuse(REASON_BUILD_FAILED)


def _build(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    source_unified_revision: Any,
    source_evidence_snapshot: Any,
    delivery_plan: dict[str, Any],
    migration: dict[str, Any],
) -> FrontendSurfaceResult:
    units = _units(source_evidence_snapshot)

    frontend_units = [
        unit
        for unit in units
        if getattr(unit, "source", None)
        == SOURCE_FRONTEND
    ]

    delivery_rows = _frontend_rows(
        delivery_plan.get("rows")
    )

    migration_rows = _frontend_rows(
        migration.get("rows")
    )

    delivery_metrics = delivery_plan.get("metrics")

    migration_metrics = migration.get("metrics")

    if not isinstance(
        delivery_metrics, dict
    ) or not isinstance(migration_metrics, dict):
        return _refuse(REASON_ROWS_INCOMPLETE)

    # Frontend row completeness. The persisted row cap may hide rows;
    # a hidden Frontend row can never be authorized from an aggregate
    # count, so the persisted Frontend rows must equal the FULL-scope
    # count on BOTH planes, which must in turn equal the request-local
    # Frontend unit count.
    expected = len(frontend_units)

    if not (
        len(delivery_rows)
        == delivery_metrics.get("frontend_unit_count")
        == expected
    ):
        return _refuse(REASON_ROWS_INCOMPLETE)

    if not (
        len(migration_rows)
        == migration_metrics.get(
            "frontend_direct_bypass_count"
        )
        == expected
    ):
        return _refuse(REASON_ROWS_INCOMPLETE)

    controlled: list[FrontendControlledDirectUnit] = []

    for unit in frontend_units:
        if (
            getattr(unit, "kind", None)
            != KIND_FRONTEND_ATTACHMENT
        ):
            return _refuse(
                REASON_ROW_BINDING_UNAVAILABLE
            )

        # The ONE deterministic evidence ref, recomputed from the unit's
        # own fields. A row can only match when its persisted source /
        # kind / occurrence / digest agree exactly.
        ref = evidence_ref_for_fields(
            unit.source,
            unit.kind,
            unit.request_occurrence,
            unit.content_digest,
        )

        delivery_row = _exactly_one_match(
            delivery_rows, ref
        )

        migration_row = _exactly_one_match(
            migration_rows, ref
        )

        if delivery_row is None or migration_row is None:
            return _refuse(
                REASON_ROW_BINDING_UNAVAILABLE
            )

        # The Delivery row must be a direct-data / already-present
        # baseline; a foreign channel can never be adopted as
        # controlled direct data.
        if (
            delivery_row.get(
                "immediate_delivery_mode"
            )
            != _ROW_IMMEDIATE_DIRECT_DATA
            or delivery_row.get(
                "full_content_access_mode"
            )
            != _ROW_FULL_ALREADY_PRESENT
            or delivery_row.get(
                "currently_model_visible"
            )
            is not True
        ):
            return _refuse(
                REASON_ROW_BINDING_UNAVAILABLE
            )

        # The Migration row must be exactly the Frontend direct-bypass
        # gap needing a controlled Frontend channel.
        if (
            migration_row.get("authority_coverage")
            != COVERAGE_FRONTEND_GAP
            or migration_row.get(
                "replacement_requirement"
            )
            != REQ_FRONTEND_CHANNEL
        ):
            return _refuse(
                REASON_ROW_BINDING_UNAVAILABLE
            )

        raw_attachment = _native_raw_attachment(
            unit
        )

        if raw_attachment is None:
            return _refuse(
                REASON_ENVELOPE_UNAVAILABLE
            )

        controlled.append(
            FrontendControlledDirectUnit(
                evidence_ref=ref,
                source_occurrence=(
                    unit.request_occurrence
                ),
                source_scope=unit.source_scope,
                content_digest=unit.content_digest,
                model_visible_chars=(
                    unit.model_visible_chars
                ),
                model_visible_estimated_tokens=(
                    unit
                    .model_visible_estimated_tokens
                ),
                control_action=(
                    ACTION_PRESERVE_DIRECT_DATA
                ),
                native_raw_attachment=(
                    raw_attachment
                ),
            )
        )

    return FrontendSurfaceResult(
        ok=True,
        reason=None,
        surface=FrontendControlledDirectSurface(
            version=VERSION,
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            source_unified_revision=(
                source_unified_revision
            ),
            units=tuple(controlled),
        ),
    )


def _native_raw_attachment(
    unit: Any,
) -> str | None:
    """The exact canonical raw envelope of a Frontend evidence unit.

    It is read from the unit's in-memory ``native_payload`` (the
    ``FrontendMemorySegment``); a unit without a proven raw envelope has
    no controllable exact attachment, so it is refused rather than
    reconstructed.
    """

    payload = getattr(unit, "native_payload", None)

    raw = getattr(payload, "raw_attachment", None)

    if not isinstance(raw, str) or not raw:
        return None

    return raw


def surface_bound_to_request(
    surface: Any,
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    source_unified_revision: Any,
) -> bool:
    """True when the surface is THIS request's exact controlled surface.

    A foreign or mismatched surface must never become another request's
    control basis.
    """

    return bool(
        isinstance(
            surface,
            FrontendControlledDirectSurface,
        )
        and surface.version == VERSION
        and surface.conversation_id
        == conversation_id
        and surface.cognitive_request_id
        == cognitive_request_id
        and surface.source_unified_revision
        == source_unified_revision
    )