from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any

from ombrebrain.context.frontend_controlled_direct_surface import (
    FrontendSurfaceResult,
    migration_bound_to_request,
    surface_bound_to_request,
)
from ombrebrain.context.memory_authority_migration_shadow import (
    COVERAGE_FRONTEND_GAP,
    COVERAGE_OB_CUE,
    COVERAGE_OB_UNKNOWN,
    COVERAGE_OB_WITHHOLD,
    delivery_plan_bound_to_request,
    is_valid_memory_authority_migration_artifact,
)
from ombrebrain.context.memory_delivery_shadow import (
    evidence_ref_for_fields,
)
from ombrebrain.context.memory_lifecycle_event import (
    parse_iso_utc,
)
from ombrebrain.context.memory_source_evidence import (
    KIND_FRONTEND_ATTACHMENT,
    KIND_OB_CANDIDATE,
    SCOPE_CURRENT,
    SCOPE_HISTORY,
    SCOPE_RETRIEVAL,
    SOURCE_FRONTEND,
    SOURCE_OB,
)
from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    now_iso,
    read_json,
    state_root,
)
from ombrebrain.context.validators.freshness import (
    is_valid_revision,
)
from ombrebrain.gateway.canonicalizer import (
    REWRITE_KEEP_EXACT,
    extract_frontend_memory_segments,
    rewrite_frontend_memory_attachments,
)
from ombrebrain.gateway.operit_adapter import (
    OperitAdapter,
)


# Unified Memory Authority Compatibility Candidate Shadow v0 -- SHADOW
# ONLY, CANDIDATE ONLY.
#
# The FIRST target-body candidate this repository allows, and a
# deliberately ultra-narrow one:
#
#     Frontend attachment
#         -> Frontend Controlled Direct Surface
#             -> explicit controller decision preserve_direct_data
#                 -> candidate final body is EXACTLY the same bytes
#
# It answers ONE question, and only that question:
#
#     "could unified control EXPLICITLY adopt the CURRENT Frontend
#      direct exposure WITHOUT changing a single byte the model sees?"
#
# It is a ZERO-BEHAVIOUR-CHANGE compatibility candidate, NOT the live
# unification:
#
#   - strategy ``compatibility_preserve_current_frontend_direct.v1``;
#   - every Frontend occurrence is authorized as
#     ``preserve_direct_data`` -> the low-level canonical transform
#     action ``keep_exact``;
#   - the candidate body is DEFINED as the ORIGINAL final body bytes.
#     It is never reserialized, because a JSON reserialization could
#     change whitespace / escaping / formatting for no semantic reason.
#     A stored artifact MUST prove
#         candidate_body_bytes == final_body_bytes
#         candidate_body_sha256 == final_body_sha256
#         candidate_body_byte_identical == true
#   - OB is DELEGATED, never reimplemented: the existing OB control
#     plane keeps owning OB rendering, and OB rows are mapped from the
#     already-validated Migration full-scope metrics. ``ob_control
#     _strategy`` says so explicitly. An OB UNKNOWN stays a gap.
#
# Hard rules:
#   - it consumes ONLY THIS request's request-local
#     ``workspace.memory_source_evidence`` / ``workspace
#     .memory_delivery_plan`` / ``workspace.memory_authority_migration``
#     and the ACTUAL ``final_body``. It never reads a persisted
#     "latest" and never rebuilds evidence / Delivery / Migration;
#   - it never becomes the returned body in v0: the coordinator ignores
#     any body-like value. The already-selected ``final_body`` is always
#     returned unchanged;
#   - it never emits a live Frontend DROP, never removes a Frontend
#     attachment, never converts Frontend to Flash, never invents a
#     Frontend memref / Recall / item identity and never mutates a
#     memory;
#   - privacy: the persisted artifact and any log line carry only
#     counts, booleans, enums, revisions and sha256 digests. No body,
#     candidate body, raw attachment, Frontend body, OB content, memory
#     id, cue, memref, user text, query or system prompt is ever
#     persisted or logged.

VERSION = (
    "memory-live-authority-candidate-shadow.v1"
)
MODE = "shadow_only"
AUTHORITY = "candidate_only"
STRATEGY = (
    "compatibility_preserve_current_frontend_direct.v1"
)

# Explicit delegation declaration: the compatibility candidate does NOT
# own OB rendering. The existing controlled OB plane keeps that.
OB_CONTROL_STRATEGY = (
    "delegate_existing_ob_control_plane"
)

# Source contracts this candidate strictly binds to.
_MIGRATION_VERSION = (
    "memory-authority-migration-shadow.v1"
)

# Hard bound on persisted per-request candidate rows. It mirrors the
# source Migration persisted-row cap: candidate rows only ever mirror
# ALREADY persisted Migration rows, so hidden rows are never invented.
MAX_CANDIDATE_ROWS = 64

# The CLOSED candidate interpretation enum. A Migration coverage maps
# deterministically onto exactly ONE candidate coverage + handling.
CANDIDATE_COVERAGE_FRONTEND = (
    "covered_frontend_controlled_direct_candidate"
)
CANDIDATE_COVERAGE_OB_CUE = (
    "covered_ob_existing_cue_contract"
)
CANDIDATE_COVERAGE_OB_WITHHOLD = (
    "covered_ob_existing_withhold_contract"
)
CANDIDATE_COVERAGE_OB_UNKNOWN = (
    "gap_ob_policy_unknown"
)

CANDIDATE_HANDLING_FRONTEND = (
    "controlled_direct_passthrough"
)
CANDIDATE_HANDLING_OB_CUE = (
    "delegated_existing_ob_cue_control"
)
CANDIDATE_HANDLING_OB_WITHHOLD = (
    "delegated_existing_ob_withhold_control"
)
CANDIDATE_HANDLING_OB_UNKNOWN = (
    "unresolved_ob_policy"
)

# The CLOSED candidate coverage status. It is NEVER "live ready".
STATUS_BLOCKED = "blocked"
STATUS_COMPATIBILITY_COVERED = (
    "compatibility_covered"
)

_KNOWN_STATUSES = (
    STATUS_BLOCKED,
    STATUS_COMPATIBILITY_COVERED,
)

# live_cutover_ready is ALWAYS false in v0. The reason distinguishes
# "this candidate is still incomplete" from "the candidate is coherent
# but the live selector / rollout does not exist yet".
REASON_CANDIDATE_INCOMPLETE = (
    "compatibility_candidate_incomplete"
)
REASON_NO_LIVE_SELECTOR = (
    "live_selector_rollout_not_implemented"
)

_KNOWN_LIVE_CUTOVER_REASONS = (
    REASON_CANDIDATE_INCOMPLETE,
    REASON_NO_LIVE_SELECTOR,
)

# The CLOSED Delivery immediate-channel enum mirrored from Migration.
_MODE_DIRECT_DATA = "direct_data"
_MODE_FLASH_CUE = "flash_cue"
_MODE_DROP = "drop"
_MODE_UNKNOWN = "unknown"

_KNOWN_DELIVERY_MODES = (
    _MODE_DIRECT_DATA,
    _MODE_FLASH_CUE,
    _MODE_DROP,
    _MODE_UNKNOWN,
)

_FULL_ALREADY_PRESENT = "already_present"
_FULL_RECALL_REQUIRED = "recall_required"
_FULL_NONE = "none"
_FULL_UNKNOWN = "unknown"

_KNOWN_FULL_MODES = (
    _FULL_ALREADY_PRESENT,
    _FULL_RECALL_REQUIRED,
    _FULL_NONE,
    _FULL_UNKNOWN,
)

# The ONE deterministic candidate-row contract:
#   (source, current_delivery_mode)
#     -> (migration_coverage, candidate_handling, candidate_coverage,
#         kind, full_content_access_mode, currently_model_visible)
# It mirrors the source Migration row contract exactly, so a copied row
# is never weakened into a shape Migration could not itself produce.
_ROW_CONTRACT = {
    (SOURCE_FRONTEND, _MODE_DIRECT_DATA): (
        COVERAGE_FRONTEND_GAP,
        CANDIDATE_HANDLING_FRONTEND,
        CANDIDATE_COVERAGE_FRONTEND,
        KIND_FRONTEND_ATTACHMENT,
        _FULL_ALREADY_PRESENT,
        True,
    ),
    (SOURCE_OB, _MODE_FLASH_CUE): (
        COVERAGE_OB_CUE,
        CANDIDATE_HANDLING_OB_CUE,
        CANDIDATE_COVERAGE_OB_CUE,
        KIND_OB_CANDIDATE,
        _FULL_RECALL_REQUIRED,
        False,
    ),
    (SOURCE_OB, _MODE_DROP): (
        COVERAGE_OB_WITHHOLD,
        CANDIDATE_HANDLING_OB_WITHHOLD,
        CANDIDATE_COVERAGE_OB_WITHHOLD,
        KIND_OB_CANDIDATE,
        _FULL_NONE,
        False,
    ),
    (SOURCE_OB, _MODE_UNKNOWN): (
        COVERAGE_OB_UNKNOWN,
        CANDIDATE_HANDLING_OB_UNKNOWN,
        CANDIDATE_COVERAGE_OB_UNKNOWN,
        KIND_OB_CANDIDATE,
        _FULL_UNKNOWN,
        False,
    ),
}

_FRONTEND_SCOPES = (SCOPE_CURRENT, SCOPE_HISTORY)

# The exact metric name set. The algebra below pins them to ONE explicit
# full-request scope derived from the validated Migration artifact.
_METRIC_NAMES = (
    "unit_count",
    "frontend_controlled_direct_candidate_count",
    "ob_existing_cue_contract_count",
    "ob_existing_withhold_contract_count",
    "ob_policy_unknown_count",
    "candidate_covered_unit_count",
    "candidate_gap_unit_count",
    "frontend_exact_envelope_verified_count",
    "frontend_controlled_direct_model_visible_chars",
    "frontend_controlled_direct_model_visible"
    "_estimated_tokens",
    "source_migration_gap_unit_count",
    "adopted_frontend_gap_count",
    "remaining_gap_unit_count",
)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

# No-record reason when the exact raw Frontend envelope in the ACTUAL
# final body cannot be bound to the request-local controlled surface.
REASON_EXACT_ENVELOPE_UNAVAILABLE = (
    "frontend_exact_envelope_binding_unavailable"
)

logger = logging.getLogger(__name__)


def _path(
    conversation_id: str,
    cognitive_request_id: str,
):
    return (
        state_root()
        / "memory_live_authority_candidate"
        / conversation_id
        / (cognitive_request_id + ".json")
    )


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_nonneg_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and _SHA256_HEX.match(value) is not None
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonicalize(body: Any):
    """Parse ``body`` with the EXISTING canonicalizer.

    Returns None for anything that is not a supported Operit request, so
    a final body that cannot be canonicalized is never treated as "zero
    Frontend occurrences". The attachment grammar is never copied here.
    """

    if not isinstance(body, bytes):
        return None

    try:
        payload = json.loads(body)
    except Exception:
        return None

    if not OperitAdapter.supports(payload):
        return None

    try:
        return OperitAdapter().adapt(payload)
    except Exception:
        return None


def _scope_of(placement: Any) -> str:
    return (
        SCOPE_CURRENT
        if placement == SCOPE_CURRENT
        else SCOPE_HISTORY
    )


def _final_body_control_walk(
    final_body: Any,
    surface: Any,
) -> tuple[bool, dict[str, Any]]:
    """Bind every final-body Frontend occurrence, then keep_exact ALL.

    ``(ok, observation)``. The algorithm is occurrence-safe and exact:

      1. canonicalize the ACTUAL final body (the true model-facing
         bytes), never the forward / selected / persisted body;
      2. walk the final Frontend occurrences in canonical request order
         and FIFO-bind each to one surface unit keyed by
         ``(source_scope, exact raw envelope)``. Duplicate identical
         attachments stay distinct: they are matched by occurrence, so
         their exact envelopes bind independently;
      3. require an EXACT number match -- a final occurrence without a
         surface unit, or a surface unit without a final occurrence, is
         a binding failure;
      4. run the canonical control transform with ``keep_exact`` for
         EVERY occurrence and require the rewritten payload to deep-equal
         the original final payload and NOT to drop anything.

    Any failure is fail-closed: no candidate is built.
    """

    request = _canonicalize(final_body)

    if request is None:
        return False, {}

    segments = extract_frontend_memory_segments(
        request
    )

    queues: dict[tuple[str, str], list[Any]] = {}

    for unit in surface.units:
        key = (
            unit.source_scope,
            unit.native_raw_attachment,
        )

        queues.setdefault(key, []).append(unit)

    bound: list[Any] = []

    for segment in segments:
        key = (
            _scope_of(segment.placement),
            segment.raw_attachment,
        )

        queue = queues.get(key)

        if not queue:
            return False, {}

        bound.append(queue.pop(0))

    if any(queue for queue in queues.values()):
        return False, {}

    occurrence_count = len(segments)

    try:
        payload = json.loads(final_body)
    except Exception:
        return False, {}

    def decide(
        *,
        occurrence: int,
        placement: str,
        raw_attachment: str,
        body: str,
    ) -> Any:
        if occurrence >= len(bound):
            return None

        unit = bound[occurrence]

        if (
            placement != unit.source_scope
            or raw_attachment
            != unit.native_raw_attachment
        ):
            return None

        return REWRITE_KEEP_EXACT

    result = rewrite_frontend_memory_attachments(
        payload,
        decide,
        expected_occurrences=occurrence_count,
    )

    if not result.ok:
        return False, {}

    if (
        result.dropped_count != 0
        or result.kept_count != occurrence_count
    ):
        return False, {}

    # Every action is keep_exact, so the rewritten payload MUST
    # deep-equal the original decoded final payload.
    if result.payload != payload:
        return False, {}

    return True, {
        "frontend_occurrence_count": (
            occurrence_count
        ),
        "verified_count": occurrence_count,
    }


def _candidate_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    key = (
        row.get("source"),
        row.get("current_delivery_mode"),
    )

    contract = _ROW_CONTRACT.get(key)

    if contract is None:
        raise ValueError(
            "unmappable migration row"
        )

    (
        _coverage,
        handling,
        candidate_coverage,
    ) = contract[0], contract[1], contract[2]

    return {
        "evidence_ref": row.get("evidence_ref"),
        "source": row.get("source"),
        "kind": row.get("kind"),
        "source_scope": row.get("source_scope"),
        "source_occurrence": row.get(
            "source_occurrence"
        ),
        "content_digest": row.get("content_digest"),
        "current_delivery_mode": row.get(
            "current_delivery_mode"
        ),
        "current_full_content_access_mode": (
            row.get("current_full_content_access_mode")
        ),
        "currently_model_visible": row.get(
            "currently_model_visible"
        ),
        "authority_coverage": row.get(
            "authority_coverage"
        ),
        "replacement_requirement": row.get(
            "replacement_requirement"
        ),
        "candidate_coverage": candidate_coverage,
        "candidate_handling": handling,
    }


def _metrics(
    migration_metrics: dict[str, Any],
    *,
    verified_count: int,
) -> dict[str, Any]:
    """FULL-scope candidate metrics, derived from the validated Migration
    full-scope metrics. The mapping is CLOSED, so even a truncated
    Migration artifact yields exact full-scope algebra.
    """

    frontend = migration_metrics[
        "frontend_direct_bypass_count"
    ]
    cue = migration_metrics[
        "ob_controlled_cue_contract_count"
    ]
    withhold = migration_metrics[
        "ob_controlled_withhold_contract_count"
    ]
    unknown = migration_metrics["ob_policy_unknown_count"]

    return {
        "unit_count": migration_metrics["unit_count"],
        "frontend_controlled_direct_candidate_count":
            frontend,
        "ob_existing_cue_contract_count": cue,
        "ob_existing_withhold_contract_count": (
            withhold
        ),
        "ob_policy_unknown_count": unknown,
        "candidate_covered_unit_count": (
            frontend + cue + withhold
        ),
        "candidate_gap_unit_count": unknown,
        "frontend_exact_envelope_verified_count": (
            verified_count
        ),
        "frontend_controlled_direct_model_visible_chars":
            migration_metrics[
                "frontend_direct_model_visible_chars"
            ],
        "frontend_controlled_direct_model_visible"
        "_estimated_tokens":
            migration_metrics[
                "frontend_direct_model_visible"
                "_estimated_tokens"
            ],
        "source_migration_gap_unit_count": (
            migration_metrics["contract_gap_unit_count"]
        ),
        "adopted_frontend_gap_count": frontend,
        "remaining_gap_unit_count": unknown,
    }


def build_memory_live_authority_candidate_artifact(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    source_unified_revision: int,
    delivery_plan: dict[str, Any],
    migration: dict[str, Any],
    control_observation: dict[str, Any],
    final_body: bytes,
) -> dict[str, Any]:
    """PURE builder for one privacy-safe compatibility Candidate artifact.

    It performs NO real-request observation: the exact final-body
    Frontend control walk is observed EXACTLY ONCE by the calling
    orchestrator and passed in as ``control_observation``. This function
    never canonicalizes a body, never calls the Operit adapter, never
    reads runtime files and never reserializes a payload.

    The candidate body is DEFINED as the ORIGINAL final body bytes: the
    builder records the final bytes / sha256 and proves byte identity,
    so a reviewer can see that the compatibility controller changed
    nothing. No body is ever persisted.

    No logging; deterministic except for ``created_at``.
    """

    rows = [
        _candidate_row(row)
        for row in migration["rows"]
    ]

    metrics = _metrics(
        migration["metrics"],
        verified_count=(
            control_observation["verified_count"]
        ),
    )

    final_sha = _sha256_bytes(final_body)

    remaining = metrics["remaining_gap_unit_count"]

    ready = remaining == 0

    return {
        "version": VERSION,
        "mode": MODE,
        "authority": AUTHORITY,
        "strategy": STRATEGY,
        "live_authority": False,
        # v0 never applies the candidate and never claims a live cutover.
        "candidate_applied": False,
        "live_cutover_ready": False,
        "live_cutover_reason": (
            REASON_NO_LIVE_SELECTOR
            if ready
            else REASON_CANDIDATE_INCOMPLETE
        ),
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "source_unified_revision": (
            source_unified_revision
        ),
        "source_delivery_plan_version": (
            delivery_plan["version"]
        ),
        "source_migration_version": (
            migration["version"]
        ),
        "ob_control_strategy": OB_CONTROL_STRATEGY,
        "created_at": now_iso(),
        "frontend_control_action": (
            "preserve_direct_data"
        ),
        "compatibility_candidate_ready": ready,
        "candidate_coverage_status": (
            STATUS_COMPATIBILITY_COVERED
            if ready
            else STATUS_BLOCKED
        ),
        # The candidate body IS the original final body: only the
        # numeric / hashed proof is persisted, never either body.
        "candidate_body_byte_identical": True,
        "final_body_sha256": final_sha,
        "candidate_body_sha256": final_sha,
        "final_body_bytes": len(final_body),
        "candidate_body_bytes": len(final_body),
        "source_rows_truncated": (
            migration["source_rows_truncated"]
        ),
        "source_rows_truncated_count": (
            migration["source_rows_truncated_count"]
        ),
        "rows": rows,
        "metrics": metrics,
    }


def _valid_candidate_row(row: Any) -> bool:
    """One structurally + semantically valid candidate row."""

    if not isinstance(row, dict):
        return False

    source = row.get("source")

    if source not in (SOURCE_FRONTEND, SOURCE_OB):
        return False

    mode = row.get("current_delivery_mode")

    if mode not in _KNOWN_DELIVERY_MODES:
        return False

    contract = _ROW_CONTRACT.get((source, mode))

    if contract is None:
        return False

    (
        coverage,
        handling,
        candidate_coverage,
        kind,
        full_mode,
        visible,
    ) = contract

    if row.get("authority_coverage") != coverage:
        return False

    if row.get("candidate_handling") != handling:
        return False

    if (
        row.get("candidate_coverage")
        != candidate_coverage
    ):
        return False

    if row.get("kind") != kind:
        return False

    if (
        row.get("current_full_content_access_mode")
        != full_mode
    ):
        return False

    if row.get("currently_model_visible") is not visible:
        return False

    if source == SOURCE_FRONTEND:
        if row.get("source_scope") not in (
            _FRONTEND_SCOPES
        ):
            return False
    elif row.get("source_scope") != SCOPE_RETRIEVAL:
        return False

    if row.get("current_full_content_access_mode") not in (
        _KNOWN_FULL_MODES
    ):
        return False

    occurrence = row.get("source_occurrence")

    if not _is_nonneg_int(occurrence):
        return False

    content_digest = row.get("content_digest")

    if not _is_sha256(content_digest):
        return False

    if not _is_sha256(row.get("evidence_ref")):
        return False

    if row.get("evidence_ref") != (
        evidence_ref_for_fields(
            source,
            kind,
            occurrence,
            content_digest,
        )
    ):
        return False

    return True


def _row_coverage_counts(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    counts: Counter = Counter(
        row.get("candidate_coverage")
        for row in rows
    )

    return {
        CANDIDATE_COVERAGE_FRONTEND: counts.get(
            CANDIDATE_COVERAGE_FRONTEND, 0
        ),
        CANDIDATE_COVERAGE_OB_CUE: counts.get(
            CANDIDATE_COVERAGE_OB_CUE, 0
        ),
        CANDIDATE_COVERAGE_OB_WITHHOLD: counts.get(
            CANDIDATE_COVERAGE_OB_WITHHOLD, 0
        ),
        CANDIDATE_COVERAGE_OB_UNKNOWN: counts.get(
            CANDIDATE_COVERAGE_OB_UNKNOWN, 0
        ),
    }


def is_valid_memory_live_authority_candidate_artifact(
    artifact: Any,
) -> bool:
    """The single authoritative v0 candidate validator.

    A complete, coherent
    ``memory-live-authority-candidate-shadow.v1`` artifact is valid;
    anything else -- wrong version / mode / authority / strategy, a
    ``live_authority`` / ``candidate_applied`` / ``live_cutover_ready``
    that is not strictly False, a broken byte-identity proof, a forged
    Frontend control label, a broken full-scope algebra, an incomplete
    Frontend row set, or a ``compatibility_candidate_ready`` claim that
    contradicts the remaining OB gaps -- is INVALID as a whole. Future /
    newer versions are unsupported and therefore invalid.
    """

    if not isinstance(artifact, dict):
        return False

    if artifact.get("version") != VERSION:
        return False

    if artifact.get("mode") != MODE:
        return False

    if artifact.get("authority") != AUTHORITY:
        return False

    if artifact.get("strategy") != STRATEGY:
        return False

    if artifact.get("live_authority") is not False:
        return False

    if artifact.get("candidate_applied") is not False:
        return False

    if artifact.get("live_cutover_ready") is not False:
        return False

    if artifact.get(
        "live_cutover_reason"
    ) not in _KNOWN_LIVE_CUTOVER_REASONS:
        return False

    if artifact.get("ob_control_strategy") != (
        OB_CONTROL_STRATEGY
    ):
        return False

    if not is_valid_conversation_id(
        artifact.get("conversation_id")
    ):
        return False

    if not is_valid_cognitive_request_id(
        artifact.get("cognitive_request_id")
    ):
        return False

    if not is_valid_revision(
        artifact.get("source_unified_revision")
    ):
        return False

    if parse_iso_utc(
        artifact.get("created_at")
    ) is None:
        return False

    if (
        artifact.get("source_delivery_plan_version")
        != "memory-delivery-plan-shadow.v1"
    ):
        return False

    if (
        artifact.get("source_migration_version")
        != _MIGRATION_VERSION
    ):
        return False

    # Byte-identity proof: the candidate body IS the original final body.
    if (
        artifact.get("candidate_body_byte_identical")
        is not True
    ):
        return False

    if not _is_sha256(
        artifact.get("final_body_sha256")
    ):
        return False

    if not _is_sha256(
        artifact.get("candidate_body_sha256")
    ):
        return False

    if artifact.get("final_body_sha256") != (
        artifact.get("candidate_body_sha256")
    ):
        return False

    if not _is_nonneg_int(
        artifact.get("final_body_bytes")
    ):
        return False

    if not _is_nonneg_int(
        artifact.get("candidate_body_bytes")
    ):
        return False

    if artifact.get("final_body_bytes") != (
        artifact.get("candidate_body_bytes")
    ):
        return False

    status = artifact.get("candidate_coverage_status")

    if status not in _KNOWN_STATUSES:
        return False

    ready = artifact.get(
        "compatibility_candidate_ready"
    )

    if not _is_bool(ready):
        return False

    rows = artifact.get("rows")

    if not isinstance(rows, list):
        return False

    if len(rows) > MAX_CANDIDATE_ROWS:
        return False

    for row in rows:
        if not _valid_candidate_row(row):
            return False

    truncated = artifact.get("source_rows_truncated")

    if not _is_bool(truncated):
        return False

    truncated_count = artifact.get(
        "source_rows_truncated_count"
    )

    if not _is_nonneg_int(truncated_count):
        return False

    if truncated is not (truncated_count > 0):
        return False

    metrics = artifact.get("metrics")

    if not isinstance(metrics, dict):
        return False

    if set(metrics) != set(_METRIC_NAMES):
        return False

    for name in _METRIC_NAMES:
        if not _is_nonneg_int(metrics.get(name)):
            return False

    # Exact truncation bookkeeping: candidate rows mirror ONLY
    # already-persisted Migration rows.
    if metrics["unit_count"] != (
        len(rows) + truncated_count
    ):
        return False

    # FULL-scope algebra.
    if metrics["unit_count"] != (
        metrics["candidate_covered_unit_count"]
        + metrics["candidate_gap_unit_count"]
    ):
        return False

    if metrics["candidate_covered_unit_count"] != (
        metrics[
            "frontend_controlled_direct_candidate_count"
        ]
        + metrics["ob_existing_cue_contract_count"]
        + metrics[
            "ob_existing_withhold_contract_count"
        ]
    ):
        return False

    if metrics["candidate_gap_unit_count"] != (
        metrics["remaining_gap_unit_count"]
    ):
        return False

    if metrics["remaining_gap_unit_count"] != (
        metrics["ob_policy_unknown_count"]
    ):
        return False

    if metrics["adopted_frontend_gap_count"] != (
        metrics[
            "frontend_controlled_direct_candidate_count"
        ]
    ):
        return False

    if metrics[
        "frontend_exact_envelope_verified_count"
    ] != metrics[
        "frontend_controlled_direct_candidate_count"
    ]:
        return False

    if metrics["source_migration_gap_unit_count"] != (
        metrics[
            "frontend_controlled_direct_candidate_count"
        ]
        + metrics["ob_policy_unknown_count"]
    ):
        return False

    # Observed persisted-scope row lower bounds.
    observed = _row_coverage_counts(rows)

    if (
        metrics[
            "frontend_controlled_direct_candidate_count"
        ]
        < observed[CANDIDATE_COVERAGE_FRONTEND]
    ):
        return False

    if metrics["ob_existing_cue_contract_count"] < (
        observed[CANDIDATE_COVERAGE_OB_CUE]
    ):
        return False

    if metrics[
        "ob_existing_withhold_contract_count"
    ] < observed[CANDIDATE_COVERAGE_OB_WITHHOLD]:
        return False

    if metrics["ob_policy_unknown_count"] < (
        observed[CANDIDATE_COVERAGE_OB_UNKNOWN]
    ):
        return False

    # Frontend completeness is MANDATORY: no Frontend candidate row may
    # be hidden by truncation.
    frontend_rows = sum(
        1
        for row in rows
        if row.get("source") == SOURCE_FRONTEND
    )

    if frontend_rows != metrics[
        "frontend_controlled_direct_candidate_count"
    ]:
        return False

    # compatibility_candidate_ready is true iff there are NO remaining
    # gaps AND the Frontend control proof is complete AND the candidate
    # body is byte-identical (the latter two are structural above).
    if ready is not (
        metrics["remaining_gap_unit_count"] == 0
    ):
        return False

    if ready:
        if status != STATUS_COMPATIBILITY_COVERED:
            return False

        if (
            artifact["live_cutover_reason"]
            != REASON_NO_LIVE_SELECTOR
        ):
            return False
    else:
        if status != STATUS_BLOCKED:
            return False

        if (
            artifact["live_cutover_reason"]
            != REASON_CANDIDATE_INCOMPLETE
        ):
            return False

    return True


def _same_provenance(
    existing: dict[str, Any],
    artifact: dict[str, Any],
) -> bool:
    """Deterministic-duplicate identity, ignoring ``created_at``."""

    for name in (
        "conversation_id",
        "cognitive_request_id",
        "source_unified_revision",
        "candidate_coverage_status",
        "compatibility_candidate_ready",
        "candidate_body_byte_identical",
        "final_body_sha256",
        "candidate_body_sha256",
        "final_body_bytes",
        "candidate_body_bytes",
        "rows",
        "metrics",
        "source_rows_truncated",
        "source_rows_truncated_count",
    ):
        if existing.get(name) != artifact.get(name):
            return False

    return True


def _persist(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    artifact: dict[str, Any],
) -> tuple[bool, str | None]:
    path = _path(
        conversation_id,
        cognitive_request_id,
    )

    try:
        with LOCK:
            if path.is_file():
                existing = read_json(path)

                if existing is None:
                    return (
                        False,
                        "existing_artifact_corrupt",
                    )

                if not is_valid_memory_live_authority_candidate_artifact(
                    existing
                ):
                    return (
                        False,
                        "existing_artifact_invalid",
                    )

                if _same_provenance(
                    existing, artifact
                ):
                    return True, None

                return (
                    False,
                    "existing_artifact_mismatch",
                )

            atomic_write(path, artifact)
    except Exception:
        return False, "persist_failed"

    return True, None


def _report(
    *,
    stored: bool,
    decision: str,
    reason: str,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stored": stored,
        "mode": MODE,
        "decision": decision,
        "reason": reason,
        "strategy": STRATEGY,
    }

    if isinstance(artifact, dict):
        report["source_unified_revision"] = (
            artifact.get("source_unified_revision")
        )
        report["compatibility_candidate_ready"] = (
            artifact.get(
                "compatibility_candidate_ready"
            )
        )
        report["candidate_coverage_status"] = (
            artifact.get("candidate_coverage_status")
        )
        report["metrics"] = artifact.get("metrics")
        report["artifact"] = artifact

    return report


def update_memory_live_authority_candidate(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    delivery_plan: Any,
    migration: Any,
    surface: Any,
    source_unified_revision: Any,
    final_body: Any,
) -> dict[str, Any]:
    """Observe + persist one request's compatibility Candidate.

    Never raises. Returns a privacy-safe report. ``stored`` is False
    (with a precise reason) whenever a coherent compatibility candidate
    cannot be produced or the artifact cannot be written -- candidate
    failure is missing candidate evidence and NOTHING else.

    It consumes THIS request's request-local Delivery Plan / Migration
    only, plus the in-memory Frontend Controlled Direct Surface the
    caller built from request-local authority. It runs the
    exact-envelope final-body control walk and never returns a body.

    The surface is passed IN rather than rebuilt here so the controlled
    surface -- and its exact raw Frontend envelopes -- is built EXACTLY
    ONCE per request and never round-trips through a report dict.
    """

    if not is_valid_conversation_id(
        conversation_id
    ):
        return _report(
            stored=False,
            decision="no_record",
            reason="invalid_conversation_id",
        )

    if not is_valid_cognitive_request_id(
        cognitive_request_id
    ):
        return _report(
            stored=False,
            decision="no_record",
            reason="invalid_cognitive_request_id",
        )

    if not is_valid_revision(
        source_unified_revision
    ):
        return _report(
            stored=False,
            decision="no_record",
            reason="invalid_source_unified_revision",
        )

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
        return _report(
            stored=False,
            decision="no_record",
            reason="delivery_plan_not_bound",
        )

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
        return _report(
            stored=False,
            decision="no_record",
            reason="migration_not_bound",
        )

    if not isinstance(final_body, bytes):
        return _report(
            stored=False,
            decision="no_record",
            reason="invalid_body",
        )

    # The in-memory-only controlled surface is built EXACTLY ONCE by the
    # caller from THIS request's request-local authority. It is only
    # ever consumed here, never rebuilt, never persisted and never
    # returned.
    if not isinstance(surface, FrontendSurfaceResult):
        return _report(
            stored=False,
            decision="no_record",
            reason="surface_not_built",
        )

    if not surface.ok or surface.surface is None:
        logger.warning(
            "[gateway.context_memory_live_authority_candidate] "
            "reason=%s fail_open=true",
            surface.reason,
        )

        return _report(
            stored=False,
            decision="no_record",
            reason=(
                surface.reason or "surface_not_built"
            ),
        )

    if not surface_bound_to_request(
        surface.surface,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        source_unified_revision=(
            source_unified_revision
        ),
    ):
        return _report(
            stored=False,
            decision="no_record",
            reason="surface_not_bound",
        )

    ok, observation = _final_body_control_walk(
        final_body, surface.surface
    )

    if not ok:
        logger.warning(
            "[gateway.context_memory_live_authority_candidate] "
            "reason=frontend_exact_envelope_binding_"
            "unavailable fail_open=true"
        )

        return _report(
            stored=False,
            decision="no_record",
            reason=REASON_EXACT_ENVELOPE_UNAVAILABLE,
        )

    try:
        artifact = (
            build_memory_live_authority_candidate_artifact(
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
                source_unified_revision=(
                    source_unified_revision
                ),
                delivery_plan=delivery_plan,
                migration=migration,
                control_observation=observation,
                final_body=final_body,
            )
        )
    except Exception:
        return _report(
            stored=False,
            decision="no_record",
            reason="candidate_build_failed",
        )

    if not is_valid_memory_live_authority_candidate_artifact(
        artifact
    ):
        logger.warning(
            "[gateway.context_memory_live_authority_candidate] "
            "reason=invalid_candidate_artifact "
            "fail_open=true"
        )

        return _report(
            stored=False,
            decision="no_record",
            reason="invalid_candidate_artifact",
        )

    stored, reason = _persist(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        artifact=artifact,
    )

    if not stored:
        return _report(
            stored=False,
            decision="no_record",
            reason=reason or "persist_failed",
        )

    return _report(
        stored=True,
        decision="candidate_observed",
        reason="candidate_stored",
        artifact=artifact,
    )


def log_memory_live_authority_candidate_report(
    report: dict[str, Any],
) -> None:
    """Privacy-safe telemetry: aggregates, booleans, enums, revisions.

    No conversation id, cognitive request id, evidence ref, digest,
    body hash, memory id, cue, memref, attachment text or user text is
    ever logged.
    """

    if not isinstance(report, dict):
        return

    metrics = report.get("metrics")

    logger.info(
        "[gateway.context_memory_live_authority_candidate] %s",
        json.dumps(
            {
                "mode": report.get("mode"),
                "stored": report.get("stored"),
                "decision": report.get("decision"),
                "reason": report.get("reason"),
                "strategy": report.get("strategy"),
                "source_unified_revision": (
                    report.get(
                        "source_unified_revision"
                    )
                ),
                "compatibility_candidate_ready": (
                    report.get(
                        "compatibility_candidate_ready"
                    )
                ),
                "candidate_coverage_status": (
                    report.get(
                        "candidate_coverage_status"
                    )
                ),
                "metrics": (
                    {
                        key: metrics.get(key)
                        for key in _METRIC_NAMES
                    }
                    if isinstance(metrics, dict)
                    else None
                ),
            },
            sort_keys=True,
        ),
    )


def read_memory_live_authority_candidate(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
) -> dict[str, Any] | None:
    """Read one request-scoped Candidate artifact, or None. Read only."""

    if not (
        is_valid_conversation_id(conversation_id)
        and is_valid_cognitive_request_id(
            cognitive_request_id
        )
    ):
        return None

    return read_json(
        _path(
            conversation_id,
            cognitive_request_id,
        )
    )