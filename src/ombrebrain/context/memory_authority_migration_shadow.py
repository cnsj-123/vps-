from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any

from ombrebrain.context.memory_delivery_shadow import (
    evidence_ref_for_fields,
    is_valid_memory_delivery_plan_artifact,
)
from ombrebrain.context.memory_flash_live_exposure import (
    read_live_memory_exposure,
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
    is_bound_to_request,
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
    extract_frontend_memory_segments,
)
from ombrebrain.gateway.operit_adapter import (
    OperitAdapter,
)


# Memory Authority Migration Shadow v0 -- SHADOW ONLY, COMPARISON ONLY.
#
# The repository currently has TWO model-facing memory "mouths":
#
#   Frontend Memory: a ``relevant_memories.txt`` attachment is already
#       present in the forwarded provider request, so the model sees the
#       whole attachment directly.
#
#   OB Memory: retrieval -> Surfacing Policy -> optional Lifecycle gate
#       -> Flash -> Recall Surface -> optional Live Exposure -> a
#       bounded cue + opaque memref.
#
# PR #5 produced a request-local Source-Neutral Memory Evidence Snapshot
# and a Unified Memory Delivery Plan Shadow, but that Delivery Plan is a
# PROPOSAL (``authority=proposal_only``, ``live_authority=false``): it
# describes the channel a unit ALREADY uses, and it does NOT decide the
# actual final provider body.
#
# This module is a MIGRATION READINESS measurement, NOT the migration.
# It compares:
#
#     CURRENT actual model-facing memory exposure
#                     VS
#     SINGLE-AUTHORITY control-contract coverage
#
# over the EXACT request-local evidence of one request. It answers:
#
#   - how much Frontend Memory bypasses unified control TODAY;
#   - what would disappear if that direct mouth were removed with NO
#     replacement (numeric-only counterfactual);
#   - which OB cases already have a controlled cue / withhold contract;
#   - which requests still contain UNKNOWN control decisions.
#
# It is NOT the migration and it must not pretend to be. It never builds
# a hypothetical mutated body, never removes a Frontend attachment (not
# even in a copy), never introduces a Frontend renderer / memref /
# Recall, and never answers the still-OPEN UNIFIED_MEMORY_DELIVERY.md
# questions A-G.
#
# The target hypothesis only requires that every model-facing memory
# occurrence eventually has a controller-owned delivery decision. It
# does NOT require unified storage, a shared source schema, a shared
# identity system, a shared lifecycle or a shared retrieval system:
# the source stays federated and only model-facing AUTHORITY is
# eventually unified.
#
# Hard rules:
#   - it consumes THIS request's ``workspace.memory_source_evidence`` and
#     ``workspace.memory_delivery_plan`` only. It never rebuilds either,
#     never reads a conversation-global "latest" Unified / Delivery
#     Plan, and never re-runs retrieval / embedding / reranking /
#     Surfacing Policy / Lifecycle / Flash;
#   - it has NO live authority: it never calls
#     ``select_context_injected_body`` / ``select_live_memory_flash_body``,
#     never returns replacement body bytes and never changes
#     ``selected_body`` / ``final_body``;
#   - it performs no memory mutation (no touch / reinforce / archive /
#     delete / restore / record_usage / lifecycle write);
#   - it never invents a per-row OB live-exposure mapping: the Live
#     Exposure receipt is REQUEST-LEVEL aggregate provenance evidence,
#     so per-row OB coverage comes from the Delivery Plan CONTRACT and
#     the request-level actual exposure stays separate;
#   - current Frontend exposure is ATTACHMENT-LEVEL: one
#     ``relevant_memories.txt`` attachment is ONE occurrence. The inner
#     body is never parsed and no item-level readiness is claimed;
#   - privacy: the persisted artifact and any log line carry only
#     counts, booleans, enums, revisions and sha256 digests. No memory
#     text, cue, body, raw attachment, memory id, memref or candidate
#     fingerprint is ever emitted or persisted.

VERSION = "memory-authority-migration-shadow.v1"
MODE = "shadow_only"
AUTHORITY = "comparison_only"
HYPOTHESIS = "single_model_facing_memory_authority.v1"

# The source Delivery Plan contract this observer strictly binds to.
_DELIVERY_VERSION = "memory-delivery-plan-shadow.v1"
_DELIVERY_MODE = "shadow_only"
_DELIVERY_AUTHORITY = "proposal_only"
_DELIVERY_POLICY_STAGE = "base_surfacing_pre_lifecycle"

# Hard bound on persisted per-request migration rows. It mirrors the
# source Delivery Plan persisted-row cap: migration rows only ever
# mirror ALREADY persisted Delivery rows, so hidden migration rows are
# never invented.
MAX_MIGRATION_ROWS = 64

# The CLOSED Delivery immediate-channel enum this observer understands.
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

# The CLOSED full-content-access enum the Delivery Plan uses (kept
# separate from the delivery channel so a row is never mislabelled).
_MODE_FULL_ALREADY_PRESENT = "already_present"
_MODE_FULL_RECALL_REQUIRED = "recall_required"
_MODE_FULL_NONE = "none"
_MODE_FULL_UNKNOWN = "unknown"

_KNOWN_FULL_MODES = (
    _MODE_FULL_ALREADY_PRESENT,
    _MODE_FULL_RECALL_REQUIRED,
    _MODE_FULL_NONE,
    _MODE_FULL_UNKNOWN,
)

# The CLOSED migration coverage enum. A row maps deterministically to
# exactly ONE value; anything else invalidates the whole artifact.
COVERAGE_FRONTEND_GAP = "gap_frontend_direct_bypass"
COVERAGE_OB_CUE = "covered_ob_controlled_cue_contract"
COVERAGE_OB_WITHHOLD = (
    "covered_ob_controlled_withhold_contract"
)
COVERAGE_OB_UNKNOWN = "gap_ob_policy_unknown"

_KNOWN_COVERAGE = (
    COVERAGE_FRONTEND_GAP,
    COVERAGE_OB_CUE,
    COVERAGE_OB_WITHHOLD,
    COVERAGE_OB_UNKNOWN,
)

# The CLOSED replacement-requirement enum.
REQ_FRONTEND_CHANNEL = (
    "controlled_frontend_channel_required"
)
REQ_POLICY_RESOLUTION = "policy_resolution_required"
REQ_NONE = "none"

_KNOWN_REQUIREMENTS = (
    REQ_FRONTEND_CHANNEL,
    REQ_POLICY_RESOLUTION,
    REQ_NONE,
)

# The CLOSED top-level contract-coverage enum.
STATUS_BLOCKED = "blocked"
STATUS_COVERED = "covered"

_KNOWN_STATUSES = (STATUS_BLOCKED, STATUS_COVERED)

# live_cutover_ready is ALWAYS false in v0. The reason distinguishes
# "this request still has contract gaps" from "gaps are gone but the
# unified live renderer / selector does not exist yet".
REASON_GAPS_PRESENT = "contract_gaps_present"
REASON_NO_RENDERER = (
    "unified_live_renderer_not_implemented"
)

_KNOWN_LIVE_CUTOVER_REASONS = (
    REASON_GAPS_PRESENT,
    REASON_NO_RENDERER,
)

# No-record reasons for the two evidence-integrity gates. A whole source
# plane that cannot be observed truthfully, and a Source Evidence /
# Delivery Plan pool that disagrees, both refuse a readiness conclusion
# instead of persisting a misleading "covered" comparison.
REASON_SOURCE_UNAVAILABLE = (
    "source_observation_unavailable"
)
REASON_EVIDENCE_MISMATCH = (
    "source_delivery_evidence_mismatch"
)

# The ONE deterministic migration-row contract. Source + channel ->
# (coverage, replacement_requirement, kind, full_content_access_mode,
# currently_model_visible). It mirrors the upstream STRICT Delivery row
# contract exactly, so a copied row is never weakened into a shape the
# Delivery Plan could not itself have produced. Any other combination
# invalidates the whole artifact.
_ROW_CONTRACT = {
    (SOURCE_FRONTEND, _MODE_DIRECT_DATA): (
        COVERAGE_FRONTEND_GAP,
        REQ_FRONTEND_CHANNEL,
        KIND_FRONTEND_ATTACHMENT,
        _MODE_FULL_ALREADY_PRESENT,
        True,
    ),
    (SOURCE_OB, _MODE_FLASH_CUE): (
        COVERAGE_OB_CUE,
        REQ_NONE,
        KIND_OB_CANDIDATE,
        _MODE_FULL_RECALL_REQUIRED,
        False,
    ),
    (SOURCE_OB, _MODE_DROP): (
        COVERAGE_OB_WITHHOLD,
        REQ_NONE,
        KIND_OB_CANDIDATE,
        _MODE_FULL_NONE,
        False,
    ),
    (SOURCE_OB, _MODE_UNKNOWN): (
        COVERAGE_OB_UNKNOWN,
        REQ_POLICY_RESOLUTION,
        KIND_OB_CANDIDATE,
        _MODE_FULL_UNKNOWN,
        False,
    ),
}

_FRONTEND_SCOPES = (SCOPE_CURRENT, SCOPE_HISTORY)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

# Metric fields the validator understands. The algebra below pins them
# to ONE explicit full-request scope.
_METRIC_NAMES = (
    "unit_count",
    "contract_covered_unit_count",
    "contract_gap_unit_count",
    "frontend_direct_bypass_count",
    "frontend_replacement_required_count",
    "ob_controlled_cue_contract_count",
    "ob_controlled_withhold_contract_count",
    "ob_policy_unknown_count",
    "frontend_direct_model_visible_chars",
    "frontend_direct_model_visible_estimated_tokens",
    "counterfactual_loss_if_direct_mouth_removed_without_replacement_chars",
    "counterfactual_loss_if_direct_mouth_removed_without_replacement_estimated_tokens",
    "actual_ob_live_exposed_count",
    "actual_ob_live_exposure_estimated_tokens",
    "actual_model_visible_memory_estimated_tokens",
)

_EXPOSURE_INT_NAMES = (
    "pre_live_body_bytes",
    "final_body_bytes",
    "frontend_direct_unit_count",
    "frontend_direct_model_visible_chars",
    "frontend_direct_model_visible_estimated_tokens",
    "actual_ob_live_exposed_count",
    "actual_ob_live_exposure_estimated_tokens",
    "actual_ob_live_exposure_byte_delta",
)

logger = logging.getLogger(__name__)


def _path(
    conversation_id: str,
    cognitive_request_id: str,
):
    return (
        state_root()
        / "memory_authority_migration"
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


def _digest(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _canonicalize(body: Any):
    """Parse ``body`` with the EXISTING canonicalizer.

    Returns None for anything that is not a supported Operit request, so
    a body that cannot be canonicalized is never silently treated as
    "zero Frontend occurrences". The attachment grammar is never copied
    here; the canonical adapter is reused.
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


def _frontend_occurrence(segment: Any):
    """One final-body Frontend attachment -> one match tuple.

    ``(placement, content_digest, model_visible_chars,
    model_visible_estimated_tokens)`` -- the SAME tuple a request-local
    Source Evidence Frontend unit exposes, so occurrence / multiset
    matching is exact and duplicate identical attachments stay distinct.
    """

    body = getattr(segment, "body", None)

    if not isinstance(body, str) or not body.strip():
        return None

    raw = getattr(segment, "raw_attachment", None)

    if not isinstance(raw, str) or not raw:
        return None

    placement = getattr(
        segment,
        "placement",
        SCOPE_HISTORY,
    )

    scope = (
        SCOPE_CURRENT
        if placement == SCOPE_CURRENT
        else SCOPE_HISTORY
    )

    from ombrebrain.context.recall_types import (
        estimate_tokens,
    )

    return (
        scope,
        _digest(body),
        len(raw),
        estimate_tokens(raw),
    )


def _expected_frontend_multiset(
    source_evidence_snapshot: Any,
) -> Counter:
    units = tuple(
        getattr(
            source_evidence_snapshot,
            "units",
            (),
        )
        or ()
    )

    return Counter(
        (
            unit.source_scope,
            unit.content_digest,
            unit.model_visible_chars,
            unit.model_visible_estimated_tokens,
        )
        for unit in units
        if unit.source == SOURCE_FRONTEND
    )


def observe_frontend_final_presence(
    final_body: Any,
    source_evidence_snapshot: Any,
) -> tuple[bool, bool]:
    """``(available, verified)`` for the FINAL Frontend occurrence set.

    ``available`` is False whenever the final body cannot be
    canonicalized or a Frontend segment is malformed, so a comparison
    never fabricates current exposure from a broken final body.

    ``verified`` is True ONLY when the final Frontend multiset exactly
    equals THIS request's Source Evidence Frontend multiset. Occurrence
    counts, placement, digest and footprint must all agree; duplicate
    identical attachments stay distinct.
    """

    request = _canonicalize(final_body)

    if request is None:
        return False, False

    final_counts: Counter = Counter()

    for segment in extract_frontend_memory_segments(
        request
    ):
        occurrence = _frontend_occurrence(segment)

        if occurrence is None:
            return False, False

        final_counts[occurrence] += 1

    return (
        True,
        final_counts
        == _expected_frontend_multiset(
            source_evidence_snapshot
        ),
    )


def observe_actual_ob_live_exposure(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    pre_live_body: bytes,
    final_body: bytes,
) -> tuple[bool, dict[str, Any]]:
    """Request-level actual OB Live Exposure fact, observed EXACTLY once.

    ``(ok, data)``. When the OB live-exposure stage did not change the
    body the fact is an explicit zero -- and NO reason is inferred (it
    could be a flag, a Flash, a Surface, an eligibility or a budget
    outcome; the ONLY proven fact is that the body stage did not run).

    When the body DID change the existing validated request-scoped Live
    Exposure receipt is REQUIRED and must bind to the exact pre-live /
    final bytes. The receipt is request-scoped provenance evidence of
    the mutation, never a conversation-global "latest", and never a
    content-selection authority.
    """

    zero = {
        "applied": False,
        "exposed_count": 0,
        "estimated_tokens": 0,
        "byte_delta": 0,
    }

    if pre_live_body == final_body:
        return True, zero

    receipt = read_live_memory_exposure(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
    )

    if not isinstance(receipt, dict):
        return False, zero

    if (
        receipt.get("applied") is not True
        or receipt.get("original_sha256")
        != _sha256_bytes(pre_live_body)
        or receipt.get("selected_sha256")
        != _sha256_bytes(final_body)
        or receipt.get("original_bytes")
        != len(pre_live_body)
        or receipt.get("selected_bytes")
        != len(final_body)
        or receipt.get("byte_delta")
        != len(final_body) - len(pre_live_body)
    ):
        return False, zero

    return True, {
        "applied": True,
        "exposed_count": receipt[
            "live_exposed_count"
        ],
        "estimated_tokens": receipt[
            "estimated_tokens"
        ],
        "byte_delta": receipt["byte_delta"],
    }


def delivery_plan_bound_to_request(
    plan: Any,
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    source_unified_revision: Any,
) -> bool:
    """STRICT request-local Delivery Plan binding.

    Anything other than the exact shadow-only, proposal-only,
    pre-lifecycle plan bound to THIS conversation, THIS request and THIS
    source Unified revision is refused. There is no fallback to a
    persisted plan.
    """

    if not isinstance(plan, dict):
        return False

    if (
        plan.get("version") != _DELIVERY_VERSION
        or plan.get("mode") != _DELIVERY_MODE
        or plan.get("authority") != _DELIVERY_AUTHORITY
        or plan.get("live_authority") is not False
    ):
        return False

    if (
        plan.get("conversation_id")
        != conversation_id
    ):
        return False

    if (
        plan.get("cognitive_request_id")
        != cognitive_request_id
    ):
        return False

    if (
        plan.get("source_unified_revision")
        != source_unified_revision
    ):
        return False

    if (
        plan.get("policy_stage")
        != _DELIVERY_POLICY_STAGE
    ):
        return False

    return is_valid_memory_delivery_plan_artifact(plan)


def source_observation_available(
    source_evidence_snapshot: Any,
    delivery_plan: Any,
) -> bool:
    """True ONLY when BOTH source planes are truthfully observable.

    ``AVAILABLE + zero units`` is valid EMPTY evidence; ``UNAVAILABLE``
    is NOT. A whole source plane that cannot be observed truthfully must
    never become ``contract_gap_unit_count = 0`` / ``covered``.

    Both the request-local Source Evidence Snapshot AND the request-local
    Delivery Plan must independently report BOTH planes available, and
    the two availability views must AGREE, so a plan built from a
    different evidence pool can never silently cover an unobservable
    source.
    """

    snapshot_frontend = getattr(
        source_evidence_snapshot,
        "frontend_observation_available",
        None,
    )
    snapshot_ob = getattr(
        source_evidence_snapshot,
        "ob_observation_available",
        None,
    )

    if snapshot_frontend is not True:
        return False

    if snapshot_ob is not True:
        return False

    if not isinstance(delivery_plan, dict):
        return False

    plan_frontend = delivery_plan.get(
        "frontend_observation_available"
    )
    plan_ob = delivery_plan.get(
        "ob_observation_available"
    )

    if plan_frontend is not True:
        return False

    if plan_ob is not True:
        return False

    return (
        plan_frontend == snapshot_frontend
        and plan_ob == snapshot_ob
    )


def evidence_pool_matches_delivery(
    source_evidence_snapshot: Any,
    delivery_plan: Any,
) -> bool:
    """True ONLY when the Delivery Plan describes exactly the same source
    occurrence pool as the Source Evidence Snapshot.

    Request identity (conversation / request / revision) proves WHO the
    evidence belongs to, not WHAT the pool contains, so the full source
    split and the full unit count must agree:
    """

    units = tuple(
        getattr(
            source_evidence_snapshot,
            "units",
            (),
        )
        or ()
    )

    frontend_units = sum(
        1
        for unit in units
        if getattr(unit, "source", None)
        == SOURCE_FRONTEND
    )
    ob_units = sum(
        1
        for unit in units
        if getattr(unit, "source", None) == SOURCE_OB
    )

    if not isinstance(delivery_plan, dict):
        return False

    metrics = delivery_plan.get("metrics")

    if not isinstance(metrics, dict):
        return False

    return (
        metrics.get("frontend_unit_count")
        == frontend_units
        and metrics.get("ob_unit_count") == ob_units
        and metrics.get("unit_count") == len(units)
    )


def _migration_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    key = (
        row.get("source"),
        row.get("immediate_delivery_mode"),
    )

    contract = _ROW_CONTRACT.get(key)

    if contract is None:
        # A Delivery Plan row outside the closed mapping can never be
        # silently relabelled; the writer refuses the whole artifact.
        raise ValueError(
            "unmappable delivery row"
        )

    coverage, requirement = contract[0], contract[1]

    return {
        "evidence_ref": row.get("evidence_ref"),
        "source": row.get("source"),
        "kind": row.get("kind"),
        "source_scope": row.get("source_scope"),
        "source_occurrence": row.get(
            "source_occurrence"
        ),
        # The privacy-safe content digest is persisted ONLY so the
        # validator can recompute the occurrence evidence_ref from the
        # row's own fields. It is a sha256, never raw content, and it is
        # NOT a semantic identity claim.
        "content_digest": row.get("content_digest"),
        "current_delivery_mode": row.get(
            "immediate_delivery_mode"
        ),
        "current_full_content_access_mode": (
            row.get("full_content_access_mode")
        ),
        "currently_model_visible": row.get(
            "currently_model_visible"
        ),
        "authority_coverage": coverage,
        "replacement_requirement": requirement,
    }


def _metrics(
    delivery_metrics: dict[str, Any],
    *,
    live_exposed_count: int,
    live_exposure_estimated_tokens: int,
) -> dict[str, Any]:
    """FULL-scope migration metrics, derived from the validated Delivery
    Plan's own full-scope metrics.

    The mapping is CLOSED, so even a truncated Delivery Plan yields exact
    full-scope coverage algebra:

        every Frontend unit -> frontend direct bypass gap
        every OB flash      -> controlled cue coverage
        every OB drop       -> controlled withhold coverage
        every OB unknown    -> policy gap
    """

    cue = delivery_metrics[
        "ob_flash_cue_count"
    ]
    withhold = delivery_metrics["ob_drop_count"]
    unknown = delivery_metrics["ob_unknown_count"]

    frontend_bypass = delivery_metrics[
        "frontend_unit_count"
    ]

    covered = cue + withhold
    gap = frontend_bypass + unknown

    fe_chars = delivery_metrics[
        "current_model_visible_chars"
    ]
    fe_tokens = delivery_metrics[
        "current_model_visible_estimated_tokens"
    ]

    return {
        "unit_count": delivery_metrics["unit_count"],
        "contract_covered_unit_count": covered,
        "contract_gap_unit_count": gap,
        "frontend_direct_bypass_count": (
            frontend_bypass
        ),
        "frontend_replacement_required_count": (
            frontend_bypass
        ),
        "ob_controlled_cue_contract_count": cue,
        "ob_controlled_withhold_contract_count": (
            withhold
        ),
        "ob_policy_unknown_count": unknown,
        "frontend_direct_model_visible_chars": (
            fe_chars
        ),
        "frontend_direct_model_visible_estimated_tokens":
            fe_tokens,
        "counterfactual_loss_if_direct_mouth_"
        "removed_without_replacement_chars":
            fe_chars,
        "counterfactual_loss_if_direct_mouth_"
        "removed_without_replacement_estimated_tokens":
            fe_tokens,
        "actual_ob_live_exposed_count": (
            live_exposed_count
        ),
        "actual_ob_live_exposure_estimated_tokens":
            live_exposure_estimated_tokens,
        "actual_model_visible_memory_estimated_tokens":
            fe_tokens + live_exposure_estimated_tokens,
    }


def build_memory_authority_migration_artifact(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    source_unified_revision: int,
    delivery_plan: dict[str, Any],
    frontend_observation: tuple[bool, bool],
    live_exposure_observation: dict[str, Any],
    pre_live_body: bytes,
    final_body: bytes,
) -> dict[str, Any]:
    """PURE builder for one privacy-safe Migration Shadow artifact.

    It performs NO real-request observation. The FINAL-Frontend presence
    and the request-level OB Live Exposure are observed EXACTLY ONCE by
    the calling orchestrator and passed in. This function never
    canonicalizes a body, never calls the Operit adapter, never reads a
    Live Exposure receipt, never reads runtime files and never rereads
    request-global state: it only derives rows, metrics and labels from
    the already-validated observations. ``pre_live_body`` / ``final_body``
    are used ONLY for deterministic sha256 / byte-count bookkeeping and
    are never reparsed.

    It consumes the ALREADY BUILT request-local Delivery Plan and the
    two precomputed observations, never rebuilds any of them, never
    calls a selector and never mutates a body. No logging; deterministic
    except for ``created_at``.
    """

    delivery_metrics = delivery_plan["metrics"]

    frontend_available, frontend_verified = (
        frontend_observation
    )

    live = live_exposure_observation

    rows = [
        _migration_row(row)
        for row in delivery_plan["rows"]
    ]

    metrics = _metrics(
        delivery_metrics,
        live_exposed_count=live["exposed_count"],
        live_exposure_estimated_tokens=(
            live["estimated_tokens"]
        ),
    )

    status = (
        STATUS_BLOCKED
        if metrics["contract_gap_unit_count"] > 0
        else STATUS_COVERED
    )

    live_cutover_reason = (
        REASON_GAPS_PRESENT
        if metrics["contract_gap_unit_count"] > 0
        else REASON_NO_RENDERER
    )

    frontend_direct_unit_count = delivery_metrics[
        "frontend_unit_count"
    ]

    frontend_direct_mouth_active = (
        frontend_direct_unit_count > 0
    )
    ob_live_cue_mouth_active = (
        live["applied"] is True
    )

    current_exposure = {
        "pre_live_body_sha256": (
            _sha256_bytes(pre_live_body)
        ),
        "final_body_sha256": (
            _sha256_bytes(final_body)
        ),
        "pre_live_body_bytes": len(pre_live_body),
        "final_body_bytes": len(final_body),
        "live_body_changed": (
            pre_live_body != final_body
        ),
        "frontend_direct_unit_count": (
            frontend_direct_unit_count
        ),
        "frontend_direct_model_visible_chars":
            delivery_metrics[
                "current_model_visible_chars"
            ],
        "frontend_direct_model_visible_estimated_tokens":
            delivery_metrics[
                "current_model_visible"
                "_estimated_tokens"
            ],
        "actual_ob_live_exposure_applied": (
            live["applied"]
        ),
        "actual_ob_live_exposed_count": (
            live["exposed_count"]
        ),
        "actual_ob_live_exposure_estimated_tokens":
            live["estimated_tokens"],
        "actual_ob_live_exposure_byte_delta": (
            live["byte_delta"]
        ),
    }

    return {
        "version": VERSION,
        "mode": MODE,
        "authority": AUTHORITY,
        "hypothesis": HYPOTHESIS,
        "live_authority": False,
        # v0 never claims a live cutover: even with ZERO contract gaps
        # there is still no unified live renderer / selector, so a
        # reporter can never announce "safe to cut over".
        "live_cutover_ready": False,
        "live_cutover_reason": live_cutover_reason,
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "source_unified_revision": (
            source_unified_revision
        ),
        "source_delivery_plan_version": (
            delivery_plan["version"]
        ),
        "source_delivery_plan_mode": (
            delivery_plan["mode"]
        ),
        "source_delivery_policy_stage": (
            delivery_plan["policy_stage"]
        ),
        "created_at": now_iso(),
        "frontend_final_observation_available": (
            frontend_available
        ),
        "frontend_final_presence_verified": (
            frontend_verified
        ),
        "frontend_direct_mouth_active": (
            frontend_direct_mouth_active
        ),
        "ob_live_cue_mouth_active": (
            ob_live_cue_mouth_active
        ),
        "dual_mouth_active": (
            frontend_direct_mouth_active
            and ob_live_cue_mouth_active
        ),
        "current_exposure": current_exposure,
        "contract_coverage_status": status,
        "rows": rows,
        "source_rows_truncated": (
            delivery_plan["rows_truncated"]
        ),
        "source_rows_truncated_count": (
            delivery_plan["rows_truncated_count"]
        ),
        "metrics": metrics,
    }


def _valid_migration_row(row: Any) -> bool:
    """One structurally + semantically valid migration row.

    The migration row is downstream from a STRICT Delivery Plan, so its
    full shape is CLOSED against the upstream contract: source / kind /
    scope / delivery channel / full-content access mode / visibility /
    coverage / replacement requirement must ALL match the ONE contract
    entry for its ``(source, current_delivery_mode)`` key. A row whose
    derived label disagrees with its own source / channel -- e.g. a
    Frontend row relabelled as a controlled OB cue, or an OB unknown
    relabelled as a controlled withhold -- invalidates the whole
    artifact.

    ``evidence_ref`` is NOT merely checked for 64-hex shape: it is
    INDEPENDENTLY recomputed from the row's own persisted
    ``content_digest`` with the ONE shared Delivery helper, so a
    valid-looking random digest can never become trusted occurrence
    provenance.
    """

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
        requirement,
        kind,
        full_mode,
        visible,
    ) = contract

    if row.get("kind") != kind:
        return False

    if (
        row.get("current_full_content_access_mode")
        != full_mode
    ):
        return False

    if (
        row.get("currently_model_visible")
        is not visible
    ):
        return False

    if row.get("authority_coverage") != coverage:
        return False

    if (
        row.get("replacement_requirement")
        != requirement
    ):
        return False

    if source == SOURCE_FRONTEND:
        if row.get("source_scope") not in (
            _FRONTEND_SCOPES
        ):
            return False
    elif row.get("source_scope") != SCOPE_RETRIEVAL:
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


def _valid_current_exposure(
    exposure: Any,
) -> bool:
    """The exact current-exposure fact shape and its invariants."""

    if not isinstance(exposure, dict):
        return False

    for name in (
        "pre_live_body_sha256",
        "final_body_sha256",
    ):
        if not _is_sha256(exposure.get(name)):
            return False

    for name in _EXPOSURE_INT_NAMES:
        if not _is_nonneg_int(
            exposure.get(name)
        ):
            return False

    for name in (
        "live_body_changed",
        "actual_ob_live_exposure_applied",
    ):
        if not _is_bool(exposure.get(name)):
            return False

    if exposure["live_body_changed"] is False:
        if (
            exposure["pre_live_body_sha256"]
            != exposure["final_body_sha256"]
            or exposure["pre_live_body_bytes"]
            != exposure["final_body_bytes"]
            or exposure[
                "actual_ob_live_exposure_applied"
            ]
            is not False
            or exposure[
                "actual_ob_live_exposed_count"
            ]
            != 0
            or exposure[
                "actual_ob_live_exposure_estimated_tokens"
            ]
            != 0
            or exposure[
                "actual_ob_live_exposure_byte_delta"
            ]
            != 0
        ):
            return False

        return True

    # live_body_changed is True.
    if (
        exposure["pre_live_body_sha256"]
        == exposure["final_body_sha256"]
    ):
        return False

    if not (
        exposure["final_body_bytes"]
        > exposure["pre_live_body_bytes"]
    ):
        return False

    if (
        exposure["actual_ob_live_exposure_applied"]
        is not True
    ):
        return False

    if not (
        exposure["actual_ob_live_exposed_count"] > 0
    ):
        return False

    if not (
        exposure[
            "actual_ob_live_exposure_estimated_tokens"
        ]
        > 0
    ):
        return False

    byte_delta = (
        exposure["final_body_bytes"]
        - exposure["pre_live_body_bytes"]
    )

    if (
        exposure[
            "actual_ob_live_exposure_byte_delta"
        ]
        != byte_delta
    ):
        return False

    return byte_delta > 0


def _row_coverage_counts(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {
        COVERAGE_FRONTEND_GAP: 0,
        COVERAGE_OB_CUE: 0,
        COVERAGE_OB_WITHHOLD: 0,
        COVERAGE_OB_UNKNOWN: 0,
    }

    for row in rows:
        coverage = row.get("authority_coverage")

        if coverage in counts:
            counts[coverage] += 1

    return counts


def is_valid_memory_authority_migration_artifact(
    artifact: Any,
) -> bool:
    """The single authoritative v0 migration-shadow validator.

    A complete, structurally and semantically sound
    ``memory-authority-migration-shadow.v1`` artifact is valid;
    anything else (wrong version / mode / authority / hypothesis,
    malformed identity, a non-stored frontend observation, a
    contradictory current-exposure shape, a forged derived coverage
    label, a broken full-scope algebra, a non-false live-cutover claim)
    is INVALID as a whole. Future / newer versions are unsupported and
    therefore invalid -- they are never partially interpreted.
    """

    if not isinstance(artifact, dict):
        return False

    if artifact.get("version") != VERSION:
        return False

    if artifact.get("mode") != MODE:
        return False

    if artifact.get("authority") != AUTHORITY:
        return False

    if artifact.get("hypothesis") != HYPOTHESIS:
        return False

    if artifact.get("live_authority") is not False:
        return False

    if artifact.get("live_cutover_ready") is not False:
        return False

    if artifact.get(
        "live_cutover_reason"
    ) not in _KNOWN_LIVE_CUTOVER_REASONS:
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

    if (
        artifact.get("source_delivery_plan_version")
        != _DELIVERY_VERSION
    ):
        return False

    if (
        artifact.get("source_delivery_plan_mode")
        != _DELIVERY_MODE
    ):
        return False

    if (
        artifact.get("source_delivery_policy_stage")
        != _DELIVERY_POLICY_STAGE
    ):
        return False

    if parse_iso_utc(
        artifact.get("created_at")
    ) is None:
        return False

    # A STORED comparison has ALREADY proven final-body occurrence
    # equality, so both Frontend facts are strictly True. A pre-storage
    # "available but unverified" shape is never persisted.
    if (
        artifact.get(
            "frontend_final_observation_available"
        )
        is not True
    ):
        return False

    if (
        artifact.get(
            "frontend_final_presence_verified"
        )
        is not True
    ):
        return False

    for name in (
        "frontend_direct_mouth_active",
        "ob_live_cue_mouth_active",
        "dual_mouth_active",
    ):
        if not _is_bool(artifact.get(name)):
            return False

    exposure = artifact.get("current_exposure")

    if not _valid_current_exposure(exposure):
        return False

    # Derived actual-exposure facts must match their own numbers.
    if artifact["frontend_direct_mouth_active"] is not (
        exposure["frontend_direct_unit_count"] > 0
    ):
        return False

    if artifact["ob_live_cue_mouth_active"] is not (
        exposure["actual_ob_live_exposure_applied"]
        is True
    ):
        return False

    if artifact["dual_mouth_active"] is not (
        artifact["frontend_direct_mouth_active"]
        and artifact["ob_live_cue_mouth_active"]
    ):
        return False

    status = artifact.get("contract_coverage_status")

    if status not in _KNOWN_STATUSES:
        return False

    rows = artifact.get("rows")

    if not isinstance(rows, list):
        return False

    if len(rows) > MAX_MIGRATION_ROWS:
        return False

    for row in rows:
        if not _valid_migration_row(row):
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

    # Exact truncation bookkeeping. Migration rows mirror ONLY the
    # already-persisted Delivery rows, so the FULL unit_count is EXACTLY
    # the persisted rows plus the source Delivery truncation telemetry.
    # Internally legal aggregate algebra can never invent hidden rows
    # beyond that exact identity.
    if metrics["unit_count"] != (
        len(rows) + truncated_count
    ):
        return False

    if truncated is False and metrics["unit_count"] != len(
        rows
    ):
        return False

    if truncated is True and not (
        truncated_count > 0
        and metrics["unit_count"] > len(rows)
    ):
        return False

    # FULL-scope algebra. Every counter shares ONE explicit request
    # scope, so a forged counter that breaks the partition is rejected.
    if metrics["unit_count"] != (
        metrics["contract_covered_unit_count"]
        + metrics["contract_gap_unit_count"]
    ):
        return False

    if metrics["contract_covered_unit_count"] != (
        metrics["ob_controlled_cue_contract_count"]
        + metrics[
            "ob_controlled_withhold_contract_count"
        ]
    ):
        return False

    if metrics["contract_gap_unit_count"] != (
        metrics["frontend_direct_bypass_count"]
        + metrics["ob_policy_unknown_count"]
    ):
        return False

    if metrics["frontend_replacement_required_count"] != (
        metrics["frontend_direct_bypass_count"]
    ):
        return False

    # The counterfactual is a numeric impact estimate ONLY: it equals
    # the CURRENT verified Frontend direct footprint. It is NOT
    # "tokens saved" and NOT a recommended reduction.
    if metrics[
        "counterfactual_loss_if_direct_mouth_"
        "removed_without_replacement_chars"
    ] != metrics["frontend_direct_model_visible_chars"]:
        return False

    if metrics[
        "counterfactual_loss_if_direct_mouth_"
        "removed_without_replacement_estimated_tokens"
    ] != metrics[
        "frontend_direct_model_visible"
        "_estimated_tokens"
    ]:
        return False

    if metrics[
        "actual_model_visible_memory_estimated_tokens"
    ] != (
        metrics[
            "frontend_direct_model_visible"
            "_estimated_tokens"
        ]
        + metrics[
            "actual_ob_live_exposure"
            "_estimated_tokens"
        ]
    ):
        return False

    # Cross-lock: ``current_exposure`` and ``metrics`` persist the SAME
    # real exposure facts, so the two views must agree EXACTLY. A
    # duplicate factual representation that disagrees invalidates the
    # whole artifact -- there is no "one of them is authoritative" rule.
    if exposure["frontend_direct_unit_count"] != (
        metrics["frontend_direct_bypass_count"]
    ):
        return False

    if metrics["frontend_direct_bypass_count"] != (
        metrics["frontend_replacement_required_count"]
    ):
        return False

    if exposure[
        "frontend_direct_model_visible_chars"
    ] != metrics["frontend_direct_model_visible_chars"]:
        return False

    if metrics["frontend_direct_model_visible_chars"] != (
        metrics[
            "counterfactual_loss_if_direct_mouth_"
            "removed_without_replacement_chars"
        ]
    ):
        return False

    if exposure[
        "frontend_direct_model_visible_estimated_tokens"
    ] != metrics[
        "frontend_direct_model_visible_estimated_tokens"
    ]:
        return False

    if metrics[
        "frontend_direct_model_visible_estimated_tokens"
    ] != metrics[
        "counterfactual_loss_if_direct_mouth_"
        "removed_without_replacement_estimated_tokens"
    ]:
        return False

    if exposure["actual_ob_live_exposed_count"] != (
        metrics["actual_ob_live_exposed_count"]
    ):
        return False

    if exposure[
        "actual_ob_live_exposure_estimated_tokens"
    ] != metrics[
        "actual_ob_live_exposure_estimated_tokens"
    ]:
        return False

    if metrics[
        "actual_model_visible_memory_estimated_tokens"
    ] != (
        exposure[
            "frontend_direct_model_visible"
            "_estimated_tokens"
        ]
        + exposure[
            "actual_ob_live_exposure_estimated_tokens"
        ]
    ):
        return False

    # The persisted rows must agree with the coverage counters. When the
    # source Delivery Plan was NOT truncated the persisted rows ARE the
    # full scope, so the counters must match exactly; when it WAS
    # truncated the dropped rows are gone, so the full-scope counters
    # are only bounded below by what the persisted rows prove.
    observed = _row_coverage_counts(rows)

    if truncated is False:
        if metrics["unit_count"] != len(rows):
            return False

        if metrics[
            "frontend_direct_bypass_count"
        ] != observed[COVERAGE_FRONTEND_GAP]:
            return False

        if metrics["ob_controlled_cue_contract_count"] != (
            observed[COVERAGE_OB_CUE]
        ):
            return False

        if metrics[
            "ob_controlled_withhold_contract_count"
        ] != observed[COVERAGE_OB_WITHHOLD]:
            return False

        if metrics["ob_policy_unknown_count"] != (
            observed[COVERAGE_OB_UNKNOWN]
        ):
            return False
    else:
        for coverage, name in (
            (
                COVERAGE_FRONTEND_GAP,
                "frontend_direct_bypass_count",
            ),
            (
                COVERAGE_OB_CUE,
                "ob_controlled_cue_contract_count",
            ),
            (
                COVERAGE_OB_WITHHOLD,
                "ob_controlled_withhold_contract_count",
            ),
            (
                COVERAGE_OB_UNKNOWN,
                "ob_policy_unknown_count",
            ),
        ):
            if metrics[name] < observed[coverage]:
                return False

    # Coverage status follows the gap count exactly: never heuristically.
    if metrics["contract_gap_unit_count"] > 0:
        if status != STATUS_BLOCKED:
            return False

        if (
            artifact["live_cutover_reason"]
            != REASON_GAPS_PRESENT
        ):
            return False
    else:
        if status != STATUS_COVERED:
            return False

        if (
            artifact["live_cutover_reason"]
            != REASON_NO_RENDERER
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
        "current_exposure",
        "contract_coverage_status",
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
    """Request-scoped atomic persistence with duplicate safety.

    An existing valid artifact for the SAME request with identical
    provenance is a deterministic duplicate success and is never
    overwritten. An existing corrupt / mismatching artifact is refused
    and never silently overwritten.
    """

    path = _path(
        conversation_id,
        cognitive_request_id,
    )

    try:
        with LOCK:
            if path.is_file():
                existing = read_json(path)

                if existing is None:
                    return False, "existing_artifact_corrupt"

                if (
                    not is_valid_memory_authority_migration_artifact(
                        existing
                    )
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
    }

    if isinstance(artifact, dict):
        report[
            "source_unified_revision"
        ] = artifact.get("source_unified_revision")
        report["frontend_direct_mouth_active"] = (
            artifact.get("frontend_direct_mouth_active")
        )
        report["ob_live_cue_mouth_active"] = (
            artifact.get("ob_live_cue_mouth_active")
        )
        report["dual_mouth_active"] = (
            artifact.get("dual_mouth_active")
        )
        report["contract_coverage_status"] = (
            artifact.get("contract_coverage_status")
        )
        report["metrics"] = artifact.get("metrics")
        report["artifact"] = artifact

    return report


def update_memory_authority_migration(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    source_evidence_snapshot: Any,
    delivery_plan: Any,
    source_unified_revision: Any,
    pre_live_body: Any,
    final_body: Any,
) -> dict[str, Any]:
    """Observe + persist one request's Migration Readiness comparison.

    Never raises. Returns a privacy-safe report. ``stored`` is False
    (with a precise reason) whenever a truthful comparison cannot be
    made or the artifact cannot be written -- migration-shadow failure
    is missing comparison data and NOTHING else.

    It consumes THIS request's request-local Source Evidence Snapshot
    and Delivery Plan only and never reads a conversation-global
    "latest". It never mutates a body, never calls a selector, never
    invokes a Flash / Recall and never reinforces a memory.
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
        return _report(
            stored=False,
            decision="no_record",
            reason="evidence_not_bound",
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

    if not isinstance(
        pre_live_body, bytes
    ) or not isinstance(final_body, bytes):
        return _report(
            stored=False,
            decision="no_record",
            reason="invalid_body",
        )

    # Evidence-readiness gate. A whole source plane that cannot be
    # observed truthfully must never become "empty / covered". Both the
    # request-local Source Evidence Snapshot AND the request-local
    # Delivery Plan must report BOTH planes available and agree.
    # AVAILABLE + zero units stays valid EMPTY evidence.
    if not source_observation_available(
        source_evidence_snapshot,
        delivery_plan,
    ):
        logger.warning(
            "[gateway.context_memory_authority_migration] "
            "reason=source_observation_unavailable "
            "fail_open=true"
        )

        return _report(
            stored=False,
            decision="no_record",
            reason=REASON_SOURCE_UNAVAILABLE,
        )

    # Request identity is not semantic identity: the Delivery Plan must
    # describe exactly the same source occurrence pool (per-source split
    # and full total) as the Source Evidence Snapshot.
    if not evidence_pool_matches_delivery(
        source_evidence_snapshot,
        delivery_plan,
    ):
        logger.warning(
            "[gateway.context_memory_authority_migration] "
            "reason=source_delivery_evidence_mismatch "
            "fail_open=true"
        )

        return _report(
            stored=False,
            decision="no_record",
            reason=REASON_EVIDENCE_MISMATCH,
        )

    # Current Frontend exposure is verified against the ACTUAL final
    # body EXACTLY ONCE, here at the orchestration boundary. A
    # canonicalization failure, a malformed final Frontend segment or
    # ANY occurrence / placement / digest / footprint disagreement means
    # the current exposure cannot be truthfully observed.
    frontend_observation = (
        observe_frontend_final_presence(
            final_body,
            source_evidence_snapshot,
        )
    )

    frontend_available, frontend_verified = (
        frontend_observation
    )

    if not (
        frontend_available and frontend_verified
    ):
        logger.warning(
            "[gateway.context_memory_authority_migration] "
            "reason=final_frontend_binding_unavailable "
            "fail_open=true"
        )

        return _report(
            stored=False,
            decision="no_record",
            reason=(
                "final_frontend_binding_unavailable"
            ),
        )

    # Actual OB Live Exposure is a REQUEST-LEVEL fact, observed EXACTLY
    # ONCE here. A body that changed REQUIRES the validated request-
    # scoped receipt; a missing / corrupt / mismatching receipt means no
    # record.
    (
        live_ok,
        live_exposure_observation,
    ) = observe_actual_ob_live_exposure(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        pre_live_body=pre_live_body,
        final_body=final_body,
    )

    if not live_ok:
        logger.warning(
            "[gateway.context_memory_authority_migration] "
            "reason=live_exposure_receipt_mismatch "
            "fail_open=true"
        )

        return _report(
            stored=False,
            decision="no_record",
            reason="live_exposure_receipt_mismatch",
        )

    try:
        artifact = (
            build_memory_authority_migration_artifact(
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
                source_unified_revision=(
                    source_unified_revision
                ),
                delivery_plan=delivery_plan,
                frontend_observation=(
                    frontend_observation
                ),
                live_exposure_observation=(
                    live_exposure_observation
                ),
                pre_live_body=pre_live_body,
                final_body=final_body,
            )
        )
    except Exception:
        return _report(
            stored=False,
            decision="no_record",
            reason="migration_build_failed",
        )

    # Never persist malformed evidence: validate with the SAME v0
    # validator the reporter uses.
    if not is_valid_memory_authority_migration_artifact(
        artifact
    ):
        logger.warning(
            "[gateway.context_memory_authority_migration] "
            "reason=invalid_migration_artifact "
            "fail_open=true"
        )

        return _report(
            stored=False,
            decision="no_record",
            reason="invalid_migration_artifact",
        )

    ok, reason = _persist(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        artifact=artifact,
    )

    if not ok:
        return _report(
            stored=False,
            decision="no_record",
            reason=reason or "persist_failed",
        )

    return _report(
        stored=True,
        decision="compared",
        reason="comparison_stored",
        artifact=artifact,
    )


def log_memory_authority_migration_report(
    report: dict[str, Any],
) -> None:
    """Privacy-safe telemetry: aggregates, booleans and reasons only.

    No conversation id, cognitive request id, evidence ref, digest,
    body hash, memory id, cue, memref or memory text is ever logged.
    """

    if not isinstance(report, dict):
        return

    metrics = report.get("metrics")

    logger.info(
        "[gateway.context_memory_authority_migration] %s",
        json.dumps(
            {
                "mode": report.get("mode"),
                "stored": report.get("stored"),
                "decision": report.get("decision"),
                "reason": report.get("reason"),
                "source_unified_revision": (
                    report.get(
                        "source_unified_revision"
                    )
                ),
                "frontend_direct_mouth_active": (
                    report.get(
                        "frontend_direct_mouth_active"
                    )
                ),
                "ob_live_cue_mouth_active": (
                    report.get(
                        "ob_live_cue_mouth_active"
                    )
                ),
                "dual_mouth_active": (
                    report.get("dual_mouth_active")
                ),
                "contract_coverage_status": (
                    report.get(
                        "contract_coverage_status"
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
