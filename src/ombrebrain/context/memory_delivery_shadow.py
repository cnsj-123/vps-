from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import deque
from typing import Any

from ombrebrain.context.memory_source_evidence import (
    GRANULARITY_ATTACHMENT,
    GRANULARITY_CANDIDATE,
    SCOPE_CURRENT,
    SCOPE_HISTORY,
    SCOPE_RETRIEVAL,
    SOURCE_FRONTEND,
    SOURCE_OB,
    is_bound_to_request,
)
from ombrebrain.context.memory_lifecycle_event import (
    parse_iso_utc,
)
from ombrebrain.context.recall_types import (
    atomic_write,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    now_iso,
    state_root,
)
from ombrebrain.context.retrieval_decision_shadow import (
    candidate_fingerprint,
)
from ombrebrain.context.validators.freshness import (
    is_valid_revision,
)


# Unified Memory Delivery Plan Shadow v0 -- SHADOW ONLY.
#
# The first attempt at a UNIFIED MEMORY DELIVERY representation that
# FEDERATES the sources without collapsing them:
#
#      source-native memory evidence
#          -> Source-Neutral Evidence Adapter     (memory_source_evidence)
#              -> request-local Memory Source Evidence Snapshot
#                  -> Unified Memory Delivery Plan Shadow   (this module)
#                      -> read-only aggregate reporter
#
# This is the "unify the delivery ACTION, keep the evidence federated"
# layer. It is NOT a new retrieval algorithm, NOT a new Memory
# Authority, NOT a storage unification and NOT a live delivery
# decision.
#
# It answers ONE bounded question, and only that question:
#
#     "For THIS request, which memory units ALREADY have a Factual
#      Direct Delivery channel, which would have to pass the existing
#      eligibility -> cue -> Recall delivery path, which are RETRIEVED
#      BUT NOT SURFACED, and which can we NOT orthogonally know?"
#
# The Delivery Plan is a SHADOW SHAPE only:
#   - ``mode`` is ``shadow_only``;
#   - it is a PROPOSAL with NO live authority (``live_authority=false``,
#     ``authority=proposal_only``);
#   - it never kills / rewrites / reorders / merges a memory, never
#     invokes a Flash, never performs or alters a Recall, never
#     reinforces, never writes a memory, never picks the actual injected
#     text and never changes the provider request;
#   - it never re-runs retrieval / embedding / rerank; the candidate
#     set is purely the Source Evidence Snapshot;
#   - the base Surfacing Policy is computed ONCE upstream by the
#     coordinator and passed in. This module NEVER evaluates the
#     Surfacing Policy itself.
#
# The Delivery Plan is FEDERATED, source-specific and exact. It does
# NOT claim one universal "memory" object: a Frontend attachment unit
# and an OB candidate unit keep their own granularity and their own
# channel. It expresses three orthogonal, independently KNOWN tri-state
# facts instead of one ambiguous state enum:
#
#   currently_model_visible : is the unit's content in the live request
#                             right now?          (tri-state)
#   immediate_delivery_mode : what channel delivers it RIGHT NOW?
#                             direct_data / flash_cue / drop / unknown
#   full_content_access_mode: if more of the unit's FULL content is ever
#                             needed, how would the model get it?
#                             already_present / recall_required / none /
#                             unknown
#
# Tri-state honesty: an ELIGIBLE / INELIGIBLE verdict can only be made
# when the base Surfacing Policy is validly bound to THIS conversation
# and THIS Unified revision. A missing / malformed / stale / foreign
# policy, or an unknown decision enum, is UNKNOWN -- never a silent
# "not eligible" and never a silent "eligible". It also never guesses
# about a Flash outcome or a Recall outcome.
#
# Privacy: the persisted artifact and any log line contain only counts,
# booleans, enums, revisions and sha256 digests. No memory text, cue,
# body, raw attachment, memory id, memref / useref or candidate
# fingerprint is ever emitted or persisted.

VERSION = "memory-delivery-plan-shadow.v1"
MODE = "shadow_only"

_POLICY_VERSION = "memory-surfacing-policy.v1"
_POLICY_MODE = "shadow_only"
_ALLOW_DECISION = "allow_shadow"
_NO_SURFACE_DECISION = "no_surface"

# The CLOSED memory-surfacing-policy.v1 decision enum. Any other value
# is NOT valid evidence and must never become "ineligible".
_KNOWN_BASE_POLICY_DECISIONS = (
    _ALLOW_DECISION,
    _NO_SURFACE_DECISION,
)

# The plan stage this v0 expresses. The base Eligibility stays exactly
# the existing PRE-LIFECYCLE memory-surfacing-policy.v1 input; the
# Lifecycle Gate stays a SEPARATE, composable signal that is NOT
# consumed here.
_POLICY_STAGE = "base_surfacing_pre_lifecycle"

# This plan PROPOSES do-not-send rows outside the existing channel; it
# has no live authority.
_AUTHORITY = "proposal_only"

# Hard bound on persisted per-request delivery rows.
MAX_DELIVERY_ROWS = 64

_IMMEDIATE_DIRECT_DATA = "direct_data"
_IMMEDIATE_FLASH_CUE = "flash_cue"
_IMMEDIATE_DROP = "drop"
_IMMEDIATE_UNKNOWN = "unknown"

_KNOWN_IMMEDIATE_DELIVERY_MODES = (
    _IMMEDIATE_DIRECT_DATA,
    _IMMEDIATE_FLASH_CUE,
    _IMMEDIATE_DROP,
    _IMMEDIATE_UNKNOWN,
)

_FULL_ALREADY_PRESENT = "already_present"
_FULL_RECALL_REQUIRED = "recall_required"
_FULL_NONE = "none"
_FULL_UNKNOWN = "unknown"

_KNOWN_FULL_ACCESS_MODES = (
    _FULL_ALREADY_PRESENT,
    _FULL_RECALL_REQUIRED,
    _FULL_NONE,
    _FULL_UNKNOWN,
)

# Closed control-basis / reason vocabularies. A row whose basis / reason
# is outside these sets is malformed evidence, never a new state.
_BASIS_FRONTEND_DIRECT = (
    "existing_frontend_direct_baseline"
)
_BASIS_POLICY = "base_surfacing_policy"
_BASIS_POLICY_UNAVAILABLE = (
    "base_surfacing_unavailable"
)

_KNOWN_CONTROL_BASES = (
    _BASIS_FRONTEND_DIRECT,
    _BASIS_POLICY,
    _BASIS_POLICY_UNAVAILABLE,
)

_REASON_FRONTEND_VISIBLE = (
    "frontend_attachment_already_model_visible"
)
_REASON_ELIGIBLE = "base_surfacing_eligible"
_REASON_INELIGIBLE = "base_surfacing_ineligible"
_REASON_POLICY_UNAVAILABLE = (
    "base_surfacing_policy_unavailable"
)

_KNOWN_REASONS = (
    _REASON_FRONTEND_VISIBLE,
    _REASON_ELIGIBLE,
    _REASON_INELIGIBLE,
    _REASON_POLICY_UNAVAILABLE,
)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

logger = logging.getLogger(__name__)


def _path(
    conversation_id: str,
    cognitive_request_id: str,
):
    return (
        state_root()
        / "memory_delivery_plan"
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


def _evidence_ref(unit: Any) -> str:
    """Stable, occurrence-safe, request-local evidence reference.

    ``sha256(source + kind + occurrence + content_digest)``. It is NOT
    a memory id, NOT a content hash alone and NOT a global identity:
    duplicate content occurrences stay four distinct occurrences with
    four distinct refs. Only the digest is emitted.
    """

    payload = "\0".join(
        (
            str(unit.source),
            str(unit.kind),
            str(unit.request_occurrence),
            str(unit.content_digest),
        )
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _policy_binding(
    base_policy: Any,
    *,
    conversation_id: Any,
    expected_revision: Any,
) -> tuple[bool, list | None]:
    """Strictly bound base Surfacing Policy eligibility.

    Returns ``(available, eligible_list)``. ``available`` is False
    whenever the policy is missing, malformed, from another
    conversation, bound to another Unified revision, carries a decision
    OUTSIDE the closed v1 enum, or carries an eligible shape that does
    not match its decision -- in which case eligibility is UNKNOWN
    (``eligible_list`` is None) and must never be inferred as
    ``not eligible``. When the policy IS validly bound, ``eligible_list``
    is the real eligible list: a list for ``allow_shadow``, and an
    EXACTLY empty list for ``no_surface``, so only a genuine
    ``no_surface`` means "ineligible".
    """

    if not isinstance(base_policy, dict):
        return False, None

    if (
        base_policy.get("version") != _POLICY_VERSION
        or base_policy.get("mode") != _POLICY_MODE
    ):
        return False, None

    if (
        base_policy.get("conversation_id")
        != conversation_id
    ):
        return False, None

    if (
        base_policy.get("source_unified_revision")
        != expected_revision
    ):
        return False, None

    decision = base_policy.get("decision")

    if (
        not isinstance(decision, str)
        or decision
        not in _KNOWN_BASE_POLICY_DECISIONS
    ):
        return False, None

    eligible = base_policy.get("eligible")

    if decision == _ALLOW_DECISION:
        if not isinstance(eligible, list):
            return False, None

        return True, eligible

    # decision == no_surface: the real report ALWAYS owns an explicit
    # ``"eligible": []`` field. A missing / None / non-empty eligible
    # set is malformed evidence, never a trustworthy "ineligible".
    if not isinstance(eligible, list) or eligible:
        return False, None

    return True, []


def _eligible_queues(
    eligible_list: Any,
) -> dict[str, deque]:
    """candidate_fingerprint -> FIFO of ranks, in eligible-list order.

    Occurrence-safe: two distinct candidates with identical
    fingerprints are matched by FIFO position, never collapsed into
    one, so duplicate OB candidates cannot silently share a rank.
    """

    queues: dict[str, deque] = {}

    if not isinstance(eligible_list, list):
        return queues

    for rank, item in enumerate(eligible_list):
        fingerprint = candidate_fingerprint(item)

        queues.setdefault(
            fingerprint, deque()
        ).append(rank)

    return queues


def _match_eligible_rank(
    unit: Any,
    queues: dict[str, deque],
) -> int | None:
    """Occurrence-safe eligibility match for one OB unit."""

    fingerprint = unit.candidate_fingerprint

    if not isinstance(fingerprint, str):
        return None

    queue = queues.get(fingerprint)

    if not queue:
        return None

    return queue.popleft()


def _row(
    unit: Any,
    *,
    policy_binding_available: bool | None,
    base_surfacing_eligible: bool | None,
    base_eligible_rank: int | None,
    immediate_delivery_mode: str,
    full_content_access_mode: str,
    control_basis: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "evidence_ref": _evidence_ref(unit),
        "source": unit.source,
        "kind": unit.kind,
        "granularity": unit.granularity,
        "source_scope": unit.source_scope,
        "source_index": unit.source_index,
        "source_occurrence": (
            unit.request_occurrence
        ),
        "content_digest": unit.content_digest,
        "normalized_digest": unit.normalized_digest,
        "content_chars": unit.content_chars,
        "content_estimated_tokens": (
            unit.content_estimated_tokens
        ),
        "currently_model_visible": (
            unit.currently_model_visible
        ),
        "retrieved": unit.retrieved,
        "policy_binding_available": (
            policy_binding_available
        ),
        "base_surfacing_eligible": (
            base_surfacing_eligible
        ),
        "base_eligible_rank": base_eligible_rank,
        "immediate_delivery_mode": (
            immediate_delivery_mode
        ),
        "full_content_access_mode": (
            full_content_access_mode
        ),
        "control_basis": control_basis,
        "reason": reason,
    }


def _frontend_row(unit: Any) -> dict[str, Any]:
    """A Frontend attachment unit: already in the request.

    It is NOT retrieved, and its content is already fully available, so
    its delivery is a Factual Direct Delivery baseline -- not a Flash
    cue and not a Recall. It is never eligible / ineligible (that
    verdict is the OB candidate channel), so the base policy fields are
    an explicit ``null``, never ``false``.
    """

    return _row(
        unit,
        policy_binding_available=None,
        base_surfacing_eligible=None,
        base_eligible_rank=None,
        immediate_delivery_mode=(
            _IMMEDIATE_DIRECT_DATA
        ),
        full_content_access_mode=(
            _FULL_ALREADY_PRESENT
        ),
        control_basis=_BASIS_FRONTEND_DIRECT,
        reason=_REASON_FRONTEND_VISIBLE,
    )


def _ob_row(
    unit: Any,
    *,
    policy_available: bool,
    queues: dict[str, deque],
) -> dict[str, Any]:
    """An OB candidate unit: retrieved, not yet model-visible.

    When the base policy is validly bound, an eligible candidate would
    have a bounded Flash cue and would need an explicit Recall to fully
    load; an ineligible candidate would be dropped. When the policy
    cannot be validly bound the verdict is UNKNOWN, never a silent drop
    -- and the OB candidate is never a direct-data channel.
    """

    if policy_available:
        rank = _match_eligible_rank(unit, queues)

        if rank is None:
            return _row(
                unit,
                policy_binding_available=True,
                base_surfacing_eligible=False,
                base_eligible_rank=None,
                immediate_delivery_mode=(
                    _IMMEDIATE_DROP
                ),
                full_content_access_mode=(
                    _FULL_NONE
                ),
                control_basis=_BASIS_POLICY,
                reason=_REASON_INELIGIBLE,
            )

        return _row(
            unit,
            policy_binding_available=True,
            base_surfacing_eligible=True,
            base_eligible_rank=rank,
            immediate_delivery_mode=(
                _IMMEDIATE_FLASH_CUE
            ),
            full_content_access_mode=(
                _FULL_RECALL_REQUIRED
            ),
            control_basis=_BASIS_POLICY,
            reason=_REASON_ELIGIBLE,
        )

    return _row(
        unit,
        policy_binding_available=False,
        base_surfacing_eligible=None,
        base_eligible_rank=None,
        immediate_delivery_mode=_IMMEDIATE_UNKNOWN,
        full_content_access_mode=_FULL_UNKNOWN,
        control_basis=_BASIS_POLICY_UNAVAILABLE,
        reason=_REASON_POLICY_UNAVAILABLE,
    )


def _metrics(
    rows: list[dict[str, Any]],
    *,
    unit_count: int,
    persisted_count: int,
    truncated_count: int,
) -> dict[str, Any]:
    def is_source(row, source):
        return row.get("source") == source

    def immediate(row):
        return row.get("immediate_delivery_mode")

    def full(row):
        return row.get("full_content_access_mode")

    frontend_rows = [
        r for r in rows if is_source(r, SOURCE_FRONTEND)
    ]
    ob_rows = [
        r for r in rows if is_source(r, SOURCE_OB)
    ]

    def count(predicate):
        return sum(
            1 for row in rows if predicate(row)
        )

    return {
        "unit_count": unit_count,
        "persisted_unit_count": persisted_count,
        "truncated_unit_count": truncated_count,
        "frontend_unit_count": len(frontend_rows),
        "ob_unit_count": len(ob_rows),
        # A source unit count that is present at all is, by
        # construction, observation-available; these mirror the source
        # split for the reporter and stay 0 when the source is
        # unavailable entirely.
        "frontend_available_count": len(frontend_rows),
        "ob_available_count": len(ob_rows),
        "current_model_visible_count": count(
            lambda r: r.get(
                "currently_model_visible"
            )
            is True
        ),
        "retrieved_count": count(
            lambda r: r.get("retrieved") is True
        ),
        # The three immediate channels partition the rows that HAVE a
        # known channel; ``unknown`` rows carry no known channel, so
        # they are counted separately and never folded into ``drop``.
        "direct_data_count": count(
            lambda r: immediate(r)
            == _IMMEDIATE_DIRECT_DATA
        ),
        "flash_cue_count": count(
            lambda r: immediate(r)
            == _IMMEDIATE_FLASH_CUE
        ),
        "drop_count": count(
            lambda r: immediate(r) == _IMMEDIATE_DROP
        ),
        "unknown_count": count(
            lambda r: immediate(r)
            == _IMMEDIATE_UNKNOWN
        ),
        "full_content_already_present_count":
            count(
                lambda r: full(r)
                == _FULL_ALREADY_PRESENT
            ),
        "full_content_recall_required_count":
            count(
                lambda r: full(r)
                == _FULL_RECALL_REQUIRED
            ),
        # Cross-tab channelling: the Frontend side is direct data, the
        # OB side is cue / drop / unknown.
        "frontend_direct_data_count": count(
            lambda r: is_source(r, SOURCE_FRONTEND)
            and immediate(r)
            == _IMMEDIATE_DIRECT_DATA
        ),
        "ob_flash_cue_count": count(
            lambda r: is_source(r, SOURCE_OB)
            and immediate(r)
            == _IMMEDIATE_FLASH_CUE
        ),
        "ob_drop_count": count(
            lambda r: is_source(r, SOURCE_OB)
            and immediate(r)
            == _IMMEDIATE_DROP
        ),
        "ob_unknown_count": count(
            lambda r: is_source(r, SOURCE_OB)
            and immediate(r)
            == _IMMEDIATE_UNKNOWN
        ),
        "base_policy_available_ob_count":
            count(
                lambda r: is_source(r, SOURCE_OB)
                and r.get(
                    "policy_binding_available"
                )
                is True
            ),
        "base_policy_unavailable_ob_count":
            count(
                lambda r: is_source(r, SOURCE_OB)
                and r.get(
                    "policy_binding_available"
                )
                is False
            ),
        "combined_content_chars": sum(
            r.get("content_chars", 0)
            for r in rows
        ),
        "combined_content_estimated_tokens":
            sum(
                r.get(
                    "content_estimated_tokens", 0
                )
                for r in rows
            ),
        "current_model_visible_chars": sum(
            r.get("content_chars", 0)
            for r in rows
            if r.get(
                "currently_model_visible"
            )
            is True
        ),
        "current_model_visible_estimated_tokens":
            sum(
                r.get(
                    "content_estimated_tokens", 0
                )
                for r in rows
                if r.get(
                    "currently_model_visible"
                )
                is True
            ),
    }


def build_memory_delivery_plan_artifact(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    source_unified_revision: int,
    source_evidence_snapshot: Any,
    base_policy: Any,
) -> dict[str, Any]:
    """Pure builder for one privacy-safe Delivery Plan Shadow artifact.

    No IO, no logging, deterministic. The source evidence is the shared
    request-local snapshot; the base policy is the existing
    pre-lifecycle ``memory-surfacing-policy.v1`` report computed ONCE
    upstream. This function never evaluates the Surfacing Policy.
    """

    units = tuple(
        getattr(
            source_evidence_snapshot,
            "units",
            (),
        )
        or ()
    )

    frontend_observation_available = bool(
        getattr(
            source_evidence_snapshot,
            "frontend_observation_available",
            False,
        )
    )

    ob_observation_available = bool(
        getattr(
            source_evidence_snapshot,
            "ob_observation_available",
            False,
        )
    )

    policy_available, eligible_list = (
        _policy_binding(
            base_policy,
            conversation_id=conversation_id,
            expected_revision=(
                source_unified_revision
            ),
        )
    )

    queues = (
        _eligible_queues(eligible_list)
        if policy_available
        else {}
    )

    rows: list[dict[str, Any]] = []

    for unit in units:
        if unit.source == SOURCE_FRONTEND:
            rows.append(_frontend_row(unit))
        elif unit.source == SOURCE_OB:
            rows.append(
                _ob_row(
                    unit,
                    policy_available=(
                        policy_available
                    ),
                    queues=queues,
                )
            )

    total = len(rows)

    persisted = rows[:MAX_DELIVERY_ROWS]

    truncated_count = total - len(persisted)

    return {
        "version": VERSION,
        "mode": MODE,
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "source_unified_revision": (
            source_unified_revision
        ),
        "created_at": now_iso(),
        "authority": _AUTHORITY,
        "live_authority": False,
        "policy_stage": _POLICY_STAGE,
        # Producer facts: this plan neither invokes a Flash nor consumes
        # the Lifecycle Gate, so reviewers can see the base Eligibility
        # is the pre-lifecycle policy input only.
        "flash_required": False,
        "lifecycle_required": False,
        "frontend_observation_available": (
            frontend_observation_available
        ),
        "ob_observation_available": (
            ob_observation_available
        ),
        "base_policy_available": (
            policy_available
        ),
        "rows": persisted,
        "metrics": _metrics(
            persisted,
            unit_count=total,
            persisted_count=len(persisted),
            truncated_count=truncated_count,
        ),
        "rows_truncated": truncated_count > 0,
        "rows_truncated_count": truncated_count,
    }


def _valid_tristate(value: Any) -> bool:
    return value is None or isinstance(value, bool)


def _valid_row(row: Any) -> bool:
    """One structurally + semantically valid delivery row.

    A Frontend row must never look like an OB row and vice versa (see
    the source-specific invariants below), and a row's channel must
    match its eligibility verdict exactly.
    """

    if not isinstance(row, dict):
        return False

    if not _is_sha256(row.get("evidence_ref")):
        return False

    source = row.get("source")

    if source not in (SOURCE_FRONTEND, SOURCE_OB):
        return False

    if not _is_sha256(row.get("content_digest")):
        return False

    if not _is_sha256(row.get("normalized_digest")):
        return False

    if not _is_nonneg_int(row.get("content_chars")):
        return False

    if not _is_nonneg_int(
        row.get("content_estimated_tokens")
    ):
        return False

    if not _is_bool(row.get("currently_model_visible")):
        return False

    if not _is_bool(row.get("retrieved")):
        return False

    if not _valid_tristate(
        row.get("policy_binding_available")
    ):
        return False

    if not _valid_tristate(
        row.get("base_surfacing_eligible")
    ):
        return False

    rank = row.get("base_eligible_rank")

    if rank is not None and not _is_nonneg_int(rank):
        return False

    if row.get("immediate_delivery_mode") not in (
        _KNOWN_IMMEDIATE_DELIVERY_MODES
    ):
        return False

    if row.get("full_content_access_mode") not in (
        _KNOWN_FULL_ACCESS_MODES
    ):
        return False

    if row.get("control_basis") not in (
        _KNOWN_CONTROL_BASES
    ):
        return False

    if row.get("reason") not in _KNOWN_REASONS:
        return False

    if not _is_nonneg_int(row.get("source_index")):
        return False

    if not _is_nonneg_int(
        row.get("source_occurrence")
    ):
        return False

    if source == SOURCE_FRONTEND:
        return _valid_frontend_row(row)

    return _valid_ob_row(row)


def _valid_frontend_row(row: dict) -> bool:
    """A Frontend row is a direct-data baseline, never a cue / drop and
    never an eligible / ineligible verdict."""

    if row.get("granularity") != (
        GRANULARITY_ATTACHMENT
    ):
        return False

    if row.get("kind") != "frontend_attachment":
        return False

    if row.get("source_scope") not in (
        SCOPE_CURRENT,
        SCOPE_HISTORY,
    ):
        return False

    if row.get("currently_model_visible") is not True:
        return False

    if row.get("retrieved") is not False:
        return False

    if row.get("policy_binding_available") is not None:
        return False

    if row.get("base_surfacing_eligible") is not None:
        return False

    if row.get("base_eligible_rank") is not None:
        return False

    if row.get("immediate_delivery_mode") != (
        _IMMEDIATE_DIRECT_DATA
    ):
        return False

    if row.get("full_content_access_mode") != (
        _FULL_ALREADY_PRESENT
    ):
        return False

    if row.get("control_basis") != (
        _BASIS_FRONTEND_DIRECT
    ):
        return False

    if row.get("reason") != (
        _REASON_FRONTEND_VISIBLE
    ):
        return False

    return True


def _valid_ob_row(row: dict) -> bool:
    """An OB row is retrieved, never model-visible and never direct
    data. Eligible -> cue; ineligible -> drop; unbound policy -> unknown.
    """

    if row.get("granularity") != (
        GRANULARITY_CANDIDATE
    ):
        return False

    if row.get("kind") != "ob_candidate":
        return False

    if row.get("source_scope") != SCOPE_RETRIEVAL:
        return False

    if row.get("currently_model_visible") is not False:
        return False

    if row.get("retrieved") is not True:
        return False

    binding = row.get("policy_binding_available")

    eligible = row.get("base_surfacing_eligible")

    mode = row.get("immediate_delivery_mode")

    full = row.get("full_content_access_mode")

    basis = row.get("control_basis")

    reason = row.get("reason")

    rank = row.get("base_eligible_rank")

    if binding is True:
        if eligible is True:
            return (
                _is_nonneg_int(rank)
                and mode == _IMMEDIATE_FLASH_CUE
                and full == _FULL_RECALL_REQUIRED
                and basis == _BASIS_POLICY
                and reason == _REASON_ELIGIBLE
            )

        if eligible is False:
            return (
                rank is None
                and mode == _IMMEDIATE_DROP
                and full == _FULL_NONE
                and basis == _BASIS_POLICY
                and reason == _REASON_INELIGIBLE
            )

        return False

    if binding is False:
        return (
            eligible is None
            and rank is None
            and mode == _IMMEDIATE_UNKNOWN
            and full == _FULL_UNKNOWN
            and basis
            == _BASIS_POLICY_UNAVAILABLE
            and reason
            == _REASON_POLICY_UNAVAILABLE
        )

    # An OB row must NEVER have an unknown binding tri-state: the OB
    # side either had a usable policy or it did not.
    return False


def is_valid_memory_delivery_plan_artifact(
    artifact: Any,
) -> bool:
    """The single authoritative v0 delivery-plan validator.

    A complete, structurally sound ``memory-delivery-plan-shadow.v1``
    artifact is valid; anything else (wrong version / mode, malformed
    identity, an incomplete source set, a locally incoherent source
    state, a persisted metric that disagrees with the rows, or one
    malformed row among otherwise valid rows) is INVALID as a whole.
    Future / newer versions are unsupported and therefore invalid --
    they are never partially interpreted.
    """

    if not isinstance(artifact, dict):
        return False

    if artifact.get("version") != VERSION:
        return False

    if artifact.get("mode") != MODE:
        return False

    if artifact.get("authority") != _AUTHORITY:
        return False

    if artifact.get("live_authority") is not False:
        return False

    if artifact.get("policy_stage") != _POLICY_STAGE:
        return False

    if artifact.get("flash_required") is not False:
        return False

    if artifact.get("lifecycle_required") is not False:
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

    # Availability facts must be strict booleans.
    for name in (
        "frontend_observation_available",
        "ob_observation_available",
        "base_policy_available",
    ):
        if not _is_bool(artifact.get(name)):
            return False

    rows = artifact.get("rows")

    if not isinstance(rows, list):
        return False

    if len(rows) > MAX_DELIVERY_ROWS:
        return False

    for row in rows:
        if not _valid_row(row):
            return False

    # Source-specific invariants. A source that is NOT
    # observation-available contributes no rows at all -- availability
    # is never silently converted into "zero items".
    if not artifact[
        "frontend_observation_available"
    ] and any(
        r.get("source") == SOURCE_FRONTEND
        for r in rows
    ):
        return False

    if not artifact["ob_observation_available"] and any(
        r.get("source") == SOURCE_OB
        for r in rows
    ):
        return False

    # Every OB row's policy binding fact must equal the artifact-level
    # base-policy fact (there is no per-row disagreement).
    for row in rows:
        if row.get("source") == SOURCE_OB:
            if (
                row.get("policy_binding_available")
                is not artifact["base_policy_available"]
            ):
                return False

    truncated = artifact.get("rows_truncated")

    if not _is_bool(truncated):
        return False

    truncated_count = artifact.get(
        "rows_truncated_count"
    )

    if not _is_nonneg_int(truncated_count):
        return False

    if truncated is not (truncated_count > 0):
        return False

    total = artifact.get("metrics", {}).get(
        "unit_count"
    ) if isinstance(
        artifact.get("metrics"), dict
    ) else None

    if not _is_nonneg_int(total):
        return False

    if total != len(rows) + truncated_count:
        return False

    metrics = artifact.get("metrics")

    if not isinstance(metrics, dict):
        return False

    # These artifacts are research evidence: a hand-falsified counter is
    # not trustworthy. Recompute the request metrics from the validated
    # rows with the SAME builder helper and require agreement.
    if metrics != _metrics(
        rows,
        unit_count=total,
        persisted_count=len(rows),
        truncated_count=truncated_count,
    ):
        return False

    return True


def update_memory_delivery_plan(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    source_evidence_snapshot: Any,
    expected_unified_revision: Any,
    base_policy: Any,
) -> dict[str, Any]:
    """Observe + persist one request's Unified Memory Delivery Plan.

    Never raises. Returns a privacy-safe report. ``stored`` is False
    (with a reason) whenever a truthful plan cannot be made or the
    artifact cannot be written -- delivery-plan failure is missing
    delivery data and NOTHING else.

    It consumes THIS request's request-local Source Evidence Snapshot
    only and never re-reads a conversation-global Unified file. The
    base Surfacing Policy is passed in already computed; this function
    never evaluates it. The report carries the built artifact under
    ``plan`` for the caller to bind onto its request-local workspace;
    the artifact is safe (no memory text / id / fingerprint).

    A foreign / mismatched evidence snapshot is never accepted: it must
    be bound to THIS conversation, THIS cognitive request and THIS
    source Unified revision.
    """

    if not is_valid_conversation_id(
        conversation_id
    ):
        return {
            "stored": False,
            "mode": MODE,
            "decision": "no_record",
            "reason": "invalid_conversation_id",
        }

    if not is_valid_cognitive_request_id(
        cognitive_request_id
    ):
        return {
            "stored": False,
            "mode": MODE,
            "decision": "no_record",
            "reason": "invalid_cognitive_request_id",
        }

    if not is_valid_revision(
        expected_unified_revision
    ):
        return {
            "stored": False,
            "mode": MODE,
            "decision": "no_record",
            "reason":
                "invalid_expected_unified_revision",
        }

    if not is_bound_to_request(
        source_evidence_snapshot,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        expected_unified_revision=(
            expected_unified_revision
        ),
    ):
        return {
            "stored": False,
            "mode": MODE,
            "decision": "no_record",
            "reason": "evidence_not_bound",
        }

    artifact = build_memory_delivery_plan_artifact(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        source_unified_revision=(
            expected_unified_revision
        ),
        source_evidence_snapshot=(
            source_evidence_snapshot
        ),
        base_policy=base_policy,
    )

    # Never persist malformed evidence: validate with the SAME v0
    # validator the reporter uses. Failure stays fail-open relative to
    # the live request.
    if not is_valid_memory_delivery_plan_artifact(
        artifact
    ):
        logger.warning(
            "[gateway.context_memory_delivery] "
            "reason=invalid_delivery_plan_artifact "
            "fail_open=true"
        )

        return {
            "stored": False,
            "mode": MODE,
            "decision": "no_record",
            "reason": "invalid_delivery_plan_artifact",
        }

    try:
        atomic_write(
            _path(
                conversation_id,
                cognitive_request_id,
            ),
            artifact,
        )
    except Exception as exc:
        # Fail-open: delivery data is simply missing.
        logger.warning(
            "[gateway.context_memory_delivery] "
            "store_failed=%s fail_open=true",
            type(exc).__name__,
        )

        return {
            "stored": False,
            "mode": MODE,
            "decision": "no_record",
            "reason": "persist_failed",
        }

    return {
        "stored": True,
        "mode": MODE,
        "decision": "planned",
        "reason": "delivery_planned",
        "source_unified_revision": (
            expected_unified_revision
        ),
        "frontend_observation_available":
            artifact[
                "frontend_observation_available"
            ],
        "ob_observation_available":
            artifact["ob_observation_available"],
        "base_policy_available": artifact[
            "base_policy_available"
        ],
        "metrics": artifact["metrics"],
        "plan": artifact,
    }


def log_memory_delivery_report(
    report: dict[str, Any],
) -> None:
    """Privacy-safe telemetry: aggregate counts and reasons only.

    No conversation id, cognitive request id, evidence ref, digest,
    memory id, fingerprint, cue, body or memory text is ever logged.
    """

    metrics = (
        report.get("metrics")
        if isinstance(report, dict)
        else None
    )

    logger.info(
        "[gateway.context_memory_delivery] %s",
        json.dumps(
            {
                "mode": (
                    report.get("mode")
                    if isinstance(report, dict)
                    else None
                ),
                "stored": (
                    report.get("stored")
                    if isinstance(report, dict)
                    else None
                ),
                "decision": (
                    report.get("decision")
                    if isinstance(report, dict)
                    else None
                ),
                "reason": (
                    report.get("reason")
                    if isinstance(report, dict)
                    else None
                ),
                "frontend_observation_available": (
                    report.get(
                        "frontend_observation_available"
                    )
                    if isinstance(report, dict)
                    else None
                ),
                "ob_observation_available": (
                    report.get(
                        "ob_observation_available"
                    )
                    if isinstance(report, dict)
                    else None
                ),
                "base_policy_available": (
                    report.get(
                        "base_policy_available"
                    )
                    if isinstance(report, dict)
                    else None
                ),
                "metrics": (
                    {
                        key: metrics.get(key)
                        for key in (
                            "unit_count",
                            "frontend_unit_count",
                            "ob_unit_count",
                            "direct_data_count",
                            "flash_cue_count",
                            "drop_count",
                            "unknown_count",
                        )
                    }
                    if isinstance(metrics, dict)
                    else None
                ),
            },
            sort_keys=True,
        ),
    )
