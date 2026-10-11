from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ombrebrain.context.memory_source_evidence import (
    SOURCE_FRONTEND,
    SOURCE_OB,
    build_memory_source_evidence_snapshot,
    is_bound_to_request,
)
from ombrebrain.context.recall_types import (
    atomic_write,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    normalize_key,
    now_iso,
    state_root,
)
from ombrebrain.context.retrieval_decision_shadow import (
    candidate_text,
)
from ombrebrain.context.validators.freshness import (
    is_valid_revision,
)
from ombrebrain.gateway.canonicalizer import (
    extract_frontend_memory_segments,
)


# Frontend Memory Arbitration Shadow v1 -- shadow only.
#
# This module measures an access asymmetry that already exists in the
# baseline; it does NOT change it:
#
#   Frontend Memory  (relevant_memories.txt)
#       full content is already model-visible in the request,
#       preserved by cache rewriting, excluded from Conversation
#       Snapshot USER_TEXT-derived semantic state;
#
#   OB / cognitive memory
#       retrieved -> eligibility -> Memory Flash cue -> model decides
#       Recall -> full memory loaded only on an explicit Recall.
#
# Observation shape != future canonical memory contract. v1 measures
# two DIFFERENT units and never pretends they are the same thing:
#
#   frontend: one explicit relevant_memories.txt ATTACHMENT (bundle).
#       The repository has no proven item-level schema for the inner
#       body, so an attachment is never called one memory candidate.
#   ob: one Unified-admitted memory CANDIDATE (retrieved side).
#
# Since v0 of the Unified Memory Delivery plan, the observations are
# derived from the shared request-local Source-Neutral Memory Evidence
# Snapshot (``memory_source_evidence``). This module no longer reparses
# the forwarded body / snapshot on its own: it consumes the SAME
# evidence the Delivery Plan Shadow consumes. It remains, however, an
# OBSERVATION-ONLY measurement -- it is NOT the delivery authority and
# its observation shape is NOT the Delivery Plan.
#
# It answers, from REAL requests: how many Frontend attachments exist,
# how much model-visible context they occupy, whether they are current
# or historical, how many OB candidates the same request produced, how
# much the two sources overlap, and how much of the request's
# model-visible memory footprint comes from Frontend Memory.
#
# Two size concepts are kept strictly separate and are never collapsed:
#
#   body content size      -- the extracted inner attachment text or an
#                             OB candidate's content;
#   model-visible size     -- the exact raw Frontend attachment envelope
#                             (metadata + body) that is present in the
#                             request. OB candidate content is normally
#                             NOT model-visible, so that is 0.
#
# It is observation-only and fail-open. It never removes, rewrites or
# converts Frontend Memory, never starts a Flash, never performs or
# alters a Recall, never records usage, never reinforces anything and
# never changes the provider request. Being present in the request is
# NOT recalled/used/reinforced: those later stages are simply not
# modelled here and are never fabricated.
#
# Overlap is deliberately conservative and deterministic v1:
#   - exact content equivalence after safe extraction;
#   - normalized-text equivalence (whitespace collapse + casefold).
# No embedding, no LLM judge, no reranker, no learned score and no
# semantic threshold is used. The overlap compares an extracted
# Frontend attachment BODY against an OB candidate's content; it does
# NOT prove the Frontend system holds exactly one memory or that the
# memories are semantically the same, and it never authorises an
# automatic merge / delete.
#
# Privacy: the persisted artifact and any log line contain only
# counts, booleans, enums, revisions and sha256 digests. No Frontend
# Memory text, raw attachment envelope, OB memory text, cue, user
# text, memref, useref or raw memory id is ever emitted.

_VERSION = "memory-source-arbitration-shadow.v1"
_MODE = "shadow_only"

_KNOWN_SOURCES = (
    SOURCE_FRONTEND,
    SOURCE_OB,
)

_KIND_FRONTEND_ATTACHMENT = "frontend_attachment"
_KIND_OB_CANDIDATE = "ob_candidate"

_PLACEMENT_CURRENT = "current"
_PLACEMENT_HISTORY = "history"
_PLACEMENT_RETRIEVAL = "retrieval"

# Bound the persisted per-observation evidence. The observation pool is
# small (<=6 OB candidates by Unified budget plus the Frontend
# attachments), but this keeps the artifact size hard-bounded anyway.
_MAX_PERSISTED_OBSERVATIONS = 64


def _path(
    conversation_id: str,
    cognitive_request_id: str,
):
    return (
        state_root()
        / "memory_source_arbitration"
        / conversation_id
        / (cognitive_request_id + ".json")
    )


@dataclass(frozen=True)
class _Observation:
    """One source-neutral shadow observation.

    ``content_*`` describes extractable body content (comparable
    across sources). ``model_visible_*`` describes the exact text that
    the provider request actually carries: for a Frontend attachment
    that is the raw envelope, for an OB candidate it is 0 because OB
    content is not model-visible.

    All observed stage flags reflect real evidence: the frontend
    observation is ``currently_model_visible`` but not ``retrieved``;
    the OB observation is ``retrieved`` but not ``currently_model_visible``.
    No later stage is ever inferred.
    """

    source: str
    kind: str
    placement: str
    index: int
    content_digest: str
    normalized_digest: str
    content_chars: int
    content_estimated_tokens: int
    model_visible_chars: int
    model_visible_estimated_tokens: int
    currently_model_visible: bool
    retrieved: bool


def _digest(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _normalized_digest(text: str) -> str:
    return hashlib.sha256(
        normalize_key(text).encode("utf-8")
    ).hexdigest()


def _observations_from_units(
    units: Any,
) -> list[_Observation]:
    """Map source-neutral evidence units to v1 observations.

    Semantics are unchanged from the original v1 builder: a Frontend
    unit keeps its current/history placement and its model-visible raw
    envelope; an OB unit is retrieved but not model-visible. The
    model-visible footprint is reused STRAIGHT from the shared evidence
    unit (measured once by the adapter from the exact canonical
    envelope), so the arbitration measurement and the Delivery Plan
    Shadow can never disagree about the same occurrence.
    """

    observations: list[_Observation] = []

    if not isinstance(units, (list, tuple)):
        return observations

    for unit in units:
        if unit.source == SOURCE_FRONTEND:
            placement = unit.source_scope

            if placement not in (
                _PLACEMENT_CURRENT,
                _PLACEMENT_HISTORY,
            ):
                placement = _PLACEMENT_HISTORY

            observations.append(
                _Observation(
                    source=SOURCE_FRONTEND,
                    kind=_KIND_FRONTEND_ATTACHMENT,
                    placement=placement,
                    index=unit.source_index,
                    content_digest=unit.content_digest,
                    normalized_digest=(
                        unit.normalized_digest
                    ),
                    content_chars=unit.content_chars,
                    content_estimated_tokens=(
                        unit.content_estimated_tokens
                    ),
                    model_visible_chars=(
                        unit.model_visible_chars
                    ),
                    model_visible_estimated_tokens=(
                        unit.model_visible_estimated_tokens
                    ),
                    currently_model_visible=True,
                    retrieved=False,
                )
            )
            continue

        if unit.source == SOURCE_OB:
            observations.append(
                _Observation(
                    source=SOURCE_OB,
                    kind=_KIND_OB_CANDIDATE,
                    placement=_PLACEMENT_HISTORY,
                    index=unit.source_index,
                    content_digest=unit.content_digest,
                    normalized_digest=(
                        unit.normalized_digest
                    ),
                    content_chars=unit.content_chars,
                    content_estimated_tokens=(
                        unit.content_estimated_tokens
                    ),
                    model_visible_chars=(
                        unit.model_visible_chars
                    ),
                    model_visible_estimated_tokens=(
                        unit.model_visible_estimated_tokens
                    ),
                    currently_model_visible=False,
                    retrieved=True,
                )
            )

    return observations


def _observations_from_segments_and_memories(
    frontend_segments: Any,
    ob_memories: Any,
) -> list[_Observation]:
    """Compatibility path: build units from the raw canonical inputs.

    It reuses the SAME shared adapter unit builders the coordinator
    uses, so no second observation system exists.
    """

    from ombrebrain.context.memory_source_evidence import (
        frontend_evidence_units,
        ob_evidence_units,
    )

    units = (
        frontend_evidence_units(frontend_segments)
        + ob_evidence_units(ob_memories)
    )

    return _observations_from_units(units)


def _build_metrics(
    observations: list[_Observation],
) -> dict[str, Any]:
    frontend = [
        o
        for o in observations
        if o.source == SOURCE_FRONTEND
    ]
    ob = [
        o
        for o in observations
        if o.source == SOURCE_OB
    ]

    frontend_visible = [
        o
        for o in frontend
        if o.currently_model_visible
    ]

    # Cross-source exact / normalized overlap, counted as ordered
    # (frontend attachment, ob candidate) pairs. exact is a subset of
    # normalized by construction (byte equality implies normalized
    # equality). This compares an attachment BODY against OB candidate
    # content only -- it is not an item-level memory duplication claim.
    ob_by_exact = Counter(
        o.content_digest for o in ob
    )
    ob_by_normalized = Counter(
        o.normalized_digest for o in ob
    )

    exact_pairs = sum(
        ob_by_exact.get(o.content_digest, 0)
        for o in frontend
    )
    normalized_pairs = sum(
        ob_by_normalized.get(
            o.normalized_digest, 0
        )
        for o in frontend
    )

    # Groups of normalized-equivalent content in the combined pool
    # (size >= 2). Normalized-text equivalence only; this is not a
    # semantic-duplication count.
    normalized_counts = Counter(
        o.normalized_digest for o in observations
    )
    normalized_groups = sum(
        1
        for count in normalized_counts.values()
        if count >= 2
    )

    return {
        "frontend_attachment_count": len(frontend),
        "frontend_current_attachment_count": sum(
            1
            for o in frontend
            if o.placement == _PLACEMENT_CURRENT
        ),
        "frontend_history_attachment_count": sum(
            1
            for o in frontend
            if o.placement == _PLACEMENT_HISTORY
        ),
        # Frontend body content size (the extracted inner text).
        "frontend_body_chars": sum(
            o.content_chars for o in frontend
        ),
        "frontend_body_estimated_tokens": sum(
            o.content_estimated_tokens
            for o in frontend
        ),
        # Frontend actual model-visible footprint (exact raw
        # attachment envelope present in the request).
        "frontend_model_visible_attachment_count":
            len(frontend_visible),
        "frontend_model_visible_chars": sum(
            o.model_visible_chars
            for o in frontend_visible
        ),
        "frontend_model_visible_estimated_tokens":
            sum(
                o.model_visible_estimated_tokens
                for o in frontend_visible
            ),
        # OB retrieved candidate content size. This is NOT
        # model-visible today.
        "ob_candidate_count": len(ob),
        "ob_retrieved_candidate_content_chars":
            sum(o.content_chars for o in ob),
        "ob_retrieved_candidate_content_estimated_tokens":
            sum(
                o.content_estimated_tokens
                for o in ob
            ),
        # Overlap evidence, precise about what was compared.
        "frontend_bundle_ob_exact_pair_count":
            exact_pairs,
        "frontend_bundle_ob_normalized_pair_count":
            normalized_pairs,
        "normalized_equivalence_group_count":
            normalized_groups,
        "combined_observation_count":
            len(observations),
        "source_counts": {
            SOURCE_FRONTEND: len(frontend),
            SOURCE_OB: len(ob),
        },
    }


def _build_artifact_from_observations(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    source_unified_revision: int,
    observations: list[_Observation],
    ob_observation_available: bool,
) -> dict[str, Any]:
    metrics = _build_metrics(observations)

    persisted: list[dict[str, Any]] = []

    for observation in (
        observations[:_MAX_PERSISTED_OBSERVATIONS]
    ):
        persisted.append(
            {
                "source": observation.source,
                "kind": observation.kind,
                "placement": observation.placement,
                "index": observation.index,
                "content_digest":
                    observation.content_digest,
                "normalized_digest":
                    observation.normalized_digest,
                "content_chars":
                    observation.content_chars,
                "content_estimated_tokens":
                    observation.content_estimated_tokens,
                "model_visible_chars":
                    observation.model_visible_chars,
                "model_visible_estimated_tokens":
                    observation
                    .model_visible_estimated_tokens,
                "currently_model_visible":
                    observation.currently_model_visible,
                "retrieved": observation.retrieved,
            }
        )

    return {
        "version": _VERSION,
        "mode": _MODE,
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "source_unified_revision":
            source_unified_revision,
        "ob_observation_available":
            bool(ob_observation_available),
        "decision": "observed",
        "reason": "arbitration_observed",
        # The two sides are NOT the same unit and this v1 does not
        # claim a shared candidate contract.
        "observation_units": {
            SOURCE_FRONTEND: "attachment",
            SOURCE_OB: "candidate",
        },
        "overlap_scope": (
            "frontend_attachment_body"
            "_vs_ob_candidate_content"
        ),
        "overlap_method":
            "exact_then_normalized_text.v1",
        "normalization":
            "whitespace_collapse_casefold",
        "metrics": metrics,
        "observations": persisted,
        "observations_truncated": (
            len(observations)
            > _MAX_PERSISTED_OBSERVATIONS
        ),
    }


def build_arbitration_artifact(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    source_unified_revision: int,
    frontend_segments: Any,
    ob_memories: Any,
    ob_observation_available: bool,
) -> dict[str, Any]:
    """Pure builder for one privacy-safe arbitration artifact.

    No IO, no logging, deterministic. Frontend observations come from
    the canonical forwarded request; OB observations come from THIS
    request's own Unified memory section. Nothing is inferred beyond
    those two real inputs. The two sides are different observation
    units (attachment vs candidate) and the artifact says so instead of
    pretending a shared candidate contract already exists.

    It now derives its observations from the SHARED source-neutral
    evidence unit builders, so the arbitration measurement and the
    Delivery Plan Shadow observe the exact same source occurrences.
    """

    observations = (
        _observations_from_segments_and_memories(
            frontend_segments,
            ob_memories,
        )
    )

    return _build_artifact_from_observations(
        conversation_id=conversation_id,
        cognitive_request_id=cognitive_request_id,
        source_unified_revision=(
            source_unified_revision
        ),
        observations=observations,
        ob_observation_available=(
            ob_observation_available
        ),
    )


def update_memory_source_arbitration_from_evidence(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    source_evidence_snapshot: Any,
    expected_unified_revision: Any,
) -> dict[str, Any]:
    """Observe + persist one request's arbitration from shared evidence.

    The live coordinator path uses this: the request-local Source
    Evidence Snapshot is built ONCE and both the arbitration observer
    and the Delivery Plan Shadow consume it. This function never
    reparses the forwarded body / snapshot and never raises.

    Existing v1 semantics are preserved exactly: a valid canonical
    Frontend side plus a valid/invalid OB side produces the same
    artifact the legacy path produced, and a Frontend canonicalization
    failure still yields ``no_record`` -- partial OB availability never
    weakens the Frontend measurement.
    """

    if not is_valid_conversation_id(conversation_id):
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason": "invalid_conversation_id",
        }

    if not is_valid_cognitive_request_id(
        cognitive_request_id
    ):
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason": "invalid_cognitive_request_id",
        }

    if not is_valid_revision(
        expected_unified_revision
    ):
        return {
            "stored": False,
            "mode": _MODE,
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
            "mode": _MODE,
            "decision": "no_record",
            "reason": "evidence_not_bound",
        }

    if not (
        source_evidence_snapshot
        .frontend_observation_available
    ):
        # Malformed / unsupported Frontend: fail-safe, no failure, no
        # artifact, no reclassification of ordinary text.
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason": "request_not_canonicalizable",
        }

    observations = _observations_from_units(
        source_evidence_snapshot.units
    )

    artifact = _build_artifact_from_observations(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        source_unified_revision=(
            expected_unified_revision
        ),
        observations=observations,
        ob_observation_available=(
            source_evidence_snapshot
            .ob_observation_available
        ),
    )

    artifact["created_at"] = now_iso()

    atomic_write(
        _path(
            conversation_id,
            cognitive_request_id,
        ),
        artifact,
    )

    return {
        "stored": True,
        "mode": _MODE,
        "decision": "observed",
        "reason": "arbitration_observed",
        "source_unified_revision":
            expected_unified_revision,
        "ob_observation_available":
            artifact["ob_observation_available"],
        "metrics": dict(
            artifact["metrics"]
        ),
    }


def update_memory_source_arbitration(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    snapshot: Any,
    expected_unified_revision: Any,
    forward_body: Any,
) -> dict[str, Any]:
    """Observe and persist one request's memory-source arbitration.

    Compatibility wrapper. It builds the request-local Source Evidence
    Snapshot through the SHARED adapter and then delegates to
    ``update_memory_source_arbitration_from_evidence`` -- there is only
    one canonicalization / adaptation per call, and the same observation
    semantics as v1. Never raises. Returns a privacy-safe report.
    ``stored`` is False (with a reason) whenever a truthful observation
    cannot be made: a binding is invalid, the forwarded body is not
    canonicalizable, or the Unified revision is invalid. It performs no
    retrieval, no embedding, no re-ranking and no memory write.

    The rollout flag is owned by the caller (the Context coordinator),
    exactly like the other Memory shadow observers: this function is
    the pure observer and is only reached once the caller has decided
    to run it.
    """

    if not is_valid_conversation_id(
        conversation_id
    ):
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason": "invalid_conversation_id",
        }

    if not is_valid_cognitive_request_id(
        cognitive_request_id
    ):
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason": "invalid_cognitive_request_id",
        }

    if not is_valid_revision(
        expected_unified_revision
    ):
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason":
                "invalid_expected_unified_revision",
        }

    evidence = build_memory_source_evidence_snapshot(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        source_unified_revision=(
            expected_unified_revision
        ),
        forward_body=forward_body,
        memory_shadow_snapshot=snapshot,
    )

    return update_memory_source_arbitration_from_evidence(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        source_evidence_snapshot=evidence,
        expected_unified_revision=(
            expected_unified_revision
        ),
    )
