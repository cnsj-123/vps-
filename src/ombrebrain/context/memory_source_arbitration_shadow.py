from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ombrebrain.context.recall_types import (
    atomic_write,
    estimate_tokens,
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
from ombrebrain.gateway.operit_adapter import OperitAdapter


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

_UNIFIED_VERSION = "unified-context-candidate.v1"
_SNAPSHOT_VERSION = "memory-shadow-snapshot.v1"

_SOURCE_FRONTEND = "frontend"
_SOURCE_OB = "ob"

_KIND_FRONTEND_ATTACHMENT = "frontend_attachment"
_KIND_OB_CANDIDATE = "ob_candidate"

_PLACEMENT_CURRENT = "current"
_PLACEMENT_HISTORY = "history"

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


def _frontend_observations(
    segments: Any,
) -> list[_Observation]:
    observations: list[_Observation] = []

    if not isinstance(segments, (list, tuple)):
        return observations

    for index, segment in enumerate(segments):
        body = getattr(segment, "body", None)

        if (
            not isinstance(body, str)
            or not body.strip()
        ):
            continue

        # The exact raw canonical attachment text. It is only used in
        # memory to measure the real model-visible footprint; it is
        # never persisted or logged. If a caller supplied no raw text
        # the body is used as a conservative lower bound -- no tags are
        # ever reconstructed.
        raw_attachment = getattr(
            segment,
            "raw_attachment",
            None,
        )

        if (
            not isinstance(raw_attachment, str)
            or not raw_attachment
        ):
            raw_attachment = body

        placement = getattr(
            segment,
            "placement",
            _PLACEMENT_HISTORY,
        )

        observations.append(
            _Observation(
                source=_SOURCE_FRONTEND,
                kind=_KIND_FRONTEND_ATTACHMENT,
                placement=(
                    _PLACEMENT_CURRENT
                    if placement == _PLACEMENT_CURRENT
                    else _PLACEMENT_HISTORY
                ),
                index=index,
                content_digest=_digest(body),
                normalized_digest=(
                    _normalized_digest(body)
                ),
                content_chars=len(body),
                content_estimated_tokens=(
                    estimate_tokens(body)
                ),
                model_visible_chars=len(
                    raw_attachment
                ),
                model_visible_estimated_tokens=(
                    estimate_tokens(raw_attachment)
                ),
                # Frontend attachment content is already in the
                # model-visible request.
                currently_model_visible=True,
                # It was attached by the frontend, not retrieved by
                # the OB retrieval subsystem.
                retrieved=False,
            )
        )

    return observations


def _ob_observations(
    memories: Any,
) -> list[_Observation]:
    observations: list[_Observation] = []

    if not isinstance(memories, list):
        return observations

    for index, item in enumerate(memories):
        content = candidate_text(item)

        if not content:
            continue

        observations.append(
            _Observation(
                source=_SOURCE_OB,
                kind=_KIND_OB_CANDIDATE,
                placement=_PLACEMENT_HISTORY,
                index=index,
                content_digest=_digest(content),
                normalized_digest=(
                    _normalized_digest(content)
                ),
                content_chars=len(content),
                content_estimated_tokens=(
                    estimate_tokens(content)
                ),
                # OB candidate content is NOT in the live request;
                # only a future bounded cue / Recall may expose it.
                model_visible_chars=0,
                model_visible_estimated_tokens=0,
                currently_model_visible=False,
                # It came from THIS request's retrieval observation.
                retrieved=True,
            )
        )

    return observations


def _build_metrics(
    observations: list[_Observation],
) -> dict[str, Any]:
    frontend = [
        o
        for o in observations
        if o.source == _SOURCE_FRONTEND
    ]
    ob = [
        o
        for o in observations
        if o.source == _SOURCE_OB
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
            _SOURCE_FRONTEND: len(frontend),
            _SOURCE_OB: len(ob),
        },
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
    """

    observations = (
        _frontend_observations(frontend_segments)
        + _ob_observations(ob_memories)
    )

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
            _SOURCE_FRONTEND: "attachment",
            _SOURCE_OB: "candidate",
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


def _ob_memories_from_snapshot(
    snapshot: Any,
    *,
    conversation_id: str,
    expected_unified_revision: Any,
) -> tuple[Any, bool]:
    """Extract THIS request's OB memory section, or mark it unavailable.

    It never re-reads the shared conversation-level Unified file. Any
    identity / version / revision mismatch is treated as "not
    available" rather than guessed.
    """

    unavailable = ([], False)

    if not isinstance(snapshot, dict):
        return unavailable

    if (
        snapshot.get("version")
        != _SNAPSHOT_VERSION
    ):
        return unavailable

    if (
        snapshot.get("conversation_id")
        != conversation_id
    ):
        return unavailable

    unified = snapshot.get("unified")

    if not isinstance(unified, dict):
        return unavailable

    if unified.get("version") != _UNIFIED_VERSION:
        return unavailable

    if (
        unified.get("conversation_id")
        != conversation_id
    ):
        return unavailable

    if (
        unified.get("revision")
        != expected_unified_revision
    ):
        return unavailable

    sections = unified.get("sections")

    if not isinstance(sections, dict):
        return ([], True)

    memories = sections.get("memories")

    if not isinstance(memories, list):
        return ([], True)

    return (memories, True)


def update_memory_source_arbitration(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    snapshot: Any,
    expected_unified_revision: Any,
    forward_body: Any,
) -> dict[str, Any]:
    """Observe and persist one request's memory-source arbitration.

    Never raises. Returns a privacy-safe report. ``stored`` is False
    (with a reason) whenever a truthful observation cannot be made: a
    binding is invalid, the forwarded body is not canonicalizable, or
    the Unified revision is invalid. It performs no retrieval, no
    embedding, no re-ranking and no memory write.

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

    request = _canonicalize(forward_body)

    if request is None:
        # Malformed / unsupported request is fail-safe: no failure,
        # no artifact, no reclassification of ordinary text.
        return {
            "stored": False,
            "mode": _MODE,
            "decision": "no_record",
            "reason": "request_not_canonicalizable",
        }

    frontend_segments = (
        extract_frontend_memory_segments(request)
    )

    ob_memories, ob_available = (
        _ob_memories_from_snapshot(
            snapshot,
            conversation_id=conversation_id,
            expected_unified_revision=(
                expected_unified_revision
            ),
        )
    )

    artifact = build_arbitration_artifact(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        source_unified_revision=(
            expected_unified_revision
        ),
        frontend_segments=frontend_segments,
        ob_memories=ob_memories,
        ob_observation_available=ob_available,
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


def _canonicalize(
    forward_body: Any,
):
    """Parse the forwarded body with the existing canonicalizer.

    Returns None for anything that is not a supported Operit request,
    so a malformed body can never be classified as Frontend Memory.
    This is one canonicalization pass for the observer only; it starts
    no pipeline and reuses the existing adapter.
    """

    if not isinstance(forward_body, bytes):
        return None

    try:
        payload = json.loads(forward_body)
    except Exception:
        return None

    if not OperitAdapter.supports(payload):
        return None

    try:
        return OperitAdapter().adapt(payload)
    except Exception:
        return None
