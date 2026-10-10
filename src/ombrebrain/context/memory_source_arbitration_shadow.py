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
#       full content is already model-visible in the live request,
#       preserved by cache rewriting, excluded from Conversation
#       Snapshot USER_TEXT-derived semantic state;
#
#   OB / cognitive memory
#       retrieved -> eligibility -> Memory Flash cue -> model decides
#       Recall -> full memory loaded only on an explicit Recall.
#
# It answers, from REAL requests: how many Frontend Memory
# candidates exist, how much model-visible context they occupy,
# whether they are current or historical, how many OB candidates the
# same request produced, how much the two sources overlap, and how
# much of the request's memory footprint comes from Frontend Memory.
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
# semantic threshold is used. "exact" and "normalized" are distinct
# evidence types and neither is a claim of semantic equivalence.
#
# Privacy: the persisted artifact and any log line contain only
# counts, booleans, enums, revisions, shares, bounded numeric metrics
# and sha256 digests. No Frontend Memory text, OB memory text, cue,
# user text, attachment contents, memref, useref or raw memory id is
# ever emitted.

_VERSION = "memory-source-arbitration-shadow.v1"
_MODE = "shadow_only"

_UNIFIED_VERSION = "unified-context-candidate.v1"
_SNAPSHOT_VERSION = "memory-shadow-snapshot.v1"

_SOURCE_FRONTEND = "frontend"
_SOURCE_OB = "ob"

_PLACEMENT_CURRENT = "current"
_PLACEMENT_HISTORY = "history"

# Bound the persisted per-candidate evidence. The candidate pool is
# small (<=6 OB candidates by Unified budget plus the Frontend
# attachments), but this keeps the artifact size hard-bounded anyway.
_MAX_PERSISTED_CANDIDATES = 64


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
class _Candidate:
    """One source-neutral shadow candidate.

    Only information needed for arbitration analysis is kept. All
    observed stage flags reflect real evidence: ``currently_model_visible``
    and ``retrieved`` are set from what this request actually shows,
    and no later stage is ever inferred.
    """

    source: str
    placement: str
    index: int
    content_digest: str
    normalized_digest: str
    char_count: int
    estimated_tokens: int
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


def _frontend_candidates(
    segments: Any,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []

    if not isinstance(segments, (list, tuple)):
        return candidates

    for index, segment in enumerate(segments):
        body = getattr(segment, "body", None)

        if (
            not isinstance(body, str)
            or not body.strip()
        ):
            continue

        placement = getattr(
            segment,
            "placement",
            _PLACEMENT_HISTORY,
        )

        candidates.append(
            _Candidate(
                source=_SOURCE_FRONTEND,
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
                char_count=len(body),
                estimated_tokens=estimate_tokens(
                    body
                ),
                # Frontend Memory full content is already in the
                # model-visible request.
                currently_model_visible=True,
                # It was attached by the frontend, not retrieved by
                # the OB retrieval subsystem.
                retrieved=False,
            )
        )

    return candidates


def _ob_candidates(
    memories: Any,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []

    if not isinstance(memories, list):
        return candidates

    for index, item in enumerate(memories):
        content = candidate_text(item)

        if not content:
            continue

        candidates.append(
            _Candidate(
                source=_SOURCE_OB,
                placement=_PLACEMENT_HISTORY,
                index=index,
                content_digest=_digest(content),
                normalized_digest=(
                    _normalized_digest(content)
                ),
                char_count=len(content),
                estimated_tokens=estimate_tokens(
                    content
                ),
                # OB full content is NOT in the live request; only a
                # future bounded cue / Recall may expose it.
                currently_model_visible=False,
                # It came from THIS request's retrieval observation.
                retrieved=True,
            )
        )

    return candidates


def _build_metrics(
    candidates: list[_Candidate],
) -> dict[str, Any]:
    frontend = [
        c
        for c in candidates
        if c.source == _SOURCE_FRONTEND
    ]
    ob = [
        c
        for c in candidates
        if c.source == _SOURCE_OB
    ]

    frontend_chars = sum(
        c.char_count for c in frontend
    )
    frontend_tokens = sum(
        c.estimated_tokens for c in frontend
    )

    ob_chars = sum(c.char_count for c in ob)
    ob_tokens = sum(
        c.estimated_tokens for c in ob
    )

    visible = [
        c
        for c in frontend
        if c.currently_model_visible
    ]
    visible_tokens = sum(
        c.estimated_tokens for c in visible
    )

    # Cross-source exact / normalized overlap, counted as ordered
    # (frontend, ob) pairs. exact is a subset of normalized by
    # construction (byte equality implies normalized equality).
    ob_by_exact = Counter(
        c.content_digest for c in ob
    )
    ob_by_normalized = Counter(
        c.normalized_digest for c in ob
    )

    exact_overlap = sum(
        ob_by_exact.get(c.content_digest, 0)
        for c in frontend
    )
    normalized_overlap = sum(
        ob_by_normalized.get(
            c.normalized_digest, 0
        )
        for c in frontend
    )

    # Groups of normalized-equivalent candidates in the combined
    # pool (size >= 2). This is how "same content twice" and
    # "Frontend == OB" become visible without collapsing identities.
    normalized_counts = Counter(
        c.normalized_digest for c in candidates
    )
    duplicate_group_count = sum(
        1
        for count in normalized_counts.values()
        if count >= 2
    )

    combined_count = len(candidates)
    combined_tokens = (
        frontend_tokens + ob_tokens
    )

    if combined_tokens > 0:
        frontend_share = round(
            frontend_tokens / combined_tokens, 6
        )
        ob_share = round(
            ob_tokens / combined_tokens, 6
        )
    else:
        frontend_share = 0.0
        ob_share = 0.0

    return {
        "frontend_candidate_count": len(frontend),
        "frontend_current_count": sum(
            1
            for c in frontend
            if c.placement == _PLACEMENT_CURRENT
        ),
        "frontend_history_count": sum(
            1
            for c in frontend
            if c.placement == _PLACEMENT_HISTORY
        ),
        "frontend_chars": frontend_chars,
        "frontend_estimated_tokens":
            frontend_tokens,
        "ob_candidate_count": len(ob),
        "ob_chars": ob_chars,
        "ob_estimated_tokens": ob_tokens,
        "frontend_currently_model_visible_count":
            len(visible),
        "frontend_currently_model_visible_tokens":
            visible_tokens,
        "exact_overlap_count": exact_overlap,
        "normalized_overlap_count":
            normalized_overlap,
        "duplicate_group_count":
            duplicate_group_count,
        "combined_memory_candidate_count":
            combined_count,
        "combined_estimated_tokens":
            combined_tokens,
        "frontend_token_share": frontend_share,
        "ob_token_share": ob_share,
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

    No IO, no logging, deterministic. Frontend candidates come from
    the canonical forwarded request; OB candidates come from THIS
    request's own Unified memory section. Nothing is inferred beyond
    those two real inputs.
    """

    candidates = _frontend_candidates(
        frontend_segments
    ) + _ob_candidates(ob_memories)

    metrics = _build_metrics(candidates)

    persisted: list[dict[str, Any]] = []

    for candidate in (
        candidates[:_MAX_PERSISTED_CANDIDATES]
    ):
        persisted.append(
            {
                "source": candidate.source,
                "placement": candidate.placement,
                "index": candidate.index,
                "candidate": True,
                "content_digest":
                    candidate.content_digest,
                "normalized_digest":
                    candidate.normalized_digest,
                "char_count":
                    candidate.char_count,
                "estimated_tokens":
                    candidate.estimated_tokens,
                "currently_model_visible":
                    candidate.currently_model_visible,
                "retrieved": candidate.retrieved,
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
        "overlap_method":
            "exact_then_normalized_text.v1",
        "normalization":
            "whitespace_collapse_casefold",
        "metrics": metrics,
        "candidates": persisted,
        "candidates_truncated": (
            len(candidates)
            > _MAX_PERSISTED_CANDIDATES
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
