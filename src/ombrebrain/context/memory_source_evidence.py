from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ombrebrain.context.recall_types import (
    estimate_tokens,
    normalize_key,
)
from ombrebrain.context.retrieval_decision_shadow import (
    candidate_fingerprint,
    candidate_text,
)
from ombrebrain.gateway.canonicalizer import (
    extract_frontend_memory_segments,
)
from ombrebrain.gateway.operit_adapter import OperitAdapter


# Source-Neutral Memory Evidence Adapter — request-local, IN-MEMORY ONLY.
#
#     source-native memory evidence
#         -> Source-Neutral Evidence Adapter       (this module)
#             -> request-local Memory Source Evidence Snapshot
#                 -> consumers (Arbitration Shadow / Delivery Shadow)
#
# This module asks ONE question: "for THIS request, what memory evidence
# units already exist, described in a source-neutral CONTROL
# vocabulary?" It does NOT unify storage, does NOT unify retrieval and
# does NOT turn a source payload into a universal "memory" object.
#
# Contract: ``memory-source-evidence.v1``.
#
# Two sources are federated and stay federated:
#
#   frontend: one relevant_memories.txt ATTACHMENT == ONE evidence unit.
#       The repository has no proven item-level schema for the inner
#       body, so one attachment is never split into "memories". The
#       inner content is never parsed.
#   ob: one Unified-admitted memory CANDIDATE == ONE evidence unit.
#
# The adapter normalizes ONLY the control metadata the consumers agree
# on: source, kind, granularity, source scope, digest / normalized
# digest, size, visibility state, retrieval state and request
# occurrence. The real source object is retained as an in-memory
# ``native_payload`` reference so a downstream shadow observer can reuse
# the exact source object; it is never called a universal memory and is
# never persisted or logged.
#
# Hard rules:
#   - frontend extraction reuses the canonicalizer
#     (``extract_frontend_memory_segments``); the attachment grammar /
#     filename classification is never copied here, and ordinary user
#     text is never reclassified as Frontend Memory;
#   - OB evidence is read ONLY from THIS request's own memory shadow
#     snapshot (strict version / conversation / revision binding). It
#     never re-reads a conversation-global Unified file and never reruns
#     retrieval / embedding / vector search / rerank / BucketManager;
#   - the complete snapshot is in-memory only: never persisted, never
#     logged, never placed in Unified / Preview / Gate / Mutation /
#     Real Injection / the request body / a cache key / a header;
#   - duplicate occurrences stay DISTINCT: two identical frontend
#     attachments are two units, and two identical OB candidates are
#     two units. This is an OCCURRENCE adapter, never an identity claim.

_VERSION = "memory-source-evidence.v1"

_UNIFIED_VERSION = "unified-context-candidate.v1"
_SNAPSHOT_VERSION = "memory-shadow-snapshot.v1"

SOURCE_FRONTEND = "frontend"
SOURCE_OB = "ob"

KIND_FRONTEND_ATTACHMENT = "frontend_attachment"
KIND_OB_CANDIDATE = "ob_candidate"

GRANULARITY_ATTACHMENT = "attachment"
GRANULARITY_CANDIDATE = "candidate"

# Source-neutral placement / provenance scope.
SCOPE_CURRENT = "current"
SCOPE_HISTORY = "history"
SCOPE_RETRIEVAL = "retrieval"

# There is deliberately NO silent truncation of the in-memory unit
# pool. The snapshot is request-local and never persisted, OB Unified
# memories are already tightly bounded upstream and the persisted
# Delivery artifact carries its own explicit hard cap with honest
# truncation telemetry. A silent slice here would let a downstream
# consumer believe it observed the COMPLETE source set, so it is never
# applied.


@dataclass(frozen=True)
class MemorySourceEvidenceUnit:
    """One source-neutral evidence unit for THIS request.

    ``native_payload`` is the exact source-native object (a
    ``FrontendMemorySegment`` or the OB candidate dict). It exists so a
    downstream shadow observer can reuse the real source object without
    reparsing; it is excluded from ``repr`` and is never persisted or
    logged. This is NOT a universal memory object.

    ``content_chars`` / ``content_estimated_tokens`` measure the
    extractable BODY content (comparable across sources).
    ``model_visible_chars`` / ``model_visible_estimated_tokens`` measure
    the text the provider request actually carries RIGHT NOW: for a
    Frontend attachment that is the exact raw attachment envelope
    (metadata + body), for an OB candidate it is 0 because OB content is
    not model-visible. The two concepts are kept strictly separate and
    are never collapsed. Only the numeric footprint leaves the
    request-local snapshot; the raw envelope is never persisted.
    """

    source: str
    kind: str
    granularity: str
    source_index: int
    source_scope: str
    content_digest: str
    normalized_digest: str
    content_chars: int
    content_estimated_tokens: int
    model_visible_chars: int
    model_visible_estimated_tokens: int
    currently_model_visible: bool
    retrieved: bool
    request_occurrence: int
    candidate_fingerprint: str | None = None
    native_payload: Any = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class MemorySourceEvidenceSnapshot:
    """Request-bound, in-memory-only source evidence for one request.

    The snapshot is deliberately not serializable: ``native_payload``
    references carry real source content. It is never logged, never
    persisted, never model-visible and never a universal memory schema.
    """

    version: str
    conversation_id: str
    cognitive_request_id: str
    source_unified_revision: Any
    frontend_observation_available: bool
    ob_observation_available: bool
    units: tuple[MemorySourceEvidenceUnit, ...]


def _digest(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _normalized_digest(text: str) -> str:
    return hashlib.sha256(
        normalize_key(text).encode("utf-8")
    ).hexdigest()


def _canonicalize(
    forward_body: Any,
):
    """Parse the forwarded body with the existing canonicalizer.

    Returns None for anything that is not a supported Operit request, so
    a malformed body can never be classified as Frontend Memory. This is
    ONE canonicalization pass for the request; it starts no pipeline and
    reuses the existing adapter.
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


def frontend_evidence_units(
    segments: Any,
) -> list[MemorySourceEvidenceUnit]:
    """One FrontendMemorySegment -> one ``frontend_attachment`` unit.

    The inner body is never parsed: one attachment is one unit,
    regardless of the shape of its content (list-like text, JSON-like
    text, XML-like text, blank-line paragraphs or duplicate sentences).

    The model-visible footprint is measured from the segment's exact
    canonical ``raw_attachment`` envelope, never from the bare body. A
    segment without a valid raw attachment has no proven model-visible
    envelope, so it is refused rather than silently labelling its body
    as the footprint. The live canonical path always supplies it.
    """

    units: list[MemorySourceEvidenceUnit] = []

    if not isinstance(segments, (list, tuple)):
        return units

    for index, segment in enumerate(segments):
        body = getattr(segment, "body", None)

        if (
            not isinstance(body, str)
            or not body.strip()
        ):
            continue

        raw_attachment = getattr(
            segment,
            "raw_attachment",
            None,
        )

        if (
            not isinstance(raw_attachment, str)
            or not raw_attachment
        ):
            # No actual canonical envelope: there is no exact
            # model-visible footprint to report, so this malformed
            # unit is refused instead of being measured as its body.
            continue

        placement = getattr(
            segment,
            "placement",
            SCOPE_HISTORY,
        )

        units.append(
            MemorySourceEvidenceUnit(
                source=SOURCE_FRONTEND,
                kind=KIND_FRONTEND_ATTACHMENT,
                granularity=GRANULARITY_ATTACHMENT,
                source_index=index,
                source_scope=(
                    SCOPE_CURRENT
                    if placement == SCOPE_CURRENT
                    else SCOPE_HISTORY
                ),
                content_digest=_digest(body),
                normalized_digest=(
                    _normalized_digest(body)
                ),
                content_chars=len(body),
                content_estimated_tokens=(
                    estimate_tokens(body)
                ),
                # The exact raw envelope (metadata + body) is what the
                # provider request actually carries: this is the real
                # model-visible footprint, larger than the body.
                model_visible_chars=len(
                    raw_attachment
                ),
                model_visible_estimated_tokens=(
                    estimate_tokens(raw_attachment)
                ),
                # A Frontend attachment body is already in the
                # model-visible request; it was attached by the
                # frontend, not retrieved by the OB subsystem.
                currently_model_visible=True,
                retrieved=False,
                request_occurrence=index,
                native_payload=segment,
            )
        )

    return units


def ob_evidence_units(
    memories: Any,
) -> list[MemorySourceEvidenceUnit]:
    """One Unified-admitted memory candidate -> one ``ob_candidate`` unit."""

    units: list[MemorySourceEvidenceUnit] = []

    if not isinstance(memories, list):
        return units

    for index, item in enumerate(memories):
        content = candidate_text(item)

        if not content:
            continue

        units.append(
            MemorySourceEvidenceUnit(
                source=SOURCE_OB,
                kind=KIND_OB_CANDIDATE,
                granularity=GRANULARITY_CANDIDATE,
                source_index=index,
                source_scope=SCOPE_RETRIEVAL,
                content_digest=_digest(content),
                normalized_digest=(
                    _normalized_digest(content)
                ),
                content_chars=len(content),
                content_estimated_tokens=(
                    estimate_tokens(content)
                ),
                # OB candidate content is NOT in the live request; only
                # a bounded cue / an explicit Recall may expose it, so
                # its model-visible footprint is exactly 0 and it is
                # never currently model-visible.
                model_visible_chars=0,
                model_visible_estimated_tokens=0,
                currently_model_visible=False,
                retrieved=True,
                request_occurrence=index,
                candidate_fingerprint=(
                    candidate_fingerprint(item)
                ),
                native_payload=item,
            )
        )

    return units


def ob_memories_from_snapshot(
    snapshot: Any,
    *,
    conversation_id: Any,
    expected_unified_revision: Any,
) -> tuple[Any, bool]:
    """Extract THIS request's OB memory section, or mark it unavailable.

    Strict binding (never guessed, never a fallback):

        snapshot.version == memory-shadow-snapshot.v1
        snapshot.conversation_id == current conversation
        unified.version == unified-context-candidate.v1
        unified.conversation_id == current conversation
        unified.revision == expected source Unified revision

    Anything else is ``ob_observation_available = false`` -- UNKNOWN is
    never quietly turned into "zero candidates". It never re-reads the
    shared conversation-level Unified file.

    A valid EMPTY OB observation is the ONLY shape that yields
    ``([], True)``: ``sections`` is a dict AND ``sections["memories"]``
    is a list AND that list is empty. A malformed structure (missing /
    ``None`` / non-dict ``sections``, missing / ``None`` / non-list
    ``memories``) is UNAVAILABLE and yields ``([], False)`` -- zero
    valid memories are never fabricated from malformed evidence.
    """

    unavailable = ([], False)

    if not isinstance(snapshot, dict):
        return unavailable

    if snapshot.get("version") != _SNAPSHOT_VERSION:
        return unavailable

    if snapshot.get("conversation_id") != conversation_id:
        return unavailable

    unified = snapshot.get("unified")

    if not isinstance(unified, dict):
        return unavailable

    if unified.get("version") != _UNIFIED_VERSION:
        return unavailable

    if unified.get("conversation_id") != conversation_id:
        return unavailable

    if (
        unified.get("revision")
        != expected_unified_revision
    ):
        return unavailable

    sections = unified.get("sections")

    if not isinstance(sections, dict):
        return unavailable

    memories = sections.get("memories")

    if not isinstance(memories, list):
        return unavailable

    return (memories, True)


def build_memory_source_evidence_snapshot(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    source_unified_revision: Any,
    forward_body: Any,
    memory_shadow_snapshot: Any,
) -> MemorySourceEvidenceSnapshot:
    """Build the request-local evidence snapshot. Never raises on input.

    Frontend availability and OB availability are represented
    INDEPENDENTLY: a canonicalization success with zero attachments is
    ``frontend_observation_available = true`` and zero units, which is
    different from a Frontend canonicalization failure. A source failure
    is never converted into "zero items without an availability bit".
    """

    request = _canonicalize(forward_body)

    frontend_available = request is not None

    frontend_units = (
        frontend_evidence_units(
            extract_frontend_memory_segments(request)
        )
        if frontend_available
        else []
    )

    ob_memories, ob_available = (
        ob_memories_from_snapshot(
            memory_shadow_snapshot,
            conversation_id=conversation_id,
            expected_unified_revision=(
                source_unified_revision
            ),
        )
    )

    ob_units = (
        ob_evidence_units(ob_memories)
        if ob_available
        else []
    )

    # Serialization / occurrence order: Frontend occurrences in canonical
    # request order, then OB occurrences in canonical retrieval order.
    # Order is NOT priority and is never called rank.
    units = tuple(frontend_units + ob_units)

    return MemorySourceEvidenceSnapshot(
        version=_VERSION,
        conversation_id=conversation_id,
        cognitive_request_id=cognitive_request_id,
        source_unified_revision=source_unified_revision,
        frontend_observation_available=(
            frontend_available
        ),
        ob_observation_available=ob_available,
        units=units,
    )


def is_bound_to_request(
    snapshot: Any,
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    expected_unified_revision: Any,
) -> bool:
    """True when the snapshot is THIS request's exact evidence.

    A foreign or mismatched snapshot must never become the runtime
    evidence for another request.
    """

    return bool(
        isinstance(
            snapshot,
            MemorySourceEvidenceSnapshot,
        )
        and snapshot.version == _VERSION
        and snapshot.conversation_id
        == conversation_id
        and snapshot.cognitive_request_id
        == cognitive_request_id
        and snapshot.source_unified_revision
        == expected_unified_revision
    )
