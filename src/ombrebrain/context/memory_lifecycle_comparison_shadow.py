from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import deque
from datetime import datetime, timezone
from typing import Any

from ombrebrain.context.memory_flash import (
    candidate_memory_id,
)
from ombrebrain.context.recall_types import (
    atomic_write,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    now_iso,
    state_root,
)
from ombrebrain.context.memory_lifecycle_event import (
    memory_key,
    parse_iso_utc,
)
from ombrebrain.context.memory_lifecycle_state import (
    resolve_lifecycle_config,
)
from ombrebrain.context.memory_surfacing_lifecycle import (
    evaluate_lifecycle_surfacing_candidate,
    lifecycle_shadow_enabled,
    lifecycle_surfacing_enabled,
    resolve_surfacing_lifecycle_config,
)
from ombrebrain.context.retrieval_decision_shadow import (
    candidate_fingerprint,
)
from ombrebrain.context.validators.freshness import (
    is_valid_revision,
)


# Memory Lifecycle Comparison Shadow v1 — OBSERVATION ONLY.
#
#     LEGACY OB lifecycle                    COGNITIVE lifecycle
#     ------------------                     -------------------
#     importance / activation_count /        immutable ``used`` events
#     last_active / resolved / digested      -> strength / accessibility /
#       -> DecayEngine.calculate_score()         salience
#       -> auto-resolve / archive               -> supplementary surfacing
#                                                  constraint
#
# These are two DIFFERENT lifetime models. This module does not decide
# which is right and does not become a third lifecycle authority. On
# the SAME real, request-local retrieval candidates it records where
# the two current policies AGREE and where they MATERIALLY DISAGREE,
# as a privacy-safe, request-scoped artifact for later analysis.
#
# Hard boundaries (enforced by construction + guardrail tests):
#
#   - it never archives / resolves / touches / deletes / prefers a
#     memory and never changes activation_count / last_active /
#     importance;
#   - it never writes a cognitive lifecycle event, a state snapshot or
#     a reinforcement receipt, and never fabricates Used evidence;
#   - it never changes Retrieval, ranking, the Surfacing Policy, Flash,
#     Recall, Dream or the forwarded provider body;
#   - it never starts or stops DecayEngine and never calls its mutation
#     or background APIs -- the narrow binding may ONLY use the
#     side-effect-free ``preview_decay_cycle_decision`` read surface;
#   - any failure is fail-open: the comparison data is simply missing
#     and NOTHING else changes.
#
# The Legacy preview uses the REAL runtime DecayEngine semantics (the
# same predicates and the same score math as the live decay cycle),
# never a copied second formula. The Cognitive side reuses the REAL
# lifecycle formulas and the same supplementary evaluator the live
# gate uses, so the threshold decision lives in ONE place.
#
#     unobserved != forgotten
#
# A memory with no valid ``used`` lifecycle event is NEUTRAL, never
# fabricated as strength=0 / forgotten / suppressed.
#
# Privacy: the persisted artifact carries request binding ids already
# used by other request-scoped Context artifacts, revisions, counts,
# enums, booleans, finite numbers, thresholds, the candidate digest and
# the hashed memory key only. No memory text, name, query, cue, memref,
# useref or raw memory id is ever persisted, and log lines carry only
# aggregate counts / reasons / revisions / thresholds.

logger = logging.getLogger("ombre_brain.gateway")

COMPARISON_SHADOW_ENV = (
    "OMBRE_GATEWAY_CONTEXT_LIFECYCLE_COMPARISON_SHADOW"
)

VERSION = "memory-lifecycle-comparison-shadow.v1"

MODE = "shadow_only"

_UNIFIED_VERSION = "unified-context-candidate.v1"
_SNAPSHOT_VERSION = "memory-shadow-snapshot.v1"

_POLICY_VERSION = "memory-surfacing-policy.v1"
_POLICY_MODE = "shadow_only"
_ALLOW_DECISION = "allow_shadow"

# Hard bound on persisted per-request candidate evidence.
MAX_COMPARISON_ROWS = 64

# Generous in-memory safety cap on captured Legacy observations. Real
# retrieval candidate sets are far smaller; this only prevents an
# unbounded snapshot if a caller ever supplies one.
_MAX_LEGACY_OBSERVATIONS = 256

# Deterministic comparison categories.
_CAT_BINDING_UNAVAILABLE = "binding_unavailable"
_CAT_LEGACY_UNAVAILABLE = "legacy_unavailable"
_CAT_LEGACY_NOT_APPLICABLE = "legacy_not_applicable"
_CAT_COGNITIVE_UNAVAILABLE = "cognitive_evidence_unavailable"

# Counterfactual gate decisions: what the CURRENT lifecycle gate policy
# would decide IF it were applied to this base policy. These are
# computed regardless of whether the live gate is enabled.
_GATE_PRIMARY = "primary_preserved"
_GATE_NEUTRAL = "neutral_unobserved"
_GATE_KEEP = "would_keep"
_GATE_SUPPRESS = "would_suppress"
_GATE_NOT_BASE_ELIGIBLE = "not_base_eligible"
_GATE_BINDING_UNAVAILABLE = "binding_unavailable"
_GATE_BASE_POLICY_UNAVAILABLE = "base_policy_unavailable"
_GATE_COGNITIVE_EVIDENCE_UNAVAILABLE = (
    "cognitive_evidence_unavailable"
)

# Actual live-gate participation. ``gate_disabled`` means the live
# Lifecycle Surfacing gate cannot change anything under its current
# flags, so a counterfactual suppression here is NOT a live effect.
_GATE_DISABLED = "gate_disabled"

_KNOWN_COUNTERFACTUAL_DECISIONS = (
    _GATE_PRIMARY,
    _GATE_NEUTRAL,
    _GATE_KEEP,
    _GATE_SUPPRESS,
    _GATE_NOT_BASE_ELIGIBLE,
    _GATE_BINDING_UNAVAILABLE,
    _GATE_BASE_POLICY_UNAVAILABLE,
    _GATE_COGNITIVE_EVIDENCE_UNAVAILABLE,
)

_KNOWN_LIVE_GATE_DECISIONS = (
    _KNOWN_COUNTERFACTUAL_DECISIONS
    + (_GATE_DISABLED,)
)

# All deterministic comparison categories the builder may emit.
_KNOWN_COMPARISON_CATEGORIES = (
    _CAT_BINDING_UNAVAILABLE,
    _CAT_LEGACY_UNAVAILABLE,
    _CAT_LEGACY_NOT_APPLICABLE,
    _CAT_COGNITIVE_UNAVAILABLE,
    "legacy_archive_cognitive_unobserved",
    "legacy_archive_cognitive_suppress",
    "legacy_archive_cognitive_keep",
    "legacy_keep_cognitive_unobserved",
    "legacy_keep_cognitive_suppress",
    "legacy_keep_cognitive_keep",
)

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

# Cognitive counterfactual bucket names.
_COG_UNOBSERVED = "unobserved"
_COG_SUPPRESS = "suppress"
_COG_KEEP = "keep"

# Legacy archive/keep bucket names.
_LEG_ARCHIVE = "archive"
_LEG_KEEP = "keep"

# The real runtime DecayEngine, bound narrowly by server startup. Only
# its read-only preview API is ever called.
_ENGINE: Any = None


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def lifecycle_comparison_enabled() -> bool:
    """May the comparison observer run at all? Default OFF."""

    return _truthy(
        os.environ.get(COMPARISON_SHADOW_ENV)
    )


def bind_legacy_lifecycle_comparison_engine(
    decay_engine: Any,
) -> None:
    """Bind the EXISTING runtime DecayEngine for read-only previews.

    It stores the reference only: it never starts / stops the engine,
    never wraps it and never creates a second DecayEngine. Passing None
    simply leaves the comparison without Legacy evidence.
    """

    global _ENGINE
    _ENGINE = decay_engine


def legacy_comparison_engine_bound() -> bool:
    return _ENGINE is not None


def _finite(value: Any) -> float | None:
    """Finite float or None. Rejects NaN / Infinity / booleans."""

    if isinstance(value, bool):
        return None

    if not isinstance(value, (int, float)):
        return None

    number = float(value)

    if not math.isfinite(number):
        return None

    return number


def _path(
    conversation_id: str,
    cognitive_request_id: str,
):
    return (
        state_root()
        / "memory_lifecycle_comparison"
        / conversation_id
        / (cognitive_request_id + ".json")
    )


def capture_legacy_lifecycle_observations(
    retrieval_memories: Any,
) -> list[dict[str, Any]]:
    """Request-local Legacy preview evidence, captured BEFORE projection.

    The raw retrieval candidates still carry the canonical metadata the
    Legacy plane needs. This runs the bound runtime DecayEngine's
    read-only preview for each candidate, in canonical retrieval order,
    and returns only request-local evidence: the index, the (never
    persisted) memory id, the candidate fingerprint and the preview
    result. It never reads a bucket, never mutates metadata and never
    crosses into the Cognitive plane.
    """

    memories = (
        retrieval_memories
        if isinstance(retrieval_memories, list)
        else []
    )

    observations: list[dict[str, Any]] = []

    for index, candidate in enumerate(
        memories[:_MAX_LEGACY_OBSERVATIONS]
    ):
        metadata = (
            candidate.get("metadata")
            if isinstance(candidate, dict)
            else None
        )

        preview = None
        available = False

        if (
            _ENGINE is not None
            and isinstance(metadata, dict)
        ):
            try:
                result = (
                    _ENGINE.preview_decay_cycle_decision(
                        metadata
                    )
                )
            except Exception:
                result = None

            if isinstance(result, dict):
                preview = result
                available = True

        observations.append(
            {
                "retrieval_index": index,
                "memory_id":
                    candidate_memory_id(candidate),
                "candidate_fingerprint":
                    candidate_fingerprint(candidate),
                "legacy_available": available,
                "legacy": preview,
            }
        )

    return observations


# ------------------------------------------------------------------
# Pure builder
# ------------------------------------------------------------------


def _fingerprint_queues(
    items: Any,
) -> dict[str, deque]:
    """fingerprint -> FIFO of positional indices, in canonical order."""

    queues: dict[str, deque] = {}

    if not isinstance(items, list):
        return queues

    for index, item in enumerate(items):
        fingerprint = candidate_fingerprint(item)
        queues.setdefault(
            fingerprint, deque()
        ).append(index)

    return queues


def _unified_memories(
    unified: Any,
    *,
    expected_conversation_id: Any,
    expected_revision: Any,
) -> list | None:
    """THIS request's Unified admitted memories, or None if unusable.

    A request-local Unified is valid ONLY when its contract version,
    its ``conversation_id`` and its ``revision`` all match THIS
    request. A foreign Unified that happens to carry the same numeric
    revision is never accepted.
    """

    if (
        not isinstance(unified, dict)
        or unified.get("version") != _UNIFIED_VERSION
        or unified.get("conversation_id")
        != expected_conversation_id
        or unified.get("revision") != expected_revision
    ):
        return None

    sections = unified.get("sections")

    if not isinstance(sections, dict):
        return None

    memories = sections.get("memories")

    if not isinstance(memories, list):
        return None

    return memories


def _base_policy_binding(
    base_policy: Any,
    *,
    conversation_id: Any,
    expected_revision: Any,
) -> tuple[bool, list | None]:
    """Strictly bound base Surfacing Policy eligibility.

    Returns ``(available, eligible_list)``. ``available`` is False
    whenever the policy is missing, malformed, from another
    conversation, or bound to another Unified revision -- in which case
    eligibility is UNKNOWN (``eligible_list`` is None) and must never be
    inferred as ``not eligible``. When the policy IS validly bound,
    ``eligible_list`` is the real eligible list (an empty list when the
    policy decided ``no_surface``), so a real ``no_surface`` correctly
    means "not eligible".
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

    if not isinstance(decision, str):
        # A malformed decision is UNKNOWN, never "not eligible".
        return False, None

    if decision != _ALLOW_DECISION:
        return True, []

    eligible = base_policy.get("eligible")

    if not isinstance(eligible, list):
        # A valid allow_shadow policy always carries an eligible list;
        # anything else is malformed, so eligibility is UNKNOWN.
        return False, None

    return True, eligible


def _legacy_view(
    observation: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Normalised Legacy fields for one observation."""

    preview = observation.get("legacy")
    available = bool(
        observation.get("legacy_available")
    ) and isinstance(preview, dict)

    if not available:
        return (
            {
                "available": False,
                "applicable": False,
                "skip_reason": None,
                "current_resolved": False,
                "auto_resolve_candidate": False,
                "score_current": None,
                "score_if_auto_resolve_succeeds":
                    None,
                "threshold": None,
                "archive_candidate_current": False,
                "archive_candidate_after_auto_resolve_if_successful":
                    False,
                "nominal_archive_candidate": False,
            },
            False,
        )

    threshold = _finite(preview.get("threshold"))

    return (
        {
            "available": True,
            "applicable": bool(
                preview.get("applicable")
            ),
            "skip_reason": (
                preview.get("skip_reason")
                if isinstance(
                    preview.get("skip_reason"),
                    str,
                )
                else None
            ),
            "current_resolved": bool(
                preview.get("current_resolved")
            ),
            "auto_resolve_candidate": bool(
                preview.get(
                    "auto_resolve_candidate"
                )
            ),
            "score_current": _finite(
                preview.get("score_current")
            ),
            "score_if_auto_resolve_succeeds":
                _finite(
                    preview.get(
                        "score_if_auto_resolve_succeeds"
                    )
                ),
            "threshold": threshold,
            "archive_candidate_current": bool(
                preview.get(
                    "archive_candidate_current"
                )
            ),
            "archive_candidate_after_auto_resolve_if_successful":
                bool(
                    preview.get(
                        "archive_candidate_after_auto_resolve_if_successful"
                    )
                ),
            "nominal_archive_candidate": bool(
                preview.get(
                    "nominal_archive_candidate"
                )
            ),
        },
        True,
    )


def _cognitive_view(
    memory_id: Any,
    *,
    as_of: datetime,
    config: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], bool]:
    """Read-only cognitive evaluation of one memory.

    Returns ``(view, evidence_available)`` where the view carries the
    tri-state ``available`` / ``observed`` contract:

      available=True,  observed=True   valid Used history observed
      available=True,  observed=False  evidence read, no valid Used
                                       history -> neutral_unobserved
      available=False, observed=None   evidence could not be evaluated
                                       -> cognitive_evidence_unavailable

    Unavailable is NEVER represented as ``observed=False``: unknown is
    not the same as "no history".
    """

    def _unavailable() -> dict[str, Any]:
        return {
            "available": False,
            "observed": None,
            "use_count": 0,
            "strength": None,
            "accessibility": None,
            "salience": None,
            "age_since_last_used_hours": None,
            "min_accessibility": thresholds[
                "min_accessibility"
            ],
            "min_salience": thresholds[
                "min_salience"
            ],
            "would_suppress_if_supplementary": False,
            "would_keep_if_supplementary": False,
            "invalid_event_count": 0,
        }

    # No usable identity: the evidence chain for this candidate cannot
    # be read at all -> unavailable, not "no history".
    if not isinstance(memory_id, str) or not memory_id:
        return _unavailable(), False

    try:
        result = (
            evaluate_lifecycle_surfacing_candidate(
                candidate={"id": memory_id},
                as_of=as_of,
                config=config,
                thresholds=thresholds,
            )
        )
    except Exception:
        return _unavailable(), False

    if not isinstance(result, dict):
        return _unavailable(), False

    observed = bool(result.get("observed"))

    invalid = result.get("invalid_event_count")

    if (
        not isinstance(invalid, int)
        or isinstance(invalid, bool)
    ):
        invalid = 0

    if not observed:
        return (
            {
                "available": True,
                "observed": False,
                "use_count": 0,
                "strength": None,
                "accessibility": None,
                "salience": None,
                "age_since_last_used_hours":
                    None,
                "min_accessibility": thresholds[
                    "min_accessibility"
                ],
                "min_salience": thresholds[
                    "min_salience"
                ],
                "would_suppress_if_supplementary":
                    False,
                "would_keep_if_supplementary":
                    False,
                "invalid_event_count": invalid,
            },
            True,
        )

    kept = bool(result.get("kept"))

    return (
        {
            "available": True,
            "observed": True,
            "use_count": int(
                result.get("use_count") or 0
            ),
            "strength": _finite(
                result.get("strength")
            ),
            "accessibility": _finite(
                result.get("accessibility")
            ),
            "salience": _finite(
                result.get("salience")
            ),
            "age_since_last_used_hours":
                _finite(
                    result.get(
                        "age_since_last_used_hours"
                    )
                ),
            "min_accessibility": thresholds[
                "min_accessibility"
            ],
            "min_salience": thresholds[
                "min_salience"
            ],
            "would_suppress_if_supplementary":
                not kept,
            "would_keep_if_supplementary":
                kept,
            "invalid_event_count": invalid,
        },
        True,
    )


def _counterfactual_gate_decision(
    *,
    binding_available: bool,
    base_policy_available: bool,
    base_eligible: bool,
    base_rank: int | None,
    cognitive_available: bool,
    cognitive: dict[str, Any],
) -> str:
    """What the CURRENT lifecycle gate would decide IF applied.

    This is a COUNTERFACTUAL: it is computed regardless of whether the
    live Lifecycle Surfacing gate is enabled. It is never labelled an
    actual live effect (see ``effective_live_gate_decision``).

    The primary (rank 0) base-eligible candidate is ALWAYS preserved,
    so ``would_suppress_if_supplementary`` and this decision are
    deliberately represented separately. A missing base policy or
    missing cognitive evidence is UNKNOWN, never "suppressed" and never
    "not eligible".
    """

    if not binding_available:
        return _GATE_BINDING_UNAVAILABLE

    if not base_policy_available:
        return _GATE_BASE_POLICY_UNAVAILABLE

    if not base_eligible:
        return _GATE_NOT_BASE_ELIGIBLE

    if base_rank == 0:
        return _GATE_PRIMARY

    if not cognitive_available:
        return _GATE_COGNITIVE_EVIDENCE_UNAVAILABLE

    if not cognitive.get("observed"):
        return _GATE_NEUTRAL

    if cognitive.get("would_keep_if_supplementary"):
        return _GATE_KEEP

    return _GATE_SUPPRESS


def _effective_live_gate_decision(
    counterfactual: str,
    *,
    live_gate_active: bool,
) -> str:
    """The actual current lifecycle gate participation.

    When the live gate cannot change anything under its current flags,
    no counterfactual suppression is an actual live effect, so this is
    ``gate_disabled``. Otherwise the real gate would apply the
    counterfactual decision.
    """

    if not live_gate_active:
        return _GATE_DISABLED

    return counterfactual


def _category(
    *,
    binding_available: bool,
    legacy: dict[str, Any],
    legacy_available: bool,
    cognitive_available: bool,
    cognitive: dict[str, Any],
) -> str:
    if not binding_available:
        return _CAT_BINDING_UNAVAILABLE

    if not legacy_available:
        return _CAT_LEGACY_UNAVAILABLE

    if not legacy.get("applicable"):
        return _CAT_LEGACY_NOT_APPLICABLE

    if not cognitive_available:
        return _CAT_COGNITIVE_UNAVAILABLE

    legacy_side = (
        _LEG_ARCHIVE
        if legacy.get("nominal_archive_candidate")
        else _LEG_KEEP
    )

    if not cognitive.get("observed"):
        cognitive_side = _COG_UNOBSERVED
    elif cognitive.get(
        "would_suppress_if_supplementary"
    ):
        cognitive_side = _COG_SUPPRESS
    else:
        cognitive_side = _COG_KEEP

    return (
        "legacy_"
        + legacy_side
        + "_cognitive_"
        + cognitive_side
    )


def _round(value: float | None) -> float | None:
    if value is None:
        return None

    return round(value, 6)


def _index(value: Any) -> int:
    """Nonnegative retrieval index, or 0 when the evidence is invalid."""

    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    return 0


def _finalize_legacy(
    legacy: dict[str, Any],
) -> dict[str, Any]:
    row = dict(legacy)
    row["score_current"] = _round(
        legacy.get("score_current")
    )
    row["score_if_auto_resolve_succeeds"] = _round(
        legacy.get(
            "score_if_auto_resolve_succeeds"
        )
    )
    return row


def _finalize_cognitive(
    cognitive: dict[str, Any],
    *,
    counterfactual: str,
    live_gate_active: bool,
) -> dict[str, Any]:
    row = dict(cognitive)
    row["strength"] = _round(
        cognitive.get("strength")
    )
    row["accessibility"] = _round(
        cognitive.get("accessibility")
    )
    row["salience"] = _round(
        cognitive.get("salience")
    )
    row["age_since_last_used_hours"] = _round(
        cognitive.get(
            "age_since_last_used_hours"
        )
    )
    row["counterfactual_gate_decision"] = (
        counterfactual
    )
    row["effective_live_gate_decision"] = (
        _effective_live_gate_decision(
            counterfactual,
            live_gate_active=live_gate_active,
        )
    )
    return row


def build_lifecycle_comparison_artifact(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    source_unified_revision: Any,
    unified: Any,
    base_policy: Any,
    legacy_observations: Any,
    retrieval_candidate_count: int,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Build the privacy-safe, request-scoped comparison artifact.

    Read-only with respect to every memory authority (it reads THIS
    request's Unified snapshot and the existing cognitive lifecycle
    event files) but NOT mathematically pure: its Cognitive evaluation
    performs read-only IO against immutable lifecycle events. It writes
    nothing but the returned dict.
    """

    moment = as_of or datetime.now(timezone.utc)

    if (
        moment.tzinfo is None
        or moment.utcoffset() is None
    ):
        moment = moment.replace(
            tzinfo=timezone.utc
        )

    observations = (
        legacy_observations
        if isinstance(
            legacy_observations, list
        )
        else []
    )

    unified_memories = _unified_memories(
        unified,
        expected_conversation_id=(
            conversation_id
        ),
        expected_revision=(
            source_unified_revision
        ),
    )

    binding_available = unified_memories is not None

    admitted_queue = _fingerprint_queues(
        unified_memories
        if unified_memories is not None
        else []
    )

    (
        base_policy_available,
        eligible_list,
    ) = _base_policy_binding(
        base_policy,
        conversation_id=conversation_id,
        expected_revision=(
            source_unified_revision
        ),
    )

    eligible_queue = _fingerprint_queues(
        eligible_list
        if eligible_list is not None
        else []
    )

    # Actual live-gate participation under the CURRENT flags. It only
    # records whether the real gate could change behaviour; it never
    # changes the counterfactual analysis.
    live_gate_active = bool(
        lifecycle_surfacing_enabled()
        and lifecycle_shadow_enabled()
    )

    config = resolve_lifecycle_config()

    thresholds = (
        resolve_surfacing_lifecycle_config()
    )

    rows: list[dict[str, Any]] = []

    for observation in observations:
        if not isinstance(observation, dict):
            continue

        if len(rows) >= MAX_COMPARISON_ROWS:
            break

        retrieval_index = _index(
            observation.get("retrieval_index")
        )

        fingerprint = observation.get(
            "candidate_fingerprint"
        )

        valid_fingerprint = (
            isinstance(fingerprint, str)
            and _SHA256_HEX.match(fingerprint)
            is not None
        )

        key = memory_key(
            observation.get("memory_id")
        )

        if not valid_fingerprint or key is None:
            # Cannot truthfully bind this retrieval candidate's
            # identity, so its eligibility is UNKNOWN.
            unavailable_cognitive, _ = (
                _cognitive_view(
                    None,
                    as_of=moment,
                    config=config,
                    thresholds=thresholds,
                )
            )

            rows.append(
                {
                    "memory_key": None,
                    "candidate_fingerprint": None,
                    "retrieval_index":
                        retrieval_index,
                    "binding": {
                        "unified_admitted": None,
                        "base_policy_available":
                            False,
                        "base_surfacing_eligible":
                            None,
                        "base_eligible_rank": None,
                    },
                    "legacy": _finalize_legacy(
                        _legacy_view(observation)[0]
                    ),
                    "cognitive":
                        _finalize_cognitive(
                            unavailable_cognitive,
                            counterfactual=(
                                _GATE_BINDING_UNAVAILABLE
                            ),
                            live_gate_active=(
                                live_gate_active
                            ),
                        ),
                    "comparison": {
                        "category":
                            _CAT_BINDING_UNAVAILABLE
                    },
                }
            )

            continue

        memory_id = observation.get("memory_id")

        admitted_index = None
        queue = admitted_queue.get(fingerprint)

        if queue:
            admitted_index = queue.popleft()

        base_rank = None
        eligible_queue_for_fp = (
            eligible_queue.get(fingerprint)
        )

        if eligible_queue_for_fp:
            base_rank = (
                eligible_queue_for_fp.popleft()
            )

        if base_policy_available:
            base_eligible = base_rank is not None
            base_surfacing_eligible = (
                base_eligible
            )
            base_eligible_rank = base_rank
        else:
            # UNKNOWN base eligibility: never infer "not eligible".
            base_eligible = False
            base_surfacing_eligible = None
            base_eligible_rank = None

        legacy, legacy_available = _legacy_view(
            observation
        )

        cognitive, cognitive_available = (
            _cognitive_view(
                memory_id,
                as_of=moment,
                config=config,
                thresholds=thresholds,
            )
        )

        binding_ok = binding_available

        counterfactual = (
            _counterfactual_gate_decision(
                binding_available=binding_ok,
                base_policy_available=(
                    base_policy_available
                ),
                base_eligible=base_eligible,
                base_rank=base_rank,
                cognitive_available=(
                    cognitive_available
                ),
                cognitive=cognitive,
            )
        )

        category = _category(
            binding_available=binding_ok,
            legacy=legacy,
            legacy_available=legacy_available,
            cognitive_available=(
                cognitive_available
            ),
            cognitive=cognitive,
        )

        binding_row = {
            "unified_admitted": (
                (admitted_index is not None)
                if binding_ok
                else None
            ),
            "base_policy_available":
                base_policy_available,
            "base_surfacing_eligible":
                base_surfacing_eligible,
            "base_eligible_rank":
                base_eligible_rank,
        }

        rows.append(
            {
                "memory_key": key,
                "candidate_fingerprint":
                    fingerprint,
                "retrieval_index":
                    retrieval_index,
                "binding": binding_row,
                "legacy": _finalize_legacy(
                    legacy
                ),
                "cognitive": _finalize_cognitive(
                    cognitive,
                    counterfactual=counterfactual,
                    live_gate_active=(
                        live_gate_active
                    ),
                ),
                "comparison": {
                    "category": category
                },
            }
        )

    total = (
        retrieval_candidate_count
        if isinstance(
            retrieval_candidate_count, int
        )
        and not isinstance(
            retrieval_candidate_count, bool
        )
        and retrieval_candidate_count >= 0
        else len(observations)
    )

    persisted = len(rows)

    truncated_count = max(0, total - persisted)

    return {
        "version": VERSION,
        "mode": MODE,
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "source_unified_revision":
            source_unified_revision,
        "created_at": now_iso(),
        "live_lifecycle_gate_active":
            live_gate_active,
        "observation_truncated":
            truncated_count > 0,
        "observation_truncated_count":
            truncated_count,
        "metrics": _metrics(
            rows,
            retrieval_candidate_count=total,
            persisted_count=persisted,
            truncated_count=truncated_count,
        ),
        "candidates": rows,
    }


def _count(
    rows: list[dict[str, Any]],
    predicate,
) -> int:
    return sum(1 for row in rows if predicate(row))


def _auto_resolve_direction(
    legacy: dict[str, Any],
) -> str | None:
    """Whether a successful auto-resolve would change archive outcome.

    Direction is NOT assumed: the current formula may make one direction
    more likely, but the comparison must not encode that assumption.
    """

    if not legacy.get("auto_resolve_candidate"):
        return None

    current = bool(
        legacy.get("archive_candidate_current")
    )

    after = bool(
        legacy.get(
            "archive_candidate_after_auto_resolve_if_successful"
        )
    )

    if current == after:
        return None

    return "created" if after else "removed"


def _auto_resolve_changed(
    legacy: dict[str, Any],
) -> bool:
    return _auto_resolve_direction(legacy) is not None


def _metrics(
    rows: list[dict[str, Any]],
    *,
    retrieval_candidate_count: int,
    persisted_count: int,
    truncated_count: int,
) -> dict[str, Any]:
    def legacy(row):
        return row.get("legacy") or {}

    def cognitive(row):
        return row.get("cognitive") or {}

    def binding(row):
        return row.get("binding") or {}

    def category(row):
        return (row.get("comparison") or {}).get(
            "category"
        )

    return {
        "retrieval_candidate_count":
            retrieval_candidate_count,
        "observed_candidate_count":
            persisted_count,
        "observation_truncated_count":
            truncated_count,
        "legacy_available_count": _count(
            rows, lambda r: legacy(r).get(
                "available"
            )
        ),
        "legacy_applicable_count": _count(
            rows, lambda r: legacy(r).get(
                "applicable"
            )
        ),
        "legacy_not_applicable_count":
            _count(
                rows,
                lambda r: legacy(r).get(
                    "available"
                )
                and not legacy(r).get(
                    "applicable"
                ),
            ),
        "legacy_auto_resolve_candidate_count":
            _count(
                rows,
                lambda r: legacy(r).get(
                    "auto_resolve_candidate"
                ),
            ),
        "legacy_nominal_archive_candidate_count":
            _count(
                rows,
                lambda r: legacy(r).get(
                    "nominal_archive_candidate"
                ),
            ),
        "legacy_auto_resolve_changed_archive_outcome_count":
            _count(
                rows,
                lambda r: _auto_resolve_changed(
                    legacy(r)
                ),
            ),
        "legacy_auto_resolve_created_archive_candidate_count":
            _count(
                rows,
                lambda r: _auto_resolve_direction(
                    legacy(r)
                )
                == "created",
            ),
        "legacy_auto_resolve_removed_archive_candidate_count":
            _count(
                rows,
                lambda r: _auto_resolve_direction(
                    legacy(r)
                )
                == "removed",
            ),
        "cognitive_available_count": _count(
            rows,
            lambda r: cognitive(r).get(
                "available"
            )
            is True,
        ),
        # The three Cognitive representation counts partition the rows
        # for which a Cognitive evaluation was attempted and
        # represented. binding_unavailable rows are the DOCUMENTED
        # exclusion: they carry no usable identity, so they are never
        # counted as observed, neutral OR evidence_unavailable (an
        # unknown identity must not inflate "evidence unavailable"
        # either).
        "cognitive_observed_count": _count(
            rows,
            lambda r: category(r)
            != _CAT_BINDING_UNAVAILABLE
            and cognitive(r).get("available")
            is True
            and cognitive(r).get("observed")
            is True,
        ),
        "cognitive_neutral_unobserved_count":
            _count(
                rows,
                lambda r: category(r)
                != _CAT_BINDING_UNAVAILABLE
                and cognitive(r).get(
                    "available"
                )
                is True
                and cognitive(r).get(
                    "observed"
                )
                is False,
            ),
        "cognitive_evidence_unavailable_count":
            _count(
                rows,
                lambda r: category(r)
                != _CAT_BINDING_UNAVAILABLE
                and cognitive(r).get(
                    "available"
                )
                is False,
            ),
        "cognitive_represented_count": _count(
            rows,
            lambda r: category(r)
            != _CAT_BINDING_UNAVAILABLE,
        ),
        "cognitive_counterfactual_suppress_count":
            _count(
                rows,
                lambda r: cognitive(r).get(
                    "would_suppress_if_supplementary"
                ),
            ),
        "cognitive_counterfactual_gate_suppress_count":
            _count(
                rows,
                lambda r: cognitive(r).get(
                    "counterfactual_gate_decision"
                )
                == _GATE_SUPPRESS,
            ),
        "cognitive_live_gate_suppress_count":
            _count(
                rows,
                lambda r: cognitive(r).get(
                    "effective_live_gate_decision"
                )
                == _GATE_SUPPRESS,
            ),
        "cognitive_primary_preserved_count":
            _count(
                rows,
                lambda r: cognitive(r).get(
                    "counterfactual_gate_decision"
                )
                == _GATE_PRIMARY,
            ),
        "unified_admitted_count": _count(
            rows,
            lambda r: binding(r).get(
                "unified_admitted"
            )
            is True,
        ),
        "base_surfacing_eligible_count":
            _count(
                rows,
                lambda r: binding(r).get(
                    "base_surfacing_eligible"
                )
                is True,
            ),
        "base_policy_unavailable_count":
            _count(
                rows,
                lambda r: binding(r).get(
                    "base_policy_available"
                )
                is False,
            ),
        "both_archive_and_suppress_count":
            _count(
                rows,
                lambda r: legacy(r).get(
                    "nominal_archive_candidate"
                )
                and cognitive(r).get(
                    "would_suppress_if_supplementary"
                ),
            ),
        "legacy_archive_cognitive_unobserved_count":
            _count(
                rows,
                lambda r: category(r)
                == "legacy_archive_cognitive_unobserved",
            ),
        "legacy_archive_cognitive_suppress_count":
            _count(
                rows,
                lambda r: category(r)
                == "legacy_archive_cognitive_suppress",
            ),
        "legacy_archive_cognitive_keep_count":
            _count(
                rows,
                lambda r: category(r)
                == "legacy_archive_cognitive_keep",
            ),
        "legacy_keep_cognitive_unobserved_count":
            _count(
                rows,
                lambda r: category(r)
                == "legacy_keep_cognitive_unobserved",
            ),
        "legacy_keep_cognitive_suppress_count":
            _count(
                rows,
                lambda r: category(r)
                == "legacy_keep_cognitive_suppress",
            ),
        "legacy_keep_cognitive_keep_count":
            _count(
                rows,
                lambda r: category(r)
                == "legacy_keep_cognitive_keep",
            ),
        "binding_unavailable_count": _count(
            rows,
            lambda r: category(r)
            == _CAT_BINDING_UNAVAILABLE,
        ),
        "invalid_event_count": sum(
            int(cognitive(r).get(
                "invalid_event_count"
            ) or 0)
            for r in rows
        ),
    }


# ------------------------------------------------------------------
# Artifact validator (single authoritative v1 contract)
# ------------------------------------------------------------------


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and _SHA256_HEX.match(value) is not None
    )


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_nonneg_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _is_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_finite_or_none(value: Any) -> bool:
    return value is None or _is_finite(value)


def _is_unit(value: Any) -> bool:
    return _is_finite(value) and (
        0.0 <= float(value) <= 1.0
    )


def _valid_legacy_row(legacy: Any) -> bool:
    if not isinstance(legacy, dict):
        return False

    if not _is_bool(legacy.get("available")):
        return False

    if not _is_bool(legacy.get("applicable")):
        return False

    skip_reason = legacy.get("skip_reason")

    if not (
        skip_reason is None
        or isinstance(skip_reason, str)
    ):
        return False

    for name in (
        "current_resolved",
        "auto_resolve_candidate",
        "archive_candidate_current",
        "archive_candidate_after_auto_resolve_if_successful",
        "nominal_archive_candidate",
    ):
        if not _is_bool(legacy.get(name)):
            return False

    if not _is_finite_or_none(
        legacy.get("score_current")
    ):
        return False

    if not _is_finite_or_none(
        legacy.get("score_if_auto_resolve_succeeds")
    ):
        return False

    if not _is_finite_or_none(
        legacy.get("threshold")
    ):
        return False

    if legacy.get("available") is False:
        for name in (
            "score_current",
            "score_if_auto_resolve_succeeds",
            "threshold",
        ):
            if legacy.get(name) is not None:
                return False

    return True


def _valid_cognitive_row(cognitive: Any) -> bool:
    if not isinstance(cognitive, dict):
        return False

    available = cognitive.get("available")
    observed = cognitive.get("observed")

    if not _is_bool(available):
        return False

    if observed is not None and not _is_bool(
        observed
    ):
        return False

    # available / observed tri-state contract.
    if available is False and observed is not None:
        return False

    if available is True and observed is None:
        return False

    if not _is_nonneg_int(
        cognitive.get("use_count")
    ):
        return False

    if not _is_finite_or_none(
        cognitive.get("strength")
    ):
        return False

    if not _is_finite_or_none(
        cognitive.get("accessibility")
    ):
        return False

    if not _is_finite_or_none(
        cognitive.get("salience")
    ):
        return False

    age = cognitive.get("age_since_last_used_hours")

    if age is not None:
        if not _is_finite(age):
            return False

        if float(age) < 0.0:
            return False

    if not _is_unit(
        cognitive.get("min_accessibility")
    ):
        return False

    if not _is_unit(
        cognitive.get("min_salience")
    ):
        return False

    if not _is_bool(
        cognitive.get(
            "would_suppress_if_supplementary"
        )
    ):
        return False

    if not _is_bool(
        cognitive.get("would_keep_if_supplementary")
    ):
        return False

    if not _is_nonneg_int(
        cognitive.get("invalid_event_count")
    ):
        return False

    if cognitive.get(
        "counterfactual_gate_decision"
    ) not in _KNOWN_COUNTERFACTUAL_DECISIONS:
        return False

    if cognitive.get(
        "effective_live_gate_decision"
    ) not in _KNOWN_LIVE_GATE_DECISIONS:
        return False

    lifecycle_numeric = (
        "strength",
        "accessibility",
        "salience",
        "age_since_last_used_hours",
    )

    if observed is False:
        for name in lifecycle_numeric:
            if cognitive.get(name) is not None:
                return False

        if (
            cognitive.get(
                "would_suppress_if_supplementary"
            )
            is not False
        ):
            return False

    if observed is True:
        if cognitive.get("use_count", 0) <= 0:
            return False

        for name in (
            "strength",
            "accessibility",
            "salience",
        ):
            if cognitive.get(name) is None:
                return False

    return True


def _valid_binding_row(binding: Any) -> bool:
    if not isinstance(binding, dict):
        return False

    unified_admitted = binding.get(
        "unified_admitted"
    )

    if (
        unified_admitted is not None
        and not _is_bool(unified_admitted)
    ):
        return False

    base_policy_available = binding.get(
        "base_policy_available"
    )

    if not _is_bool(base_policy_available):
        return False

    eligible = binding.get(
        "base_surfacing_eligible"
    )

    if eligible is not None and not _is_bool(
        eligible
    ):
        return False

    rank = binding.get("base_eligible_rank")

    if rank is not None and not _is_nonneg_int(rank):
        return False

    if base_policy_available is False:
        if eligible is not None or rank is not None:
            return False
    else:
        if not _is_bool(eligible):
            return False

        if eligible is False and rank is not None:
            return False

        if eligible is True and rank is None:
            return False

    return True


def _valid_candidate_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False

    if not _is_nonneg_int(row.get("retrieval_index")):
        return False

    comparison = row.get("comparison")

    if not isinstance(comparison, dict):
        return False

    category = comparison.get("category")

    if category not in _KNOWN_COMPARISON_CATEGORIES:
        return False

    fingerprint = row.get("candidate_fingerprint")
    key = row.get("memory_key")

    if fingerprint is None:
        # Only the exact binding-unavailable shape may omit identity.
        if key is not None:
            return False

        if category != _CAT_BINDING_UNAVAILABLE:
            return False
    else:
        if not _is_sha256(fingerprint):
            return False

        if not _is_sha256(key):
            return False

    if not _valid_binding_row(row.get("binding")):
        return False

    if not _valid_legacy_row(row.get("legacy")):
        return False

    if not _valid_cognitive_row(
        row.get("cognitive")
    ):
        return False

    return True


def is_valid_lifecycle_comparison_artifact(
    artifact: Any,
) -> bool:
    """The single authoritative v1 comparison-artifact validator.

    A complete, structurally sound ``memory-lifecycle-comparison-shadow.v1``
    artifact is valid; anything else (wrong version / mode, malformed
    identity, non-finite numbers, an inconsistent binding or
    cognitive tri-state, an unknown category or gate decision, or one
    malformed row among otherwise valid rows) is INVALID as a whole.
    Future / newer versions are unsupported for this v1 reporter and are
    therefore invalid -- they are never partially interpreted.
    """

    if not isinstance(artifact, dict):
        return False

    if artifact.get("version") != VERSION:
        return False

    if artifact.get("mode") != MODE:
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

    if parse_iso_utc(artifact.get("created_at")) is None:
        return False

    if not _is_bool(
        artifact.get("live_lifecycle_gate_active")
    ):
        return False

    if not _is_bool(
        artifact.get("observation_truncated")
    ):
        return False

    if not _is_nonneg_int(
        artifact.get("observation_truncated_count")
    ):
        return False

    if not isinstance(artifact.get("metrics"), dict):
        return False

    candidates = artifact.get("candidates")

    if not isinstance(candidates, list):
        return False

    if len(candidates) > MAX_COMPARISON_ROWS:
        return False

    for row in candidates:
        if not _valid_candidate_row(row):
            return False

    return True


# ------------------------------------------------------------------
# Observer entry point (called by the Context coordinator)
# ------------------------------------------------------------------


def _retrieval_candidate_count(
    snapshot: Any,
) -> int:
    evidence = (
        snapshot.get("evidence")
        if isinstance(snapshot, dict)
        else None
    )

    if isinstance(evidence, list):
        return len(evidence)

    return 0


def update_memory_lifecycle_comparison(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    snapshot: Any,
    expected_unified_revision: Any,
    base_policy: Any,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Observe + persist one request's Legacy vs Cognitive comparison.

    Never raises. Returns a privacy-safe report. ``stored`` is False
    (with a reason) whenever a truthful comparison cannot be made or
    the artifact cannot be written -- comparison failure is missing
    comparison data and NOTHING else. It consumes THIS request's
    request-local snapshot only and never re-reads a conversation-global
    Unified file.
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
            "reason":
                "invalid_cognitive_request_id",
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

    if (
        not isinstance(snapshot, dict)
        or snapshot.get("version")
        != _SNAPSHOT_VERSION
        or snapshot.get("conversation_id")
        != conversation_id
    ):
        return {
            "stored": False,
            "mode": MODE,
            "decision": "no_record",
            "reason":
                "snapshot_unavailable",
        }

    unified = snapshot.get("unified")

    if (
        not isinstance(unified, dict)
        or unified.get("version")
        != _UNIFIED_VERSION
        or unified.get("conversation_id")
        != conversation_id
        or unified.get("revision")
        != expected_unified_revision
    ):
        # A foreign Unified -- even one carrying the SAME numeric
        # revision -- is never request-local authority. Fail closed
        # for EVIDENCE only (never for the live request) and never
        # fall back to a shared latest file.
        return {
            "stored": False,
            "mode": MODE,
            "decision": "no_record",
            "reason": "unified_binding_invalid",
        }

    artifact = build_lifecycle_comparison_artifact(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        source_unified_revision=(
            expected_unified_revision
        ),
        unified=unified,
        base_policy=base_policy,
        legacy_observations=snapshot.get(
            "legacy_lifecycle_observations"
        ),
        retrieval_candidate_count=(
            _retrieval_candidate_count(snapshot)
        ),
        as_of=as_of,
    )

    # Never persist malformed evidence: validate with the SAME v1
    # validator the reporter uses. Failure stays fail-open relative to
    # the live request.
    if not is_valid_lifecycle_comparison_artifact(
        artifact
    ):
        logger.warning(
            "[gateway.context_memory_lifecycle_comparison] "
            "reason=invalid_comparison_artifact "
            "fail_open=true"
        )

        return {
            "stored": False,
            "mode": MODE,
            "decision": "no_record",
            "reason": "invalid_comparison_artifact",
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
        # Fail-open: comparison data is simply missing.
        logger.warning(
            "[gateway.context_memory_lifecycle_comparison] "
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
        "decision": "observed",
        "reason": "lifecycle_comparison_observed",
        "source_unified_revision":
            expected_unified_revision,
        "metrics": artifact["metrics"],
    }


def log_lifecycle_comparison_report(
    report: dict[str, Any],
) -> None:
    """Privacy-safe telemetry: aggregate counts and reasons only.

    No conversation id, cognitive request id, memory key, candidate
    fingerprint, raw memory id or memory text is ever logged.
    """

    metrics = (
        report.get("metrics")
        if isinstance(report, dict)
        else None
    )

    logger.info(
        "[gateway.context_memory_lifecycle_comparison] %s",
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
                "source_unified_revision": (
                    report.get(
                        "source_unified_revision"
                    )
                    if isinstance(report, dict)
                    else None
                ),
                "metrics": metrics,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )