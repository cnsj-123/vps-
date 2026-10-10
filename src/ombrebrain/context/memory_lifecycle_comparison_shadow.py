from __future__ import annotations

import json
import logging
import math
import os
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
)
from ombrebrain.context.memory_lifecycle_state import (
    resolve_lifecycle_config,
)
from ombrebrain.context.memory_surfacing_lifecycle import (
    evaluate_lifecycle_surfacing_candidate,
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

# Effective gate decisions (what the ACTUAL live gate would do).
_GATE_PRIMARY = "primary_preserved"
_GATE_NEUTRAL = "neutral_unobserved"
_GATE_KEEP = "would_keep"
_GATE_SUPPRESS = "would_suppress"
_GATE_NOT_BASE_ELIGIBLE = "not_base_eligible"
_GATE_BINDING_UNAVAILABLE = "binding_unavailable"

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
    expected_revision: Any,
) -> list | None:
    """THIS request's Unified admitted memories, or None if unusable."""

    if (
        not isinstance(unified, dict)
        or unified.get("version") != _UNIFIED_VERSION
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


def _base_eligible(
    base_policy: Any,
) -> list | None:
    """The base policy's eligible list, or None if not a real allow."""

    if not isinstance(base_policy, dict):
        return None

    if (
        base_policy.get("version") != _POLICY_VERSION
        or base_policy.get("mode") != _POLICY_MODE
    ):
        return None

    if (
        base_policy.get("decision")
        != _ALLOW_DECISION
    ):
        return []

    eligible = base_policy.get("eligible")

    if not isinstance(eligible, list):
        return []

    return eligible


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
    """Read-only cognitive evaluation of one memory, or unavailable."""

    unavailable = {
        "observed": False,
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
        return unavailable, False

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


def _effective_gate_decision(
    *,
    admitted: bool,
    base_eligible: bool,
    base_rank: int | None,
    binding_available: bool,
    cognitive: dict[str, Any],
) -> str:
    """What the ACTUAL live gate would decide for this candidate.

    The primary (rank 0) base-eligible candidate is ALWAYS preserved,
    so ``would_suppress_if_supplementary`` and the effective gate
    decision are deliberately represented separately.
    """

    if not binding_available:
        return _GATE_BINDING_UNAVAILABLE

    if not base_eligible:
        return _GATE_NOT_BASE_ELIGIBLE

    if base_rank == 0:
        return _GATE_PRIMARY

    if not cognitive.get("observed"):
        return _GATE_NEUTRAL

    if cognitive.get("would_keep_if_supplementary"):
        return _GATE_KEEP

    return _GATE_SUPPRESS


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

    Pure and read-only with respect to every memory authority: it reads
    THIS request's Unified snapshot and the existing cognitive formulas
    and writes nothing but the returned dict.
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

    eligible_list = _base_eligible(base_policy)

    eligible_queue = _fingerprint_queues(
        eligible_list
        if eligible_list is not None
        else []
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

        fingerprint = observation.get(
            "candidate_fingerprint"
        )

        if (
            not isinstance(fingerprint, str)
            or not fingerprint
        ):
            # Cannot truthfully bind this retrieval candidate.
            rows.append(
                {
                    "memory_key": None,
                    "candidate_fingerprint": None,
                    "retrieval_index":
                        observation.get(
                            "retrieval_index"
                        ),
                    "binding": {
                        "unified_admitted": None,
                        "base_surfacing_eligible":
                            None,
                        "base_eligible_rank": None,
                    },
                    "legacy": _legacy_view(
                        observation
                    )[0],
                    "cognitive": _cognitive_view(
                        None,
                        as_of=moment,
                        config=config,
                        thresholds=thresholds,
                    )[0],
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

        base_eligible = base_rank is not None

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

        effective_gate_decision = (
            _effective_gate_decision(
                admitted=admitted_index
                is not None,
                base_eligible=base_eligible,
                base_rank=base_rank,
                binding_available=binding_ok,
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

        cognitive_row = dict(cognitive)
        cognitive_row["strength"] = _round(
            cognitive.get("strength")
        )
        cognitive_row[
            "accessibility"
        ] = _round(
            cognitive.get("accessibility")
        )
        cognitive_row["salience"] = _round(
            cognitive.get("salience")
        )
        cognitive_row[
            "age_since_last_used_hours"
        ] = _round(
            cognitive.get(
                "age_since_last_used_hours"
            )
        )
        cognitive_row[
            "effective_gate_decision"
        ] = effective_gate_decision

        legacy_row = dict(legacy)
        legacy_row["score_current"] = _round(
            legacy.get("score_current")
        )
        legacy_row[
            "score_if_auto_resolve_succeeds"
        ] = _round(
            legacy.get(
                "score_if_auto_resolve_succeeds"
            )
        )

        rows.append(
            {
                "memory_key": memory_key(
                    memory_id
                ),
                "candidate_fingerprint":
                    fingerprint,
                "retrieval_index":
                    observation.get(
                        "retrieval_index"
                    ),
                "binding": {
                    "unified_admitted":
                        admitted_index
                        is not None,
                    "base_surfacing_eligible":
                        base_eligible,
                    "base_eligible_rank":
                        base_rank,
                },
                "legacy": legacy_row,
                "cognitive": cognitive_row,
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
        "cognitive_observed_count": _count(
            rows,
            lambda r: cognitive(r).get(
                "observed"
            ),
        ),
        "cognitive_neutral_unobserved_count":
            _count(
                rows,
                lambda r: cognitive(r).get(
                    "observed"
                )
                is False,
            ),
        "cognitive_counterfactual_suppress_count":
            _count(
                rows,
                lambda r: cognitive(r).get(
                    "would_suppress_if_supplementary"
                ),
            ),
        "cognitive_effective_suppress_count":
            _count(
                rows,
                lambda r: cognitive(r).get(
                    "effective_gate_decision"
                )
                == _GATE_SUPPRESS,
            ),
        "cognitive_primary_preserved_count":
            _count(
                rows,
                lambda r: cognitive(r).get(
                    "effective_gate_decision"
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
        or unified.get("revision")
        != expected_unified_revision
    ):
        return {
            "stored": False,
            "mode": MODE,
            "decision": "no_record",
            "reason":
                "unified_observation_unavailable",
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