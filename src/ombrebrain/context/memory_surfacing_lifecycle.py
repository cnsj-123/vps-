from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

from ombrebrain.context.memory_flash import (
    candidate_memory_id,
)
from ombrebrain.context.memory_lifecycle_event import (
    LIFECYCLE_SHADOW_ENV,
    event_age_hours,
    lifecycle_use_facts,
    parse_iso_utc,
    read_lifecycle_events,
)
from ombrebrain.context.memory_lifecycle_state import (
    accessibility_for_age,
    resolve_lifecycle_config,
    salience_for_age,
    strength_for_use_count,
)


# Memory Surfacing Lifecycle v1 — shadow-only integration layer.
#
#     retrieved  ->  surfaced
#
# The existing Memory Surfacing Policy answers "which already-retrieved
# candidates are eligible to become a Memory Flash cue?" with a pure,
# rule-based, canonical-order decision. This module adds exactly ONE
# constraint on top of that answer and changes nothing else:
#
#     retrieved candidate set          (frozen)
#       -> Surfacing Policy base       (memory-surfacing-policy.v1)
#         -> lifecycle Surfacing gate  (this module)
#           -> Memory Flash            (memory-flash.v1)
#
#     strength      != accessibility
#     accessibility != salience
#     low accessibility != deleted
#     forgotten     != erased
#
# What it does NOT do, by construction:
#
#   - it never re-runs retrieval and never re-ranks candidates: the
#     retained candidates keep the exact canonical Retrieval order;
#   - it never changes the primary (first base-eligible) candidate:
#     that candidate is always preserved, so lifecycle can influence
#     ``supplementary`` surfacing but can never overturn Retrieval's
#     strongest candidate or produce a total lifecycle blackout;
#   - it never reads a canonical memory store: for a ``use_count > 0``
#     memory the last explicit use is a sufficient decay reference,
#     and a ``use_count == 0`` memory is neutral;
#   - it never trusts the derived Lifecycle State snapshot: state is a
#     cache carrying an old ``as_of``, so accessibility / salience are
#     re-derived here from the IMMUTABLE lifecycle events at the
#     CURRENT ``as_of``, reusing the frozen v1 formulas rather than a
#     second copy of the math;
#   - it never writes: no lifecycle event, no lifecycle state, no
#     source memory and no reinforcement;
#   - it never calls the reinforcement observer: surfacing is not
#     reinforcement, and nothing here is negative reinforcement.
#
# Cold-start neutrality is load-bearing. The vast majority of memories
# predate the lifecycle event system, so ``use_count == 0`` means "no
# lifecycle history", NOT "never used" and NOT "forgotten":
#
#     unobserved != forgotten
#
# A memory with no valid ``used`` lifecycle event is therefore always
# neutral -- this phase deliberately never derives an age from a
# created timestamp for it.
#
# It is fail-NEUTRAL, not fail-closed. Any failure (integration
# exception, invalid ``as_of``, unreadable events, unexpected formula
# error, flag mismatch) keeps the base policy eligibility untouched:
# a lifecycle fault must never make an otherwise surfaced memory
# disappear.
#
# Two flags are required and the integration flag is NOT a pipeline
# starter: the Memory Flash stage must already be running, and the
# lifecycle shadow must also be ON. While the lifecycle shadow is OFF
# the whole lifecycle subsystem takes no part in behaviour, so no old
# lifecycle state may quietly influence Surfacing.
#
# The two thresholds are a deterministic Surfacing policy v1, not a
# claim about human memory.

logger = logging.getLogger("ombre_brain.gateway")

SURFACING_LIFECYCLE_VERSION = (
    "memory-surfacing-lifecycle.v1"
)

MODE = "shadow_only"

# The integration flag. Default OFF.
INTEGRATION_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_SURFACING_LIFECYCLE"
)

# Bounded gate thresholds. Defaults are policy v1, not science.
MIN_ACCESSIBILITY_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_SURFACING_MIN_ACCESSIBILITY"
)

MIN_SALIENCE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_SURFACING_MIN_SALIENCE"
)

_DEFAULT_MIN_ACCESSIBILITY = 0.30
_DEFAULT_MIN_SALIENCE = 0.25

# The base policy contract this layer attaches to. The name is repeated
# here (rather than imported) because memory_surfacing_policy imports
# memory_flash, not the other way round.
_POLICY_VERSION = "memory-surfacing-policy.v1"
_POLICY_MODE = "shadow_only"
_ALLOW_DECISION = "allow_shadow"

_DECISION_APPLIED = "applied"
_DECISION_NEUTRAL = "neutral"

_REASON_APPLIED = "lifecycle_surfacing_applied"
_REASON_DISABLED = "lifecycle_disabled"
_REASON_SUPPRESSED = (
    "lifecycle_below_surface_threshold"
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def lifecycle_surfacing_enabled() -> bool:
    """May the lifecycle gate constrain Surfacing at all? Default OFF."""

    return _truthy(os.environ.get(INTEGRATION_ENV))


def lifecycle_shadow_enabled() -> bool:
    """Is the Memory Lifecycle subsystem participating in behaviour?"""

    return _truthy(
        os.environ.get(LIFECYCLE_SHADOW_ENV)
    )


def _bounded_unit(
    raw: Any,
    *,
    default: float,
) -> float:
    """Finite float in ``[0.0, 1.0]``. Invalid -> default, else clamp."""

    if raw is None:
        return default

    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default

    if not math.isfinite(value):
        return default

    if value < 0.0:
        return 0.0

    if value > 1.0:
        return 1.0

    return value


def resolve_surfacing_lifecycle_config() -> (
    dict[str, float]
):
    """Bounded gate thresholds. Never raises."""

    return {
        "min_accessibility": _bounded_unit(
            os.environ.get(
                MIN_ACCESSIBILITY_ENV
            ),
            default=_DEFAULT_MIN_ACCESSIBILITY,
        ),
        "min_salience": _bounded_unit(
            os.environ.get(MIN_SALIENCE_ENV),
            default=_DEFAULT_MIN_SALIENCE,
        ),
    }


def _resolve_as_of(
    as_of: Any,
) -> datetime | None:
    """``None`` -> now UTC. A fixed, timezone-aware value otherwise."""

    if as_of is None:
        return datetime.now(timezone.utc)

    if isinstance(as_of, datetime):
        if (
            as_of.tzinfo is None
            or as_of.utcoffset() is None
        ):
            return None

        return as_of.astimezone(timezone.utc)

    return parse_iso_utc(as_of)


def _summary(
    *,
    decision: str,
    reason: str,
    thresholds: dict[str, float],
    evaluated_count: int = 0,
    observed_count: int = 0,
    neutral_count: int = 0,
    passed_count: int = 0,
    blocked_count: int = 0,
    primary_preserved_count: int = 0,
    invalid_event_count: int = 0,
) -> dict[str, Any]:
    """Request-level, privacy-safe summary. No per-memory state."""

    return {
        "version":
            SURFACING_LIFECYCLE_VERSION,
        "mode":
            MODE,
        "decision":
            decision,
        "reason":
            reason,
        "evaluated_count":
            evaluated_count,
        "observed_count":
            observed_count,
        "neutral_count":
            neutral_count,
        "passed_count":
            passed_count,
        "blocked_count":
            blocked_count,
        "primary_preserved_count":
            primary_preserved_count,
        "invalid_event_count":
            invalid_event_count,
        "min_accessibility":
            thresholds["min_accessibility"],
        "min_salience":
            thresholds["min_salience"],
    }


def _neutral(
    policy_report: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """An unchanged copy carrying a neutral summary (no suppression)."""

    output = dict(policy_report)

    output["lifecycle_surface"] = _summary(
        decision=_DECISION_NEUTRAL,
        reason=reason,
        thresholds=(
            resolve_surfacing_lifecycle_config()
        ),
    )

    return output


def _log(summary: dict[str, Any]) -> None:
    """Privacy-safe telemetry: counts, reasons and thresholds only."""

    logger.info(
        "[gateway.context_memory_surfacing_lifecycle] %s",
        json.dumps(
            {
                "mode":
                    summary.get("mode"),
                "decision":
                    summary.get("decision"),
                "reason":
                    summary.get("reason"),
                "evaluated_count":
                    summary.get(
                        "evaluated_count"
                    ),
                "observed_count":
                    summary.get(
                        "observed_count"
                    ),
                "neutral_count":
                    summary.get(
                        "neutral_count"
                    ),
                "passed_count":
                    summary.get("passed_count"),
                "blocked_count":
                    summary.get("blocked_count"),
                "primary_preserved_count":
                    summary.get(
                        "primary_preserved_count"
                    ),
                "invalid_event_count":
                    summary.get(
                        "invalid_event_count"
                    ),
                "min_accessibility":
                    summary.get(
                        "min_accessibility"
                    ),
                "min_salience":
                    summary.get(
                        "min_salience"
                    ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _evaluate_supplementary(
    *,
    candidate: Any,
    as_of: datetime,
    config: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[bool, bool, int]:
    """One supplementary candidate: (observed, kept, invalid_count).

    Unobserved (no valid used lifecycle event) is always kept --
    ``unobserved != forgotten``. The derived Lifecycle State snapshot
    is never consulted; accessibility / salience come from the
    immutable events at ``as_of``.
    """

    memory_id = candidate_memory_id(candidate)

    if memory_id is None:
        return False, True, 0

    store = read_lifecycle_events(memory_id)

    events = (
        store.get("events")
        if isinstance(store, dict)
        else None
    )

    invalid_count = (
        store.get("invalid_event_count")
        if isinstance(store, dict)
        else 0
    )

    if not isinstance(invalid_count, int) or (
        isinstance(invalid_count, bool)
    ):
        invalid_count = 0

    if not isinstance(events, list) or not events:
        # Zero valid events: neutral, never suppressed.
        return False, True, invalid_count

    facts = lifecycle_use_facts(
        events, as_of=as_of
    )

    use_count = facts["use_count"]
    last_used = facts["last_used_at"]

    if use_count <= 0 or last_used is None:
        return False, True, invalid_count

    strength = strength_for_use_count(
        use_count, config=config
    )

    age_since_last_used = event_age_hours(
        at=last_used, as_of=as_of
    )

    accessibility = accessibility_for_age(
        strength,
        age_since_last_used,
        config=config,
    )

    salience = salience_for_age(
        use_count=use_count,
        age_since_last_used_hours=(
            age_since_last_used
        ),
        config=config,
    )

    # Long-term trace OR recent prominence. Never AND.
    kept = (
        accessibility
        >= thresholds["min_accessibility"]
        or salience >= thresholds["min_salience"]
    )

    return True, kept, invalid_count


def _apply_gate(
    policy_report: dict[str, Any],
    *,
    as_of: datetime,
) -> dict[str, Any]:
    """Filter supplementary candidates in place of a copy.

    Only the base-eligible candidates are read (each one's own event
    directory); nothing is scanned globally and no canonical memory is
    touched.
    """

    config = resolve_lifecycle_config()

    thresholds = (
        resolve_surfacing_lifecycle_config()
    )

    eligible = policy_report.get("eligible")

    if not isinstance(eligible, list) or not eligible:
        return dict(policy_report)

    retained: list[Any] = []
    suppressed: list[dict[str, Any]] = []

    evaluated_count = 0
    observed_count = 0
    neutral_count = 0
    passed_count = 0
    blocked_count = 0
    primary_preserved_count = 0
    invalid_event_count = 0

    for rank, candidate in enumerate(eligible):
        if rank == 0:
            # Primary preservation: Retrieval's strongest current
            # candidate is never lifecycle-suppressed.
            primary_preserved_count = 1
            retained.append(candidate)
            continue

        evaluated_count += 1

        observed, kept, invalid = (
            _evaluate_supplementary(
                candidate=candidate,
                as_of=as_of,
                config=config,
                thresholds=thresholds,
            )
        )

        invalid_event_count += invalid

        if not observed:
            neutral_count += 1
            retained.append(candidate)
            continue

        observed_count += 1

        if kept:
            passed_count += 1
            retained.append(candidate)
            continue

        blocked_count += 1

        suppressed.append(
            {
                "memory_id":
                    candidate_memory_id(
                        candidate
                    ),
                "reason": _REASON_SUPPRESSED,
            }
        )

    output = dict(policy_report)

    base_ineligible = policy_report.get(
        "ineligible"
    )

    output["eligible"] = retained

    output["ineligible"] = (
        list(base_ineligible)
        if isinstance(base_ineligible, list)
        else []
    ) + suppressed

    output["eligible_candidate_count"] = len(
        retained
    )

    summary = _summary(
        decision=_DECISION_APPLIED,
        reason=_REASON_APPLIED,
        thresholds=thresholds,
        evaluated_count=evaluated_count,
        observed_count=observed_count,
        neutral_count=neutral_count,
        passed_count=passed_count,
        blocked_count=blocked_count,
        primary_preserved_count=(
            primary_preserved_count
        ),
        invalid_event_count=invalid_event_count,
    )

    output["lifecycle_surface"] = summary

    _log(summary)

    return output


def apply_lifecycle_surfacing_gate(
    policy_report: Any,
    *,
    as_of: Any = None,
) -> Any:
    """Constrain an existing Surfacing Policy report with lifecycle.

    Pure, synchronous and read-only. It returns a COPY of the base
    ``memory-surfacing-policy.v1`` report with the supplementary
    candidates filtered by the lifecycle gate. The caller's object is
    never mutated, and no numeric lifecycle state is ever attached to
    a candidate.

    Fail-NEUTRAL: while either flag is OFF, while the base policy is
    not the real contract, while the base policy already decided
    ``no_surface``, or on any failure, the base eligibility is
    returned untouched. Lifecycle can only further constrain an
    already eligible set -- it can never revive an ineligible
    candidate and never re-rank what survives.
    """

    if not isinstance(policy_report, dict):
        return policy_report

    # Integration flag OFF: behaviour is exactly the frozen version.
    if not lifecycle_surfacing_enabled():
        return dict(policy_report)

    if (
        policy_report.get("version")
        != _POLICY_VERSION
        or policy_report.get("mode") != _POLICY_MODE
    ):
        return dict(policy_report)

    # A base ``no_surface`` is final: lifecycle must not revive it.
    if (
        policy_report.get("decision")
        != _ALLOW_DECISION
    ):
        return dict(policy_report)

    # Lifecycle Shadow OFF means the whole subsystem participates in
    # no behaviour: neutral, never a suppression.
    if not lifecycle_shadow_enabled():
        return _neutral(
            policy_report,
            reason=_REASON_DISABLED,
        )

    resolved = _resolve_as_of(as_of)

    if resolved is None:
        # An invalid ``as_of`` is a failure: keep base eligibility.
        return dict(policy_report)

    try:
        return _apply_gate(
            policy_report, as_of=resolved
        )
    except Exception:
        # Lifecycle integration must never make a memory disappear.
        return dict(policy_report)
