from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ombrebrain.context.memory_lifecycle_event import (
    event_age_hours,
    iso_utc,
    lifecycle_use_facts,
    memory_key,
    parse_iso_utc,
)
from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    read_json,
    state_root,
)


# Memory Lifecycle State v1 — derived snapshot, never source of truth.
#
#     existence     != accessibility
#     accessibility != salience
#     salience      != strength
#     strength      != confidence
#
# "Forgetting" here is NOT deletion. It is the deterministic, lazy
# lowering of ``accessibility`` and ``salience`` over time while the
# memory still exists in the canonical store:
#
#     exists=True, strength=0.65, accessibility=0.04, salience=0.00
#
# is the whole point of this phase.
#
# This state is a DERIVED CACHE. The source of truth is the immutable
# lifecycle event receipts: deleting every state snapshot must still
# allow an identical recomputation, and a corrupt snapshot is never
# trusted -- it is overwritten by a fresh derivation from events.
#
# The formulas are a deterministic system policy v1, not a claim about
# real human memory.
#
# Nothing here mutates a canonical memory, writes ``activation_count``
# / ``last_active`` / ``importance``, calls a legacy touch / decay /
# archive API, changes Retrieval, changes the Surfacing Policy or
# runs a background job. Decay is derived at ``as_of``, so no hourly
# JSON is ever rewritten for "forgetting".

STATE_VERSION = "memory-lifecycle-state.v1"

MODE = "shadow_only"

_REFERENCE_LAST_USED = "last_used"
_REFERENCE_SOURCE_CREATED = "source_created"
_REFERENCE_FALLBACK = "observation_fallback"

_REFERENCE_SOURCES = (
    _REFERENCE_LAST_USED,
    _REFERENCE_SOURCE_CREATED,
    _REFERENCE_FALLBACK,
)

# Environment surface. Every knob is optional, bounded and degrades
# to a safe default, so configuration can never open an unbounded or
# degenerate decay formula.
_ENV_INITIAL_STRENGTH = (
    "OMBRE_MEMORY_LIFECYCLE_STRENGTH_INITIAL"
)
_ENV_STRENGTH_GAIN = (
    "OMBRE_MEMORY_LIFECYCLE_STRENGTH_GAIN"
)
_ENV_ACCESS_HALF_LIFE = (
    "OMBRE_MEMORY_LIFECYCLE_ACCESS_HALF_LIFE_HOURS"
)
_ENV_SALIENCE_HALF_LIFE = (
    "OMBRE_MEMORY_LIFECYCLE_SALIENCE_HALF_LIFE_HOURS"
)
_ENV_ACCESS_FLOOR = (
    "OMBRE_MEMORY_LIFECYCLE_ACCESS_FLOOR"
)
_ENV_LOW_ACCESS_THRESHOLD = (
    "OMBRE_MEMORY_LIFECYCLE_LOW_ACCESS_THRESHOLD"
)

# Defaults.
_DEFAULT_INITIAL_STRENGTH = 0.20
_DEFAULT_STRENGTH_GAIN = 0.15
_DEFAULT_ACCESS_HALF_LIFE = 168.0
_DEFAULT_SALIENCE_HALF_LIFE = 24.0
_DEFAULT_ACCESS_FLOOR = 0.02
_DEFAULT_LOW_ACCESS_THRESHOLD = 0.10

# Hard bounds.
_INITIAL_STRENGTH_BOUNDS = (0.0, 0.8)
_STRENGTH_GAIN_BOUNDS = (0.01, 0.5)
_ACCESS_HALF_LIFE_BOUNDS = (1.0, 8760.0)
_SALIENCE_HALF_LIFE_BOUNDS = (1.0, 720.0)
_ACCESS_FLOOR_BOUNDS = (0.0, 0.25)
_LOW_ACCESS_THRESHOLD_BOUNDS = (0.0, 0.5)


def _bounded_float(
    raw: Any,
    *,
    default: float,
    low: float,
    high: float,
) -> float:
    """Finite float in ``[low, high]``. Invalid -> default."""

    if raw is None:
        return default

    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default

    if not math.isfinite(value):
        return default

    if value < low:
        return low

    if value > high:
        return high

    return value


def resolve_lifecycle_config() -> dict[str, float]:
    """Bounded lifecycle parameters. Never raises."""

    return {
        "initial_strength": _bounded_float(
            os.environ.get(
                _ENV_INITIAL_STRENGTH
            ),
            default=_DEFAULT_INITIAL_STRENGTH,
            low=_INITIAL_STRENGTH_BOUNDS[0],
            high=_INITIAL_STRENGTH_BOUNDS[1],
        ),
        "strength_gain": _bounded_float(
            os.environ.get(_ENV_STRENGTH_GAIN),
            default=_DEFAULT_STRENGTH_GAIN,
            low=_STRENGTH_GAIN_BOUNDS[0],
            high=_STRENGTH_GAIN_BOUNDS[1],
        ),
        "accessibility_half_life_hours":
            _bounded_float(
                os.environ.get(
                    _ENV_ACCESS_HALF_LIFE
                ),
                default=_DEFAULT_ACCESS_HALF_LIFE,
                low=(
                    _ACCESS_HALF_LIFE_BOUNDS[0]
                ),
                high=(
                    _ACCESS_HALF_LIFE_BOUNDS[1]
                ),
            ),
        "salience_half_life_hours":
            _bounded_float(
                os.environ.get(
                    _ENV_SALIENCE_HALF_LIFE
                ),
                default=(
                    _DEFAULT_SALIENCE_HALF_LIFE
                ),
                low=(
                    _SALIENCE_HALF_LIFE_BOUNDS[
                        0
                    ]
                ),
                high=(
                    _SALIENCE_HALF_LIFE_BOUNDS[
                        1
                    ]
                ),
            ),
        "accessibility_floor": _bounded_float(
            os.environ.get(_ENV_ACCESS_FLOOR),
            default=_DEFAULT_ACCESS_FLOOR,
            low=_ACCESS_FLOOR_BOUNDS[0],
            high=_ACCESS_FLOOR_BOUNDS[1],
        ),
        "low_accessibility_threshold":
            _bounded_float(
                os.environ.get(
                    _ENV_LOW_ACCESS_THRESHOLD
                ),
                default=(
                    _DEFAULT_LOW_ACCESS_THRESHOLD
                ),
                low=(
                    _LOW_ACCESS_THRESHOLD_BOUNDS[
                        0
                    ]
                ),
                high=(
                    _LOW_ACCESS_THRESHOLD_BOUNDS[
                        1
                    ]
                ),
            ),
    }


def _clamp_unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0

    return max(0.0, min(1.0, value))


def strength_for_use_count(
    use_count: int,
    *,
    config: dict[str, float],
) -> float:
    """Long-term reinforcement trace from explicit used events only.

    Deterministic and monotonic in ``use_count``:

        strength = 1 - (1 - INITIAL) * (1 - GAIN) ** use_count

    It never decays with wall-clock time -- time lowers
    accessibility / salience, not the long-term trace.
    """

    if (
        not isinstance(use_count, int)
        or isinstance(use_count, bool)
        or use_count < 0
    ):
        use_count = 0

    initial = config["initial_strength"]
    gain = config["strength_gain"]

    try:
        value = 1.0 - (
            1.0 - initial
        ) * ((1.0 - gain) ** use_count)
    except (OverflowError, ValueError):
        return _clamp_unit(initial)

    return _clamp_unit(value)


def accessibility_for_age(
    strength: float,
    age_hours: float,
    *,
    config: dict[str, float],
) -> float:
    """Current recallability. Lazily derived, never a source write."""

    half_life = config[
        "accessibility_half_life_hours"
    ]

    floor = config["accessibility_floor"]

    age = (
        age_hours
        if math.isfinite(age_hours)
        and age_hours > 0
        else 0.0
    )

    try:
        decay = 2.0 ** (-age / half_life)
    except (OverflowError, ValueError):
        decay = 0.0

    value = floor + (
        1.0 - floor
    ) * _clamp_unit(strength) * decay

    return _clamp_unit(value)


def salience_for_age(
    *,
    use_count: int,
    age_since_last_used_hours: float,
    config: dict[str, float],
) -> float:
    """Short-term prominence from the most recent explicit used only.

    Source ``importance`` / ``confidence`` / retrieval score never
    enter this formula.
    """

    if (
        not isinstance(use_count, int)
        or isinstance(use_count, bool)
        or use_count < 0
        or use_count == 0
    ):
        return 0.0

    half_life = config[
        "salience_half_life_hours"
    ]

    age = (
        age_since_last_used_hours
        if math.isfinite(
            age_since_last_used_hours
        )
        and age_since_last_used_hours > 0
        else 0.0
    )

    try:
        value = 2.0 ** (-age / half_life)
    except (OverflowError, ValueError):
        value = 0.0

    return _clamp_unit(value)


def _is_hex64(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False

    return all(
        char in "0123456789abcdef"
        for char in value
    )


def lifecycle_state_path(
    key: Any,
) -> Path | None:
    if not _is_hex64(key):
        return None

    return (
        state_root()
        / "memory_lifecycle_state"
        / (key + ".json")
    )


def build_lifecycle_state(
    *,
    memory_id: str,
    events: list[dict[str, Any]],
    as_of: datetime,
    config: dict[str, float],
    exists: bool,
    source_exists_checked: bool,
    source_created_at: Any,
    event_file_count: int,
    invalid_event_count: int,
) -> dict[str, Any]:
    """Derive one state snapshot from that memory's event receipts."""

    key = memory_key(memory_id)

    if key is None:
        raise ValueError("invalid memory_id")

    facts = lifecycle_use_facts(
        events, as_of=as_of
    )

    use_count = facts["use_count"]
    last_used = facts["last_used_at"]

    created_at = parse_iso_utc(
        source_created_at
    )

    if last_used is not None:
        reference_at = last_used
        reference_source = _REFERENCE_LAST_USED
    elif created_at is not None:
        reference_at = created_at
        reference_source = (
            _REFERENCE_SOURCE_CREATED
        )
    else:
        reference_at = as_of
        reference_source = _REFERENCE_FALLBACK

    strength = strength_for_use_count(
        use_count, config=config
    )

    accessibility = accessibility_for_age(
        strength,
        event_age_hours(
            at=reference_at, as_of=as_of
        ),
        config=config,
    )

    salience = salience_for_age(
        use_count=use_count,
        age_since_last_used_hours=(
            event_age_hours(
                at=last_used, as_of=as_of
            )
        ),
        config=config,
    )

    threshold = config[
        "low_accessibility_threshold"
    ]

    return {
        "version": STATE_VERSION,
        "mode": MODE,
        "memory_id": memory_id,
        "memory_key": key,
        "exists": bool(exists),
        "event_count": int(event_file_count),
        "use_count": int(use_count),
        "strength": strength,
        "accessibility": accessibility,
        "salience": salience,
        "last_used_at": (
            iso_utc(last_used)
            if last_used is not None
            else None
        ),
        "reference_at": iso_utc(
            reference_at
        ),
        "reference_source": reference_source,
        "as_of": iso_utc(as_of),
        "accessibility_half_life_hours":
            config["accessibility_half_life_hours"],
        "salience_half_life_hours":
            config["salience_half_life_hours"],
        "strength_gain":
            config["strength_gain"],
        "initial_strength":
            config["initial_strength"],
        "accessibility_floor":
            config["accessibility_floor"],
        "low_accessibility_threshold":
            config[
                "low_accessibility_threshold"
            ],
        "source_created_at": (
            iso_utc(created_at)
            if created_at is not None
            else None
        ),
        "source_exists_checked": bool(
            source_exists_checked
        ),
        "derived_from_event_count":
            int(use_count),
        "future_event_count":
            int(facts["future_event_count"]),
        "invalid_event_count":
            int(invalid_event_count),
        "low_accessibility": bool(
            accessibility < threshold
        ),
    }


def _is_count(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _is_unit(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _is_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _matches(
    value: Any,
    expected: float,
) -> bool:
    """Loose float equality. Never a strict ``==`` on a formula."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and math.isfinite(float(expected))
        and math.isclose(
            float(value),
            float(expected),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )


def is_valid_lifecycle_state_artifact(
    artifact: Any,
    *,
    memory_id: Any = None,
) -> bool:
    """Structural identity of one derived state snapshot.

    A snapshot is only a cache, so a reader verifies the contract, the
    memory identity, the numeric ranges, the reference semantics AND --
    because the state is DERIVED -- that every derived field still
    matches the deterministic v1 formula recomputed from the provenance
    and the config the artifact itself carries:

        strength          <- use_count + initial_strength + gain
        accessibility     <- strength + reference_at + as_of + half life
        salience          <- use_count + last_used_at + as_of + half life
        low_accessibility <- accessibility + threshold

    The artifact's own config snapshot is used, never the current
    environment, so a later config change never invalidates an old
    snapshot. A tampered snapshot (``strength=99``, a valid-range but
    wrong strength, a wrong reference, a broken count relation) is
    never returned as if it were real.

    This validates consistency with OUR deterministic v1 formula only.
    It makes no claim about human memory.
    """

    if not isinstance(artifact, dict):
        return False

    if (
        artifact.get("version") != STATE_VERSION
        or artifact.get("mode") != MODE
    ):
        return False

    artifact_memory_id = artifact.get(
        "memory_id"
    )

    key = memory_key(artifact_memory_id)

    if key is None:
        return False

    if artifact.get("memory_key") != key:
        return False

    if (
        memory_id is not None
        and artifact_memory_id != memory_id
    ):
        return False

    if not isinstance(artifact.get("exists"), bool):
        return False

    if not isinstance(
        artifact.get("source_exists_checked"),
        bool,
    ):
        return False

    if not isinstance(
        artifact.get("low_accessibility"), bool
    ):
        return False

    for field in (
        "event_count",
        "use_count",
        "derived_from_event_count",
        "future_event_count",
        "invalid_event_count",
    ):
        if not _is_count(artifact.get(field)):
            return False

    # Every valid v1 event is a ``used`` event, so the derived count is
    # exactly the use count...
    if artifact["use_count"] != artifact[
        "derived_from_event_count"
    ]:
        return False

    # ...and the on-disk directory holds only lifecycle event JSON, so
    # every event file is either derived from or counted invalid.
    if artifact["event_count"] != (
        artifact["derived_from_event_count"]
        + artifact["invalid_event_count"]
    ):
        return False

    for field in (
        "strength",
        "accessibility",
        "salience",
    ):
        if not _is_unit(artifact.get(field)):
            return False

    for field in (
        "strength_gain",
        "initial_strength",
        "accessibility_floor",
        "low_accessibility_threshold",
    ):
        if not _is_unit(artifact.get(field)):
            return False

    if not _is_positive(
        artifact.get(
            "accessibility_half_life_hours"
        )
    ):
        return False

    if not _is_positive(
        artifact.get("salience_half_life_hours")
    ):
        return False

    source = artifact.get("reference_source")

    if source not in _REFERENCE_SOURCES:
        return False

    as_of = parse_iso_utc(artifact.get("as_of"))

    reference_at = parse_iso_utc(
        artifact.get("reference_at")
    )

    if as_of is None or reference_at is None:
        return False

    raw_last_used = artifact.get("last_used_at")

    last_used = (
        parse_iso_utc(raw_last_used)
        if raw_last_used is not None
        else None
    )

    if (
        raw_last_used is not None
        and last_used is None
    ):
        return False

    raw_created = artifact.get("source_created_at")

    created = (
        parse_iso_utc(raw_created)
        if raw_created is not None
        else None
    )

    if raw_created is not None and created is None:
        return False

    use_count = artifact["use_count"]

    # An explicit use is the ONLY way to have a last-used time, and it
    # always owns the decay reference.
    if use_count > 0:
        if last_used is None:
            return False

        if source != _REFERENCE_LAST_USED:
            return False

        if reference_at != last_used:
            return False
    else:
        if last_used is not None:
            return False

        if source == _REFERENCE_LAST_USED:
            return False

    # A non-used reference must be the exact time it claims to be.
    if source == _REFERENCE_SOURCE_CREATED:
        if created is None:
            return False

        if reference_at != created:
            return False
    elif source == _REFERENCE_FALLBACK:
        if reference_at != as_of:
            return False

    expected_strength = strength_for_use_count(
        use_count, config=artifact
    )

    if not _matches(
        artifact["strength"], expected_strength
    ):
        return False

    expected_accessibility = accessibility_for_age(
        artifact["strength"],
        event_age_hours(
            at=reference_at, as_of=as_of
        ),
        config=artifact,
    )

    if not _matches(
        artifact["accessibility"],
        expected_accessibility,
    ):
        return False

    expected_salience = salience_for_age(
        use_count=use_count,
        age_since_last_used_hours=(
            event_age_hours(
                at=last_used, as_of=as_of
            )
        ),
        config=artifact,
    )

    if not _matches(
        artifact["salience"], expected_salience
    ):
        return False

    return artifact["low_accessibility"] == (
        artifact["accessibility"]
        < artifact[
            "low_accessibility_threshold"
        ]
    )


def persist_lifecycle_state(
    artifact: Any,
) -> dict[str, Any]:
    """Overwrite the derived snapshot for one memory.

    The snapshot is a cache: it is always re-derived from events, so a
    corrupt snapshot is replaced rather than trusted. A failed write
    never affects the canonical source and never invalidates the
    in-memory result.
    """

    if not is_valid_lifecycle_state_artifact(
        artifact
    ):
        return {
            "stored": False,
            "reason": "invalid_lifecycle_state",
        }

    path = lifecycle_state_path(
        artifact["memory_key"]
    )

    if path is None:
        return {
            "stored": False,
            "reason": "invalid_lifecycle_state",
        }

    with LOCK:
        atomic_write(path, artifact)

    return {
        "stored": True,
        "reason": "lifecycle_state_stored",
    }


def read_memory_lifecycle_state(
    memory_id: Any,
) -> dict[str, Any] | None:
    """Read one derived snapshot, identity-validated.

    The snapshot carries the ``as_of`` it was derived at, so a caller
    can always tell how fresh it is. A stale snapshot is never "now".
    """

    key = memory_key(memory_id)

    if key is None:
        return None

    path = lifecycle_state_path(key)

    if path is None:
        return None

    artifact = read_json(path)

    if not is_valid_lifecycle_state_artifact(
        artifact, memory_id=memory_id
    ):
        return None

    return artifact


def lifecycle_state_status(
    memory_id: Any,
) -> dict[str, Any]:
    """Privacy-safe freshness status of one state snapshot."""

    artifact = read_memory_lifecycle_state(
        memory_id
    )

    if artifact is None:
        return {"exists": False}

    return {
        "exists": True,
        "version": artifact.get("version"),
        "mode": artifact.get("mode"),
        "as_of": artifact.get("as_of"),
        "use_count": artifact.get("use_count"),
        "event_count":
            artifact.get("event_count"),
        "low_accessibility":
            artifact.get("low_accessibility"),
    }