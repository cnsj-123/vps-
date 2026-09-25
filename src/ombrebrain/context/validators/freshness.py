from __future__ import annotations

from typing import Any

# Default reason strings. Callers pass the pipeline-specific wording
# (e.g. "preview_revision_mismatch") so existing telemetry stays
# unchanged.
DEFAULT_INVALID_REASON = "invalid_revision"
DEFAULT_MISMATCH_REASON = "revision_mismatch"


def is_valid_revision(
    value: Any,
) -> bool:
    """Revisions are 1-based."""

    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
    )


def validate_context_freshness(
    *,
    checked_revision: Any,
    expected_revision: Any,
    invalid_reason: str = (
        DEFAULT_INVALID_REASON
    ),
    mismatch_reason: str = (
        DEFAULT_MISMATCH_REASON
    ),
) -> dict[str, Any]:
    """Check that a stage was derived from the expected upstream revision.

    This is the single shared form of a check that is otherwise
    hand-written in the Injection Gate, the Mutation Shadow builder
    and the Real Injection selector: a stage records the upstream
    revision it was built from (``checked_revision``) and that must
    be a valid 1-based revision that equals the upstream stage's own
    revision (``expected_revision``).

    Pure and read-only. It never raises and never reads state, so it
    can be used both for enforcement and for observation.
    """

    result = {
        "valid": False,
        "reason": invalid_reason,
        "checked_revision":
            checked_revision,
        "expected_revision":
            expected_revision,
    }

    if not is_valid_revision(
        checked_revision
    ) or not is_valid_revision(
        expected_revision
    ):
        return result

    if checked_revision != expected_revision:
        result["reason"] = mismatch_reason
        return result

    result["valid"] = True
    result["reason"] = None

    return result
