from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


# Lifecycle Comparison Aggregate Report — READ ONLY.
#
# It scans the request-scoped, privacy-safe artifacts written by
# ``memory_lifecycle_comparison_shadow``:
#
#     <state_root>/
#       memory_lifecycle_comparison/
#         <conversation_id>/
#           <cognitive_request_id>.json
#
# and produces an AGGREGATE summary. It is a maintenance / research
# tool, not a runtime authority:
#
#   - it creates no database, no rolling global aggregate and no
#     runtime accumulator;
#   - it writes, creates, moves or deletes nothing (not even a
#     malformed artifact);
#   - it prints only aggregate statistics and finite numbers, never a
#     conversation id, cognitive request id, memory key, candidate
#     fingerprint, raw memory id, memory text, name, query or cue;
#   - a malformed artifact only increments the invalid count and is
#     skipped from numeric analysis; it never crashes the report.
#
# It reads BOTH lifecycle planes only to count where their current
# semantics differ. A disagreement here does NOT prove which lifecycle
# policy is better.
#
#     correlation != causation, and observation != decision

VERSION = "memory-lifecycle-comparison-shadow.v1"
MODE = "shadow_only"

DEFAULT_LIMIT = 500

_COMPARISON_DIRNAME = "memory_lifecycle_comparison"

_CAT_BINDING_UNAVAILABLE = "binding_unavailable"
_CAT_LEGACY_UNAVAILABLE = "legacy_unavailable"
_CAT_LEGACY_NOT_APPLICABLE = "legacy_not_applicable"
_CAT_COGNITIVE_UNAVAILABLE = (
    "cognitive_evidence_unavailable"
)

# The deterministic comparison categories the runtime observer may
# emit. Unknown / future categories are still counted (see
# ``category_counts``) so an artifact from a newer version is never
# silently dropped, but zero-filled here for a stable report shape.
_KNOWN_CATEGORIES = (
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

_GATE_PRIMARY = "primary_preserved"
_GATE_NEUTRAL = "neutral_unobserved"
_GATE_KEEP = "would_keep"
_GATE_SUPPRESS = "would_suppress"
_GATE_NOT_BASE_ELIGIBLE = "not_base_eligible"
_GATE_BINDING_UNAVAILABLE = "binding_unavailable"

_KNOWN_GATE_DECISIONS = (
    _GATE_PRIMARY,
    _GATE_NEUTRAL,
    _GATE_KEEP,
    _GATE_SUPPRESS,
    _GATE_NOT_BASE_ELIGIBLE,
    _GATE_BINDING_UNAVAILABLE,
)

_DEFAULT_STATE_ROOT = "/app/buckets/.context"


def _state_root() -> Path:
    return Path(
        (
            os.environ.get(
                "OMBRE_CONTEXT_STATE_DIR"
            )
            or _DEFAULT_STATE_ROOT
        ).strip()
    )


def _finite(
    value: Any,
) -> float | None:
    """Finite float or None. Rejects NaN / Infinity / booleans."""

    if isinstance(value, bool):
        return None

    if not isinstance(value, (int, float)):
        return None

    number = float(value)

    if not math.isfinite(number):
        return None

    return number


def _as_bool(value: Any) -> bool:
    return value is True


def _as_count(value: Any) -> int:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    return 0


def _artifact_paths(
    root: Path,
    *,
    limit: int,
) -> list[Path]:
    """Deterministic, bounded list of artifact files (read-only)."""

    base = root / _COMPARISON_DIRNAME

    if not base.is_dir():
        return []

    paths: list[Path] = []

    for path in base.rglob("*.json"):
        if path.is_file():
            paths.append(path)

    paths.sort(
        key=lambda item: item.as_posix()
    )

    if limit >= 0:
        return paths[:limit]

    return paths


def _load_artifact(
    path: Path,
) -> dict[str, Any] | None:
    """Read one artifact, or None when it is malformed / unreadable."""

    try:
        raw = path.read_text(
            encoding="utf-8"
        )
    except OSError:
        return None

    if not raw.strip():
        return None

    try:
        artifact = json.loads(raw)
    except (ValueError, TypeError):
        return None

    if not isinstance(artifact, dict):
        return None

    if (
        artifact.get("version") != VERSION
        or artifact.get("mode") != MODE
    ):
        return None

    candidates = artifact.get("candidates")

    if not isinstance(candidates, list):
        return None

    return artifact


def _valid_row(
    row: Any,
) -> bool:
    """Structural check of one persisted candidate row."""

    if not isinstance(row, dict):
        return False

    if not isinstance(
        row.get("binding"), dict
    ):
        return False

    if not isinstance(
        row.get("legacy"), dict
    ):
        return False

    if not isinstance(
        row.get("cognitive"), dict
    ):
        return False

    comparison = row.get("comparison")

    if not isinstance(comparison, dict):
        return False

    return isinstance(
        comparison.get("category"), str
    )


def _percentile(
    ordered: list[float],
    fraction: float,
) -> float:
    """Linear-interpolated percentile of a non-empty sorted list."""

    if len(ordered) == 1:
        return ordered[0]

    position = (
        len(ordered) - 1
    ) * fraction

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * weight
    )


def _distribution(
    values: list[float],
) -> dict[str, Any]:
    """Bounded, transparent distribution. Empty -> all None."""

    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }

    ordered = sorted(values)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "median": _percentile(ordered, 0.5),
        "p75": _percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def build_report(
    *,
    root: str | Path | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Aggregate the persisted comparison artifacts. Pure / read-only.

    Never writes, creates or deletes anything. Malformed artifacts only
    increment ``invalid_artifact_count``.
    """

    resolved_root = (
        Path(root)
        if root is not None
        else _state_root()
    )

    paths = _artifact_paths(
        resolved_root, limit=limit
    )

    artifact_count = 0
    valid_artifact_count = 0
    invalid_artifact_count = 0

    candidate_count = 0
    observation_truncated_count = 0

    category_counts: dict[str, int] = {
        name: 0 for name in _KNOWN_CATEGORIES
    }

    gate_counts: dict[str, int] = {
        name: 0 for name in _KNOWN_GATE_DECISIONS
    }

    legacy_available_count = 0
    legacy_applicable_count = 0
    legacy_auto_resolve_candidate_count = 0
    legacy_nominal_archive_candidate_count = 0

    cognitive_observed_count = 0
    cognitive_neutral_unobserved_count = 0
    cognitive_counterfactual_suppress_count = 0
    primary_preserved_count = 0

    both_archive_and_suppress_count = 0
    binding_unavailable_count = 0
    invalid_event_count = 0

    legacy_score_values: list[float] = []
    strength_values: list[float] = []
    accessibility_values: list[float] = []
    salience_values: list[float] = []
    use_count_values: list[float] = []

    for path in paths:
        artifact_count += 1

        artifact = _load_artifact(path)

        if artifact is None:
            invalid_artifact_count += 1
            continue

        valid_artifact_count += 1

        observation_truncated_count += (
            _as_count(
                artifact.get(
                    "observation_truncated_count"
                )
            )
        )

        for row in artifact.get(
            "candidates", []
        ):
            if not _valid_row(row):
                # A malformed row is skipped from numeric analysis
                # rather than crashing or being mutated in place.
                continue

            candidate_count += 1

            legacy = row["legacy"]
            cognitive = row["cognitive"]
            category = row[
                "comparison"
            ]["category"]

            category_counts[category] = (
                category_counts.get(category, 0)
                + 1
            )

            if _as_bool(
                legacy.get("available")
            ):
                legacy_available_count += 1

            if _as_bool(
                legacy.get("applicable")
            ):
                legacy_applicable_count += 1

            if _as_bool(
                legacy.get(
                    "auto_resolve_candidate"
                )
            ):
                legacy_auto_resolve_candidate_count += 1

            nominal_archive = _as_bool(
                legacy.get(
                    "nominal_archive_candidate"
                )
            )

            if nominal_archive:
                legacy_nominal_archive_candidate_count += 1

            observed = cognitive.get("observed")
            suppression = _as_bool(
                cognitive.get(
                    "would_suppress_if_supplementary"
                )
            )

            if observed is True:
                cognitive_observed_count += 1
            elif observed is False:
                cognitive_neutral_unobserved_count += 1

            if suppression:
                cognitive_counterfactual_suppress_count += 1

            effective = cognitive.get(
                "effective_gate_decision"
            )

            if effective == _GATE_PRIMARY:
                primary_preserved_count += 1

            if isinstance(effective, str):
                gate_counts[effective] = (
                    gate_counts.get(
                        effective, 0
                    )
                    + 1
                )

            if (
                nominal_archive
                and suppression
            ):
                both_archive_and_suppress_count += 1

            if (
                category
                == _CAT_BINDING_UNAVAILABLE
            ):
                binding_unavailable_count += 1

            invalid_event_count += _as_count(
                cognitive.get(
                    "invalid_event_count"
                )
            )

            score = _finite(
                legacy.get("score_current")
            )

            if score is not None:
                legacy_score_values.append(
                    score
                )

            if observed is True:
                use_count = _finite(
                    cognitive.get("use_count")
                )

                if use_count is not None:
                    use_count_values.append(
                        use_count
                    )

                strength = _finite(
                    cognitive.get("strength")
                )

                if strength is not None:
                    strength_values.append(
                        strength
                    )

                accessibility = _finite(
                    cognitive.get(
                        "accessibility"
                    )
                )

                if accessibility is not None:
                    accessibility_values.append(
                        accessibility
                    )

                salience = _finite(
                    cognitive.get("salience")
                )

                if salience is not None:
                    salience_values.append(
                        salience
                    )

    return {
        "version": VERSION,
        "mode": MODE,
        "report": "aggregate_only",
        "artifact_count": artifact_count,
        "valid_artifact_count": (
            valid_artifact_count
        ),
        "invalid_artifact_count": (
            invalid_artifact_count
        ),
        "candidate_count": candidate_count,
        "observation_truncated_count": (
            observation_truncated_count
        ),
        "category_counts": category_counts,
        "effective_gate_decision_counts": (
            gate_counts
        ),
        "legacy_available_count": (
            legacy_available_count
        ),
        "legacy_applicable_count": (
            legacy_applicable_count
        ),
        "legacy_auto_resolve_candidate_count": (
            legacy_auto_resolve_candidate_count
        ),
        "legacy_nominal_archive_candidate_count": (
            legacy_nominal_archive_candidate_count
        ),
        "cognitive_observed_count": (
            cognitive_observed_count
        ),
        "cognitive_neutral_unobserved_count": (
            cognitive_neutral_unobserved_count
        ),
        "cognitive_counterfactual_suppress_count": (
            cognitive_counterfactual_suppress_count
        ),
        "primary_preserved_count": (
            primary_preserved_count
        ),
        "both_archive_and_suppress_count": (
            both_archive_and_suppress_count
        ),
        "binding_unavailable_count": (
            binding_unavailable_count
        ),
        "invalid_event_count": (
            invalid_event_count
        ),
        "legacy_archive_cognitive_unobserved_count":
            category_counts.get(
                "legacy_archive_cognitive_unobserved",
                0,
            ),
        "legacy_archive_cognitive_keep_count":
            category_counts.get(
                "legacy_archive_cognitive_keep", 0
            ),
        "legacy_keep_cognitive_suppress_count":
            category_counts.get(
                "legacy_keep_cognitive_suppress", 0
            ),
        "distributions": {
            "legacy_score_current": _distribution(
                legacy_score_values
            ),
            "cognitive_strength": _distribution(
                strength_values
            ),
            "cognitive_accessibility": (
                _distribution(
                    accessibility_values
                )
            ),
            "cognitive_salience": _distribution(
                salience_values
            ),
            "cognitive_use_count": _distribution(
                use_count_values
            ),
        },
    }


def _render_text(
    report: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append(
        "Lifecycle Comparison — aggregate (read-only)"
    )
    lines.append(
        "artifacts: %s (valid %s / invalid %s)"
        % (
            report["artifact_count"],
            report["valid_artifact_count"],
            report["invalid_artifact_count"],
        )
    )
    lines.append(
        "candidates: %s"
        % report["candidate_count"]
    )
    lines.append("")

    lines.append("categories:")
    for name in sorted(
        report["category_counts"]
    ):
        lines.append(
            "  %-42s %s"
            % (
                name,
                report[
                    "category_counts"
                ][name],
            )
        )

    lines.append("")
    lines.append("effective gate decisions:")
    for name in sorted(
        report[
            "effective_gate_decision_counts"
        ]
    ):
        lines.append(
            "  %-42s %s"
            % (
                name,
                report[
                    "effective_gate_decision_counts"
                ][name],
            )
        )

    lines.append("")
    lines.append("counts:")
    for name in (
        "legacy_available_count",
        "legacy_applicable_count",
        "legacy_auto_resolve_candidate_count",
        "legacy_nominal_archive_candidate_count",
        "cognitive_observed_count",
        "cognitive_neutral_unobserved_count",
        "cognitive_counterfactual_suppress_count",
        "primary_preserved_count",
        "both_archive_and_suppress_count",
        "binding_unavailable_count",
        "invalid_event_count",
        "observation_truncated_count",
    ):
        lines.append(
            "  %-42s %s"
            % (name, report[name])
        )

    lines.append("")
    lines.append("distributions:")
    for name in sorted(
        report["distributions"]
    ):
        stats = report[
            "distributions"
        ][name]
        lines.append(
            "  %s (n=%s)"
            % (name, stats["count"])
        )
        lines.append(
            "    min=%s p25=%s median=%s "
            "p75=%s max=%s"
            % (
                stats["min"],
                stats["p25"],
                stats["median"],
                stats["p75"],
                stats["max"],
            )
        )

    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the read-only Lifecycle Comparison "
            "Shadow artifacts."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "maximum number of artifacts to read "
            "(default %d)" % DEFAULT_LIMIT
        ),
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "state root to scan (default: "
            "OMBRE_CONTEXT_STATE_DIR)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of text",
    )

    args = parser.parse_args(argv)

    report = build_report(
        root=args.root,
        limit=max(0, args.limit),
    )

    if args.json:
        print(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_render_text(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())