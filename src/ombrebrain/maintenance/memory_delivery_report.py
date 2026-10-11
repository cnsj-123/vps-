from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from ombrebrain.context.memory_delivery_shadow import (
    VERSION,
    MODE,
    is_valid_memory_delivery_plan_artifact,
)


# Unified Memory Delivery Plan — aggregate report. READ ONLY.
#
# It scans the request-scoped, privacy-safe artifacts written by
# ``memory_delivery_shadow``:
#
#     <state_root>/
#       memory_delivery_plan/
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
#     conversation id, cognitive request id, evidence ref, digest,
#     memory id, fingerprint, cue, body or memory text;
#   - it validates each artifact with the SAME authoritative v0
#     validator the writer uses. An artifact that fails validation is
#     counted as INVALID and contributes ZERO to every statistic --
#     a malformed row is never silently skipped inside a "valid"
#     artifact;
#   - a missing artifact and a malformed artifact are different
#     observations and are counted separately (``artifact_count`` vs
#     ``invalid_artifact_count``), so "no data" is never reported as
#     "zero units";
#   - a persisted metric and a re-derived metric are never conflated:
#     every counter comes from a row that already passed the v0
#     validator, and the persisted metric is only read for the
#     request-level roll-up after that validation.
#
# It never calls a live delivery component, never invokes a Flash,
# never performs a Recall and never touches storage beyond reading the
# artifact files.

DEFAULT_LIMIT = 500

_DELIVERY_DIRNAME = "memory_delivery_plan"

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


def _as_count(value: Any) -> int:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    return 0


def _reject_constant(name: str) -> Any:
    """Reject NaN / Infinity / -Infinity anywhere in an artifact.

    ``json.loads`` accepts these by default; an experiment that guides
    delivery migration must never treat such an artifact as valid, so
    the parse itself fails and the artifact is counted invalid.
    """

    raise ValueError(
        "non-finite JSON constant: %s" % name
    )


def _artifact_paths(
    root: Path,
    *,
    limit: int,
) -> list[Path]:
    """Deterministic, bounded list of artifact files (read-only)."""

    if limit < 0:
        return []

    base = root / _DELIVERY_DIRNAME

    if not base.is_dir():
        return []

    paths: list[Path] = []

    for path in base.rglob("*.json"):
        if path.is_file():
            paths.append(path)

    paths.sort(
        key=lambda item: item.as_posix()
    )

    return paths[:limit]


def _load_artifact(
    path: Path,
) -> dict[str, Any] | None:
    """Parse one artifact, or None when it is unreadable / malformed."""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None

    if not raw.strip():
        return None

    try:
        artifact = json.loads(
            raw,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError):
        return None

    if not isinstance(artifact, dict):
        return None

    return artifact


def build_report(
    *,
    root: str | Path | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Aggregate the persisted delivery-plan artifacts. Read-only.

    Never writes, creates or deletes anything. A negative limit is
    clamped to ``0`` (never "scan everything"), so the read stays
    bounded even for programmatic callers. Malformed / invalid
    artifacts only increment ``invalid_artifact_count``.
    """

    resolved_root = (
        Path(root)
        if root is not None
        else _state_root()
    )

    paths = _artifact_paths(
        resolved_root,
        limit=max(0, int(limit)),
    )

    artifact_count = 0
    valid_artifact_count = 0
    invalid_artifact_count = 0

    request_with_frontend_count = 0
    request_with_ob_count = 0
    request_with_both_count = 0

    unit_count = 0
    frontend_unit_count = 0
    ob_unit_count = 0

    direct_data_count = 0
    flash_cue_count = 0
    drop_count = 0
    unknown_count = 0

    frontend_direct_data_count = 0
    ob_flash_cue_count = 0
    ob_drop_count = 0
    ob_unknown_count = 0

    policy_unavailable_ob_count = 0

    current_model_visible_unit_count = 0
    current_model_visible_estimated_tokens = 0

    truncated_unit_count = 0

    truncated_artifact_count = 0

    for path in paths:
        artifact_count += 1

        artifact = _load_artifact(path)

        if artifact is None or (
            not is_valid_memory_delivery_plan_artifact(
                artifact
            )
        ):
            # Invalid as a WHOLE: contributes zero to every statistic.
            invalid_artifact_count += 1
            continue

        valid_artifact_count += 1

        metrics = artifact["metrics"]

        frontend_units = _as_count(
            metrics.get("frontend_unit_count")
        )
        ob_units = _as_count(
            metrics.get("ob_unit_count")
        )

        unit_count += _as_count(
            metrics.get("unit_count")
        )
        frontend_unit_count += frontend_units
        ob_unit_count += ob_units

        if frontend_units > 0:
            request_with_frontend_count += 1

        if ob_units > 0:
            request_with_ob_count += 1

        if frontend_units > 0 and ob_units > 0:
            request_with_both_count += 1

        direct_data_count += _as_count(
            metrics.get("direct_data_count")
        )
        flash_cue_count += _as_count(
            metrics.get("flash_cue_count")
        )
        drop_count += _as_count(
            metrics.get("drop_count")
        )
        unknown_count += _as_count(
            metrics.get("unknown_count")
        )

        frontend_direct_data_count += _as_count(
            metrics.get(
                "frontend_direct_data_count"
            )
        )
        ob_flash_cue_count += _as_count(
            metrics.get("ob_flash_cue_count")
        )
        ob_drop_count += _as_count(
            metrics.get("ob_drop_count")
        )
        ob_unknown_count += _as_count(
            metrics.get("ob_unknown_count")
        )

        policy_unavailable_ob_count += _as_count(
            metrics.get(
                "base_policy_unavailable_ob_count"
            )
        )

        current_model_visible_unit_count += (
            _as_count(
                metrics.get(
                    "current_model_visible_count"
                )
            )
        )
        current_model_visible_estimated_tokens += (
            _as_count(
                metrics.get(
                    "current_model_visible"
                    "_estimated_tokens"
                )
            )
        )

        truncated_unit_count += _as_count(
            artifact.get("rows_truncated_count")
        )

        if artifact.get("rows_truncated") is True:
            truncated_artifact_count += 1

    return {
        "version": VERSION,
        "mode": MODE,
        "report": "aggregate_only",
        "artifact_count": artifact_count,
        "valid_artifact_count": valid_artifact_count,
        "invalid_artifact_count": (
            invalid_artifact_count
        ),
        "request_with_frontend_count": (
            request_with_frontend_count
        ),
        "request_with_ob_count": (
            request_with_ob_count
        ),
        "request_with_both_count": (
            request_with_both_count
        ),
        "unit_count": unit_count,
        "frontend_unit_count": (
            frontend_unit_count
        ),
        "ob_unit_count": ob_unit_count,
        "direct_data_count": direct_data_count,
        "flash_cue_count": flash_cue_count,
        "drop_count": drop_count,
        "unknown_count": unknown_count,
        "frontend_direct_data_count": (
            frontend_direct_data_count
        ),
        "ob_flash_cue_count": (
            ob_flash_cue_count
        ),
        "ob_drop_count": ob_drop_count,
        "ob_unknown_count": ob_unknown_count,
        "policy_unavailable_ob_count": (
            policy_unavailable_ob_count
        ),
        "current_model_visible_unit_count": (
            current_model_visible_unit_count
        ),
        "current_model_visible_estimated_tokens": (
            current_model_visible_estimated_tokens
        ),
        "truncated_artifact_count": (
            truncated_artifact_count
        ),
        "truncated_unit_count": (
            truncated_unit_count
        ),
    }


def _render_text(
    report: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append(
        "Unified Memory Delivery — aggregate "
        "(read-only)"
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
        "requests: frontend=%s ob=%s both=%s"
        % (
            report["request_with_frontend_count"],
            report["request_with_ob_count"],
            report["request_with_both_count"],
        )
    )
    lines.append("")

    lines.append("units:")
    for name in (
        "unit_count",
        "frontend_unit_count",
        "ob_unit_count",
        "current_model_visible_unit_count",
        "current_model_visible_estimated_tokens",
    ):
        lines.append(
            "  %-42s %s"
            % (name, report[name])
        )

    lines.append("")
    lines.append("channels:")
    for name in (
        "direct_data_count",
        "flash_cue_count",
        "drop_count",
        "unknown_count",
        "frontend_direct_data_count",
        "ob_flash_cue_count",
        "ob_drop_count",
        "ob_unknown_count",
        "policy_unavailable_ob_count",
        "truncated_artifact_count",
        "truncated_unit_count",
    ):
        lines.append(
            "  %-42s %s"
            % (name, report[name])
        )

    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the read-only Unified Memory "
            "Delivery Plan Shadow artifacts."
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
