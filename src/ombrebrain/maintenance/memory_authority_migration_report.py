from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ombrebrain.context.memory_authority_migration_shadow import (
    MODE,
    VERSION,
    is_valid_memory_authority_migration_artifact,
)


# Memory Authority Migration Shadow — aggregate report. READ ONLY.
#
# It scans the request-scoped, privacy-safe artifacts written by
# ``memory_authority_migration_shadow``:
#
#     <state_root>/
#       memory_authority_migration/
#         <conversation_id>/
#           <cognitive_request_id>.json
#
# and produces an AGGREGATE summary. It is a maintenance / research
# tool, not a runtime authority and NOT an architecture decider:
#
#   - it creates no database, no rolling global aggregate and no
#     runtime accumulator;
#   - it writes, creates, moves or deletes nothing (not even a
#     malformed artifact);
#   - it prints only aggregate statistics and finite numbers, never a
#     conversation id, cognitive request id, evidence ref, digest,
#     body hash, memory id, fingerprint, cue, body or memory text;
#   - it validates each artifact with the SAME authoritative v0
#     validator the writer uses. An artifact that fails validation is
#     counted as INVALID and contributes ZERO to every statistic -- a
#     malformed row is never silently skipped inside a "valid" artifact;
#   - a missing artifact and a malformed artifact are different
#     observations and are counted separately (``artifact_count`` vs
#     ``invalid_artifact_count``), so "no data" is never reported as
#     "zero units";
#   - it never emits a "safe to cut over" state: v0 has NO live-ready
#     state, so the reporter says nothing of the kind;
#   - full-scope coverage aggregates are read from the FULL-SCOPE
#     metrics the artifact already carries. A truncated source Delivery
#     Plan still yields exact full-scope coverage algebra (the mapping
#     is closed), so persisted comparison rows are never confused with
#     the full request unit_count; the truncation counters are exposed
#     separately.
#
# It never calls a live delivery component, never invokes a Flash,
# never performs a Recall and never touches storage beyond reading the
# artifact files.

DEFAULT_LIMIT = 500

_MIGRATION_DIRNAME = "memory_authority_migration"

_DEFAULT_STATE_ROOT = "/app/buckets/.context"

_STATUS_BLOCKED = "blocked"


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

    base = root / _MIGRATION_DIRNAME

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
    """Aggregate the persisted migration artifacts (read-only)."""

    if root is None:
        base = _state_root()
    else:
        base = Path(root)

    paths = _artifact_paths(base, limit=limit)

    artifact_count = 0
    valid_artifact_count = 0
    invalid_artifact_count = 0

    request_with_frontend_direct_mouth_count = 0
    request_with_ob_live_cue_mouth_count = 0
    request_with_dual_mouth_count = 0

    contract_coverage_blocked_request_count = 0
    contract_coverage_covered_request_count = 0

    unit_count = 0
    contract_covered_unit_count = 0
    contract_gap_unit_count = 0

    frontend_direct_bypass_count = 0
    frontend_replacement_required_count = 0

    ob_controlled_cue_contract_count = 0
    ob_controlled_withhold_contract_count = 0
    ob_policy_unknown_count = 0

    frontend_direct_model_visible_chars = 0
    frontend_direct_model_visible_estimated_tokens = 0

    counterfactual_loss_estimated_tokens = 0

    actual_ob_live_exposed_count = 0
    actual_ob_live_exposure_estimated_tokens = 0

    source_rows_truncated_artifact_count = 0
    source_rows_truncated_count = 0

    for path in paths:
        artifact_count += 1

        artifact = _load_artifact(path)

        if artifact is None or (
            not is_valid_memory_authority_migration_artifact(
                artifact
            )
        ):
            # Invalid as a WHOLE: contributes zero to every statistic.
            invalid_artifact_count += 1
            continue

        valid_artifact_count += 1

        metrics = artifact["metrics"]

        if (
            artifact.get(
                "frontend_direct_mouth_active"
            )
            is True
        ):
            request_with_frontend_direct_mouth_count += 1

        if (
            artifact.get("ob_live_cue_mouth_active")
            is True
        ):
            request_with_ob_live_cue_mouth_count += 1

        if artifact.get("dual_mouth_active") is True:
            request_with_dual_mouth_count += 1

        if (
            artifact.get("contract_coverage_status")
            == _STATUS_BLOCKED
        ):
            contract_coverage_blocked_request_count += 1
        else:
            contract_coverage_covered_request_count += 1

        unit_count += _as_count(
            metrics.get("unit_count")
        )
        contract_covered_unit_count += _as_count(
            metrics.get(
                "contract_covered_unit_count"
            )
        )
        contract_gap_unit_count += _as_count(
            metrics.get("contract_gap_unit_count")
        )

        frontend_direct_bypass_count += _as_count(
            metrics.get(
                "frontend_direct_bypass_count"
            )
        )
        frontend_replacement_required_count += (
            _as_count(
                metrics.get(
                    "frontend_replacement_required_count"
                )
            )
        )

        ob_controlled_cue_contract_count += (
            _as_count(
                metrics.get(
                    "ob_controlled_cue_contract_count"
                )
            )
        )
        ob_controlled_withhold_contract_count += (
            _as_count(
                metrics.get(
                    "ob_controlled_withhold_contract_count"
                )
            )
        )
        ob_policy_unknown_count += _as_count(
            metrics.get(
                "ob_policy_unknown_count"
            )
        )

        frontend_direct_model_visible_chars += (
            _as_count(
                metrics.get(
                    "frontend_direct_model_visible"
                    "_chars"
                )
            )
        )
        frontend_direct_model_visible_estimated_tokens += (
            _as_count(
                metrics.get(
                    "frontend_direct_model_visible"
                    "_estimated_tokens"
                )
            )
        )

        counterfactual_loss_estimated_tokens += (
            _as_count(
                metrics.get(
                    "counterfactual_loss_if_direct_"
                    "mouth_removed_without_replacement"
                    "_estimated_tokens"
                )
            )
        )

        actual_ob_live_exposed_count += _as_count(
            metrics.get(
                "actual_ob_live_exposed_count"
            )
        )
        actual_ob_live_exposure_estimated_tokens += (
            _as_count(
                metrics.get(
                    "actual_ob_live_exposure"
                    "_estimated_tokens"
                )
            )
        )

        source_rows_truncated_count += _as_count(
            artifact.get(
                "source_rows_truncated_count"
            )
        )

        if (
            artifact.get("source_rows_truncated")
            is True
        ):
            source_rows_truncated_artifact_count += 1

    return {
        "version": VERSION,
        "mode": MODE,
        "report": "aggregate_only",
        "artifact_count": artifact_count,
        "valid_artifact_count": valid_artifact_count,
        "invalid_artifact_count": (
            invalid_artifact_count
        ),
        "request_with_frontend_direct_mouth_count": (
            request_with_frontend_direct_mouth_count
        ),
        "request_with_ob_live_cue_mouth_count": (
            request_with_ob_live_cue_mouth_count
        ),
        "request_with_dual_mouth_count": (
            request_with_dual_mouth_count
        ),
        "contract_coverage_blocked_request_count": (
            contract_coverage_blocked_request_count
        ),
        "contract_coverage_covered_request_count": (
            contract_coverage_covered_request_count
        ),
        "unit_count": unit_count,
        "contract_covered_unit_count": (
            contract_covered_unit_count
        ),
        "contract_gap_unit_count": (
            contract_gap_unit_count
        ),
        "frontend_direct_bypass_count": (
            frontend_direct_bypass_count
        ),
        "frontend_replacement_required_count": (
            frontend_replacement_required_count
        ),
        "ob_controlled_cue_contract_count": (
            ob_controlled_cue_contract_count
        ),
        "ob_controlled_withhold_contract_count": (
            ob_controlled_withhold_contract_count
        ),
        "ob_policy_unknown_count": (
            ob_policy_unknown_count
        ),
        "frontend_direct_model_visible_chars": (
            frontend_direct_model_visible_chars
        ),
        "frontend_direct_model_visible_estimated_tokens":
            frontend_direct_model_visible_estimated_tokens,
        "counterfactual_loss_if_direct_mouth_"
        "removed_without_replacement_estimated_tokens":
            counterfactual_loss_estimated_tokens,
        "actual_ob_live_exposed_count": (
            actual_ob_live_exposed_count
        ),
        "actual_ob_live_exposure_estimated_tokens":
            actual_ob_live_exposure_estimated_tokens,
        "source_rows_truncated_artifact_count": (
            source_rows_truncated_artifact_count
        ),
        "source_rows_truncated_count": (
            source_rows_truncated_count
        ),
    }


def _render_text(
    report: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append(
        "Memory Authority Migration — aggregate "
        "(read-only, comparison-only)"
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
        "requests: frontend_direct=%s "
        "ob_live_cue=%s dual=%s"
        % (
            report[
                "request_with_frontend_direct"
                "_mouth_count"
            ],
            report[
                "request_with_ob_live_cue"
                "_mouth_count"
            ],
            report["request_with_dual_mouth_count"],
        )
    )
    lines.append(
        "coverage: blocked_requests=%s "
        "covered_requests=%s"
        % (
            report[
                "contract_coverage_blocked"
                "_request_count"
            ],
            report[
                "contract_coverage_covered"
                "_request_count"
            ],
        )
    )
    lines.append("")

    lines.append("units:")
    for name in (
        "unit_count",
        "contract_covered_unit_count",
        "contract_gap_unit_count",
        "frontend_direct_bypass_count",
        "frontend_replacement_required_count",
    ):
        lines.append(
            "  %-58s %s"
            % (name, report[name])
        )

    lines.append("")
    lines.append("ob_contracts:")
    for name in (
        "ob_controlled_cue_contract_count",
        "ob_controlled_withhold_contract_count",
        "ob_policy_unknown_count",
    ):
        lines.append(
            "  %-58s %s"
            % (name, report[name])
        )

    lines.append("")
    lines.append("exposure:")
    for name in (
        "frontend_direct_model_visible_chars",
        "frontend_direct_model_visible_estimated_tokens",
        "counterfactual_loss_if_direct_mouth_"
        "removed_without_replacement_estimated_tokens",
        "actual_ob_live_exposed_count",
        "actual_ob_live_exposure_estimated_tokens",
        "source_rows_truncated_artifact_count",
        "source_rows_truncated_count",
    ):
        lines.append(
            "  %-58s %s"
            % (name, report[name])
        )

    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the read-only Memory Authority "
            "Migration Shadow artifacts."
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
