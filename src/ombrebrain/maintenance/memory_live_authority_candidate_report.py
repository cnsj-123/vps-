from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ombrebrain.context.memory_live_authority_candidate_shadow import (
    MODE,
    VERSION,
    is_valid_memory_live_authority_candidate_artifact,
)


# Unified Memory Authority Compatibility Candidate Shadow — aggregate
# report. READ ONLY.
#
# It scans the request-scoped, privacy-safe artifacts written by
# ``memory_live_authority_candidate_shadow``:
#
#     <state_root>/
#       memory_live_authority_candidate/
#         <conversation_id>/
#           <cognitive_request_id>.json
#
# and produces an AGGREGATE summary of COMPATIBILITY CANDIDATE
# coherence only. It is a maintenance / research tool, not a runtime
# authority and NOT an architecture decider:
#
#   - it creates no database, no rolling global aggregate and no
#     runtime accumulator;
#   - it writes, creates, moves or deletes nothing (not even a
#     malformed artifact);
#   - it prints only aggregate statistics and finite numbers, never a
#     conversation id, cognitive request id, evidence ref, digest,
#     body hash, memory id, fingerprint, cue, body, attachment text or
#     memory text;
#   - it validates each artifact with the SAME authoritative v0
#     validator the writer uses. An artifact that fails validation is
#     counted as INVALID and contributes ZERO to every statistic -- a
#     malformed row is never silently skipped inside a "valid" artifact;
#   - a missing artifact and a malformed artifact are different
#     observations and are counted separately (``artifact_count`` vs
#     ``invalid_artifact_count``), so "no data" is never reported as
#     "zero units";
#   - it NEVER emits a "safe to cut over" state. v0 has NO live-ready
#     state: ``live_cutover_ready`` is always false, so
#     ``live_cutover_ready_true_count`` MUST remain 0 for every valid v1
#     artifact. The compatibility candidate only proves that unified
#     control COULD adopt the CURRENT Frontend direct exposure without
#     changing what the model sees;
#   - full-scope aggregates are read from the FULL-SCOPE metrics the
#     artifact already carries. A truncated source row set still yields
#     exact full-scope algebra (the mapping is closed), so persisted
#     candidate rows are never confused with the full request
#     ``unit_count``; the truncation counters are exposed separately.
#
# It never initializes the server, the MCP layer, a BucketManager, a
# DecayEngine or an EmbeddingEngine, never calls a live delivery
# component, never invokes a Flash, never performs a Recall and never
# touches storage beyond reading the artifact files.

DEFAULT_LIMIT = 500

_CANDIDATE_DIRNAME = "memory_live_authority_candidate"

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

    ``json.loads`` accepts these by default; a compatibility experiment
    that guides delivery migration must never treat such an artifact as
    valid, so the parse itself fails and the artifact is counted
    invalid.
    """

    raise ValueError(
        "non-finite JSON constant: %s" % name
    )


def _visit_budget(limit: int) -> int:
    """Hard upper bound on filesystem entries examined for ``limit``.

    Core to the bounded-maintenance contract: the reporter may NEVER
    observe more than this many directory entries, so a state tree with
    an unbounded / hostile number of entries can not force the reporter
    to enumerate (or sort) the whole namespace. The budget is a small,
    documented function of ``limit``: ``max(2 * limit, limit)`` for a
    positive limit, and ``0`` when nothing may be read.
    """

    if limit <= 0:
        return 0

    return max(2 * limit, limit)


def _iter_artifact_paths(
    base: Path,
    *,
    limit: int,
    stats: dict[str, Any],
):
    """Bounded, lazy traversal yielding artifact files one at a time.

    The filesystem iteration itself is bounded. It uses ``os.scandir``
    so directory entries are produced lazily, and it stops as soon as
    EITHER ``limit`` artifacts have been selected OR the entry-visit
    budget has been reached. It never materializes an unbounded
    namespace (no eager listing or sorting of the whole directory
    tree), so neither the top-level conversation iterator nor a
    conversation's child iterator is ever drained past the bound.

    Every filesystem entry examined -- top-level or child -- counts
    against the budget. Because the scan is genuinely bounded the
    result may UNDERFILL: it yields the traversal-order prefix of
    artifacts, not a globally lexicographic selection, and a prefix of
    empty / junk / non-JSON entries yields fewer than ``limit`` files.

    Aggregate-only telemetry (integers / boolean, never a name, path or
    id) is written into ``stats``:

        scan_visit_budget         the entry-visit budget for this scan
        scan_visited_entry_count  entries actually examined
        scan_budget_exhausted     True when the budget (not ``limit``)
                                  ended the scan
    """

    visit_budget = _visit_budget(limit)

    stats["scan_visit_budget"] = visit_budget
    stats["scan_visited_entry_count"] = 0
    stats["scan_budget_exhausted"] = False

    if limit <= 0:
        return

    visited = 0
    selected = 0
    budget_exhausted = False

    try:
        root_iterator = os.scandir(base)
    except OSError:
        return

    with root_iterator:
        stop = False

        for root_entry in root_iterator:
            visited += 1

            if visited >= visit_budget:
                budget_exhausted = True
                break

            try:
                is_conversation_dir = (
                    root_entry.is_dir(
                        follow_symlinks=False
                    )
                )
            except OSError:
                is_conversation_dir = False

            if not is_conversation_dir:
                continue

            try:
                child_iterator = os.scandir(
                    root_entry.path
                )
            except OSError:
                continue

            with child_iterator:
                for child_entry in child_iterator:
                    visited += 1

                    try:
                        is_artifact = (
                            child_entry.name.endswith(
                                ".json"
                            )
                            and child_entry.is_file(
                                follow_symlinks=False
                            )
                        )
                    except OSError:
                        is_artifact = False

                    if is_artifact:
                        yield Path(child_entry.path)

                        selected += 1

                        if selected >= limit:
                            stop = True
                            break

                    if visited >= visit_budget:
                        budget_exhausted = True
                        stop = True
                        break

            if stop:
                break

    stats["scan_visited_entry_count"] = visited
    stats["scan_budget_exhausted"] = budget_exhausted


def _artifact_paths(
    root: Path,
    *,
    limit: int,
) -> tuple[list[Path], dict[str, Any]]:
    """Bounded list of artifact files plus aggregate scan telemetry
    (read-only).

    ``limit`` is the MAXIMUM number of artifacts selected / parsed, not
    a guaranteed count: traversal also enforces a hard filesystem-entry
    visit budget and stops at whichever bound is reached first, so a
    huge (or hostile) state tree is never enumerated before slicing and
    the result may underfill. ``limit <= 0`` selects no file and opens
    no directory at all.
    """

    stats: dict[str, Any] = {}

    base = root / _CANDIDATE_DIRNAME

    paths = list(
        _iter_artifact_paths(
            base,
            limit=limit,
            stats=stats,
        )
    )

    return paths, stats


def _path_identity(path: Path) -> tuple[str, str]:
    """Derive ``(conversation_id, cognitive_request_id)`` from the
    artifact path ``<base>/<conversation_id>/<cognitive_request_id>.json``.
    """

    return path.parent.name, path.stem


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
    """Aggregate the persisted candidate artifacts (read-only)."""

    if root is None:
        base = _state_root()
    else:
        base = Path(root)

    paths, scan_stats = _artifact_paths(
        base, limit=limit
    )

    artifact_count = 0
    valid_artifact_count = 0
    invalid_artifact_count = 0

    compatibility_candidate_ready_request_count = 0
    compatibility_candidate_blocked_request_count = 0

    request_with_frontend_control_count = 0

    frontend_controlled_direct_candidate_count = 0

    frontend_controlled_direct_model_visible_estimated_tokens = (
        0
    )

    adopted_frontend_gap_count = 0

    ob_existing_cue_contract_count = 0
    ob_existing_withhold_contract_count = 0
    ob_policy_unknown_count = 0

    remaining_gap_unit_count = 0

    byte_identical_candidate_count = 0

    source_rows_truncated_artifact_count = 0

    live_cutover_ready_true_count = 0

    for path in paths:
        artifact_count += 1

        artifact = _load_artifact(path)

        if artifact is None or (
            not is_valid_memory_live_authority_candidate_artifact(
                artifact
            )
        ):
            # Invalid as a WHOLE: contributes zero to every statistic.
            invalid_artifact_count += 1
            continue

        # PATH BINDING: the artifact must belong to the exact path it
        # was found under. A valid artifact copied under another request
        # filename or conversation directory is NOT a second valid
        # observation and contributes ZERO statistics.
        path_conversation_id, path_request_id = (
            _path_identity(path)
        )

        if (
            artifact.get("conversation_id")
            != path_conversation_id
            or artifact.get("cognitive_request_id")
            != path_request_id
        ):
            invalid_artifact_count += 1
            continue

        valid_artifact_count += 1

        metrics = artifact["metrics"]

        if (
            artifact.get("compatibility_candidate_ready")
            is True
        ):
            compatibility_candidate_ready_request_count += 1
        else:
            compatibility_candidate_blocked_request_count += 1

        if _as_count(
            metrics.get(
                "frontend_controlled_direct_candidate_count"
            )
        ) > 0:
            request_with_frontend_control_count += 1

        frontend_controlled_direct_candidate_count += (
            _as_count(
                metrics.get(
                    "frontend_controlled_direct_candidate_count"
                )
            )
        )

        frontend_controlled_direct_model_visible_estimated_tokens += (
            _as_count(
                metrics.get(
                    "frontend_controlled_direct_model_visible"
                    "_estimated_tokens"
                )
            )
        )

        adopted_frontend_gap_count += _as_count(
            metrics.get("adopted_frontend_gap_count")
        )

        ob_existing_cue_contract_count += _as_count(
            metrics.get(
                "ob_existing_cue_contract_count"
            )
        )
        ob_existing_withhold_contract_count += _as_count(
            metrics.get(
                "ob_existing_withhold_contract_count"
            )
        )
        ob_policy_unknown_count += _as_count(
            metrics.get("ob_policy_unknown_count")
        )

        remaining_gap_unit_count += _as_count(
            metrics.get("remaining_gap_unit_count")
        )

        if (
            artifact.get(
                "candidate_body_byte_identical"
            )
            is True
        ):
            byte_identical_candidate_count += 1

        if (
            artifact.get("source_rows_truncated")
            is True
        ):
            source_rows_truncated_artifact_count += 1

        # v0 has NO live-ready state. A valid v1 artifact MUST keep this
        # false; the counter exists only so a reviewer can confirm it
        # stays at 0. The validator already rejects ``true``, so this
        # can never be incremented by a valid artifact.
        if artifact.get("live_cutover_ready") is True:
            live_cutover_ready_true_count += 1

    return {
        "version": VERSION,
        "mode": MODE,
        "report": "aggregate_only",
        "artifact_count": artifact_count,
        "valid_artifact_count": valid_artifact_count,
        "invalid_artifact_count": (
            invalid_artifact_count
        ),
        "compatibility_candidate_ready_request_count": (
            compatibility_candidate_ready_request_count
        ),
        "compatibility_candidate_blocked_request_count": (
            compatibility_candidate_blocked_request_count
        ),
        "request_with_frontend_control_count": (
            request_with_frontend_control_count
        ),
        "frontend_controlled_direct_candidate_count": (
            frontend_controlled_direct_candidate_count
        ),
        "frontend_controlled_direct_model_visible"
        "_estimated_tokens": (
            frontend_controlled_direct_model_visible_estimated_tokens
        ),
        "adopted_frontend_gap_count": (
            adopted_frontend_gap_count
        ),
        "ob_existing_cue_contract_count": (
            ob_existing_cue_contract_count
        ),
        "ob_existing_withhold_contract_count": (
            ob_existing_withhold_contract_count
        ),
        "ob_policy_unknown_count": (
            ob_policy_unknown_count
        ),
        "remaining_gap_unit_count": (
            remaining_gap_unit_count
        ),
        "byte_identical_candidate_count": (
            byte_identical_candidate_count
        ),
        "source_rows_truncated_artifact_count": (
            source_rows_truncated_artifact_count
        ),
        "live_cutover_ready_true_count": (
            live_cutover_ready_true_count
        ),
        "scan_visit_budget": _as_count(
            scan_stats.get("scan_visit_budget")
        ),
        "scan_visited_entry_count": _as_count(
            scan_stats.get(
                "scan_visited_entry_count"
            )
        ),
        "scan_budget_exhausted": (
            scan_stats.get(
                "scan_budget_exhausted"
            )
            is True
        ),
    }


def _render_text(
    report: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append(
        "Unified Memory Authority Compatibility Candidate — "
        "aggregate (read-only, candidate-only)"
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
        "requests: ready=%s blocked=%s "
        "with_frontend_control=%s"
        % (
            report[
                "compatibility_candidate_ready"
                "_request_count"
            ],
            report[
                "compatibility_candidate_blocked"
                "_request_count"
            ],
            report[
                "request_with_frontend_control_count"
            ],
        )
    )
    lines.append("")

    lines.append("coverage:")
    for name in (
        "frontend_controlled_direct_candidate_count",
        "adopted_frontend_gap_count",
        "ob_existing_cue_contract_count",
        "ob_existing_withhold_contract_count",
        "ob_policy_unknown_count",
        "remaining_gap_unit_count",
    ):
        lines.append(
            "  %-58s %s"
            % (name, report[name])
        )

    lines.append("")
    lines.append("body_identity:")
    for name in (
        "byte_identical_candidate_count",
        "frontend_controlled_direct_model_visible"
        "_estimated_tokens",
    ):
        lines.append(
            "  %-58s %s"
            % (name, report[name])
        )

    lines.append("")
    lines.append("truncation_and_cutover:")
    for name in (
        "source_rows_truncated_artifact_count",
        "live_cutover_ready_true_count",
    ):
        lines.append(
            "  %-58s %s"
            % (name, report[name])
        )

    lines.append("")
    lines.append("scan_bound:")
    for name in (
        "scan_visit_budget",
        "scan_visited_entry_count",
    ):
        lines.append(
            "  %-58s %s"
            % (name, report[name])
        )
    lines.append(
        "  %-58s %s"
        % (
            "scan_budget_exhausted",
            report["scan_budget_exhausted"],
        )
    )

    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the read-only Unified Memory Authority "
            "Compatibility Candidate Shadow artifacts."
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