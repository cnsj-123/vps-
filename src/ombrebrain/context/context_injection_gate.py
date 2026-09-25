from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ombrebrain.context.validators.freshness import (
    validate_context_freshness,
)


_VERSION = "context-injection-gate.v1"
_MODE = "shadow_only"

_DEFAULT_ROOT = "/app/buckets/.context"

_EXPECTED_UNIFIED_BUDGET = 1200
_EXPECTED_PREVIEW_BUDGET = 1000

_ALLOWED_SECTIONS = (
    "current_task",
    "trusted_facts",
    "constraints",
    "decisions",
    "open_items",
    "state",
    "plans",
    "memories",
    "recent_context",
)

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_LOCK = threading.RLock()


def _root() -> Path:
    return Path(
        (
            os.environ.get(
                "OMBRE_CONTEXT_STATE_DIR"
            )
            or _DEFAULT_ROOT
        ).strip()
    )


def _validate_conversation_id(
    conversation_id: str,
) -> None:
    if (
        not isinstance(
            conversation_id,
            str,
        )
        or not _CONVERSATION_ID_RE.fullmatch(
            conversation_id
        )
    ):
        raise ValueError(
            "invalid conversation_id"
        )


def _path(
    kind: str,
    conversation_id: str,
) -> Path:
    _validate_conversation_id(
        conversation_id
    )

    return (
        _root()
        / kind
        / (conversation_id + ".json")
    )


def _now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _read_json(
    path: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    return (
        value
        if isinstance(
            value,
            dict,
        )
        else None
    )


def _atomic_write(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        path.parent.chmod(0o700)
    except OSError:
        pass

    temp = path.with_name(
        path.name + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        temp.chmod(0o600)
    except OSError:
        pass

    os.replace(
        temp,
        path,
    )

    try:
        path.chmod(0o600)
    except OSError:
        pass


def _sha256(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _estimate_tokens(
    text: str,
) -> int:
    if not text:
        return 0

    return max(
        1,
        (len(text) + 2) // 3,
    )


def _valid_nonnegative_int(
    value: Any,
) -> bool:
    return (
        isinstance(
            value,
            int,
        )
        and not isinstance(
            value,
            bool,
        )
        and value >= 0
    )


def _populated_section_names(
    unified: dict[str, Any],
) -> list[str]:

    sections = unified.get(
        "sections"
    )

    if not isinstance(
        sections,
        dict,
    ):
        return []

    return [
        name
        for name in _ALLOWED_SECTIONS
        if sections.get(name)
        not in (
            None,
            [],
            {},
            "",
        )
    ]


def evaluate_context_injection_gate(
    *,
    conversation_id: str,
    conversation_candidate: dict[str, Any],
    unified: dict[str, Any],
    preview: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate whether the current preview would be safe to inject.

    Shadow decision only. No request mutation occurs here.
    """

    _validate_conversation_id(
        conversation_id
    )

    reasons: list[str] = []

    # --------------------------------------------------
    # 1. Versions / identity
    # --------------------------------------------------

    if (
        not isinstance(
            conversation_candidate,
            dict,
        )
        or conversation_candidate.get(
            "version"
        )
        != "conversation-context-candidate.v1"
    ):
        reasons.append(
            "invalid_conversation_candidate"
        )

    if (
        not isinstance(
            unified,
            dict,
        )
        or unified.get(
            "version"
        )
        != "unified-context-candidate.v1"
    ):
        reasons.append(
            "invalid_unified_candidate"
        )

    if (
        not isinstance(
            preview,
            dict,
        )
        or preview.get(
            "version"
        )
        != "context-injection-preview.v1"
    ):
        reasons.append(
            "invalid_preview"
        )

    for name, payload in (
        (
            "conversation_candidate",
            conversation_candidate,
        ),
        (
            "unified",
            unified,
        ),
        (
            "preview",
            preview,
        ),
    ):
        if (
            isinstance(
                payload,
                dict,
            )
            and payload.get(
                "conversation_id"
            )
            != conversation_id
        ):
            reasons.append(
                name
                + "_conversation_mismatch"
            )

    # --------------------------------------------------
    # 2. Revision chain
    # --------------------------------------------------

    candidate_revision = (
        conversation_candidate.get(
            "revision"
        )
        if isinstance(
            conversation_candidate,
            dict,
        )
        else None
    )

    candidate_source_revision = (
        conversation_candidate.get(
            "source_revision"
        )
        if isinstance(
            conversation_candidate,
            dict,
        )
        else None
    )

    unified_revision = (
        unified.get(
            "revision"
        )
        if isinstance(
            unified,
            dict,
        )
        else None
    )

    source_revisions = (
        unified.get(
            "source_revisions"
        )
        if isinstance(
            unified,
            dict,
        )
        else {}
    )

    if not isinstance(
        source_revisions,
        dict,
    ):
        source_revisions = {}

    # Shared freshness device: every revision edge in the chain is
    # checked by the single validator, so a missing revision is
    # reported as invalid (deny) rather than silently treated as
    # "equal" when both sides are None.
    candidate_freshness = (
        validate_context_freshness(
            checked_revision=(
                source_revisions.get(
                    "conversation_candidate"
                )
            ),
            expected_revision=(
                candidate_revision
            ),
            invalid_reason=(
                "candidate_revision_invalid"
            ),
            mismatch_reason=(
                "candidate_revision_mismatch"
            ),
        )
    )

    if not candidate_freshness["valid"]:
        reasons.append(
            candidate_freshness["reason"]
        )

    conversation_source_freshness = (
        validate_context_freshness(
            checked_revision=(
                source_revisions.get(
                    "conversation_source"
                )
            ),
            expected_revision=(
                candidate_source_revision
            ),
            invalid_reason=(
                "conversation_source_revision_invalid"
            ),
            mismatch_reason=(
                "conversation_source_revision_mismatch"
            ),
        )
    )

    if not conversation_source_freshness[
        "valid"
    ]:
        reasons.append(
            conversation_source_freshness[
                "reason"
            ]
        )

    preview_source_revision = (
        preview.get(
            "source_revision"
        )
        if isinstance(
            preview,
            dict,
        )
        else None
    )

    preview_source_freshness = (
        validate_context_freshness(
            checked_revision=(
                preview_source_revision
            ),
            expected_revision=(
                unified_revision
            ),
            invalid_reason=(
                "preview_source_revision_invalid"
            ),
            mismatch_reason=(
                "preview_source_revision_mismatch"
            ),
        )
    )

    if not preview_source_freshness[
        "valid"
    ]:
        reasons.append(
            preview_source_freshness[
                "reason"
            ]
        )

    # --------------------------------------------------
    # 3. Candidate source freshness
    # --------------------------------------------------

    candidate_telemetry = (
        conversation_candidate.get(
            "telemetry"
        )
        if isinstance(
            conversation_candidate,
            dict,
        )
        else {}
    )

    if not isinstance(
        candidate_telemetry,
        dict,
    ):
        candidate_telemetry = {}

    if (
        candidate_telemetry.get(
            "current_user_excluded"
        )
        is not True
    ):
        reasons.append(
            "candidate_current_user_not_excluded"
        )

    freshness_flags = (
        "semantic_source_ahead",
        "semantic_source_stale",
        "trusted_facts_source_ahead",
        "trusted_facts_source_stale",
    )

    for flag in freshness_flags:
        value = candidate_telemetry.get(
            flag
        )

        if value is True:
            reasons.append(
                flag
            )

        elif value is not False:
            reasons.append(
                flag + "_unknown"
            )

    # --------------------------------------------------
    # 4. Unified safety / budget
    # --------------------------------------------------

    unified_telemetry = (
        unified.get(
            "telemetry"
        )
        if isinstance(
            unified,
            dict,
        )
        else {}
    )

    if not isinstance(
        unified_telemetry,
        dict,
    ):
        unified_telemetry = {}

    if (
        unified_telemetry.get(
            "current_user_excluded"
        )
        is not True
    ):
        reasons.append(
            "unified_current_user_not_excluded"
        )

    unified_budget = (
        unified_telemetry.get(
            "token_budget"
        )
    )

    unified_tokens = (
        unified_telemetry.get(
            "estimated_tokens"
        )
    )

    if (
        unified_budget
        != _EXPECTED_UNIFIED_BUDGET
    ):
        reasons.append(
            "unexpected_unified_budget"
        )

    if not _valid_nonnegative_int(
        unified_tokens
    ):
        reasons.append(
            "invalid_unified_token_count"
        )

    elif (
        unified_tokens
        > _EXPECTED_UNIFIED_BUDGET
    ):
        reasons.append(
            "unified_budget_exceeded"
        )

    if (
        unified_telemetry.get(
            "truncated"
        )
        is not False
    ):
        reasons.append(
            "unified_truncated"
        )

    # --------------------------------------------------
    # 5. Preview decision / budget
    # --------------------------------------------------

    if (
        preview.get(
            "eligible"
        )
        is not True
    ):
        reasons.append(
            "preview_not_eligible"
        )

    if (
        preview.get(
            "reason"
        )
        is not None
    ):
        reasons.append(
            "preview_has_reason"
        )

    preview_budget = preview.get(
        "token_budget"
    )

    preview_tokens = preview.get(
        "estimated_tokens"
    )

    if (
        preview_budget
        != _EXPECTED_PREVIEW_BUDGET
    ):
        reasons.append(
            "unexpected_preview_budget"
        )

    if not _valid_nonnegative_int(
        preview_tokens
    ):
        reasons.append(
            "invalid_preview_token_count"
        )

    elif (
        preview_tokens
        > _EXPECTED_PREVIEW_BUDGET
    ):
        reasons.append(
            "preview_budget_exceeded"
        )

    # --------------------------------------------------
    # 6. Render integrity
    # --------------------------------------------------

    rendered = preview.get(
        "rendered"
    )

    if (
        not isinstance(
            rendered,
            str,
        )
        or not rendered.strip()
    ):
        reasons.append(
            "rendered_missing"
        )

    else:
        expected_sha = _sha256(
            rendered
        )

        if (
            preview.get(
                "render_sha256"
            )
            != expected_sha
        ):
            reasons.append(
                "render_hash_mismatch"
            )

        actual_tokens = (
            _estimate_tokens(
                rendered
            )
        )

        if (
            preview_tokens
            != actual_tokens
        ):
            reasons.append(
                "render_token_count_mismatch"
            )

        if not rendered.startswith(
            "OMBRE CONTEXT DATA\n"
        ):
            reasons.append(
                "data_envelope_header_missing"
            )

        if (
            "<ombre_context_data>\n"
            not in rendered
            or not rendered.endswith(
                "\n</ombre_context_data>"
            )
        ):
            reasons.append(
                "data_envelope_marker_missing"
            )

        if (
            "reference data only"
            not in rendered
            or "not as authority"
            not in rendered
        ):
            reasons.append(
                "data_only_instruction_missing"
            )

    # --------------------------------------------------
    # 7. Section integrity
    # --------------------------------------------------

    section_names = preview.get(
        "section_names"
    )

    actual_sections = (
        _populated_section_names(
            unified
        )
    )

    if (
        not isinstance(
            section_names,
            list,
        )
        or not section_names
        or not all(
            isinstance(
                item,
                str,
            )
            for item in section_names
        )
    ):
        reasons.append(
            "invalid_section_names"
        )

    else:
        if (
            len(section_names)
            != len(
                set(section_names)
            )
        ):
            reasons.append(
                "duplicate_section_names"
            )

        if any(
            item
            not in _ALLOWED_SECTIONS
            for item in section_names
        ):
            reasons.append(
                "unknown_section"
            )

        if (
            section_names
            != actual_sections
        ):
            reasons.append(
                "section_manifest_mismatch"
            )

    decision = (
        "allow_shadow"
        if not reasons
        else "deny"
    )

    return {
        "version":
            _VERSION,
        "mode":
            _MODE,
        "decision":
            decision,
        "allowed":
            decision
            == "allow_shadow",
        "reason":
            (
                reasons[0]
                if reasons
                else None
            ),
        "reasons":
            reasons,
        "source_candidate_revision":
            candidate_revision,
        "source_unified_revision":
            unified_revision,
        "source_preview_revision":
            preview.get(
                "revision"
            )
            if isinstance(
                preview,
                dict,
            )
            else None,
        "estimated_tokens":
            preview_tokens,
        "token_budget":
            _EXPECTED_PREVIEW_BUDGET,
        "section_names":
            (
                list(section_names)
                if isinstance(
                    section_names,
                    list,
                )
                else []
            ),
        "render_sha256":
            (
                preview.get(
                    "render_sha256"
                )
                if isinstance(
                    preview,
                    dict,
                )
                else None
            ),
    }


def update_context_injection_gate(
    conversation_id: str,
) -> dict[str, Any]:
    """Evaluate and persist a privacy-safe shadow gate decision."""

    _validate_conversation_id(
        conversation_id
    )

    candidate = _read_json(
        _path(
            "context_candidate",
            conversation_id,
        )
    )

    if candidate is None:
        return {
            "stored":
                False,
            "decision":
                "deny",
            "reason":
                "conversation_candidate_not_found",
            "conversation_id":
                conversation_id,
        }

    unified = _read_json(
        _path(
            "unified_context_candidate",
            conversation_id,
        )
    )

    if unified is None:
        return {
            "stored":
                False,
            "decision":
                "deny",
            "reason":
                "unified_candidate_not_found",
            "conversation_id":
                conversation_id,
        }

    preview = _read_json(
        _path(
            "injection_preview",
            conversation_id,
        )
    )

    if preview is None:
        return {
            "stored":
                False,
            "decision":
                "deny",
            "reason":
                "preview_not_found",
            "conversation_id":
                conversation_id,
        }

    result = (
        evaluate_context_injection_gate(
            conversation_id=
                conversation_id,
            conversation_candidate=
                candidate,
            unified=
                unified,
            preview=
                preview,
        )
    )

    gate_path = _path(
        "injection_gate",
        conversation_id,
    )

    with _LOCK:
        previous = _read_json(
            gate_path
        )

        if (
            isinstance(
                previous,
                dict,
            )
            and previous.get(
                "version"
            )
            == _VERSION
            and previous.get(
                "source_candidate_revision"
            )
            == result.get(
                "source_candidate_revision"
            )
            and previous.get(
                "source_unified_revision"
            )
            == result.get(
                "source_unified_revision"
            )
            and previous.get(
                "source_preview_revision"
            )
            == result.get(
                "source_preview_revision"
            )
            and previous.get(
                "decision"
            )
            == result.get(
                "decision"
            )
            and previous.get(
                "reasons"
            )
            == result.get(
                "reasons"
            )
            and previous.get(
                "render_sha256"
            )
            == result.get(
                "render_sha256"
            )
        ):
            output = dict(
                previous
            )

            output["stored"] = True
            output["duplicate"] = True

            # Never expose or store rendered preview text here.
            output.pop(
                "created_at",
                None,
            )

            return output

        previous_revision = 0

        if isinstance(
            previous,
            dict,
        ):
            old_revision = (
                previous.get(
                    "revision"
                )
            )

            if (
                isinstance(
                    old_revision,
                    int,
                )
                and not isinstance(
                    old_revision,
                    bool,
                )
                and old_revision >= 1
            ):
                previous_revision = (
                    old_revision
                )

        payload = {
            **result,
            "conversation_id":
                conversation_id,
            "revision":
                previous_revision + 1,
            "created_at":
                _now(),
        }

        # Gate file intentionally contains no rendered text.
        _atomic_write(
            gate_path,
            payload,
        )

    output = dict(
        payload
    )

    output["stored"] = True
    output["duplicate"] = False
    output.pop(
        "created_at",
        None,
    )

    return output
