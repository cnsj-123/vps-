from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from ombrebrain.context.context_request_mutation_shadow import (
    build_context_request_mutation_shadow_from_runtime,
)
from ombrebrain.context.validators.freshness import (
    validate_context_freshness,
)


_VERSION = "context-real-injection.v1"
_MODE = "limited_real_injection"

# Phase 4A-3D master switch.
# Default OFF. Real injection happens only when this is
# explicitly enabled by the operator.
_ENABLE_ENV = "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION"

_DEFAULT_ROOT = "/app/buckets/.context"

_EXPECTED_PREVIEW_VERSION = (
    "context-injection-preview.v1"
)
_EXPECTED_GATE_VERSION = (
    "context-injection-gate.v1"
)
_EXPECTED_GATE_MODE = "shadow_only"
_EXPECTED_GATE_DECISION = "allow_shadow"
_EXPECTED_MUTATION_VERSION = (
    "context-request-mutation-shadow.v1"
)
_EXPECTED_MUTATION_MODE = "shadow_only"
_EXPECTED_MUTATION_REASON = (
    "shadow_mutation_safe"
)

_MAX_INSERTED_CONTEXT_TOKENS = 1000

# Final DATA-ONLY envelope defense.
# Phase 4A-3D never blindly trusts the Preview module output.
_ENVELOPE_HEADER = "OMBRE CONTEXT DATA\n"
_ENVELOPE_OPEN = "<ombre_context_data>\n"
_ENVELOPE_CLOSE = "\n</ombre_context_data>"
_ENVELOPE_AUTHORITY_WORDING = (
    "reference data only",
    "not as authority",
)

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
)

_INVARIANT_KEYS = (
    "boundary_preserved",
    "history_preserved",
    "system_preserved",
    "tools_preserved",
    "params_preserved",
    "model_preserved",
    "cache_marker_count_preserved",
    "message_count_preserved",
)


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _root() -> Path:
    return Path(
        (
            os.environ.get(
                "OMBRE_CONTEXT_STATE_DIR"
            )
            or _DEFAULT_ROOT
        ).strip()
    )


def _read_json_file(
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


def _sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


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


def _valid_revision(
    value: Any,
) -> bool:
    """Revisions are 1-based."""

    return (
        isinstance(
            value,
            int,
        )
        and not isinstance(
            value,
            bool,
        )
        and value >= 1
    )


def _estimate_tokens(
    text: str,
) -> int:
    """Same estimator as the Injection Preview module."""

    if not text:
        return 0

    return max(
        1,
        (len(text) + 2) // 3,
    )


def _valid_data_envelope(
    rendered: str,
) -> bool:
    """Final DATA-ONLY envelope defense."""

    if not rendered.startswith(
        _ENVELOPE_HEADER
    ):
        return False

    if _ENVELOPE_OPEN not in rendered:
        return False

    if not rendered.endswith(
        _ENVELOPE_CLOSE
    ):
        return False

    return all(
        wording in rendered
        for wording in _ENVELOPE_AUTHORITY_WORDING
    )


def _base_report(
    body: Any,
    conversation_id: Any,
    *,
    enabled: bool,
    reason: str,
) -> dict[str, Any]:
    """Privacy-safe report.

    Only booleans, counts, enums, revisions, token counts,
    byte sizes and SHA256 digests are ever reported.
    No rendered Context, memory, user or system text.
    """

    if isinstance(
        body,
        bytes,
    ):
        size = len(body)
        digest = _sha256_bytes(body)
    else:
        size = 0
        digest = None

    report: dict[str, Any] = {
        "version":
            _VERSION,
        "mode":
            _MODE,
        "enabled":
            enabled,
        "applied":
            False,
        "reason":
            reason,
        "conversation_id":
            (
                conversation_id
                if isinstance(
                    conversation_id,
                    str,
                )
                else None
            ),

        "original_bytes":
            size,
        "selected_bytes":
            size,
        "byte_delta":
            0,

        "inserted_context_tokens":
            None,

        "original_sha256":
            digest,
        "selected_sha256":
            digest,
        "inserted_context_sha256":
            None,
    }

    for key in _INVARIANT_KEYS:
        report[key] = False

    return report


def _denied(
    report: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Keep the exact original body and record why.

    selected_* always stays equal to the original body here.
    """

    report["applied"] = False
    report["reason"] = reason

    return report


def _copy_mutation_telemetry(
    report: dict[str, Any],
    mutation: dict[str, Any],
) -> None:
    """Copy privacy-safe mutation telemetry for observability."""

    tokens = mutation.get(
        "inserted_context_tokens"
    )

    if _valid_nonnegative_int(
        tokens
    ):
        report[
            "inserted_context_tokens"
        ] = tokens

    context_sha = mutation.get(
        "inserted_context_sha256"
    )

    if (
        isinstance(
            context_sha,
            str,
        )
        and context_sha
    ):
        report[
            "inserted_context_sha256"
        ] = context_sha

    for key in _INVARIANT_KEYS:
        report[key] = (
            mutation.get(key)
            is True
        )


def select_context_injected_body(
    forward_body: bytes,
    *,
    conversation_id: str,
) -> tuple[bytes, dict[str, Any]]:
    """Phase 4A-3D limited real injection selector.

    Returns the body that may be sent upstream.

    Contract:
      - default-OFF: without the explicit master switch the exact
        forward_body is returned;
      - only an already cache-stabilized forward_body is accepted;
      - the real mutation logic is never duplicated here: the existing
        Phase 4A-3C shadow builder is re-invoked and re-verified;
      - every failure (missing Preview/Gate, deny, unsafe mutation,
        revision or hash mismatch, token budget, invalid report,
        exception) returns the exact forward_body;
      - this function never raises for request-borne input and never
        returns an HTTP error;
      - the report never contains rendered Context text.
    """

    if not _truthy(
        os.environ.get(
            _ENABLE_ENV
        )
    ):
        return (
            forward_body,
            _base_report(
                forward_body,
                conversation_id,
                enabled=False,
                reason="master_disabled",
            ),
        )

    report = _base_report(
        forward_body,
        conversation_id,
        enabled=True,
        reason="not_applied",
    )

    try:
        return _select_enabled(
            forward_body,
            conversation_id=conversation_id,
            report=report,
        )
    except Exception:
        # Absolute fail-open: any unexpected error keeps the
        # original upstream body.
        return (
            forward_body,
            _denied(
                report,
                "selector_exception",
            ),
        )


def _select_enabled(
    forward_body: Any,
    *,
    conversation_id: Any,
    report: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:

    # --------------------------------------------------
    # 1. Input contract
    # --------------------------------------------------

    if not isinstance(
        forward_body,
        bytes,
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_body",
            ),
        )

    if (
        not isinstance(
            conversation_id,
            str,
        )
        or not _CONVERSATION_ID_RE.fullmatch(
            conversation_id
        )
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_conversation_id",
            ),
        )

    # --------------------------------------------------
    # 2. Preview
    # --------------------------------------------------

    preview = _read_json_file(
        _root()
        / "injection_preview"
        / (conversation_id + ".json")
    )

    if preview is None:
        return (
            forward_body,
            _denied(
                report,
                "preview_not_found",
            ),
        )

    if (
        preview.get("version")
        != _EXPECTED_PREVIEW_VERSION
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_preview",
            ),
        )

    if (
        preview.get("eligible")
        is not True
        or preview.get("reason")
        is not None
    ):
        return (
            forward_body,
            _denied(
                report,
                "preview_not_eligible",
            ),
        )

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
        return (
            forward_body,
            _denied(
                report,
                "rendered_missing",
            ),
        )

    preview_revision = preview.get(
        "revision"
    )

    if not _valid_revision(
        preview_revision
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_preview_revision",
            ),
        )

    # Unified revision this Preview was rendered from.
    preview_source_revision = preview.get(
        "source_revision"
    )

    if not _valid_revision(
        preview_source_revision
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_preview_source_revision",
            ),
        )

    # --------------------------------------------------
    # 3. Gate
    # --------------------------------------------------

    gate = _read_json_file(
        _root()
        / "injection_gate"
        / (conversation_id + ".json")
    )

    if gate is None:
        return (
            forward_body,
            _denied(
                report,
                "gate_not_found",
            ),
        )

    if (
        gate.get("version")
        != _EXPECTED_GATE_VERSION
        or gate.get("mode")
        != _EXPECTED_GATE_MODE
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_gate",
            ),
        )

    if (
        gate.get("decision")
        != _EXPECTED_GATE_DECISION
        or gate.get("allowed")
        is not True
        or gate.get("reason")
        is not None
        or gate.get("reasons")
        != []
    ):
        return (
            forward_body,
            _denied(
                report,
                "gate_not_allowed",
            ),
        )

    # --------------------------------------------------
    # 4. Gate / Preview consistency
    # --------------------------------------------------

    # Gate must be evaluating exactly the Preview that will be
    # injected. Shared freshness validator is the single source of
    # truth for this revision-chain rule.
    gate_preview_freshness = (
        validate_context_freshness(
            checked_revision=gate.get(
                "source_preview_revision"
            ),
            expected_revision=(
                preview_revision
            ),
            invalid_reason=(
                "invalid_gate_source_revision"
            ),
            mismatch_reason=(
                "preview_revision_mismatch"
            ),
        )
    )

    if not gate_preview_freshness[
        "valid"
    ]:
        return (
            forward_body,
            _denied(
                report,
                gate_preview_freshness[
                    "reason"
                ],
            ),
        )

    # ...and that Preview must have been rendered from the same
    # Unified revision the Gate evaluated.
    gate_unified_freshness = (
        validate_context_freshness(
            checked_revision=gate.get(
                "source_unified_revision"
            ),
            expected_revision=(
                preview_source_revision
            ),
            invalid_reason=(
                "invalid_gate_unified_revision"
            ),
            mismatch_reason=(
                "unified_revision_mismatch"
            ),
        )
    )

    if not gate_unified_freshness[
        "valid"
    ]:
        return (
            forward_body,
            _denied(
                report,
                gate_unified_freshness[
                    "reason"
                ],
            ),
        )

    # Render hash must agree across Preview, Gate and a local
    # recomputation of the rendered text.
    preview_render_sha = preview.get(
        "render_sha256"
    )

    if (
        not isinstance(
            preview_render_sha,
            str,
        )
        or preview_render_sha
        != gate.get(
            "render_sha256"
        )
        or preview_render_sha
        != _sha256_text(
            rendered
        )
    ):
        return (
            forward_body,
            _denied(
                report,
                "render_hash_mismatch",
            ),
        )

    # --------------------------------------------------
    # 5. DATA-ONLY envelope and token budget
    # --------------------------------------------------

    # Final defense: the rendered text must still be the DATA-ONLY
    # envelope, even if Preview/Gate were tampered with consistently.
    if not _valid_data_envelope(
        rendered
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_data_envelope",
            ),
        )

    preview_tokens = preview.get(
        "estimated_tokens"
    )

    gate_tokens = gate.get(
        "estimated_tokens"
    )

    if (
        not _valid_nonnegative_int(
            preview_tokens
        )
        or not _valid_nonnegative_int(
            gate_tokens
        )
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_token_count",
            ),
        )

    if (
        preview_tokens
        > _MAX_INSERTED_CONTEXT_TOKENS
        or gate_tokens
        > _MAX_INSERTED_CONTEXT_TOKENS
    ):
        return (
            forward_body,
            _denied(
                report,
                "token_budget_exceeded",
            ),
        )

    if (
        preview.get("token_budget")
        != _MAX_INSERTED_CONTEXT_TOKENS
        or gate.get("token_budget")
        != _MAX_INSERTED_CONTEXT_TOKENS
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_token_budget",
            ),
        )

    # Recompute the token count from the rendered text with the same
    # estimator the Preview module uses. Three fields agreeing with
    # each other is not enough.
    recomputed_tokens = _estimate_tokens(
        rendered
    )

    if (
        preview_tokens != recomputed_tokens
        or gate_tokens != recomputed_tokens
    ):
        return (
            forward_body,
            _denied(
                report,
                "token_recompute_mismatch",
            ),
        )

    # --------------------------------------------------
    # 6. Re-run the existing shadow mutation.
    # Mutation logic is never duplicated here.
    # --------------------------------------------------

    try:
        mutated_body, mutation = (
            build_context_request_mutation_shadow_from_runtime(
                forward_body,
                conversation_id=
                    conversation_id,
            )
        )
    except Exception:
        return (
            forward_body,
            _denied(
                report,
                "mutation_failed",
            ),
        )

    if not isinstance(
        mutation,
        dict,
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_mutation_report",
            ),
        )

    _copy_mutation_telemetry(
        report,
        mutation,
    )

    if (
        mutation.get("version")
        != _EXPECTED_MUTATION_VERSION
        or mutation.get("mode")
        != _EXPECTED_MUTATION_MODE
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_mutation_report",
            ),
        )

    if mutation.get("built") is not True:
        return (
            forward_body,
            _denied(
                report,
                "mutation_not_built",
            ),
        )

    if (
        mutation.get("safe_to_mutate")
        is not True
    ):
        return (
            forward_body,
            _denied(
                report,
                "mutation_not_safe",
            ),
        )

    if (
        mutation.get("would_inject")
        is not True
    ):
        return (
            forward_body,
            _denied(
                report,
                "mutation_would_not_inject",
            ),
        )

    if (
        mutation.get("upstream_mutated")
        is not False
    ):
        return (
            forward_body,
            _denied(
                report,
                "mutation_upstream_mutated",
            ),
        )

    if (
        mutation.get("reason")
        != _EXPECTED_MUTATION_REASON
    ):
        return (
            forward_body,
            _denied(
                report,
                "mutation_reason_mismatch",
            ),
        )

    # --------------------------------------------------
    # 7. Mutation invariants
    # --------------------------------------------------

    for key in _INVARIANT_KEYS:
        if mutation.get(key) is not True:
            return (
                forward_body,
                _denied(
                    report,
                    "mutation_invariant_failed",
                ),
            )

    # --------------------------------------------------
    # 8. Cross-check Preview / Gate / Mutation telemetry
    # --------------------------------------------------

    mutation_tokens = mutation.get(
        "inserted_context_tokens"
    )

    if not _valid_nonnegative_int(
        mutation_tokens
    ):
        return (
            forward_body,
            _denied(
                report,
                "invalid_token_count",
            ),
        )

    if (
        mutation_tokens
        > _MAX_INSERTED_CONTEXT_TOKENS
    ):
        return (
            forward_body,
            _denied(
                report,
                "token_budget_exceeded",
            ),
        )

    if mutation_tokens != preview_tokens:
        return (
            forward_body,
            _denied(
                report,
                "token_count_mismatch",
            ),
        )

    if (
        mutation.get(
            "inserted_context_sha256"
        )
        != preview_render_sha
    ):
        return (
            forward_body,
            _denied(
                report,
                "context_sha_mismatch",
            ),
        )

    # --------------------------------------------------
    # 9. Body integrity
    # --------------------------------------------------

    if (
        not isinstance(
            mutated_body,
            bytes,
        )
        or mutated_body
        == forward_body
    ):
        return (
            forward_body,
            _denied(
                report,
                "mutation_body_invalid",
            ),
        )

    if (
        mutation.get("mutated_sha256")
        != _sha256_bytes(
            mutated_body
        )
    ):
        return (
            forward_body,
            _denied(
                report,
                "mutated_sha_mismatch",
            ),
        )

    if (
        mutation.get("original_sha256")
        != _sha256_bytes(
            forward_body
        )
    ):
        return (
            forward_body,
            _denied(
                report,
                "original_sha_mismatch",
            ),
        )

    # --------------------------------------------------
    # 10. Safe limited real injection.
    # --------------------------------------------------

    report.update(
        {
            "applied":
                True,
            "reason":
                "injection_applied",
            "selected_bytes":
                len(mutated_body),
            "byte_delta":
                (
                    len(mutated_body)
                    - len(forward_body)
                ),
            "selected_sha256":
                _sha256_bytes(
                    mutated_body
                ),
        }
    )

    return (
        mutated_body,
        report,
    )