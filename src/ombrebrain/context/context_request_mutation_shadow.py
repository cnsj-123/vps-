from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from ombrebrain.gateway.cache_fingerprint import (
    cache_fingerprint_summary_from_body,
)
from ombrebrain.gateway.operit_adapter import (
    OperitAdapter,
)


_VERSION = "context-request-mutation-shadow.v1"
_MODE = "shadow_only"

_DEFAULT_ROOT = "/app/buckets/.context"

_CONVERSATION_ID_RE = re.compile(
    r"^ctx_[0-9a-f]{16}$"
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


def _base_report(
    body: bytes,
) -> dict[str, Any]:
    return {
        "version":
            _VERSION,
        "mode":
            _MODE,
        "built":
            False,
        "safe_to_mutate":
            False,
        "would_inject":
            False,

        # Critical invariant:
        # this module never sends anything upstream.
        "upstream_mutated":
            False,

        "reason":
            "not_built",

        "original_bytes":
            len(body),
        "mutated_bytes":
            len(body),
        "byte_delta":
            0,

        "original_sha256":
            _sha256_bytes(
                body
            ),
        "mutated_sha256":
            _sha256_bytes(
                body
            ),

        "current_user_index":
            None,
        "insertion_block_index":
            None,

        "current_user_blocks_before":
            None,
        "current_user_blocks_after":
            None,

        "inserted_context_tokens":
            None,
        "inserted_context_sha256":
            None,

        "boundary_message_index_before":
            None,
        "boundary_message_index_after":
            None,

        "boundary_prefix_sha256_before":
            None,
        "boundary_prefix_sha256_after":
            None,

        "parent_prefix_sha256_before":
            None,
        "parent_prefix_sha256_after":
            None,

        "boundary_preserved":
            False,
        "history_preserved":
            False,
        "system_preserved":
            False,
        "tools_preserved":
            False,
        "params_preserved":
            False,
        "model_preserved":
            False,
        "cache_marker_count_preserved":
            False,
        "message_count_preserved":
            False,
    }


def build_context_request_mutation_shadow(
    body: bytes,
    *,
    preview: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[
    bytes,
    dict[str, Any],
]:
    """
    Build the request that WOULD be sent if Context injection
    were enabled.

    Shadow-only contract:
      - never sends anything upstream;
      - caller must continue using the original forward_body;
      - on any uncertainty, returns the exact original body;
      - system/tools/history are never modified;
      - Context is inserted only as the first block of the
        current user message;
      - the existing cache boundary must remain byte-semantically
        stable according to the cache fingerprint.
    """

    report = _base_report(
        body
    )

    # --------------------------------------------------
    # 1. Gate must already allow this exact Preview.
    # --------------------------------------------------

    if not isinstance(
        gate,
        dict,
    ):
        report["reason"] = (
            "invalid_gate"
        )
        return body, report

    if (
        gate.get("version")
        != "context-injection-gate.v1"
        or gate.get("mode")
        != "shadow_only"
        or gate.get("decision")
        != "allow_shadow"
        or gate.get("allowed")
        is not True
    ):
        report["reason"] = (
            "gate_not_allowed"
        )
        return body, report

    if not isinstance(
        preview,
        dict,
    ):
        report["reason"] = (
            "invalid_preview"
        )
        return body, report

    if (
        preview.get("version")
        != "context-injection-preview.v1"
        or preview.get("eligible")
        is not True
        or preview.get("reason")
        is not None
    ):
        report["reason"] = (
            "preview_not_eligible"
        )
        return body, report

    if (
        gate.get(
            "source_preview_revision"
        )
        != preview.get(
            "revision"
        )
    ):
        report["reason"] = (
            "preview_revision_mismatch"
        )
        return body, report

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
        report["reason"] = (
            "rendered_missing"
        )
        return body, report

    rendered_sha256 = (
        _sha256_text(
            rendered
        )
    )

    if (
        preview.get(
            "render_sha256"
        )
        != rendered_sha256
        or gate.get(
            "render_sha256"
        )
        != rendered_sha256
    ):
        report["reason"] = (
            "render_hash_mismatch"
        )
        return body, report

    # --------------------------------------------------
    # 2. Parse only the already cache-stabilized body.
    # --------------------------------------------------

    try:
        payload = json.loads(
            body
        )
    except Exception:
        report["reason"] = (
            "non_json"
        )
        return body, report

    if not OperitAdapter.supports(
        payload
    ):
        report["reason"] = (
            "unsupported_request"
        )
        return body, report

    messages = payload.get(
        "messages"
    )

    if (
        not isinstance(
            messages,
            list,
        )
        or not messages
    ):
        report["reason"] = (
            "messages_missing"
        )
        return body, report

    current_index = (
        len(messages) - 1
    )

    current = messages[
        current_index
    ]

    if (
        not isinstance(
            current,
            dict,
        )
        or current.get(
            "role"
        )
        != "user"
    ):
        report["reason"] = (
            "current_user_missing"
        )
        return body, report

    content = current.get(
        "content"
    )

    # Conservative Phase 3C rule:
    # do not normalize string content yet.
    if not isinstance(
        content,
        list,
    ):
        report["reason"] = (
            "current_content_not_blocks"
        )
        return body, report

    if not content:
        report["reason"] = (
            "current_content_empty"
        )
        return body, report

    report[
        "current_user_index"
    ] = current_index

    report[
        "current_user_blocks_before"
    ] = len(content)

    # Prevent accidental double injection.
    for block in content:
        if (
            isinstance(
                block,
                dict,
            )
            and block.get(
                "type"
            )
            == "text"
            and block.get(
                "text"
            )
            == rendered
        ):
            report["reason"] = (
                "context_already_present"
            )
            return body, report

    # --------------------------------------------------
    # 3. Cache boundary must already be before current.
    # --------------------------------------------------

    before = (
        cache_fingerprint_summary_from_body(
            body
        )
    )

    if not isinstance(
        before,
        dict,
    ):
        report["reason"] = (
            "fingerprint_unavailable"
        )
        return body, report

    boundary_index = before.get(
        "boundary_message_index"
    )

    boundary_sha = before.get(
        "boundary_prefix_sha256"
    )

    if (
        not isinstance(
            boundary_index,
            int,
        )
        or isinstance(
            boundary_index,
            bool,
        )
        or not isinstance(
            boundary_sha,
            str,
        )
    ):
        report["reason"] = (
            "cache_boundary_missing"
        )
        return body, report

    if (
        boundary_index
        >= current_index
    ):
        report["reason"] = (
            "cache_boundary_not_before_current"
        )
        return body, report

    report[
        "boundary_message_index_before"
    ] = boundary_index

    report[
        "boundary_prefix_sha256_before"
    ] = boundary_sha

    report[
        "parent_prefix_sha256_before"
    ] = before.get(
        "parent_prefix_sha256"
    )

    # --------------------------------------------------
    # 4. Build SHADOW mutation in memory.
    # --------------------------------------------------

    mutated = deepcopy(
        payload
    )

    mutated_current = (
        mutated[
            "messages"
        ][
            current_index
        ]
    )

    mutated_content = (
        mutated_current[
            "content"
        ]
    )

    context_block = {
        "type":
            "text",
        "text":
            rendered,
    }

    # Context comes before the user's original content.
    # The user's actual request therefore remains the final
    # user-authored content in this message.
    mutated_content.insert(
        0,
        context_block,
    )

    mutated_body = json.dumps(
        mutated,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    # --------------------------------------------------
    # 5. Verify immutable regions.
    # --------------------------------------------------

    history_preserved = (
        mutated[
            "messages"
        ][
            :current_index
        ]
        ==
        payload[
            "messages"
        ][
            :current_index
        ]
    )

    system_preserved = (
        mutated.get(
            "system"
        )
        ==
        payload.get(
            "system"
        )
    )

    tools_preserved = (
        mutated.get(
            "tools"
        )
        ==
        payload.get(
            "tools"
        )
    )

    after = (
        cache_fingerprint_summary_from_body(
            mutated_body
        )
    )

    if not isinstance(
        after,
        dict,
    ):
        report["reason"] = (
            "mutated_fingerprint_unavailable"
        )
        return body, report

    boundary_preserved = (
        after.get(
            "boundary_message_index"
        )
        ==
        before.get(
            "boundary_message_index"
        )
        and after.get(
            "boundary_prefix_sha256"
        )
        ==
        before.get(
            "boundary_prefix_sha256"
        )
        and after.get(
            "parent_boundary_index"
        )
        ==
        before.get(
            "parent_boundary_index"
        )
        and after.get(
            "parent_prefix_sha256"
        )
        ==
        before.get(
            "parent_prefix_sha256"
        )
    )

    cache_marker_preserved = (
        after.get(
            "message_cache_markers"
        )
        ==
        before.get(
            "message_cache_markers"
        )
    )

    message_count_preserved = (
        after.get(
            "messages_count"
        )
        ==
        before.get(
            "messages_count"
        )
    )

    system_fingerprint_preserved = (
        after.get(
            "system_sha256"
        )
        ==
        before.get(
            "system_sha256"
        )
    )

    tools_fingerprint_preserved = (
        after.get(
            "tools_sha256"
        )
        ==
        before.get(
            "tools_sha256"
        )
    )

    params_preserved = (
        after.get(
            "params_sha256"
        )
        ==
        before.get(
            "params_sha256"
        )
    )

    model_preserved = (
        after.get(
            "model_sha256"
        )
        ==
        before.get(
            "model_sha256"
        )
    )

    report.update(
        {
            "history_preserved":
                history_preserved,

            "system_preserved":
                (
                    system_preserved
                    and
                    system_fingerprint_preserved
                ),

            "tools_preserved":
                (
                    tools_preserved
                    and
                    tools_fingerprint_preserved
                ),

            "params_preserved":
                params_preserved,

            "model_preserved":
                model_preserved,

            "cache_marker_count_preserved":
                cache_marker_preserved,

            "message_count_preserved":
                message_count_preserved,

            "boundary_preserved":
                boundary_preserved,

            "boundary_message_index_after":
                after.get(
                    "boundary_message_index"
                ),

            "boundary_prefix_sha256_after":
                after.get(
                    "boundary_prefix_sha256"
                ),

            "parent_prefix_sha256_after":
                after.get(
                    "parent_prefix_sha256"
                ),
        }
    )

    invariants = (
        history_preserved,
        system_preserved,
        tools_preserved,
        system_fingerprint_preserved,
        tools_fingerprint_preserved,
        params_preserved,
        model_preserved,
        cache_marker_preserved,
        message_count_preserved,
        boundary_preserved,
    )

    if not all(
        invariants
    ):
        report["reason"] = (
            "mutation_invariant_failed"
        )
        return body, report

    # --------------------------------------------------
    # 6. Safe SHADOW result.
    # --------------------------------------------------

    report.update(
        {
            "built":
                True,
            "safe_to_mutate":
                True,
            "would_inject":
                True,

            # Still shadow-only.
            "upstream_mutated":
                False,

            "reason":
                "shadow_mutation_safe",

            "mutated_bytes":
                len(
                    mutated_body
                ),

            "byte_delta":
                (
                    len(mutated_body)
                    - len(body)
                ),

            "mutated_sha256":
                _sha256_bytes(
                    mutated_body
                ),

            "insertion_block_index":
                0,

            "current_user_blocks_after":
                len(
                    mutated_content
                ),

            "inserted_context_tokens":
                preview.get(
                    "estimated_tokens"
                ),

            "inserted_context_sha256":
                rendered_sha256,
        }
    )

    return (
        mutated_body,
        report,
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


def build_context_request_mutation_shadow_from_runtime(
    body: bytes,
    *,
    conversation_id: str,
) -> tuple[
    bytes,
    dict[str, Any],
]:
    """Load the local Preview/Gate and build a shadow mutation.

    The returned mutated body is observation-only. The caller must
    never substitute it for the real upstream body in Phase 4A-3C.
    """

    _validate_conversation_id(
        conversation_id
    )

    preview = _read_json_file(
        _root()
        / "injection_preview"
        / (conversation_id + ".json")
    )

    if preview is None:
        report = _base_report(
            body
        )

        report["reason"] = (
            "preview_not_found"
        )

        return body, report

    gate = _read_json_file(
        _root()
        / "injection_gate"
        / (conversation_id + ".json")
    )

    if gate is None:
        report = _base_report(
            body
        )

        report["reason"] = (
            "gate_not_found"
        )

        return body, report

    return (
        build_context_request_mutation_shadow(
            body,
            preview=preview,
            gate=gate,
        )
    )

