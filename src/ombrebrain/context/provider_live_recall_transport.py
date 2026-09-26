from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ombrebrain.context.live_recall_authorization import (
    live_exposure_flag_enabled,
    live_recall_bridge_enabled,
    live_recall_enabled,
)
from ombrebrain.context.live_recall_transport_capability import (
    mint_live_recall_transport_capability,
)
from ombrebrain.context.live_recall_usage_attribution import (
    live_recall_usage_attribution_enabled,
)
from ombrebrain.context.live_memory_transport_capability import (
    mint_live_memory_transport_capability,
)
from ombrebrain.context.memory_flash_live_exposure import (
    read_live_memory_exposure,
)
from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    is_valid_fingerprint,
    now_iso,
    read_json,
    state_root,
)


# Provider-Bound Live Recall Transport v1.
#
# This module is the ONLY place where the Gateway mutates an upstream
# request to attach the Anthropic Messages MCP Connector:
#
#   Memory Flash (memref + cue, already in the body)
#     -> short-lived hidden MCP capability (never seen by the model)
#       -> mcp_servers[].authorization_token
#         -> Anthropic MCP Connector calls back into /mcp
#
# The Gateway itself stays a request transformer + raw response
# passthrough. It never parses upstream ``tool_use``, never buffers the
# model stream, never constructs an agent loop and never sends a second
# provider request. The Provider owns
# ``mcp_tool_use -> remote MCP -> mcp_tool_result -> continuation``.
#
# Mutation scope is deliberately tiny and verified:
#   - only ``tools`` and ``mcp_servers`` may be appended to;
#   - existing ``tools`` / ``mcp_servers`` keep their exact order and
#     values;
#   - the only outbound header change is appending the
#     ``mcp-client-2025-11-20`` beta token to ``anthropic-beta``;
#   - ``messages`` / ``system`` / ``model`` / params / ``stream`` are
#     byte-for-byte preserved.
#
# Default OFF, and fail-open relative to its own input: every failure
# returns the exact input body plus an equivalent headers copy. Once a
# mutation is built, the transport receipt must be persisted AND
# re-read before the mutated body may be returned -- the system never
# emits a capability it cannot prove it minted.
#
# Usage Attribution (``OMBRE_GATEWAY_CONTEXT_USAGE_ATTRIBUTION``) is
# additive and default OFF. OFF keeps the frozen v1 behavior exactly:
# an ``obrcap_`` capability and a Recall-only toolset. ON mints the v2
# ``obmtcap_`` capability and enables the second transport-only tool
# in the same toolset configuration:
#
#     "configs": {
#       "Recall":    {"enabled": true},
#       "UseMemory": {"enabled": true}
#     }
#
# Nothing else changes: same server entry, same URL, same beta token,
# same preserved fields.
#
# The raw capability appears in exactly one place: the outbound
# ``mcp_servers[].authorization_token``. Never in the receipt, the
# report, a log, the model context or a tool argument.

_VERSION = "provider-live-recall-transport.v1"
_MODE = "anthropic_mcp_connector"

_PROVIDER_ANTHROPIC_MCP = "anthropic_mcp_connector"

_SERVER_NAME = "ombre-live-recall"
_MCP_TOOLSET_TYPE = "mcp_toolset"

# The single Recall tool is the frozen v1 toolset. With the Usage
# Attribution flag ON the SAME toolset additionally enables
# ``UseMemory`` -- the explicit usage-attribution tool -- and the
# capability becomes a v2 ``obmtcap_`` token. Nothing else about the
# request changes.
_TOOLSET_CONFIGS_V1 = {"Recall": {"enabled": True}}
_TOOLSET_CONFIGS_V2 = {
    "Recall": {"enabled": True},
    "UseMemory": {"enabled": True},
}

_BETA_HEADER = "anthropic-beta"
_BETA_TOKEN = "mcp-client-2025-11-20"

_ENV_TRANSPORT = (
    "OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_TRANSPORT"
)
_ENV_PROVIDER = (
    "OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_PROVIDER"
)
_ENV_MCP_URL = (
    "OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_MCP_URL"
)

# Fields the transport may never change.
_PRESERVED_KEYS = (
    "messages",
    "system",
    "model",
    "max_tokens",
    "stream",
    "temperature",
    "top_p",
    "stop_sequences",
    "metadata",
)

# ZDR note: the Anthropic MCP Connector is an independent provider
# capability. This module never claims ZDR compatibility; the
# deployment operator is responsible for confirming the Provider's
# data retention policy before enabling
# OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_TRANSPORT.


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def live_recall_transport_enabled() -> bool:
    """The transport rollout gate. Default OFF."""

    return _truthy(os.environ.get(_ENV_TRANSPORT))


def configured_provider() -> str:
    return (
        str(os.environ.get(_ENV_PROVIDER) or "")
        .strip()
        .lower()
    )


def _configured_mcp_url() -> str | None:
    """Syntax-only validation of the operator-configured MCP URL.

    Requires ``https://...`` with a host and with no query and no
    fragment. The URL is never requested here -- no SSRF probing.
    """

    raw = os.environ.get(_ENV_MCP_URL)

    if not isinstance(raw, str):
        return None

    text = raw.strip()

    if not text:
        return None

    try:
        parts = urlsplit(text)
    except ValueError:
        return None

    if parts.scheme != "https":
        return None

    if not parts.netloc:
        return None

    if parts.query or parts.fragment:
        return None

    return text


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _is_nonnegative_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


# ------------------------------------------------------
# Beta header
# ------------------------------------------------------


def _beta_key(
    headers: dict[str, str],
) -> str | None:
    for key in headers:
        if key.lower() == _BETA_HEADER:
            return key

    return None


def _beta_tokens(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []

    return [
        token.strip()
        for token in value.split(",")
        if token.strip()
    ]


def _has_beta_token(
    headers: dict[str, str],
) -> bool:
    key = _beta_key(headers)

    if key is None:
        return False

    return _BETA_TOKEN in _beta_tokens(
        headers.get(key)
    )


def _add_beta_token(
    headers: dict[str, str],
) -> bool:
    """Append the MCP beta token once. Returns whether it was added."""

    key = _beta_key(headers)

    if key is None:
        headers[_BETA_HEADER] = _BETA_TOKEN
        return True

    tokens = _beta_tokens(headers.get(key))

    if _BETA_TOKEN in tokens:
        return False

    tokens.append(_BETA_TOKEN)

    headers[key] = ", ".join(tokens)

    return True


# ------------------------------------------------------
# Receipt
# ------------------------------------------------------


def provider_live_recall_transport_path(
    conversation_id: str,
    cognitive_request_id: str,
) -> Path:
    return (
        state_root()
        / "provider_live_recall_transport"
        / conversation_id
        / (cognitive_request_id + ".json")
    )


def is_valid_provider_live_recall_transport_receipt(
    receipt: Any,
    *,
    conversation_id: Any = None,
    cognitive_request_id: Any = None,
    selected_body_sha256: Any = None,
) -> bool:
    """Structural identity of one transport receipt.

    Only privacy-safe fields are checked; a raw capability, the MCP
    URL plaintext, a memref, a cue, a memory id or user text can never
    make a receipt valid because none of them is a field.
    """

    if not isinstance(receipt, dict):
        return False

    if (
        receipt.get("version") != _VERSION
        or receipt.get("mode") != _MODE
    ):
        return False

    if receipt.get("server_name") != _SERVER_NAME:
        return False

    if receipt.get("applied") is not True:
        return False

    receipt_conversation = receipt.get(
        "conversation_id"
    )

    receipt_request = receipt.get(
        "cognitive_request_id"
    )

    if (
        not is_valid_conversation_id(
            receipt_conversation
        )
        or not is_valid_cognitive_request_id(
            receipt_request
        )
    ):
        return False

    if (
        conversation_id is not None
        and receipt_conversation != conversation_id
    ):
        return False

    if (
        cognitive_request_id is not None
        and receipt_request != cognitive_request_id
    ):
        return False

    created_at = receipt.get("created_at")

    if (
        not isinstance(created_at, str)
        or not created_at
    ):
        return False

    if not is_valid_fingerprint(
        receipt.get("capability_sha256")
    ):
        return False

    expires_at = receipt.get(
        "capability_expires_at"
    )

    if (
        not isinstance(expires_at, str)
        or not expires_at
    ):
        return False

    for key in (
        "source_live_exposure_selected_sha256",
        "source_live_exposure_render_sha256",
        "mcp_url_sha256",
        "original_body_sha256",
        "selected_body_sha256",
    ):
        if not is_valid_fingerprint(
            receipt.get(key)
        ):
            return False

    if (
        selected_body_sha256 is not None
        and receipt.get("selected_body_sha256")
        != selected_body_sha256
    ):
        return False

    tools_before = receipt.get("tools_count_before")

    tools_after = receipt.get("tools_count_after")

    servers_before = receipt.get(
        "mcp_servers_count_before"
    )

    servers_after = receipt.get(
        "mcp_servers_count_after"
    )

    if (
        not _is_nonnegative_int(tools_before)
        or not _is_nonnegative_int(tools_after)
        or not _is_nonnegative_int(servers_before)
        or not _is_nonnegative_int(servers_after)
    ):
        return False

    if (
        tools_after != tools_before + 1
        or servers_after != servers_before + 1
    ):
        return False

    if receipt.get("beta_present_after") is not True:
        return False

    if receipt.get("beta_present_before") not in (
        True,
        False,
    ):
        return False

    if (
        receipt.get("original_body_sha256")
        == receipt.get("selected_body_sha256")
    ):
        return False

    for key in (
        "messages_preserved",
        "system_preserved",
        "model_preserved",
        "params_preserved",
        "stream_preserved",
    ):
        if receipt.get(key) is not True:
            return False

    return True


def _persist_receipt(
    path: Path,
    receipt: dict[str, Any],
    *,
    conversation_id: str,
    cognitive_request_id: str,
) -> tuple[bool, bool]:
    """Persist and re-read. Returns ``(ok, duplicate)``.

    An existing receipt with the same body binding is a duplicate
    success and is never overwritten; an existing corrupt or
    mismatching receipt is refused.
    """

    duplicate = False

    try:
        with LOCK:
            if path.is_file():
                existing = read_json(path)

                if not (
                    is_valid_provider_live_recall_transport_receipt(
                        existing,
                        conversation_id=(
                            conversation_id
                        ),
                        cognitive_request_id=(
                            cognitive_request_id
                        ),
                    )
                ):
                    return False, False

                if (
                    existing.get("selected_body_sha256")
                    == receipt.get(
                        "selected_body_sha256"
                    )
                    and existing.get(
                        "capability_sha256"
                    )
                    == receipt.get(
                        "capability_sha256"
                    )
                ):
                    duplicate = True
                else:
                    return False, False
            else:
                atomic_write(path, receipt)

            persisted = read_json(path)

            if not (
                is_valid_provider_live_recall_transport_receipt(
                    persisted,
                    conversation_id=conversation_id,
                    cognitive_request_id=(
                        cognitive_request_id
                    ),
                    selected_body_sha256=receipt.get(
                        "selected_body_sha256"
                    ),
                )
            ):
                return False, duplicate

            return True, duplicate
    except Exception:
        return False, duplicate


# ------------------------------------------------------
# Report
# ------------------------------------------------------


def _base_report(enabled: bool) -> dict[str, Any]:
    """Privacy-safe report. Only booleans, counts, enums and hashes."""

    return {
        "version": _VERSION,
        "mode": _MODE,
        "enabled": bool(enabled),
        "applied": False,
        "reason": "not_applied",
        "provider": None,
        "tools_delta": 0,
        "mcp_servers_delta": 0,
        "beta_added": False,
        "capability_ttl": None,
        "receipt_valid": False,
        "duplicate": False,
        "tools_count_before": None,
        "tools_count_after": None,
        "mcp_servers_count_before": None,
        "mcp_servers_count_after": None,
        "original_body_sha256": None,
        "selected_body_sha256": None,
        "beta_present_before": None,
        "beta_present_after": None,
        "messages_preserved": None,
        "system_preserved": None,
        "model_preserved": None,
        "params_preserved": None,
        "stream_preserved": None,
    }


def _unchanged(
    body: bytes,
    headers: dict[str, str],
    report: dict[str, Any],
    reason: str,
) -> tuple[bytes, dict[str, str], dict[str, Any]]:
    report["applied"] = False
    report["reason"] = reason

    return body, dict(headers), report


# ------------------------------------------------------
# Transport
# ------------------------------------------------------


def prepare_provider_live_recall_transport(
    body: bytes,
    headers: dict[str, str],
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
) -> tuple[bytes, dict[str, str], dict[str, Any]]:
    """Attach the Anthropic MCP Connector to one upstream request.

    Returns ``(body, headers, report)``. The body and headers are
    returned unchanged (with an equivalent headers copy) on any
    refusal or failure. This function never raises.
    """

    safe_headers = (
        dict(headers)
        if isinstance(headers, dict)
        else {}
    )

    if not live_recall_transport_enabled():
        return (
            body,
            safe_headers,
            _base_report(False)
            | {"reason": "transport_disabled"},
        )

    report = _base_report(True)

    try:
        provider = configured_provider()

        report["provider"] = provider or None

        if provider != _PROVIDER_ANTHROPIC_MCP:
            return _unchanged(
                body,
                safe_headers,
                report,
                "unsupported_provider",
            )

        mcp_url = _configured_mcp_url()

        if mcp_url is None:
            return _unchanged(
                body,
                safe_headers,
                report,
                "invalid_mcp_url",
            )

        # The transport is never a feature starter: the Live Recall
        # Bridge, the live Recall master switch and the Live Exposure
        # flag must all already be ON. No flag is turned on here.
        if not (
            live_recall_bridge_enabled()
            and live_recall_enabled()
            and live_exposure_flag_enabled()
        ):
            return _unchanged(
                body,
                safe_headers,
                report,
                "prerequisites_disabled",
            )

        if not isinstance(body, bytes):
            return _unchanged(
                body,
                safe_headers,
                report,
                "invalid_body",
            )

        if (
            not is_valid_conversation_id(conversation_id)
            or not is_valid_cognitive_request_id(
                cognitive_request_id
            )
        ):
            return _unchanged(
                body,
                safe_headers,
                report,
                "invalid_request_identity",
            )

        # CID/RID alone are not enough: a real live-exposed
        # provenance receipt must exist for THIS request.
        exposure = read_live_memory_exposure(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
        )

        if not isinstance(exposure, dict):
            return _unchanged(
                body,
                safe_headers,
                report,
                "live_exposure_unavailable",
            )

        usage_attribution = (
            live_recall_usage_attribution_enabled()
        )

        if usage_attribution:
            # Usage Attribution is ON: the SAME transport mints a v2
            # capability (``obmtcap_``) that also grants UseMemory.
            # The v1 ``obrcap_`` credential is never widened.
            raw_token, mint_report = (
                mint_live_memory_transport_capability(
                    conversation_id=conversation_id,
                    cognitive_request_id=(
                        cognitive_request_id
                    ),
                )
            )
        else:
            raw_token, mint_report = (
                mint_live_recall_transport_capability(
                    conversation_id=conversation_id,
                    cognitive_request_id=(
                        cognitive_request_id
                    ),
                )
            )

        report["capability_ttl"] = mint_report.get(
            "ttl_seconds"
        )

        if not isinstance(raw_token, str):
            return _unchanged(
                body,
                safe_headers,
                report,
                "capability_not_minted",
            )

        try:
            payload = json.loads(body)
        except Exception:
            return _unchanged(
                body,
                safe_headers,
                report,
                "non_json_body",
            )

        if not isinstance(payload, dict):
            return _unchanged(
                body,
                safe_headers,
                report,
                "unsupported_request",
            )

        existing_servers = payload.get("mcp_servers")

        if existing_servers is None:
            servers: list[Any] = []
        elif isinstance(existing_servers, list):
            servers = list(existing_servers)
        else:
            return _unchanged(
                body,
                safe_headers,
                report,
                "invalid_mcp_servers",
            )

        if any(
            isinstance(server, dict)
            and server.get("name") == _SERVER_NAME
            for server in servers
        ):
            return _unchanged(
                body,
                safe_headers,
                report,
                "mcp_server_name_collision",
            )

        existing_tools = payload.get("tools")

        if existing_tools is None:
            tools: list[Any] = []
        elif isinstance(existing_tools, list):
            tools = list(existing_tools)
        else:
            return _unchanged(
                body,
                safe_headers,
                report,
                "invalid_tools",
            )

        if any(
            isinstance(tool, dict)
            and tool.get("mcp_server_name")
            == _SERVER_NAME
            for tool in tools
        ):
            return _unchanged(
                body,
                safe_headers,
                report,
                "mcp_toolset_collision",
            )

        beta_present_before = _has_beta_token(
            safe_headers
        )

        report["tools_count_before"] = len(tools)
        report["mcp_servers_count_before"] = len(
            servers
        )
        report["beta_present_before"] = (
            beta_present_before
        )

        # -------- build the mutation --------

        mutated = dict(payload)

        mutated["mcp_servers"] = servers + [
            {
                "type": "url",
                "url": mcp_url,
                "name": _SERVER_NAME,
                "authorization_token": raw_token,
            }
        ]

        mutated["tools"] = tools + [
            {
                "type": _MCP_TOOLSET_TYPE,
                "mcp_server_name": _SERVER_NAME,
                "default_config": {"enabled": False},
                "configs": dict(
                    _TOOLSET_CONFIGS_V2
                    if usage_attribution
                    else _TOOLSET_CONFIGS_V1
                ),
            }
        ]

        selected_body = json.dumps(
            mutated,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        # -------- verify the invariant --------

        try:
            verified = json.loads(selected_body)
        except Exception:
            return _unchanged(
                body,
                safe_headers,
                report,
                "mutation_invariant_failed",
            )

        messages_preserved = verified.get(
            "messages"
        ) == payload.get("messages")

        system_preserved = verified.get(
            "system"
        ) == payload.get("system")

        model_preserved = verified.get(
            "model"
        ) == payload.get("model")

        stream_preserved = verified.get(
            "stream"
        ) == payload.get("stream")

        params_preserved = all(
            verified.get(key) == payload.get(key)
            for key in _PRESERVED_KEYS
        ) and set(verified.keys()) == (
            set(payload.keys())
            | {"tools", "mcp_servers"}
        )

        tools_prefix_preserved = (
            verified.get("tools", [])[: len(tools)]
            == tools
        )

        servers_prefix_preserved = (
            verified.get("mcp_servers", [])[
                : len(servers)
            ]
            == servers
        )

        report.update(
            {
                "messages_preserved":
                    messages_preserved,
                "system_preserved":
                    system_preserved,
                "model_preserved": model_preserved,
                "params_preserved":
                    params_preserved,
                "stream_preserved":
                    stream_preserved,
                "tools_count_after":
                    len(mutated["tools"]),
                "mcp_servers_count_after":
                    len(mutated["mcp_servers"]),
                "original_body_sha256":
                    _sha256_bytes(body),
                "selected_body_sha256":
                    _sha256_bytes(selected_body),
            }
        )

        if not all(
            (
                messages_preserved,
                system_preserved,
                model_preserved,
                params_preserved,
                stream_preserved,
                tools_prefix_preserved,
                servers_prefix_preserved,
            )
        ):
            return _unchanged(
                body,
                safe_headers,
                report,
                "mutation_invariant_failed",
            )

        outbound_headers = dict(safe_headers)

        beta_added = _add_beta_token(outbound_headers)

        beta_present_after = _has_beta_token(
            outbound_headers
        )

        if not beta_present_after:
            return _unchanged(
                body,
                safe_headers,
                report,
                "beta_header_not_applied",
            )

        # -------- receipt BEFORE the body may leave --------

        receipt: dict[str, Any] = {
            "version": _VERSION,
            "mode": _MODE,
            "conversation_id": conversation_id,
            "cognitive_request_id": (
                cognitive_request_id
            ),
            "applied": True,
            "created_at": now_iso(),
            "capability_sha256": mint_report.get(
                "token_sha256"
            ),
            "capability_expires_at": mint_report.get(
                "expires_at"
            ),
            "source_live_exposure_selected_sha256":
                exposure.get("selected_sha256"),
            "source_live_exposure_render_sha256":
                exposure.get("render_sha256"),
            "mcp_url_sha256": _sha256_text(mcp_url),
            "server_name": _SERVER_NAME,
            "original_body_sha256": report[
                "original_body_sha256"
            ],
            "selected_body_sha256": report[
                "selected_body_sha256"
            ],
            "tools_count_before": report[
                "tools_count_before"
            ],
            "tools_count_after": report[
                "tools_count_after"
            ],
            "mcp_servers_count_before": report[
                "mcp_servers_count_before"
            ],
            "mcp_servers_count_after": report[
                "mcp_servers_count_after"
            ],
            "beta_present_before": (
                beta_present_before
            ),
            "beta_present_after": True,
            "messages_preserved": True,
            "system_preserved": True,
            "model_preserved": True,
            "params_preserved": True,
            "stream_preserved": True,
        }

        if not (
            is_valid_provider_live_recall_transport_receipt(
                receipt,
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
            )
        ):
            return _unchanged(
                body,
                safe_headers,
                report,
                "receipt_invalid",
            )

        ok, duplicate = _persist_receipt(
            provider_live_recall_transport_path(
                conversation_id,
                cognitive_request_id,
            ),
            receipt,
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
        )

        if not ok:
            return _unchanged(
                body,
                safe_headers,
                report,
                "receipt_persistence_failed",
            )

        report.update(
            {
                "applied": True,
                "reason": "transport_applied",
                "tools_delta": (
                    report["tools_count_after"]
                    - report["tools_count_before"]
                ),
                "mcp_servers_delta": (
                    report[
                        "mcp_servers_count_after"
                    ]
                    - report[
                        "mcp_servers_count_before"
                    ]
                ),
                "beta_added": beta_added,
                "beta_present_after": True,
                "receipt_valid": True,
                "duplicate": duplicate,
            }
        )

        return selected_body, outbound_headers, report

    except Exception:
        return _unchanged(
            body,
            safe_headers,
            report,
            "transport_exception",
        )


__all__ = [
    "configured_provider",
    "is_valid_provider_live_recall_transport_receipt",
    "live_recall_transport_enabled",
    "prepare_provider_live_recall_transport",
    "provider_live_recall_transport_path",
]