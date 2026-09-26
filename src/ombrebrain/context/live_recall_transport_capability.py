from __future__ import annotations

import hashlib
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ombrebrain.context.memory_flash_live_exposure import (
    read_live_memory_exposure,
)
from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    is_valid_fingerprint,
    parse_positive_int,
    read_json,
    state_root,
)


# Live Recall Transport Capability v1 — provider_bound.
#
# This module mints and resolves exactly ONE new kind of credential:
# a short-lived, request-scoped, Recall-only transport capability.
#
# It is NOT a Recall authorization and NOT a memref lookup:
#
#     capability            != recall authorization
#     capability            != recall_requested
#     capability            != memory_loaded
#     capability            != used
#
# Minting a capability creates NO Recall Request, NO Related Recall,
# NO memory_usage and NO lifecycle event. It only proves that the
# holder is the Provider request that this Gateway already bound to a
# trusted ``conversation_id`` + ``cognitive_request_id`` -- and that
# THIS request really did live-expose at least one memref.
#
# The raw token is a 256-bit random value; it is never derived from a
# user OAuth / static MCP / dashboard token, never encodes a CID, a
# RID or a memref, and is never written to disk, logged, receipted or
# reported. Only its SHA256 is persisted, as the exact artifact
# filename and field.
#
# The capability is deliberately NOT bound to a specific memref: one
# Live Flash may expose 2-3 memrefs and the model must be able to
# Recall each of them separately. Which memref is legal is still
# decided by the Live Recall Authorization Bridge
# (``memref in exposed_memrefs``), never by this token.
#
# Default OFF at the transport layer; fail-closed here: any unexpected
# failure returns None instead of a token.

_VERSION = "memory-live-recall-transport-capability.v1"
_MODE = "provider_bound"
_ALLOWED_TOOL = "Recall"

_TOKEN_PREFIX = "obrcap_"
_TOKEN_RE = re.compile(r"^obrcap_[0-9a-f]{64}$")

# Capability TTL. Short by design; the hard cap can never be raised
# past 900 seconds.
_ENV_TTL = (
    "OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_CAPABILITY_TTL_SECONDS"
)
_DEFAULT_TTL_SECONDS = 300
_HARD_TTL_SECONDS = 900

# Safe refusal / status reasons — a closed enum. No filesystem error,
# exception message or request identity is ever surfaced.
_REASON_NOT_MINTED = "not_minted"
_REASON_MINTED = "capability_minted"
_REASON_INVALID_IDENTITY = "invalid_request_identity"
_REASON_EXPOSURE_UNAVAILABLE = "live_exposure_unavailable"
_REASON_PERSISTENCE_FAILED = (
    "capability_persistence_failed"
)
_REASON_MINT_EXCEPTION = "capability_mint_exception"

SAFE_MINT_REASONS = frozenset(
    {
        _REASON_NOT_MINTED,
        _REASON_MINTED,
        _REASON_INVALID_IDENTITY,
        _REASON_EXPOSURE_UNAVAILABLE,
        _REASON_PERSISTENCE_FAILED,
        _REASON_MINT_EXCEPTION,
    }
)


# ------------------------------------------------------
# Primitives
# ------------------------------------------------------


def _sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _now(
    now: datetime | None = None,
) -> datetime:
    """Resolve the clock. Tests always pass a fixed ``now``."""

    if isinstance(now, datetime):
        return (
            now
            if now.tzinfo is not None
            else now.replace(tzinfo=timezone.utc)
        )

    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return (
        moment.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def _is_nonnegative_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def resolve_capability_ttl_seconds() -> int:
    """Resolve the capability TTL (default 300s, hard cap 900s).

    An invalid value degrades to the safe default; an over-long value
    is clamped to the hard cap, so configuration can never mint a
    long-lived capability.
    """

    return parse_positive_int(
        os.environ.get(_ENV_TTL),
        default=_DEFAULT_TTL_SECONDS,
        cap=_HARD_TTL_SECONDS,
    )


def live_recall_transport_capability_dir() -> Path:
    return (
        state_root()
        / "live_recall_transport_capability"
    )


def live_recall_transport_capability_path(
    token_sha256: str,
) -> Path:
    """The EXACT artifact path for one token; never a directory scan."""

    return (
        live_recall_transport_capability_dir()
        / (token_sha256 + ".json")
    )


def is_live_recall_transport_capability_token(
    value: Any,
) -> bool:
    """Format-only check: ``obrcap_`` + 64 lowercase hex chars."""

    return (
        isinstance(value, str)
        and bool(_TOKEN_RE.fullmatch(value))
    )


# ------------------------------------------------------
# Artifact validation
# ------------------------------------------------------


def is_valid_live_recall_transport_capability_artifact(
    artifact: Any,
    *,
    token_sha256: Any = None,
    conversation_id: Any = None,
    cognitive_request_id: Any = None,
    now: datetime | None = None,
) -> bool:
    """Independent structural identity of one capability artifact.

    Reading the file at the right path proves nothing, so every reader
    re-validates the contract, the tool binding, the token hash, the
    request identity, the created / expiry timestamps and the source
    Live Exposure binding. Anything off makes the artifact unusable.

    When ``now`` is given the expiry is enforced as well
    (``now < expires_at``); the boundary is explicit and consistent: a
    capability is valid strictly before ``expires_at`` and invalid at
    and after it.
    """

    if not isinstance(artifact, dict):
        return False

    if (
        artifact.get("version") != _VERSION
        or artifact.get("mode") != _MODE
    ):
        return False

    if artifact.get("allowed_tool") != _ALLOWED_TOOL:
        return False

    stored_sha = artifact.get("token_sha256")

    if not is_valid_fingerprint(stored_sha):
        return False

    if (
        token_sha256 is not None
        and stored_sha != token_sha256
    ):
        return False

    artifact_conversation = artifact.get(
        "conversation_id"
    )

    artifact_request = artifact.get(
        "cognitive_request_id"
    )

    if (
        not is_valid_conversation_id(
            artifact_conversation
        )
        or not is_valid_cognitive_request_id(
            artifact_request
        )
    ):
        return False

    if (
        conversation_id is not None
        and artifact_conversation != conversation_id
    ):
        return False

    if (
        cognitive_request_id is not None
        and artifact_request != cognitive_request_id
    ):
        return False

    created = _parse_iso(artifact.get("created_at"))

    expires = _parse_iso(artifact.get("expires_at"))

    if created is None or expires is None:
        return False

    lifetime = (expires - created).total_seconds()

    # A capability can never outlive the hard cap, and an expiry that
    # is not strictly after its creation is corrupt.
    if lifetime < 1 or lifetime > _HARD_TTL_SECONDS:
        return False

    if not is_valid_fingerprint(
        artifact.get(
            "source_live_exposure_selected_sha256"
        )
    ) or not is_valid_fingerprint(
        artifact.get(
            "source_live_exposure_render_sha256"
        )
    ):
        return False

    source_count = artifact.get(
        "source_live_exposure_count"
    )

    if (
        not _is_nonnegative_int(source_count)
        or source_count < 1
    ):
        return False

    if now is not None and not (
        _now(now) < expires
    ):
        return False

    return True


# ------------------------------------------------------
# Mint
# ------------------------------------------------------


def _base_report() -> dict[str, Any]:
    return {
        "version": _VERSION,
        "mode": _MODE,
        "enabled": True,
        "minted": False,
        "reason": _REASON_NOT_MINTED,
        "ttl_seconds": None,
        "expires_at": None,
        "token_sha256": None,
        "source_live_exposure_count": None,
    }


def _persist(
    path: Path,
    artifact: dict[str, Any],
) -> bool:
    """Persist and re-read before the token may ever be handed out."""

    try:
        with LOCK:
            if path.is_file():
                existing = read_json(path)

                if not (
                    is_valid_live_recall_transport_capability_artifact(
                        existing,
                        token_sha256=artifact.get(
                            "token_sha256"
                        ),
                    )
                ):
                    # A corrupt / tampered artifact is never replaced.
                    return False
            else:
                atomic_write(path, artifact)

            persisted = read_json(path)

            return (
                is_valid_live_recall_transport_capability_artifact(
                    persisted,
                    token_sha256=artifact.get(
                        "token_sha256"
                    ),
                )
            )
    except Exception:
        return False


def mint_live_recall_transport_capability(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    now: datetime | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Mint one short-lived, Recall-only transport capability.

    Returns ``(raw_token, report)``. The raw token is returned to the
    transport ONLY; the report is privacy-safe and carries just the
    token SHA256, the TTL and counts.

    Preconditions (no fallback to Memory Flash, the Recall Surface or
    the Exposure Ledger):

      - valid trusted ``conversation_id`` / ``cognitive_request_id``;
      - a validated Live Exposure receipt exists for THIS request;
      - ``live_exposed_count >= 1`` and ``exposed_memrefs`` non-empty.

    Any failure returns ``(None, report)`` and never raises.
    """

    report = _base_report()

    try:
        if (
            not is_valid_conversation_id(conversation_id)
            or not is_valid_cognitive_request_id(
                cognitive_request_id
            )
        ):
            report["reason"] = _REASON_INVALID_IDENTITY
            return None, report

        receipt = read_live_memory_exposure(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
        )

        if not isinstance(receipt, dict):
            report["reason"] = (
                _REASON_EXPOSURE_UNAVAILABLE
            )
            return None, report

        memrefs = receipt.get("exposed_memrefs")

        count = receipt.get("live_exposed_count")

        if (
            not isinstance(memrefs, list)
            or not memrefs
            or not _is_nonnegative_int(count)
            or count < 1
        ):
            report["reason"] = (
                _REASON_EXPOSURE_UNAVAILABLE
            )
            return None, report

        selected_sha = receipt.get("selected_sha256")

        render_sha = receipt.get("render_sha256")

        if not is_valid_fingerprint(
            selected_sha
        ) or not is_valid_fingerprint(render_sha):
            report["reason"] = (
                _REASON_EXPOSURE_UNAVAILABLE
            )
            return None, report

        ttl_seconds = resolve_capability_ttl_seconds()

        issued = _now(now)

        expires = issued + timedelta(
            seconds=ttl_seconds
        )

        raw_token = (
            _TOKEN_PREFIX + secrets.token_hex(32)
        )

        token_sha256 = _sha256_text(raw_token)

        artifact: dict[str, Any] = {
            "version": _VERSION,
            "mode": _MODE,
            "token_sha256": token_sha256,
            "conversation_id": conversation_id,
            "cognitive_request_id": (
                cognitive_request_id
            ),
            "created_at": _iso(issued),
            "expires_at": _iso(expires),
            "allowed_tool": _ALLOWED_TOOL,
            "source_live_exposure_selected_sha256":
                selected_sha,
            "source_live_exposure_render_sha256":
                render_sha,
            "source_live_exposure_count": count,
        }

        if not _persist(
            live_recall_transport_capability_path(
                token_sha256
            ),
            artifact,
        ):
            report["reason"] = (
                _REASON_PERSISTENCE_FAILED
            )
            return None, report

        report.update(
            {
                "minted": True,
                "reason": _REASON_MINTED,
                "ttl_seconds": ttl_seconds,
                "expires_at": artifact["expires_at"],
                "token_sha256": token_sha256,
                "source_live_exposure_count": count,
            }
        )

        return raw_token, report

    except Exception:
        report["reason"] = _REASON_MINT_EXCEPTION

        return None, report


# ------------------------------------------------------
# Resolve
# ------------------------------------------------------


def resolve_live_recall_transport_capability(
    token: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Resolve a capability token into its trusted request binding.

    Returns ``{"conversation_id", "cognitive_request_id"}`` or None.

    The lookup is EXACT: format check -> hash the token -> read THAT
    artifact path -> validate the artifact -> validate the expiry ->
    re-read THIS artifact's Live Exposure receipt and require it to
    still match the capability's source binding. No glob, no rglob, no
    directory scan, and no fallback when the exposure receipt is gone.
    """

    try:
        if not is_live_recall_transport_capability_token(
            token
        ):
            return None

        token_sha256 = _sha256_text(token)

        artifact = read_json(
            live_recall_transport_capability_path(
                token_sha256
            )
        )

        if not (
            is_valid_live_recall_transport_capability_artifact(
                artifact,
                token_sha256=token_sha256,
                # Expiry is ALWAYS enforced when resolving; the
                # caller's clock is only a test seam.
                now=_now(now),
            )
        ):
            return None

        conversation_id = artifact["conversation_id"]

        cognitive_request_id = artifact[
            "cognitive_request_id"
        ]

        receipt = read_live_memory_exposure(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
        )

        if not isinstance(receipt, dict):
            return None

        if (
            receipt.get("selected_sha256")
            != artifact.get(
                "source_live_exposure_selected_sha256"
            )
            or receipt.get("render_sha256")
            != artifact.get(
                "source_live_exposure_render_sha256"
            )
            or receipt.get("live_exposed_count")
            != artifact.get(
                "source_live_exposure_count"
            )
        ):
            return None

        return {
            "conversation_id": conversation_id,
            "cognitive_request_id": (
                cognitive_request_id
            ),
        }

    except Exception:
        return None


def live_recall_transport_capability_expired(
    token: Any,
    *,
    now: datetime | None = None,
) -> bool:
    """Privacy-safe expiry probe for logs only.

    True only when the token has a valid format, an artifact exists
    that is structurally valid, and that artifact's expiry has already
    passed. Never exposes the token, the hash or the request identity.
    """

    try:
        if not is_live_recall_transport_capability_token(
            token
        ):
            return False

        token_sha256 = _sha256_text(token)

        artifact = read_json(
            live_recall_transport_capability_path(
                token_sha256
            )
        )

        if not (
            is_valid_live_recall_transport_capability_artifact(
                artifact,
                token_sha256=token_sha256,
            )
        ):
            return False

        expires = _parse_iso(
            artifact.get("expires_at")
        )

        if expires is None:
            return False

        return not (_now(now) < expires)

    except Exception:
        return False