from __future__ import annotations

import hashlib
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ombrebrain.context.live_recall_transport_capability import (
    resolve_capability_ttl_seconds,
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
    read_json,
    state_root,
)


# Live Memory Transport Capability v2 — provider_bound.
#
# v1 (``obrcap_``) is a Recall-only capability and stays Recall-only
# forever: an old token can never be silently widened to a new tool.
# v2 is a SEPARATE credential, minted only when the Usage Attribution
# rollout flag is ON, and it is the only capability that may ever
# reach ``UseMemory``:
#
#     obrcap_  -> allowed_tools = ["Recall"]
#     obmtcap_ -> allowed_tools = ["Recall", "UseMemory"]
#
# Everything else is bound exactly like v1:
#
#   - trusted ``conversation_id`` + ``cognitive_request_id`` only;
#   - a validated Live Exposure receipt must exist for THIS request;
#   - TTL default 300s, hard cap 900s (same knob as v1);
#   - only the token SHA256 is persisted, as the exact artifact file
#     name; the raw token is never written, logged or receipted;
#   - resolution is an exact-path lookup -- no glob, no rglob, no
#     directory scan and no fallback;
#   - the capability is NOT a Recall authorization, NOT a
#     ``recall_requested``, NOT a ``memory_loaded`` and NOT a ``used``:
#     minting it creates no Recall Request, no Related Recall, no
#     Usage Surface, no memory_usage and no lifecycle event.
#
# The raw token is 256 random bits and encodes nothing: not a CID, not
# a RID, not a memref, not a memory id. The model-facing allowed tool
# set is chosen by the server from the token KIND, never from anything
# the model sends.
#
# Default OFF at the transport layer; fail-closed here: any unexpected
# failure returns None instead of a token.

_VERSION = "memory-live-memory-transport-capability.v2"
_MODE = "provider_bound"

_ALLOWED_TOOLS = ("Recall", "UseMemory")

_TOKEN_PREFIX = "obmtcap_"
_TOKEN_RE = re.compile(r"^obmtcap_[0-9a-f]{64}$")

# The same hard cap v1 enforces (900s); declared locally so this
# module's validator never depends on a mutable import.
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


def resolve_memory_capability_ttl_seconds() -> int:
    """The v2 TTL: the same knob and the same hard cap as v1."""

    return resolve_capability_ttl_seconds()


def live_memory_transport_capability_dir() -> Path:
    return (
        state_root()
        / "live_memory_transport_capability"
    )


def live_memory_transport_capability_path(
    token_sha256: str,
) -> Path:
    """The EXACT artifact path for one token; never a directory scan."""

    return (
        live_memory_transport_capability_dir()
        / (token_sha256 + ".json")
    )


def is_live_memory_transport_capability_token(
    value: Any,
) -> bool:
    """Format-only check: ``obmtcap_`` + 64 lowercase hex chars.

    A v1 ``obrcap_`` token never matches, so a Recall-only credential
    can never be resolved as a v2 capability.
    """

    return (
        isinstance(value, str)
        and bool(_TOKEN_RE.fullmatch(value))
    )


# ------------------------------------------------------
# Artifact validation
# ------------------------------------------------------


def is_valid_live_memory_transport_capability_artifact(
    artifact: Any,
    *,
    token_sha256: Any = None,
    conversation_id: Any = None,
    cognitive_request_id: Any = None,
    now: datetime | None = None,
) -> bool:
    """Independent structural identity of one v2 capability artifact.

    Reading the file at the right path proves nothing, so every reader
    re-validates the contract, the tool binding, the token hash, the
    request identity, the created / expiry timestamps and the source
    Live Exposure binding. Anything off makes the artifact unusable.

    ``allowed_tools`` must be EXACTLY ``["Recall", "UseMemory"]``: a
    tampered artifact can never grant a narrower or a wider set.

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

    allowed_tools = artifact.get("allowed_tools")

    if (
        not isinstance(allowed_tools, list)
        or tuple(allowed_tools) != _ALLOWED_TOOLS
    ):
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

    # A capability can never outlive v1's hard cap, and an expiry that
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
        "allowed_tools": list(_ALLOWED_TOOLS),
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
                    is_valid_live_memory_transport_capability_artifact(
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
                is_valid_live_memory_transport_capability_artifact(
                    persisted,
                    token_sha256=artifact.get(
                        "token_sha256"
                    ),
                )
            )
    except Exception:
        return False


def mint_live_memory_transport_capability(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    now: datetime | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Mint one short-lived ``Recall`` + ``UseMemory`` capability.

    Returns ``(raw_token, report)``. The raw token is returned to the
    transport ONLY; the report is privacy-safe and carries just the
    token SHA256, the TTL, the allowed tools and counts.

    Preconditions (identical to v1, no fallback to Memory Flash, the
    Recall Surface or the Exposure Ledger):

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

        ttl_seconds = (
            resolve_memory_capability_ttl_seconds()
        )

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
            "allowed_tools": list(_ALLOWED_TOOLS),
            "source_live_exposure_selected_sha256":
                selected_sha,
            "source_live_exposure_render_sha256":
                render_sha,
            "source_live_exposure_count": count,
        }

        if not _persist(
            live_memory_transport_capability_path(
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


def resolve_live_memory_transport_capability(
    token: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Resolve a v2 token into its trusted request binding.

    Returns ``{"conversation_id", "cognitive_request_id",
    "allowed_tools"}`` or None.

    The lookup is EXACT: format check -> hash the token -> read THAT
    artifact path -> validate the artifact -> validate the expiry ->
    re-read THIS artifact's Live Exposure receipt and require it to
    still match the capability's source binding. No glob, no rglob, no
    directory scan, and no fallback when the exposure receipt is gone.
    """

    try:
        if not is_live_memory_transport_capability_token(
            token
        ):
            return None

        token_sha256 = _sha256_text(token)

        artifact = read_json(
            live_memory_transport_capability_path(
                token_sha256
            )
        )

        if not (
            is_valid_live_memory_transport_capability_artifact(
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
            "allowed_tools": list(_ALLOWED_TOOLS),
        }

    except Exception:
        return None


def live_memory_transport_capability_expired(
    token: Any,
    *,
    now: datetime | None = None,
) -> bool:
    """Privacy-safe expiry probe for logs only.

    True only when the token has a valid v2 format, an artifact exists
    that is structurally valid, and that artifact's expiry has already
    passed. Never exposes the token, the hash or the request identity.
    """

    try:
        if not is_live_memory_transport_capability_token(
            token
        ):
            return False

        token_sha256 = _sha256_text(token)

        artifact = read_json(
            live_memory_transport_capability_path(
                token_sha256
            )
        )

        if not (
            is_valid_live_memory_transport_capability_artifact(
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


__all__ = [
    "SAFE_MINT_REASONS",
    "is_live_memory_transport_capability_token",
    "is_valid_live_memory_transport_capability_artifact",
    "live_memory_transport_capability_expired",
    "live_memory_transport_capability_path",
    "mint_live_memory_transport_capability",
    "resolve_live_memory_transport_capability",
    "resolve_memory_capability_ttl_seconds",
]