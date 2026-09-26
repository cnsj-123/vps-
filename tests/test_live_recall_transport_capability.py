from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from _recall_fixtures import (
    CAPABILITY_TTL_ENV,
    CID,
    CID_B,
    RID,
    RID_B,
    memory,
    seed_live_exposure,
)

from ombrebrain.context.live_recall_transport_capability import (
    _ALLOWED_TOOL,
    _DEFAULT_TTL_SECONDS,
    _MODE,
    _VERSION,
    is_live_recall_transport_capability_token,
    live_recall_transport_capability_expired,
    live_recall_transport_capability_path,
    mint_live_recall_transport_capability,
    resolve_capability_ttl_seconds,
    resolve_live_recall_transport_capability,
)


T0 = datetime(
    2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc
)

_TOKEN_RE = re.compile(r"^obrcap_[0-9a-f]{64}$")

_MEMORIES = [
    memory("mem-1"),
    memory("mem-2"),
]

# Live Exposure v1 caps the FULL rendered envelope at 192 tokens, so
# two short cues are used to keep the fixture well inside the budget.
_EXPOSED_COUNT = len(_MEMORIES)


@contextlib.contextmanager
def cap_env(root, *, ttl=None, **extra):
    values = {
        "OMBRE_CONTEXT_STATE_DIR": str(root),
        CAPABILITY_TTL_ENV: (
            str(ttl)
            if ttl is not None
            else str(_DEFAULT_TTL_SECONDS)
        ),
    }

    values.update(extra)

    with patch.dict(
        os.environ, values, clear=False
    ):
        yield


def _sha256(token: str) -> str:
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def _artifact_path(root, token):
    with cap_env(root):
        return live_recall_transport_capability_path(
            _sha256(token)
        )


def _artifact_text(root, token):
    return _artifact_path(
        root, token
    ).read_text(encoding="utf-8")


def _artifact(root, token):
    return json.loads(_artifact_text(root, token))


def _write_artifact(root, token, artifact):
    _artifact_path(root, token).write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )


def _exposure_path(root, conversation_id=CID, rid=RID):
    return (
        Path(root)
        / "memory_live_exposure"
        / conversation_id
        / (rid + ".json")
    )


class LiveRecallTransportCapabilityTests(
    unittest.TestCase
):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

        self.root = self._tmp.name

        self.addCleanup(self._tmp.cleanup)

        with cap_env(self.root):
            seed_live_exposure(
                self.root, list(_MEMORIES)
            )

    # --------------------------------------------------
    # TTL resolution
    # --------------------------------------------------

    def test_default_ttl_is_300(self):
        with cap_env(
            self.root, ttl=_DEFAULT_TTL_SECONDS
        ):
            self.assertEqual(
                resolve_capability_ttl_seconds(), 300
            )

        with patch.dict(
            os.environ,
            {CAPABILITY_TTL_ENV: "abc"},
            clear=False,
        ):
            self.assertEqual(
                resolve_capability_ttl_seconds(), 300
            )

        with patch.dict(
            os.environ,
            {CAPABILITY_TTL_ENV: "0"},
            clear=False,
        ):
            self.assertEqual(
                resolve_capability_ttl_seconds(), 300
            )

    def test_overlong_ttl_is_clamped(self):
        with patch.dict(
            os.environ,
            {CAPABILITY_TTL_ENV: "100000"},
            clear=False,
        ):
            self.assertEqual(
                resolve_capability_ttl_seconds(), 900
            )

        with patch.dict(
            os.environ,
            {CAPABILITY_TTL_ENV: "45"},
            clear=False,
        ):
            self.assertEqual(
                resolve_capability_ttl_seconds(), 45
            )

    # --------------------------------------------------
    # Mint
    # --------------------------------------------------

    def test_mint_requires_valid_live_exposure(self):
        with tempfile.TemporaryDirectory() as empty:
            with cap_env(empty):
                token, report = (
                    mint_live_recall_transport_capability(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        now=T0,
                    )
                )

        self.assertIsNone(token)
        self.assertFalse(report["minted"])
        self.assertEqual(
            report["reason"],
            "live_exposure_unavailable",
        )

    def test_mint_rejects_invalid_identity(self):
        with cap_env(self.root):
            token, report = (
                mint_live_recall_transport_capability(
                    conversation_id="nope",
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

        self.assertIsNone(token)
        self.assertEqual(
            report["reason"],
            "invalid_request_identity",
        )

    def test_mint_returns_token_and_privacy_safe_artifact(self):
        with cap_env(self.root):
            token, report = (
                mint_live_recall_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

        self.assertIsInstance(token, str)
        self.assertTrue(
            is_live_recall_transport_capability_token(token)
        )
        self.assertRegex(token, _TOKEN_RE)

        self.assertTrue(report["minted"])
        self.assertEqual(
            report["reason"], "capability_minted"
        )
        self.assertEqual(report["ttl_seconds"], 300)
        self.assertEqual(
            report["token_sha256"], _sha256(token)
        )
        self.assertEqual(
            report["source_live_exposure_count"],
            _EXPOSED_COUNT,
        )

        artifact_text = _artifact_text(
            self.root, token
        )

        # Raw token / memref / cue / memory id never on disk.
        self.assertNotIn(token, artifact_text)
        self.assertNotIn("obrcap_", artifact_text)
        self.assertNotIn("memref_", artifact_text)
        self.assertNotIn("mem-1", artifact_text)

        # Raw token never in the report either.
        self.assertNotIn(
            token, json.dumps(report)
        )
        self.assertNotIn(
            "obrcap_", json.dumps(report)
        )

        artifact = json.loads(artifact_text)

        self.assertEqual(artifact["version"], _VERSION)
        self.assertEqual(artifact["mode"], _MODE)
        self.assertEqual(
            artifact["allowed_tool"], _ALLOWED_TOOL
        )
        self.assertEqual(artifact["conversation_id"], CID)
        self.assertEqual(
            artifact["cognitive_request_id"], RID
        )
        self.assertEqual(
            artifact["source_live_exposure_count"],
            _EXPOSED_COUNT,
        )

        self.assertEqual(
            artifact["created_at"],
            "2026-01-01T00:00:00Z",
        )
        self.assertEqual(
            artifact["expires_at"],
            "2026-01-01T00:05:00Z",
        )

    def test_mint_creates_no_recall_state(self):
        with cap_env(self.root):
            mint_live_recall_transport_capability(
                conversation_id=CID,
                cognitive_request_id=RID,
                now=T0,
            )

        for name in (
            "recall_request",
            "related_recall",
            "memory_usage",
            "memory_lifecycle_event",
        ):
            self.assertFalse(
                (Path(self.root) / name).exists(),
                name,
            )

    # --------------------------------------------------
    # Resolve
    # --------------------------------------------------

    def test_resolve_returns_trusted_binding(self):
        with cap_env(self.root):
            token, _ = (
                mint_live_recall_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

            binding = (
                resolve_live_recall_transport_capability(
                    token, now=T0
                )
            )

        self.assertEqual(
            binding,
            {
                "conversation_id": CID,
                "cognitive_request_id": RID,
            },
        )

    def test_resolve_rejects_bad_token_format(self):
        with cap_env(self.root):
            for token in (
                None,
                "",
                "obrcap_",
                "obrcap_" + "z" * 64,
                "obrcap_" + "0" * 63,
                "obrcap_" + "0" * 65,
                "Bearer obrcap_" + "0" * 64,
                "obrcap_" + "0" * 64 + " ",
            ):
                self.assertIsNone(
                    resolve_live_recall_transport_capability(
                        token, now=T0
                    ),
                    token,
                )

    def test_resolve_expiry_boundary(self):
        with cap_env(self.root):
            token, _ = (
                mint_live_recall_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

            self.assertIsNotNone(
                resolve_live_recall_transport_capability(
                    token,
                    now=T0 + timedelta(seconds=299),
                )
            )

            self.assertIsNone(
                resolve_live_recall_transport_capability(
                    token,
                    now=T0 + timedelta(seconds=300),
                )
            )

            self.assertIsNone(
                resolve_live_recall_transport_capability(
                    token,
                    now=T0 + timedelta(seconds=301),
                )
            )

            self.assertTrue(
                live_recall_transport_capability_expired(
                    token,
                    now=T0 + timedelta(seconds=301),
                )
            )

            self.assertFalse(
                live_recall_transport_capability_expired(
                    token,
                    now=T0 + timedelta(seconds=299),
                )
            )

    def test_resolve_uses_exact_path_not_a_scan(self):
        with cap_env(self.root):
            token, _ = (
                mint_live_recall_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

            path = _artifact_path(
                self.root, token
            )

            # A valid-looking artifact under another name must NOT be
            # found: the resolver never scans the directory.
            moved = path.with_name("deadbeef.json")

            moved.write_text(
                path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            path.unlink()

            with cap_env(self.root):
                self.assertIsNone(
                    resolve_live_recall_transport_capability(
                        token, now=T0
                    )
                )

    # --------------------------------------------------
    # Tampering
    # --------------------------------------------------

    def test_tampered_artifact_fields_are_rejected(self):
        def case(mutate):
            with cap_env(self.root):
                token, _ = (
                    mint_live_recall_transport_capability(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                        now=T0,
                    )
                )

            artifact = _artifact(self.root, token)

            mutate(artifact)

            _write_artifact(
                self.root, token, artifact
            )

            with cap_env(self.root):
                return (
                    resolve_live_recall_transport_capability(
                        token, now=T0
                    )
                )

        self.assertIsNone(
            case(
                lambda a: a.update(
                    {"conversation_id": CID_B}
                )
            )
        )
        self.assertIsNone(
            case(
                lambda a: a.update(
                    {"cognitive_request_id": RID_B}
                )
            )
        )
        self.assertIsNone(
            case(lambda a: a.update({"version": "x"}))
        )
        self.assertIsNone(
            case(lambda a: a.update({"mode": "x"}))
        )
        self.assertIsNone(
            case(
                lambda a: a.update(
                    {"allowed_tool": "hold"}
                )
            )
        )
        self.assertIsNone(
            case(
                lambda a: a.update(
                    {"token_sha256": "0" * 64}
                )
            )
        )
        self.assertIsNone(
            case(
                lambda a: a.update(
                    {
                        "expires_at":
                            "2025-01-01T00:00:00Z"
                    }
                )
            )
        )
        self.assertIsNone(
            case(
                lambda a: a.update(
                    {
                        "expires_at":
                            "2027-01-01T00:00:00Z"
                    }
                )
            )
        )
        self.assertIsNone(
            case(
                lambda a: a.update(
                    {"created_at": "not-a-date"}
                )
            )
        )
        self.assertIsNone(
            case(
                lambda a: a.update(
                    {
                        "source_live_exposure_selected_sha256":
                            "0" * 64
                    }
                )
            )
        )
        self.assertIsNone(
            case(
                lambda a: a.update(
                    {
                        "source_live_exposure_render_sha256":
                            "0" * 64
                    }
                )
            )
        )
        self.assertIsNone(
            case(
                lambda a: a.update(
                    {"source_live_exposure_count": 1}
                )
            )
        )

    def test_deleted_exposure_receipt_invalidates(self):
        with cap_env(self.root):
            token, _ = (
                mint_live_recall_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

            _exposure_path(self.root).unlink()

            self.assertIsNone(
                resolve_live_recall_transport_capability(
                    token, now=T0
                )
            )

    def test_replaced_exposure_receipt_invalidates(self):
        with cap_env(self.root):
            token, _ = (
                mint_live_recall_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

            path = _exposure_path(self.root)

            receipt = json.loads(
                path.read_text(encoding="utf-8")
            )

            receipt["render_sha256"] = "f" * 64

            path.write_text(
                json.dumps(receipt),
                encoding="utf-8",
            )

            self.assertIsNone(
                resolve_live_recall_transport_capability(
                    token, now=T0
                )
            )


if __name__ == "__main__":
    unittest.main()