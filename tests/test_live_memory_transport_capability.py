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

from ombrebrain.context.live_memory_transport_capability import (
    _ALLOWED_TOOLS,
    _MODE,
    _VERSION,
    is_live_memory_transport_capability_token,
    is_valid_live_memory_transport_capability_artifact,
    live_memory_transport_capability_expired,
    live_memory_transport_capability_path,
    mint_live_memory_transport_capability,
    resolve_live_memory_transport_capability,
)
from ombrebrain.context.live_recall_transport_capability import (
    is_live_recall_transport_capability_token,
    live_recall_transport_capability_path,
    mint_live_recall_transport_capability,
    resolve_live_recall_transport_capability,
)


T0 = datetime(
    2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc
)

_TOKEN_RE = re.compile(r"^obmtcap_[0-9a-f]{64}$")

_V1_TOKEN_RE = re.compile(r"^obrcap_[0-9a-f]{64}$")

_MEMORIES = [
    memory("mem-1"),
    memory("mem-2"),
]


@contextlib.contextmanager
def cap_env(root, *, ttl=None, **extra):
    values = {
        "OMBRE_CONTEXT_STATE_DIR": str(root),
        CAPABILITY_TTL_ENV: (
            str(ttl) if ttl is not None else "300"
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
        return live_memory_transport_capability_path(
            _sha256(token)
        )


def _artifact(root, token):
    return json.loads(
        _artifact_path(root, token).read_text(
            encoding="utf-8"
        )
    )


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


class LiveMemoryTransportCapabilityTests(
    unittest.TestCase
):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

        self.root = self._tmp.name

        self.addCleanup(self._tmp.cleanup)

        seed_live_exposure(
            self.root, list(_MEMORIES)
        )

    def _mint(self, **kwargs):
        with cap_env(self.root):
            return mint_live_memory_transport_capability(
                conversation_id=kwargs.pop(
                    "conversation_id", CID
                ),
                cognitive_request_id=kwargs.pop(
                    "cognitive_request_id", RID
                ),
                **kwargs,
            )

    def _mint_token(self, **kwargs):
        token, report = self._mint(**kwargs)

        self.assertIsInstance(token, str)

        self.assertTrue(report["minted"])

        return token

    def _resolve(self, token, **kwargs):
        with cap_env(self.root):
            return resolve_live_memory_transport_capability(
                token, **kwargs
            )

    # --------------------------------------------------
    # Format
    # --------------------------------------------------

    def test_token_format_is_v2_only(self):
        token = self._mint_token()

        self.assertRegex(token, _TOKEN_RE)

        self.assertTrue(
            is_live_memory_transport_capability_token(
                token
            )
        )

        # A v1 credential is never a v2 credential.
        self.assertFalse(
            is_live_memory_transport_capability_token(
                "obrcap_" + "a" * 64
            )
        )

        self.assertFalse(
            is_live_recall_transport_capability_token(
                token
            )
        )

        for value in (
            None,
            "",
            "obmtcap_",
            "obmtcap_" + "A" * 64,
            "obmtcap_" + "a" * 63,
            "obmtcap_" + "a" * 65,
            "obmtcap_" + "g" * 64,
            "obrcap_" + "a" * 64,
        ):
            self.assertFalse(
                is_live_memory_transport_capability_token(
                    value
                ),
                value,
            )

    # --------------------------------------------------
    # Mint preconditions
    # --------------------------------------------------

    def test_invalid_identity_is_refused(self):
        for kwargs in (
            {"conversation_id": "nope"},
            {"cognitive_request_id": "nope"},
            {"conversation_id": None},
            {"cognitive_request_id": None},
        ):
            token, report = self._mint(**kwargs)

            self.assertIsNone(token)
            self.assertFalse(report["minted"])
            self.assertEqual(
                report["reason"],
                "invalid_request_identity",
            )

    def test_missing_live_exposure_is_refused(self):
        with tempfile.TemporaryDirectory() as empty:
            with cap_env(empty):
                token, report = (
                    mint_live_memory_transport_capability(
                        conversation_id=CID,
                        cognitive_request_id=RID,
                    )
                )

        self.assertIsNone(token)
        self.assertEqual(
            report["reason"],
            "live_exposure_unavailable",
        )

    def test_zero_exposure_is_refused(self):
        receipt_path = _exposure_path(self.root)

        receipt = json.loads(
            receipt_path.read_text(encoding="utf-8")
        )

        receipt["live_exposed_count"] = 0
        receipt["exposed_memrefs"] = []

        receipt_path.write_text(
            json.dumps(receipt), encoding="utf-8"
        )

        _token, report = self._mint()

        self.assertFalse(report["minted"])
        self.assertEqual(
            report["reason"],
            "live_exposure_unavailable",
        )

    # --------------------------------------------------
    # Artifact
    # --------------------------------------------------

    def test_artifact_is_v2_with_both_tools(self):
        token = self._mint_token()

        artifact = _artifact(self.root, token)

        self.assertEqual(artifact["version"], _VERSION)
        self.assertEqual(artifact["mode"], _MODE)
        self.assertEqual(
            artifact["version"],
            "memory-live-memory-transport-capability.v2",
        )
        self.assertEqual(
            artifact["mode"], "provider_bound"
        )
        self.assertEqual(
            artifact["allowed_tools"],
            ["Recall", "UseMemory"],
        )
        self.assertEqual(
            artifact["allowed_tools"],
            list(_ALLOWED_TOOLS),
        )
        self.assertEqual(
            artifact["token_sha256"], _sha256(token)
        )
        self.assertEqual(artifact["conversation_id"], CID)
        self.assertEqual(
            artifact["cognitive_request_id"], RID
        )

        self.assertTrue(
            is_valid_live_memory_transport_capability_artifact(
                artifact,
                token_sha256=_sha256(token),
                conversation_id=CID,
                cognitive_request_id=RID,
            )
        )

    def test_raw_token_is_never_persisted_or_reported(self):
        token, report = self._mint()

        artifact_text = _artifact_path(
            self.root, token
        ).read_text(encoding="utf-8")

        report_text = json.dumps(report)

        # The capability artifact and the report never carry the raw
        # token, a memref, a memory id or a cue.
        for text in (
            artifact_text,
            report_text,
        ):
            self.assertNotIn(token, text)
            self.assertNotIn("obmtcap_", text)
            self.assertNotIn("obrcap_", text)
            self.assertNotIn("memref_", text)
            self.assertNotIn("mem-1", text)

        # Only the SHA256 lands on disk, as the exact file name.
        self.assertIn(
            _sha256(token), artifact_text
        )
        self.assertEqual(
            _artifact_path(self.root, token).name,
            _sha256(token) + ".json",
        )

    def test_artifact_lives_in_its_own_directory(self):
        token = self._mint_token()

        self.assertIn(
            "live_memory_transport_capability",
            str(_artifact_path(self.root, token)),
        )

        # The v1 capability store is never involved.
        with cap_env(self.root):
            v1_path = (
                live_recall_transport_capability_path(
                    _sha256(token)
                )
            )

        self.assertFalse(v1_path.exists())

    # --------------------------------------------------
    # TTL
    # --------------------------------------------------

    def test_ttl_defaults_and_caps(self):
        with cap_env(self.root):
            _token, report = (
                mint_live_memory_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

        self.assertEqual(report["ttl_seconds"], 300)

        with cap_env(self.root, ttl=60):
            _token, report = (
                mint_live_memory_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

        self.assertEqual(report["ttl_seconds"], 60)

        with cap_env(self.root, ttl=99999):
            _token, report = (
                mint_live_memory_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

        self.assertEqual(report["ttl_seconds"], 900)

        with cap_env(self.root, ttl="not-a-number"):
            _token, report = (
                mint_live_memory_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    now=T0,
                )
            )

        self.assertEqual(report["ttl_seconds"], 300)

    def test_expiry_is_enforced(self):
        token = self._mint_token()

        artifact = _artifact(self.root, token)

        expires = datetime.fromisoformat(
            artifact["expires_at"].replace(
                "Z", "+00:00"
            )
        )

        self.assertIsNotNone(
            self._resolve(
                token,
                now=expires - timedelta(seconds=1),
            )
        )

        self.assertIsNone(
            self._resolve(token, now=expires)
        )

        self.assertIsNone(
            self._resolve(
                token,
                now=expires + timedelta(seconds=1),
            )
        )

    def test_expired_probe(self):
        token = self._mint_token()

        artifact = _artifact(self.root, token)

        expires = datetime.fromisoformat(
            artifact["expires_at"].replace(
                "Z", "+00:00"
            )
        )

        with cap_env(self.root):
            self.assertFalse(
                live_memory_transport_capability_expired(
                    token,
                    now=expires - timedelta(seconds=1),
                )
            )

            self.assertTrue(
                live_memory_transport_capability_expired(
                    token, now=expires
                )
            )

            self.assertFalse(
                live_memory_transport_capability_expired(
                    "obmtcap_" + "a" * 64
                )
            )

    # --------------------------------------------------
    # Resolve
    # --------------------------------------------------

    def test_resolve_returns_trusted_binding_and_tools(self):
        token = self._mint_token()

        binding = self._resolve(token)

        self.assertEqual(
            binding,
            {
                "conversation_id": CID,
                "cognitive_request_id": RID,
                "allowed_tools": [
                    "Recall",
                    "UseMemory",
                ],
            },
        )

    def test_resolve_is_exact_path_only(self):
        token = self._mint_token()

        # A token whose hash has no artifact never resolves, and no
        # directory scan can invent one.
        self.assertIsNone(
            self._resolve("obmtcap_" + "0" * 64)
        )

        # A v1 credential never resolves through the v2 resolver.
        with cap_env(self.root):
            v1_token, _report = (
                mint_live_recall_transport_capability(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                )
            )

        self.assertIsNone(self._resolve(v1_token))

        # ...and the v2 credential never resolves through v1.
        with cap_env(self.root):
            self.assertIsNone(
                resolve_live_recall_transport_capability(
                    token
                )
            )

    def test_tampered_artifact_never_resolves(self):
        token = self._mint_token()

        base = _artifact(self.root, token)

        for mutate in (
            lambda a: a.update(
                {"allowed_tools": ["Recall"]}
            ),
            lambda a: a.update(
                {
                    "allowed_tools": [
                        "Recall",
                        "UseMemory",
                        "hold",
                    ]
                }
            ),
            lambda a: a.update({"allowed_tools": "Recall"}),
            lambda a: a.update(
                {"conversation_id": CID_B}
            ),
            lambda a: a.update(
                {"cognitive_request_id": RID_B}
            ),
            lambda a: a.update({"version": "x"}),
            lambda a: a.update({"mode": "x"}),
            lambda a: a.update({"token_sha256": "0" * 64}),
            lambda a: a.update(
                {"created_at": "2020-01-01T00:00:00Z"}
            ),
            lambda a: a.update(
                {"expires_at": "2030-01-01T00:00:00Z"}
            ),
            lambda a: a.update(
                {
                    "source_live_exposure_count": 99
                }
            ),
            lambda a: a.update(
                {
                    "source_live_exposure_render_sha256":
                        "0" * 64
                }
            ),
        ):
            artifact = dict(base)

            mutate(artifact)

            _write_artifact(
                self.root, token, artifact
            )

            self.assertIsNone(
                self._resolve(token),
                str(artifact)[:80],
            )

        _write_artifact(self.root, token, base)

        self.assertIsNotNone(self._resolve(token))

    def test_missing_exposure_receipt_never_resolves(self):
        token = self._mint_token()

        _exposure_path(self.root).unlink()

        self.assertIsNone(self._resolve(token))

    def test_corrupt_artifact_is_not_replaced(self):
        fixed_hex = "ab" * 32

        with patch(
            "secrets.token_hex",
            return_value=fixed_hex,
        ):
            token, report = self._mint()

        self.assertTrue(report["minted"])
        self.assertEqual(
            token, "obmtcap_" + fixed_hex
        )

        corrupt = {"version": "x"}

        _write_artifact(self.root, token, corrupt)

        before = _artifact_path(
            self.root, token
        ).read_text(encoding="utf-8")

        with patch(
            "secrets.token_hex",
            return_value=fixed_hex,
        ):
            token_again, report_again = self._mint()

        self.assertIsNone(token_again)
        self.assertFalse(report_again["minted"])
        self.assertEqual(
            report_again["reason"],
            "capability_persistence_failed",
        )

        self.assertEqual(
            _artifact_path(
                self.root, token
            ).read_text(encoding="utf-8"),
            before,
        )

    def test_mint_creates_no_recall_or_usage_artifact(self):
        self._mint_token()

        for directory in (
            "recall_request",
            "related_recall",
            "memory_usage",
            "live_recall_usage_surface",
            "memory_lifecycle_events",
        ):
            self.assertFalse(
                (Path(self.root) / directory).exists(),
                directory,
            )


if __name__ == "__main__":
    unittest.main()