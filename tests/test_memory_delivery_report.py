from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ombrebrain.context import (
    memory_delivery_shadow as delivery,
)
from ombrebrain.context import (
    memory_source_evidence as evidence,
)
from ombrebrain.context.memory_delivery_shadow import (
    MAX_DELIVERY_ROWS,
    update_memory_delivery_plan,
)
from ombrebrain.context.memory_source_evidence import (
    MemorySourceEvidenceSnapshot,
    frontend_evidence_units,
    ob_evidence_units,
)
from ombrebrain.gateway import (
    FrontendMemorySegment,
    OperitAdapter,
    extract_frontend_memory_segments,
)
from ombrebrain.maintenance import (
    memory_delivery_report as reporter,
)


CID = "ctx_0123456789abcdef"
RID = "ctxreq_" + "a" * 32

_SENTINEL_FRONTEND = "FRONTEND-SECRET-BODY"
_SENTINEL_OB = "OB-SECRET-CONTENT"


def _attachment(filename, body, attachment_id="x"):
    return (
        '<attachment id="'
        + attachment_id
        + '" filename="'
        + filename
        + '" type="text/plain">'
        + body
        + "</attachment>"
    )


def _body(messages):
    return json.dumps(
        {"model": "example", "messages": messages}
    ).encode("utf-8")


def _frontend_current_body(*bodies):
    text = "q " + " ".join(
        _attachment(
            "relevant_memories.txt",
            body,
            "front_%d" % index,
        )
        for index, body in enumerate(bodies)
    )

    return _body(
        [{"role": "user", "content": text}]
    )


def _frontend_units(*bodies):
    payload = json.loads(
        _frontend_current_body(*bodies)
    )
    request = OperitAdapter().adapt(payload)
    return frontend_evidence_units(
        extract_frontend_memory_segments(request)
    )


def _memory(memory_id, content):
    return {"id": memory_id, "content": content}


def _ob_units(memories):
    return ob_evidence_units(memories)


def _snap(
    units,
    *,
    frontend_available=True,
    ob_available=True,
    revision=5,
    rid=RID,
):
    return MemorySourceEvidenceSnapshot(
        version=evidence._VERSION,
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        frontend_observation_available=(
            frontend_available
        ),
        ob_observation_available=ob_available,
        units=tuple(units),
    )


def _policy(
    *,
    decision="allow_shadow",
    eligible=None,
    conversation_id=CID,
    revision=5,
):
    return {
        "version": "memory-surfacing-policy.v1",
        "mode": "shadow_only",
        "conversation_id": conversation_id,
        "source_unified_revision": revision,
        "decision": decision,
        "eligible": (
            []
            if eligible is None
            and decision == "no_surface"
            else eligible
        ),
    }


@contextlib.contextmanager
def _state_env(root):
    with patch.dict(
        os.environ,
        {"OMBRE_CONTEXT_STATE_DIR": str(root)},
        clear=False,
    ):
        yield


def _persist(
    root,
    *,
    units,
    cid=CID,
    rid=RID,
    frontend_available=True,
    ob_available=True,
    policy=None,
):
    snap = _snap(
        units,
        frontend_available=frontend_available,
        ob_available=ob_available,
        rid=rid,
    )

    with _state_env(root):
        result = update_memory_delivery_plan(
            conversation_id=cid,
            cognitive_request_id=rid,
            source_evidence_snapshot=snap,
            expected_unified_revision=5,
            base_policy=policy,
        )

    if not result.get("stored"):
        raise AssertionError(result)

    return result["plan"]


def _delivery_dir(root):
    return Path(root) / "memory_delivery_plan" / CID


def _write_raw(root, name, text):
    directory = _delivery_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class EmptyScanTests(unittest.TestCase):

    def test_missing_root_is_all_zeros(self):
        with tempfile.TemporaryDirectory() as root:
            report = reporter.build_report(root=root)

        self.assertEqual(report["artifact_count"], 0)
        self.assertEqual(report["valid_artifact_count"], 0)
        self.assertEqual(report["invalid_artifact_count"], 0)
        self.assertEqual(report["unit_count"], 0)
        self.assertEqual(report["report"], "aggregate_only")
        self.assertEqual(
            report["version"], delivery.VERSION
        )
        self.assertEqual(report["mode"], delivery.MODE)

    def test_negative_limit_scans_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            _persist(root, units=_frontend_units("MEM-A"))
            report = reporter.build_report(
                root=root, limit=-1
            )

        self.assertEqual(report["artifact_count"], 0)


class AggregationTests(unittest.TestCase):

    def test_valid_frontend_and_ob_are_aggregated(self):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=(
                    _frontend_units("MEM-A")
                    + _ob_units(
                        [_memory("m1", "OB-A")]
                    )
                ),
                policy=_policy(),
            )
            report = reporter.build_report(root=root)

        self.assertEqual(report["artifact_count"], 1)
        self.assertEqual(
            report["valid_artifact_count"], 1
        )
        self.assertEqual(
            report["invalid_artifact_count"], 0
        )
        self.assertEqual(
            report["request_with_frontend_count"], 1
        )
        self.assertEqual(
            report["request_with_ob_count"], 1
        )
        self.assertEqual(
            report["request_with_both_count"], 1
        )
        self.assertEqual(report["frontend_unit_count"], 1)
        self.assertEqual(report["ob_unit_count"], 1)
        self.assertEqual(report["unit_count"], 2)
        self.assertEqual(
            report["frontend_direct_data_count"], 1
        )

    def test_limit_truncates_scan(self):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units("MEM-A"),
                rid="ctxreq_" + "b" * 32,
            )
            _persist(
                root,
                units=_frontend_units("MEM-B"),
                rid="ctxreq_" + "c" * 32,
            )
            report = reporter.build_report(
                root=root, limit=1
            )

        self.assertEqual(report["artifact_count"], 1)
        self.assertEqual(
            report["valid_artifact_count"], 1
        )


class InvalidArtifactTests(unittest.TestCase):

    def test_missing_and_malformed_counted_separately(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units("MEM-A"),
                rid="ctxreq_" + "b" * 32,
            )
            _write_raw(
                root, "bad.json", "{not json"
            )
            report = reporter.build_report(root=root)

        # One valid + one malformed = artifact_count 2.
        self.assertEqual(report["artifact_count"], 2)
        self.assertEqual(
            report["valid_artifact_count"], 1
        )
        self.assertEqual(
            report["invalid_artifact_count"], 1
        )

    def test_empty_file_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            _write_raw(root, "empty.json", "   ")
            report = reporter.build_report(root=root)

        self.assertEqual(report["artifact_count"], 1)
        self.assertEqual(
            report["invalid_artifact_count"], 1
        )
        self.assertEqual(report["unit_count"], 0)

    def test_non_finite_constant_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            _write_raw(
                root,
                "nan.json",
                '{"metrics": {"unit_count": NaN}}',
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["invalid_artifact_count"], 1
        )
        self.assertEqual(report["unit_count"], 0)

    def test_foreign_version_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            _write_raw(
                root,
                "foreign.json",
                json.dumps(
                    {
                        "version":
                            "memory-delivery-plan-shadow.v2",
                        "mode": "shadow_only",
                    }
                ),
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["invalid_artifact_count"], 1
        )

    def test_hand_falsified_metric_rejected_by_whole(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units("MEM-A"),
            )
            path = _delivery_dir(root) / (RID + ".json")
            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )
            # Inflate a persisted counter by hand.
            artifact["metrics"]["unit_count"] = 999
            path.write_text(
                json.dumps(artifact), encoding="utf-8"
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["invalid_artifact_count"], 1
        )
        self.assertEqual(report["unit_count"], 0)

    def test_one_bad_row_invalidates_whole_artifact(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units("MEM-A"),
            )
            path = _delivery_dir(root) / (RID + ".json")
            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )
            artifact["rows"][0]["evidence_ref"] = (
                "not-a-sha256"
            )
            path.write_text(
                json.dumps(artifact), encoding="utf-8"
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["invalid_artifact_count"], 1
        )
        self.assertEqual(
            report["frontend_unit_count"], 0
        )


class ReadOnlyTests(unittest.TestCase):

    def test_report_never_writes_or_deletes(self):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units("MEM-A"),
            )
            malformed = _write_raw(
                root, "bad.json", "{not json"
            )

            before = sorted(
                p.as_posix()
                for p in Path(root).rglob("*")
            )
            malformed_before = malformed.read_text(
                encoding="utf-8"
            )

            reporter.build_report(root=root)

            after = sorted(
                p.as_posix()
                for p in Path(root).rglob("*")
            )

            self.assertEqual(before, after)
            self.assertEqual(
                malformed.read_text(encoding="utf-8"),
                malformed_before,
            )

    def test_report_carries_no_identifiers_or_text(self):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=(
                    _frontend_units(
                        _SENTINEL_FRONTEND
                    )
                    + _ob_units(
                        [
                            _memory(
                                "m1",
                                _SENTINEL_OB,
                            )
                        ]
                    )
                ),
                policy=_policy(),
            )
            report = reporter.build_report(root=root)

        rendered = json.dumps(report, sort_keys=True)

        self.assertNotIn(CID, rendered)
        self.assertNotIn(RID, rendered)
        self.assertNotIn(
            _SENTINEL_FRONTEND, rendered
        )
        self.assertNotIn(_SENTINEL_OB, rendered)
        self.assertNotIn("ctx_", rendered)
        self.assertNotIn("ctxreq_", rendered)


class TruncationAggregationTests(unittest.TestCase):

    def test_untruncated_artifacts_report_zero(self):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units("MEM-A"),
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["truncated_artifact_count"], 0
        )
        self.assertEqual(
            report["truncated_unit_count"], 0
        )

    def test_truncated_artifacts_expose_counters(self):
        memories = [
            _memory("m%d" % index, "OB-%d" % index)
            for index in range(
                MAX_DELIVERY_ROWS + 6
            )
        ]

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_ob_units(memories),
                policy=_policy(eligible=memories),
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["valid_artifact_count"], 1
        )
        self.assertEqual(
            report["truncated_artifact_count"], 1
        )
        self.assertEqual(
            report["truncated_unit_count"], 6
        )


class CliTests(unittest.TestCase):

    def test_cli_json_smoke(self):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units("MEM-A"),
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = reporter.main(
                    ["--root", root, "--json"]
                )

        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(
            payload["valid_artifact_count"], 1
        )
        self.assertEqual(payload["unit_count"], 1)

    def test_cli_text_smoke(self):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units("MEM-A"),
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = reporter.main(["--root", root])

        self.assertEqual(code, 0)
        self.assertIn(
            "Unified Memory Delivery",
            buffer.getvalue(),
        )

    def test_cli_defaults_to_state_env(self):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units("MEM-A"),
            )
            with _state_env(root):
                report = reporter.build_report()

        self.assertEqual(
            report["valid_artifact_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
