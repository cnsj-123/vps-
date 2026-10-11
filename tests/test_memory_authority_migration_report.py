from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ombrebrain.context import (
    memory_authority_migration_shadow as migration,
)
from ombrebrain.context.memory_authority_migration_shadow import (
    MAX_MIGRATION_ROWS,
    update_memory_authority_migration,
)
from ombrebrain.context.memory_delivery_shadow import (
    build_memory_delivery_plan_artifact,
)
from ombrebrain.context.memory_source_evidence import (
    MemorySourceEvidenceSnapshot,
    frontend_evidence_units,
    ob_evidence_units,
)
from ombrebrain.gateway import (
    OperitAdapter,
    extract_frontend_memory_segments,
)
from ombrebrain.maintenance import (
    memory_authority_migration_report as reporter,
)


CID = "ctx_0123456789abcdef"
RID = "ctxreq_" + "a" * 32
REV = 5

_SENTINEL_FRONTEND = "FRONTEND_SECRET_MIGRATION"
_SENTINEL_OB = "OB_CONTENT_SECRET_MIGRATION"


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


def _frontend_body(*bodies):
    text = "q " + " ".join(
        _attachment(
            "relevant_memories.txt",
            body,
            "front_%d" % index,
        )
        for index, body in enumerate(bodies)
    )

    return _body([{"role": "user", "content": text}])


def _frontend_units_from(body):
    payload = json.loads(body)
    request = OperitAdapter().adapt(payload)
    return frontend_evidence_units(
        extract_frontend_memory_segments(request)
    )


def _memory(memory_id, content):
    return {"id": memory_id, "content": content}


def _snap(units, *, rid=RID, revision=REV):
    return MemorySourceEvidenceSnapshot(
        version="memory-source-evidence.v1",
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        frontend_observation_available=True,
        ob_observation_available=True,
        units=tuple(units),
    )


def _plan(snap, *, rid=RID, revision=REV):
    return build_memory_delivery_plan_artifact(
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        source_evidence_snapshot=snap,
        base_policy=None,
    )


def _state_env(root):
    return patch.dict(
        os.environ,
        {"OMBRE_CONTEXT_STATE_DIR": str(root)},
        clear=False,
    )


def _migration_dir(root):
    return (
        Path(root)
        / "memory_authority_migration"
        / CID
    )


def _persist(
    root,
    *,
    units,
    body=None,
    rid=RID,
):
    snap = _snap(units, rid=rid)

    if body is None:
        body = _body(
            [{"role": "user", "content": "q"}]
        )

    with _state_env(root):
        result = update_memory_authority_migration(
            conversation_id=CID,
            cognitive_request_id=rid,
            source_evidence_snapshot=snap,
            delivery_plan=_plan(snap, rid=rid),
            source_unified_revision=REV,
            pre_live_body=body,
            final_body=body,
        )

    if not result.get("stored"):
        raise AssertionError(result)

    return result["artifact"]


def _write_raw(root, name, text):
    directory = _migration_dir(root)
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
            report["version"], migration.VERSION
        )
        self.assertEqual(report["mode"], migration.MODE)

    def test_negative_limit_scans_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(
                    _frontend_body("MEM-A")
                ),
                body=_frontend_body("MEM-A"),
            )
            report = reporter.build_report(
                root=root, limit=-1
            )

        self.assertEqual(report["artifact_count"], 0)


class AggregationTests(unittest.TestCase):

    def test_valid_artifacts_are_aggregated(self):
        front = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(front),
                body=front,
                rid="ctxreq_" + "b" * 32,
            )
            _persist(
                root,
                units=ob_evidence_units(
                    [_memory("m1", "OB-A")]
                ),
                rid="ctxreq_" + "c" * 32,
            )
            report = reporter.build_report(root=root)

        self.assertEqual(report["artifact_count"], 2)
        self.assertEqual(
            report["valid_artifact_count"], 2
        )
        self.assertEqual(
            report["invalid_artifact_count"], 0
        )
        self.assertEqual(
            report[
                "request_with_frontend_direct_mouth_count"
            ],
            1,
        )
        self.assertEqual(
            report[
                "contract_coverage_blocked_request_count"
            ],
            2,
        )
        self.assertEqual(report["unit_count"], 2)
        self.assertEqual(
            report["frontend_direct_bypass_count"], 1
        )
        self.assertEqual(
            report["ob_policy_unknown_count"], 1
        )

    def test_limit_truncates_scan(self):
        with tempfile.TemporaryDirectory() as root:
            for index in range(3):
                _persist(
                    root,
                    units=_frontend_units_from(
                        _frontend_body("MEM-%d" % index)
                    ),
                    body=_frontend_body("MEM-%d" % index),
                    rid="ctxreq_"
                    + str(index) * 32,
                )
            report = reporter.build_report(
                root=root, limit=2
            )

        self.assertEqual(report["artifact_count"], 2)


class InvalidArtifactTests(unittest.TestCase):

    def test_missing_and_malformed_counted_separately(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(
                    _frontend_body("MEM-A")
                ),
                body=_frontend_body("MEM-A"),
            )
            _write_raw(root, "bad.json", "{not json")
            report = reporter.build_report(root=root)

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
                            "memory-authority-"
                            "migration-shadow.v2",
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
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
            )
            path = _migration_dir(root) / (
                RID + ".json"
            )
            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )
            artifact["metrics"]["unit_count"] = 999
            path.write_text(
                json.dumps(artifact), encoding="utf-8"
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["invalid_artifact_count"], 1
        )
        self.assertEqual(report["unit_count"], 0)
        self.assertEqual(
            report["frontend_direct_bypass_count"], 0
        )

    def test_live_cutover_ready_true_is_rejected(self):
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
            )
            path = _migration_dir(root) / (
                RID + ".json"
            )
            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )
            artifact["live_cutover_ready"] = True
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
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
            )
            path = _migration_dir(root) / (
                RID + ".json"
            )
            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )
            artifact["rows"][0][
                "authority_coverage"
            ] = "gap_ob_policy_unknown"
            path.write_text(
                json.dumps(artifact), encoding="utf-8"
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["invalid_artifact_count"], 1
        )
        self.assertEqual(report["unit_count"], 0)


class TruncationAggregationTests(unittest.TestCase):

    def test_untruncated_artifacts_report_zero(self):
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report[
                "source_rows_truncated_artifact_count"
            ],
            0,
        )
        self.assertEqual(
            report["source_rows_truncated_count"], 0
        )

    def test_truncated_artifacts_expose_counters(self):
        memories = [
            _memory("m%d" % index, "OB-%d" % index)
            for index in range(
                MAX_MIGRATION_ROWS + 6
            )
        ]

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=ob_evidence_units(memories),
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["valid_artifact_count"], 1
        )
        self.assertEqual(
            report[
                "source_rows_truncated_artifact_count"
            ],
            1,
        )
        self.assertEqual(
            report["source_rows_truncated_count"], 6
        )
        # Full-scope counts are preserved even when rows were truncated.
        self.assertEqual(
            report["unit_count"], MAX_MIGRATION_ROWS + 6
        )


class ReadOnlyTests(unittest.TestCase):

    def test_report_never_writes_or_deletes(self):
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
            )
            malformed = _write_raw(
                root, "bad.json", "{not json"
            )

            before = sorted(
                path.as_posix()
                for path in Path(root).rglob("*")
            )
            malformed_before = malformed.read_text(
                encoding="utf-8"
            )

            reporter.build_report(root=root)

            after = sorted(
                path.as_posix()
                for path in Path(root).rglob("*")
            )

            self.assertEqual(before, after)
            self.assertEqual(
                malformed.read_text(encoding="utf-8"),
                malformed_before,
            )

    def test_report_carries_no_identifiers_or_text(self):
        body = _frontend_body(_SENTINEL_FRONTEND)

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=(
                    _frontend_units_from(body)
                    + ob_evidence_units(
                        [
                            _memory(
                                "m1",
                                _SENTINEL_OB,
                            )
                        ]
                    )
                ),
                body=body,
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


class CliTests(unittest.TestCase):

    def test_cli_json_smoke(self):
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
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
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = reporter.main(["--root", root])

        self.assertEqual(code, 0)
        self.assertIn(
            "Memory Authority Migration",
            buffer.getvalue(),
        )
        self.assertNotIn(
            "safe to cut over", buffer.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
