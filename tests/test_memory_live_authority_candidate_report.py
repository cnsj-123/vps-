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
    memory_live_authority_candidate_shadow as candidate,
)
from ombrebrain.context.frontend_controlled_direct_surface import (
    build_frontend_controlled_direct_surface,
)
from ombrebrain.context.memory_authority_migration_shadow import (
    MAX_MIGRATION_ROWS,
    build_memory_authority_migration_artifact,
    observe_frontend_final_presence,
)
from ombrebrain.context.memory_delivery_shadow import (
    build_memory_delivery_plan_artifact,
)
from ombrebrain.context.memory_live_authority_candidate_shadow import (
    update_memory_live_authority_candidate,
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
    memory_live_authority_candidate_report as reporter,
)


CID = "ctx_0123456789abcdef"
RID = "ctxreq_" + "a" * 32
REV = 5

_SENTINEL_FRONTEND = "FRONTEND_CONTROL_SECRET"
_SENTINEL_OB = "OB_CONTENT_CONTROL_SECRET"
_SENTINEL_MEMID = "OB_MEMORY_ID_CONTROL_SECRET"


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


def _stack(units, *, body=None, rid=RID, revision=REV):
    snap = _snap(units, rid=rid, revision=revision)

    if body is None:
        body = _body(
            [{"role": "user", "content": "q"}]
        )

    plan = _plan(snap, rid=rid, revision=revision)

    migration = build_memory_authority_migration_artifact(
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        delivery_plan=plan,
        frontend_observation=(
            observe_frontend_final_presence(body, snap)
        ),
        live_exposure_observation={
            "applied": False,
            "exposed_count": 0,
            "estimated_tokens": 0,
            "byte_delta": 0,
        },
        pre_live_body=body,
        final_body=body,
    )

    surface = build_frontend_controlled_direct_surface(
        conversation_id=CID,
        cognitive_request_id=rid,
        source_unified_revision=revision,
        source_evidence_snapshot=snap,
        delivery_plan=plan,
        migration=migration,
    )

    return plan, migration, surface, body


def _state_env(root):
    return patch.dict(
        os.environ,
        {"OMBRE_CONTEXT_STATE_DIR": str(root)},
        clear=False,
    )


def _candidate_dir(root):
    return (
        Path(root)
        / "memory_live_authority_candidate"
        / CID
    )


def _persist(
    root,
    *,
    units,
    body=None,
    rid=RID,
):
    plan, migration, surface, body = _stack(
        units, body=body, rid=rid
    )

    with _state_env(root):
        result = update_memory_live_authority_candidate(
            conversation_id=CID,
            cognitive_request_id=rid,
            delivery_plan=plan,
            migration=migration,
            surface=surface,
            source_unified_revision=REV,
            final_body=body,
        )

    if not result.get("stored"):
        raise AssertionError(result)

    return result["artifact"]


def _write_raw(root, name, text):
    directory = _candidate_dir(root)
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
        self.assertEqual(report["report"], "aggregate_only")
        self.assertEqual(
            report["live_cutover_ready_true_count"], 0
        )
        self.assertEqual(
            report["version"], candidate.VERSION
        )
        self.assertEqual(report["mode"], candidate.MODE)

    def test_negative_limit_scans_nothing(self):
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
            )
            report = reporter.build_report(
                root=root, limit=-1
            )

        self.assertEqual(report["artifact_count"], 0)


class AggregationTests(unittest.TestCase):

    def test_valid_frontend_artifact_is_aggregated(self):
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
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
            report[
                "compatibility_candidate_ready"
                "_request_count"
            ],
            1,
        )
        self.assertEqual(
            report[
                "compatibility_candidate_blocked"
                "_request_count"
            ],
            0,
        )
        self.assertEqual(
            report[
                "request_with_frontend_control_count"
            ],
            1,
        )
        self.assertEqual(
            report[
                "frontend_controlled_direct_candidate_count"
            ],
            1,
        )
        self.assertEqual(
            report["adopted_frontend_gap_count"], 1
        )
        self.assertEqual(
            report["byte_identical_candidate_count"], 1
        )
        self.assertEqual(
            report["remaining_gap_unit_count"], 0
        )
        self.assertEqual(
            report["live_cutover_ready_true_count"], 0
        )

    def test_blocked_artifact_counts_as_blocked(self):
        body = _frontend_body("MEM-A")

        units = (
            _frontend_units_from(body)
            + ob_evidence_units(
                [_memory("m1", "OB-A")]
            )
        )

        with tempfile.TemporaryDirectory() as root:
            _persist(root, units=units, body=body)
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["valid_artifact_count"], 1
        )
        self.assertEqual(
            report[
                "compatibility_candidate_blocked"
                "_request_count"
            ],
            1,
        )
        self.assertEqual(
            report["ob_policy_unknown_count"], 1
        )
        self.assertEqual(
            report["remaining_gap_unit_count"], 1
        )

    def test_limit_truncates_scan(self):
        with tempfile.TemporaryDirectory() as root:
            for index in range(3):
                body = _frontend_body("MEM-%d" % index)

                _persist(
                    root,
                    units=_frontend_units_from(body),
                    body=body,
                    rid="ctxreq_" + str(index) * 32,
                )

            report = reporter.build_report(
                root=root, limit=2
            )

        self.assertEqual(report["artifact_count"], 2)


class InvalidArtifactTests(unittest.TestCase):

    def test_missing_and_malformed_counted_separately(
        self,
    ):
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
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

    def test_foreign_version_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            _write_raw(
                root,
                "foreign.json",
                json.dumps(
                    {
                        "version":
                            "memory-live-authority-"
                            "candidate-shadow.v2",
                        "mode": "shadow_only",
                    }
                ),
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["invalid_artifact_count"], 1
        )

    def test_live_cutover_ready_true_is_rejected(self):
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
            )
            path = _candidate_dir(root) / (
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
        self.assertEqual(
            report["live_cutover_ready_true_count"], 0
        )
        self.assertEqual(
            report[
                "frontend_controlled_direct_candidate_count"
            ],
            0,
        )

    def test_falsified_metric_rejected_by_whole(self):
        body = _frontend_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            _persist(
                root,
                units=_frontend_units_from(body),
                body=body,
            )
            path = _candidate_dir(root) / (
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
        self.assertEqual(
            report[
                "frontend_controlled_direct_candidate_count"
            ],
            0,
        )

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
            path = _candidate_dir(root) / (
                RID + ".json"
            )
            artifact = json.loads(
                path.read_text(encoding="utf-8")
            )
            artifact["rows"][0][
                "candidate_handling"
            ] = "delegated_existing_ob_cue_control"
            path.write_text(
                json.dumps(artifact), encoding="utf-8"
            )
            report = reporter.build_report(root=root)

        self.assertEqual(
            report["invalid_artifact_count"], 1
        )


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

    def test_ob_only_truncation_still_builds(self):
        body = _frontend_body("MEM-A")

        # Frontend unit FIRST so its row survives the persisted row cap,
        # then enough OB units to push some OB rows past the cap. All
        # Frontend rows stay persisted, so the candidate is still built.
        memories = [
            _memory("m%d" % index, "OB-%d" % index)
            for index in range(
                MAX_MIGRATION_ROWS + 5
            )
        ]

        units = (
            _frontend_units_from(body)
            + ob_evidence_units(memories)
        )

        with tempfile.TemporaryDirectory() as root:
            _persist(root, units=units, body=body)
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
            report[
                "frontend_controlled_direct_candidate_count"
            ],
            1,
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

        units = (
            _frontend_units_from(body)
            + ob_evidence_units(
                [
                    _memory(
                        _SENTINEL_MEMID,
                        _SENTINEL_OB,
                    )
                ]
            )
        )

        with tempfile.TemporaryDirectory() as root:
            _persist(root, units=units, body=body)
            report = reporter.build_report(root=root)

        rendered = json.dumps(report, sort_keys=True)

        self.assertNotIn(CID, rendered)
        self.assertNotIn(RID, rendered)
        self.assertNotIn(_SENTINEL_FRONTEND, rendered)
        self.assertNotIn(_SENTINEL_OB, rendered)
        self.assertNotIn(_SENTINEL_MEMID, rendered)
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
        self.assertEqual(
            payload["live_cutover_ready_true_count"], 0
        )

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
            "Compatibility Candidate",
            buffer.getvalue(),
        )
        self.assertNotIn(
            "safe to cut over", buffer.getvalue()
        )


if __name__ == "__main__":
    unittest.main()