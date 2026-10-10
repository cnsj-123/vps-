from __future__ import annotations

import ast
import asyncio
import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from _lifecycle_fixtures import (
    CID,
    RID,
    T0,
    hours,
    iso,
    lifecycle_env,
    tree,
    usage_artifact,
    write_usage,
)
from _recall_fixtures import RID_B, memory

from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)
from ombrebrain.context.memory_lifecycle_comparison_shadow import (
    COMPARISON_SHADOW_ENV,
    MAX_COMPARISON_ROWS,
    MODE,
    VERSION,
    _metrics,
    bind_legacy_lifecycle_comparison_engine,
    build_lifecycle_comparison_artifact,
    capture_legacy_lifecycle_observations,
    is_valid_lifecycle_comparison_artifact,
    legacy_comparison_engine_bound,
    lifecycle_comparison_enabled,
    log_lifecycle_comparison_report,
    update_memory_lifecycle_comparison,
)
from ombrebrain.context.memory_lifecycle_event import (
    memory_key,
    record_lifecycle_event_from_usage,
)
from ombrebrain.context.memory_lifecycle_state import (
    resolve_lifecycle_config,
)
from ombrebrain.context.memory_surfacing_lifecycle import (
    INTEGRATION_ENV,
    apply_lifecycle_surfacing_gate,
    evaluate_lifecycle_surfacing_candidate,
    resolve_surfacing_lifecycle_config,
)
from ombrebrain.context.retrieval_decision_shadow import (
    candidate_fingerprint,
)
from ombrebrain.context.unified_context_candidate import (
    build_memory_shadow_snapshot,
)
from ombrebrain.maintenance import (
    lifecycle_comparison_report as reporter,
)
from decay_engine import DecayEngine


# Guardrail / integration / concurrency / privacy tests for the Legacy
# vs Cognitive Lifecycle Shadow Comparison v0.
#
# The comparison is OBSERVATION ONLY: it must never mutate any memory
# state, never change live behaviour, default OFF, and only ever
# produce a privacy-safe request-scoped artifact. These tests freeze
# those boundaries behaviourally AND as source contracts, and exercise
# real composition (raw candidates -> snapshot -> base policy ->
# observer -> persisted artifact) rather than only pure builders.

_REPO_ROOT = Path(__file__).resolve().parents[1]

_COMPARISON_MODULE = (
    "src/ombrebrain/context/"
    "memory_lifecycle_comparison_shadow.py"
)

_REPORTER_MODULE = (
    "src/ombrebrain/maintenance/"
    "lifecycle_comparison_report.py"
)

# Spec §41: the comparison path may call the side-effect-free Legacy
# preview API and NOTHING else on the engine / canonical store.
_FORBIDDEN_LEGACY_CALLS = {
    "touch",
    "touch_many",
    "update",
    "archive",
    "delete",
    "restore_archived",
    "hard_delete_test_bucket",
    "run_decay_cycle",
    "start",
    "ensure_started",
    "stop",
}

# Spec §42: the comparison must never write cognitive lifecycle
# evidence.
_FORBIDDEN_COGNITIVE_CALLS = {
    "record_usage",
    "record_lifecycle_event_from_usage",
    "observe_memory_usage_lifecycle",
    "process_completed_usage_for_lifecycle",
    "persist_lifecycle_state",
    "derive_memory_lifecycle_state",
}

_A = "mem-alpha"
_B = "mem-beta"
_C = "mem-gamma"
_D = "mem-delta"

_COUNTER = {"n": 0}


def _next_recall_id() -> str:
    _COUNTER["n"] += 1
    return "recall_" + format(_COUNTER["n"], "032x")


def record_use(root, memory_id, *, used_at) -> None:
    """Persist one real ``used`` lifecycle event through the real path."""

    recall_id = _next_recall_id()

    write_usage(
        root,
        usage_artifact(
            recall_id=recall_id,
            loaded_memory_ids=(memory_id,),
            used_memory_ids=(memory_id,),
            at=iso(used_at),
        ),
    )

    with lifecycle_env(root, enabled=True):
        result = record_lifecycle_event_from_usage(
            CID, RID, recall_id, 2
        )

    if not result.get("stored"):
        raise AssertionError(result)


@contextlib.contextmanager
def state_env(root):
    with patch.dict(
        os.environ,
        {"OMBRE_CONTEXT_STATE_DIR": str(root)},
        clear=False,
    ):
        yield


@contextlib.contextmanager
def comparison_env(root, *, enabled=True):
    with patch.dict(
        os.environ,
        {
            "OMBRE_CONTEXT_STATE_DIR": str(root),
            COMPARISON_SHADOW_ENV: (
                "1" if enabled else "0"
            ),
        },
        clear=False,
    ):
        yield


def candidate(memory_id, content=None, *, metadata=None):
    return {
        "id": memory_id,
        "content": (
            content
            if content is not None
            else ("cue for " + memory_id)
        ),
        "context_relevance": 0.9,
        "metadata": (
            metadata
            if metadata is not None
            else {"last_active": "2020-01-01T00:00:00Z"}
        ),
    }


def eligible_policy(
    eligible,
    *,
    decision="allow_shadow",
    conversation_id=CID,
    source_unified_revision=5,
    version="memory-surfacing-policy.v1",
    mode="shadow_only",
):
    return {
        "version": version,
        "mode": mode,
        "conversation_id": conversation_id,
        "source_unified_revision":
            source_unified_revision,
        "decision": decision,
        "reason": "surfacing_eligible",
        "eligible": list(eligible),
        "ineligible": [],
    }


def _preview(
    *,
    applicable=True,
    skip_reason=None,
    nominal_archive=False,
    score_current=1.5,
    score_after=None,
    auto_resolve=False,
    current_resolved=False,
):
    return {
        "applicable": applicable,
        "skip_reason": skip_reason,
        "current_resolved": current_resolved,
        "auto_resolve_candidate": auto_resolve,
        "score_current": score_current,
        "score_if_auto_resolve_succeeds": score_after,
        "threshold": 0.3,
        "archive_candidate_current": nominal_archive,
        "archive_candidate_after_auto_resolve_if_successful":
            nominal_archive,
        "nominal_archive_candidate": nominal_archive,
    }


class SpyDecayEngine:
    """Read-only Legacy engine stand-in.

    Implements ONLY the side-effect-free preview surface. Any access to
    a legacy mutation / lifecycle API raises, so a regression that
    starts bridging the planes fails here instead of silently
    mutating state.
    """

    def __init__(self, previews=None):
        self.calls: list = []
        self._previews = list(previews or [])
        self._index = 0

    def preview_decay_cycle_decision(self, metadata):
        self.calls.append(
            dict(metadata)
            if isinstance(metadata, dict)
            else metadata
        )

        if self._previews:
            index = min(
                self._index,
                len(self._previews) - 1,
            )

            self._index += 1

            return dict(self._previews[index])

        return _preview()

    def __getattr__(self, name):
        if name in (
            _FORBIDDEN_LEGACY_CALLS
            | _FORBIDDEN_COGNITIVE_CALLS
        ):
            raise AssertionError(
                "comparison touched engine." + name
            )

        raise AttributeError(name)


class SpyBucketManager:
    """Records every mutation call. The comparison must make none."""

    def __init__(self):
        self.updates = []
        self.archives = []
        self.deletes = []
        self.touches = []
        self.list_all_calls = 0

    async def list_all(self, *, include_archive=False):
        self.list_all_calls += 1
        return []

    async def update(self, memory_id, **kwargs):
        self.updates.append((memory_id, kwargs))
        return True

    async def archive(self, memory_id):
        self.archives.append(memory_id)
        return True

    async def delete(self, memory_id):
        self.deletes.append(memory_id)
        return True

    async def touch(self, memory_id, **kwargs):
        self.touches.append((memory_id, kwargs))
        return True


class _EngineBoundTestCase(unittest.TestCase):
    def tearDown(self):
        bind_legacy_lifecycle_comparison_engine(None)


def _ast(relative):
    return ast.parse(
        (_REPO_ROOT / relative).read_text(
            encoding="utf-8"
        )
    )


def _call_attr_names(tree_ast):
    names = set()

    for node in ast.walk(tree_ast):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ):
            names.add(node.func.attr)

    return names


# ------------------------------------------------------------------
# 1. Flag + binding
# ------------------------------------------------------------------


class FlagAndBindingTests(_EngineBoundTestCase):

    def test_default_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(COMPARISON_SHADOW_ENV, None)

            self.assertFalse(
                lifecycle_comparison_enabled()
            )

    def test_truthy_on(self):
        for value in ("1", "true", "yes", "on", "ON"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ,
                    {COMPARISON_SHADOW_ENV: value},
                    clear=False,
                ):
                    self.assertTrue(
                        lifecycle_comparison_enabled()
                    )

    def test_binding_is_reference_only(self):
        engine = SpyDecayEngine()

        self.assertFalse(
            legacy_comparison_engine_bound()
        )

        bind_legacy_lifecycle_comparison_engine(engine)

        self.assertTrue(
            legacy_comparison_engine_bound()
        )

        # Storing the reference must not invoke anything on it.
        self.assertEqual(engine.calls, [])

    def test_binding_none_disables_legacy(self):
        bind_legacy_lifecycle_comparison_engine(None)

        observations = (
            capture_legacy_lifecycle_observations(
                [candidate(_A)]
            )
        )

        self.assertEqual(len(observations), 1)
        self.assertFalse(
            observations[0]["legacy_available"]
        )
        self.assertIsNone(observations[0]["legacy"])

    def test_snapshot_stays_compatible_without_observations(
        self,
    ):
        memories = [memory(_A)]

        snapshot = build_memory_shadow_snapshot(
            conversation_id=CID,
            unified={
                "version":
                    "unified-context-candidate.v1",
                "conversation_id": CID,
                "revision": 5,
                "sections": {"memories": memories},
            },
            retrieval_memories=memories,
        )

        self.assertNotIn(
            "legacy_lifecycle_observations", snapshot
        )


# ------------------------------------------------------------------
# 2. Request-local Legacy capture
# ------------------------------------------------------------------


class CaptureTests(_EngineBoundTestCase):

    def test_capture_uses_only_preview_and_preserves_order(
        self,
    ):
        engine = SpyDecayEngine(
            [
                _preview(score_current=2.0),
                _preview(nominal_archive=True),
            ]
        )

        bind_legacy_lifecycle_comparison_engine(engine)

        first = candidate(
            _A, metadata={"importance": 5}
        )
        second = candidate(
            _B, metadata={"importance": 3}
        )

        before_first = dict(first["metadata"])
        before_second = dict(second["metadata"])

        observations = (
            capture_legacy_lifecycle_observations(
                [first, second]
            )
        )

        self.assertEqual(
            [o["retrieval_index"] for o in observations],
            [0, 1],
        )
        self.assertTrue(
            all(
                o["legacy_available"]
                for o in observations
            )
        )
        self.assertEqual(
            observations[0]["candidate_fingerprint"],
            candidate_fingerprint(first),
        )
        self.assertEqual(
            observations[1]["legacy"][
                "nominal_archive_candidate"
            ],
            True,
        )

        # Exactly one preview call per candidate, no input mutation.
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(first["metadata"], before_first)
        self.assertEqual(second["metadata"], before_second)

    def test_capture_never_mutates_bucket_state(self):
        manager = SpyBucketManager()

        engine = DecayEngine(
            {"decay": {"threshold": 0.3}}, manager
        )

        bind_legacy_lifecycle_comparison_engine(engine)

        capture_legacy_lifecycle_observations(
            [
                candidate(
                    _A,
                    metadata={
                        "importance": 3,
                        "activation_count": 1,
                        "last_active":
                            "2020-01-01T00:00:00Z",
                    },
                )
            ]
        )

        self.assertEqual(manager.updates, [])
        self.assertEqual(manager.archives, [])
        self.assertEqual(manager.deletes, [])
        self.assertEqual(manager.touches, [])
        self.assertEqual(manager.list_all_calls, 0)

    def test_preview_exception_is_fail_open(self):
        class Exploding(SpyDecayEngine):
            def preview_decay_cycle_decision(self, meta):
                raise RuntimeError("boom")

        bind_legacy_lifecycle_comparison_engine(
            Exploding()
        )

        observations = (
            capture_legacy_lifecycle_observations(
                [candidate(_A)]
            )
        )

        self.assertEqual(len(observations), 1)
        self.assertFalse(
            observations[0]["legacy_available"]
        )


# ------------------------------------------------------------------
# 3. Pure builder — categories / metrics / neutrality
# ------------------------------------------------------------------


class BuilderTests(_EngineBoundTestCase):

    def _build(
        self,
        *,
        candidates,
        eligible,
        observations,
        retrieval_candidate_count=None,
        live_gate_active=False,
    ):
        return build_lifecycle_comparison_artifact(
            conversation_id=CID,
            cognitive_request_id=RID,
            source_unified_revision=5,
            unified={
                "version":
                    "unified-context-candidate.v1",
                "conversation_id": CID,
                "revision": 5,
                "sections": {"memories": candidates},
            },
            base_policy=eligible_policy(eligible),
            legacy_observations=observations,
            retrieval_candidate_count=(
                retrieval_candidate_count
                if retrieval_candidate_count
                is not None
                else len(observations)
            ),
            live_lifecycle_gate_active=live_gate_active,
            as_of=T0,
        )

    def test_cold_start_is_neutral_not_suppressed(self):
        with tempfile.TemporaryDirectory() as root:
            engine = SpyDecayEngine(
                [
                    _preview(),
                    _preview(nominal_archive=True),
                ]
            )

            bind_legacy_lifecycle_comparison_engine(engine)

            candidates = [candidate(_A), candidate(_B)]

            observations = (
                capture_legacy_lifecycle_observations(
                    candidates
                )
            )

            with state_env(root):
                artifact = self._build(
                    candidates=candidates,
                    eligible=candidates,
                    observations=observations,
                )

        rows = artifact["candidates"]

        self.assertEqual(
            [r["comparison"]["category"] for r in rows],
            [
                "legacy_keep_cognitive_unobserved",
                "legacy_archive_cognitive_unobserved",
            ],
        )

        for row in rows:
            self.assertIs(
                row["cognitive"]["observed"], False
            )
            self.assertIs(
                row["cognitive"][
                    "would_suppress_if_supplementary"
                ],
                False,
            )

        # Primary (rank 0) preserved; the supplementary unobserved
        # candidate is neutral, never suppressed.
        self.assertEqual(
            rows[0]["cognitive"][
                "counterfactual_gate_decision"
            ],
            "primary_preserved",
        )
        self.assertEqual(
            rows[1]["cognitive"][
                "counterfactual_gate_decision"
            ],
            "neutral_unobserved",
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics[
                "cognitive_neutral_unobserved_count"
            ],
            2,
        )
        self.assertEqual(
            metrics[
                "cognitive_counterfactual_gate_suppress_count"
            ],
            0,
        )
        self.assertEqual(
            metrics["cognitive_primary_preserved_count"],
            1,
        )

    def test_dormant_history_yields_suppress_category(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(
                root, _B, used_at=T0 - hours(500)
            )

            engine = SpyDecayEngine(
                [_preview(), _preview()]
            )

            bind_legacy_lifecycle_comparison_engine(engine)

            candidates = [candidate(_A), candidate(_B)]

            observations = (
                capture_legacy_lifecycle_observations(
                    candidates
                )
            )

            with state_env(root):
                artifact = self._build(
                    candidates=candidates,
                    eligible=candidates,
                    observations=observations,
                )

        rows = artifact["candidates"]

        self.assertEqual(
            rows[0]["comparison"]["category"],
            "legacy_keep_cognitive_unobserved",
        )
        self.assertEqual(
            rows[1]["comparison"]["category"],
            "legacy_keep_cognitive_suppress",
        )
        self.assertIs(
            rows[1]["cognitive"][
                "would_suppress_if_supplementary"
            ],
            True,
        )
        self.assertEqual(
            rows[1]["cognitive"][
                "counterfactual_gate_decision"
            ],
            "would_suppress",
        )

    def test_primary_preservation_is_separate_from_suppression(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            record_use(
                root, _A, used_at=T0 - hours(500)
            )

            engine = SpyDecayEngine([_preview()])

            bind_legacy_lifecycle_comparison_engine(engine)

            candidates = [candidate(_A)]

            observations = (
                capture_legacy_lifecycle_observations(
                    candidates
                )
            )

            with state_env(root):
                artifact = self._build(
                    candidates=candidates,
                    eligible=candidates,
                    observations=observations,
                )

        row = artifact["candidates"][0]

        # The cognitive counterfactual WOULD suppress it ...
        self.assertIs(
            row["cognitive"][
                "would_suppress_if_supplementary"
            ],
            True,
        )
        # ... but the effective gate preserves the primary.
        self.assertEqual(
            row["cognitive"][
                "counterfactual_gate_decision"
            ],
            "primary_preserved",
        )

    def test_binding_unavailable_category(self):
        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        candidates = [candidate(_A)]
        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        with tempfile.TemporaryDirectory() as root:
            with state_env(root):
                artifact = build_lifecycle_comparison_artifact(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    source_unified_revision=5,
                    unified=None,
                    base_policy=eligible_policy([]),
                    legacy_observations=observations,
                    retrieval_candidate_count=1,
                    live_lifecycle_gate_active=False,
                    as_of=T0,
                )

        row = artifact["candidates"][0]

        self.assertEqual(
            row["comparison"]["category"],
            "binding_unavailable",
        )
        self.assertIsNone(row["binding"]["unified_admitted"])

    def test_legacy_not_applicable_category(self):
        engine = SpyDecayEngine(
            [
                _preview(
                    applicable=False,
                    skip_reason="type_plan",
                    score_current=None,
                )
            ]
        )

        bind_legacy_lifecycle_comparison_engine(engine)

        candidates = [candidate(_A)]
        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        with tempfile.TemporaryDirectory() as root:
            with state_env(root):
                artifact = self._build(
                    candidates=candidates,
                    eligible=candidates,
                    observations=observations,
                )

        self.assertEqual(
            artifact["candidates"][0]["comparison"][
                "category"
            ],
            "legacy_not_applicable",
        )

    def test_observation_cap_truncates_deterministically(self):
        total = MAX_COMPARISON_ROWS + 6

        engine = SpyDecayEngine([_preview()])

        bind_legacy_lifecycle_comparison_engine(engine)

        candidates = [
            candidate("mem-%03d" % index)
            for index in range(total)
        ]

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        with tempfile.TemporaryDirectory() as root:
            with state_env(root):
                artifact = self._build(
                    candidates=candidates,
                    eligible=candidates,
                    observations=observations,
                    retrieval_candidate_count=total,
                )

        self.assertEqual(
            len(artifact["candidates"]),
            MAX_COMPARISON_ROWS,
        )
        self.assertEqual(
            artifact["observation_truncated_count"],
            6,
        )
        self.assertTrue(
            artifact["observation_truncated"]
        )
        self.assertEqual(
            artifact["metrics"][
                "retrieval_candidate_count"
            ],
            total,
        )
        self.assertEqual(
            artifact["metrics"][
                "observed_candidate_count"
            ],
            MAX_COMPARISON_ROWS,
        )

    def test_numeric_safety(self):
        engine = SpyDecayEngine(
            [
                {
                    "applicable": True,
                    "skip_reason": None,
                    "current_resolved": False,
                    "auto_resolve_candidate": False,
                    "score_current": float("nan"),
                    "score_if_auto_resolve_succeeds":
                        float("inf"),
                    "threshold": 0.3,
                    "archive_candidate_current": False,
                    "archive_candidate_after_auto_resolve_if_successful":
                        False,
                    "nominal_archive_candidate": False,
                }
            ]
        )

        bind_legacy_lifecycle_comparison_engine(engine)

        candidates = [candidate(_A)]
        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        with tempfile.TemporaryDirectory() as root:
            with state_env(root):
                artifact = self._build(
                    candidates=candidates,
                    eligible=candidates,
                    observations=observations,
                )

        legacy = artifact["candidates"][0]["legacy"]

        self.assertIsNone(legacy["score_current"])
        self.assertIsNone(
            legacy["score_if_auto_resolve_succeeds"]
        )

        # No NaN / Infinity may be serializable.
        json.dumps(artifact, allow_nan=False)


# ------------------------------------------------------------------
# 4. Persistence: fail-open + artifact privacy
# ------------------------------------------------------------------


class PersistenceTests(_EngineBoundTestCase):

    def _snapshot(self, candidates, observations):
        return build_memory_shadow_snapshot(
            conversation_id=CID,
            unified={
                "version":
                    "unified-context-candidate.v1",
                "conversation_id": CID,
                "revision": 5,
                "sections": {"memories": candidates},
            },
            retrieval_memories=candidates,
            legacy_lifecycle_observations=observations,
        )

    def test_persist_failure_is_fail_open_and_privacy_safe(
        self,
    ):
        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        candidates = [candidate(_A)]
        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        with tempfile.TemporaryDirectory() as root:
            with state_env(root):
                with patch(
                    "ombrebrain.context."
                    "memory_lifecycle_comparison_shadow."
                    "atomic_write",
                    side_effect=OSError("disk full"),
                ):
                    with self.assertLogs(
                        "ombre_brain.gateway",
                        level="WARNING",
                    ) as captured:
                        report = (
                            update_memory_lifecycle_comparison(
                                conversation_id=CID,
                                cognitive_request_id=RID,
                                snapshot=self._snapshot(
                                    candidates,
                                    observations,
                                ),
                                expected_unified_revision=5,
                                base_policy=eligible_policy(
                                    candidates
                                ),
                                live_lifecycle_gate_active=False,
                            )
                        )

            logged = "\n".join(
                record.getMessage()
                for record in captured.records
            )

        self.assertFalse(report["stored"])
        self.assertEqual(
            report["reason"], "persist_failed"
        )
        self.assertIn("fail_open=true", logged)
        self.assertIn("store_failed=", logged)

        for secret in (CID, RID):
            self.assertNotIn(secret, logged)

        # No artifact was written.
        comparison_dir = (
            Path(root)
            / "memory_lifecycle_comparison"
        )

        self.assertFalse(comparison_dir.exists())

    def test_valid_ids_required(self):
        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        candidates = [candidate(_A)]
        snapshot = self._snapshot(
            candidates, []
        )

        for conversation_id, rid, reason in (
            ("bad", RID, "invalid_conversation_id"),
            (
                CID,
                "bad",
                "invalid_cognitive_request_id",
            ),
        ):
            with self.subTest(reason=reason):
                with tempfile.TemporaryDirectory() as root:
                    with state_env(root):
                        report = (
                            update_memory_lifecycle_comparison(
                                conversation_id=(
                                    conversation_id
                                ),
                                cognitive_request_id=rid,
                                snapshot=snapshot,
                                expected_unified_revision=5,
                                base_policy=eligible_policy(
                                    candidates
                                ),
                                live_lifecycle_gate_active=False,
                            )
                        )

                self.assertFalse(report["stored"])
                self.assertEqual(report["reason"], reason)

    def test_artifact_privacy_and_hashed_identity(self):
        raw_id = "SECRET_RAW_ID_ALPHA_0001"
        content = "SECRET_MEMORY_CONTENT_ALPHA"
        name = "SECRET_MEMORY_NAME_ALPHA"

        mem = memory(raw_id, content, name=name)

        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations([mem])
        )

        with tempfile.TemporaryDirectory() as root:
            with state_env(root):
                artifact = build_lifecycle_comparison_artifact(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    source_unified_revision=5,
                    unified={
                        "version":
                            "unified-context-candidate.v1",
                        "conversation_id": CID,
                        "revision": 5,
                        "sections": {"memories": [mem]},
                    },
                    base_policy=eligible_policy([mem]),
                    legacy_observations=observations,
                    retrieval_candidate_count=1,
                    live_lifecycle_gate_active=False,
                    as_of=T0,
                )

        rendered = json.dumps(artifact)

        # Memory-derived secrets must never be persisted.
        for secret in (
            raw_id,
            content,
            name,
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, rendered)

        # Request binding ids are allowed (spec §18/§19): the
        # artifact is explicitly bound to this request.
        self.assertEqual(artifact["conversation_id"], CID)
        self.assertEqual(
            artifact["cognitive_request_id"], RID
        )

        row = artifact["candidates"][0]

        # Only the cryptographic identity is persisted for memory.
        self.assertEqual(
            row["memory_key"], memory_key(raw_id)
        )
        self.assertEqual(
            row["candidate_fingerprint"],
            candidate_fingerprint(mem),
        )

    def test_log_report_is_privacy_safe(self):
        report = {
            "stored": True,
            "mode": MODE,
            "decision": "observed",
            "reason": "lifecycle_comparison_observed",
            "source_unified_revision": 5,
            "metrics": {"observed_candidate_count": 2},
        }

        with self.assertLogs(
            "ombre_brain.gateway", level="INFO"
        ) as captured:
            log_lifecycle_comparison_report(report)

        logged = "\n".join(
            record.getMessage()
            for record in captured.records
        )

        self.assertIn(
            "[gateway.context_memory_lifecycle_comparison]",
            logged,
        )

        for secret in (
            CID,
            RID,
            memory_key(_A),
            candidate_fingerprint(candidate(_A)),
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, logged)


# ------------------------------------------------------------------
# 5. Same-conversation isolation + no global re-read
# ------------------------------------------------------------------


class IsolationTests(_EngineBoundTestCase):

    def _persist(self, root, rid, revision, candidates):
        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        snapshot = build_memory_shadow_snapshot(
            conversation_id=CID,
            unified={
                "version":
                    "unified-context-candidate.v1",
                "conversation_id": CID,
                "revision": revision,
                "sections": {"memories": candidates},
            },
            retrieval_memories=candidates,
            legacy_lifecycle_observations=observations,
        )

        report = update_memory_lifecycle_comparison(
            conversation_id=CID,
            cognitive_request_id=rid,
            snapshot=snapshot,
            expected_unified_revision=revision,
            base_policy=eligible_policy(candidates),
            live_lifecycle_gate_active=False,
        )

        self.assertTrue(report["stored"])

        path = (
            Path(root)
            / "memory_lifecycle_comparison"
            / CID
            / (rid + ".json")
        )

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def test_same_conversation_requests_are_isolated(self):
        with tempfile.TemporaryDirectory() as root:
            engine = SpyDecayEngine([_preview()])
            bind_legacy_lifecycle_comparison_engine(engine)

            a_candidates = [
                candidate(
                    _A,
                    "A only content",
                    metadata={
                        "last_active":
                            "2020-01-01T00:00:00Z"
                    },
                )
            ]
            b_candidates = [
                candidate(
                    _B,
                    "B only content",
                    metadata={
                        "last_active":
                            "2020-02-02T00:00:00Z"
                    },
                )
            ]

            with state_env(root):
                artifact_a = self._persist(
                    root, RID, 5, a_candidates
                )

                # Simulate request B overwriting the shared
                # conversation-global "latest" Unified file.
                global_path = (
                    Path(root)
                    / "unified_context_candidate"
                    / (CID + ".json")
                )
                global_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                global_path.write_text(
                    json.dumps(
                        {
                            "version":
                                "unified-context-candidate.v1",
                            "conversation_id": CID,
                            "revision": 6,
                            "sections": {
                                "memories": b_candidates
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                artifact_b = self._persist(
                    root, RID_B, 6, b_candidates
                )

        fp_a = candidate_fingerprint(a_candidates[0])
        fp_b = candidate_fingerprint(b_candidates[0])

        a_fingerprints = [
            row["candidate_fingerprint"]
            for row in artifact_a["candidates"]
        ]
        b_fingerprints = [
            row["candidate_fingerprint"]
            for row in artifact_b["candidates"]
        ]

        self.assertEqual(a_fingerprints, [fp_a])
        self.assertEqual(b_fingerprints, [fp_b])
        self.assertNotIn(fp_b, a_fingerprints)
        self.assertNotIn(fp_a, b_fingerprints)

        self.assertEqual(
            artifact_a["source_unified_revision"], 5
        )
        self.assertEqual(
            artifact_b["source_unified_revision"], 6
        )

        # A's memory_key is the hashed A identity only.
        self.assertEqual(
            artifact_a["candidates"][0]["memory_key"],
            memory_key(_A),
        )
        self.assertEqual(
            artifact_b["candidates"][0]["memory_key"],
            memory_key(_B),
        )


# ------------------------------------------------------------------
# 6. Cognitive gate equivalence after refactor
# ------------------------------------------------------------------


class EquivalenceTests(unittest.TestCase):

    def test_public_evaluator_matches_live_gate(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(root, _B, used_at=T0 - hours(500))
            record_use(root, _C, used_at=T0)

            eligible = [
                candidate(_A),
                candidate(_B),
                candidate(_C),
                candidate(_D),
            ]

            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR": str(root),
                    INTEGRATION_ENV: "1",
                    "OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW":
                        "1",
                },
                clear=False,
            ):
                output = apply_lifecycle_surfacing_gate(
                    eligible_policy(eligible),
                    as_of=T0,
                )

                config = resolve_lifecycle_config()
                thresholds = (
                    resolve_surfacing_lifecycle_config()
                )

                retained_ids = [
                    item["id"]
                    for item in output["eligible"]
                ]

                for cand in eligible[1:]:
                    evaluation = (
                        evaluate_lifecycle_surfacing_candidate(
                            candidate=cand,
                            as_of=T0,
                            config=config,
                            thresholds=thresholds,
                        )
                    )

                    expected_retained = (
                        not evaluation["observed"]
                        or evaluation["kept"]
                    )

                    with self.subTest(
                        memory_id=cand["id"]
                    ):
                        self.assertEqual(
                            cand["id"] in retained_ids,
                            expected_retained,
                        )

        # Primary is always retained; dormant B is suppressed.
        self.assertEqual(retained_ids[0], _A)
        self.assertIn(_C, retained_ids)
        self.assertIn(_D, retained_ids)
        self.assertNotIn(_B, retained_ids)


# ------------------------------------------------------------------
# 7. Static safety / authority boundaries
# ------------------------------------------------------------------


class StaticSafetyTests(unittest.TestCase):

    def test_comparison_module_has_no_legacy_mutation(self):
        names = _call_attr_names(
            _ast(_COMPARISON_MODULE)
        )

        for forbidden in (
            _FORBIDDEN_LEGACY_CALLS
            | _FORBIDDEN_COGNITIVE_CALLS
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, names)

    def test_comparison_module_only_calls_preview_on_engine(
        self,
    ):
        tree_ast = _ast(_COMPARISON_MODULE)

        engine_calls = []

        for node in ast.walk(tree_ast):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "_ENGINE"
            ):
                engine_calls.append(node.func.attr)

        self.assertEqual(
            engine_calls,
            ["preview_decay_cycle_decision"],
        )

    def test_comparison_module_has_no_write_authority_imports(
        self,
    ):
        tree_ast = _ast(_COMPARISON_MODULE)

        forbidden_imports = (
            "bucket_manager",
            "embedding",
            "openai",
            "anthropic",
            "asyncio",
            "threading",
            "memory_reinforcement",
        )

        for node in ast.walk(tree_ast):
            if isinstance(node, ast.Import):
                modules = [
                    alias.name
                    for alias in node.names
                ]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                modules = []

            for module in modules:
                lowered = module.lower()

                for forbidden in forbidden_imports:
                    with self.subTest(
                        module=module,
                        forbidden=forbidden,
                    ):
                        self.assertNotIn(
                            forbidden, lowered
                        )

    def test_comparison_module_never_reads_global_unified(self):
        source = (
            _REPO_ROOT / _COMPARISON_MODULE
        ).read_text(encoding="utf-8")

        # No import of the conversation-global Unified reader and no
        # reference to its on-disk path token.
        self.assertNotIn(
            "unified_context_candidate",
            source.replace(
                "unified-context-candidate.v1", ""
            ),
        )

    def test_reporter_module_is_read_only(self):
        names = _call_attr_names(_ast(_REPORTER_MODULE))

        forbidden = {
            "write_text",
            "write_bytes",
            "mkdir",
            "unlink",
            "rmdir",
            "remove",
            "rename",
            "replace",
            "chmod",
            "touch",
            "open",
        }

        for name in forbidden:
            with self.subTest(name=name):
                self.assertNotIn(name, names)

        source = (
            _REPO_ROOT / _REPORTER_MODULE
        ).read_text(encoding="utf-8")

        self.assertNotIn("atomic_write", source)
        self.assertNotIn("shutil", source)


# ------------------------------------------------------------------
# 8. Aggregate reporter
# ------------------------------------------------------------------


def _report_row(category, **overrides):
    row = {
        "memory_key": "a" * 64,
        "candidate_fingerprint": "f" * 64,
        "retrieval_index": 0,
        "binding": {
            "unified_admitted": True,
            "base_policy_available": True,
            "base_surfacing_eligible": True,
            "base_eligible_rank": 1,
        },
        "legacy": {
            "available": True,
            "applicable": True,
            "skip_reason": None,
            "current_resolved": False,
            "auto_resolve_candidate": False,
            "score_current": 1.5,
            "score_if_auto_resolve_succeeds": None,
            "threshold": 0.3,
            "archive_candidate_current": False,
            "archive_candidate_after_auto_resolve_if_successful":
                False,
            "nominal_archive_candidate": False,
        },
        "cognitive": {
            "available": True,
            "observed": True,
            "use_count": 2,
            "strength": 0.4,
            "accessibility": 0.2,
            "salience": 0.1,
            "age_since_last_used_hours": 48.0,
            "min_accessibility": 0.3,
            "min_salience": 0.25,
            "would_suppress_if_supplementary": False,
            "would_keep_if_supplementary": True,
            "counterfactual_gate_decision": "would_keep",
            "effective_live_gate_decision": "gate_disabled",
            "invalid_event_count": 0,
        },
        "comparison": {"category": category},
    }

    for key, value in overrides.items():
        row[key] = value

    return row


def _artifact(
    rows,
    *,
    truncated=0,
    conversation_id=CID,
    request_id=RID,
    revision=5,
    live_gate_active=False,
):
    total = len(rows) + truncated

    return {
        "version": VERSION,
        "mode": MODE,
        "conversation_id": conversation_id,
        "cognitive_request_id": request_id,
        "source_unified_revision": revision,
        "created_at": iso(T0),
        "live_lifecycle_gate_active": live_gate_active,
        "observation_truncated": truncated > 0,
        "observation_truncated_count": truncated,
        # The validator recomputes these from the rows with the SAME
        # builder helper, so a fixture artifact must carry the real
        # metrics -- an empty / hand-written dict is now invalid.
        "metrics": _metrics(
            rows,
            retrieval_candidate_count=total,
            persisted_count=len(rows),
            truncated_count=truncated,
        ),
        "candidates": rows,
    }


def _write_artifact(root, conversation, request, payload):
    path = (
        Path(root)
        / "memory_lifecycle_comparison"
        / conversation
        / (request + ".json")
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload), encoding="utf-8"
    )

    return path


class ReporterTests(unittest.TestCase):

    def test_aggregates_deterministically(self):
        with tempfile.TemporaryDirectory() as root:
            _write_artifact(
                root,
                CID,
                RID,
                _artifact(
                    [
                        _report_row(
                            "legacy_keep_cognitive_keep"
                        ),
                        _report_row(
                            "legacy_keep_cognitive_suppress",
                            cognitive={
                                **_report_row(
                                    "x"
                                )["cognitive"],
                                "would_suppress_if_supplementary":
                                    True,
                                "would_keep_if_supplementary":
                                    False,
                                "counterfactual_gate_decision":
                                    "would_suppress",
                            },
                        ),
                    ]
                ),
            )

            first = reporter.build_report(root=root)
            second = reporter.build_report(root=root)

        self.assertEqual(first, second)
        self.assertEqual(first["artifact_count"], 1)
        self.assertEqual(first["valid_artifact_count"], 1)
        self.assertEqual(first["candidate_count"], 2)
        self.assertEqual(
            first["category_counts"][
                "legacy_keep_cognitive_keep"
            ],
            1,
        )
        self.assertEqual(
            first["category_counts"][
                "legacy_keep_cognitive_suppress"
            ],
            1,
        )
        self.assertEqual(
            first["cognitive_counterfactual_suppress_count"],
            1,
        )
        self.assertEqual(
            first["distributions"][
                "legacy_score_current"
            ]["count"],
            2,
        )

    def test_malformed_artifact_counts_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            _write_artifact(
                root,
                CID,
                RID,
                _artifact(
                    [
                        _report_row(
                            "legacy_keep_cognitive_keep"
                        )
                    ]
                ),
            )

            bad = (
                Path(root)
                / "memory_lifecycle_comparison"
                / CID
                / "broken.json"
            )

            bad.write_text(
                "{not json", encoding="utf-8"
            )

            report = reporter.build_report(root=root)

        self.assertEqual(report["artifact_count"], 2)
        self.assertEqual(
            report["valid_artifact_count"], 1
        )
        self.assertEqual(
            report["invalid_artifact_count"], 1
        )
        self.assertEqual(report["candidate_count"], 1)

    def test_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as root:
            for index in range(5):
                _write_artifact(
                    root,
                    CID,
                    "ctxreq_" + format(index, "032x"),
                    _artifact(
                        [
                            _report_row(
                                "legacy_keep_cognitive_keep"
                            )
                        ]
                    ),
                )

            report = reporter.build_report(
                root=root, limit=2
            )

        self.assertEqual(report["artifact_count"], 2)

    def test_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            _write_artifact(
                root,
                CID,
                RID,
                _artifact(
                    [
                        _report_row(
                            "legacy_keep_cognitive_keep"
                        )
                    ]
                ),
            )

            before = tree(root)

            reporter.build_report(root=root)

            after = tree(root)

        self.assertEqual(after, before)

    def test_empty_input_has_no_nan(self):
        with tempfile.TemporaryDirectory() as root:
            report = reporter.build_report(root=root)

        self.assertEqual(report["artifact_count"], 0)
        self.assertEqual(report["candidate_count"], 0)

        for name, stats in report[
            "distributions"
        ].items():
            with self.subTest(name=name):
                self.assertEqual(stats["count"], 0)

                for field in (
                    "min",
                    "p25",
                    "median",
                    "p75",
                    "max",
                ):
                    self.assertIsNone(stats[field])

        json.dumps(report, allow_nan=False)

    def test_all_output_values_are_finite(self):
        with tempfile.TemporaryDirectory() as root:
            _write_artifact(
                root,
                CID,
                RID,
                _artifact(
                    [
                        _report_row(
                            "legacy_keep_cognitive_keep"
                        ),
                        _report_row(
                            "legacy_keep_cognitive_keep"
                        ),
                    ]
                ),
            )

            report = reporter.build_report(root=root)

            rendered = json.dumps(report, allow_nan=False)

        self.assertNotIn("NaN", rendered)
        self.assertNotIn("Infinity", rendered)

        for name, stats in report[
            "distributions"
        ].items():
            if stats["count"] == 0:
                continue

            for field in (
                "min",
                "p25",
                "median",
                "p75",
                "max",
            ):
                with self.subTest(
                    name=name, field=field
                ):
                    self.assertTrue(
                        math.isfinite(stats[field])
                    )

    def test_no_identifiers_emitted(self):
        with tempfile.TemporaryDirectory() as root:
            _write_artifact(
                root,
                CID,
                RID,
                _artifact(
                    [
                        _report_row(
                            "legacy_keep_cognitive_keep"
                        )
                    ]
                ),
            )

            report = reporter.build_report(root=root)

        rendered = json.dumps(report)

        for secret in (
            CID,
            RID,
            "a" * 64,
            "f" * 64,
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, rendered)

    def test_cli_json_output(self):
        with tempfile.TemporaryDirectory() as root:
            _write_artifact(
                root,
                CID,
                RID,
                _artifact(
                    [
                        _report_row(
                            "legacy_keep_cognitive_keep"
                        )
                    ]
                ),
            )

            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = reporter.main(
                    ["--root", root, "--json"]
                )

        self.assertEqual(code, 0)

        parsed = json.loads(stream.getvalue())

        self.assertEqual(parsed["artifact_count"], 1)


# ------------------------------------------------------------------
# 9. Pipeline composition
# ------------------------------------------------------------------


_PIPE_FLAGS = (
    "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION",
    "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_EXPOSURE_LEDGER_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW",
    INTEGRATION_ENV,
    "OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW",
    COMPARISON_SHADOW_ENV,
)

_FLASH_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW"
)
_CONFIDENCE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW"
)


def _isolated_env():
    return {flag: "0" for flag in _PIPE_FLAGS}


def _confidence():
    return {
        "version": "context-confidence-gate.v1",
        "mode": "shadow_only",
        "conversation_id": CID,
        "decision": "allow_shadow",
        "allowed": True,
        "reason": None,
        "reasons": [],
        "stored": True,
        "duplicate": False,
        "revision": 1,
        "source_unified_revision": 5,
    }


class PipelineCompositionTests(
    unittest.IsolatedAsyncioTestCase,
):
    def tearDown(self):
        bind_legacy_lifecycle_comparison_engine(None)

    def _snapshot(self, memories):
        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                memories
            )
        )

        return build_memory_shadow_snapshot(
            conversation_id=CID,
            unified={
                "version":
                    "unified-context-candidate.v1",
                "conversation_id": CID,
                "revision": 5,
                "sections": {"memories": memories},
                "telemetry": {
                    "current_user_excluded": True,
                    "memory_count": len(memories),
                },
            },
            retrieval_memories=memories,
            legacy_lifecycle_observations=observations,
        )

    async def _run(
        self,
        *,
        root,
        comparison_on,
        flash_on,
        memories,
        comparison_impl=None,
        policy_mock=None,
        integration_on=False,
        lifecycle_shadow_on=False,
    ):
        body = b'{"messages":[]}'

        environ = {
            **_isolated_env(),
            _CONFIDENCE_ENV: "1",
            _FLASH_ENV: (
                "1" if flash_on else "0"
            ),
            COMPARISON_SHADOW_ENV: (
                "1" if comparison_on else "0"
            ),
            INTEGRATION_ENV: (
                "1" if integration_on else "0"
            ),
            "OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW": (
                "1" if lifecycle_shadow_on else "0"
            ),
            "OMBRE_CONTEXT_STATE_DIR": root,
        }

        patches = [
            patch.object(
                coordinator,
                "observe_context_sources",
                Mock(return_value=CID),
            ),
            patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                AsyncMock(
                    return_value={
                        "stored": True,
                        "revision": 5,
                        "memory_shadow_snapshot":
                            self._snapshot(memories),
                    }
                ),
            ),
            patch.object(
                coordinator,
                "update_context_confidence_gate",
                Mock(return_value=_confidence()),
            ),
        ]

        if comparison_impl is not None:
            patches.append(
                patch.object(
                    coordinator,
                    "update_memory_lifecycle_comparison",
                    comparison_impl,
                )
            )

        if policy_mock is not None:
            patches.append(
                patch.object(
                    coordinator,
                    "evaluate_surfacing_policy",
                    policy_mock,
                )
            )

        with patch.dict(
            os.environ, environ, clear=False
        ):
            with contextlib.ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)

                selected = await (
                    coordinator.run_context_pipeline(body)
                )

        self.assertEqual(selected, body)

        return body

    def _comparison_dir(self, root):
        return (
            Path(root)
            / "memory_lifecycle_comparison"
            / CID
        )

    def _flash_semantics(self, root):
        """The Flash artifact minus per-request volatile fields.

        ``cognitive_request_id`` and ``created_at`` are minted per
        request independently of the comparison observer, so they are
        dropped before comparing a fault run against the baseline.
        """

        files = sorted(
            (
                Path(root)
                / "memory_flash"
                / CID
            ).glob("*.json")
        )

        artifact = json.loads(
            files[0].read_text(encoding="utf-8")
        )

        for volatile in (
            "cognitive_request_id",
            "created_at",
        ):
            artifact.pop(volatile, None)

        return artifact

    async def test_comparison_only_writes_only_comparison(
        self,
    ):
        memories = [memory(_A), memory(_B)]

        with tempfile.TemporaryDirectory() as root:
            body = await self._run(
                root=root,
                comparison_on=True,
                flash_on=False,
                memories=memories,
            )

            comparison_dir = self._comparison_dir(root)

            self.assertTrue(comparison_dir.is_dir())

            files = sorted(
                comparison_dir.glob("*.json")
            )

            self.assertEqual(len(files), 1)

            artifact = json.loads(
                files[0].read_text(encoding="utf-8")
            )

            self.assertEqual(
                artifact["version"], VERSION
            )
            self.assertEqual(artifact["mode"], MODE)
            self.assertEqual(
                len(artifact["candidates"]), 2
            )
            self.assertEqual(
                artifact["source_unified_revision"], 5
            )

            # No other Memory stage fabricated an artifact.
            self.assertFalse(
                (Path(root) / "memory_flash").exists()
            )
            self.assertFalse(
                (Path(root) / "recall_surface").exists()
            )
            self.assertFalse(
                (Path(root) / "exposure_ledger").exists()
            )

        self.assertEqual(body, b'{"messages":[]}')

    async def test_flag_off_writes_no_artifact(self):
        memories = [memory(_A)]

        with tempfile.TemporaryDirectory() as root:
            await self._run(
                root=root,
                comparison_on=False,
                flash_on=True,
                memories=memories,
            )

            self.assertFalse(
                self._comparison_dir(root).exists()
            )

    async def test_shared_base_policy_computed_once(self):
        from ombrebrain.context.memory_surfacing_policy import (
            evaluate_surfacing_policy,
        )

        memories = [memory(_A), memory(_B)]

        spy = Mock(wraps=evaluate_surfacing_policy)

        with tempfile.TemporaryDirectory() as root:
            await self._run(
                root=root,
                comparison_on=True,
                flash_on=True,
                memories=memories,
                policy_mock=spy,
            )

        self.assertEqual(spy.call_count, 1)

    async def test_observer_failure_leaves_flash_unchanged(
        self,
    ):
        memories = [memory(_A), memory(_B)]

        with tempfile.TemporaryDirectory() as baseline_root:
            await self._run(
                root=baseline_root,
                comparison_on=False,
                flash_on=True,
                memories=memories,
            )

            baseline = self._flash_semantics(
                baseline_root
            )

        def explode(**kwargs):
            raise RuntimeError("comparison boom")

        with tempfile.TemporaryDirectory() as fault_root:
            await self._run(
                root=fault_root,
                comparison_on=True,
                flash_on=True,
                memories=memories,
                comparison_impl=Mock(
                    side_effect=explode
                ),
            )

            after = self._flash_semantics(fault_root)

            # No comparison artifact was written by the failing
            # observer, and no stale fallback was created.
            self.assertFalse(
                self._comparison_dir(fault_root).exists()
            )

        self.assertEqual(after, baseline)

    def _comparison_artifact(self, root):
        files = sorted(
            self._comparison_dir(root).glob("*.json")
        )

        self.assertEqual(len(files), 1)

        return json.loads(
            files[0].read_text(encoding="utf-8")
        )

    async def _live_gate_case(
        self,
        *,
        flash_on,
        integration_on,
        lifecycle_shadow_on,
    ):
        memories = [memory(_A), memory(_B)]

        policy = eligible_policy(
            memories, source_unified_revision=5
        )

        with tempfile.TemporaryDirectory() as root:
            # Both candidates carry a real, DORMANT ``used`` history:
            # the supplementary candidate's counterfactual MUST still be
            # computed (would_suppress) regardless of Flash.
            record_use(
                root, _A, used_at=T0 - hours(500)
            )
            record_use(
                root, _B, used_at=T0 - hours(500)
            )

            await self._run(
                root=root,
                comparison_on=True,
                flash_on=flash_on,
                memories=memories,
                policy_mock=Mock(
                    return_value=policy
                ),
                integration_on=integration_on,
                lifecycle_shadow_on=lifecycle_shadow_on,
            )

            return self._comparison_artifact(root)

    async def test_live_gate_matrix_a_flash_off(self):
        # comparison ON, flash OFF, integration ON, shadow ON.
        artifact = await self._live_gate_case(
            flash_on=False,
            integration_on=True,
            lifecycle_shadow_on=True,
        )

        self.assertIs(
            artifact["live_lifecycle_gate_active"],
            False,
        )

        decisions = [
            row["cognitive"][
                "counterfactual_gate_decision"
            ]
            for row in artifact["candidates"]
        ]

        # The counterfactual is STILL computed while Flash is OFF.
        self.assertIn("would_suppress", decisions)

        for row in artifact["candidates"]:
            self.assertEqual(
                row["cognitive"][
                    "effective_live_gate_decision"
                ],
                "gate_disabled",
            )

    async def test_live_gate_matrix_b_integration_off(self):
        # comparison ON, flash ON, integration OFF, shadow ON.
        artifact = await self._live_gate_case(
            flash_on=True,
            integration_on=False,
            lifecycle_shadow_on=True,
        )

        self.assertIs(
            artifact["live_lifecycle_gate_active"],
            False,
        )

        for row in artifact["candidates"]:
            self.assertEqual(
                row["cognitive"][
                    "effective_live_gate_decision"
                ],
                "gate_disabled",
            )

    async def test_live_gate_matrix_c_lifecycle_shadow_off(
        self,
    ):
        # comparison ON, flash ON, integration ON, shadow OFF.
        artifact = await self._live_gate_case(
            flash_on=True,
            integration_on=True,
            lifecycle_shadow_on=False,
        )

        self.assertIs(
            artifact["live_lifecycle_gate_active"],
            False,
        )

        for row in artifact["candidates"]:
            self.assertEqual(
                row["cognitive"][
                    "effective_live_gate_decision"
                ],
                "gate_disabled",
            )

    async def test_live_gate_matrix_d_all_on(self):
        # comparison ON, flash ON, integration ON, shadow ON.
        artifact = await self._live_gate_case(
            flash_on=True,
            integration_on=True,
            lifecycle_shadow_on=True,
        )

        self.assertIs(
            artifact["live_lifecycle_gate_active"],
            True,
        )

        for row in artifact["candidates"]:
            self.assertEqual(
                row["cognitive"][
                    "effective_live_gate_decision"
                ],
                row["cognitive"][
                    "counterfactual_gate_decision"
                ],
            )


# ------------------------------------------------------------------
# Review fixes: base-policy / unified binding, tri-state, validator,
# gate semantics, auto-resolve outcome metric, interleaving, bounds.
# ------------------------------------------------------------------

_UNSET = object()

_OTHER_CID = "ctx_fedcba9876543210"


def _build_artifact(
    *,
    candidates,
    observations,
    eligible=None,
    base_policy=_UNSET,
    revision=5,
    unified_candidates=None,
    unified_conversation=CID,
    retrieval_candidate_count=None,
    live_gate_active=False,
):
    if base_policy is _UNSET:
        base_policy = eligible_policy(
            eligible if eligible is not None else [],
            source_unified_revision=revision,
        )

    return build_lifecycle_comparison_artifact(
        conversation_id=CID,
        cognitive_request_id=RID,
        source_unified_revision=revision,
        unified={
            "version": "unified-context-candidate.v1",
            "conversation_id": unified_conversation,
            "revision": revision,
            "sections": {
                "memories": (
                    unified_candidates
                    if unified_candidates is not None
                    else candidates
                )
            },
        },
        base_policy=base_policy,
        legacy_observations=observations,
        retrieval_candidate_count=(
            retrieval_candidate_count
            if retrieval_candidate_count is not None
            else len(observations)
        ),
        live_lifecycle_gate_active=live_gate_active,
        as_of=T0,
    )


class BasePolicyBindingTests(_EngineBoundTestCase):
    """Review §2/§13: a missing / foreign / malformed Base Policy is
    UNKNOWN, never 'not eligible', and no conversation-global policy
    file is ever consulted."""

    def _row(
        self, base_policy, *, revision=5
    ):
        candidates = [candidate(_A)]

        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        artifact = _build_artifact(
            candidates=candidates,
            observations=observations,
            base_policy=base_policy,
            revision=revision,
        )

        return artifact["candidates"][0]

    def _assert_unknown(self, row):
        binding = row["binding"]

        self.assertIs(
            binding["base_policy_available"], False
        )
        self.assertIsNone(
            binding["base_surfacing_eligible"]
        )
        self.assertIsNone(binding["base_eligible_rank"])
        self.assertEqual(
            row["cognitive"][
                "counterfactual_gate_decision"
            ],
            "base_policy_unavailable",
        )
        # Absence never overwrites the legitimate comparison category.
        self.assertEqual(
            row["comparison"]["category"],
            "legacy_keep_cognitive_unobserved",
        )

    def test_absent_policy_is_unknown_not_ineligible(self):
        self._assert_unknown(self._row(None))

    def test_malformed_policy_is_unknown(self):
        for bad in (
            [],
            "nope",
            3,
            True,
            {"version": "memory-surfacing-policy.v1"},
        ):
            with self.subTest(bad=bad):
                self._assert_unknown(self._row(bad))

    def test_foreign_conversation_same_revision_is_unknown(
        self,
    ):
        self._assert_unknown(
            self._row(
                eligible_policy(
                    [candidate(_A)],
                    conversation_id=_OTHER_CID,
                )
            )
        )

    def test_stale_revision_is_unknown(self):
        self._assert_unknown(
            self._row(
                eligible_policy(
                    [candidate(_A)],
                    source_unified_revision=4,
                )
            )
        )

    def test_future_revision_is_unknown(self):
        self._assert_unknown(
            self._row(
                eligible_policy(
                    [candidate(_A)],
                    source_unified_revision=6,
                )
            )
        )

    def test_wrong_version_is_unknown(self):
        self._assert_unknown(
            self._row(
                eligible_policy(
                    [candidate(_A)],
                    version="memory-surfacing-policy.v2",
                )
            )
        )

    def test_wrong_mode_is_unknown(self):
        self._assert_unknown(
            self._row(
                eligible_policy(
                    [candidate(_A)], mode="live"
                )
            )
        )

    def test_non_dict_eligible_list_is_unknown(self):
        policy = eligible_policy([candidate(_A)])
        policy["eligible"] = "not-a-list"
        self._assert_unknown(self._row(policy))

    def test_unknown_string_decision_is_unknown(self):
        for bad in (
            "banana",
            "corrupt",
            "future_v2_decision",
            "",
        ):
            with self.subTest(decision=bad):
                self._assert_unknown(
                    self._row(
                        eligible_policy(
                            [candidate(_A)],
                            decision=bad,
                        )
                    )
                )

    def test_non_string_decision_is_unknown(self):
        for bad in (None, 3, True, ["allow_shadow"]):
            with self.subTest(decision=bad):
                self._assert_unknown(
                    self._row(
                        eligible_policy(
                            [candidate(_A)],
                            decision=bad,
                        )
                    )
                )

    def test_contradictory_no_surface_is_unknown(self):
        # The real v1 contract carries an EMPTY eligible set for
        # ``no_surface``. A contradictory policy that still lists
        # eligible candidates is malformed evidence, never a
        # trustworthy "not eligible".
        self._assert_unknown(
            self._row(
                eligible_policy(
                    [candidate(_A)],
                    decision="no_surface",
                )
            )
        )

    def test_valid_bound_no_surface_is_not_eligible(self):
        row = self._row(
            eligible_policy(
                [], decision="no_surface"
            )
        )

        binding = row["binding"]

        self.assertIs(
            binding["base_policy_available"], True
        )
        self.assertIs(
            binding["base_surfacing_eligible"], False
        )
        self.assertIsNone(binding["base_eligible_rank"])
        self.assertEqual(
            row["cognitive"][
                "counterfactual_gate_decision"
            ],
            "not_base_eligible",
        )

    def test_no_conversation_global_policy_file_consulted(
        self,
    ):
        candidates = [candidate(_A)]

        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        with tempfile.TemporaryDirectory() as root:
            with state_env(root):
                # A conversation-global policy file exists and marks
                # the candidate eligible -- it must be ignored.
                global_dir = (
                    Path(root)
                    / "memory_surfacing_policy"
                )
                global_dir.mkdir(
                    parents=True, exist_ok=True
                )
                (global_dir / (CID + ".json")).write_text(
                    json.dumps(
                        eligible_policy(candidates)
                    ),
                    encoding="utf-8",
                )

                artifact = _build_artifact(
                    candidates=candidates,
                    observations=observations,
                    base_policy=None,
                )

        self._assert_unknown(
            artifact["candidates"][0]
        )


class UnifiedBindingTests(_EngineBoundTestCase):
    """Review §3: a foreign request-local Unified -- even with the SAME
    numeric revision -- is never request-local authority, and no shared
    'latest' file is ever read as a fallback."""

    def test_builder_marks_foreign_unified_binding_unavailable(
        self,
    ):
        candidates = [candidate(_A)]

        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        artifact = _build_artifact(
            candidates=candidates,
            observations=observations,
            eligible=candidates,
            unified_conversation=_OTHER_CID,
        )

        row = artifact["candidates"][0]

        self.assertIsNone(
            row["binding"]["unified_admitted"]
        )
        self.assertEqual(
            row["comparison"]["category"],
            "binding_unavailable",
        )

    def test_foreign_unified_same_revision_writes_nothing(
        self,
    ):
        candidates = [candidate(_A)]

        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        # Hand-built snapshot: request-local conversation CID but a
        # Unified bound to a FOREIGN conversation at the same revision.
        snapshot = {
            "version": "memory-shadow-snapshot.v1",
            "mode": "shadow_only",
            "conversation_id": CID,
            "revision": 5,
            "unified": {
                "version":
                    "unified-context-candidate.v1",
                "conversation_id": _OTHER_CID,
                "revision": 5,
                "sections": {"memories": candidates},
            },
            "evidence": [],
            "legacy_lifecycle_observations":
                observations,
        }

        with tempfile.TemporaryDirectory() as root:
            with state_env(root):
                report = update_memory_lifecycle_comparison(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    snapshot=snapshot,
                    expected_unified_revision=5,
                    base_policy=eligible_policy(
                        candidates
                    ),
                    live_lifecycle_gate_active=False,
                )

            self.assertFalse(report["stored"])
            self.assertEqual(
                report["reason"],
                "unified_binding_invalid",
            )
            self.assertFalse(
                (
                    Path(root)
                    / "memory_lifecycle_comparison"
                ).exists()
            )

    def test_shared_latest_unified_file_is_ignored(
        self,
    ):
        candidates = [candidate(_A)]

        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        snapshot = {
            "version": "memory-shadow-snapshot.v1",
            "mode": "shadow_only",
            "conversation_id": CID,
            "revision": 5,
            "unified": {
                "version":
                    "unified-context-candidate.v1",
                "conversation_id": _OTHER_CID,
                "revision": 5,
                "sections": {"memories": candidates},
            },
            "evidence": [],
            "legacy_lifecycle_observations":
                observations,
        }

        with tempfile.TemporaryDirectory() as root:
            with state_env(root):
                # A shared conversation-global Unified file for CID
                # exists (as a concurrent request would have written).
                unified_dir = (
                    Path(root)
                    / "unified_context_candidate"
                )
                unified_dir.mkdir(
                    parents=True, exist_ok=True
                )
                (unified_dir / (CID + ".json")).write_text(
                    json.dumps(
                        snapshot["unified"]
                    ),
                    encoding="utf-8",
                )

                report = update_memory_lifecycle_comparison(
                    conversation_id=CID,
                    cognitive_request_id=RID,
                    snapshot=snapshot,
                    expected_unified_revision=5,
                    base_policy=eligible_policy(
                        candidates
                    ),
                    live_lifecycle_gate_active=False,
                )

            self.assertFalse(report["stored"])
            self.assertEqual(
                report["reason"],
                "unified_binding_invalid",
            )


class CognitiveTriStateTests(_EngineBoundTestCase):
    """Review §4/§5: unobserved != evidence unavailable, and the three
    representation counts partition the evaluated rows."""

    def test_unavailable_is_not_neutral(self):
        # Two candidates: the primary is preserved regardless, so the
        # observable "evidence unavailable" counterfactual is the
        # supplementary (rank 1) candidate.
        candidates = [candidate(_A), candidate(_B)]

        engine = SpyDecayEngine(
            [_preview(), _preview()]
        )
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        with tempfile.TemporaryDirectory() as root:
            with state_env(root):
                with patch(
                    "ombrebrain.context."
                    "memory_lifecycle_comparison_shadow."
                    "evaluate_lifecycle_surfacing_candidate",
                    side_effect=RuntimeError("boom"),
                ):
                    artifact = _build_artifact(
                        candidates=candidates,
                        observations=observations,
                        eligible=candidates,
                    )

        row = artifact["candidates"][0]
        cognitive = row["cognitive"]

        self.assertIs(cognitive["available"], False)
        self.assertIsNone(cognitive["observed"])
        self.assertIsNone(cognitive["strength"])
        self.assertIsNone(cognitive["accessibility"])
        self.assertIsNone(cognitive["salience"])
        self.assertIs(
            cognitive[
                "would_suppress_if_supplementary"
            ],
            False,
        )
        self.assertEqual(
            row["comparison"]["category"],
            "cognitive_evidence_unavailable",
        )

        # The primary is preserved; the supplementary candidate's
        # counterfactual is explicitly "evidence unavailable".
        self.assertEqual(
            row["cognitive"][
                "counterfactual_gate_decision"
            ],
            "primary_preserved",
        )
        self.assertEqual(
            artifact["candidates"][1]["cognitive"][
                "counterfactual_gate_decision"
            ],
            "cognitive_evidence_unavailable",
        )
        self.assertEqual(
            artifact["candidates"][1]["comparison"][
                "category"
            ],
            "cognitive_evidence_unavailable",
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics[
                "cognitive_evidence_unavailable_count"
            ],
            2,
        )
        self.assertEqual(
            metrics["cognitive_neutral_unobserved_count"],
            0,
        )
        self.assertEqual(
            metrics["cognitive_observed_count"], 0
        )

    def test_observed_and_neutral_are_counted_separately(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            # _B observed and kept; _C observed and dormant; _A neutral.
            record_use(root, _B, used_at=T0)
            record_use(
                root, _C, used_at=T0 - hours(500)
            )

            candidates = [
                candidate(_A),
                candidate(_B),
                candidate(_C),
            ]

            engine = SpyDecayEngine(
                [_preview(), _preview(), _preview()]
            )
            bind_legacy_lifecycle_comparison_engine(engine)

            observations = (
                capture_legacy_lifecycle_observations(
                    candidates
                )
            )

            with state_env(root):
                artifact = _build_artifact(
                    candidates=candidates,
                    observations=observations,
                    eligible=candidates,
                )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics["cognitive_observed_count"], 2
        )
        self.assertEqual(
            metrics["cognitive_neutral_unobserved_count"],
            1,
        )
        self.assertEqual(
            metrics[
                "cognitive_evidence_unavailable_count"
            ],
            0,
        )
        self.assertEqual(
            metrics["cognitive_represented_count"], 3
        )
        self.assertEqual(
            metrics["cognitive_observed_count"]
            + metrics[
                "cognitive_neutral_unobserved_count"
            ]
            + metrics[
                "cognitive_evidence_unavailable_count"
            ],
            metrics["cognitive_represented_count"],
        )

    def test_binding_unavailable_excluded_from_all_three(
        self,
    ):
        # A candidate with no usable identity: it must not inflate
        # evidence_unavailable, neutral OR observed.
        candidates = [{"content": "no identity here"}]

        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        artifact = _build_artifact(
            candidates=candidates,
            observations=observations,
            eligible=candidates,
        )

        row = artifact["candidates"][0]

        self.assertEqual(
            row["comparison"]["category"],
            "binding_unavailable",
        )
        self.assertIsNone(row["memory_key"])
        self.assertIsNone(
            row["candidate_fingerprint"]
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics["cognitive_represented_count"], 0
        )
        self.assertEqual(
            metrics["cognitive_observed_count"], 0
        )
        self.assertEqual(
            metrics["cognitive_neutral_unobserved_count"],
            0,
        )
        self.assertEqual(
            metrics[
                "cognitive_evidence_unavailable_count"
            ],
            0,
        )


class ArtifactValidatorTests(unittest.TestCase):
    """Review §6/§8: one authoritative validator, reject-by-whole."""

    def _valid(self):
        return _artifact(
            [
                _report_row(
                    "legacy_keep_cognitive_keep"
                )
            ]
        )

    def test_valid_artifact_passes(self):
        self.assertTrue(
            is_valid_lifecycle_comparison_artifact(
                self._valid()
            )
        )

    def test_reports_are_valid(self):
        # The builder output in the reporter fixtures must satisfy the
        # same validator the writer uses.
        with tempfile.TemporaryDirectory() as root:
            _write_artifact(
                root,
                CID,
                RID,
                self._valid(),
            )

            report = reporter.build_report(root=root)

        self.assertEqual(report["valid_artifact_count"], 1)
        self.assertEqual(report["invalid_artifact_count"], 0)

    def test_nan_and_infinity_rejected(self):
        for bad in (
            float("nan"),
            float("inf"),
            float("-inf"),
        ):
            with self.subTest(bad=bad):
                artifact = self._valid()
                artifact["candidates"][0]["legacy"][
                    "score_current"
                ] = bad

                self.assertFalse(
                    is_valid_lifecycle_comparison_artifact(
                        artifact
                    )
                )

    def test_invalid_identifiers_rejected(self):
        cases = {
            "version": "memory-lifecycle-comparison-shadow.v2",
            "mode": "live",
            "conversation_id": "nope",
            "cognitive_request_id": "nope",
            "source_unified_revision": -1,
            "created_at": "not-a-time",
            "live_lifecycle_gate_active": "yes",
            "observation_truncated": 1,
            "observation_truncated_count": -1,
            "metrics": [],
            "candidates": {},
        }

        for key, value in cases.items():
            with self.subTest(key=key):
                artifact = self._valid()
                artifact[key] = value

                self.assertFalse(
                    is_valid_lifecycle_comparison_artifact(
                        artifact
                    )
                )

    def test_malformed_identity_rejected(self):
        for field, value in (
            ("memory_key", "k" * 64),
            ("memory_key", "deadbeef"),
            ("candidate_fingerprint", "Z" * 64),
            ("candidate_fingerprint", None),
        ):
            with self.subTest(field=field, value=value):
                artifact = self._valid()
                artifact["candidates"][0][field] = value

                self.assertFalse(
                    is_valid_lifecycle_comparison_artifact(
                        artifact
                    )
                )

    def test_wrong_binding_types_rejected(self):
        eco = self._valid()["candidates"][0][
            "binding"
        ]

        for mutation in (
            {"base_policy_available": "yes"},
            {
                "base_policy_available": False,
                "base_surfacing_eligible": True,
            },
            {
                "base_policy_available": True,
                "base_surfacing_eligible": True,
                "base_eligible_rank": None,
            },
            {
                "base_policy_available": True,
                "base_surfacing_eligible": False,
                "base_eligible_rank": 0,
            },
            {"base_eligible_rank": -1},
        ):
            with self.subTest(mutation=mutation):
                artifact = self._valid()
                artifact["candidates"][0][
                    "binding"
                ].update(mutation)

                self.assertFalse(
                    is_valid_lifecycle_comparison_artifact(
                        artifact
                    )
                )

        # Sanity: the untouched binding is valid.
        artifact = self._valid()
        artifact["candidates"][0]["binding"] = (
            dict(eco)
        )

        self.assertTrue(
            is_valid_lifecycle_comparison_artifact(
                artifact
            )
        )

    def test_tri_state_violations_rejected(self):
        for mutation in (
            {"available": False, "observed": False},
            {"available": True, "observed": None},
            {"observed": True, "use_count": 0},
            {
                "observed": False,
                "strength": 0.5,
            },
        ):
            with self.subTest(mutation=mutation):
                base = _report_row(
                    "legacy_keep_cognitive_keep"
                )

                base["cognitive"].update(mutation)

                # Keep a consistent category for this row.
                artifact = _artifact([base])

                self.assertFalse(
                    is_valid_lifecycle_comparison_artifact(
                        artifact
                    )
                )

    def test_unknown_category_and_decision_rejected(self):
        artifact = self._valid()
        artifact["candidates"][0]["comparison"][
            "category"
        ] = "not_a_category"

        self.assertFalse(
            is_valid_lifecycle_comparison_artifact(
                artifact
            )
        )

        artifact = self._valid()
        artifact["candidates"][0]["cognitive"][
            "counterfactual_gate_decision"
        ] = "not_a_decision"

        self.assertFalse(
            is_valid_lifecycle_comparison_artifact(
                artifact
            )
        )

        artifact = self._valid()
        artifact["candidates"][0]["cognitive"][
            "effective_live_gate_decision"
        ] = "not_a_decision"

        self.assertFalse(
            is_valid_lifecycle_comparison_artifact(
                artifact
            )
        )

    def test_one_malformed_row_invalidates_the_whole(self):
        artifact = _artifact(
            [
                _report_row(
                    "legacy_keep_cognitive_keep"
                ),
                _report_row(
                    "legacy_keep_cognitive_keep"
                ),
            ]
        )

        self.assertTrue(
            is_valid_lifecycle_comparison_artifact(
                artifact
            )
        )

        artifact["candidates"][1]["comparison"][
            "category"
        ] = "bogus"

        self.assertFalse(
            is_valid_lifecycle_comparison_artifact(
                artifact
            )
        )

    def test_reporter_zeroes_a_partially_malformed_artifact(
        self,
    ):
        good = _artifact(
            [
                _report_row(
                    "legacy_keep_cognitive_keep"
                ),
                _report_row(
                    "legacy_keep_cognitive_keep"
                ),
            ]
        )

        bad = _artifact(
            [
                _report_row(
                    "legacy_keep_cognitive_keep"
                ),
                _report_row(
                    "legacy_keep_cognitive_keep"
                ),
            ]
        )

        bad["candidates"][1]["comparison"][
            "category"
        ] = "bogus"

        with tempfile.TemporaryDirectory() as root:
            _write_artifact(root, CID, RID, good)
            _write_artifact(
                root,
                CID,
                "ctxreq_" + "1" * 32,
                bad,
            )

            report = reporter.build_report(root=root)

        self.assertEqual(report["artifact_count"], 2)
        self.assertEqual(
            report["valid_artifact_count"], 1
        )
        self.assertEqual(
            report["invalid_artifact_count"], 1
        )
        # Only the valid artifact's rows contribute.
        self.assertEqual(report["candidate_count"], 2)

    def test_reporter_counts_nan_and_infinity_invalid(
        self,
    ):
        nan = _artifact(
            [_report_row("legacy_keep_cognitive_keep")]
        )
        nan["candidates"][0]["legacy"][
            "score_current"
        ] = float("nan")

        inf = _artifact(
            [_report_row("legacy_keep_cognitive_keep")]
        )
        inf["candidates"][0]["legacy"][
            "score_current"
        ] = float("inf")

        with tempfile.TemporaryDirectory() as root:
            _write_artifact(root, CID, RID, self._valid())
            _write_artifact(
                root,
                CID,
                "ctxreq_" + "2" * 32,
                nan,
            )
            _write_artifact(
                root,
                CID,
                "ctxreq_" + "3" * 32,
                inf,
            )

            report = reporter.build_report(root=root)

        self.assertEqual(report["artifact_count"], 3)
        self.assertEqual(
            report["valid_artifact_count"], 1
        )
        self.assertEqual(
            report["invalid_artifact_count"], 2
        )
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(
            report["distributions"][
                "legacy_score_current"
            ]["count"],
            1,
        )


class AdversarialValidatorMatrixTests(_EngineBoundTestCase):
    """Review §4: mutate ONE field on a REAL builder artifact and prove
    the WHOLE artifact is rejected and contributes ZERO to the report.

    Every case starts from a genuinely built artifact (real evaluators,
    real base policy, recomputed metrics), asserts that baseline is
    valid, then falsifies exactly one derived label / shape. Where the
    mutation is row-level, the metrics are refreshed first so the ONLY
    remaining inconsistency is the intended semantic one.
    """

    def _assert_whole_invalid_and_zeroed(self, artifact):
        self.assertFalse(
            is_valid_lifecycle_comparison_artifact(
                artifact
            )
        )

        with tempfile.TemporaryDirectory() as root:
            _write_artifact(root, CID, RID, artifact)

            report = reporter.build_report(root=root)

        self.assertEqual(report["artifact_count"], 1)
        self.assertEqual(
            report["valid_artifact_count"], 0
        )
        self.assertEqual(
            report["invalid_artifact_count"], 1
        )
        self.assertEqual(report["candidate_count"], 0)

        for name, count in report[
            "category_counts"
        ].items():
            with self.subTest(category=name):
                self.assertEqual(count, 0)

        self.assertEqual(
            report["distributions"][
                "cognitive_strength"
            ]["count"],
            0,
        )

    def _refresh_metrics(self, artifact):
        rows = artifact["candidates"]

        artifact["metrics"] = _metrics(
            rows,
            retrieval_candidate_count=artifact[
                "metrics"
            ]["retrieval_candidate_count"],
            persisted_count=len(rows),
            truncated_count=artifact[
                "observation_truncated_count"
            ],
        )

    def _primary_artifact(self):
        # Real builder artifact: one OBSERVED + KEPT primary
        # candidate (legacy KEEP, cognitive KEEP, rank 0).
        with tempfile.TemporaryDirectory() as root:
            record_use(root, _A, used_at=T0)

            candidates = [candidate(_A)]

            engine = SpyDecayEngine([_preview()])
            bind_legacy_lifecycle_comparison_engine(
                engine
            )

            observations = (
                capture_legacy_lifecycle_observations(
                    candidates
                )
            )

            with state_env(root):
                return _build_artifact(
                    candidates=candidates,
                    observations=observations,
                    eligible=candidates,
                )

    def _kept_pair_artifact(self):
        # Real builder artifact: rank 0 primary + rank 1 observed keep.
        with tempfile.TemporaryDirectory() as root:
            record_use(root, _A, used_at=T0)
            record_use(root, _B, used_at=T0)

            candidates = [
                candidate(_A),
                candidate(_B),
            ]

            engine = SpyDecayEngine(
                [_preview(), _preview()]
            )
            bind_legacy_lifecycle_comparison_engine(
                engine
            )

            observations = (
                capture_legacy_lifecycle_observations(
                    candidates
                )
            )

            with state_env(root):
                return _build_artifact(
                    candidates=candidates,
                    observations=observations,
                    eligible=candidates,
                )

    def _binding_unavailable_artifact(self):
        candidates = [{"content": "no identity"}]

        engine = SpyDecayEngine([_preview()])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        return _build_artifact(
            candidates=candidates,
            observations=observations,
            eligible=candidates,
        )

    # 1. legacy KEEP + cognitive KEEP, category changed to a suppress
    #    category that contradicts the row's own evidence.
    def test_case1_category_contradicts_evidence(self):
        artifact = self._primary_artifact()

        row = artifact["candidates"][0]

        self.assertEqual(
            row["comparison"]["category"],
            "legacy_keep_cognitive_keep",
        )

        row["comparison"][
            "category"
        ] = "legacy_archive_cognitive_suppress"

        self._refresh_metrics(artifact)

        self._assert_whole_invalid_and_zeroed(
            artifact
        )

    # 2. base_policy_available=false but counterfactual=would_suppress.
    def test_case2_policy_unavailable_but_suppress(self):
        artifact = self._primary_artifact()

        binding = artifact["candidates"][0]["binding"]

        binding["base_policy_available"] = False
        binding["base_surfacing_eligible"] = None
        binding["base_eligible_rank"] = None

        artifact["candidates"][0]["cognitive"][
            "counterfactual_gate_decision"
        ] = "would_suppress"

        self._refresh_metrics(artifact)

        self._assert_whole_invalid_and_zeroed(
            artifact
        )

    # 3. base eligible rank=0 but counterfactual=would_suppress.
    def test_case3_primary_rank_but_suppress(self):
        artifact = self._primary_artifact()

        row = artifact["candidates"][0]

        self.assertEqual(
            row["binding"]["base_eligible_rank"], 0
        )

        row["cognitive"][
            "counterfactual_gate_decision"
        ] = "would_suppress"

        self._refresh_metrics(artifact)

        self._assert_whole_invalid_and_zeroed(
            artifact
        )

    # 4. live_lifecycle_gate_active=false but effective=would_suppress.
    def test_case4_live_off_effective_suppress(self):
        artifact = self._primary_artifact()

        self.assertIs(
            artifact["live_lifecycle_gate_active"],
            False,
        )

        row = artifact["candidates"][0]

        self.assertEqual(
            row["cognitive"][
                "counterfactual_gate_decision"
            ],
            "primary_preserved",
        )

        row["cognitive"][
            "effective_live_gate_decision"
        ] = "would_suppress"

        self._refresh_metrics(artifact)

        self._assert_whole_invalid_and_zeroed(
            artifact
        )

    # 5. live=true, counterfactual=would_keep, effective=would_suppress.
    def test_case5_live_on_effective_mismatch(self):
        artifact = self._kept_pair_artifact()

        artifact["live_lifecycle_gate_active"] = True

        row = artifact["candidates"][1]

        self.assertEqual(
            row["cognitive"][
                "counterfactual_gate_decision"
            ],
            "would_keep",
        )

        row["cognitive"][
            "effective_live_gate_decision"
        ] = "would_suppress"

        self._refresh_metrics(artifact)

        self._assert_whole_invalid_and_zeroed(
            artifact
        )

    # 6. available=false, observed=null, use_count=5.
    def test_case6_unavailable_with_use_count(self):
        artifact = self._primary_artifact()

        cog = artifact["candidates"][0]["cognitive"]

        cog["available"] = False
        cog["observed"] = None
        cog["use_count"] = 5
        cog["strength"] = None
        cog["accessibility"] = None
        cog["salience"] = None
        cog["age_since_last_used_hours"] = None
        cog["would_suppress_if_supplementary"] = False
        cog["would_keep_if_supplementary"] = False

        self._refresh_metrics(artifact)

        self._assert_whole_invalid_and_zeroed(
            artifact
        )

    # 7. available=false but strength=0.5.
    def test_case7_unavailable_with_strength(self):
        artifact = self._primary_artifact()

        cog = artifact["candidates"][0]["cognitive"]

        cog["available"] = False
        cog["observed"] = None
        cog["use_count"] = 0
        cog["strength"] = 0.5
        cog["accessibility"] = None
        cog["salience"] = None
        cog["age_since_last_used_hours"] = None
        cog["would_suppress_if_supplementary"] = False
        cog["would_keep_if_supplementary"] = False

        self._refresh_metrics(artifact)

        self._assert_whole_invalid_and_zeroed(
            artifact
        )

    # 8. available=true, observed=false, would_keep=true.
    def test_case8_neutral_with_keep_claim(self):
        artifact = self._primary_artifact()

        cog = artifact["candidates"][0]["cognitive"]

        cog["available"] = True
        cog["observed"] = False
        cog["use_count"] = 0
        cog["strength"] = None
        cog["accessibility"] = None
        cog["salience"] = None
        cog["age_since_last_used_hours"] = None
        cog["would_suppress_if_supplementary"] = False
        cog["would_keep_if_supplementary"] = True

        self._refresh_metrics(artifact)

        self._assert_whole_invalid_and_zeroed(
            artifact
        )

    # 9. binding_unavailable category but base_policy_available=true.
    def test_case9_binding_unavailable_with_claim(self):
        artifact = self._binding_unavailable_artifact()

        row = artifact["candidates"][0]

        self.assertEqual(
            row["comparison"]["category"],
            "binding_unavailable",
        )

        row["binding"]["base_policy_available"] = True
        row["binding"]["base_surfacing_eligible"] = False
        row["binding"]["base_eligible_rank"] = None

        self._refresh_metrics(artifact)

        self._assert_whole_invalid_and_zeroed(
            artifact
        )

    # 10. valid rows but top-level metrics manually falsified.
    def test_case10_falsified_metrics(self):
        artifact = self._primary_artifact()

        self.assertTrue(
            is_valid_lifecycle_comparison_artifact(
                artifact
            )
        )

        artifact["metrics"][
            "observed_candidate_count"
        ] += 1

        self._assert_whole_invalid_and_zeroed(
            artifact
        )


class ReporterBoundTests(unittest.TestCase):
    """Review §16: a programmatic limit can never become unbounded."""

    def test_negative_limit_scans_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            _write_artifact(
                root,
                CID,
                RID,
                _artifact(
                    [
                        _report_row(
                            "legacy_keep_cognitive_keep"
                        )
                    ]
                ),
            )

            report = reporter.build_report(
                root=root, limit=-1
            )

        self.assertEqual(report["artifact_count"], 0)
        self.assertEqual(report["candidate_count"], 0)

    def test_negative_limit_cli_scans_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            _write_artifact(
                root,
                CID,
                RID,
                _artifact(
                    [
                        _report_row(
                            "legacy_keep_cognitive_keep"
                        )
                    ]
                ),
            )

            buffer = io.StringIO()

            with contextlib.redirect_stdout(buffer):
                code = reporter.main(
                    [
                        "--root",
                        root,
                        "--limit",
                        "-5",
                        "--json",
                    ]
                )

        self.assertEqual(code, 0)

        payload = json.loads(buffer.getvalue())

        self.assertEqual(payload["artifact_count"], 0)


class GateSemanticsTests(_EngineBoundTestCase):
    """Review §9/§10: the counterfactual is preserved whether or not
    the live gate participates, and live participation is captured."""

    def _artifact(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(
                root, _B, used_at=T0 - hours(500)
            )

            candidates = [
                candidate(_A),
                candidate(_B),
            ]

            engine = SpyDecayEngine(
                [_preview(), _preview()]
            )
            bind_legacy_lifecycle_comparison_engine(engine)

            observations = (
                capture_legacy_lifecycle_observations(
                    candidates
                )
            )

            with state_env(root):
                return _build_artifact(
                    candidates=candidates,
                    observations=observations,
                    eligible=candidates,
                )

    def test_gate_disabled_preserves_counterfactual(self):
        artifact = self._artifact()

        row = artifact["candidates"][1]

        self.assertEqual(
            row["cognitive"][
                "counterfactual_gate_decision"
            ],
            "would_suppress",
        )
        self.assertEqual(
            row["cognitive"][
                "effective_live_gate_decision"
            ],
            "gate_disabled",
        )
        self.assertIs(
            artifact["live_lifecycle_gate_active"],
            False,
        )

    def test_gate_active_effective_matches_counterfactual(
        self,
    ):
        # The builder records the explicit request-local orchestration
        # fact it is given; it no longer infers it from environment
        # flags. With live participation TRUE the effective decision
        # must equal the (unchanged) counterfactual.
        with tempfile.TemporaryDirectory() as root:
            record_use(
                root, _B, used_at=T0 - hours(500)
            )

            candidates = [
                candidate(_A),
                candidate(_B),
            ]

            engine = SpyDecayEngine(
                [_preview(), _preview()]
            )
            bind_legacy_lifecycle_comparison_engine(engine)

            observations = (
                capture_legacy_lifecycle_observations(
                    candidates
                )
            )

            with state_env(root):
                artifact = _build_artifact(
                    candidates=candidates,
                    observations=observations,
                    eligible=candidates,
                    live_gate_active=True,
                )

        self.assertIs(
            artifact["live_lifecycle_gate_active"],
            True,
        )

        row = artifact["candidates"][1]

        self.assertEqual(
            row["cognitive"][
                "counterfactual_gate_decision"
            ],
            "would_suppress",
        )
        self.assertEqual(
            row["cognitive"][
                "effective_live_gate_decision"
            ],
            "would_suppress",
        )


class AutoResolveOutcomeMetricTests(_EngineBoundTestCase):
    """Review §11: the legacy outcome-change metric is explicit."""

    _CREATED = {
        "applicable": True,
        "skip_reason": None,
        "current_resolved": False,
        "auto_resolve_candidate": True,
        "score_current": 0.2,
        "score_if_auto_resolve_succeeds": 1.0,
        "threshold": 0.3,
        "archive_candidate_current": False,
        "archive_candidate_after_auto_resolve_if_successful":
            True,
        "nominal_archive_candidate": False,
    }

    _REMOVED = {
        "applicable": True,
        "skip_reason": None,
        "current_resolved": False,
        "auto_resolve_candidate": True,
        "score_current": 0.2,
        "score_if_auto_resolve_succeeds": 1.0,
        "threshold": 0.3,
        "archive_candidate_current": True,
        "archive_candidate_after_auto_resolve_if_successful":
            False,
        "nominal_archive_candidate": False,
    }

    def test_created_direction_counted(self):
        candidates = [candidate(_A)]

        engine = SpyDecayEngine([dict(self._CREATED)])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        artifact = _build_artifact(
            candidates=candidates,
            observations=observations,
            eligible=candidates,
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics[
                "legacy_auto_resolve_changed_archive_outcome_count"
            ],
            1,
        )
        self.assertEqual(
            metrics[
                "legacy_auto_resolve_created_archive_candidate_count"
            ],
            1,
        )
        self.assertEqual(
            metrics[
                "legacy_auto_resolve_removed_archive_candidate_count"
            ],
            0,
        )

    def test_removed_direction_counted(self):
        candidates = [candidate(_A)]

        engine = SpyDecayEngine([dict(self._REMOVED)])
        bind_legacy_lifecycle_comparison_engine(engine)

        observations = (
            capture_legacy_lifecycle_observations(
                candidates
            )
        )

        artifact = _build_artifact(
            candidates=candidates,
            observations=observations,
            eligible=candidates,
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics[
                "legacy_auto_resolve_changed_archive_outcome_count"
            ],
            1,
        )
        self.assertEqual(
            metrics[
                "legacy_auto_resolve_created_archive_candidate_count"
            ],
            0,
        )
        self.assertEqual(
            metrics[
                "legacy_auto_resolve_removed_archive_candidate_count"
            ],
            1,
        )

    def test_reporter_matches_builder_metric(self):
        row = _report_row(
            "legacy_keep_cognitive_keep"
        )
        row["legacy"]["auto_resolve_candidate"] = True
        row["legacy"][
            "archive_candidate_current"
        ] = False
        row["legacy"][
            "archive_candidate_after_auto_resolve_if_successful"
        ] = True

        with tempfile.TemporaryDirectory() as root:
            _write_artifact(
                root,
                CID,
                RID,
                _artifact([row]),
            )

            report = reporter.build_report(root=root)

        self.assertEqual(
            report[
                "legacy_auto_resolve_changed_archive_outcome_count"
            ],
            1,
        )
        self.assertEqual(
            report[
                "legacy_auto_resolve_created_archive_candidate_count"
            ],
            1,
        )


class InterleavingRaceTests(
    unittest.IsolatedAsyncioTestCase,
):
    """Review §12: a genuinely interleaved same-conversation race.

    Request A captures its candidates + Legacy preview evidence, then
    pauses at a deterministic point. Request B runs to completion and
    advances the conversation-scoped latest artifacts to B. A then
    resumes. A's artifact must contain A evidence only.
    """

    def tearDown(self):
        bind_legacy_lifecycle_comparison_engine(None)

    def _snap(self, memories):
        observations = (
            capture_legacy_lifecycle_observations(
                memories
            )
        )

        return build_memory_shadow_snapshot(
            conversation_id=CID,
            unified={
                "version":
                    "unified-context-candidate.v1",
                "conversation_id": CID,
                "revision": 5,
                "sections": {"memories": memories},
                "telemetry": {
                    "current_user_excluded": True,
                    "memory_count": len(memories),
                },
            },
            retrieval_memories=memories,
            legacy_lifecycle_observations=observations,
        )

    async def _artifacts(self, root):
        directory = (
            Path(root)
            / "memory_lifecycle_comparison"
            / CID
        )

        found = []

        for path in sorted(directory.glob("*.json")):
            found.append(
                json.loads(
                    path.read_text(encoding="utf-8")
                )
            )

        return found

    async def test_same_conversation_interleaving_is_request_local(
        self,
    ):
        a_mem = [memory(_A)]
        b_mem = [memory(_B)]

        engine = SpyDecayEngine(
            [_preview(), _preview()]
        )
        bind_legacy_lifecycle_comparison_engine(engine)

        a_ready = asyncio.Event()
        a_release = asyncio.Event()
        calls = {"n": 0}

        async def controlled_runtime(conversation_id):
            calls["n"] += 1

            if calls["n"] == 1:
                # Request A has captured its own candidates and Legacy
                # preview evidence; pause here.
                snapshot = self._snap(a_mem)
                a_ready.set()
                await a_release.wait()

                return {
                    "stored": True,
                    "revision": 5,
                    "memory_shadow_snapshot": snapshot,
                }

            return {
                "stored": True,
                "revision": 5,
                "memory_shadow_snapshot": self._snap(
                    b_mem
                ),
            }

        with tempfile.TemporaryDirectory() as root:
            environ = {
                **_isolated_env(),
                _CONFIDENCE_ENV: "1",
                _FLASH_ENV: "0",
                COMPARISON_SHADOW_ENV: "1",
                "OMBRE_CONTEXT_STATE_DIR": root,
            }

            with patch.dict(
                os.environ, environ, clear=False
            ):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            coordinator,
                            "observe_context_sources",
                            Mock(return_value=CID),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            coordinator,
                            "update_unified_context_"
                            "candidate_from_runtime",
                            controlled_runtime,
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            coordinator,
                            "update_context_confidence_gate",
                            Mock(
                                return_value=_confidence()
                            ),
                        )
                    )

                    body = b'{"messages":[]}'

                    task_a = asyncio.create_task(
                        coordinator.run_context_pipeline(
                            body
                        )
                    )

                    # A is now deterministically paused.
                    await asyncio.wait_for(
                        a_ready.wait(), timeout=5
                    )

                    # Request B completes fully and advances the
                    # conversation-scoped latest artifacts to B.
                    task_b = asyncio.create_task(
                        coordinator.run_context_pipeline(
                            body
                        )
                    )

                    await asyncio.wait_for(
                        task_b, timeout=5
                    )

                    # Simulate B having advanced the shared,
                    # conversation-scoped "latest" artifacts to B.
                    policy_dir = (
                        Path(root)
                        / "memory_surfacing_policy"
                    )
                    policy_dir.mkdir(
                        parents=True, exist_ok=True
                    )
                    (
                        policy_dir / (CID + ".json")
                    ).write_text(
                        json.dumps(
                            eligible_policy(b_mem)
                        ),
                        encoding="utf-8",
                    )

                    unified_dir = (
                        Path(root)
                        / "unified_context_candidate"
                    )
                    unified_dir.mkdir(
                        parents=True, exist_ok=True
                    )
                    (
                        unified_dir / (CID + ".json")
                    ).write_text(
                        json.dumps(
                            {
                                "version":
                                    "unified-context-"
                                    "candidate.v1",
                                "conversation_id": CID,
                                "revision": 5,
                                "sections": {
                                    "memories": b_mem
                                },
                            }
                        ),
                        encoding="utf-8",
                    )

                    # A resumes after B has already written its state.
                    a_release.set()

                    await asyncio.wait_for(
                        task_a, timeout=5
                    )

            artifacts = await self._artifacts(root)

        self.assertEqual(len(artifacts), 2)

        fingerprint_a = candidate_fingerprint(
            {"id": _A, "content": a_mem[0]["content"]}
        )
        fingerprint_b = candidate_fingerprint(
            {"id": _B, "content": b_mem[0]["content"]}
        )

        by_fingerprint = {}

        for artifact in artifacts:
            keys = {
                row["candidate_fingerprint"]
                for row in artifact["candidates"]
            }
            by_fingerprint[frozenset(keys)] = artifact

        # Each request wrote exactly its own evidence, no bleed-over.
        self.assertIn(
            frozenset({fingerprint_a}),
            by_fingerprint,
        )
        self.assertIn(
            frozenset({fingerprint_b}),
            by_fingerprint,
        )

        for artifact in artifacts:
            keys = {
                row["memory_key"]
                for row in artifact["candidates"]
            }

            if fingerprint_a in {
                row["candidate_fingerprint"]
                for row in artifact["candidates"]
            }:
                self.assertEqual(
                    keys, {memory_key(_A)}
                )
            else:
                self.assertEqual(
                    keys, {memory_key(_B)}
                )


if __name__ == "__main__":
    unittest.main()