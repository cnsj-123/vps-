from __future__ import annotations

import ast
import contextlib
import json
import os
import tempfile
import unittest
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from unittest.mock import (
    AsyncMock,
    Mock,
    patch,
)

from _lifecycle_fixtures import (
    CID,
    LIFECYCLE_ENV,
    RID,
    T0,
    hours,
    iso,
    lifecycle_env,
    tree,
    usage_artifact,
    write_usage,
)

from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)
from ombrebrain.context.memory_lifecycle_event import (
    lifecycle_events_dir,
    memory_key,
    read_lifecycle_events,
    record_lifecycle_event_from_usage,
)
from ombrebrain.context.memory_lifecycle_state import (
    accessibility_for_age,
    build_lifecycle_state,
    lifecycle_state_path,
    persist_lifecycle_state,
    resolve_lifecycle_config,
    salience_for_age,
    strength_for_use_count,
)
from ombrebrain.context.memory_surfacing_lifecycle import (
    INTEGRATION_ENV,
    MIN_ACCESSIBILITY_ENV,
    MIN_SALIENCE_ENV,
    SURFACING_LIFECYCLE_VERSION,
    apply_lifecycle_surfacing_gate,
    resolve_surfacing_lifecycle_config,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]

_MODULE_SOURCE = (
    _REPO_ROOT
    / "src"
    / "ombrebrain"
    / "context"
    / "memory_surfacing_lifecycle.py"
)

# The base candidate ids / contents used by these tests.
A = "mem-alpha"
B = "mem-beta"
C = "mem-gamma"

_RECALL_COUNTER = {"n": 0}


def _next_recall_id() -> str:
    _RECALL_COUNTER["n"] += 1

    return "recall_" + format(
        _RECALL_COUNTER["n"], "032x"
    )


def record_use(
    root,
    memory_id,
    *,
    used_at,
) -> None:
    """Persist one real ``used`` lifecycle event for a memory.

    Every event goes through the authorized usage-derived write path,
    so this is the same immutable receipt production would create.
    """

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


def add_corrupt_event(
    root,
    memory_id,
) -> None:
    with state_env(root):
        directory = lifecycle_events_dir(
            memory_key(memory_id)
        )

        assert directory is not None

        directory.mkdir(
            parents=True, exist_ok=True
        )

        (directory / "corrupt.json").write_text(
            json.dumps({"version": "not-an-event"}),
            encoding="utf-8",
        )


def write_state_snapshot(
    root,
    memory_id,
    payload,
) -> None:
    with state_env(root):
        path = lifecycle_state_path(
            memory_key(memory_id)
        )

        assert path is not None

        path.parent.mkdir(
            parents=True, exist_ok=True
        )

        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


@contextlib.contextmanager
def state_env(root):
    """Bind only the Context state dir (no lifecycle flags)."""

    with patch.dict(
        os.environ,
        {"OMBRE_CONTEXT_STATE_DIR": str(root)},
        clear=False,
    ):
        yield


@contextlib.contextmanager
def gate_env(
    root,
    *,
    integration=True,
    shadow=True,
    **extra,
):
    env = {
        "OMBRE_CONTEXT_STATE_DIR": str(root),
        INTEGRATION_ENV: (
            "1" if integration else "0"
        ),
        LIFECYCLE_ENV: (
            "1" if shadow else "0"
        ),
    }

    env.update(extra)

    with patch.dict(os.environ, env, clear=False):
        yield


def candidate(memory_id, content=None):
    return {
        "id": memory_id,
        "content": (
            content
            if content is not None
            else ("cue for " + memory_id)
        ),
        "context_relevance": 0.9,
        "metadata": {
            "last_active": "2020-01-01T00:00:00Z",
        },
    }


def base_policy(
    eligible,
    *,
    ineligible=None,
    decision="allow_shadow",
    reason="surfacing_eligible",
):
    return {
        "version": "memory-surfacing-policy.v1",
        "mode": "shadow_only",
        "conversation_id": CID,
        "source_unified_revision": 5,
        "source_confidence_revision": 1,
        "confidence_decision": "allow_shadow",
        "confidence_reason": None,
        "confidence_binding": None,
        "decision": decision,
        "reason": reason,
        "reasons": [reason],
        "retrieved_candidate_count": (
            len(eligible)
            + len(ineligible or [])
        ),
        "eligible_candidate_count": len(eligible),
        "evidence_count": len(eligible),
        "eligible": list(eligible),
        "ineligible": list(ineligible or []),
    }


def eligible_ids(report):
    return [
        item["id"] for item in report["eligible"]
    ]


class ConfigTests(unittest.TestCase):

    def test_defaults(self):
        with patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop(
                MIN_ACCESSIBILITY_ENV, None
            )
            os.environ.pop(
                MIN_SALIENCE_ENV, None
            )

            config = (
                resolve_surfacing_lifecycle_config()
            )

        self.assertAlmostEqual(
            config["min_accessibility"], 0.30
        )
        self.assertAlmostEqual(
            config["min_salience"], 0.25
        )

    def test_invalid_falls_back_to_default(self):
        for bad in (
            "nan",
            "inf",
            "-inf",
            "abc",
            "",
            "1e999",
        ):
            with self.subTest(bad=bad):
                with patch.dict(
                    os.environ,
                    {
                        MIN_ACCESSIBILITY_ENV: bad,
                        MIN_SALIENCE_ENV: bad,
                    },
                    clear=False,
                ):
                    config = (
                        resolve_surfacing_lifecycle_config()
                    )

                self.assertAlmostEqual(
                    config["min_accessibility"],
                    0.30,
                )
                self.assertAlmostEqual(
                    config["min_salience"], 0.25
                )

    def test_out_of_range_is_clamped(self):
        for raw, expected in (
            ("-5", 0.0),
            ("0", 0.0),
            ("2.5", 1.0),
            ("1", 1.0),
        ):
            with self.subTest(raw=raw):
                with patch.dict(
                    os.environ,
                    {
                        MIN_ACCESSIBILITY_ENV: raw,
                        MIN_SALIENCE_ENV: raw,
                    },
                    clear=False,
                ):
                    config = (
                        resolve_surfacing_lifecycle_config()
                    )

                self.assertAlmostEqual(
                    config["min_accessibility"],
                    expected,
                )
                self.assertAlmostEqual(
                    config["min_salience"],
                    expected,
                )


class FlagTests(unittest.TestCase):

    def test_integration_off_is_exact_base_policy(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            # A dormant supplementary memory that WOULD be suppressed.
            record_use(
                root,
                B,
                used_at=T0 - hours(500),
            )

            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            with gate_env(
                root,
                integration=False,
                shadow=True,
            ):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(output, policy)
        self.assertNotIn(
            "lifecycle_surface", output
        )
        self.assertIsNot(output, policy)

    def test_lifecycle_shadow_off_is_neutral(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            record_use(
                root,
                B,
                used_at=T0 - hours(500),
            )

            policy = base_policy(
                [candidate(A), candidate(B), candidate(C)]
            )

            with gate_env(
                root,
                integration=True,
                shadow=False,
            ):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(
            eligible_ids(output), [A, B, C]
        )

        summary = output["lifecycle_surface"]

        self.assertEqual(
            summary["decision"], "neutral"
        )
        self.assertEqual(
            summary["reason"], "lifecycle_disabled"
        )
        self.assertEqual(
            summary["blocked_count"], 0
        )
        self.assertEqual(
            output["decision"], "allow_shadow"
        )
        self.assertEqual(
            output["reason"], "surfacing_eligible"
        )

    def test_base_no_surface_is_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(
                root,
                B,
                used_at=T0,
            )

            policy = base_policy(
                [],
                ineligible=[
                    {
                        "memory_id": B,
                        "reason":
                            "anti_echo_recent_24h",
                    }
                ],
                decision="no_surface",
                reason="anti_echo_recent_24h",
            )

            with gate_env(
                root,
                integration=True,
                shadow=True,
            ):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(output, policy)
        self.assertNotIn(
            "lifecycle_surface", output
        )

    def test_invalid_as_of_is_neutral(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(
                root,
                B,
                used_at=T0 - hours(500),
            )

            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            for bad in (
                datetime(2026, 1, 1),
                "not-a-time",
                123,
            ):
                with self.subTest(bad=bad):
                    with gate_env(root):
                        output = (
                            apply_lifecycle_surfacing_gate(
                                policy, as_of=bad
                            )
                        )

                    self.assertEqual(output, policy)

    def test_garbage_policy_is_returned_as_is(self):
        with gate_env("/tmp"):
            for value in (None, 1, "x", []):
                with self.subTest(value=value):
                    self.assertEqual(
                        apply_lifecycle_surfacing_gate(
                            value
                        ),
                        value,
                    )


class GateSemanticsTests(unittest.TestCase):

    def test_primary_always_retained(self):
        with tempfile.TemporaryDirectory() as root:
            # Dormant: accessibility near the floor, salience ~ 0.
            record_use(
                root,
                A,
                used_at=T0 - timedelta(days=500),
            )

            policy = base_policy([candidate(A)])

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(eligible_ids(output), [A])
        self.assertEqual(
            output["decision"], "allow_shadow"
        )
        self.assertEqual(
            output["lifecycle_surface"][
                "primary_preserved_count"
            ],
            1,
        )
        self.assertEqual(
            output["lifecycle_surface"][
                "evaluated_count"
            ],
            0,
        )
        self.assertEqual(
            output["lifecycle_surface"][
                "blocked_count"
            ],
            0,
        )

    def test_unobserved_supplementary_is_neutral(self):
        with tempfile.TemporaryDirectory() as root:
            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(
            eligible_ids(output), [A, B]
        )

        summary = output["lifecycle_surface"]

        self.assertEqual(
            summary["neutral_count"], 1
        )
        self.assertEqual(
            summary["observed_count"], 0
        )
        self.assertEqual(
            summary["blocked_count"], 0
        )

    def test_recent_used_supplementary_is_retained(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            record_use(root, B, used_at=T0)

            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(
            eligible_ids(output), [A, B]
        )
        self.assertEqual(
            output["lifecycle_surface"][
                "passed_count"
            ],
            1,
        )

    def test_salience_rescues_a_fading_memory(self):
        with tempfile.TemporaryDirectory() as root:
            age = 48.0

            record_use(
                root,
                B,
                used_at=T0 - hours(age),
            )

            config = resolve_lifecycle_config()

            strength = strength_for_use_count(
                1, config=config
            )

            accessibility = accessibility_for_age(
                strength, age, config=config
            )

            salience = salience_for_age(
                use_count=1,
                age_since_last_used_hours=age,
                config=config,
            )

            # The premise: accessibility alone would block it...
            self.assertLess(
                accessibility, 0.30
            )
            # ...but salience keeps it.
            self.assertGreaterEqual(
                salience, 0.25
            )

            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(
            eligible_ids(output), [A, B]
        )
        self.assertEqual(
            output["lifecycle_surface"][
                "passed_count"
            ],
            1,
        )

    def test_dormant_supplementary_is_suppressed(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            age = 72.0

            record_use(
                root,
                B,
                used_at=T0 - hours(age),
            )

            config = resolve_lifecycle_config()

            strength = strength_for_use_count(
                1, config=config
            )

            accessibility = accessibility_for_age(
                strength, age, config=config
            )

            salience = salience_for_age(
                use_count=1,
                age_since_last_used_hours=age,
                config=config,
            )

            self.assertLess(accessibility, 0.30)
            self.assertLess(salience, 0.25)

            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(eligible_ids(output), [A])

        self.assertEqual(
            output["eligible_candidate_count"], 1
        )
        self.assertEqual(
            output["retrieved_candidate_count"], 2
        )
        self.assertEqual(
            output["eligible"],
            [policy["eligible"][0]],
        )
        self.assertIn(
            {
                "memory_id": B,
                "reason":
                    "lifecycle_below_surface_threshold",
            },
            output["ineligible"],
        )
        self.assertEqual(
            output["lifecycle_surface"][
                "blocked_count"
            ],
            1,
        )

    def test_reinforcement_extends_surfacing_time(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            age = 60.0

            # Same age, one vs two explicit uses.
            record_use(
                root,
                "mem-one",
                used_at=T0 - hours(age),
            )

            record_use(
                root,
                "mem-two",
                used_at=T0 - hours(age),
            )

            record_use(
                root,
                "mem-two",
                used_at=T0 - hours(age),
            )

            config = resolve_lifecycle_config()

            one = accessibility_for_age(
                strength_for_use_count(
                    1, config=config
                ),
                age,
                config=config,
            )

            two = accessibility_for_age(
                strength_for_use_count(
                    2, config=config
                ),
                age,
                config=config,
            )

            # The premise: only the reinforced trace clears the bar.
            self.assertLess(one, 0.30)
            self.assertGreaterEqual(two, 0.30)

            policy = base_policy(
                [
                    candidate(A),
                    candidate("mem-one"),
                    candidate("mem-two"),
                ]
            )

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(
            eligible_ids(output),
            [A, "mem-two"],
        )

    def test_multiple_candidate_primary_test(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(
                root,
                A,
                used_at=T0 - timedelta(days=400),
            )

            record_use(
                root,
                B,
                used_at=T0 - timedelta(days=400),
            )

            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(eligible_ids(output), [A])

    def test_order_is_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            # All recent & used, so none is suppressed.
            for memory_id in (B, C):
                record_use(
                    root, memory_id, used_at=T0
                )

            policy = base_policy(
                [
                    candidate(A),
                    candidate(B),
                    candidate(C),
                ]
            )

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(
            eligible_ids(output), [A, B, C]
        )
        self.assertEqual(
            output["eligible"],
            policy["eligible"],
        )

    def test_caller_policy_is_not_mutated(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(
                root,
                B,
                used_at=T0 - hours(500),
            )

            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            snapshot = json.dumps(policy)

            with gate_env(root):
                apply_lifecycle_surfacing_gate(
                    policy, as_of=T0
                )

        self.assertEqual(
            json.dumps(policy), snapshot
        )

    def test_summary_counts_are_consistent(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(root, B, used_at=T0)

            record_use(
                root,
                C,
                used_at=T0 - hours(500),
            )

            policy = base_policy(
                [candidate(A), candidate(B), candidate(C)]
            )

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        summary = output["lifecycle_surface"]

        self.assertEqual(
            summary["version"],
            SURFACING_LIFECYCLE_VERSION,
        )
        self.assertEqual(summary["mode"], "shadow_only")
        self.assertEqual(
            summary["decision"], "applied"
        )
        self.assertEqual(
            summary["observed_count"]
            + summary["neutral_count"],
            summary["evaluated_count"],
        )
        self.assertEqual(
            summary["passed_count"]
            + summary["blocked_count"],
            summary["observed_count"],
        )
        self.assertEqual(
            summary["evaluated_count"], 2
        )
        self.assertNotIn(
            "accessibility", summary
        )
        self.assertNotIn("salience", summary)
        self.assertNotIn("strength", summary)


class EventIntegrityTests(unittest.TestCase):

    def test_corrupt_event_is_not_a_use(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(root, B, used_at=T0)

            add_corrupt_event(root, B)

            with state_env(root):
                store = read_lifecycle_events(B)

            self.assertEqual(
                len(store["events"]), 1
            )
            self.assertEqual(
                store["invalid_event_count"], 1
            )

            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        # The valid used event decides; the corrupt file is ignored.
        self.assertEqual(
            eligible_ids(output), [A, B]
        )
        self.assertEqual(
            output["lifecycle_surface"][
                "observed_count"
            ],
            1,
        )
        self.assertEqual(
            output["lifecycle_surface"][
                "invalid_event_count"
            ],
            1,
        )

    def test_only_corrupt_is_neutral(self):
        with tempfile.TemporaryDirectory() as root:
            add_corrupt_event(root, B)

            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            with gate_env(root):
                output = (
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )
                )

        self.assertEqual(
            eligible_ids(output), [A, B]
        )

        summary = output["lifecycle_surface"]

        self.assertEqual(
            summary["neutral_count"], 1
        )
        self.assertEqual(
            summary["observed_count"], 0
        )
        self.assertEqual(
            summary["blocked_count"], 0
        )
        self.assertEqual(
            summary["invalid_event_count"], 1
        )


class StateCacheIndependenceTests(unittest.TestCase):

    def _decide(self, root):
        policy = base_policy(
            [candidate(A), candidate(B)]
        )

        with gate_env(root):
            output = apply_lifecycle_surfacing_gate(
                policy, as_of=T0
            )

        return eligible_ids(output), output

    def test_state_snapshot_is_never_authority(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            record_use(root, B, used_at=T0)

            # No snapshot at all.
            ids_none, _ = self._decide(root)

            with state_env(root):
                # A VALID but stale snapshot claiming a far-future
                # derivation (accessibility ~ floor).
                store = read_lifecycle_events(B)

                state = build_lifecycle_state(
                    memory_id=B,
                    events=store["events"],
                    as_of=T0 + timedelta(days=365),
                    config=resolve_lifecycle_config(),
                    exists=True,
                    source_exists_checked=True,
                    source_created_at=None,
                    event_file_count=(
                        store["event_file_count"]
                    ),
                    invalid_event_count=(
                        store["invalid_event_count"]
                    ),
                )

                persist_lifecycle_state(state)

            ids_valid, _ = self._decide(root)

            # A corrupt snapshot claiming high values.
            write_state_snapshot(
                root,
                B,
                {
                    "version":
                        "memory-lifecycle-state.v1",
                    "mode": "shadow_only",
                    "memory_id": B,
                    "memory_key": memory_key(B),
                    "accessibility": 0.99,
                    "salience": 0.99,
                },
            )

            ids_corrupt, _ = self._decide(root)

            # Snapshot deleted entirely.
            with state_env(root):
                path = lifecycle_state_path(
                    memory_key(B)
                )

                path.unlink()

            ids_deleted, _ = self._decide(root)

        self.assertEqual(
            ids_none, [A, B]
        )
        self.assertEqual(ids_valid, ids_none)
        self.assertEqual(ids_corrupt, ids_none)
        self.assertEqual(ids_deleted, ids_none)


class ReadOnlyTests(unittest.TestCase):

    def test_gate_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(root, B, used_at=T0)

            before = tree(root)

            with state_env(root):
                store_before = read_lifecycle_events(B)

            policy = base_policy(
                [candidate(A), candidate(B)]
            )

            with gate_env(root):
                apply_lifecycle_surfacing_gate(
                    policy, as_of=T0
                )

            after = tree(root)

            with state_env(root):
                store_after = read_lifecycle_events(B)

        self.assertEqual(after, before)
        self.assertEqual(
            store_after, store_before
        )

    def test_no_reinforcement_or_scheduler(self):
        source = _MODULE_SOURCE.read_text(
            encoding="utf-8"
        )

        for forbidden in (
            "record_lifecycle_event_from_usage",
            "observe_memory_usage_lifecycle",
            "persist_lifecycle_state",
            "BucketManager",
            "memory_lifecycle_events",
            "sorted(",
            ".sort(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(
                    forbidden, source
                )

        tree_ast = ast.parse(source)

        forbidden_calls = {
            "touch",
            "archive",
            "update",
            "delete",
            "create",
            "sort",
        }

        for node in ast.walk(tree_ast):
            if (
                isinstance(node, ast.Call)
                and isinstance(
                    node.func, ast.Attribute
                )
            ):
                self.assertNotIn(
                    node.func.attr,
                    forbidden_calls,
                )

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

                for forbidden in (
                    "bucket_manager",
                    "embedding",
                    "retrieval",
                    "openai",
                    "anthropic",
                    "asyncio",
                    "threading",
                    "decay_engine",
                ):
                    self.assertNotIn(
                        forbidden, lowered
                    )


class LoggingTests(unittest.TestCase):

    def test_log_is_privacy_safe(self):
        with tempfile.TemporaryDirectory() as root:
            record_use(root, B, used_at=T0)

            policy = base_policy(
                [
                    candidate(
                        A, "SECRET_ALPHA_TEXT"
                    ),
                    candidate(
                        B, "SECRET_BETA_TEXT"
                    ),
                ]
            )

            with gate_env(root):
                with self.assertLogs(
                    "ombre_brain.gateway",
                    level="INFO",
                ) as captured:
                    apply_lifecycle_surfacing_gate(
                        policy, as_of=T0
                    )

        logged = "\n".join(
            record.getMessage()
            for record in captured.records
        )

        self.assertIn(
            "[gateway.context_memory_surfacing_lifecycle]",
            logged,
        )

        for secret in (
            A,
            B,
            "SECRET_ALPHA_TEXT",
            "SECRET_BETA_TEXT",
            CID,
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, logged)


CID_PIPE = "ctx_0123456789abcdef"

_FLASH_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW"
)
_LEDGER_ENV = (
    "OMBRE_GATEWAY_CONTEXT_EXPOSURE_LEDGER_SHADOW"
)
_SURFACE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW"
)
_CONFIDENCE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW"
)

_ALL_FLAGS = (
    "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION",
    _CONFIDENCE_ENV,
    _FLASH_ENV,
    _LEDGER_ENV,
    _SURFACE_ENV,
    INTEGRATION_ENV,
    LIFECYCLE_ENV,
)

_ISOLATED_ENV = {
    flag: "0" for flag in _ALL_FLAGS
}


def _old_ts():
    return (
        datetime.now(timezone.utc)
        - timedelta(days=5)
    ).isoformat()


def _memory(memory_id, name=None):
    item = {
        "id": memory_id,
        "content": "cue for " + memory_id,
        "context_relevance": 0.9,
        "metadata": {
            "last_active": _old_ts(),
        },
    }

    if name is not None:
        item["metadata"]["name"] = name

    return item


def _allow_confidence():
    return {
        "version": "context-confidence-gate.v1",
        "mode": "shadow_only",
        "conversation_id": CID_PIPE,
        "decision": "allow_shadow",
        "allowed": True,
        "reason": None,
        "reasons": [],
        "stored": True,
        "duplicate": False,
        "revision": 1,
        "source_unified_revision": 5,
    }


def _snapshot(memories, *, revision=5):
    from ombrebrain.context.retrieval_decision_shadow import (
        build_candidate_evidence,
    )

    return {
        "version": "memory-shadow-snapshot.v1",
        "mode": "shadow_only",
        "conversation_id": CID_PIPE,
        "revision": revision,
        "unified": {
            "version":
                "unified-context-candidate.v1",
            "conversation_id": CID_PIPE,
            "revision": revision,
            "sections": {
                "memories": memories,
            },
            "telemetry": {
                "current_user_excluded": True,
                "memory_count": len(memories),
            },
        },
        "evidence": build_candidate_evidence(
            memories
        ),
    }


class PipelineRegressionTests(
    unittest.IsolatedAsyncioTestCase,
):

    async def _run(
        self,
        *,
        root,
        integration_on,
        lifecycle_on,
        memories,
    ):
        body = b'{"messages":[]}'

        environ = {
            **_ISOLATED_ENV,
            _CONFIDENCE_ENV: "1",
            _FLASH_ENV: "1",
            _LEDGER_ENV: "1",
            _SURFACE_ENV: "1",
            INTEGRATION_ENV: (
                "1" if integration_on else "0"
            ),
            LIFECYCLE_ENV: (
                "1" if lifecycle_on else "0"
            ),
            "OMBRE_CONTEXT_STATE_DIR": root,
        }

        with patch.dict(
            os.environ, environ, clear=False
        ):
            with patch.object(
                coordinator,
                "observe_context_sources",
                Mock(return_value=CID_PIPE),
            ), patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                AsyncMock(
                    return_value={
                        "stored": True,
                        "revision": 5,
                        "memory_shadow_snapshot":
                            _snapshot(memories),
                    }
                ),
            ), patch.object(
                coordinator,
                "update_context_confidence_gate",
                Mock(return_value=_allow_confidence()),
            ), patch.object(
                coordinator,
                "update_context_injection_preview",
                Mock(
                    return_value={
                        "stored": True,
                        "revision": 1,
                        "source_revision": 5,
                    }
                ),
            ), patch.object(
                coordinator,
                "update_context_injection_gate",
                Mock(
                    return_value={
                        "stored": True,
                        "revision": 1,
                        "decision": "allow_shadow",
                        "allowed": True,
                    }
                ),
            ):
                selected = await (
                    coordinator.run_context_pipeline(body)
                )

        self.assertIs(selected, body)

        return self._artifacts(root)

    def _artifacts(self, root):
        flash_dir = (
            Path(root)
            / "memory_flash"
            / CID_PIPE
        )

        ledger_dir = (
            Path(root)
            / "exposure_ledger"
            / CID_PIPE
        )

        surface_dir = (
            Path(root)
            / "recall_surface"
            / CID_PIPE
        )

        flash = json.loads(
            sorted(flash_dir.iterdir())[0].read_text(
                encoding="utf-8"
            )
        )

        ledger = json.loads(
            sorted(ledger_dir.iterdir())[0].read_text(
                encoding="utf-8"
            )
        )

        surface = None

        if surface_dir.is_dir():
            files = sorted(
                surface_dir.iterdir()
            )

            if files:
                surface = json.loads(
                    files[0].read_text(
                        encoding="utf-8"
                    )
                )

        return flash, ledger, surface

    async def test_integration_off_matches_baseline(
        self,
    ):
        memories = [_memory(A), _memory(B)]

        with tempfile.TemporaryDirectory() as root_off:
            flash_off, ledger_off, _ = (
                await self._run(
                    root=root_off,
                    integration_on=False,
                    lifecycle_on=False,
                    memories=memories,
                )
            )

        with tempfile.TemporaryDirectory() as root_on:
            flash_on, ledger_on, _ = (
                await self._run(
                    root=root_on,
                    integration_on=True,
                    lifecycle_on=False,
                    memories=memories,
                )
            )

        for artifact_off, artifact_on in (
            (flash_off, flash_on),
            (ledger_off, ledger_on),
        ):
            for field in (
                "decision",
                "reason",
                "retrieved_candidate_count",
                "eligible_candidate_count",
                "surfaced_count",
                "estimated_tokens",
                "token_budget",
            ):
                with self.subTest(field=field):
                    self.assertEqual(
                        artifact_off.get(field),
                        artifact_on.get(field),
                    )

        self.assertEqual(
            [
                item["memory_id"]
                for item in flash_off["flashes"]
            ],
            [
                item["memory_id"]
                for item in flash_on["flashes"]
            ],
        )
        self.assertEqual(
            flash_off["surfaced_count"], 2
        )

    async def test_integration_off_never_calls_gate(
        self,
    ):
        memories = [_memory(A), _memory(B)]

        with tempfile.TemporaryDirectory() as root:
            environ = {
                **_ISOLATED_ENV,
                _CONFIDENCE_ENV: "1",
                _FLASH_ENV: "1",
                _LEDGER_ENV: "1",
                "OMBRE_CONTEXT_STATE_DIR": root,
            }

            gate = Mock()

            with patch.dict(
                os.environ, environ, clear=False
            ):
                with patch.object(
                    coordinator,
                    "observe_context_sources",
                    Mock(return_value=CID_PIPE),
                ), patch.object(
                    coordinator,
                    "update_unified_context_candidate_from_runtime",
                    AsyncMock(
                        return_value={
                            "stored": True,
                            "revision": 5,
                            "memory_shadow_snapshot":
                                _snapshot(memories),
                        }
                    ),
                ), patch.object(
                    coordinator,
                    "update_context_confidence_gate",
                    Mock(
                        return_value=(
                            _allow_confidence()
                        )
                    ),
                ), patch.object(
                    coordinator,
                    "apply_lifecycle_surfacing_gate",
                    gate,
                ):
                    await coordinator.run_context_pipeline(
                        b'{"messages":[]}'
                    )

        gate.assert_not_called()

    async def test_dormant_supplementary_not_surfaced(
        self,
    ):
        memories = [_memory(A), _memory(B)]

        with tempfile.TemporaryDirectory() as root:
            # B was explicitly used, but a long time ago.
            record_use(
                root,
                B,
                used_at=(
                    datetime.now(timezone.utc)
                    - timedelta(hours=96)
                ),
            )

            flash, ledger, surface = (
                await self._run(
                    root=root,
                    integration_on=True,
                    lifecycle_on=True,
                    memories=memories,
                )
            )

        surfaced_ids = [
            item["memory_id"]
            for item in flash["flashes"]
        ]

        self.assertEqual(surfaced_ids, [A])
        self.assertEqual(
            flash["surfaced_count"], 1
        )
        self.assertEqual(
            flash["retrieved_candidate_count"], 2
        )

        # retrieved != surfaced: the ledger still records B as
        # retrieved, but never as surfaced.
        self.assertEqual(
            ledger["retrieved_count"], 2
        )
        self.assertEqual(
            ledger["surfaced_count"], 1
        )
        self.assertIn(
            B, ledger["retrieved_memory_ids"]
        )
        self.assertNotIn(
            B,
            [
                item["memory_id"]
                for item in flash["flashes"]
            ],
        )

        # Recall Surface is built from the real Flash only.
        self.assertIsNotNone(surface)
        self.assertEqual(
            surface["memory_count"], 1
        )
        self.assertEqual(
            set(surface["mapping"].values()),
            {A},
        )
        self.assertNotIn(B, surface["mapping"].values())


if __name__ == "__main__":
    unittest.main()
