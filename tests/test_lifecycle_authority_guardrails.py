from __future__ import annotations

import ast
import unittest
from pathlib import Path


# Guardrail tests for the Memory Lifecycle Authority Map
# (docs/cognitive-memory/LIFECYCLE_AUTHORITY_MAP.md).
#
# These freeze the CURRENT authority separation between legacy OB
# (activation / decay / archive) and the new cognitive lifecycle
# (used / reinforcement / strength-accessibility-salience). They are
# deliberately source-contract / behavioural-spy tests: a regression
# that starts bridging the two planes must fail here instead of
# silently changing live behaviour.
#
# They assert separation ONLY. They do not assert that either plane
# works, and they do not change any default.

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"

# Legacy mutation entry points on BucketManager.
_LEGACY_MUTATION_CALLS = {
    "touch",
    "touch_many",
    "archive",
    "delete",
    "hard_delete_test_bucket",
}

# New-lifecycle "used" / reinforcement writers. `used` is evidence, and
# reinforcement is derived from it -- never from retrieval / surfacing.
_LIFECYCLE_EVIDENCE_CALLS = {
    "record_usage",
    "record_lifecycle_event_from_usage",
    "observe_memory_usage_lifecycle",
    "process_completed_usage_for_lifecycle",
    "derive_memory_lifecycle_state",
}

_NEW_LIFECYCLE_MODULES = (
    "src/ombrebrain/context/memory_lifecycle_event.py",
    "src/ombrebrain/context/memory_lifecycle_state.py",
    "src/ombrebrain/context/memory_reinforcement.py",
    "src/ombrebrain/context/memory_surfacing_lifecycle.py",
    "src/ombrebrain/context/memory_usage_signal.py",
    "src/ombrebrain/context/live_recall_usage_attribution.py",
    "src/ombrebrain/context/live_recall_usage_surface.py",
    "src/ombrebrain/context/live_recall_usage_ref_receipt.py",
    "src/ombrebrain/context/live_recall_usage_lifecycle_integration.py",
)

# The exact legacy-decay lazy-start path documented in §9.2 of the map.
# Under stdio these are the ONLY way DecayEngine can start.
_DOCUMENTED_LAZY_START_FILES = {
    "src/tools/breath/__init__.py",
    "src/tools/dream/__init__.py",
    "src/tools/hold/__init__.py",
    "src/tools/grow/__init__.py",
    "src/tools/anchor/core.py",
    "src/tools/plan/core.py",
    "src/tools/i/core.py",
}

# The two lifecycle architecture documents. They must not carry
# machine-local file:///workspace links (broken on GitHub).
_ARCHITECTURE_DOCS = (
    "docs/cognitive-memory/LIFECYCLE_AUTHORITY_MAP.md",
    "docs/cognitive-memory/ARCHITECTURE_STATUS.md",
)


def _parse(relative: str) -> ast.Module:
    path = _REPO_ROOT / relative
    return ast.parse(path.read_text(encoding="utf-8"))


def _calls(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(
            node.func, ast.Attribute
        ):
            yield node


def _imported_modules(tree: ast.Module) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return modules


def _call_attr_names(tree: ast.Module) -> set[str]:
    return {node.func.attr for node in _calls(tree)}


def _decay_engine_startup_target(node: ast.AST) -> str | None:
    """Return the called attr name when *node* is a target-aware DecayEngine
    startup call, i.e. ``<something>.decay_engine.<attr>(...)``.

    The receiver must be an attribute expression whose own attribute name is
    ``decay_engine`` (e.g. ``rt.decay_engine``). This deliberately does NOT
    match unrelated ``foo.ensure_started()`` calls, ``self.start()``, or the
    RuntimeLifecycle dynamic ``getattr(self.decay_engine, "start", None)``
    path (its ``func`` is a ``Name``, not an ``Attribute``).
    """

    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    receiver = func.value
    if not (
        isinstance(receiver, ast.Attribute)
        and receiver.attr == "decay_engine"
    ):
        return None
    return func.attr


def _class_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(
                        item, (ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    and item.name == method_name
                ):
                    return item
    raise AssertionError(
        f"{class_name}.{method_name} not found"
    )


def _run_async_step_wiring(
    method: ast.AST,
) -> dict[str, ast.AST | None]:
    """Map each ``_run_async_step`` label to its callback call node."""

    wiring: dict[str, ast.AST | None] = {}
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "_run_async_step"
        ):
            continue
        label = None
        if node.args and isinstance(node.args[0], ast.Constant):
            label = node.args[0].value
        callback = node.args[1] if len(node.args) > 1 else None
        wiring[label] = callback
    return wiring


def _is_decay_engine_getattr(node: ast.AST | None, attr: str) -> bool:
    """Match ``getattr(self.decay_engine, <attr>, None)``."""

    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not (isinstance(func, ast.Name) and func.id == "getattr"):
        return False
    if len(node.args) < 2:
        return False
    target = node.args[0]
    if not (
        isinstance(target, ast.Attribute)
        and target.attr == "decay_engine"
    ):
        return False
    return (
        isinstance(node.args[1], ast.Constant)
        and node.args[1].value == attr
    )


class RetrievalAndSurfacingAreReadOnlyTests(unittest.TestCase):
    """A / B / C / I: retrieval and surfacing never mutate or reinforce."""

    _RETRIEVAL_MODULES = (
        "src/ombrebrain/context/related_recall.py",
        "src/ombrebrain/context/memory_flash.py",
        "src/ombrebrain/context/memory_recall_surface.py",
        "src/ombrebrain/context/memory_flash_live_exposure.py",
        "src/ombrebrain/context/live_recall_authorization.py",
    )

    def test_A_gateway_context_retrieval_does_not_mutate(self):
        for relative in self._RETRIEVAL_MODULES:
            with self.subTest(relative=relative):
                attrs = _call_attr_names(_parse(relative))
                self.assertEqual(
                    attrs & _LEGACY_MUTATION_CALLS,
                    set(),
                    "retrieval/context module calls a legacy mutation",
                )

    def test_B_related_recall_does_not_mutate_on_load(self):
        attrs = _call_attr_names(
            _parse("src/ombrebrain/context/related_recall.py")
        )
        self.assertEqual(attrs & _LEGACY_MUTATION_CALLS, set())

    def test_C_flash_and_recall_do_not_write_used_or_reinforcement(self):
        for relative in self._RETRIEVAL_MODULES:
            with self.subTest(relative=relative):
                attrs = _call_attr_names(_parse(relative))
                self.assertEqual(
                    attrs & _LIFECYCLE_EVIDENCE_CALLS,
                    set(),
                    "retrieval/flash module writes usage or "
                    "reinforcement evidence",
                )

    def test_I_breath_search_and_surface_never_touch(self):
        for relative in (
            "src/tools/breath/search.py",
            "src/tools/breath/surface.py",
            "src/tools/breath/importance.py",
        ):
            with self.subTest(relative=relative):
                attrs = _call_attr_names(_parse(relative))
                self.assertEqual(
                    attrs & {"touch", "touch_many"}, set()
                )


class NewLifecycleNeverCallsLegacyTests(unittest.TestCase):
    """E / H: the cognitive plane never reaches into legacy mutation."""

    def test_E_new_lifecycle_modules_have_no_legacy_mutation(self):
        for relative in _NEW_LIFECYCLE_MODULES:
            with self.subTest(relative=relative):
                tree = _parse(relative)
                attrs = _call_attr_names(tree)
                self.assertEqual(
                    attrs
                    & {"touch", "touch_many", "archive", "delete",
                       "run_decay_cycle", "hard_delete_test_bucket"},
                    set(),
                )
                for node in _calls(tree):
                    if node.func.attr != "update":
                        continue
                    self.assertNotIn(
                        "bump_active",
                        {kw.arg for kw in node.keywords},
                        "lifecycle module uses update(bump_active=...)",
                    )

    def test_H_lifecycle_surfacing_gate_is_read_only(self):
        relative = (
            "src/ombrebrain/context/memory_surfacing_lifecycle.py"
        )
        tree = _parse(relative)
        attrs = _call_attr_names(tree)
        self.assertEqual(
            attrs
            & {"touch", "touch_many", "archive", "delete", "update"},
            set(),
        )
        for module in _imported_modules(tree):
            lowered = module.lower()
            self.assertNotIn("bucket_manager", lowered)
            self.assertNotIn("decay_engine", lowered)


class LegacyPlaneNeverReadsCognitiveStateTests(unittest.TestCase):
    """F / G: neither legacy entry point reaches into the new plane."""

    def test_F_trace_reinforce_cannot_create_used_or_receipt(self):
        for relative in (
            "src/tools/trace/_reinforce.py",
            "src/tools/trace/core.py",
        ):
            with self.subTest(relative=relative):
                tree = _parse(relative)
                self.assertEqual(
                    _call_attr_names(tree) & _LIFECYCLE_EVIDENCE_CALLS,
                    set(),
                )
                for module in _imported_modules(tree):
                    lowered = module.lower()
                    self.assertNotIn("memory_lifecycle", lowered)
                    self.assertNotIn("memory_reinforcement", lowered)
                    self.assertNotIn("memory_usage", lowered)
                    self.assertNotIn("live_recall", lowered)

    def test_F_reinforce_still_is_the_legacy_activation_action(self):
        tree = _parse("src/tools/trace/_reinforce.py")
        self.assertIn("touch", _call_attr_names(tree))

    def test_G_decay_engine_is_unaware_of_cognitive_state(self):
        relative = "src/decay_engine.py"
        tree = _parse(relative)

        for module in _imported_modules(tree):
            lowered = module.lower()
            self.assertNotIn("memory_lifecycle", lowered)
            self.assertNotIn("memory_reinforcement", lowered)
            self.assertNotIn("memory_usage", lowered)
            self.assertNotIn("live_recall", lowered)

        cognitive_identifiers: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                cognitive_identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                cognitive_identifiers.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(
                node.value, str
            ):
                cognitive_identifiers.add(node.value)

        self.assertEqual(
            cognitive_identifiers
            & {"strength", "accessibility", "salience"},
            set(),
        )


class UsageEvidenceCallSiteExclusivityTests(unittest.TestCase):
    """D: only the explicit UseMemory path may create used/reinforcement."""

    def _call_sites(self, name: str) -> set[str]:
        found: set[str] = set()
        for path in sorted(_SRC.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # Count both ``mod.name(...)`` and a bare imported
                # ``name(...)`` -- the latter is how the lifecycle
                # modules actually call each other.
                called = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else func.id
                    if isinstance(func, ast.Name)
                    else None
                )
                if called == name:
                    found.add(str(path.relative_to(_REPO_ROOT)))
        return found

    def test_record_usage_only_from_usage_attribution(self):
        self.assertEqual(
            self._call_sites("record_usage"),
            {
                "src/ombrebrain/context/"
                "live_recall_usage_attribution.py"
            },
        )

    def test_reinforcement_entrypoints_are_exclusive(self):
        self.assertEqual(
            self._call_sites("record_lifecycle_event_from_usage"),
            {
                "src/ombrebrain/context/"
                "memory_reinforcement.py"
            },
        )
        self.assertEqual(
            self._call_sites("observe_memory_usage_lifecycle"),
            {
                "src/ombrebrain/context/"
                "memory_reinforcement.py"
            },
        )
        self.assertEqual(
            self._call_sites(
                "process_completed_usage_for_lifecycle"
            ),
            {
                "src/ombrebrain/context/"
                "live_recall_usage_lifecycle_integration.py"
            },
        )


class RuntimeLifecycleStartupTests(unittest.TestCase):
    """J: RuntimeLifecycle.start/stop drive the decay engine it owns.

    ``server_app.py`` is side-effectful and imports dependencies the CI
    test environment does not install, so this freezes the lifespan
    wiring as a source contract instead of importing the production
    server.
    """

    def test_J_start_starts_provided_decay_engine(self):
        tree = _parse("src/server_app.py")

        start = _class_method(tree, "RuntimeLifecycle", "start")
        self.assertTrue(
            _is_decay_engine_getattr(
                _run_async_step_wiring(start).get(
                    "decay engine start"
                ),
                "start",
            ),
            "RuntimeLifecycle.start must start its decay engine",
        )

        stop = _class_method(tree, "RuntimeLifecycle", "stop")
        self.assertTrue(
            _is_decay_engine_getattr(
                _run_async_step_wiring(stop).get(
                    "decay engine stop"
                ),
                "stop",
            ),
            "RuntimeLifecycle.stop must stop its decay engine",
        )

    def test_J_start_without_decay_engine_is_safe(self):
        tree = _parse("src/server_app.py")
        step = _class_method(
            tree, "RuntimeLifecycle", "_run_async_step"
        )
        # A missing/None decay_engine must be a no-op, not a crash.
        has_none_guard = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "callback"
            for node in ast.walk(step)
        )
        self.assertTrue(
            has_none_guard,
            "_run_async_step must no-op on a None callback",
        )


class ServerWiringContractTests(unittest.TestCase):
    """K: streamable-http wires decay_engine; stdio current wiring does not."""

    def _lifecycle_assignments(self):
        tree = _parse("src/server.py")
        assignments = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(
                node.value, ast.Call
            ):
                continue
            call = node.value
            if not (
                isinstance(call.func, ast.Name)
                and call.func.id == "RuntimeLifecycle"
            ):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = {
                        kw.arg for kw in call.keywords
                    }
        return assignments

    def test_K_http_wires_decay_engine_and_stdio_does_not(self):
        assignments = self._lifecycle_assignments()
        self.assertEqual(
            set(assignments),
            {"_runtime_lifecycle", "_stdio_runtime_lifecycle"},
        )
        self.assertIn(
            "decay_engine", assignments["_runtime_lifecycle"]
        )
        self.assertNotIn(
            "decay_engine",
            assignments["_stdio_runtime_lifecycle"],
        )

    def test_K_alternate_stdio_start_is_the_documented_lazy_path(self):
        # DecayEngine can only start through two legitimate paths:
        #   A. RuntimeLifecycle.start()
        #      -> getattr(self.decay_engine, "start", None)  (covered by J)
        #   B. lazy `<recv>.decay_engine.ensure_started()` from a tool, where
        #      DecayEngine.ensure_started() calls self.start().
        # Target-aware rule: only calls whose receiver is a
        # `*.decay_engine` attribute expression count. Scan ALL of src so a
        # new, unexpected DecayEngine startup authority anywhere fails here.
        ensure_files: set[str] = set()
        direct_start_files: set[str] = set()
        for path in sorted(_SRC.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            rel = str(path.relative_to(_REPO_ROOT))
            for node in ast.walk(tree):
                target = _decay_engine_startup_target(node)
                if target == "ensure_started":
                    ensure_files.add(rel)
                elif target == "start":
                    direct_start_files.add(rel)
        # The only `*.decay_engine.ensure_started()` callers are the seven
        # documented lazy tool entrypoints.
        self.assertEqual(ensure_files, _DOCUMENTED_LAZY_START_FILES)
        # Zero unexpected direct `*.decay_engine.start()` calls. The
        # intentional eager start is the RuntimeLifecycle getattr path,
        # tested separately, and is not a direct attribute call.
        self.assertEqual(
            direct_start_files,
            set(),
            "unexpected direct *.decay_engine.start() call site",
        )

    def test_K_startup_receiver_rule_is_target_aware(self):
        # The target-aware rule must match DecayEngine-bearing receivers but
        # ignore unrelated ensure_started()/start() calls, and must catch a
        # direct *.decay_engine.start().
        cases = {
            "rt.decay_engine.ensure_started()": "ensure_started",
            "something.decay_engine.ensure_started()": "ensure_started",
            "self.decay_engine.ensure_started()": "ensure_started",
            "something.decay_engine.start()": "start",
            "self.ensure_started()": None,
            "outbox.ensure_started()": None,
            "service.start()": None,
            "getattr(self.decay_engine, 'start', None)": None,
        }
        for src, expected in cases.items():
            node = ast.parse(src).body[0].value
            self.assertEqual(
                _decay_engine_startup_target(node), expected, src
            )

    def test_K_decay_engine_ensure_started_calls_start(self):
        # Path B: ensure_started() must actually start the background loop.
        tree = _parse("src/decay_engine.py")
        method = _class_method(tree, "DecayEngine", "ensure_started")
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "start"
                for node in ast.walk(method)
            ),
            "DecayEngine.ensure_started must call self.start()",
        )


class ArchitectureDocHygieneTests(unittest.TestCase):
    def test_architecture_docs_have_no_machine_local_links(self):
        for rel in _ARCHITECTURE_DOCS:
            text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("file:///workspace", text, rel)


if __name__ == "__main__":
    unittest.main()
