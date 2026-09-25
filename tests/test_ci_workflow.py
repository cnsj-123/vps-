from __future__ import annotations

import unittest
from pathlib import Path

import yaml


_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "python-tests.yml"
)


class CiWorkflowTests(
    unittest.TestCase,
):
    """Static checks for the minimal GitHub Actions CI.

    The repository has no actionlint, so the workflow is validated
    by reading it directly: triggers, Python version, install step
    and the compile / test commands. The workflow must stay
    side-effect free: no deploy, no SSH, no secrets, no push.
    """

    def setUp(self):
        self.assertTrue(
            _WORKFLOW_PATH.is_file(),
            str(_WORKFLOW_PATH),
        )

        self.source = _WORKFLOW_PATH.read_text(
            encoding="utf-8"
        )

        self.workflow = yaml.safe_load(
            self.source
        )

        # PyYAML reads the bare YAML 1.1 key ``on`` as boolean True.
        self.triggers = (
            self.workflow.get("on")
            or self.workflow.get(True)
            or {}
        )

    def test_workflow_parses_as_yaml(
        self,
    ):
        self.assertIsInstance(
            self.workflow,
            dict,
        )

        self.assertIn(
            "jobs",
            self.workflow,
        )

    def test_push_and_pull_request_branches(
        self,
    ):
        for trigger in (
            "push",
            "pull_request",
        ):
            with self.subTest(
                trigger=trigger
            ):
                self.assertIn(
                    trigger,
                    self.triggers,
                )

                self.assertEqual(
                    self.triggers[trigger][
                        "branches"
                    ],
                    [
                        "feature/gateway-v2",
                        "main",
                    ],
                )

    def test_workflow_dispatch_enabled(
        self,
    ):
        self.assertIn(
            "workflow_dispatch",
            self.triggers,
        )

    def test_python_version_is_pinned(
        self,
    ):
        versions = [
            step["with"][
                "python-version"
            ]
            for step in self._steps()
            if "python-version"
            in step.get("with", {})
        ]

        self.assertEqual(
            versions,
            ["3.12"],
        )

    def test_install_step_does_not_add_second_requirements_system(
        self,
    ):
        runs = self._run_commands()

        install = [
            run
            for run in runs
            if "pip install" in run
        ]

        self.assertTrue(
            install,
            "expected a pip install step",
        )

        joined = "\n".join(
            install
        )

        # The project has no dependency manifest, so CI installs the
        # test dependencies directly instead of inventing a second
        # requirements system.
        for forbidden in (
            "-r requirements",
            "requirements.txt",
            "requirements-dev.txt",
            "pyproject.toml",
            "setup.py",
        ):
            with self.subTest(
                forbidden=forbidden
            ):
                self.assertNotIn(
                    forbidden,
                    joined,
                )

        # ...and the packages the suite actually imports are there.
        for package in (
            "starlette",
            "httpx",
        ):
            with self.subTest(
                package=package
            ):
                self.assertIn(
                    package,
                    joined,
                )

    def test_compile_and_test_commands(
        self,
    ):
        runs = self._run_commands()

        self.assertTrue(
            any(
                "compileall -q src" in run
                for run in runs
            ),
            "expected python -m compileall -q src",
        )

        self.assertTrue(
            any(
                "unittest discover -s tests -v"
                in run
                for run in runs
            ),
            "expected the unit test discovery command",
        )

    def test_workflow_has_no_side_effects(
        self,
    ):
        # Only the executable parts are scanned: the surrounding
        # comments intentionally describe what CI must NOT do.
        executable = "\n".join(
            self._run_commands()
            + self._uses_refs()
        ).lower()

        for forbidden in (
            "secrets.",
            "ssh",
            "scp ",
            "deploy",
            "vps",
            "git push",
            "anthropic",
            "openai",
            "curl ",
            "wget ",
            "aws ",
        ):
            with self.subTest(
                forbidden=forbidden
            ):
                self.assertNotIn(
                    forbidden,
                    executable,
                )

    def test_permissions_are_read_only(
        self,
    ):
        self.assertEqual(
            self.workflow.get(
                "permissions"
            ),
            {
                "contents": "read",
            },
        )

    def _steps(self) -> list:
        steps: list = []

        for job in (
            self.workflow["jobs"].values()
        ):
            steps.extend(
                job.get("steps", [])
            )

        return steps

    def _run_commands(self) -> list[str]:
        return [
            step["run"]
            for step in self._steps()
            if step.get("run")
        ]

    def _uses_refs(self) -> list[str]:
        return [
            step["uses"]
            for step in self._steps()
            if step.get("uses")
        ]


if __name__ == "__main__":
    unittest.main()