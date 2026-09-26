from __future__ import annotations

import json
import math
import tempfile
import unittest
from copy import deepcopy
from datetime import timedelta

from _lifecycle_fixtures import (
    CID,
    M1,
    M2,
    RID,
    T0,
    guarded_bucket_manager,
    hours,
    iso,
    lifecycle_env,
    source_bucket,
)

from ombrebrain.context.memory_lifecycle_event import (
    build_lifecycle_event,
    memory_key,
    persist_lifecycle_event,
)
from ombrebrain.context.memory_lifecycle_state import (
    STATE_VERSION,
    accessibility_for_age,
    is_valid_lifecycle_state_artifact,
    lifecycle_state_path,
    read_memory_lifecycle_state,
    resolve_lifecycle_config,
    salience_for_age,
    strength_for_use_count,
)
from ombrebrain.context.memory_reinforcement import (
    derive_memory_lifecycle_state,
)


def recall_id(seed: int) -> str:
    return "recall_" + format(seed, "032x")


def _seed(
    memory_id: str,
    *,
    times: tuple[str, ...],
    first_recall: int = 1,
    index: int = 2,
) -> None:
    for offset, at in enumerate(times):
        persist_lifecycle_event(
            build_lifecycle_event(
                conversation_id=CID,
                cognitive_request_id=RID,
                recall_id=recall_id(
                    first_recall + offset
                ),
                memory_id=memory_id,
                source_usage_event_at=at,
                source_usage_event_index=index,
            )
        )


class LifecycleFormulaTests(unittest.TestCase):
    def setUp(self):
        self.config = resolve_lifecycle_config()

    def test_strength_formula_examples(self):
        expected = {
            0: 0.20,
            1: 0.32,
            2: 0.422,
        }

        for use_count, value in expected.items():
            with self.subTest(use_count=use_count):
                self.assertAlmostEqual(
                    strength_for_use_count(
                        use_count,
                        config=self.config,
                    ),
                    value,
                    places=3,
                )

    def test_strength_monotonic_and_bounded(self):
        previous = -1.0

        for use_count in (0, 1, 2, 5, 20, 500):
            with self.subTest(use_count=use_count):
                value = strength_for_use_count(
                    use_count,
                    config=self.config,
                )

                self.assertGreaterEqual(
                    value, previous
                )
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
                self.assertTrue(
                    math.isfinite(value)
                )

                previous = value

    def test_accessibility_decay_by_half_life(
        self,
    ):
        floor = self.config[
            "accessibility_floor"
        ]

        strength = 0.32

        fresh = accessibility_for_age(
            strength, 0.0, config=self.config
        )

        one_half_life = accessibility_for_age(
            strength, 168.0, config=self.config
        )

        two_half_lives = accessibility_for_age(
            strength, 336.0, config=self.config
        )

        self.assertAlmostEqual(
            fresh,
            floor + (1.0 - floor) * strength,
            places=6,
        )
        self.assertAlmostEqual(
            one_half_life,
            floor
            + (1.0 - floor) * strength * 0.5,
            places=6,
        )
        self.assertAlmostEqual(
            two_half_lives,
            floor
            + (1.0 - floor) * strength * 0.25,
            places=6,
        )

        self.assertGreater(fresh, one_half_life)
        self.assertGreater(
            one_half_life, two_half_lives
        )
        self.assertGreaterEqual(
            two_half_lives, floor
        )
        self.assertLessEqual(fresh, 1.0)

    def test_accessibility_floor_and_clamp(self):
        floor = self.config[
            "accessibility_floor"
        ]

        for age in (0.0, 1.0, 168.0, 100000.0):
            with self.subTest(age=age):
                value = accessibility_for_age(
                    1.0,
                    age,
                    config=self.config,
                )

                self.assertGreaterEqual(
                    value, floor - 1e-9
                )
                self.assertLessEqual(value, 1.0)
                self.assertTrue(
                    math.isfinite(value)
                )

    def test_negative_age_never_boosts(self):
        self.assertEqual(
            accessibility_for_age(
                0.32,
                -100.0,
                config=self.config,
            ),
            accessibility_for_age(
                0.32, 0.0, config=self.config
            ),
        )

    def test_salience_decay_by_half_life(self):
        fresh = salience_for_age(
            use_count=1,
            age_since_last_used_hours=0.0,
            config=self.config,
        )
        one_half_life = salience_for_age(
            use_count=1,
            age_since_last_used_hours=24.0,
            config=self.config,
        )
        two_half_lives = salience_for_age(
            use_count=1,
            age_since_last_used_hours=48.0,
            config=self.config,
        )

        self.assertAlmostEqual(fresh, 1.0)
        self.assertAlmostEqual(
            one_half_life, 0.5
        )
        self.assertAlmostEqual(
            two_half_lives, 0.25
        )

    def test_salience_zero_without_used(self):
        self.assertEqual(
            salience_for_age(
                use_count=0,
                age_since_last_used_hours=0.0,
                config=self.config,
            ),
            0.0,
        )


class LifecycleConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = resolve_lifecycle_config()

        self.assertAlmostEqual(
            config["initial_strength"], 0.20
        )
        self.assertAlmostEqual(
            config["strength_gain"], 0.15
        )
        self.assertAlmostEqual(
            config[
                "accessibility_half_life_hours"
            ],
            168.0,
        )
        self.assertAlmostEqual(
            config["salience_half_life_hours"],
            24.0,
        )
        self.assertAlmostEqual(
            config["accessibility_floor"], 0.02
        )
        self.assertAlmostEqual(
            config[
                "low_accessibility_threshold"
            ],
            0.10,
        )

    def test_invalid_values_degrade_to_default(
        self,
    ):
        env = {
            "OMBRE_MEMORY_LIFECYCLE_STRENGTH_INITIAL":
                "not-a-number",
            "OMBRE_MEMORY_LIFECYCLE_STRENGTH_GAIN":
                "nan",
            "OMBRE_MEMORY_LIFECYCLE_ACCESS_HALF_LIFE_HOURS":
                "inf",
            "OMBRE_MEMORY_LIFECYCLE_SALIENCE_HALF_LIFE_HOURS":
                "-3",
            "OMBRE_MEMORY_LIFECYCLE_ACCESS_FLOOR":
                "",
            "OMBRE_MEMORY_LIFECYCLE_LOW_ACCESS_THRESHOLD":
                "None",
        }

        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root, **env):
                config = resolve_lifecycle_config()

        self.assertAlmostEqual(
            config["initial_strength"], 0.20
        )
        self.assertAlmostEqual(
            config["strength_gain"], 0.15
        )

        # "-3" is finite, so it is clamped to the hard lower bound.
        self.assertAlmostEqual(
            config["salience_half_life_hours"],
            1.0,
        )
        self.assertAlmostEqual(
            config["accessibility_floor"], 0.02
        )
        self.assertAlmostEqual(
            config[
                "low_accessibility_threshold"
            ],
            0.10,
        )

    def test_out_of_range_values_are_clamped(
        self,
    ):
        env = {
            "OMBRE_MEMORY_LIFECYCLE_STRENGTH_INITIAL":
                "5",
            "OMBRE_MEMORY_LIFECYCLE_STRENGTH_GAIN":
                "9",
            "OMBRE_MEMORY_LIFECYCLE_ACCESS_HALF_LIFE_HOURS":
                "0",
            "OMBRE_MEMORY_LIFECYCLE_SALIENCE_HALF_LIFE_HOURS":
                "999999",
            "OMBRE_MEMORY_LIFECYCLE_ACCESS_FLOOR":
                "0.9",
            "OMBRE_MEMORY_LIFECYCLE_LOW_ACCESS_THRESHOLD":
                "0.9",
        }

        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root, **env):
                config = resolve_lifecycle_config()

        self.assertAlmostEqual(
            config["initial_strength"], 0.8
        )
        self.assertAlmostEqual(
            config["strength_gain"], 0.5
        )
        self.assertAlmostEqual(
            config[
                "accessibility_half_life_hours"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            config["salience_half_life_hours"],
            720.0,
        )
        self.assertAlmostEqual(
            config["accessibility_floor"], 0.25
        )
        self.assertAlmostEqual(
            config[
                "low_accessibility_threshold"
            ],
            0.5,
        )

    def test_degenerate_config_stays_finite(self):
        env = {
            "OMBRE_MEMORY_LIFECYCLE_ACCESS_HALF_LIFE_HOURS":
                "0",
            "OMBRE_MEMORY_LIFECYCLE_STRENGTH_GAIN":
                "1",
        }

        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root, **env):
                config = resolve_lifecycle_config()

        value = accessibility_for_age(
            1.0, 10.0, config=config
        )

        self.assertTrue(math.isfinite(value))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


class LifecycleStateDerivationTests(
    unittest.IsolatedAsyncioTestCase,
):
    async def _derive(
        self,
        root,
        memory_id,
        *,
        as_of,
        buckets=None,
    ):
        return (
            await derive_memory_lifecycle_state(
                memory_id,
                bucket_manager=(
                    guarded_bucket_manager(
                        buckets or {}
                    )
                ),
                as_of=as_of,
            )
        )

    async def test_zero_use_state(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                report = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={
                        M1: source_bucket(
                            M1,
                            created=iso(
                                T0 - hours(10)
                            ),
                        )
                    },
                )

        state = report["state"]

        self.assertEqual(state["use_count"], 0)
        self.assertAlmostEqual(
            state["strength"], 0.20
        )
        self.assertEqual(state["salience"], 0.0)
        self.assertIsNone(
            state["last_used_at"]
        )
        self.assertEqual(
            state["reference_source"],
            "source_created",
        )
        self.assertEqual(
            state["reference_at"],
            iso(T0 - hours(10)),
        )

    async def test_use_count_drives_strength(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _seed(
                    M1,
                    times=(
                        iso(T0 - hours(2)),
                        iso(T0 - hours(1)),
                    ),
                )

                report = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={M1: source_bucket(M1)},
                )

        state = report["state"]

        self.assertEqual(state["use_count"], 2)
        self.assertEqual(
            state["derived_from_event_count"], 2
        )
        self.assertAlmostEqual(
            state["strength"], 0.422, places=3
        )
        self.assertEqual(
            state["last_used_at"],
            iso(T0 - hours(1)),
        )
        self.assertEqual(
            state["reference_source"], "last_used"
        )

    async def test_multiple_recalls_accumulate(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _seed(
                    M1,
                    times=(
                        iso(T0 - hours(3)),
                        iso(T0 - hours(2)),
                        iso(T0 - hours(1)),
                    ),
                )

                report = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={M1: source_bucket(M1)},
                )

        self.assertEqual(
            report["state"]["use_count"], 3
        )

    async def test_reinforcement_raises_accessibility(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _seed(
                    M1,
                    times=(iso(T0 - hours(1)),),
                )

                _seed(
                    M2,
                    times=(
                        iso(T0 - hours(1)),
                        iso(T0 - hours(1)),
                        iso(T0 - hours(1)),
                    ),
                    first_recall=10,
                )

                one = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={M1: source_bucket(M1)},
                )

                three = await self._derive(
                    root,
                    M2,
                    as_of=T0,
                    buckets={M2: source_bucket(M2)},
                )

        self.assertEqual(one["state"]["use_count"], 1)
        self.assertEqual(
            three["state"]["use_count"], 3
        )
        self.assertEqual(
            one["state"]["last_used_at"],
            three["state"]["last_used_at"],
        )
        self.assertGreater(
            three["state"]["strength"],
            one["state"]["strength"],
        )
        self.assertGreater(
            three["state"]["accessibility"],
            one["state"]["accessibility"],
        )
        self.assertAlmostEqual(
            three["state"]["salience"],
            one["state"]["salience"],
        )

    async def test_time_lowers_accessibility_not_strength(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _seed(
                    M1,
                    times=(iso(T0),),
                )

                now = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={M1: source_bucket(M1)},
                )

                later = await self._derive(
                    root,
                    M1,
                    as_of=T0 + hours(168),
                    buckets={M1: source_bucket(M1)},
                )

                year = await self._derive(
                    root,
                    M1,
                    as_of=T0 + timedelta(days=365),
                    buckets={M1: source_bucket(M1)},
                )

        self.assertAlmostEqual(
            now["state"]["strength"],
            later["state"]["strength"],
        )
        self.assertAlmostEqual(
            now["state"]["strength"],
            year["state"]["strength"],
        )
        self.assertGreater(
            now["state"]["accessibility"],
            later["state"]["accessibility"],
        )
        self.assertGreater(
            later["state"]["accessibility"],
            year["state"]["accessibility"],
        )
        self.assertAlmostEqual(
            later["state"]["accessibility"],
            0.02
            + 0.98 * 0.32 * 0.5,
            places=6,
        )
        self.assertAlmostEqual(
            now["state"]["salience"], 1.0
        )
        self.assertGreater(
            now["state"]["salience"],
            later["state"]["salience"],
        )

    async def test_forgetting_never_deletes(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _seed(
                    M1,
                    times=(iso(T0),),
                )

                far = await self._derive(
                    root,
                    M1,
                    as_of=T0 + timedelta(days=365),
                    buckets={M1: source_bucket(M1)},
                )

                state = read_memory_lifecycle_state(
                    M1
                )

                self.assertTrue(
                    far["state"]["exists"]
                )
                self.assertEqual(
                    far["state"]["event_count"], 1
                )
                self.assertEqual(
                    state["event_count"], 1
                )
                self.assertEqual(
                    state["use_count"], 1
                )
                self.assertTrue(
                    far["state"][
                        "low_accessibility"
                    ]
                )

    async def test_exists_false_keeps_events(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _seed(
                    M1,
                    times=(iso(T0),),
                )

                report = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={},
                )

                state = read_memory_lifecycle_state(
                    M1
                )

        self.assertFalse(report["state"]["exists"])
        self.assertTrue(
            report["state"][
                "source_exists_checked"
            ]
        )
        self.assertEqual(state["use_count"], 1)
        self.assertEqual(state["event_count"], 1)

    async def test_replay_from_events(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _seed(
                    M1,
                    times=(
                        iso(T0 - hours(3)),
                        iso(T0 - hours(2)),
                        iso(T0 - hours(1)),
                    ),
                )

                first = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={M1: source_bucket(M1)},
                )

                lifecycle_state_path(
                    memory_key(M1)
                ).unlink()

                second = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={M1: source_bucket(M1)},
                )

        for field in (
            "use_count",
            "strength",
            "last_used_at",
            "accessibility",
            "salience",
            "reference_at",
            "reference_source",
        ):
            with self.subTest(field=field):
                self.assertEqual(
                    first["state"][field],
                    second["state"][field],
                )

    async def test_corrupt_snapshot_is_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _seed(
                    M1,
                    times=(iso(T0),),
                )

                await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={M1: source_bucket(M1)},
                )

                path = lifecycle_state_path(
                    memory_key(M1)
                )

                corrupt = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

                corrupt["strength"] = 999
                corrupt["memory_id"] = M2
                corrupt["version"] = "bogus"

                path.write_text(
                    json.dumps(corrupt),
                    encoding="utf-8",
                )

                self.assertIsNone(
                    read_memory_lifecycle_state(
                        M1
                    )
                )

                report = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={M1: source_bucket(M1)},
                )

                healed = (
                    read_memory_lifecycle_state(
                        M1
                    )
                )

        self.assertAlmostEqual(
            report["state"]["strength"], 0.32
        )
        self.assertEqual(
            healed["version"], STATE_VERSION
        )
        self.assertAlmostEqual(
            healed["strength"], 0.32
        )

    async def test_reference_falls_back_to_observation(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                report = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={
                        M1: source_bucket(
                            M1, created=None
                        )
                    },
                )

        state = report["state"]

        self.assertEqual(
            state["reference_source"],
            "observation_fallback",
        )
        self.assertEqual(
            state["reference_at"], iso(T0)
        )
        self.assertIsNone(
            state["source_created_at"]
        )
        # ``last_active`` is NEVER the lifecycle reference.
        self.assertNotEqual(
            state["reference_at"],
            "2020-01-01T00:00:00Z",
        )

    async def test_unparseable_created_falls_back(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                buckets = {
                    M1: source_bucket(
                        M1, created="not-a-date"
                    )
                }

                report = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets=buckets,
                )

        self.assertEqual(
            report["state"]["reference_source"],
            "observation_fallback",
        )
        self.assertIsNone(
            report["state"]["source_created_at"]
        )

    async def test_future_event_clamped_to_zero_age(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _seed(
                    M1,
                    times=(iso(T0 + hours(5)),),
                )

                report = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={M1: source_bucket(M1)},
                )

        state = report["state"]

        self.assertEqual(
            state["future_event_count"], 1
        )
        self.assertAlmostEqual(
            state["accessibility"],
            0.02 + 0.98 * 0.32,
            places=6,
        )
        self.assertGreaterEqual(
            state["accessibility"], 0.0
        )

    async def test_numeric_integrity_and_json(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                _seed(
                    M1,
                    times=(
                        iso(T0 - hours(2)),
                        iso(T0 - hours(1)),
                    ),
                )

                report = await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets={M1: source_bucket(M1)},
                )

                path = lifecycle_state_path(
                    memory_key(M1)
                )

                text = path.read_text(
                    encoding="utf-8"
                )

        state = report["state"]

        for field in (
            "strength",
            "accessibility",
            "salience",
        ):
            with self.subTest(field=field):
                self.assertTrue(
                    math.isfinite(state[field])
                )
                self.assertGreaterEqual(
                    state[field], 0.0
                )
                self.assertLessEqual(
                    state[field], 1.0
                )

        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertTrue(
            is_valid_lifecycle_state_artifact(
                json.loads(text)
            )
        )

    async def test_derive_respects_disabled_flag(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root, enabled=False):
                report = (
                    await derive_memory_lifecycle_state(
                        M1, as_of=T0
                    )
                )

                self.assertFalse(report["state_stored"])
                self.assertIsNone(report["state"])
                self.assertEqual(
                    report["reason"],
                    "lifecycle_disabled",
                )

                self.assertFalse(
                    lifecycle_state_path(
                        memory_key(M1)
                    ).exists()
                )

    async def test_invalid_as_of_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                report = (
                    await derive_memory_lifecycle_state(
                        M1, as_of="not-a-time"
                    )
                )

        self.assertEqual(
            report["reason"], "invalid_as_of"
        )
        self.assertIsNone(report["state"])

    async def test_persist_false_returns_but_does_not_write(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                report = (
                    await derive_memory_lifecycle_state(
                        M1,
                        bucket_manager=(
                            guarded_bucket_manager(
                                {
                                    M1:
                                        source_bucket(
                                            M1
                                        )
                                }
                            )
                        ),
                        as_of=T0,
                        persist=False,
                    )
                )

                path = lifecycle_state_path(
                    memory_key(M1)
                )

        self.assertIsNotNone(report["state"])
        self.assertFalse(report["state_stored"])
        self.assertFalse(path.exists())

    async def test_state_artifact_not_mutating_source(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            with lifecycle_env(root):
                buckets = {
                    M1: source_bucket(
                        M1,
                        created=iso(
                            T0 - hours(5)
                        ),
                    )
                }

                pristine = deepcopy(buckets)

                _seed(
                    M1,
                    times=(iso(T0),),
                )

                await self._derive(
                    root,
                    M1,
                    as_of=T0,
                    buckets=buckets,
                )

        self.assertEqual(buckets, pristine)


if __name__ == "__main__":
    unittest.main()