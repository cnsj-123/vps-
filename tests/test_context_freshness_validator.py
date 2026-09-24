from __future__ import annotations

import unittest

from ombrebrain.context.validators import (
    validate_context_freshness as package_reexport,
)
from ombrebrain.context.validators.freshness import (
    DEFAULT_INVALID_REASON,
    DEFAULT_MISMATCH_REASON,
    validate_context_freshness,
)


class FreshnessValidatorContractTests(
    unittest.TestCase
):

    def test_result_shape(self):
        result = validate_context_freshness(
            checked_revision=1,
            expected_revision=1,
        )

        self.assertEqual(
            set(result),
            {
                "valid",
                "reason",
                "checked_revision",
                "expected_revision",
            },
        )

    def test_valid_revision_chain(self):
        result = validate_context_freshness(
            checked_revision=5,
            expected_revision=5,
        )

        self.assertTrue(result["valid"])
        self.assertIsNone(result["reason"])
        self.assertEqual(
            result["checked_revision"],
            5,
        )
        self.assertEqual(
            result["expected_revision"],
            5,
        )

    def test_stale_revision(self):
        result = validate_context_freshness(
            checked_revision=4,
            expected_revision=5,
            mismatch_reason=(
                "preview_revision_mismatch"
            ),
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "preview_revision_mismatch",
        )
        self.assertEqual(
            result["checked_revision"],
            4,
        )
        self.assertEqual(
            result["expected_revision"],
            5,
        )

    def test_stale_revision_default_reason(self):
        result = validate_context_freshness(
            checked_revision=4,
            expected_revision=5,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            DEFAULT_MISMATCH_REASON,
        )

    def test_invalid_checked_revision(self):
        for bad in (
            None,
            True,
            False,
            0,
            -1,
            "5",
            5.0,
            [5],
        ):
            result = validate_context_freshness(
                checked_revision=bad,
                expected_revision=5,
                invalid_reason=(
                    "invalid_gate_source_revision"
                ),
            )

            self.assertFalse(
                result["valid"],
                bad,
            )
            self.assertEqual(
                result["reason"],
                "invalid_gate_source_revision",
                bad,
            )
            self.assertEqual(
                result["checked_revision"],
                bad,
            )

    def test_invalid_expected_revision(self):
        result = validate_context_freshness(
            checked_revision=5,
            expected_revision="5",
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            DEFAULT_INVALID_REASON,
        )

    def test_invalid_revision_wins_over_mismatch(self):
        # An invalid checked revision is reported before any
        # equality comparison is attempted.
        result = validate_context_freshness(
            checked_revision=None,
            expected_revision=5,
            invalid_reason="invalid_a",
            mismatch_reason="mismatch_b",
        )

        self.assertEqual(
            result["reason"],
            "invalid_a",
        )

    def test_package_reexport(self):
        self.assertIs(
            package_reexport,
            validate_context_freshness,
        )

    def test_pure_and_total(self):
        # Same inputs always produce the same structured result,
        # and no input can make the validator raise.
        first = validate_context_freshness(
            checked_revision=object(),
            expected_revision=object(),
        )

        second = validate_context_freshness(
            checked_revision=object(),
            expected_revision=object(),
        )

        self.assertEqual(
            first["valid"],
            second["valid"],
        )
        self.assertEqual(
            first["reason"],
            second["reason"],
        )
        self.assertFalse(first["valid"])
        self.assertEqual(
            first["reason"],
            DEFAULT_INVALID_REASON,
        )


class FreshnessValidatorEdgeTests(
    unittest.TestCase
):
    """Supplemental coverage: revision boundaries and reason paths."""

    def test_both_revisions_invalid(self):
        result = validate_context_freshness(
            checked_revision=None,
            expected_revision=None,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            DEFAULT_INVALID_REASON,
        )
        self.assertIsNone(
            result["checked_revision"]
        )
        self.assertIsNone(
            result["expected_revision"]
        )

    def test_invalid_expected_revision_reason(
        self,
    ):
        result = validate_context_freshness(
            checked_revision=5,
            expected_revision=0,
            invalid_reason="invalid_expected",
            mismatch_reason="mismatch",
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            "invalid_expected",
        )

    def test_boundary_revision_one_is_valid(
        self,
    ):
        result = validate_context_freshness(
            checked_revision=1,
            expected_revision=1,
        )

        self.assertTrue(result["valid"])
        self.assertIsNone(result["reason"])

    def test_zero_is_not_a_valid_revision(self):
        result = validate_context_freshness(
            checked_revision=0,
            expected_revision=0,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["reason"],
            DEFAULT_INVALID_REASON,
        )

    def test_boolean_is_never_a_revision(self):
        # bool subclasses int in Python, so the validator must
        # reject it explicitly.
        for bad in (True, False):
            result = validate_context_freshness(
                checked_revision=bad,
                expected_revision=1,
            )

            self.assertFalse(
                result["valid"],
                bad,
            )
            self.assertEqual(
                result["reason"],
                DEFAULT_INVALID_REASON,
                bad,
            )

    def test_large_revisions_compare_exactly(
        self,
    ):
        big = 2 ** 53

        fresh = validate_context_freshness(
            checked_revision=big,
            expected_revision=big,
        )

        self.assertTrue(fresh["valid"])

        stale = validate_context_freshness(
            checked_revision=big - 1,
            expected_revision=big,
            mismatch_reason="stale",
        )

        self.assertFalse(stale["valid"])
        self.assertEqual(
            stale["reason"],
            "stale",
        )

    def test_invalid_and_mismatch_reasons_are_distinct(
        self,
    ):
        # Invalid input is rejected before equality is evaluated.
        invalid = validate_context_freshness(
            checked_revision=None,
            expected_revision=5,
            invalid_reason="a",
            mismatch_reason="b",
        )

        self.assertEqual(invalid["reason"], "a")

        # Two valid but different revisions take the mismatch path.
        mismatch = validate_context_freshness(
            checked_revision=4,
            expected_revision=5,
            invalid_reason="a",
            mismatch_reason="b",
        )

        self.assertEqual(
            mismatch["reason"],
            "b",
        )


if __name__ == "__main__":
    unittest.main()
