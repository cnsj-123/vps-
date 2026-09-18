from __future__ import annotations

import os
import tempfile
import unittest

from ombrebrain.gateway.gateway_runtime import (
    cache_usage_summary,
    clear_upstream_override,
    record_cache_usage,
    resolve_upstream_base,
    set_upstream_override,
    validate_upstream_url,
)


class GatewayRuntimeTests(
    unittest.TestCase
):

    def setUp(self):
        self.temp = (
            tempfile.TemporaryDirectory()
        )

        self.old_dir = os.environ.get(
            "OMBRE_GATEWAY_STATE_DIR"
        )

        os.environ[
            "OMBRE_GATEWAY_STATE_DIR"
        ] = self.temp.name

    def tearDown(self):
        if self.old_dir is None:
            os.environ.pop(
                "OMBRE_GATEWAY_STATE_DIR",
                None,
            )
        else:
            os.environ[
                "OMBRE_GATEWAY_STATE_DIR"
            ] = self.old_dir

        self.temp.cleanup()

    def test_validate_upstream(self):
        self.assertEqual(
            validate_upstream_url(
                "https://relay.example/v1/messages/"
            ),
            "https://relay.example/v1/messages",
        )

        with self.assertRaises(
            ValueError
        ):
            validate_upstream_url(
                "ftp://relay.example"
            )

        with self.assertRaises(
            ValueError
        ):
            validate_upstream_url(
                "https://user:secret@relay.example"
            )

        with self.assertRaises(
            ValueError
        ):
            validate_upstream_url(
                "https://relay.example/v1?token=secret"
            )

    def test_dashboard_override_wins(self):
        set_upstream_override(
            "https://new.example/v1/messages"
        )

        value, source = resolve_upstream_base(
            "https://old.example/v1/messages"
        )

        self.assertEqual(
            value,
            "https://new.example/v1/messages",
        )

        self.assertEqual(
            source,
            "dashboard",
        )

        clear_upstream_override()

        value, source = resolve_upstream_base(
            "https://old.example/v1/messages"
        )

        self.assertEqual(
            source,
            "environment",
        )

    def test_hit_rate_is_real_token_ratio(self):
        record = record_cache_usage(
            {
                "observed": True,
                "input_tokens": 500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 52000,
                "output_tokens": 100,
            },
            upstream=(
                "https://relay.example/v1/messages"
            ),
        )

        self.assertIsNotNone(
            record
        )

        self.assertAlmostEqual(
            record["hit_rate"],
            52000 / 52500,
        )

        self.assertEqual(
            record["status"],
            "hit",
        )

        self.assertEqual(
            record["upstream_host"],
            "relay.example",
        )

    def test_missing_usage_is_unknown(self):
        record = record_cache_usage(
            {
                "observed": True,
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 80,
            }
        )

        self.assertEqual(
            record["status"],
            "unknown",
        )

        self.assertIsNone(
            record["hit_rate"]
        )

    def test_summary_only_averages_valid_rounds(self):
        record_cache_usage(
            {
                "observed": True,
                "input_tokens": 100,
                "cache_read_input_tokens": 900,
            }
        )

        record_cache_usage(
            {
                "observed": True,
                "input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        )

        result = cache_usage_summary(
            20
        )

        self.assertEqual(
            result["count"],
            2,
        )

        self.assertEqual(
            result["valid_count"],
            1,
        )

        self.assertAlmostEqual(
            result["average_hit_rate"],
            0.9,
        )


if __name__ == "__main__":
    unittest.main()
