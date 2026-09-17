from __future__ import annotations

import json
import unittest

from ombrebrain.gateway.fingertips_probe import (
    fingertips_ownership_summary_from_body,
)
from ombrebrain.gateway.rewriter import (
    _rewrite_historical_text,
)


def attachment(
    attachment_id: str,
    filename: str,
    body: str = "secret",
) -> str:
    return (
        '<attachment '
        f'id="{attachment_id}" '
        f'filename="{filename}" '
        'type="text/plain">'
        f"{body}"
        "</attachment>"
    )


class GatewayAttachmentOwnershipTests(
    unittest.TestCase
):

    def test_modern_operit_id_is_removed(self):
        text = (
            "hello "
            + attachment(
                "message_insert_extra_bundle_123",
                "Time:12:00",
            )
        )

        rewritten, report = (
            _rewrite_historical_text(text)
        )

        self.assertEqual(rewritten.strip(), "hello")
        self.assertEqual(
            report["dynamic_context_removed"],
            1,
        )

    def test_time_filename_without_operit_id_survives(self):
        marker = attachment(
            "ordinary_document_123",
            "Time:notes.txt",
        )

        text = "hello " + marker

        rewritten, report = (
            _rewrite_historical_text(text)
        )

        self.assertIn(marker, rewritten)
        self.assertEqual(
            report["dynamic_context_removed"],
            0,
        )

    def test_legacy_operit_id_is_removed(self):
        text = (
            "hello "
            + attachment(
                "message_insert_extra_weather_123",
                "anything.txt",
            )
        )

        rewritten, report = (
            _rewrite_historical_text(text)
        )

        self.assertEqual(rewritten.strip(), "hello")
        self.assertEqual(
            report["dynamic_context_removed"],
            1,
        )

    def test_fingertips_is_preserved(self):
        marker = attachment(
            "fingertips_123",
            "fingertips.txt",
        )

        rewritten, report = (
            _rewrite_historical_text(
                "hello " + marker
            )
        )

        self.assertIn(marker, rewritten)
        self.assertEqual(
            report["fingertips_removed"],
            0,
        )

    def test_attachment_only_dynamic_is_guarded(self):
        marker = attachment(
            "message_insert_extra_bundle_123",
            "Time:12:00",
        )

        rewritten, report = (
            _rewrite_historical_text(marker)
        )

        self.assertEqual(rewritten, marker)
        self.assertEqual(
            report["dynamic_context_removed"],
            0,
        )
        self.assertEqual(
            report["empty_guard_preserved"],
            1,
        )

    def test_probe_contains_no_raw_metadata(self):
        secret_id = "fingertips_private_abc"
        secret_filename = "fingertips.txt"

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": attachment(
                                secret_id,
                                secret_filename,
                            ),
                        }
                    ],
                }
            ]
        }

        body = json.dumps(payload).encode()

        summary = (
            fingertips_ownership_summary_from_body(
                body
            )
        )

        self.assertIsNotNone(summary)
        assert summary is not None

        rendered = json.dumps(summary)

        self.assertNotIn(secret_id, rendered)
        self.assertNotIn(
            secret_filename,
            rendered,
        )

        self.assertEqual(
            summary["candidates"],
            1,
        )

        self.assertEqual(
            summary["features"]
            ["id_starts_fingertips"],
            1,
        )

        self.assertEqual(
            summary["features"]
            ["filename_exact_fingertips_txt"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
