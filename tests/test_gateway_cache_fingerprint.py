from __future__ import annotations

import json
import unittest

from ombrebrain.gateway.cache_fingerprint import (
    cache_fingerprint_summary_from_body,
)


class GatewayCacheFingerprintTests(
    unittest.TestCase
):

    def _body(self):
        return json.dumps({
            "model": "private-model-name",
            "tools": [
                {
                    "name": "private_tool",
                    "description": "private tool text",
                }
            ],
            "system": [
                {
                    "type": "text",
                    "text": "private system text",
                }
            ],
            "tool_choice": {
                "type": "auto",
            },
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "private history",
                            "cache_control": {
                                "type": "ephemeral",
                                "ttl": "1h",
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "private current",
                        }
                    ],
                },
            ],
        }).encode()

    def test_summary_contains_no_prompt_text(self):
        summary = (
            cache_fingerprint_summary_from_body(
                self._body()
            )
        )

        self.assertIsNotNone(summary)

        rendered = json.dumps(summary)

        self.assertNotIn(
            "private-model-name",
            rendered,
        )
        self.assertNotIn(
            "private_tool",
            rendered,
        )
        self.assertNotIn(
            "private system text",
            rendered,
        )
        self.assertNotIn(
            "private history",
            rendered,
        )
        self.assertNotIn(
            "private current",
            rendered,
        )

        self.assertEqual(
            summary["tools_count"],
            1,
        )
        self.assertEqual(
            summary["messages_count"],
            2,
        )
        self.assertEqual(
            summary["boundary_message_index"],
            0,
        )

    def test_same_structure_same_hashes(self):
        a = cache_fingerprint_summary_from_body(
            self._body()
        )
        b = cache_fingerprint_summary_from_body(
            self._body()
        )

        self.assertEqual(a, b)

    def test_system_change_changes_system_only(self):
        payload = json.loads(self._body())

        before = (
            cache_fingerprint_summary_from_body(
                json.dumps(payload).encode()
            )
        )

        payload["system"][0]["text"] += " changed"

        after = (
            cache_fingerprint_summary_from_body(
                json.dumps(payload).encode()
            )
        )

        self.assertNotEqual(
            before["system_sha256"],
            after["system_sha256"],
        )

        self.assertEqual(
            before["tools_sha256"],
            after["tools_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
