from __future__ import annotations

import json
import unittest

from ombrebrain.gateway.response_usage import (
    ResponseUsageObserver,
)


class GatewayResponseUsageTests(
    unittest.TestCase
):

    def test_sse_usage_across_chunks(self):
        observer = ResponseUsageObserver(
            "text/event-stream"
        )

        stream = (
            'event: message_start\n'
            'data: {"type":"message_start",'
            '"message":{"usage":{'
            '"input_tokens":123,'
            '"cache_creation_input_tokens":456,'
            '"cache_read_input_tokens":789}}}\n\n'
            'event: content_block_delta\n'
            'data: {"type":"content_block_delta",'
            '"delta":{"type":"text_delta",'
            '"text":"private-secret-text"}}\n\n'
            'event: message_delta\n'
            'data: {"type":"message_delta",'
            '"usage":{"output_tokens":42}}\n\n'
        ).encode()

        # Deliberately split in awkward places.
        for start in range(0, len(stream), 17):
            observer.feed(
                stream[start:start + 17]
            )

        result = observer.finish()

        self.assertTrue(result["observed"])
        self.assertEqual(
            result["transport"],
            "sse",
        )
        self.assertEqual(
            result["input_tokens"],
            123,
        )
        self.assertEqual(
            result[
                "cache_creation_input_tokens"
            ],
            456,
        )
        self.assertEqual(
            result[
                "cache_read_input_tokens"
            ],
            789,
        )
        self.assertEqual(
            result["output_tokens"],
            42,
        )

        rendered = json.dumps(result)

        self.assertNotIn(
            "private-secret-text",
            rendered,
        )

    def test_message_delta_zeroes_do_not_erase_start_usage(self):
        observer = ResponseUsageObserver(
            "text/event-stream"
        )

        stream = (
            'event: message_start\n'
            'data: {"type":"message_start",'
            '"message":{"usage":{'
            '"input_tokens":1016,'
            '"output_tokens":1,'
            '"cache_creation_input_tokens":0,'
            '"cache_read_input_tokens":46171}}}\n\n'
            'event: message_delta\n'
            'data: {"type":"message_delta",'
            '"delta":{"stop_reason":"end_turn"},'
            '"usage":{'
            '"input_tokens":0,'
            '"output_tokens":115,'
            '"cache_creation_input_tokens":0,'
            '"cache_read_input_tokens":0}}\n\n'
        ).encode()

        observer.feed(stream)

        result = observer.finish()

        self.assertEqual(
            result["input_tokens"],
            1016,
        )
        self.assertEqual(
            result["cache_creation_input_tokens"],
            0,
        )
        self.assertEqual(
            result["cache_read_input_tokens"],
            46171,
        )
        self.assertEqual(
            result["output_tokens"],
            115,
        )

    def test_message_start_explicit_zero_cache_fields_are_present(self):
        observer = ResponseUsageObserver(
            "text/event-stream"
        )

        stream = (
            'event: message_start\n'
            'data: {"type":"message_start",'
            '"message":{"usage":{'
            '"input_tokens":0,'
            '"output_tokens":1,'
            '"cache_creation_input_tokens":0,'
            '"cache_read_input_tokens":0}}}\n\n'
        ).encode()

        observer.feed(stream)
        result = observer.finish()

        self.assertTrue(
            result["message_start_seen"]
        )
        self.assertTrue(
            result["message_start_usage_present"]
        )
        self.assertTrue(
            result["message_start_input_present"]
        )
        self.assertTrue(
            result["message_start_output_present"]
        )
        self.assertTrue(
            result["message_start_cache_creation_present"]
        )
        self.assertTrue(
            result["message_start_cache_read_present"]
        )

        self.assertEqual(
            result["cache_read_input_tokens"],
            0,
        )

    def test_message_start_missing_cache_fields_is_detected(self):
        observer = ResponseUsageObserver(
            "text/event-stream"
        )

        stream = (
            'event: message_start\n'
            'data: {"type":"message_start",'
            '"message":{"usage":{'
            '"input_tokens":0,'
            '"output_tokens":1}}}\n\n'
        ).encode()

        observer.feed(stream)
        result = observer.finish()

        self.assertTrue(
            result["message_start_seen"]
        )
        self.assertTrue(
            result["message_start_usage_present"]
        )
        self.assertFalse(
            result["message_start_cache_creation_present"]
        )
        self.assertFalse(
            result["message_start_cache_read_present"]
        )

    def test_json_usage(self):
        observer = ResponseUsageObserver(
            "application/json"
        )

        body = json.dumps({
            "content": [
                {
                    "type": "text",
                    "text": "private-response",
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 30,
                "cache_read_input_tokens": 40,
            },
        }).encode()

        observer.feed(body)

        result = observer.finish()

        self.assertTrue(result["observed"])
        self.assertEqual(
            result["cache_read_input_tokens"],
            40,
        )

        rendered = json.dumps(result)

        self.assertNotIn(
            "private-response",
            rendered,
        )

    def test_no_usage_is_safe(self):
        observer = ResponseUsageObserver(
            "text/event-stream"
        )

        observer.feed(
            b'data: {"type":"ping"}\n\n'
        )

        result = observer.finish()

        self.assertFalse(result["observed"])
        self.assertNotIn(
            "input_tokens",
            result,
        )


if __name__ == "__main__":
    unittest.main()
