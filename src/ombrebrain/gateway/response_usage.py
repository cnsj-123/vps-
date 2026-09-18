from __future__ import annotations

import json
from typing import Any


_ALLOWED_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

_MAX_LINE_BYTES = 256 * 1024
_MAX_JSON_BYTES = 512 * 1024


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return value

    return None


def _extract_usage_dict(
    payload: Any,
) -> dict[str, int | float]:
    """
    Extract only numeric token accounting fields.

    No response text, model output, tool payload,
    identifiers, or arbitrary metadata is returned.
    """

    candidates: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        usage = payload.get("usage")

        if isinstance(usage, dict):
            candidates.append(usage)

        message = payload.get("message")

        if isinstance(message, dict):
            usage = message.get("usage")

            if isinstance(usage, dict):
                candidates.append(usage)

    result: dict[str, int | float] = {}

    for usage in candidates:
        for key in _ALLOWED_USAGE_KEYS:
            value = _safe_number(usage.get(key))

            if value is not None:
                result[key] = value

    return result


class ResponseUsageObserver:
    """
    Streaming-safe, privacy-safe response observer.

    Bytes are observed and returned unchanged by the caller.

    SSE:
    - buffers only until newline
    - ignores lines without usage/cache markers
    - parses only candidate JSON event envelopes

    JSON:
    - bounded buffer
    - extracts usage only at finish()
    """

    def __init__(
        self,
        content_type: str = "",
    ) -> None:
        self._content_type = (
            content_type or ""
        ).lower()

        self._is_sse = (
            "text/event-stream"
            in self._content_type
        )

        self._line_buffer = bytearray()
        self._json_buffer = bytearray()

        self._usage: dict[
            str,
            int | float,
        ] = {}

        self._oversize_lines = 0
        self._json_truncated = False

    def _merge_usage(
        self,
        usage: dict[str, int | float],
    ) -> None:
        for key, value in usage.items():
            self._usage[key] = value

    def _inspect_sse_line(
        self,
        raw_line: bytes,
    ) -> None:
        line = raw_line.strip()

        if not line.startswith(b"data:"):
            return

        data = line[5:].strip()

        if not data or data == b"[DONE]":
            return

        # Avoid parsing ordinary model text events.
        if (
            b'"usage"' not in data
            and b'"cache_' not in data
        ):
            return

        try:
            payload = json.loads(
                data.decode("utf-8")
            )
        except Exception:
            return

        usage = _extract_usage_dict(payload)

        event_type = (
            payload.get("type")
            if isinstance(payload, dict)
            else None
        )

        if event_type == "message_delta":
            # Streaming contract:
            # input/cache accounting belongs to the
            # request and is already known at
            # message_start. Some compatible relays
            # emit zero placeholders for these fields
            # again in message_delta. Never let those
            # overwrite the initial accounting.
            if "output_tokens" in usage:
                self._usage["output_tokens"] = (
                    usage["output_tokens"]
                )

            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                if (
                    key not in self._usage
                    and key in usage
                ):
                    self._usage[key] = usage[key]

            return

        self._merge_usage(usage)

    def feed(
        self,
        chunk: bytes,
    ) -> None:
        if not chunk:
            return

        if self._is_sse:
            self._line_buffer.extend(chunk)

            while True:
                newline = self._line_buffer.find(
                    b"\n"
                )

                if newline < 0:
                    break

                line = bytes(
                    self._line_buffer[:newline]
                )

                del self._line_buffer[
                    :newline + 1
                ]

                if len(line) > _MAX_LINE_BYTES:
                    self._oversize_lines += 1
                    continue

                self._inspect_sse_line(line)

            if (
                len(self._line_buffer)
                > _MAX_LINE_BYTES
            ):
                self._line_buffer.clear()
                self._oversize_lines += 1

            return

        if self._json_truncated:
            return

        remaining = (
            _MAX_JSON_BYTES
            - len(self._json_buffer)
        )

        if len(chunk) > remaining:
            self._json_buffer.clear()
            self._json_truncated = True
            return

        self._json_buffer.extend(chunk)

    def finish(self) -> dict[str, Any]:
        if self._is_sse:
            if self._line_buffer:
                line = bytes(self._line_buffer)
                self._line_buffer.clear()

                if len(line) <= _MAX_LINE_BYTES:
                    self._inspect_sse_line(line)
                else:
                    self._oversize_lines += 1

        elif (
            not self._json_truncated
            and self._json_buffer
        ):
            try:
                payload = json.loads(
                    self._json_buffer.decode(
                        "utf-8"
                    )
                )
            except Exception:
                payload = None

            if payload is not None:
                self._merge_usage(
                    _extract_usage_dict(payload)
                )

        result: dict[str, Any] = {
            "observed": bool(self._usage),
            "transport": (
                "sse"
                if self._is_sse
                else "json_or_other"
            ),
        }

        for key in _ALLOWED_USAGE_KEYS:
            if key in self._usage:
                result[key] = self._usage[key]

        if self._oversize_lines:
            result["oversize_lines"] = (
                self._oversize_lines
            )

        if self._json_truncated:
            result["json_truncated"] = True

        return result
