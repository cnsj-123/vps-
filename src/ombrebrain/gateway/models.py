from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SegmentKind(str, Enum):
    USER_TEXT = "user_text"
    DYNAMIC_CONTEXT = "dynamic_context"
    PERCEPTION = "perception"
    FINGERTIPS = "fingertips"
    FRONTEND_MEMORY = "frontend_memory"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CanonicalBlock:
    kind: SegmentKind
    source_message_index: int
    source_block_index: int
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass(frozen=True)
class CanonicalMessage:
    role: str
    original_index: int
    blocks: tuple[CanonicalBlock, ...]

    @property
    def kinds(self) -> tuple[SegmentKind, ...]:
        return tuple(block.kind for block in self.blocks)


@dataclass(frozen=True)
class CanonicalRequest:
    protocol: str
    system: Any
    history: tuple[CanonicalMessage, ...]
    current: CanonicalMessage | None
    passthrough: dict[str, Any]
    source_payload: dict[str, Any]

    def roundtrip_payload(self) -> dict[str, Any]:
        """
        Phase 2A is observation-only.

        Until mutation mode is explicitly enabled, the original request can
        always be reconstructed byte-semantically from this untouched payload.
        """
        return deepcopy(self.source_payload)
