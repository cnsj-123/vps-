from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class State:
    relationship: dict[str, float]
    current_focus: str | None
    last_seen: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relationship": dict(self.relationship),
            "current_focus": self.current_focus,
            "last_seen": self.last_seen,
        }

    @classmethod
    def empty(cls) -> "State":
        return cls(
            relationship={},
            current_focus=None,
            last_seen=None,
        )
