from __future__ import annotations

from dataclasses import replace
from typing import Any

from .models import State
from .store import StateStore


class StateService:
    ALLOWED_FIELDS = frozenset({
        "relationship",
        "current_focus",
        "last_seen",
    })

    RELATIONSHIP_FIELDS = frozenset({
        "closeness",
        "trust",
    })

    MAX_DELTA = 0.05

    def __init__(self, store: StateStore):
        self.store = store

    def get(self) -> tuple[State, int]:
        return self.store.get_state()

    def update(
        self,
        patch: dict[str, Any],
        *,
        changed_by: str = "system",
        expected_revision: int | None = None,
    ) -> tuple[State, int]:
        unknown = set(patch) - self.ALLOWED_FIELDS
        if unknown:
            raise ValueError(
                f"unsupported state fields: {sorted(unknown)}"
            )

        current, revision = self.store.get_state()

        if expected_revision is not None and revision != expected_revision:
            raise ValueError(
                f"state revision conflict: expected {expected_revision}, actual {revision}"
            )

        relationship = dict(current.relationship)

        if "relationship" in patch:
            incoming = patch["relationship"]

            if not isinstance(incoming, dict):
                raise ValueError("relationship must be an object")

            unknown_relation = set(incoming) - self.RELATIONSHIP_FIELDS
            if unknown_relation:
                raise ValueError(
                    f"unsupported relationship fields: {sorted(unknown_relation)}"
                )

            for key, value in incoming.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(
                        f"relationship.{key} must be numeric"
                    )

                value = float(value)

                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"relationship.{key} must be between 0 and 1"
                    )

                previous = float(relationship.get(key, value))

                if key in relationship and abs(value - previous) > self.MAX_DELTA + 1e-12:
                    raise ValueError(
                        f"relationship.{key} change exceeds {self.MAX_DELTA}"
                    )

                relationship[key] = value

        current_focus = current.current_focus
        if "current_focus" in patch:
            value = patch["current_focus"]
            if value is not None and not isinstance(value, str):
                raise ValueError("current_focus must be a string or null")
            current_focus = value

        last_seen = current.last_seen
        if "last_seen" in patch:
            value = patch["last_seen"]
            if value is not None and not isinstance(value, str):
                raise ValueError("last_seen must be a string or null")
            last_seen = value

        new_state = replace(
            current,
            relationship=relationship,
            current_focus=current_focus,
            last_seen=last_seen,
        )

        new_revision = self.store.save_state(
            new_state,
            changed_by=changed_by,
            expected_revision=revision,
        )

        return new_state, new_revision
