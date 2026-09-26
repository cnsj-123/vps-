from __future__ import annotations

import contextlib
import json
import os
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from _recall_fixtures import (
    CID,
    RID,
    bucket,
)


# Shared, minimal fixtures for the Memory Lifecycle shadow tests.
#
# Test-only. Not collected by unittest discovery (the module name does
# not match ``test*.py``).

LIFECYCLE_ENV = (
    "OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW"
)

# A fixed instants -- tests never sleep and never use the wall clock.
T0 = datetime(
    2026, 9, 26, 6, 44, 31, tzinfo=timezone.utc
)

M1 = "mem-1"
M2 = "mem-2"
M3 = "mem-3"

FORBIDDEN_METHODS = (
    "touch",
    "archive",
    "update",
    "delete",
    "create",
    "save",
    "write",
)


def iso(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def hours(count: float) -> timedelta:
    return timedelta(hours=count)


def usage_artifact(
    *,
    recall_id: str,
    conversation_id: str = CID,
    cognitive_request_id: str = RID,
    anchor_memory_id: str = M1,
    loaded_memory_ids: tuple[str, ...] = (
        M1,
    ),
    used_memory_ids: tuple[str, ...] = (),
    at: str | None = None,
    requested_at: str | None = None,
    extra_events: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    """A validated ``memory-usage-signal.v1`` artifact.

    ``used`` is always a subset of ``loaded`` and the event log
    mirrors the state arrays exactly, so the artifact passes the
    real Usage validator.
    """

    requested = requested_at or at or iso(T0)
    loaded_at = at or iso(T0)

    events: list[dict[str, Any]] = [
        {
            "stage": "recall_requested",
            "memory_id": anchor_memory_id,
            "at": requested,
        }
    ]

    for memory_id in loaded_memory_ids:
        events.append(
            {
                "stage": "memory_loaded",
                "memory_id": memory_id,
                "at": loaded_at,
            }
        )

    for memory_id in used_memory_ids:
        events.append(
            {
                "stage": "used",
                "memory_id": memory_id,
                "at": loaded_at,
            }
        )

    events.extend(
        dict(event) for event in extra_events
    )

    return {
        "version": "memory-usage-signal.v1",
        "mode": "shadow_only",
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "recall_id": recall_id,
        "anchor_memory_id": anchor_memory_id,
        "loaded_memory_ids": list(
            loaded_memory_ids
        ),
        "used_memory_ids": list(
            used_memory_ids
        ),
        "loaded_count": len(
            loaded_memory_ids
        ),
        "used_count": len(used_memory_ids),
        "events": events,
    }


def write_usage(
    root: str | Path,
    artifact: dict[str, Any],
    *,
    filename: str | None = None,
) -> Path:
    path = (
        Path(root)
        / "memory_usage"
        / artifact["conversation_id"]
        / artifact["cognitive_request_id"]
        / (
            (filename or artifact["recall_id"])
            + ".json"
        )
    )

    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    path.write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    return path


def source_bucket(
    memory_id: str,
    *,
    created: str | None = None,
    importance: int = 9,
    activation_count: int = 7,
    last_active: str = "2020-01-01T00:00:00Z",
    content: str | None = None,
    bucket_type: str = "dynamic",
) -> dict[str, Any]:
    """A canonical ``BucketManager.get()``-shaped source memory.

    It carries every legacy field the lifecycle shadow must NEVER
    write: ``activation_count`` / ``last_active`` / ``importance`` /
    ``type`` / ``score`` / ``metadata`` / ``content``.
    """

    item = bucket(
        memory_id,
        content
        if content is not None
        else ("body for " + memory_id),
        source_type=bucket_type,
    )

    metadata = item["metadata"]

    metadata["importance"] = importance
    metadata["activation_count"] = (
        activation_count
    )
    metadata["last_active"] = last_active

    if created is not None:
        metadata["created"] = created

    item["score"] = 0.87

    return item


def guarded_bucket_manager(
    buckets: dict[str, dict[str, Any]] | None = None,
) -> "GuardedBucketManager":
    """Canonical repository whose only legal call is ``get``.

    Every legacy mutation entry point raises, so a regression that
    starts calling ``touch`` / ``archive`` / ``update`` / ``delete``
    / ``create`` fails the test instead of silently mutating a
    canonical memory.
    """

    return GuardedBucketManager(buckets)


class GuardedBucketManager:
    def __init__(
        self,
        buckets: dict[str, dict[str, Any]] | None,
    ) -> None:
        self.buckets = dict(buckets or {})
        self.get_calls: list[str] = []

        for name in FORBIDDEN_METHODS:
            setattr(
                self,
                name,
                Mock(
                    side_effect=AssertionError(
                        "lifecycle called "
                        + name
                    )
                ),
            )

    async def get(
        self, bucket_id: str
    ) -> dict[str, Any] | None:
        self.get_calls.append(bucket_id)

        return self.buckets.get(bucket_id)


@contextlib.contextmanager
def lifecycle_env(
    root: str | Path,
    *,
    enabled: bool = True,
    **extra: str,
):
    """Bind the state dir and the lifecycle flag for one test."""

    env = {
        "OMBRE_CONTEXT_STATE_DIR": str(root),
        LIFECYCLE_ENV: (
            "1" if enabled else "0"
        ),
    }

    env.update(extra)

    with patch.dict(
        os.environ, env, clear=False
    ):
        yield


def tree(root: str | Path) -> dict[str, bytes]:
    """Every file under ``root`` as path -> bytes."""

    base = Path(root)

    if not base.is_dir():
        return {}

    return {
        str(
            path.relative_to(base)
        ): path.read_bytes()
        for path in sorted(
            base.rglob("*")
        )
        if path.is_file()
    }


def tree_without_lifecycle(
    root: str | Path,
) -> dict[str, bytes]:
    """The tree with the shadow-only lifecycle dirs removed."""

    return {
        name: payload
        for name, payload in tree(root).items()
        if not name.startswith(
            "memory_lifecycle_events/"
        )
        and not name.startswith(
            "memory_lifecycle_state/"
        )
    }