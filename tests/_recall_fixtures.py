from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch


# Shared, minimal fixtures for the Recall Control Plane tests.
#
# Test-only. Not collected by unittest discovery (the module name does
# not match ``test*.py``).

CID = "ctx_0123456789abcdef"
CID_B = "ctx_fedcba9876543210"

RID = "ctxreq_" + "a" * 32
RID_B = "ctxreq_" + "b" * 32

RECALL_ENABLED_ENV = (
    "OMBRE_GATEWAY_CONTEXT_RECALL_ENABLED"
)
RECALL_SHADOW_ENV = (
    "OMBRE_GATEWAY_CONTEXT_RELATED_RECALL_SHADOW"
)
RECALL_SURFACE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW"
)


def old_ts(days: int = 5) -> str:
    return (
        datetime.now(timezone.utc)
        - timedelta(days=days)
    ).isoformat()


def memory(
    memory_id: str,
    content: str | None = None,
    *,
    name: str | None = None,
    last_active: str | None = None,
    source_type: str | None = None,
) -> dict[str, Any]:
    """A retrieval-shaped memory candidate."""

    item: dict[str, Any] = {
        "id": memory_id,
        "content": (
            content
            if content is not None
            else ("content for " + memory_id)
        ),
        "context_relevance": 0.9,
        "metadata": {
            "last_active": (
                last_active
                if last_active is not None
                else old_ts()
            ),
        },
    }

    if name is not None:
        item["metadata"]["name"] = name

    if source_type is not None:
        item["metadata"]["type"] = source_type

    return item


def bucket(
    memory_id: str,
    content: str | None = None,
    *,
    name: str | None = None,
    source_type: str | None = None,
) -> dict[str, Any]:
    """A canonical BucketManager.get()-shaped memory."""

    metadata: dict[str, Any] = {
        "last_active": old_ts(),
    }

    if name is not None:
        metadata["name"] = name

    if source_type is not None:
        metadata["type"] = source_type

    return {
        "id": memory_id,
        "metadata": metadata,
        "content": (
            content
            if content is not None
            else ("content for " + memory_id)
        ),
        "path": f"/fake/buckets/{memory_id}.md",
    }


def flash_artifact(
    memories: list[dict[str, Any]],
    *,
    conversation_id: str = CID,
    cognitive_request_id: str = RID,
    revision: int = 5,
    decision: str = "surfaced",
) -> dict[str, Any]:
    flashes = []

    for index, item in enumerate(memories):
        cue = (
            item.get("metadata", {}).get(
                "name"
            )
            or str(item.get("content") or "")
        )[:160]

        flash: dict[str, Any] = {
            "memory_id": item.get("id"),
            "cue": cue,
            "reason": "surfacing_eligible",
            "estimated_tokens": 1,
            "rank": index + 1,
        }

        source_type = (
            item.get("metadata", {}).get("type")
        )

        if source_type:
            flash["source_type"] = source_type

        flashes.append(flash)

    return {
        "version": "memory-flash.v1",
        "mode": "shadow_only",
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "source_unified_revision": revision,
        "source_confidence_revision": 1,
        "decision": decision,
        "reason": (
            "flash_surfaced"
            if flashes
            else "no_eligible_candidates"
        ),
        "retrieved_candidate_count": len(
            memories
        ),
        "eligible_candidate_count": len(
            memories
        ),
        "surfaced_count": len(flashes),
        "estimated_tokens": len(flashes),
        "token_budget": 96,
        "max_items": 3,
        "max_chars": 160,
        "flashes": flashes,
    }


def write_recall(
    root: str | Path,
    artifact: dict[str, Any],
) -> Path:
    path = (
        Path(root)
        / "related_recall"
        / artifact["conversation_id"]
        / artifact["cognitive_request_id"]
        / (artifact["recall_id"] + ".json")
    )

    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    path.write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    return path


def write_flash(
    root: str | Path,
    artifact: dict[str, Any],
) -> Path:
    path = (
        Path(root)
        / "memory_flash"
        / artifact["conversation_id"]
        / (
            artifact["cognitive_request_id"]
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


def recall_request(
    anchor_memory_id: str = "mem-1",
    *,
    conversation_id: str = CID,
    cognitive_request_id: str = RID,
    revision: int = 5,
    requested_scope: str = "related",
) -> dict[str, Any]:
    """A validated ``memory-recall-request.v1`` artifact shape."""

    return {
        "version": "memory-recall-request.v1",
        "mode": "shadow_only",
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "anchor_memory_id": anchor_memory_id,
        "source_flash_request_id":
            cognitive_request_id,
        "source_flash_unified_revision":
            revision,
        "requested_scope": requested_scope,
        "created_at": old_ts(0),
    }


def recall_artifact(
    memory_ids: list[str],
    *,
    conversation_id: str = CID,
    cognitive_request_id: str = RID,
    anchor_memory_id: str | None = None,
    recall_id: str = "recall_" + "0" * 32,
    revision: int = 5,
) -> dict[str, Any]:
    """A ``related-memory-recall.v1`` artifact shape."""

    anchor = (
        anchor_memory_id
        if anchor_memory_id is not None
        else (memory_ids[0] if memory_ids else "mem-1")
    )

    return {
        "version": "related-memory-recall.v1",
        "mode": "shadow_only",
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "recall_id": recall_id,
        "anchor_memory_id": anchor,
        "source_flash_request_id":
            cognitive_request_id,
        "source_flash_unified_revision":
            revision,
        "requested_scope": "related",
        "request_fingerprint": "f" * 64,
        "created_at": old_ts(0),
        "retrieved_count": len(memory_ids),
        "included_count": len(memory_ids),
        "estimated_tokens": len(memory_ids),
        "token_budget": 512,
        "memories": [
            {
                "memory_id": memory_id,
                "content": "body " + memory_id,
                "reason": (
                    "anchor_memory"
                    if index == 0
                    else "related_memory"
                ),
                "rank": index + 1,
            }
            for index, memory_id in enumerate(
                memory_ids
            )
        ],
    }


class FakeBucketManager:
    """Canonical ``BucketManager.get`` surface, no filesystem writes."""

    def __init__(
        self,
        buckets: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.buckets = dict(buckets or {})
        self.get_calls: list[str] = []

    async def get(
        self, bucket_id: str
    ) -> dict[str, Any] | None:
        self.get_calls.append(bucket_id)

        return self.buckets.get(bucket_id)


class FakeRetrievalAdapter:
    """Canonical ``retrieve_with_observation`` surface."""

    def __init__(
        self,
        results: list[dict[str, Any]] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self.results = list(results or [])
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    async def retrieve_with_observation(
        self,
        query: str,
        *,
        max_results: int = 8,
        **_: Any,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        self.calls.append((query, max_results))

        if self.raises is not None:
            raise self.raises

        selected = self.results[:max_results]

        return (
            list(selected),
            {
                "candidates": list(
                    self.results
                ),
                "vector_scores": {},
                "telemetry": {},
            },
        )


@contextlib.contextmanager
def recall_env(
    root: str | Path,
    *,
    enabled: bool = True,
    shadow: bool = True,
    surface: bool = False,
    **extra: str,
):
    env = {
        "OMBRE_CONTEXT_STATE_DIR": str(root),
        RECALL_ENABLED_ENV: (
            "1" if enabled else "0"
        ),
        RECALL_SHADOW_ENV: (
            "1" if shadow else "0"
        ),
        RECALL_SURFACE_ENV: (
            "1" if surface else "0"
        ),
    }

    env.update(extra)

    with patch.dict(
        os.environ, env, clear=False
    ):
        yield