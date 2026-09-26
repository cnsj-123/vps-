from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ombrebrain.context.memory_flash_live_exposure import (
    render_memory_flash,
    select_live_memory_flash_body,
)
from ombrebrain.context.memory_recall_surface import (
    recall_surface_for_model,
    update_recall_surface,
)
from ombrebrain.context.recall_request import (
    recall_request_fingerprint,
)
from ombrebrain.context.recall_types import (
    estimate_tokens,
    memory_usage_path,
    recall_request_path,
    related_recall_path,
)


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

    fingerprint = recall_request_fingerprint(
        conversation_id=conversation_id,
        cognitive_request_id=cognitive_request_id,
        anchor_memory_id=anchor,
        requested_scope="related",
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
        "request_fingerprint": fingerprint,
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


# ------------------------------------------------------
# Provider-bound Live Recall transport fixtures
# ------------------------------------------------------

FLASH_SHADOW_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW"
)
SURFACE_SHADOW_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW"
)
LIVE_EXPOSURE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_LIVE_EXPOSURE"
)
LIVE_BUDGET_ENV = (
    "OMBRE_MEMORY_FLASH_LIVE_TOKEN_BUDGET"
)

CAPABILITY_TTL_ENV = (
    "OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_CAPABILITY_TTL_SECONDS"
)

TRANSPORT_ENV = (
    "OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_TRANSPORT"
)
TRANSPORT_PROVIDER_ENV = (
    "OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_PROVIDER"
)
TRANSPORT_MCP_URL_ENV = (
    "OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_MCP_URL"
)
BRIDGE_ENV = (
    "OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_BRIDGE"
)


def anthropic_payload() -> dict[str, Any]:
    """A minimal Messages-shaped body with a cache boundary before the
    current user message (the shape Live Exposure requires)."""

    return {
        "model": "test-model",
        "system": [
            {
                "type": "text",
                "text": "system-secret",
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": "1h",
                },
            }
        ],
        "tools": [
            {
                "name": "test_tool",
                "description": "tool-secret",
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "old-user",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "old-assistant",
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
                        "text": "current-user-secret",
                    }
                ],
            },
        ],
        "stream": True,
    }


def anthropic_body() -> bytes:
    return json.dumps(
        anthropic_payload(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def seed_live_exposure(
    root: str | Path,
    memories: list[dict[str, Any]],
    *,
    conversation_id: str = CID,
    cognitive_request_id: str = RID,
    revision: int = 5,
    exposed_count: int | None = None,
    body: bytes | None = None,
) -> list[str]:
    """Seed Flash + Recall Surface + a valid Live Exposure receipt.

    Returns the ordered surface memrefs. The live token budget is
    pinned to the exact cost of the exposed items so later items
    cannot fit.
    """

    write_flash(
        root,
        flash_artifact(
            memories,
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            revision=revision,
        ),
    )

    with patch.dict(
        os.environ,
        {
            "OMBRE_CONTEXT_STATE_DIR": str(root),
            FLASH_SHADOW_ENV: "1",
            SURFACE_SHADOW_ENV: "1",
            LIVE_EXPOSURE_ENV: "1",
        },
        clear=False,
    ):
        update_recall_surface(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            flash_report=flash_artifact(
                memories,
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
                revision=revision,
            ),
        )

        surfaces = recall_surface_for_model(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
        )["surfaces"]

        memrefs = [
            surface["memref"]
            for surface in surfaces
        ]

        cues = [
            surface["cue"] for surface in surfaces
        ]

        count = (
            len(memrefs)
            if exposed_count is None
            else exposed_count
        )

        items = [
            {
                "memref": memrefs[index],
                "cue": cues[index],
            }
            for index in range(count)
        ]

        budget = estimate_tokens(
            render_memory_flash(items)
        )

        with patch.dict(
            os.environ,
            {LIVE_BUDGET_ENV: str(budget)},
            clear=False,
        ):
            select_live_memory_flash_body(
                body if body is not None else anthropic_body(),
                conversation_id=conversation_id,
                cognitive_request_id=(
                    cognitive_request_id
                ),
            )

    return memrefs


# ------------------------------------------------------
# Explicit Memory Usage Attribution fixtures
# ------------------------------------------------------

USAGE_ATTRIBUTION_ENV = (
    "OMBRE_GATEWAY_CONTEXT_USAGE_ATTRIBUTION"
)
USAGE_REFS_MAX_ENV = (
    "OMBRE_GATEWAY_CONTEXT_USAGE_REFS_MAX"
)
USAGE_REFS_TOKEN_BUDGET_ENV = (
    "OMBRE_GATEWAY_CONTEXT_USAGE_REFS_TOKEN_BUDGET"
)
LIFECYCLE_SHADOW_ENV = (
    "OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW"
)

DEFAULT_RECALL_ID = "recall_" + "1" * 32


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def projected_model_result(
    memory_ids: list[str],
    *,
    kinds: list[str] | None = None,
    content_prefix: str = "recalled body ",
) -> dict[str, Any]:
    """A validated ``memory-live-recall-result.v1`` model projection.

    Ranks are sequential from 1 and the first kind must be ``anchor``.
    Pass a SHORT ``memory_ids`` list to model the bounded projection
    dropping the tail: those loaded items were never output to the
    model and therefore can never obtain a ``useref``.
    """

    memories = []

    for index, memory_id in enumerate(memory_ids):
        kind = (
            kinds[index]
            if kinds is not None
            else (
                "anchor"
                if index == 0
                else "related"
            )
        )

        memories.append(
            {
                "rank": index + 1,
                "kind": kind,
                "content": content_prefix + memory_id,
            }
        )

    return {
        "version": "memory-live-recall-result.v1",
        "mode": "trusted_live_bridge",
        "status": "recalled",
        "reason": "recall_completed",
        "memory_count": len(memories),
        "memories": memories,
    }


def seed_recall_control_plane(
    root: str | Path,
    memory_ids: list[str],
    *,
    conversation_id: str = CID,
    cognitive_request_id: str = RID,
    recall_id: str = DEFAULT_RECALL_ID,
    revision: int = 5,
    loaded_memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Write a validated Recall Request + Related Recall + Usage Signal.

    Nothing is recorded as ``used``: the seeded Usage Signal holds
    exactly ``recall_requested`` + ``memory_loaded`` events, so the
    tests can prove that only an explicit attribution moves ``used``.

    ``loaded_memory_ids`` may be a SUBSET of ``memory_ids`` (an item
    that the Recall loaded into the artifact but that the bounded
    model projection must never be able to attribute).
    """

    anchor = memory_ids[0] if memory_ids else "mem-1"

    request = recall_request(
        anchor,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        revision=revision,
    )

    request["recall_id"] = recall_id

    request["request_fingerprint"] = (
        recall_request_fingerprint(
            conversation_id=conversation_id,
            cognitive_request_id=(
                cognitive_request_id
            ),
            anchor_memory_id=anchor,
            requested_scope="related",
        )
    )

    request["status"] = "requested"

    artifact = recall_artifact(
        memory_ids,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
        revision=revision,
    )

    loaded = (
        list(memory_ids)
        if loaded_memory_ids is None
        else list(loaded_memory_ids)
    )

    at = old_ts(0)

    usage = {
        "version": "memory-usage-signal.v1",
        "mode": "shadow_only",
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "recall_id": recall_id,
        "anchor_memory_id": anchor,
        "requested_at": at,
        "loaded_memory_ids": loaded,
        "used_memory_ids": [],
        "loaded_count": len(loaded),
        "used_count": 0,
        "events": [
            {
                "stage": "recall_requested",
                "memory_id": anchor,
                "at": at,
            }
        ]
        + [
            {
                "stage": "memory_loaded",
                "memory_id": memory_id,
                "at": at,
            }
            for memory_id in loaded
        ],
    }

    with patch.dict(
        os.environ,
        {"OMBRE_CONTEXT_STATE_DIR": str(root)},
        clear=False,
    ):
        _write_json(
            recall_request_path(
                conversation_id,
                cognitive_request_id,
                recall_id,
            ),
            request,
        )

        _write_json(
            related_recall_path(
                conversation_id,
                cognitive_request_id,
                recall_id,
            ),
            artifact,
        )

        _write_json(
            memory_usage_path(
                conversation_id,
                cognitive_request_id,
                recall_id,
            ),
            usage,
        )

    return {
        "recall_id": recall_id,
        "anchor_memory_id": anchor,
        "request_fingerprint":
            request["request_fingerprint"],
        "memory_ids": list(memory_ids),
        "loaded_memory_ids": loaded,
        "artifact": artifact,
    }


def read_usage(
    root: str | Path,
    *,
    conversation_id: str = CID,
    cognitive_request_id: str = RID,
    recall_id: str = DEFAULT_RECALL_ID,
) -> dict[str, Any] | None:
    """Read one persisted Usage Signal artifact, raw."""

    path = (
        Path(root)
        / "memory_usage"
        / conversation_id
        / cognitive_request_id
        / (recall_id + ".json")
    )

    if not path.is_file():
        return None

    return json.loads(
        path.read_text(encoding="utf-8")
    )