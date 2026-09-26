from __future__ import annotations

import os
from typing import Any

from ombrebrain.context import memory_usage_signal
from ombrebrain.context import memory_recall_surface
from ombrebrain.context.recall_request import (
    RECALL_REQUEST_VERSION,
    authorize_recall_request,
    new_recall_id,
    recall_control_plane_enabled,
    recall_request_fingerprint,
)
from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    bound_text,
    estimate_tokens,
    is_valid_recall_id,
    normalize_key,
    now_iso,
    parse_positive_int,
    read_json,
    related_recall_dir,
    related_recall_path,
)
from ombrebrain.context.unified_context_candidate import (
    bound_context_service,
)


# Related Memory Recall v1 — bounded neighborhood, shadow only.
#
# Flash creation:        NO re-retrieval (a cue is not a new search).
# Explicit Recall:       YES, it may run canonical retrieval once.
#
# This module is the ONLY place a Recall Request may load memory
# content, and it is deliberately narrow:
#
#   - it reuses the existing canonical retrieval
#     (``ContextRetrievalAdapter.retrieve_with_observation``) exactly
#     once -- there is no second retrieval engine, no second scorer and
#     no hidden retrieval loop;
#   - it loads the anchor through the canonical bucket repository
#     (``BucketManager.get``) -- it never builds a filesystem path from
#     an id;
#   - it returns a very small neighborhood, hard-bounded by item cap,
#     token budget and per-memory character cap;
#   - the anchor is first, related candidates follow canonical order;
#   - current-user echo, duplicate ids and exact duplicate content are
#     removed;
#   - it never mutates a memory source, never reinforces anything and
#     never calls a model.
#
# A Recall result is NOT injected back into Unified / Preview. It is a
# separate artifact the AI may or may not use.

_VERSION = "related-memory-recall.v1"
_MODE = "shadow_only"

_DEFAULT_MAX_RELATED_MEMORIES = 4
_MAX_RELATED_MEMORIES_CAP = 8

_DEFAULT_TOKEN_BUDGET = 512
_TOKEN_BUDGET_CAP = 1200

_DEFAULT_MAX_CHARS_PER_MEMORY = 1200
_MAX_CHARS_PER_MEMORY_CAP = 3000

# Deterministic query construction bounds (no model call, no free-form
# model query).
_ANCHOR_QUERY_CHARS = 400
_USER_QUERY_CHARS = 300

_REASON_INVALID_REQUEST = "invalid_recall_request"
_REASON_REF_NOT_FOUND = "recall_ref_not_found"
_REASON_ANCHOR_UNAVAILABLE = (
    "anchor_memory_unavailable"
)
_REASON_DISABLED = "recall_disabled"
_REASON_UNAVAILABLE = "recall_unavailable"
_REASON_CONTEXT_UNAVAILABLE = (
    "context_unavailable"
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def resolve_related_recall_budget(
    *,
    max_related_memories: Any = None,
    token_budget: Any = None,
    max_chars_per_memory: Any = None,
) -> dict[str, int]:
    """Related Recall-local caps. Conservative and always bounded.

    These caps are Recall-local only: they never change the canonical
    retrieval weights, thresholds, recency, ranking or the normal
    Context retrieval result count.
    """

    if max_related_memories is None:
        max_related_memories = os.environ.get(
            "OMBRE_RELATED_RECALL_MAX_MEMORIES"
        )

    if token_budget is None:
        token_budget = os.environ.get(
            "OMBRE_RELATED_RECALL_TOKEN_BUDGET"
        )

    if max_chars_per_memory is None:
        max_chars_per_memory = os.environ.get(
            "OMBRE_RELATED_RECALL_MAX_CHARS"
        )

    return {
        "max_related_memories":
            parse_positive_int(
                max_related_memories,
                default=(
                    _DEFAULT_MAX_RELATED_MEMORIES
                ),
                cap=(
                    _MAX_RELATED_MEMORIES_CAP
                ),
            ),
        "token_budget":
            parse_positive_int(
                token_budget,
                default=_DEFAULT_TOKEN_BUDGET,
                cap=_TOKEN_BUDGET_CAP,
            ),
        "max_chars_per_memory":
            parse_positive_int(
                max_chars_per_memory,
                default=(
                    _DEFAULT_MAX_CHARS_PER_MEMORY
                ),
                cap=(
                    _MAX_CHARS_PER_MEMORY_CAP
                ),
            ),
    }


def build_recall_query(
    *,
    anchor_representation: Any,
    current_query: Any,
    anchor_chars: int = _ANCHOR_QUERY_CHARS,
    user_chars: int = _USER_QUERY_CHARS,
) -> str:
    """Deterministic, bounded recall query. No model, no free-form.

    Only two inputs are used: the anchor memory's already-existing
    compact representation and THIS request's current user query.
    The system prompt, the full transcript, the memory store and the
    Flash artifact JSON are never part of it.
    """

    anchor_text = bound_text(
        anchor_representation,
        max_chars=anchor_chars,
    )

    query_text = bound_text(
        current_query,
        max_chars=user_chars,
    )

    return "\n".join(
        part
        for part in (anchor_text, query_text)
        if part
    )


def _anchor_representation(
    bucket: Any,
) -> str:
    """metadata.name first, then content. Nothing is generated."""

    if not isinstance(bucket, dict):
        return ""

    metadata = bucket.get("metadata")

    if isinstance(metadata, dict):
        name = bound_text(
            metadata.get("name"),
            max_chars=_ANCHOR_QUERY_CHARS,
        )

        if name:
            return name

    return bound_text(
        bucket.get("content"),
        max_chars=_ANCHOR_QUERY_CHARS,
    )


def _source_type(
    item: Any,
) -> str:
    if not isinstance(item, dict):
        return ""

    metadata = item.get("metadata")

    if not isinstance(metadata, dict):
        return ""

    value = metadata.get("type")

    if (
        isinstance(value, str)
        and value.strip()
    ):
        return value.strip()

    return ""


def _fit_content(
    raw: Any,
    *,
    max_chars: int,
    remaining_tokens: int,
) -> tuple[str, int]:
    """Bound one memory's content to the char cap AND remaining tokens.

    Returns ("", 0) when nothing fits. The result can never exceed
    the per-memory character cap or the remaining token budget.
    """

    text = bound_text(raw, max_chars=max_chars)

    if not text or remaining_tokens < 1:
        return "", 0

    cost = estimate_tokens(text)

    if cost <= remaining_tokens:
        return text, cost

    # Truncate so the heuristic cost fits the remaining budget. The
    # token heuristic is ceil(len / 3), so room * 3 characters is a
    # safe upper bound; re-check to stay conservative.
    room = max(1, remaining_tokens * 3)

    text = text[:room]

    cost = estimate_tokens(text)

    if cost > remaining_tokens:
        return "", 0

    return text, cost


def _retrieved_results(
    results: Any,
) -> list[dict[str, Any]]:
    if not isinstance(results, list):
        return []

    return [
        item
        for item in results
        if isinstance(item, dict)
    ]


def _refusal(
    reason: str,
    *,
    mode: str = _MODE,
) -> dict[str, Any]:
    return {
        "version": _VERSION,
        "mode": mode,
        "stored": False,
        "duplicate": False,
        "decision": "refused",
        "reason": reason,
        "recall_id": None,
        "retrieved_count": 0,
        "included_count": 0,
        "estimated_tokens": 0,
        "token_budget": None,
        "recall": None,
    }


async def build_related_recall(
    *,
    recall_request: Any,
    current_query: Any = "",
    bucket_manager: Any = None,
    retrieval_adapter: Any = None,
    budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build (do not persist) one bounded related recall artifact.

    Never raises: malformed input, an unavailable anchor or a failing
    retrieval degrades to a structured refusal, never to an exception.
    """

    if not isinstance(recall_request, dict):
        return _refusal(_REASON_INVALID_REQUEST)

    conversation_id = recall_request.get(
        "conversation_id"
    )

    cognitive_request_id = (
        recall_request.get(
            "cognitive_request_id"
        )
    )

    anchor_memory_id = recall_request.get(
        "anchor_memory_id"
    )

    if (
        recall_request.get("version")
        != RECALL_REQUEST_VERSION
        or recall_request.get("mode") != _MODE
        or not isinstance(
            conversation_id, str
        )
        or not isinstance(
            cognitive_request_id, str
        )
        or not isinstance(
            anchor_memory_id, str
        )
        or not anchor_memory_id.strip()
    ):
        return _refusal(_REASON_INVALID_REQUEST)

    anchor_memory_id = anchor_memory_id.strip()

    resolved = resolve_related_recall_budget(
        max_related_memories=(
            budget.get(
                "max_related_memories"
            )
            if isinstance(budget, dict)
            else None
        ),
        token_budget=(
            budget.get("token_budget")
            if isinstance(budget, dict)
            else None
        ),
        max_chars_per_memory=(
            budget.get(
                "max_chars_per_memory"
            )
            if isinstance(budget, dict)
            else None
        ),
    )

    max_related = resolved[
        "max_related_memories"
    ]
    token_budget = resolved["token_budget"]
    max_chars = resolved[
        "max_chars_per_memory"
    ]

    # --------------------------------------------------
    # 1. Anchor through the canonical repository
    # --------------------------------------------------

    anchor_bucket = None

    getter = getattr(
        bucket_manager, "get", None
    )

    if callable(getter):
        try:
            anchor_bucket = await getter(
                anchor_memory_id
            )
        except Exception:
            anchor_bucket = None

    if not isinstance(anchor_bucket, dict):
        # Deterministic refusal. Never silently substitutes another
        # memory for a missing anchor.
        return _refusal(
            _REASON_ANCHOR_UNAVAILABLE
        )

    # --------------------------------------------------
    # 2. One canonical retrieval for the related neighborhood
    # --------------------------------------------------

    query = build_recall_query(
        anchor_representation=(
            _anchor_representation(
                anchor_bucket
            )
        ),
        current_query=current_query,
    )

    results: list[dict[str, Any]] = []

    if query and retrieval_adapter is not None:
        retrieve = getattr(
            retrieval_adapter,
            "retrieve_with_observation",
            None,
        )

        if callable(retrieve):
            try:
                raw_results, _observation = (
                    await retrieve(
                        query,
                        max_results=max_related,
                    )
                )

                results = _retrieved_results(
                    raw_results
                )
            except Exception:
                results = []

    retrieved_count = len(results)

    # --------------------------------------------------
    # 3. Compose: anchor first, then canonical order
    # --------------------------------------------------

    memories: list[dict[str, Any]] = []
    used_tokens = 0

    echo_key = normalize_key(
        current_query
    ) if isinstance(
        current_query, str
    ) and current_query.strip() else ""

    seen_ids: set[str] = {
        anchor_memory_id
    }

    seen_text: set[str] = set()

    anchor_content, anchor_cost = (
        _fit_content(
            anchor_bucket.get("content"),
            max_chars=max_chars,
            remaining_tokens=(
                token_budget - used_tokens
            ),
        )
    )

    if not anchor_content:
        return _refusal(
            _REASON_ANCHOR_UNAVAILABLE
        )

    anchor_item: dict[str, Any] = {
        "memory_id": anchor_memory_id,
        "content": anchor_content,
        "reason": "anchor_memory",
        "rank": 1,
    }

    anchor_source_type = _source_type(
        anchor_bucket
    )

    if anchor_source_type:
        anchor_item["source_type"] = (
            anchor_source_type
        )

    memories.append(anchor_item)

    used_tokens += anchor_cost

    seen_text.add(
        normalize_key(anchor_content)
    )

    duplicates = 0
    echo_rejected = 0
    budget_rejected = 0

    for candidate in results:
        if len(memories) >= max_related:
            break

        memory_id = candidate.get("id")

        if (
            not isinstance(memory_id, str)
            or not memory_id.strip()
        ):
            continue

        memory_id = memory_id.strip()

        # Same stable id as the anchor or an already included memory.
        if memory_id in seen_ids:
            duplicates += 1
            continue

        text_key = normalize_key(
            candidate.get("content")
        ) if isinstance(
            candidate.get("content"), str
        ) else ""

        if text_key and text_key in seen_text:
            duplicates += 1
            continue

        # Never wrap the current user message itself as a past memory.
        if echo_key and text_key == echo_key:
            echo_rejected += 1
            continue

        content, cost = _fit_content(
            candidate.get("content"),
            max_chars=max_chars,
            remaining_tokens=(
                token_budget - used_tokens
            ),
        )

        if not content:
            budget_rejected += 1
            continue

        item: dict[str, Any] = {
            "memory_id": memory_id,
            "content": content,
            "reason": "related_memory",
            "rank": len(memories) + 1,
        }

        relevance = candidate.get(
            "context_relevance"
        )

        if (
            isinstance(relevance, (int, float))
            and not isinstance(relevance, bool)
        ):
            item["context_relevance"] = round(
                float(relevance), 4
            )

        source_type = _source_type(candidate)

        if source_type:
            item["source_type"] = source_type

        memories.append(item)

        seen_ids.add(memory_id)

        if text_key:
            seen_text.add(text_key)

        used_tokens += cost

    return {
        "stored": True,
        "version": _VERSION,
        "mode": _MODE,
        "conversation_id": conversation_id,
        "cognitive_request_id":
            cognitive_request_id,
        "recall_id": new_recall_id(),
        "anchor_memory_id": anchor_memory_id,
        "source_flash_request_id":
            recall_request.get(
                "source_flash_request_id"
            ),
        "source_flash_unified_revision":
            recall_request.get(
                "source_flash_unified_revision"
            ),
        "requested_scope":
            recall_request.get(
                "requested_scope"
            ),
        "request_fingerprint":
            recall_request_fingerprint(
                conversation_id=(
                    conversation_id
                ),
                cognitive_request_id=(
                    cognitive_request_id
                ),
                anchor_memory_id=(
                    anchor_memory_id
                ),
                requested_scope=(
                    recall_request.get(
                        "requested_scope"
                    )
                    or ""
                ),
            ),
        "created_at": now_iso(),
        "retrieved_count": retrieved_count,
        "included_count": len(memories),
        "estimated_tokens": used_tokens,
        "token_budget": token_budget,
        "max_related_memories": max_related,
        "max_chars_per_memory": max_chars,
        "duplicate_rejected": duplicates,
        "echo_rejected": echo_rejected,
        "budget_rejected": budget_rejected,
        "memories": memories,
    }


def _find_by_fingerprint(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    """Existing recall with the same request fingerprint, if any.

    Called under ``LOCK`` only, so two concurrent retries of the same
    Recall Request cannot both create an artifact.
    """

    try:
        directory = related_recall_dir(
            conversation_id,
            cognitive_request_id,
        )
    except ValueError:
        return None

    if not directory.is_dir():
        return None

    for path in sorted(
        directory.glob("recall_*.json")
    ):
        artifact = read_json(path)

        if not isinstance(artifact, dict):
            continue

        if (
            artifact.get("version") == _VERSION
            and artifact.get(
                "request_fingerprint"
            )
            == fingerprint
        ):
            return artifact

    return None


def _duplicate_report(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": _VERSION,
        "mode": _MODE,
        "stored": True,
        "duplicate": True,
        "decision": "recalled",
        "reason": "duplicate_recall_request",
        "recall_id": artifact.get(
            "recall_id"
        ),
        "retrieved_count": artifact.get(
            "retrieved_count"
        ),
        "included_count": artifact.get(
            "included_count"
        ),
        "estimated_tokens": artifact.get(
            "estimated_tokens"
        ),
        "token_budget": artifact.get(
            "token_budget"
        ),
        "recall": artifact,
    }


def _recall_report(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": _VERSION,
        "mode": _MODE,
        "stored": True,
        "duplicate": False,
        "decision": "recalled",
        "reason": "recall_completed",
        "recall_id": artifact.get(
            "recall_id"
        ),
        "retrieved_count": artifact.get(
            "retrieved_count"
        ),
        "included_count": artifact.get(
            "included_count"
        ),
        "estimated_tokens": artifact.get(
            "estimated_tokens"
        ),
        "token_budget": artifact.get(
            "token_budget"
        ),
        "recall": artifact,
    }


def _resolve_dependencies(
    *,
    bucket_manager: Any,
    retrieval_adapter: Any,
) -> tuple[Any, Any]:
    """Use the explicitly supplied canonical deps, else the bound
    ContextService. Never creates a second retrieval engine."""

    service = None

    if (
        bucket_manager is None
        or retrieval_adapter is None
    ):
        service = bound_context_service()

    if bucket_manager is None and service is not None:
        bucket_manager = getattr(
            service, "bucket_mgr", None
        )

    if (
        retrieval_adapter is None
        and service is not None
    ):
        retrieval_adapter = getattr(
            service, "retrieval", None
        )

    return bucket_manager, retrieval_adapter


async def request_related_recall(
    *,
    conversation_id: Any,
    cognitive_request_id: Any,
    anchor_memory_id: Any = None,
    anchor_ref: Any = None,
    requested_scope: Any = "related",
    source_flash_unified_revision: Any = None,
    current_query: Any = "",
    bucket_manager: Any = None,
    retrieval_adapter: Any = None,
    budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    """The single Recall control-plane entry point.

    This is the minimal public API the next phase's thin provider
    adapter needs. Live model trigger wiring is deferred: nothing
    calls this automatically, and a Memory Flash never triggers it.

    The trusted caller binds ``conversation_id`` and
    ``cognitive_request_id`` (the model may never pass them), and
    supplies the current user query from the real request. The model
    may only ever name an anchor, and only through an opaque
    ``memref`` (``anchor_ref``) or the already-surfaced memory id.

    Fail-open: any failure returns a structured refusal and never
    raises, never blocks a request and never changes a model
    response.
    """

    try:
        if not recall_control_plane_enabled():
            return _refusal(_REASON_DISABLED)

        anchor = anchor_memory_id

        if anchor_ref is not None:
            # Opaque, request-scoped reference. A memref from another
            # request, another conversation or a guess resolves to
            # None and is refused.
            anchor = (
                memory_recall_surface.resolve_recall_ref(
                    conversation_id=(
                        conversation_id
                    ),
                    cognitive_request_id=(
                        cognitive_request_id
                    ),
                    memref=anchor_ref,
                )
            )

            if anchor is None:
                return _refusal(
                    _REASON_REF_NOT_FOUND
                )

        authorization = (
            authorize_recall_request(
                conversation_id=(
                    conversation_id
                ),
                cognitive_request_id=(
                    cognitive_request_id
                ),
                anchor_memory_id=anchor,
                requested_scope=(
                    requested_scope
                ),
                source_flash_unified_revision=(
                    source_flash_unified_revision
                ),
            )
        )

        if not authorization.get("valid"):
            return _refusal(
                authorization.get("reason")
                or _REASON_INVALID_REQUEST
            )

        recall_request = authorization[
            "recall_request"
        ]

        fingerprint = (
            recall_request_fingerprint(
                conversation_id=(
                    recall_request[
                        "conversation_id"
                    ]
                ),
                cognitive_request_id=(
                    recall_request[
                        "cognitive_request_id"
                    ]
                ),
                anchor_memory_id=(
                    recall_request[
                        "anchor_memory_id"
                    ]
                ),
                requested_scope=(
                    recall_request[
                        "requested_scope"
                    ]
                ),
            )
        )

        # Idempotency: a provider retry of the SAME recall request
        # folds onto the existing artifact and never double-counts
        # ``recall_requested``.
        with LOCK:
            existing = _find_by_fingerprint(
                conversation_id=(
                    recall_request[
                        "conversation_id"
                    ]
                ),
                cognitive_request_id=(
                    recall_request[
                        "cognitive_request_id"
                    ]
                ),
                fingerprint=fingerprint,
            )

        if isinstance(existing, dict):
            return _duplicate_report(
                existing
            )

        resolved_bucket, resolved_retrieval = (
            _resolve_dependencies(
                bucket_manager=bucket_manager,
                retrieval_adapter=(
                    retrieval_adapter
                ),
            )
        )

        if resolved_bucket is None:
            return _refusal(
                _REASON_CONTEXT_UNAVAILABLE
            )

        artifact = await build_related_recall(
            recall_request=recall_request,
            current_query=current_query,
            bucket_manager=resolved_bucket,
            retrieval_adapter=(
                resolved_retrieval
            ),
            budget=budget,
        )

        if artifact.get("stored") is not True:
            # build_related_recall() returns a refusal report
            # (no artifact) when the anchor is unavailable.
            return _refusal(
                artifact.get("reason")
                or _REASON_INVALID_REQUEST
            )

        # ``stored`` is an in-memory readiness marker, not part of the
        # persisted artifact contract.
        artifact = dict(artifact)

        artifact.pop("stored", None)

        path = related_recall_path(
            artifact["conversation_id"],
            artifact["cognitive_request_id"],
            artifact["recall_id"],
        )

        with LOCK:
            # Re-check under the lock: a concurrent identical recall
            # may have finished while this one awaited retrieval. The
            # earlier artifact wins deterministically.
            existing = _find_by_fingerprint(
                conversation_id=(
                    artifact["conversation_id"]
                ),
                cognitive_request_id=(
                    artifact[
                        "cognitive_request_id"
                    ]
                ),
                fingerprint=fingerprint,
            )

            if isinstance(existing, dict):
                return _duplicate_report(
                    existing
                )

            atomic_write(path, artifact)

        # Recall result -> Usage Signal. Loaded != used: this records
        # ``recall_requested`` and ``memory_loaded`` only. It never
        # records ``used`` and never reinforces a memory.
        try:
            memory_usage_signal.record_recall(
                recall_artifact=artifact
            )
        except Exception:
            pass

        return _recall_report(artifact)

    except Exception:
        # Fail-open: never 500, never block, never change a response.
        return _refusal(_REASON_UNAVAILABLE)


def read_related_recall(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
) -> dict[str, Any] | None:
    """Read one raw related-recall artifact (internal memory data)."""

    if not is_valid_recall_id(recall_id):
        return None

    try:
        path = related_recall_path(
            conversation_id,
            cognitive_request_id,
            recall_id,
        )
    except ValueError:
        return None

    return read_json(path)


def related_recall_status(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
) -> dict[str, Any]:
    """Privacy-safe status of one related-recall artifact."""

    artifact = read_related_recall(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if not isinstance(artifact, dict):
        return {"exists": False}

    return {
        "exists": True,
        "version": artifact.get("version"),
        "mode": artifact.get("mode"),
        "decision": "recalled",
        "retrieved_count":
            artifact.get("retrieved_count"),
        "included_count":
            artifact.get("included_count"),
        "estimated_tokens":
            artifact.get("estimated_tokens"),
        "token_budget":
            artifact.get("token_budget"),
        "source_flash_unified_revision":
            artifact.get(
                "source_flash_unified_revision"
            ),
    }