from __future__ import annotations

import os
from typing import Any

from ombrebrain.context import memory_usage_signal
from ombrebrain.context import memory_recall_surface
from ombrebrain.context.recall_request import (
    RECALL_REQUEST_VERSION,
    authorize_recall_request,
    find_recall_request_by_fingerprint,
    new_recall_id,
    recall_control_plane_enabled,
    recall_request_fingerprint,
    record_recall_request,
)
from ombrebrain.context.recall_types import (
    LOCK,
    atomic_write,
    bound_text,
    estimate_tokens,
    is_valid_cognitive_request_id,
    is_valid_conversation_id,
    is_valid_fingerprint,
    is_valid_recall_id,
    normalize_key,
    now_iso,
    parse_positive_int,
    read_json,
    related_recall_path,
)
from ombrebrain.context.unified_context_candidate import (
    bound_context_service,
)
from ombrebrain.context.validators.freshness import (
    is_valid_revision,
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
_REASON_INVALID_RECALL_ID = "invalid_recall_id"
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
    recall_id: Any = None,
    duplicate: bool = False,
) -> dict[str, Any]:
    return {
        "version": _VERSION,
        "mode": mode,
        "stored": False,
        "duplicate": duplicate,
        "decision": "refused",
        "reason": reason,
        "recall_id": recall_id,
        "retrieved_count": 0,
        "included_count": 0,
        "estimated_tokens": 0,
        "token_budget": None,
        "recall": None,
    }


async def build_related_recall(
    *,
    recall_request: Any,
    recall_id: Any = None,
    current_query: Any = "",
    bucket_manager: Any = None,
    retrieval_adapter: Any = None,
    budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build (do not persist) one bounded related recall artifact.

    ``recall_id`` is minted by the Recall control plane BEFORE the
    anchor is loaded, so a trusted caller passes the id this request
    already owns and the Related Recall artifact reuses it. A
    malformed id is refused; ``None`` mints one only for standalone
    use, never for the control plane flow.

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

    if recall_id is None:
        recall_id = new_recall_id()
    elif not is_valid_recall_id(recall_id):
        return _refusal(
            _REASON_INVALID_RECALL_ID
        )

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
        "recall_id": recall_id,
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


def _is_positive_rank(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
    )


def is_valid_related_recall_artifact(
    artifact: Any,
    *,
    conversation_id: Any = None,
    cognitive_request_id: Any = None,
    recall_id: Any = None,
    request_fingerprint: Any = None,
) -> bool:
    """Structural identity of one persisted Related Recall artifact.

    A correct path proves nothing. Readers verify the contract, the
    mode, the complete request identity, the source binding, the
    scope, the RECOMPUTED request fingerprint and the memory shape,
    so a corrupt artifact can never be reused as a duplicate success:

      - ``source_flash_request_id`` must bind to this request's id
        (the v1 Flash is request scoped);
      - ``request_fingerprint`` is recomputed from the artifact's own
        identity and compared -- the fingerprint an artifact carries
        about itself is never trusted;
      - ``memories`` must be a self-consistent list: unique ids,
        unique normalized content, sequential ranks starting at 1,
        and the anchor first.
    """

    if not isinstance(artifact, dict):
        return False

    if (
        artifact.get("version") != _VERSION
        or artifact.get("mode") != _MODE
    ):
        return False

    artifact_conversation = artifact.get(
        "conversation_id"
    )

    artifact_request = artifact.get(
        "cognitive_request_id"
    )

    artifact_recall = artifact.get("recall_id")

    if (
        conversation_id is not None
        and artifact_conversation
        != conversation_id
    ):
        return False

    if (
        cognitive_request_id is not None
        and artifact_request
        != cognitive_request_id
    ):
        return False

    if (
        recall_id is not None
        and artifact_recall != recall_id
    ):
        return False

    if (
        not is_valid_conversation_id(
            artifact_conversation
        )
        or not is_valid_cognitive_request_id(
            artifact_request
        )
    ):
        return False

    if not is_valid_recall_id(artifact_recall):
        return False

    anchor = artifact.get("anchor_memory_id")

    if (
        not isinstance(anchor, str)
        or not anchor.strip()
    ):
        return False

    anchor = anchor.strip()

    # The v1 Flash is request scoped: the recall must have been built
    # from THIS request's Flash.
    if (
        artifact.get("source_flash_request_id")
        != artifact_request
    ):
        return False

    if not is_valid_revision(
        artifact.get(
            "source_flash_unified_revision"
        )
    ):
        return False

    if artifact.get("requested_scope") != "related":
        return False

    fingerprint = artifact.get(
        "request_fingerprint"
    )

    if not is_valid_fingerprint(fingerprint):
        return False

    expected = recall_request_fingerprint(
        conversation_id=artifact_conversation,
        cognitive_request_id=artifact_request,
        anchor_memory_id=anchor,
        requested_scope="related",
    )

    if fingerprint != expected:
        return False

    if (
        request_fingerprint is not None
        and fingerprint != request_fingerprint
    ):
        return False

    memories = artifact.get("memories")

    if not isinstance(memories, list):
        return False

    if artifact.get("included_count") != len(
        memories
    ):
        return False

    seen_ids: set[str] = set()
    seen_text: set[str] = set()

    for index, item in enumerate(memories):
        if not isinstance(item, dict):
            return False

        memory_id = item.get("memory_id")

        if (
            not isinstance(memory_id, str)
            or not memory_id.strip()
        ):
            return False

        memory_id = memory_id.strip()

        content = item.get("content")

        if (
            not isinstance(content, str)
            or not content.strip()
            or len(content)
            > _MAX_CHARS_PER_MEMORY_CAP
        ):
            return False

        reason = item.get("reason")

        if reason not in (
            "anchor_memory",
            "related_memory",
        ):
            return False

        rank = item.get("rank")

        # Ranks are 1-based and strictly sequential.
        if not _is_positive_rank(rank):
            return False

        if rank != index + 1:
            return False

        if memory_id in seen_ids:
            return False

        seen_ids.add(memory_id)

        text_key = normalize_key(content)

        if text_key in seen_text:
            return False

        seen_text.add(text_key)

        if index == 0:
            # The anchor memory is always first.
            if (
                memory_id != anchor
                or reason != "anchor_memory"
                or rank != 1
            ):
                return False

    return True


def _read_related_recall_checked(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    """Read one recall artifact for THIS request, fully validated."""

    artifact = read_related_recall(
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    )

    if not is_valid_related_recall_artifact(
        artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
        request_fingerprint=fingerprint,
    ):
        return None

    return artifact


def _recall_report(
    artifact: dict[str, Any],
    *,
    duplicate: bool = False,
) -> dict[str, Any]:
    return {
        "version": _VERSION,
        "mode": _MODE,
        "stored": True,
        "duplicate": duplicate,
        "decision": "recalled",
        "reason": (
            "duplicate_recall_request"
            if duplicate
            else "recall_completed"
        ),
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
    ``cognitive_request_id`` (the model may never pass them). The
    model may only ever name an anchor, and only through an opaque
    ``memref`` (``anchor_ref``) or the already-surfaced memory id.

    ``current_query`` MUST come from trusted request context (the
    Gateway / Coordinator that owns the live request) and MUST NOT be
    accepted from model tool arguments. It is deliberately not part of
    any model-facing schema.

    Ordering (this is the whole point of the control plane):

        authorize against the real Flash
          -> fingerprint / reuse-or-persist the Recall Request
            -> mint recall_id (if new)
              -> persist recall_requested
                -> load anchor + one canonical related retrieval
                  -> persist Related Recall (if loaded)
                    -> record memory_loaded

    So ``recall_requested`` exists even when nothing can be loaded,
    and the recall id is stable before any load is attempted.

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

        # An invalid authorization is not a Recall Request: nothing is
        # persisted and no recall_requested event exists.
        if not authorization.get("valid"):
            return _refusal(
                authorization.get("reason")
                or _REASON_INVALID_REQUEST
            )

        recall_request = authorization[
            "recall_request"
        ]

        cid = recall_request["conversation_id"]

        rid = recall_request[
            "cognitive_request_id"
        ]

        fingerprint = (
            recall_request_fingerprint(
                conversation_id=cid,
                cognitive_request_id=rid,
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

        # Idempotency is anchored to the persisted Recall Request, not
        # to the Related Recall artifact: a recall that failed to load
        # has no recall artifact and must still be deduplicated.
        with LOCK:
            existing_request = (
                find_recall_request_by_fingerprint(
                    conversation_id=cid,
                    cognitive_request_id=rid,
                    fingerprint=fingerprint,
                )
            )

        if isinstance(existing_request, dict):
            recall_id = existing_request.get(
                "recall_id"
            )
            request_duplicate = True
        else:
            recall_id = new_recall_id()

            recorded = record_recall_request(
                recall_request=recall_request,
                recall_id=recall_id,
            )

            if not recorded.get("stored"):
                return _refusal(
                    recorded.get("reason")
                    or _REASON_INVALID_REQUEST
                )

            recall_id = (
                recorded.get("recall_id")
                or recall_id
            )
            request_duplicate = bool(
                recorded.get("duplicate")
            )

        if not is_valid_recall_id(recall_id):
            return _refusal(
                _REASON_INVALID_RECALL_ID
            )

        # This exact recall already completed: reuse the persisted
        # result. No second retrieval and no overwrite.
        existing_recall = (
            _read_related_recall_checked(
                conversation_id=cid,
                cognitive_request_id=rid,
                recall_id=recall_id,
                fingerprint=fingerprint,
            )
        )

        if existing_recall is not None:
            try:
                memory_usage_signal.record_memory_loaded(
                    conversation_id=cid,
                    cognitive_request_id=rid,
                    recall_id=recall_id,
                )
            except Exception:
                pass

            return _recall_report(
                existing_recall,
                duplicate=True,
            )

        # The AI's request is now a recorded fact, independent of
        # whether anything can be loaded. Idempotent per recall id.
        try:
            memory_usage_signal.record_recall_requested(
                conversation_id=cid,
                cognitive_request_id=rid,
                recall_id=recall_id,
            )
        except Exception:
            pass

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
                _REASON_CONTEXT_UNAVAILABLE,
                recall_id=recall_id,
                duplicate=request_duplicate,
            )

        artifact = await build_related_recall(
            recall_request=recall_request,
            recall_id=recall_id,
            current_query=current_query,
            bucket_manager=resolved_bucket,
            retrieval_adapter=(
                resolved_retrieval
            ),
            budget=budget,
        )

        if artifact.get("stored") is not True:
            # Anchor unavailable / invalid id: the Recall Request and
            # its recall id survive, with zero loaded memories.
            return _refusal(
                artifact.get("reason")
                or _REASON_INVALID_REQUEST,
                recall_id=recall_id,
                duplicate=request_duplicate,
            )

        # ``stored`` is an in-memory readiness marker, not part of the
        # persisted artifact contract.
        artifact = dict(artifact)

        artifact.pop("stored", None)

        path = related_recall_path(
            cid, rid, artifact["recall_id"]
        )

        with LOCK:
            # Re-check under the lock: a concurrent identical recall
            # may have finished while this one awaited retrieval. The
            # earlier artifact wins deterministically.
            previous = (
                _read_related_recall_checked(
                    conversation_id=cid,
                    cognitive_request_id=rid,
                    recall_id=recall_id,
                    fingerprint=fingerprint,
                )
            )

            if previous is not None:
                artifact = previous
            else:
                atomic_write(path, artifact)

        # Loaded != used, and loaded only from what was really
        # persisted: ``record_memory_loaded`` re-reads and re-validates
        # the artifact from disk itself. This never records ``used``
        # and never reinforces a memory.
        try:
            memory_usage_signal.record_memory_loaded(
                conversation_id=cid,
                cognitive_request_id=rid,
                recall_id=recall_id,
            )
        except Exception:
            pass

        return _recall_report(
            artifact,
            duplicate=request_duplicate,
        )

    except Exception:
        # Fail-open: never 500, never block, never change a response.
        return _refusal(_REASON_UNAVAILABLE)


def read_related_recall(
    *,
    conversation_id: str,
    cognitive_request_id: str,
    recall_id: str,
) -> dict[str, Any] | None:
    """Read one raw related-recall artifact (internal memory data).

    Identity-validated: a corrupt artifact is never returned as if it
    were a valid recall result.
    """

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

    artifact = read_json(path)

    if not is_valid_related_recall_artifact(
        artifact,
        conversation_id=conversation_id,
        cognitive_request_id=(
            cognitive_request_id
        ),
        recall_id=recall_id,
    ):
        return None

    return artifact


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