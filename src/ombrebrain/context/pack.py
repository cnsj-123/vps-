from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ombrebrain.state.models import State
from ombrebrain.context.relevance import ContextRelevanceFilter


def _estimate_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, (len(value) + 2) // 3)
    if isinstance(value, dict):
        return _estimate_tokens(" ".join(f"{k}:{v}" for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return sum(_estimate_tokens(item) for item in value)
    return _estimate_tokens(str(value))


@dataclass(frozen=True)
class ContextPack:
    state: State
    plans: tuple[dict[str, Any], ...]
    memories: tuple[Any, ...]
    token_budget: int = 1000
    estimated_tokens: int = 0
    truncated: bool = False
    state_tokens: int = 0
    plan_tokens: int = 0
    memory_tokens: int = 0
    candidate_count: int = 0
    retrieval_candidate_count: int = 0
    relevance_rejected: int = 0
    included_count: int = 0
    budget_rejected: int = 0
    anti_echo: dict[str, int] | None = None
    dedup: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "plans": [dict(item) for item in self.plans],
            "memories": list(self.memories),
            "token_budget": self.token_budget,
            "estimated_tokens": self.estimated_tokens,
            "truncated": self.truncated,
            "state_tokens": self.state_tokens,
            "plan_tokens": self.plan_tokens,
            "memory_tokens": self.memory_tokens,
            "candidate_count": self.candidate_count,
            "retrieval_candidate_count": self.retrieval_candidate_count,
            "relevance_rejected": self.relevance_rejected,
            "included_count": self.included_count,
            "budget_rejected": self.budget_rejected,
            "anti_echo": dict(self.anti_echo or {}),
            "dedup": dict(self.dedup or {}),
        }


@dataclass(frozen=True)
class ContextPackBuilder:
    token_budget: int = 1000
    max_plans: int = 5
    max_memories: int = 8
    relevance_filter: ContextRelevanceFilter = ContextRelevanceFilter()

    @staticmethod
    def select_active_plans(all_buckets):
        plans = []

        for b in all_buckets:
            meta = b.get("metadata") or {}

            if meta.get("type") != "plan":
                continue

            status = (meta.get("status") or "active").lower()

            if status != "active":
                continue

            if meta.get("pinned", False):
                continue

            if meta.get("protected", False):
                continue

            plans.append({
                "id": b.get("id"),
                "name": meta.get("name") or "",
                "content": b.get("content") or "",
                "status": status,
                "created_at": meta.get("created"),
                "updated_at": meta.get("last_active"),
                "related_bucket": meta.get("related_bucket"),
                "importance": meta.get("importance", 7),
                "weight": float(meta.get("weight", 0.5) or 0.5),
                "why_remembered": meta.get("why_remembered", ""),
            })

        plans.sort(
            key=lambda p: float(p.get("weight") or 0.5),
            reverse=True,
        )

        return plans

    def build(
        self,
        state: State,
        plans: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        memories: list[Any] | tuple[Any, ...] = (),
        retrieval_candidate_count: int = 0,
        relevance_rejected: int = 0,
        anti_echo: dict[str, int] | None = None,
        dedup: dict[str, int] | None = None,
    ) -> ContextPack:
        active_plans = tuple(
            dict(plan)
            for plan in plans
            if str(plan.get("status") or "active").lower() == "active"
        )[:self.max_plans]

        candidates = [
            memory for memory in memories
            if isinstance(memory, dict)
        ]

        filtered_memories = self.relevance_filter.filter(candidates)

        state_tokens = _estimate_tokens(state.to_dict())
        plan_tokens = _estimate_tokens(active_plans)

        used_tokens = state_tokens + plan_tokens
        memory_tokens_used = 0
        budget_rejected = 0

        selected = []
        truncated = False

        for memory in filtered_memories[:self.max_memories]:
            memory_tokens = _estimate_tokens(memory)

            if used_tokens + memory_tokens <= self.token_budget:
                selected.append(memory)
                used_tokens += memory_tokens
                memory_tokens_used += memory_tokens
            else:
                budget_rejected += 1
                truncated = True
                break

        return ContextPack(
            state=state,
            plans=active_plans,
            memories=tuple(selected),
            token_budget=self.token_budget,
            estimated_tokens=used_tokens,
            truncated=truncated,
            state_tokens=state_tokens,
            plan_tokens=plan_tokens,
            memory_tokens=memory_tokens_used,
            candidate_count=len(candidates),
            retrieval_candidate_count=retrieval_candidate_count,
            relevance_rejected=relevance_rejected,
            included_count=len(selected),
            budget_rejected=budget_rejected,
            anti_echo=dict(anti_echo or {}),
            dedup=dict(dedup or {}),
        )
