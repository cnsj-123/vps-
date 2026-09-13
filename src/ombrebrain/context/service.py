from __future__ import annotations
from typing import Any
from ombrebrain.context.pack import ContextPack, ContextPackBuilder
from ombrebrain.state.service import StateService
from ombrebrain.context.retrieval import ContextRetrievalAdapter

class ContextService:
    def __init__(
        self,
        state_service: StateService,
        bucket_mgr: Any,
        embedding_engine: Any = None,
        token_budget: int = 1000,
    ):
        self.state_service = state_service
        self.bucket_mgr = bucket_mgr
        self.builder = ContextPackBuilder(token_budget=token_budget)
        self.retrieval = ContextRetrievalAdapter(bucket_mgr=bucket_mgr, embedding_engine=embedding_engine)

    async def get_pack(
        self,
        memories: list[Any] | tuple[Any, ...] = (),
        query: str = "",
    ) -> ContextPack:
        state, _revision = self.state_service.get()
        buckets = await self.bucket_mgr.list_all(include_archive=False)
        plans = self.builder.select_active_plans(buckets)
        retrieval_candidate_count = 0
        relevance_rejected = 0
        anti_echo = {}
        dedup = {}
        if query.strip() and not memories:
            memories = await self.retrieval.retrieve(query, max_results=self.builder.max_memories)
            telemetry = self.retrieval.last_telemetry or {}
            retrieval_candidate_count = int(telemetry.get("candidate_count", 0))
            relevance_rejected = int(telemetry.get("relevance_rejected", 0))
            anti_echo = {k: int(v) for k, v in telemetry.items() if k.startswith("anti_echo_")}
            dedup = {k: int(v) for k, v in telemetry.items() if k.startswith("dedup_")}
        return self.builder.build(
            state=state,
            plans=plans,
            memories=memories,
            retrieval_candidate_count=retrieval_candidate_count,
            relevance_rejected=relevance_rejected,
            anti_echo=anti_echo,
            dedup=dedup,
        )
