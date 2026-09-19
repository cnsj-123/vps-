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

    async def get_candidates(
        self,
        query: str = "",
    ) -> dict[str, Any]:
        """Return unbudgeted context candidates for unified assembly.

        This reuses the existing State / Plan / Memory retrieval path but
        deliberately does not apply ContextPackBuilder's token budget.
        The unified candidate owns the single final budget.
        """
        state, state_revision = self.state_service.get()

        buckets = await self.bucket_mgr.list_all(
            include_archive=False
        )

        plans = self.builder.select_active_plans(
            buckets
        )

        memories: list[dict[str, Any]] = []

        retrieval_candidate_count = 0
        relevance_rejected = 0
        retrieval_quality: dict[str, Any] = {
            "embedding_enabled": False,
            "semantic_score_count": 0,
            "raw_match_count": 0,
            "processed_candidate_count": 0,
            "included_count": 0,
            "max_semantic_score": None,
            "relevance_threshold":
                float(
                    self.retrieval.relevance_threshold
                ),
            "outcome": "empty_query",
        }
        anti_echo: dict[str, int] = {}
        dedup: dict[str, int] = {}

        if query.strip():
            # EmbeddingEngine may be hot-reloaded after ContextService
            # initialization. Follow BucketManager's current engine instead
            # of retaining a stale reference.
            if hasattr(
                self.bucket_mgr,
                "embedding_engine",
            ):
                self.retrieval.embedding_engine = (
                    self.bucket_mgr.embedding_engine
                )

            memories = await self.retrieval.retrieve(
                query,
                max_results=self.builder.max_memories,
            )

            telemetry = (
                self.retrieval.last_telemetry
                or {}
            )

            retrieval_candidate_count = int(
                telemetry.get(
                    "candidate_count",
                    0,
                )
            )

            relevance_rejected = int(
                telemetry.get(
                    "relevance_rejected",
                    0,
                )
            )

            retrieval_quality = {
                "embedding_enabled":
                    bool(
                        telemetry.get(
                            "embedding_enabled",
                            False,
                        )
                    ),
                "semantic_score_count":
                    int(
                        telemetry.get(
                            "semantic_score_count",
                            0,
                        )
                    ),
                "raw_match_count":
                    int(
                        telemetry.get(
                            "raw_match_count",
                            0,
                        )
                    ),
                "processed_candidate_count":
                    int(
                        telemetry.get(
                            "processed_candidate_count",
                            retrieval_candidate_count,
                        )
                    ),
                "included_count":
                    int(
                        telemetry.get(
                            "included_count",
                            len(memories),
                        )
                    ),
                "max_semantic_score":
                    telemetry.get(
                        "max_semantic_score"
                    ),
                "relevance_threshold":
                    telemetry.get(
                        "relevance_threshold",
                        self.retrieval.relevance_threshold,
                    ),
                "outcome":
                    str(
                        telemetry.get(
                            "outcome",
                            "unknown",
                        )
                    ),
            }

            anti_echo = {
                k: int(v)
                for k, v in telemetry.items()
                if k.startswith(
                    "anti_echo_"
                )
            }

            dedup = {
                k: int(v)
                for k, v in telemetry.items()
                if k.startswith(
                    "dedup_"
                )
            }

        return {
            "state":
                state.to_dict(),
            "state_revision":
                state_revision,
            "plans": [
                dict(item)
                for item in plans
            ],
            "memories": [
                dict(item)
                for item in memories
                if isinstance(item, dict)
            ],
            "telemetry": {
                "retrieval_candidate_count":
                    retrieval_candidate_count,
                "relevance_rejected":
                    relevance_rejected,
                "retrieval_quality":
                    retrieval_quality,
                "anti_echo":
                    anti_echo,
                "dedup":
                    dedup,
            },
        }

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
