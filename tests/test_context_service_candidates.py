from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from ombrebrain.context.service import (
    ContextService,
)
from ombrebrain.state.models import State


class FakeStateService:

    def get(self):
        return (
            State(
                relationship={
                    "trust": 0.5,
                },
                current_focus=
                    "Context",
                last_seen=None,
            ),
            7,
        )


class ContextServiceCandidateTests(
    unittest.IsolatedAsyncioTestCase
):

    async def test_candidates_are_not_pack_budgeted(
        self,
    ):
        bucket_mgr = AsyncMock()

        bucket_mgr.list_all.return_value = [
            {
                "id": "p1",
                "content": "A" * 600,
                "metadata": {
                    "type": "plan",
                    "status": "active",
                    "name": "Plan 1",
                    "weight": 1.0,
                },
            },
            {
                "id": "p2",
                "content": "B" * 600,
                "metadata": {
                    "type": "plan",
                    "status": "active",
                    "name": "Plan 2",
                    "weight": 0.9,
                },
            },
        ]

        service = ContextService(
            state_service=
                FakeStateService(),
            bucket_mgr=
                bucket_mgr,
            embedding_engine=None,
            token_budget=20,
        )

        result = await service.get_candidates(
            query=""
        )

        self.assertEqual(
            result[
                "state_revision"
            ],
            7,
        )

        self.assertEqual(
            len(
                result[
                    "plans"
                ]
            ),
            2,
        )

    async def test_query_reuses_existing_retrieval(
        self,
    ):
        bucket_mgr = AsyncMock()

        bucket_mgr.list_all.return_value = []

        service = ContextService(
            state_service=
                FakeStateService(),
            bucket_mgr=
                bucket_mgr,
            embedding_engine=None,
        )

        service.retrieval.retrieve = (
            AsyncMock(
                return_value=[
                    {
                        "id": "m1",
                        "content":
                            "memory",
                        "context_relevance":
                            0.9,
                    }
                ]
            )
        )

        service.retrieval.last_telemetry = {
            "candidate_count": 3,
            "relevance_rejected": 2,
            "anti_echo_candidates": 1,
            "dedup_candidates": 1,
        }

        result = await service.get_candidates(
            query="current user query"
        )

        self.assertEqual(
            len(
                result[
                    "memories"
                ]
            ),
            1,
        )

        self.assertEqual(
            result[
                "telemetry"
            ][
                "retrieval_candidate_count"
            ],
            3,
        )

        service.retrieval.retrieve.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
