from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import ombrebrain.context.unified_context_candidate as unified


CID = "ctx_0123456789abcdef"


class UnifiedRuntimeTests(
    unittest.IsolatedAsyncioTestCase
):

    def setUp(self):
        self.temp = (
            tempfile.TemporaryDirectory()
        )

        self.old_root = os.environ.get(
            "OMBRE_CONTEXT_STATE_DIR"
        )

        os.environ[
            "OMBRE_CONTEXT_STATE_DIR"
        ] = self.temp.name

        self.old_service = (
            unified._CONTEXT_SERVICE
        )

    def tearDown(self):
        unified._CONTEXT_SERVICE = (
            self.old_service
        )

        if self.old_root is None:
            os.environ.pop(
                "OMBRE_CONTEXT_STATE_DIR",
                None,
            )
        else:
            os.environ[
                "OMBRE_CONTEXT_STATE_DIR"
            ] = self.old_root

        self.temp.cleanup()

    def write(
        self,
        kind,
        value,
    ):
        p = (
            Path(self.temp.name)
            / kind
            / (CID + ".json")
        )

        p.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        p.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    async def test_current_user_used_for_retrieval_but_not_persisted(
        self,
    ):
        query = (
            "PRIVATE_CURRENT_USER_QUERY"
        )

        self.write(
            "compact",
            {
                "conversation_id":
                    CID,
                "source_revision":
                    10,
                "recent_messages": [
                    {
                        "role":
                            "assistant",
                        "text":
                            "old",
                        "source_index":
                            20,
                    },
                    {
                        "role":
                            "user",
                        "text":
                            query,
                        "source_index":
                            21,
                    },
                ],
            },
        )

        self.write(
            "context_candidate",
            {
                "version":
                    "conversation-context-candidate.v1",
                "conversation_id":
                    CID,
                "revision":
                    3,
                "source_revision":
                    10,
                "sections": {
                    "current_task": None,
                    "trusted_facts": [],
                    "constraints": [],
                    "decisions": [],
                    "open_items": [],
                    "recent_context": [],
                },
                "telemetry": {
                    "latest_user_source_index":
                        21,
                    "current_user_excluded":
                        True,
                },
            },
        )

        service = AsyncMock()

        service.get_candidates.return_value = {
            "state": {},
            "state_revision": 1,
            "plans": [],
            "memories": [],
            "telemetry": {},
        }

        unified.bind_context_service(
            service
        )

        result = await (
            unified.update_unified_context_candidate_from_runtime(
                CID
            )
        )

        service.get_candidates.assert_awaited_once_with(
            query=query
        )

        self.assertTrue(
            result[
                "retrieval_query_used"
            ]
        )

        output = (
            Path(self.temp.name)
            / "unified_context_candidate"
            / (CID + ".json")
        ).read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            query,
            output,
        )

    async def test_stale_compact_query_is_not_used(
        self,
    ):
        self.write(
            "compact",
            {
                "conversation_id":
                    CID,
                "source_revision":
                    9,
                "recent_messages": [
                    {
                        "role":
                            "user",
                        "text":
                            "STALE_QUERY",
                        "source_index":
                            10,
                    }
                ],
            },
        )

        self.write(
            "context_candidate",
            {
                "version":
                    "conversation-context-candidate.v1",
                "conversation_id":
                    CID,
                "revision":
                    4,
                "source_revision":
                    10,
                "sections": {
                    "current_task": None,
                    "trusted_facts": [],
                    "constraints": [],
                    "decisions": [],
                    "open_items": [],
                    "recent_context": [],
                },
                "telemetry": {
                    "latest_user_source_index":
                        10,
                    "current_user_excluded":
                        True,
                },
            },
        )

        service = AsyncMock()

        service.get_candidates.return_value = {
            "state": {},
            "state_revision": 1,
            "plans": [],
            "memories": [],
            "telemetry": {},
        }

        unified.bind_context_service(
            service
        )

        await (
            unified.update_unified_context_candidate_from_runtime(
                CID
            )
        )

        service.get_candidates.assert_awaited_once_with(
            query=""
        )


if __name__ == "__main__":
    unittest.main()
