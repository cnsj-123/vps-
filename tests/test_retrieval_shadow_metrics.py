from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ombrebrain.context.retrieval_shadow_metrics import (
    record_retrieval_shadow_metrics,
    retrieval_shadow_metrics_status,
)


CID = "ctx_0123456789abcdef"


def telemetry(
    *,
    included=0,
    raw=0,
    rejected=0,
    outcome="no_search_matches",
    drop_total=0,
    drop_recent=0,
    drop_id=0,
    drop_text=0,
):
    return {
        "relevance_rejected":
            rejected,
        "retrieval_quality": {
            "included_count":
                included,
            "raw_match_count":
                raw,
            "outcome":
                outcome,
            "decision_shadow": {
                "would_drop_total":
                    drop_total,
                "would_drop_recent_24h":
                    drop_recent,
                "would_drop_duplicate_id":
                    drop_id,
                "would_drop_exact_text_duplicate":
                    drop_text,
            },
        },
    }


class RetrievalShadowMetricsTests(
    unittest.TestCase
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

    def tearDown(self):
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

    def test_aggregates_safe_metrics(
        self,
    ):
        record_retrieval_shadow_metrics(
            conversation_id=CID,
            revision=1,
            telemetry=telemetry(
                included=2,
                raw=2,
                outcome="included",
            ),
            retrieval_query_used=True,
        )

        result = (
            record_retrieval_shadow_metrics(
                conversation_id=CID,
                revision=2,
                telemetry=telemetry(
                    included=1,
                    raw=1,
                    outcome="included",
                    drop_total=1,
                    drop_recent=1,
                ),
                retrieval_query_used=True,
            )
        )

        self.assertEqual(
            result["sample_count"],
            2,
        )

        self.assertEqual(
            result[
                "retrieval_hit_requests"
            ],
            2,
        )

        self.assertEqual(
            result[
                "retrieved_memories"
            ],
            3,
        )

        self.assertEqual(
            result[
                "shadow_would_drop_total"
            ],
            1,
        )

        self.assertEqual(
            result[
                "shadow_would_drop_recent_24h"
            ],
            1,
        )

        self.assertEqual(
            result[
                "shadow_drop_rate_percent"
            ],
            33.3,
        )

        self.assertEqual(
            result["outcomes"],
            {
                "included": 2,
            },
        )

    def test_same_revision_is_idempotent(
        self,
    ):
        first = (
            record_retrieval_shadow_metrics(
                conversation_id=CID,
                revision=7,
                telemetry=telemetry(
                    included=2,
                    outcome="included",
                ),
                retrieval_query_used=True,
            )
        )

        second = (
            record_retrieval_shadow_metrics(
                conversation_id=CID,
                revision=7,
                telemetry=telemetry(
                    included=999,
                    outcome="included",
                ),
                retrieval_query_used=True,
            )
        )

        self.assertFalse(
            first["duplicate"]
        )

        self.assertTrue(
            second["duplicate"]
        )

        self.assertEqual(
            second["sample_count"],
            1,
        )

        self.assertEqual(
            second[
                "retrieved_memories"
            ],
            2,
        )

    def test_window_is_capped_at_100(
        self,
    ):
        for revision in range(
            1,
            106,
        ):
            record_retrieval_shadow_metrics(
                conversation_id=CID,
                revision=revision,
                telemetry=telemetry(
                    included=1,
                    outcome="included",
                ),
                retrieval_query_used=True,
            )

        status = (
            retrieval_shadow_metrics_status()
        )

        self.assertEqual(
            status["window_size"],
            100,
        )

        self.assertEqual(
            status["sample_count"],
            100,
        )

        self.assertEqual(
            status[
                "retrieved_memories"
            ],
            100,
        )

    def test_raw_text_is_never_persisted(
        self,
    ):
        secret = (
            "PRIVATE_QUERY_OR_MEMORY_TEXT"
        )

        value = telemetry(
            included=1,
            outcome="included",
        )

        value["query"] = secret
        value["content"] = secret

        value[
            "retrieval_quality"
        ][
            "memory_text"
        ] = secret

        record_retrieval_shadow_metrics(
            conversation_id=CID,
            revision=1,
            telemetry=value,
            retrieval_query_used=True,
        )

        rendered = (
            Path(self.temp.name)
            / "retrieval_shadow_metrics.json"
        ).read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            secret,
            rendered,
        )

        parsed = json.loads(
            rendered
        )

        self.assertEqual(
            parsed["version"],
            "retrieval-shadow-metrics.v1",
        )


if __name__ == "__main__":
    unittest.main()
