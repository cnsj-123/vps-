from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import (
    ThreadPoolExecutor,
)
from pathlib import Path
from unittest.mock import (
    AsyncMock,
    Mock,
    patch,
)

from ombrebrain.context import (
    context_pipeline_coordinator as coordinator,
)
from ombrebrain.context import (
    memory_source_arbitration_shadow as arbitration,
)
from ombrebrain.context.recall_types import (
    estimate_tokens,
)
from ombrebrain.gateway import (
    FrontendMemorySegment,
    OperitAdapter,
    extract_frontend_memory_segments,
)


CID = "ctx_0123456789abcdef"

_ARBITRATION_ENV = (
    "OMBRE_GATEWAY_CONTEXT_MEMORY_SOURCE_ARBITRATION_SHADOW"
)

_ALL_FLAGS = (
    "OMBRE_GATEWAY_CONTEXT_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_SNAPSHOT_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_COMPACT_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_SEMANTIC_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_SEMANTIC_STATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_TRUSTED_FACTS_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_CANDIDATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_REAL_INJECTION",
    "OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_EXPOSURE_LEDGER_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW",
    "OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_LIVE_EXPOSURE",
    _ARBITRATION_ENV,
)

_ISOLATED_ENV = {
    flag: "0" for flag in _ALL_FLAGS
}


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _attachment(
    filename,
    body,
    attachment_id="message_insert_extra_bundle_test",
):
    return (
        '<attachment id="'
        + attachment_id
        + '" filename="'
        + filename
        + '" type="text/plain">'
        + body
        + "</attachment>"
    )


def _body(messages):
    return json.dumps(
        {
            "model": "example",
            "messages": messages,
        }
    ).encode("utf-8")


def _segment(
    placement,
    body,
    *,
    attachment_id="front_seg",
):
    """Build one canonical Frontend Memory attachment segment.

    ``raw_attachment`` is the exact envelope the request carries, so
    the model-visible footprint can be measured from real text.
    """

    return FrontendMemorySegment(
        placement=placement,
        source_message_index=0,
        source_block_index=0,
        body=body,
        raw_attachment=_attachment(
            "relevant_memories.txt",
            body,
            attachment_id,
        ),
    )


def _current_raw(body, index=0):
    """The exact attachment envelope _frontend_current_body emits."""

    return _attachment(
        "relevant_memories.txt",
        body,
        "front_%d" % index,
    )


def _frontend_current_body(
    *bodies,
    user_text="current question",
):
    text = user_text + " " + " ".join(
        _attachment(
            "relevant_memories.txt",
            body,
            "front_%d" % index,
        )
        for index, body in enumerate(bodies)
    )

    return _body(
        [{"role": "user", "content": text}]
    )


def _frontend_history_body(
    history_body,
):
    history = (
        "old question "
        + _attachment(
            "relevant_memories.txt",
            history_body,
            "front_hist",
        )
    )

    return _body(
        [
            {
                "role": "user",
                "content": history,
            },
            {
                "role": "assistant",
                "content": "answer",
            },
            {
                "role": "user",
                "content": "current question",
            },
        ]
    )


def _memory(memory_id, content):
    return {
        "id": memory_id,
        "content": content,
    }


def _unified(memories, revision=5):
    return {
        "version":
            "unified-context-candidate.v1",
        "conversation_id": CID,
        "revision": revision,
        "sections": {"memories": memories},
    }


def _snapshot(memories, revision=5):
    return {
        "version":
            "memory-shadow-snapshot.v1",
        "mode": "shadow_only",
        "conversation_id": CID,
        "revision": revision,
        "unified": _unified(
            memories,
            revision=revision,
        ),
    }


def _arbitration_dir(root):
    return (
        Path(root)
        / "memory_source_arbitration"
        / CID
    )


def _read_all(directory):
    return [
        json.loads(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(directory.iterdir())
    ]


# ---------------------------------------------------------------------
# Canonicalization helper (public extraction contract)
# ---------------------------------------------------------------------


class FrontendMemoryExtractionTests(
    unittest.TestCase,
):

    def _extract(self, body):
        payload = json.loads(body)
        request = OperitAdapter().adapt(payload)
        return extract_frontend_memory_segments(
            request
        )

    def test_extracts_only_relevant_memories(
        self,
    ):
        body = _body(
            [
                {
                    "role": "user",
                    "content": (
                        "hello "
                        + _attachment(
                            "relevant_memories.txt",
                            "MEM-A",
                            "a",
                        )
                        + " "
                        + _attachment(
                            "notes.txt",
                            "not a memory",
                            "b",
                        )
                        + " "
                        + _attachment(
                            "message_insert_extra_bundle_1",
                            "dynamic",
                            "c",
                        )
                    ),
                }
            ]
        )

        segments = self._extract(body)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].body, "MEM-A")
        self.assertEqual(
            segments[0].placement, "current"
        )
        # The exact raw envelope is exposed for model-visible
        # footprint measurement.
        self.assertEqual(
            segments[0].raw_attachment,
            _attachment(
                "relevant_memories.txt",
                "MEM-A",
                "a",
            ),
        )

    def test_placement_current_and_history(
        self,
    ):
        body = _frontend_history_body("HIST")
        segments = self._extract(body)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].body, "HIST")
        self.assertEqual(
            segments[0].placement, "history"
        )
        self.assertEqual(
            segments[0].raw_attachment,
            _attachment(
                "relevant_memories.txt",
                "HIST",
                "front_hist",
            ),
        )

    def test_self_closing_and_empty_are_skipped(
        self,
    ):
        self_closing = _body(
            [
                {
                    "role": "user",
                    "content": (
                        '<attachment '
                        'filename="relevant_memories.txt" '
                        "/>"
                    ),
                }
            ]
        )

        self.assertEqual(
            self._extract(self_closing), ()
        )

        empty = _body(
            [
                {
                    "role": "user",
                    "content": (
                        '<attachment '
                        'filename="relevant_memories.txt">'
                        "</attachment>"
                    ),
                }
            ]
        )

        self.assertEqual(
            self._extract(empty), ()
        )

    def test_malformed_attachment_is_not_frontend(
        self,
    ):
        # Missing closing tag: the canonical splitter never
        # classifies it, so it must stay ordinary user text.
        body = _body(
            [
                {
                    "role": "user",
                    "content": (
                        '<attachment '
                        'filename="relevant_memories.txt">'
                        "unterminated"
                    ),
                }
            ]
        )

        self.assertEqual(
            self._extract(body), ()
        )

    def test_plain_text_is_never_reclassified(
        self,
    ):
        body = _body(
            [
                {
                    "role": "user",
                    "content": (
                        "please read relevant_memories.txt"
                    ),
                }
            ]
        )

        self.assertEqual(
            self._extract(body), ()
        )


# ---------------------------------------------------------------------
# Pure builder metrics
# ---------------------------------------------------------------------


class ArbitrationBuilderTests(
    unittest.TestCase,
):

    def _artifact(
        self,
        frontend_segments=(),
        ob_memories=(),
        revision=5,
        ob_available=True,
    ):
        return arbitration.build_arbitration_artifact(
            conversation_id=CID,
            cognitive_request_id=(
                "ctxreq_" + "a" * 32
            ),
            source_unified_revision=revision,
            frontend_segments=frontend_segments,
            ob_memories=ob_memories,
            ob_observation_available=(
                ob_available
            ),
        )

    def test_empty_pool_is_all_zero(self):
        artifact = self._artifact()

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics["frontend_attachment_count"], 0
        )
        self.assertEqual(
            metrics["ob_candidate_count"], 0
        )
        self.assertEqual(
            metrics["combined_observation_count"],
            0,
        )
        self.assertEqual(
            metrics[
                "frontend_model_visible_estimated_tokens"
            ],
            0,
        )
        self.assertEqual(
            metrics[
                "ob_retrieved_candidate_content"
                "_estimated_tokens"
            ],
            0,
        )

    def test_no_ambiguous_share_metrics(self):
        metrics = self._artifact()["metrics"]

        # There is deliberately no cross-source token share: Frontend
        # body content is model-visible while OB candidate content is
        # not, so a share would be semantically wrong.
        self.assertNotIn(
            "frontend_token_share", metrics
        )
        self.assertNotIn(
            "ob_token_share", metrics
        )
        self.assertNotIn(
            "combined_estimated_tokens", metrics
        )
        self.assertNotIn(
            "combined_memory_candidate_count",
            metrics,
        )

    def test_one_attachment_is_one_frontend_attachment(
        self,
    ):
        artifact = self._artifact(
            frontend_segments=(
                _segment("current", "ALPHA"),
            ),
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics["frontend_attachment_count"], 1
        )
        self.assertEqual(
            metrics[
                "frontend_current_attachment_count"
            ],
            1,
        )
        self.assertEqual(
            metrics[
                "frontend_history_attachment_count"
            ],
            0,
        )

        observation = artifact["observations"][0]

        # Attachment-level observation, never a claimed memory item.
        self.assertEqual(
            observation["kind"],
            "frontend_attachment",
        )
        self.assertEqual(
            observation["source"], "frontend"
        )

    def test_frontend_body_vs_model_visible_metrics(
        self,
    ):
        segment = _segment("current", "ALPHA")

        raw = segment.raw_attachment

        artifact = self._artifact(
            frontend_segments=(segment,),
            ob_memories=[
                _memory("m1", "BETA")
            ],
        )

        metrics = artifact["metrics"]

        # Body content size uses the extracted inner body only.
        self.assertEqual(
            metrics["frontend_body_chars"], 5
        )
        self.assertEqual(
            metrics["frontend_body_estimated_tokens"],
            estimate_tokens("ALPHA"),
        )

        # Model-visible footprint uses the exact raw envelope.
        self.assertEqual(
            metrics[
                "frontend_model_visible_chars"
            ],
            len(raw),
        )
        self.assertEqual(
            metrics[
                "frontend_model_visible"
                "_estimated_tokens"
            ],
            estimate_tokens(raw),
        )
        self.assertEqual(
            metrics[
                "frontend_model_visible"
                "_attachment_count"
            ],
            1,
        )

        # The envelope can only be larger than the bare body.
        self.assertGreaterEqual(
            metrics[
                "frontend_model_visible"
                "_estimated_tokens"
            ],
            metrics[
                "frontend_body_estimated_tokens"
            ],
        )

        self.assertEqual(
            metrics["ob_candidate_count"], 1
        )
        self.assertEqual(
            metrics[
                "ob_retrieved_candidate_content"
                "_estimated_tokens"
            ],
            estimate_tokens("BETA"),
        )
        self.assertEqual(
            metrics["combined_observation_count"],
            2,
        )

    def test_ob_candidates_are_not_model_visible(
        self,
    ):
        artifact = self._artifact(
            ob_memories=[_memory("m1", "OB-1")]
        )

        observation = artifact["observations"][0]

        self.assertEqual(
            observation["source"], "ob"
        )
        self.assertEqual(
            observation["kind"], "ob_candidate"
        )
        self.assertFalse(
            observation["currently_model_visible"]
        )
        self.assertTrue(observation["retrieved"])
        self.assertEqual(
            observation["model_visible_chars"], 0
        )
        self.assertEqual(
            observation[
                "model_visible_estimated_tokens"
            ],
            0,
        )

    def test_observation_never_recalled_used_reinforced(
        self,
    ):
        artifact = self._artifact(
            frontend_segments=(
                _segment("history", "MEM"),
            ),
        )

        observation = artifact["observations"][0]

        self.assertEqual(
            set(observation.keys()),
            {
                "source",
                "kind",
                "placement",
                "index",
                "content_digest",
                "normalized_digest",
                "content_chars",
                "content_estimated_tokens",
                "model_visible_chars",
                "model_visible_estimated_tokens",
                "currently_model_visible",
                "retrieved",
            },
        )
        # Frontend Memory is present in the request, which is not
        # recalled / used / reinforced and must not be modelled so.
        self.assertTrue(
            observation["currently_model_visible"]
        )
        self.assertFalse(observation["retrieved"])
        self.assertEqual(
            observation["kind"],
            "frontend_attachment",
        )

    def test_exact_overlap(self):
        artifact = self._artifact(
            frontend_segments=(
                _segment("current", "SAME"),
            ),
            ob_memories=[_memory("m1", "SAME")],
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics[
                "frontend_bundle_ob_exact_pair_count"
            ],
            1,
        )
        self.assertEqual(
            metrics[
                "frontend_bundle_ob"
                "_normalized_pair_count"
            ],
            1,
        )
        self.assertEqual(
            metrics[
                "normalized_equivalence_group_count"
            ],
            1,
        )
        # The overlap is scoped to attachment body vs candidate
        # content, not an item-level memory duplication claim.
        self.assertEqual(
            artifact["overlap_scope"],
            "frontend_attachment_body"
            "_vs_ob_candidate_content",
        )

    def test_normalized_overlap_only(self):
        artifact = self._artifact(
            frontend_segments=(
                _segment(
                    "current", "Same   Memory"
                ),
            ),
            ob_memories=[
                _memory("m1", "same memory")
            ],
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics[
                "frontend_bundle_ob_exact_pair_count"
            ],
            0,
        )
        self.assertEqual(
            metrics[
                "frontend_bundle_ob"
                "_normalized_pair_count"
            ],
            1,
        )
        self.assertEqual(
            metrics[
                "normalized_equivalence_group_count"
            ],
            1,
        )

    def test_same_content_twice_is_one_group(
        self,
    ):
        artifact = self._artifact(
            ob_memories=[
                _memory("m1", "DUP"),
                _memory("m2", "DUP"),
            ],
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics[
                "normalized_equivalence_group_count"
            ],
            1,
        )
        self.assertEqual(
            metrics[
                "frontend_bundle_ob_exact_pair_count"
            ],
            0,
        )

    def test_same_frontend_content_twice_is_one_group(
        self,
    ):
        artifact = self._artifact(
            frontend_segments=(
                _segment("current", "DUP"),
                _segment("history", "DUP"),
            ),
        )

        metrics = artifact["metrics"]

        self.assertEqual(
            metrics["frontend_attachment_count"], 2
        )
        self.assertEqual(
            metrics[
                "normalized_equivalence_group_count"
            ],
            1,
        )
        # No OB side, so there is no cross-source overlap.
        self.assertEqual(
            metrics[
                "frontend_bundle_ob_exact_pair_count"
            ],
            0,
        )
        self.assertEqual(
            metrics[
                "frontend_bundle_ob"
                "_normalized_pair_count"
            ],
            0,
        )

    def test_same_id_different_content_not_collapsed(
        self,
    ):
        artifact = self._artifact(
            ob_memories=[
                _memory("same-id", "ONE"),
                _memory("same-id", "TWO"),
            ],
        )

        self.assertEqual(
            artifact["metrics"][
                "normalized_equivalence_group_count"
            ],
            0,
        )
        self.assertEqual(
            artifact["metrics"]["ob_candidate_count"],
            2,
        )

    def test_metrics_only_no_raw_text(self):
        serialized = json.dumps(
            self._artifact(
                frontend_segments=(
                    _segment(
                        "current",
                        "SECRET-FRONTEND",
                    ),
                ),
                ob_memories=[
                    _memory("m1", "SECRET-OB")
                ],
            )
        )

        self.assertNotIn(
            "SECRET-FRONTEND", serialized
        )
        self.assertNotIn("SECRET-OB", serialized)
        self.assertNotIn("m1", serialized)
        # The raw envelope is transient in memory only: neither its
        # text, its metadata nor a raw field is persisted.
        self.assertNotIn(
            "raw_attachment", serialized
        )
        self.assertNotIn("<attachment", serialized)
        self.assertNotIn(
            "filename=", serialized
        )
        self.assertNotIn(
            "front_seg", serialized
        )


# ---------------------------------------------------------------------
# Persistence / binding / fail-safe
# ---------------------------------------------------------------------


class ArbitrationPersistenceTests(
    unittest.TestCase,
):

    def test_persists_bound_artifact(self):
        body = _frontend_current_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root
                },
                clear=False,
            ):
                report = (
                    arbitration
                    .update_memory_source_arbitration(
                        conversation_id=CID,
                        cognitive_request_id=(
                            "ctxreq_" + "b" * 32
                        ),
                        snapshot=_snapshot(
                            [_memory("m1", "OB-A")]
                        ),
                        expected_unified_revision=5,
                        forward_body=body,
                    )
                )

            self.assertTrue(report["stored"])

            artifacts = _read_all(
                _arbitration_dir(root)
            )

            raw_text = (
                next(
                    _arbitration_dir(root)
                    .iterdir()
                )
                .read_text(encoding="utf-8")
            )

        self.assertEqual(len(artifacts), 1)

        artifact = artifacts[0]

        self.assertEqual(
            artifact["source_unified_revision"], 5
        )
        self.assertTrue(
            artifact["ob_observation_available"]
        )
        self.assertEqual(
            artifact["metrics"][
                "frontend_attachment_count"
            ],
            1,
        )
        self.assertEqual(
            artifact["metrics"]["ob_candidate_count"],
            1,
        )

        # No raw Frontend text / envelope is persisted.
        self.assertNotIn("MEM-A", raw_text)
        self.assertNotIn(
            "relevant_memories.txt", raw_text
        )
        self.assertNotIn(
            "raw_attachment", raw_text
        )

    def test_malformed_body_is_fail_safe(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root
                },
                clear=False,
            ):
                report = (
                    arbitration
                    .update_memory_source_arbitration(
                        conversation_id=CID,
                        cognitive_request_id=(
                            "ctxreq_" + "c" * 32
                        ),
                        snapshot=_snapshot([]),
                        expected_unified_revision=5,
                        forward_body=b"{not json",
                    )
                )

            self.assertFalse(report["stored"])
            self.assertEqual(
                report["reason"],
                "request_not_canonicalizable",
            )
            self.assertFalse(
                _arbitration_dir(root).exists()
            )

    def test_non_bytes_body_is_fail_safe(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root
                },
                clear=False,
            ):
                report = (
                    arbitration
                    .update_memory_source_arbitration(
                        conversation_id=CID,
                        cognitive_request_id=(
                            "ctxreq_" + "d" * 32
                        ),
                        snapshot=_snapshot([]),
                        expected_unified_revision=5,
                        forward_body="not-bytes",
                    )
                )

            self.assertFalse(report["stored"])

    def test_invalid_bindings_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root
                },
                clear=False,
            ):
                bad_conversation = (
                    arbitration
                    .update_memory_source_arbitration(
                        conversation_id="nope",
                        cognitive_request_id=(
                            "ctxreq_" + "e" * 32
                        ),
                        snapshot=_snapshot([]),
                        expected_unified_revision=5,
                        forward_body=b"{}",
                    )
                )

                bad_request = (
                    arbitration
                    .update_memory_source_arbitration(
                        conversation_id=CID,
                        cognitive_request_id="nope",
                        snapshot=_snapshot([]),
                        expected_unified_revision=5,
                        forward_body=b"{}",
                    )
                )

                bad_revision = (
                    arbitration
                    .update_memory_source_arbitration(
                        conversation_id=CID,
                        cognitive_request_id=(
                            "ctxreq_" + "f" * 32
                        ),
                        snapshot=_snapshot([]),
                        expected_unified_revision=0,
                        forward_body=b"{}",
                    )
                )

        self.assertEqual(
            bad_conversation["reason"],
            "invalid_conversation_id",
        )
        self.assertEqual(
            bad_request["reason"],
            "invalid_cognitive_request_id",
        )
        self.assertEqual(
            bad_revision["reason"],
            "invalid_expected_unified_revision",
        )

    def test_snapshot_binding_mismatch_marks_ob_unavailable(
        self,
    ):
        body = _frontend_current_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root
                },
                clear=False,
            ):
                # Snapshot conversation differs.
                other = _snapshot(
                    [_memory("m1", "OB-A")]
                )
                other["conversation_id"] = (
                    "ctx_ffffffffffffffff"
                )

                wrong_conversation = (
                    arbitration
                    .update_memory_source_arbitration(
                        conversation_id=CID,
                        cognitive_request_id=(
                            "ctxreq_" + "1" * 32
                        ),
                        snapshot=other,
                        expected_unified_revision=5,
                        forward_body=body,
                    )
                )

                # Snapshot revision differs from the expected one.
                wrong_revision = (
                    arbitration
                    .update_memory_source_arbitration(
                        conversation_id=CID,
                        cognitive_request_id=(
                            "ctxreq_" + "2" * 32
                        ),
                        snapshot=_snapshot(
                            [_memory("m1", "OB-A")],
                            revision=9,
                        ),
                        expected_unified_revision=5,
                        forward_body=body,
                    )
                )

        for report in (
            wrong_conversation,
            wrong_revision,
        ):
            self.assertTrue(report["stored"])
            self.assertFalse(
                report["ob_observation_available"]
            )
            self.assertEqual(
                report["metrics"][
                    "ob_candidate_count"
                ],
                0,
            )
            # Frontend side is still observed truthfully.
            self.assertEqual(
                report["metrics"][
                    "frontend_attachment_count"
                ],
                1,
            )

    def test_never_reads_shared_unified_from_disk(
        self,
    ):
        body = _frontend_current_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            # Decoy: a "latest" Unified for this conversation on
            # disk that must never be read by the observer.
            decoy_dir = (
                Path(root)
                / "unified_context_candidate"
            )
            decoy_dir.mkdir(
                parents=True, exist_ok=True
            )
            (
                decoy_dir
                / (CID + ".json")
            ).write_text(
                json.dumps(
                    {
                        "version":
                            "unified-context-candidate.v1",
                        "conversation_id": CID,
                        "revision": 99,
                        "sections": {
                            "memories": [
                                _memory(
                                    "decoy",
                                    "OB-DECOY",
                                )
                                for _ in range(5)
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root
                },
                clear=False,
            ):
                report = (
                    arbitration
                    .update_memory_source_arbitration(
                        conversation_id=CID,
                        cognitive_request_id=(
                            "ctxreq_" + "3" * 32
                        ),
                        snapshot=_snapshot(
                            [
                                _memory("m1", "OB-A"),
                                _memory("m2", "OB-B"),
                            ]
                        ),
                        expected_unified_revision=5,
                        forward_body=body,
                    )
                )

        self.assertTrue(report["stored"])
        self.assertEqual(
            report["source_unified_revision"], 5
        )
        self.assertEqual(
            report["metrics"]["ob_candidate_count"],
            2,
        )

    def test_same_conversation_requests_do_not_mix(
        self,
    ):
        body_a = _frontend_current_body("FRONT-A")
        body_b = _frontend_current_body("FRONT-B")

        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {
                    "OMBRE_CONTEXT_STATE_DIR":
                        root
                },
                clear=False,
            ):
                def _run(
                    request_id,
                    body,
                    snapshot,
                ):
                    return (
                        arbitration
                        .update_memory_source_arbitration(
                            conversation_id=CID,
                            cognitive_request_id=(
                                request_id
                            ),
                            snapshot=snapshot,
                            expected_unified_revision=5,
                            forward_body=body,
                        )
                    )

                with ThreadPoolExecutor(
                    max_workers=2
                ) as pool:
                    future_a = pool.submit(
                        _run,
                        "ctxreq_" + "4" * 32,
                        body_a,
                        _snapshot(
                            [
                                _memory(
                                    "a1", "OB-A"
                                )
                            ]
                        ),
                    )
                    future_b = pool.submit(
                        _run,
                        "ctxreq_" + "5" * 32,
                        body_b,
                        _snapshot(
                            [
                                _memory(
                                    "b1", "OB-B"
                                ),
                                _memory(
                                    "b2", "OB-B2"
                                ),
                            ]
                        ),
                    )
                    future_a.result()
                    future_b.result()

            artifacts = _read_all(
                _arbitration_dir(root)
            )

        self.assertEqual(len(artifacts), 2)

        by_request = {
            artifact["cognitive_request_id"]:
                artifact
            for artifact in artifacts
        }

        first = by_request[
            "ctxreq_" + "4" * 32
        ]
        second = by_request[
            "ctxreq_" + "5" * 32
        ]

        self.assertEqual(
            first["metrics"]["ob_candidate_count"],
            1,
        )
        self.assertEqual(
            first["metrics"]["frontend_body_chars"],
            len("FRONT-A"),
        )
        self.assertEqual(
            second["metrics"]["ob_candidate_count"],
            2,
        )
        self.assertEqual(
            second["metrics"]["frontend_body_chars"],
            len("FRONT-B"),
        )


# ---------------------------------------------------------------------
# Coordinator integration + regression invariants
# ---------------------------------------------------------------------


class ArbitrationCoordinatorTests(
    unittest.IsolatedAsyncioTestCase,
):

    async def _run_pipeline(
        self,
        body,
        *,
        root,
        arbitration_on,
        snapshot,
    ):
        environ = {
            **_ISOLATED_ENV,
            _ARBITRATION_ENV:
                "1" if arbitration_on else "0",
            "OMBRE_CONTEXT_STATE_DIR": root,
        }

        with patch.dict(
            os.environ,
            environ,
            clear=False,
        ):
            with patch.object(
                coordinator,
                "observe_context_sources",
                Mock(return_value=CID),
            ), patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                AsyncMock(
                    return_value={
                        "stored": True,
                        "revision": 5,
                        "memory_shadow_snapshot":
                            snapshot,
                    }
                ),
            ):
                selected = await (
                    coordinator.run_context_pipeline(
                        body
                    )
                )

        return selected

    async def test_flag_off_changes_nothing(
        self,
    ):
        body = _frontend_current_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            selected = await self._run_pipeline(
                body,
                root=root,
                arbitration_on=False,
                snapshot=_snapshot(
                    [_memory("m1", "OB-A")]
                ),
            )

            # Provider request bytes are exactly the forwarded bytes
            # and no arbitration artifact / log is produced.
            self.assertEqual(selected, body)
            self.assertFalse(
                _arbitration_dir(root).exists()
            )
            self.assertFalse(
                (
                    Path(root)
                    / "memory_source_arbitration"
                ).exists()
            )

    async def test_flag_on_keeps_bytes_and_observes(
        self,
    ):
        body = _frontend_current_body("MEM-A")

        with tempfile.TemporaryDirectory() as root:
            selected = await self._run_pipeline(
                body,
                root=root,
                arbitration_on=True,
                snapshot=_snapshot(
                    [_memory("m1", "OB-A")]
                ),
            )

            artifacts = _read_all(
                _arbitration_dir(root)
            )

            # No usage / recall / lifecycle / flash / ledger
            # artifacts exist: no false reinforcement was produced.
            children = sorted(
                path.name
                for path in Path(root).iterdir()
            )

        # Observation-only: the model sees exactly the same bytes.
        self.assertEqual(selected, body)

        # The Frontend Memory attachment is still present verbatim.
        self.assertIn(
            b"relevant_memories.txt", selected
        )
        self.assertIn(b"MEM-A", selected)

        self.assertEqual(len(artifacts), 1)

        artifact = artifacts[0]

        self.assertEqual(
            artifact["decision"], "observed"
        )
        self.assertTrue(
            artifact["ob_observation_available"]
        )
        self.assertEqual(
            artifact["metrics"][
                "frontend_model_visible"
                "_attachment_count"
            ],
            1,
        )
        # The model-visible footprint is measured from the exact
        # envelope that is really in the request.
        self.assertEqual(
            artifact["metrics"][
                "frontend_model_visible_chars"
            ],
            len(_current_raw("MEM-A")),
        )
        self.assertGreaterEqual(
            artifact["metrics"][
                "frontend_model_visible"
                "_estimated_tokens"
            ],
            artifact["metrics"][
                "frontend_body_estimated_tokens"
            ],
        )

        self.assertEqual(
            children,
            ["memory_source_arbitration"],
        )

    async def test_off_and_on_produce_identical_bytes(
        self,
    ):
        body = _frontend_history_body("MEM-HIST")

        snapshot = _snapshot(
            [_memory("m1", "OB-A")]
        )

        with tempfile.TemporaryDirectory() as off_root:
            selected_off = await self._run_pipeline(
                body,
                root=off_root,
                arbitration_on=False,
                snapshot=snapshot,
            )

        with tempfile.TemporaryDirectory() as on_root:
            selected_on = await self._run_pipeline(
                body,
                root=on_root,
                arbitration_on=True,
                snapshot=snapshot,
            )

        self.assertEqual(selected_off, body)
        self.assertEqual(selected_on, body)
        self.assertEqual(
            selected_off, selected_on
        )

    async def test_two_requests_same_conversation_get_distinct_artifacts(
        self,
    ):
        body_a = _frontend_current_body("MEM-A")
        body_b = _frontend_current_body("MEM-B")

        with tempfile.TemporaryDirectory() as root:
            selected_a = await self._run_pipeline(
                body_a,
                root=root,
                arbitration_on=True,
                snapshot=_snapshot(
                    [_memory("m1", "OB-A")]
                ),
            )
            selected_b = await self._run_pipeline(
                body_b,
                root=root,
                arbitration_on=True,
                snapshot=_snapshot(
                    [
                        _memory("m1", "OB-A"),
                        _memory("m2", "OB-B"),
                    ]
                ),
            )

            artifacts = _read_all(
                _arbitration_dir(root)
            )

        self.assertEqual(selected_a, body_a)
        self.assertEqual(selected_b, body_b)
        self.assertEqual(len(artifacts), 2)

        by_chars = sorted(
            artifact["metrics"]["frontend_body_chars"]
            for artifact in artifacts
        )

        self.assertEqual(
            by_chars,
            [len("MEM-A"), len("MEM-B")],
        )

    async def test_malformed_frontend_does_not_fail_request(
        self,
    ):
        body = _body(
            [
                {
                    "role": "user",
                    "content": (
                        '<attachment '
                        'filename="relevant_memories.txt">'
                        "unterminated"
                    ),
                }
            ]
        )

        with tempfile.TemporaryDirectory() as root:
            selected = await self._run_pipeline(
                body,
                root=root,
                arbitration_on=True,
                snapshot=_snapshot(
                    [_memory("m1", "OB-A")]
                ),
            )

            artifacts = _read_all(
                _arbitration_dir(root)
            )

        self.assertEqual(selected, body)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(
            artifacts[0]["metrics"][
                "frontend_attachment_count"
            ],
            0,
        )

    async def test_persistence_failure_is_fail_open(
        self,
    ):
        body = _frontend_current_body("MEM-A")

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        with tempfile.TemporaryDirectory() as root:
            with patch.object(
                arbitration,
                "atomic_write",
                _boom,
            ):
                selected = await self._run_pipeline(
                    body,
                    root=root,
                    arbitration_on=True,
                    snapshot=_snapshot(
                        [_memory("m1", "OB-A")]
                    ),
                )

            leftover = (
                _arbitration_dir(root).exists()
            )

        # A persistence failure must not fail the request, must not
        # change the provider bytes and must not become a live
        # availability dependency.
        self.assertEqual(selected, body)
        self.assertFalse(leftover)

    async def test_arbitration_failure_does_not_block_downstream(
        self,
    ):
        body = _frontend_current_body("MEM-A")

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        observed = {"preview": False}

        with tempfile.TemporaryDirectory() as root:
            environ = {
                **_ISOLATED_ENV,
                _ARBITRATION_ENV: "1",
                "OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW":
                    "1",
                "OMBRE_CONTEXT_STATE_DIR": root,
            }

            with patch.dict(
                os.environ,
                environ,
                clear=False,
            ), patch.object(
                coordinator,
                "observe_context_sources",
                Mock(return_value=CID),
            ), patch.object(
                coordinator,
                "update_unified_context_candidate_from_runtime",
                AsyncMock(
                    return_value={
                        "stored": True,
                        "revision": 5,
                        "memory_shadow_snapshot":
                            _snapshot(
                                [
                                    _memory(
                                        "m1", "OB-A"
                                    )
                                ]
                            ),
                    }
                ),
            ), patch.object(
                arbitration,
                "atomic_write",
                _boom,
            ), patch.object(
                coordinator,
                "update_context_injection_preview",
                Mock(
                    side_effect=lambda *a, **k: (
                        observed.__setitem__(
                            "preview", True
                        )
                        or {"stored": True}
                    )
                ),
            ):
                selected = await (
                    coordinator.run_context_pipeline(
                        body
                    )
                )

        self.assertEqual(selected, body)
        # The downstream preview stage still ran even though the
        # arbitration shadow failed to persist.
        self.assertTrue(observed["preview"])


if __name__ == "__main__":
    unittest.main()
