from __future__ import annotations

import json
import unittest

from ombrebrain.context import (
    memory_source_evidence as evidence,
)
from ombrebrain.context.memory_source_evidence import (
    MemorySourceEvidenceSnapshot,
    SCOPE_CURRENT,
    SCOPE_HISTORY,
    SCOPE_RETRIEVAL,
    SOURCE_FRONTEND,
    SOURCE_OB,
    build_memory_source_evidence_snapshot,
    frontend_evidence_units,
    ob_evidence_units,
)
from ombrebrain.gateway import (
    FrontendMemorySegment,
    OperitAdapter,
    extract_frontend_memory_segments,
)


CID = "ctx_0123456789abcdef"

_SENTINEL_FRONTEND = "FRONTEND-SECRET-BODY"
_SENTINEL_OB = "OB-SECRET-CONTENT"


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


def _frontend_current_body(*bodies):
    text = "current question " + " ".join(
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


def _memory(memory_id, content):
    return {"id": memory_id, "content": content}


def _unified(memories, revision=5):
    return {
        "version": "unified-context-candidate.v1",
        "conversation_id": CID,
        "revision": revision,
        "sections": {"memories": memories},
    }


def _snapshot(memories, revision=5):
    return {
        "version": "memory-shadow-snapshot.v1",
        "mode": "shadow_only",
        "conversation_id": CID,
        "revision": revision,
        "unified": _unified(memories, revision=revision),
    }


def _segments(forward_body):
    payload = json.loads(forward_body)
    request = OperitAdapter().adapt(payload)
    return extract_frontend_memory_segments(request)


class AdapterContractTests(unittest.TestCase):

    def test_contract_version(self):
        self.assertEqual(
            evidence._VERSION,
            "memory-source-evidence.v1",
        )

    def test_one_attachment_is_one_unit(self):
        units = frontend_evidence_units(
            _segments(
                _frontend_current_body("MEM-A")
            )
        )

        self.assertEqual(len(units), 1)

        unit = units[0]

        self.assertEqual(unit.source, SOURCE_FRONTEND)
        self.assertEqual(
            unit.kind, "frontend_attachment"
        )
        self.assertEqual(
            unit.granularity, "attachment"
        )
        self.assertEqual(unit.source_scope, SCOPE_CURRENT)
        self.assertEqual(unit.content_chars, 5)
        self.assertTrue(unit.currently_model_visible)
        self.assertFalse(unit.retrieved)

    def test_inner_body_is_never_split(self):
        # List-like, JSON-like, XML-like, blank-line paragraphs and
        # duplicate sentences all stay ONE attachment unit.
        bodies = (
            "- one\n- two\n- three",
            '{"a": 1, "b": 2}',
            "<memory><item>x</item></memory>",
            "para one\n\npara two\n\npara three",
            "same. same. same.",
        )

        for body in bodies:
            with self.subTest(body=body):
                units = frontend_evidence_units(
                    _segments(
                        _frontend_current_body(
                            body
                        )
                    )
                )
                self.assertEqual(len(units), 1)

    def test_duplicate_attachments_are_distinct_occurrences(
        self,
    ):
        units = frontend_evidence_units(
            _segments(
                _frontend_current_body(
                    "DUP", "DUP"
                )
            )
        )

        self.assertEqual(len(units), 2)
        self.assertEqual(
            units[0].content_digest,
            units[1].content_digest,
        )
        self.assertNotEqual(
            units[0].request_occurrence,
            units[1].request_occurrence,
        )

    def test_history_placement_is_preserved(self):
        body = _body(
            [
                {
                    "role": "user",
                    "content": (
                        "old "
                        + _attachment(
                            "relevant_memories.txt",
                            "HIST",
                            "hist",
                        )
                    ),
                },
                {
                    "role": "assistant",
                    "content": "a",
                },
                {
                    "role": "user",
                    "content": "now",
                },
            ]
        )

        units = frontend_evidence_units(
            _segments(body)
        )

        self.assertEqual(len(units), 1)
        self.assertEqual(
            units[0].source_scope, SCOPE_HISTORY
        )

    def test_plain_text_is_never_reclassified(self):
        body = _body(
            [
                {
                    "role": "user",
                    "content": (
                        "please read "
                        "relevant_memories.txt"
                    ),
                }
            ]
        )

        self.assertEqual(
            frontend_evidence_units(
                _segments(body)
            ),
            [],
        )

    def test_one_ob_candidate_is_one_unit(self):
        units = ob_evidence_units(
            [_memory("m1", "OB-A")]
        )

        self.assertEqual(len(units), 1)

        unit = units[0]

        self.assertEqual(unit.source, SOURCE_OB)
        self.assertEqual(unit.kind, "ob_candidate")
        self.assertEqual(
            unit.granularity, "candidate"
        )
        self.assertEqual(
            unit.source_scope, SCOPE_RETRIEVAL
        )
        self.assertFalse(unit.currently_model_visible)
        self.assertTrue(unit.retrieved)
        self.assertIsInstance(
            unit.candidate_fingerprint, str
        )

    def test_duplicate_ob_candidates_are_distinct(
        self,
    ):
        units = ob_evidence_units(
            [
                _memory("m1", "DUP"),
                _memory("m2", "DUP"),
            ]
        )

        self.assertEqual(len(units), 2)
        self.assertNotEqual(
            units[0].request_occurrence,
            units[1].request_occurrence,
        )


class AvailabilityTests(unittest.TestCase):

    def _build(
        self,
        *,
        forward_body,
        snapshot,
        revision=5,
    ):
        return build_memory_source_evidence_snapshot(
            conversation_id=CID,
            cognitive_request_id=(
                "ctxreq_" + "a" * 32
            ),
            source_unified_revision=revision,
            forward_body=forward_body,
            memory_shadow_snapshot=snapshot,
        )

    def test_both_available(self):
        snap = self._build(
            forward_body=_frontend_current_body(
                "MEM-A"
            ),
            snapshot=_snapshot([_memory("m1", "OB-A")]),
        )

        self.assertTrue(
            snap.frontend_observation_available
        )
        self.assertTrue(
            snap.ob_observation_available
        )
        self.assertEqual(len(snap.units), 2)

    def test_malformed_frontend_valid_ob(self):
        snap = self._build(
            forward_body=b"{not json",
            snapshot=_snapshot([_memory("m1", "OB-A")]),
        )

        self.assertFalse(
            snap.frontend_observation_available
        )
        self.assertTrue(
            snap.ob_observation_available
        )
        # Only the OB unit exists.
        self.assertEqual(len(snap.units), 1)
        self.assertEqual(
            snap.units[0].source, SOURCE_OB
        )

    def test_valid_frontend_invalid_ob(self):
        snap = self._build(
            forward_body=_frontend_current_body(
                "MEM-A"
            ),
            snapshot={"version": "wrong"},
        )

        self.assertTrue(
            snap.frontend_observation_available
        )
        self.assertFalse(
            snap.ob_observation_available
        )
        self.assertEqual(len(snap.units), 1)
        self.assertEqual(
            snap.units[0].source, SOURCE_FRONTEND
        )

    def test_canonicalizable_with_zero_attachments(self):
        snap = self._build(
            forward_body=_body(
                [
                    {
                        "role": "user",
                        "content": "hello",
                    }
                ]
            ),
            snapshot=_snapshot([]),
        )

        # Available, but zero units -- NOT the same as unavailable.
        self.assertTrue(
            snap.frontend_observation_available
        )
        self.assertTrue(
            snap.ob_observation_available
        )
        self.assertEqual(snap.units, ())

    def test_foreign_conversation_marks_ob_unavailable(
        self,
    ):
        snap = self._build(
            forward_body=_frontend_current_body(
                "MEM-A"
            ),
            snapshot={
                "version": "memory-shadow-snapshot.v1",
                "conversation_id":
                    "ctx_ffffffffffffffff",
                "unified": _unified(
                    [_memory("m1", "OB-A")]
                ),
            },
        )

        self.assertFalse(
            snap.ob_observation_available
        )

    def test_wrong_revision_marks_ob_unavailable(self):
        snap = self._build(
            forward_body=_frontend_current_body(
                "MEM-A"
            ),
            snapshot=_snapshot(
                [_memory("m1", "OB-A")],
                revision=99,
            ),
        )

        self.assertFalse(
            snap.ob_observation_available
        )

    def test_never_reads_shared_unified_from_disk(
        self,
    ):
        # The adapter only reads THIS request's snapshot object; a
        # decoy on disk is never consulted.
        snap = self._build(
            forward_body=_frontend_current_body(
                "MEM-A"
            ),
            snapshot=_snapshot([]),
        )

        self.assertTrue(
            snap.ob_observation_available
        )
        self.assertEqual(
            len(
                [
                    u
                    for u in snap.units
                    if u.source == SOURCE_OB
                ]
            ),
            0,
        )


class InMemoryOnlyTests(unittest.TestCase):

    def test_repr_never_carries_source_payload(self):
        # The request-local snapshot keeps source-native references for
        # downstream reuse, so ``repr`` must never surface their text:
        # the workspace repr may be logged by a debugging caller.
        snap = build_memory_source_evidence_snapshot(
            conversation_id=CID,
            cognitive_request_id=(
                "ctxreq_" + "b" * 32
            ),
            source_unified_revision=5,
            forward_body=_frontend_current_body(
                _SENTINEL_FRONTEND
            ),
            memory_shadow_snapshot=_snapshot(
                [_memory("m1", _SENTINEL_OB)]
            ),
        )

        rendered = repr(snap)

        self.assertNotIn(
            _SENTINEL_FRONTEND, rendered
        )
        self.assertNotIn(_SENTINEL_OB, rendered)

    def test_snapshot_is_not_json_serializable(self):
        snap = build_memory_source_evidence_snapshot(
            conversation_id=CID,
            cognitive_request_id=(
                "ctxreq_" + "c" * 32
            ),
            source_unified_revision=5,
            forward_body=_frontend_current_body(
                "MEM-A"
            ),
            memory_shadow_snapshot=_snapshot([]),
        )

        with self.assertRaises(TypeError):
            json.dumps(snap)


class BindingTests(unittest.TestCase):

    def _snapshot_obj(self):
        return build_memory_source_evidence_snapshot(
            conversation_id=CID,
            cognitive_request_id=(
                "ctxreq_" + "d" * 32
            ),
            source_unified_revision=5,
            forward_body=_frontend_current_body(
                "MEM-A"
            ),
            memory_shadow_snapshot=_snapshot([]),
        )

    def test_bound_to_request(self):
        snap = self._snapshot_obj()

        self.assertTrue(
            evidence.is_bound_to_request(
                snap,
                conversation_id=CID,
                cognitive_request_id=(
                    "ctxreq_" + "d" * 32
                ),
                expected_unified_revision=5,
            )
        )

    def test_foreign_request_is_not_bound(self):
        snap = self._snapshot_obj()

        self.assertFalse(
            evidence.is_bound_to_request(
                snap,
                conversation_id=CID,
                cognitive_request_id=(
                    "ctxreq_" + "e" * 32
                ),
                expected_unified_revision=5,
            )
        )

        self.assertFalse(
            evidence.is_bound_to_request(
                snap,
                conversation_id=CID,
                cognitive_request_id=(
                    "ctxreq_" + "d" * 32
                ),
                expected_unified_revision=6,
            )
        )

    def test_non_snapshot_is_not_bound(self):
        self.assertFalse(
            evidence.is_bound_to_request(
                {"version": evidence._VERSION},
                conversation_id=CID,
                cognitive_request_id=(
                    "ctxreq_" + "d" * 32
                ),
                expected_unified_revision=5,
            )
        )


class MalformedObStructureTests(unittest.TestCase):
    """Malformed OB structure is UNAVAILABLE, never "zero candidates".

    Only a valid dict ``sections`` whose ``memories`` is a list yields
    ``ob_observation_available = true``. A malformed shape must never be
    silently reported as a valid empty observation.
    """

    def _valid_empty(self):
        return {
            "version": "memory-shadow-snapshot.v1",
            "mode": "shadow_only",
            "conversation_id": CID,
            "revision": 5,
            "unified": {
                "version": "unified-context-candidate.v1",
                "conversation_id": CID,
                "revision": 5,
                "sections": {"memories": []},
            },
        }

    def test_valid_empty_list_is_available(self):
        memories, available = (
            evidence.ob_memories_from_snapshot(
                self._valid_empty(),
                conversation_id=CID,
                expected_unified_revision=5,
            )
        )

        self.assertEqual(memories, [])
        self.assertTrue(available)

    def test_valid_memories_are_available(self):
        memories, available = (
            evidence.ob_memories_from_snapshot(
                _snapshot([_memory("m1", "OB-A")]),
                conversation_id=CID,
                expected_unified_revision=5,
            )
        )

        self.assertEqual(len(memories), 1)
        self.assertTrue(available)

    def test_malformed_structure_matrix_is_unavailable(
        self,
    ):
        base = {
            "version": "unified-context-candidate.v1",
            "conversation_id": CID,
            "revision": 5,
        }

        def snap(unified):
            return {
                "version": "memory-shadow-snapshot.v1",
                "conversation_id": CID,
                "unified": unified,
            }

        cases = {
            "snapshot_none": None,
            "snapshot_empty": {},
            "snapshot_list": [1, 2],
            "snapshot_wrong_version": {
                "version": "other",
                "conversation_id": CID,
                "unified": {
                    **base,
                    "sections": {"memories": []},
                },
            },
            "snapshot_foreign_conversation": {
                "version": "memory-shadow-snapshot.v1",
                "conversation_id":
                    "ctx_ffffffffffffffff",
                "unified": {
                    **base,
                    "sections": {"memories": []},
                },
            },
            "unified_missing": snap(None),
            "unified_not_dict": snap([1, 2]),
            "unified_wrong_version": snap(
                {
                    **base,
                    "version": "other",
                    "sections": {"memories": []},
                }
            ),
            "unified_foreign_conversation": snap(
                {
                    **base,
                    "conversation_id":
                        "ctx_ffffffffffffffff",
                    "sections": {"memories": []},
                }
            ),
            "unified_wrong_revision": snap(
                {
                    **base,
                    "revision": 99,
                    "sections": {"memories": []},
                }
            ),
            "sections_missing": snap(dict(base)),
            "sections_none": snap(
                {**base, "sections": None}
            ),
            "sections_list": snap(
                {**base, "sections": []}
            ),
            "sections_str": snap(
                {**base, "sections": "x"}
            ),
            "memories_missing": snap(
                {**base, "sections": {}}
            ),
            "memories_none": snap(
                {**base, "sections": {"memories": None}}
            ),
            "memories_dict": snap(
                {**base, "sections": {"memories": {}}}
            ),
            "memories_str": snap(
                {**base, "sections": {"memories": "x"}}
            ),
        }

        for name, snapshot in cases.items():
            with self.subTest(name=name):
                memories, available = (
                    evidence
                    .ob_memories_from_snapshot(
                        snapshot,
                        conversation_id=CID,
                        expected_unified_revision=5,
                    )
                )

                self.assertEqual(memories, [])
                self.assertFalse(available)

    def test_malformed_ob_never_becomes_available_in_build(
        self,
    ):
        snap = build_memory_source_evidence_snapshot(
            conversation_id=CID,
            cognitive_request_id=(
                "ctxreq_" + "a" * 32
            ),
            source_unified_revision=5,
            forward_body=_frontend_current_body(
                "MEM-A"
            ),
            memory_shadow_snapshot={
                "version": "memory-shadow-snapshot.v1",
                "conversation_id": CID,
                "unified": {
                    "version":
                        "unified-context-candidate.v1",
                    "conversation_id": CID,
                    "revision": 5,
                    "sections": {"memories": None},
                },
            },
        )

        # Malformed OB is UNAVAILABLE -- not "zero candidates" -- while
        # the independent Frontend observation stays available.
        self.assertFalse(
            snap.ob_observation_available
        )
        self.assertTrue(
            snap.frontend_observation_available
        )
        self.assertEqual(
            [
                u
                for u in snap.units
                if u.source == SOURCE_OB
            ],
            [],
        )


class ModelVisibleFootprintTests(unittest.TestCase):

    def test_frontend_footprint_is_the_envelope(self):
        units = frontend_evidence_units(
            _segments(
                _frontend_current_body("hi")
            )
        )

        unit = units[0]

        self.assertEqual(unit.content_chars, 2)
        # The exact envelope always carries more than the bare body.
        self.assertGreater(
            unit.model_visible_chars,
            unit.content_chars,
        )
        self.assertGreaterEqual(
            unit.model_visible_estimated_tokens,
            unit.content_estimated_tokens,
        )

    def test_ob_footprint_is_exactly_zero(self):
        unit = ob_evidence_units(
            [_memory("m1", "OB-A")]
        )[0]

        self.assertEqual(unit.model_visible_chars, 0)
        self.assertEqual(
            unit.model_visible_estimated_tokens, 0
        )

    def test_segment_without_envelope_is_refused(self):
        # No canonical raw attachment => no proven model-visible
        # envelope, so the unit is refused rather than measured as its
        # bare body.
        segment = FrontendMemorySegment(
            placement=SCOPE_CURRENT,
            source_message_index=0,
            source_block_index=0,
            body="BODY",
            raw_attachment="",
        )

        self.assertEqual(
            frontend_evidence_units([segment]), []
        )


class NoSilentTruncationTests(unittest.TestCase):

    def test_all_frontend_occurrences_are_kept(self):
        count = 520

        bodies = [
            "M-%d" % index
            for index in range(count)
        ]

        units = frontend_evidence_units(
            _segments(
                _frontend_current_body(*bodies)
            )
        )

        self.assertEqual(len(units), count)

    def test_all_ob_occurrences_are_kept(self):
        count = 520

        units = ob_evidence_units(
            [
                _memory("m%d" % index, "C-%d" % index)
                for index in range(count)
            ]
        )

        self.assertEqual(len(units), count)


if __name__ == "__main__":
    unittest.main()
