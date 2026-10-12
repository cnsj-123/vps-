# Frontend Controlled Direct Channel + Unified Memory Authority Compatibility Candidate Shadow v0

```
WORKING EXPERIMENT
COMPATIBILITY MIGRATION LANE
NOT ADR
NOT LIVE AUTHORITY
NOT FROZEN SPEC
```

This document describes an **experiment**, not a contract. If this document and
the code disagree, the code is correct and this document is wrong. Nothing here is
normative, nothing here is frozen, and nothing here grants runtime authority.

It defines the **first target-body candidate** this repository allows, and a
deliberately ultra-narrow one:

```
Frontend attachment
    -> source-specific Frontend Controlled Direct Surface
        -> explicit controller decision PRESERVE_DIRECT_DATA
            -> candidate final body is EXACTLY the same bytes
```

The model-visible content is unchanged. What changes is **who owns the decision**:
today Frontend content is model-visible because it simply bypasses control; the
candidate makes it model-visible because unified authority **explicitly authorized
the exact attachment occurrence** to remain direct.

Companion documents:

- [`ARCHITECTURE_STATUS.md`](ARCHITECTURE_STATUS.md) — the reality map of the whole
  cognitive-memory chain (this experiment is recorded there as §2.28).
- [`MEMORY_AUTHORITY_MIGRATION_SHADOW.md`](MEMORY_AUTHORITY_MIGRATION_SHADOW.md) —
  the Migration Readiness Shadow this experiment reads as required upstream
  evidence; it records the CURRENT Frontend path as a `gap_frontend_direct_bypass`.
- [`UNIFIED_MEMORY_DELIVERY.md`](UNIFIED_MEMORY_DELIVERY.md) — the Unified Memory
  Delivery Plan Shadow, the upstream control contract.

Owner modules:

- `src/ombrebrain/context/frontend_controlled_direct_surface.py` — the in-memory
  Frontend Controlled Direct Surface, contract
  `frontend-controlled-direct-surface.v1`.
- `src/ombrebrain/context/memory_live_authority_candidate_shadow.py` — the Unified
  Memory Authority Compatibility Candidate Shadow, artifact
  `memory-live-authority-candidate-shadow.v1`.
- `src/ombrebrain/gateway/canonicalizer.py` — the public canonical attachment
  control transform `rewrite_frontend_memory_attachments(...)`.
- `src/ombrebrain/context/context_pipeline_coordinator.py` — integration
  (post-final-body observer).
- `src/ombrebrain/context/context_request_workspace.py` — the request-local
  `frontend_controlled_direct_surface` / `memory_live_authority_candidate` fields.
- `src/ombrebrain/maintenance/memory_live_authority_candidate_report.py` — read-only
  aggregate reporter.

The observable output is a **candidate proposition**, produced after the final
provider body has already been chosen, from evidence THIS request already produced.
It is evidence, not a decision. **`live_cutover_ready` is always `false` in v0.**

---

## 1. Why Frontend control does NOT require immediate Frontend Flash

The North Star is that every memory system may keep its own brain and storage, but
must not each own a mouth that speaks directly to the model. Today the Frontend
`relevant_memories.txt` attachment is already present in the forwarded provider
request, so the model sees the whole attachment with **no controller-owned delivery
decision** — Migration records this as `gap_frontend_direct_bypass`.

The obvious-looking fix — convert Frontend Memory to Flash — is **not** taken here.
Flash requires a bounded cue, an opaque memref, a Recall path and item-level
identity. Frontend has none of those, and inventing them is out of scope
(§18, §19). Instead this task introduces a **zero-behaviour-change compatibility
lane** that gives unified control ownership of the decision while leaving the
content exactly as it is. `preserve_direct_data` is a **migration lane**, not a
statement that Frontend `direct_data` is the permanent policy.

## 2. Attachment-level granularity

The controlled unit is exactly one **current Frontend attachment occurrence** — the
same granularity the Source Evidence Adapter and Migration already use. There is no
Frontend item parser, no bundle/sub-item identity, and no per-memory split inside an
attachment. `evidence_ref` remains a **request-occurrence reference**, never a
persistent identity. Two identical attachment envelopes remain two distinct
occurrences.

## 3. The compatibility preserve_direct_data action

The closed v0 high-level Frontend action enum contains exactly one value:

```
preserve_direct_data
```

Nothing else. No `flash`, `summarize`, `recall`, `merge`, `dedup` or `drop` is added
to the high-level Frontend surface policy in this task. Every successfully bound
Frontend occurrence receives `preserve_direct_data`, which the low-level transform
realizes as `keep_exact`.

## 4. Controller ownership vs content representation

This lane separates two things that were previously fused:

- **ownership of the decision** — now the unified controller explicitly authorizes
  each Frontend occurrence;
- **representation of the content** — unchanged: the same attachment bytes reach
  the model.

The candidate proves that ownership can move to unified control **without** changing
the representation. That is the whole point of a compatibility migration lane.

## 5. Source Evidence / Delivery / Migration strict binding

The Candidate is built **only** from THIS request's already-validated request-local
authority. It never reads a persisted "latest" file and never re-runs retrieval,
arbitration, Surfacing Policy or Migration.

- **Source Evidence** — `version = memory-source-evidence.v1`, exact
  `conversation_id` / `cognitive_request_id` / `source_unified_revision`, and
  `frontend_observation_available == true` **and** `ob_observation_available == true`.
  An unavailable plane is never turned into "zero units".
- **Delivery Plan** — `is_valid_memory_delivery_plan_artifact(plan)` with the exact
  shadow-only / proposal-only / `live_authority=false` contract, the base policy
  stage, and the exact request / revision.
- **Migration** — `is_valid_memory_authority_migration_artifact(migration)` with
  `mode=shadow_only`, `authority=comparison_only`, `live_authority=false`,
  `live_cutover_ready=false`, the exact request / revision, and
  `frontend_final_observation_available == true` **and**
  `frontend_final_presence_verified == true`.

For every Frontend Source Evidence unit the Candidate recomputes
`evidence_ref_for_fields(source, kind, source_occurrence, content_digest)` and
requires **exactly one** matching persisted Frontend Delivery row and **exactly one**
matching persisted Frontend Migration row, with the Migration row carrying
`authority_coverage=gap_frontend_direct_bypass` and
`replacement_requirement=controlled_frontend_channel_required`. Missing /
duplicate / foreign rows, forged refs, mismatched digests or occurrences produce
**no surface**. Matching is never by body text alone.

## 6. Exact raw envelope binding

This stage is **stricter than Migration**. Migration proves digest, placement,
footprint and occurrence multiset; the Candidate additionally proves, **in memory**,
that the bound final attachment envelope equals the **source-native raw attachment
envelope** for that occurrence. A body whose content is equal but whose attachment
metadata / raw envelope differs produces **no Candidate**
(`frontend_exact_envelope_binding_unavailable`). The comparison is in-memory only:
the raw envelope is never persisted.

## 7. The canonical keep/drop enforcement primitive

The future authority needs an enforcement primitive, so the canonicalizer exposes
one public, narrowly-scoped transform that reuses the **same** private attachment
grammar (`_ATTACHMENT_RE`, `_attachment_attrs`, `_classify_attachment`,
`_parse_single_attachment`) — there is no second attachment parser:

```
rewrite_frontend_memory_attachments(payload, decide)
```

The callback receives the occurrence index, placement, exact raw attachment and
exact inner body, and returns a **low-level** action:

```
keep_exact
drop
```

These are mechanical rewrite primitives, **not** high-level Memory policy. The
transform is **fail-closed as a pure transform**: an unknown action, a refused
occurrence, an occurrence-count mismatch, an unparseable target attachment or an
unsupported payload makes the whole result a **failure** — a partially rewritten
payload is never returned, and the original input object is left untouched.

## 8. Why high-level v0 emits keep only

High-level production code in v0 emits **only** `preserve_direct_data` → `keep_exact`.
`drop` exists solely so the transform is a genuine enforcement primitive testable for
future selector work; it is **never called** by the Candidate production path. This
is proven with a spy/source-contract test and a static check.

## 9. Why the candidate body is byte-identical

Because every v0 action is `keep_exact`, the rewritten payload **deep-equals** the
decoded final payload. The candidate body is therefore **defined as the original
`final_body` bytes**, and a stored artifact must prove:

```
candidate_body_bytes    == final_body_bytes
candidate_body_sha256   == final_body_sha256
candidate_body_byte_identical == true
```

This is the zero-behaviour-change compatibility proof: the controller has visited and
explicitly authorized every Frontend occurrence while preserving the exact current
provider bytes. It is **not** a semantic-dedup claim (`Frontend == OB` is never
asserted).

## 10. Why original bytes are reused instead of JSON reserialization

`json.dumps(rewritten_payload)` is **not** called, even when all actions are
`keep_exact`. Re-serializing could change whitespace, escaping or formatting with no
semantic reason. Reusing the original final bytes guarantees the model sees exactly
what it saw before — the compatibility lane changes **nothing** the model can
observe.

## 11. OB control-plane delegation

OB is treated as **already controlled by the existing OB control contract**. The
Candidate does **not** rebuild Live Exposure, does **not** create another memref or
cue, does **not** reread Recall Surface and does **not** call
`select_live_memory_flash_body`. It maps Migration's validated full-scope metrics:

```
Migration covered_ob_controlled_cue_contract      -> Candidate covered_ob_existing_cue_contract / delegated_existing_ob_cue_control
Migration covered_ob_controlled_withhold_contract -> Candidate covered_ob_existing_withhold_contract / delegated_existing_ob_withhold_control
Migration gap_ob_policy_unknown                   -> Candidate gap_ob_policy_unknown / unresolved_ob_policy
```

The artifact explicitly declares `ob_control_strategy =
delegate_existing_ob_control_plane`. It does **not** claim the new Candidate renderer
owns OB rendering; a future live unification may absorb the OB selector.

## 12. Remaining OB unknown gaps

`UNKNOWN != DROP`. The compatibility candidate does **not** magically resolve OB
policy ambiguity: an OB unit whose policy is unknown stays a gap. While
`remaining_gap_unit_count > 0`, `candidate_coverage_status = blocked`.

## 13. Row truncation / Frontend completeness rule

The persisted Delivery Plan and Migration artifacts may truncate rows at a cap. The
Candidate **cannot** authorize a Frontend occurrence whose actual row is hidden by
truncation, so it requires:

```
persisted Frontend Delivery rows  == full-scope frontend unit count
persisted Frontend Migration rows == full-scope frontend_direct_bypass_count
```

If any Frontend occurrence is hidden, no Candidate artifact is produced
(`frontend_control_rows_incomplete`); hidden Frontend actions are **never** fabricated
from aggregate counts. OB rows may remain truncated because OB is not remapped per
row here; hidden OB identities are never invented, and full OB counts come from the
validated Migration metrics (`source_rows_truncated`, `source_rows_truncated_count`).

## 14. Request-local authority

The Candidate runtime authority is the **request-local workspace** THIS request just
produced (`memory_source_evidence` / `memory_delivery_plan` /
`memory_authority_migration`) plus the request-local Frontend Controlled Surface. It
never reads a conversation-global "latest". The Frontend Controlled Surface is built
**exactly once** per request, is **in-memory only**, and is never persisted, logged,
cached or given to the model. Two concurrent requests on the same conversation cannot
contaminate each other.

### Read-only aggregate reporter

`memory_live_authority_candidate_report.py` aggregates the persisted artifacts without
creating a database, a rolling accumulator or any runtime authority. It **path-binds**
every file: a candidate artifact is accepted only when its embedded conversation /
request ids match the `<conversation_id>/<cognitive_request_id>.json` path it was read
from, so one request artifact copied under a foreign request filename or conversation
directory is counted invalid and contributes **zero** to every statistic. Its `--limit`
(default 500) is the **maximum** artifact count, not a guaranteed count: the reporter
uses a **true bounded filesystem traversal** via `os.scandir`, so directory entries are
produced lazily and the namespace is never fully listed or sorted. Traversal stops as
soon as **either** `limit` artifacts have been selected **or** a hard entry-visit budget
(`max(2 * limit, limit)`) is reached — every top-level and child entry examined counts
against that budget. The scan is therefore genuinely bounded and may **underfill** (a
bounded prefix full of empty / junk / non-JSON entries yields fewer than `limit`
artifacts); it does **not** globally sort or enumerate the full namespace, and it returns
files in bounded traversal order rather than claiming a globally lexicographic prefix.
Aggregate-only scan telemetry (`scan_visit_budget`, `scan_visited_entry_count`,
`scan_budget_exhausted`) lets a reviewer distinguish "only N artifacts exist" from "the
scan budget ended after observing N artifacts". The reporter never emits a "safe to cut
over" state and never mutates anything it reads.

## 15. Privacy

Persisted artifacts and logs carry only privacy-safe data: request-binding ids,
revisions, `evidence_ref`, `content_digest`, counts, booleans, enums, sizes and
`sha256` digests. They never contain the final body, the candidate body, a raw
attachment, Frontend body text, OB content, a raw memory id, a cue, a memref, user
text, a query or a system prompt. The in-memory surface may hold the exact raw
Frontend attachment, but it must never be serialized or logged. Candidate logs allow
only mode / stored / decision / reason / strategy / `source_unified_revision` /
`compatibility_candidate_ready` / `candidate_coverage_status` / counts and metrics —
no conversation id, request id, evidence ref, digest, memory id, memref, cue or
attachment text.

The v1 candidate artifact and **every** candidate row use **closed key schemas**: the
validator requires the exact key set the writer emits, so an unknown field — a raw
attachment, a body, a cue, a memory id, a memref or any future extension — makes the
**whole artifact invalid**. This makes the artifact contract itself forbid secrets,
rather than relying only on the writer not emitting them. A new field therefore
requires a new artifact version.

Each candidate row additionally carries a **privacy-safe numeric footprint**
(`model_visible_chars`, `model_visible_estimated_tokens`). Frontend rows copy the exact
in-memory bound surface footprint; OB rows are always exactly zero. Only these numbers
are persisted — never the raw envelope, cue, memref or body. Because the complete
Frontend row set is always persisted, the aggregate Frontend footprint metric must
equal the exact sum of the Frontend row footprints, so a forged aggregate or a single
forged row footprint cannot balance the algebra.

`read_memory_live_authority_candidate(...)` is request-scoped and read-only, and it
re-validates: it returns an artifact only when the authoritative validator accepts it
**and** its embedded conversation / request ids match the requested ones.

## 16. compatibility_candidate_ready

A separate boolean, true **only** when: both source planes are available; the Source
Evidence / Delivery / Migration bindings are valid; every Frontend occurrence has a
persisted control row; exact final raw envelope binding succeeds; every Frontend
action is `preserve_direct_data`; the candidate body is byte-identical to the final
body; and `remaining_gap_unit_count == 0`. It means the zero-behaviour-change
compatibility candidate is **coherent** — it does **not** mean live cutover is
approved.

## 17. live_cutover_ready is always false

For v0:

```
live_cutover_ready = false
```

always. When the compatibility candidate is not ready the reason is
`compatibility_candidate_incomplete`; otherwise it is
`live_selector_rollout_not_implemented`. There is still no live authority feature
flag, no live selector receipt, no canary rollout, no kill switch and no rollback
contract.

## 18. Open questions still unresolved

This task does **not** answer:

- **A.** When should Frontend `direct_data` become `flash_cue`? — OPEN.
- **B.** Should some Frontend bundle remain narrow trusted Direct Data? — OPEN.
- **C.** How will Frontend obtain stable item identity, if ever? — OPEN.
- **D.** Should Frontend get explicit Recall semantics? — OPEN.
- semantic overlap policy — OPEN.
- cross-source priority / conflicts — OPEN.
- Lifecycle composition — OPEN.
- live rollout / kill switch — OPEN.

`compatibility candidate exists` — **not** `live unified authority exists`, **not**
`Frontend mouth removed`, **not** `Frontend converted to Flash`, **not** `live
cutover approved`.

## 19. What the NEXT live-selector phase would require

This task deliberately implements **none** of it. A future live phase would still
need, at minimum:

- resolution of the delivery open questions above;
- a live authority feature flag with an explicit ownership contract;
- a live unified selector that can actually render a non-byte-identical target body
  with the existing content-integrity checks;
- a live-selector receipt (request-scoped, privacy-safe);
- a canary / percentage rollout, a kill switch and a rollback contract;
- the aggregate evidence (from the read-only reporter) that the compatibility lane
  is coherent over real traffic.

Until all of that exists, the candidate body is only ever a **proof of coherence**,
and the returned bytes remain the already-selected `final_body`.

---

## Explicit non-goals of this experiment

Not implemented, and must not be read into this work:

- no live unified selector, live rollout, percentage / canary rollout, kill switch,
  live-selector receipt or rollback contract;
- no Frontend removal / gating / rewriting / deletion / DROP; no Frontend Flash, cue,
  memref, Recall or stable item identity;
- no item-level Frontend parser; no bundle identity; no global / attachment memory id;
- no semantic dedup, embedding overlap, LLM judge, cross-source scoring or reranking;
- no OB re-implementation; OB control stays delegated to the existing OB plane;
- no retrieval / memory mutation (no touch / reinforce / archive / delete / restore /
  record_usage / lifecycle write);
- no provider-body mutation; no Lifecycle-aware Delivery; no ArchiveCandidate; no
  consolidation;
- no resolution of the open questions A–D above.