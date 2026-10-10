# Unified Memory Delivery Shadow v0

```
WORKING EXPERIMENT / SHADOW CONTROL PLANE
NOT ADR
NOT FROZEN SPECIFICATION
```

This document describes an **experiment**, not a contract. If this document and
the code disagree, the code is correct and this document is wrong. Nothing here
is normative, nothing here is frozen, and nothing here grants runtime authority.

Companion documents:

- [`ARCHITECTURE_STATUS.md`](ARCHITECTURE_STATUS.md) — the reality map of the
  whole cognitive-memory chain (this experiment is recorded there as §2.25 /
  §2.26).
- [`LIFECYCLE_AUTHORITY_MAP.md`](LIFECYCLE_AUTHORITY_MAP.md) and
  [`LIFECYCLE_SHADOW_COMPARISON.md`](LIFECYCLE_SHADOW_COMPARISON.md) — the
  lifecycle plane, which this experiment **intentionally does not consume**.

Owner modules:

- `src/ombrebrain/context/memory_source_evidence.py` — Source-Neutral Memory
  Evidence Adapter, contract `memory-source-evidence.v1`.
- `src/ombrebrain/context/memory_delivery_shadow.py` — Unified Memory Delivery
  Plan Shadow, artifact `memory-delivery-plan-shadow.v1`.
- `src/ombrebrain/maintenance/memory_delivery_report.py` — read-only aggregate
  reporter.
- `src/ombrebrain/context/context_pipeline_coordinator.py` — integration.
- `src/ombrebrain/context/context_request_workspace.py` — request-local fields.

---

## 1. Federated Memory + Unified Cognitive Control — the North Star

The working direction is:

```
Federated Memory
  -> source-specific logic
  -> evidence adapters
  -> request-local Cognitive Workspace
  -> unified cognitive control
  -> model
```

The point of "federated" is the opposite of "unified storage": each memory
source keeps its **own** storage, its **own** retrieval semantics and its
**own** granularity. What eventually unifies is the **control decision** ("what
does the model see for this request?"), never the storage.

Delivery Shadow v0 is the smallest first step toward the control half: it asks
what delivery **already** exists for each memory unit of one request, without
changing anything.

## 2. Why storage is NOT unified

Storage is not unified because the sources are genuinely different:

- **Frontend Memory** is delivered by the *client* as a
  `relevant_memories.txt` attachment. Its content is already inside the model
  request, preserved by cache rewriting, with no OB id, no revision and no
  retrieval step in the OB sense.
- **OB memory** is a retrieved candidate with a memory id, a revision, an
  eligibility policy and a cue/Recall boundary.

Forcing both into one persisted memory schema would invent identity,
provenance and lifecycle fields that one of the sources simply does not have.
This experiment deliberately does **not** attempt that. Storage stays
federated; only the *delivery proposal* is federated-but-common.

## 3. Source-native vs source-neutral

Two vocabularies must never be conflated:

- **Source-native** — the real object each source owns: a
  `FrontendMemorySegment` (a canonical attachment envelope) or an OB candidate
  dict (`{"id": ..., "content": ...}`). This is what is actually in the request
  or the retrieval result.
- **Source-neutral** — the small **control** vocabulary the consumers agree on:
  `source`, `kind`, `granularity`, `source_scope`, `content_digest`,
  `normalized_digest`, `content_chars`, `content_estimated_tokens`,
  `currently_model_visible`, `retrieved`, `request_occurrence`.

The adapter (contract `memory-source-evidence.v1`) normalizes **only** the
control metadata. It keeps the real source object as an in-memory-only
`native_payload` reference so a downstream shadow observer can reuse the exact
source object rather than reparse it. It never calls that payload a universal
"memory". `native_payload` is excluded from `repr`, is not serializable, and is
never persisted or logged.

## 4. Frontend attachment granularity

**One `relevant_memories.txt` attachment == one evidence unit.**

The repository has **no proven item-level schema** for the inner body of the
attachment. Therefore the adapter never splits an attachment, never parses its
inner text, and treats list-like, JSON-like, XML-like, multi-paragraph and
duplicate-sentence bodies as **one** unit. `granularity` is `"attachment"` and
`kind` is `"frontend_attachment"`.

Extraction reuses the canonicalizer
(`gateway.canonicalizer.extract_frontend_memory_segments`); the attachment
grammar and filename classification are never copied into this module, and
ordinary user text is **never** reclassified as Frontend Memory.

## 5. OB candidate granularity

**One Unified-admitted memory candidate == one evidence unit.** `granularity` is
`"candidate"` and `kind` is `"ob_candidate"`.

OB evidence is read **only** from THIS request's own `memory_shadow_snapshot`,
under strict version / conversation / revision binding. The adapter never
re-reads a conversation-global Unified file and never re-runs retrieval,
embedding, vector search, rerank or `BucketManager`.

## 6. The request-local Source Evidence Adapter

`build_memory_source_evidence_snapshot(...)` returns a
`MemorySourceEvidenceSnapshot` for exactly one
`(conversation_id, cognitive_request_id, source_unified_revision)`.

Hard rules:

- **In-memory only.** The snapshot is never serialized, never persisted, never
  logged, never placed in Unified / Preview / Gate / Mutation / Real Injection
  / the request body / a cache key / a header, and is never model-visible.
- **Not JSON-serializable** by construction (`native_payload` references carry
  real source content).
- **Bounded** to `MAX_EVIDENCE_UNITS = 512` against an unbounded caller pool.
- **Binding is strict**: `is_bound_to_request(snapshot, conversation_id=...,
  cognitive_request_id=..., expected_unified_revision=...)`. A foreign /
  mismatched / stale snapshot is rejected, never partially reused.
- **Occurrence, never identity** (see §13).

Consumers: the existing **Memory Source Arbitration Shadow** (refactored to
reuse it) and the new **Unified Memory Delivery Plan Shadow**.

The coordinator builds the snapshot **exactly once** per request and shares it,
so the forwarded body is canonicalized and the OB side is bound a single time.

## 7. The current two-mouth problem

Today the model receives memory through **two independent mouths** with no
shared authority:

- the **Frontend mouth** — full attachment content, already model-visible, with
  no gate, no revision and no Recall; and
- the **OB mouth** — a bounded Flash cue, gated by eligibility, with full
  content only after an explicit Recall.

There is no single component that can say "for this request, here is the
delivery state across both sources". Delivery Shadow v0 produces exactly that
**proposal** — and nothing more. It does not close either mouth, does not gate
the Frontend mouth, and does not change what the model sees.

## 8. Delivery vocabulary

A Delivery Plan row expresses three orthogonal, independently known facts
instead of one ambiguous state enum:

- `currently_model_visible` — tri-state (`true` / `false` / `null`): is the
  unit's content in the live request right now?
- `immediate_delivery_mode` — what channel delivers it **right now**?
- `full_content_access_mode` — if more of the unit's FULL content were needed,
  how would the model obtain it?

The artifact is `memory-delivery-plan-shadow.v1`, `mode = "shadow_only"`,
`authority = "proposal_only"`, `live_authority = false`.

## 9. `immediate_delivery_mode` vs `full_content_access_mode`

Closed enums:

```
immediate_delivery_mode ∈ { direct_data, flash_cue, drop, unknown }
full_content_access_mode ∈ { already_present, recall_required, none, unknown }
```

- `immediate_delivery_mode` is the **current** channel (independent of a future
  Recall).
- `full_content_access_mode` answers a **hypothetical** future need for the
  full content, and is not a promise that such a request will be honoured.

The three "known channel" values (`direct_data`, `flash_cue`, `drop`) partition
only the rows with a known channel; `unknown` rows carry **no** known channel
and are counted separately — they are never folded into `drop`.

## 10. Frontend `direct_data` is BASELINE OBSERVATION, not endorsement

A Frontend attachment unit is always:

```
immediate_delivery_mode = direct_data
full_content_access_mode = already_present
control_basis = existing_frontend_direct_baseline
reason = frontend_attachment_already_model_visible
currently_model_visible = true
retrieved = false
base_surfacing_eligible = null
```

`direct_data` is a factual statement that the content is **already** in the
request. It is **not** a new elevation, **not** an endorsement, **not** a Flash
cue and **not** an eligibility verdict — which is why the base-policy fields on
a Frontend row are an explicit `null`, never `false`. The other `full_content_*`
modes are never used for a Frontend row.

## 11. OB `flash_cue` / Recall boundary

An OB candidate unit is **retrieved, not yet model-visible**. Exactly one of
three things is expressed, and it is never `direct_data`:

| base policy state | `immediate_delivery_mode` | `full_content_access_mode` | `control_basis` |
| --- | --- | --- | --- |
| validly bound, candidate eligible | `flash_cue` | `recall_required` | `base_surfacing_policy` |
| validly bound, candidate not eligible | `drop` | `none` | `base_surfacing_policy` |
| not validly bound | `unknown` | `unknown` | `base_surfacing_unavailable` |

`flash_cue` means the existing path *would* deliver a bounded cue; full content
still requires the existing explicit Recall. Delivery Shadow neither performs
the Recall nor shortens that boundary.

## 12. `UNKNOWN != DROP`

`unknown` is not `drop`. When the base Surfacing Policy cannot be validly bound
(missing, malformed, foreign conversation, wrong Unified revision, an unknown
decision enum, or an `eligible` shape that contradicts its decision), the
eligibility verdict is **UNKNOWN** and is never silently converted into
"not eligible". An `unknown` OB row therefore carries
`base_surfacing_eligible = null` and never `false`, and its channel is `unknown`
and never `drop`.

`UNKNOWN != DROP` is the single most important honesty rule of this experiment:
the absence of evidence must never become a negative verdict.

`no_surface` is the **only** trustworthy "ineligible": the real
`memory-surfacing-policy.v1` report always owns an explicit `"eligible": []`
field, so a missing / `null` / non-empty `eligible` set on a `no_surface`
decision is malformed evidence, not a trustworthy "ineligible".

## 13. Exact / normalized overlap != identity

Two units whose `content_digest` (exact) or `normalized_digest` (normalized)
are equal are **still two distinct occurrences**. The adapter is an
**occurrence** adapter, not an identity claim:

- two identical Frontend attachments are two units;
- two identical OB candidates are two units;
- an identical-looking Frontend unit and OB unit are two units from two
  sources.

There is **no** semantic dedup, no embedding overlap, no cross-source identity
and no cross-source ranking. Duplicate occurrences are matched by **FIFO
position** against the eligible list, so duplicate OB candidates cannot silently
share one rank. `evidence_ref` is defined in §14 to keep duplicates distinct.

## 14. Why no cross-source rank exists

There is no cross-source rank because there is no shared quantity to rank on.
The Frontend attachment footprint is genuinely model-visible; the OB candidate
full content is normally **not** model-visible and only becomes a bounded cue /
Recall payload later. Ranking them on one axis would conflate "size of content"
with "cost of the model-visible channel", which are different quantities. The
plan therefore records **per-source** facts and a per-row `base_eligible_rank`
(only for OB rows, and only the existing base-policy order). It never invents a
combined score.

`evidence_ref` is the request-local reference used by a row:

```
evidence_ref = sha256(source + kind + request_occurrence + content_digest)
```

It is **not** a memory id, **not** a content hash alone and **not** a global
identity. Duplicate content occurrences stay distinct occurrences with distinct
refs. Only the digest is emitted (no text, no id).

## 15. Why Lifecycle is intentionally excluded from Delivery v0

The Lifecycle plane (`LIFECYCLE_AUTHORITY_MAP.md`,
`LIFECYCLE_SHADOW_COMPARISON.md`, `memory_surfacing_lifecycle`,
`memory_lifecycle_state`, …) is a **separate, composable signal**. Delivery v0
expresses only the **base** pre-lifecycle surfacing input:

```
policy_stage = "base_surfacing_pre_lifecycle"
```

Reasons for excluding it here:

1. Lifecycle is currently **shadow, default OFF** and threshold-driven; folding
   it into a delivery proposal would make the proposal depend on a stage that
   may not be running at all.
2. The base Surfacing Policy and the Lifecycle Gate are already separate
   authorities (§8 of `ARCHITECTURE_STATUS.md`); collapsing them into one
   delivery verdict would be an unauthorized contract change.
3. The composition of Lifecycle constraints onto a unified Delivery Plan is an
   **open architecture question** (Open Question E below), so v0 must not
   silently answer it.

The artifact therefore records `lifecycle_required = false` and
`flash_required = false`, so a reviewer can see that the plan is the base
pre-lifecycle input only.

## 16. Feature flag

```
OMBRE_GATEWAY_CONTEXT_MEMORY_DELIVERY_SHADOW
```

Default **OFF**. When OFF the plan is never built and never persisted. Enabling
it makes the request run the upstream Conversation → Unified chain (it is a
consumer, not an observability-only toggle), exactly like the other Memory
observers. It is **shadow-only**: it never mutates the provider request.

Enabling the flag changes **nothing** the model sees. Delivery Shadow is
fail-open: any failure (evidence unavailable, invalid artifact, persist failure)
is recorded as *missing delivery data* and nothing else.

For the current request, the plan is also bound onto the request-local
`ContextRequestWorkspace.memory_delivery_plan` field (an in-memory runtime
artifact for THIS request only). The persisted per-request file is the "latest
persisted observation"; it is **never** the runtime authority for a concurrent
request (§35).

## 17. Privacy

The persisted artifact and every log line contain only **counts, booleans,
enums, revisions and sha256 digests**. The following are **never** emitted or
persisted:

- memory text / content / body,
- Flash cue text,
- the raw Frontend attachment envelope,
- OB memory id / memref / useref,
- candidate fingerprint,
- conversation id in logs (the id appears only inside the artifact path and the
  artifact identity fields, following the existing request-scoped convention).

The reporter prints only finite aggregate numbers.

## 18. Concurrency

- The plan is built from THIS request's request-local Source Evidence Snapshot
  and THIS request's base policy. It never re-reads a conversation-global
  "latest" file.
- The persisted artifact path is request-scoped:
  `<state_root>/memory_delivery_plan/<conversation_id>/<cognitive_request_id>.json`.
  Two concurrent requests on the same conversation write different files.
- The request-local `workspace.memory_delivery_plan` is the in-memory authority
  for the request that produced it. A **shared latest artifact is never the
  runtime authority**.
- Persistence uses the shared `atomic_write` primitive, so a partially written
  file is never observed.

## 19. What must happen before Delivery can become live authority

At minimum, all of the following, none of which v0 does:

1. an explicit decision on **cross-source conflicts / priorities**
   (Open Question F);
2. an explicit **semantic-overlap** policy (Open Question G) and a decision on
   whether exact / normalized overlap may ever collapse occurrences;
3. a **stable Frontend item identity** (if it is ever to be more than
   attachment-granular) (Open Question C);
4. a decision on **when Frontend `direct_data` should become a Flash cue**
   (Open Question A);
5. an explicit **Lifecycle → Delivery** composition contract (Open Question E);
6. a live renderer that selects the actual injected body, with the same
   content-integrity checks the existing Real Injection path has;
7. a rollout / kill-switch design and an authority audit proving no source's
   existing guarantees are weakened.

Until then Delivery stays a **proposal with no live authority**.

## 20. What this experiment cannot prove

- It **cannot** prove that a unified model-facing delivery authority is live —
  it is not (see §19).
- It **cannot** prove anything about actual model behaviour, comprehension or
  Recall use; it observes static shapes only.
- It **cannot** prove an item-level Frontend inventory, because no item-level
  schema exists (§4).
- It **cannot** prove cross-source equivalence or ranking; none exists (§13,
  §14).
- It **cannot** prove a Lifecycle-aware delivery decision; Lifecycle is
  deliberately excluded (§15).
- It **cannot** prove the OB Flash / Recall outcome, only the *eligibility*
  that the base policy expresses.

---

## Open Architecture Questions (recorded as OPEN — not silently resolved)

**A. When should Frontend `direct_data` become `flash_cue` instead?**
OPEN. v0 always expresses the Frontend baseline as `direct_data`; it never
proposes converting it.

**B. Should some Frontend bundle be treated as narrow trusted Direct Data?**
OPEN. v0 keeps the whole attachment as one direct-data unit; it makes no
"trusted subset" distinction.

**C. How will Frontend obtain stable item identity, if ever?**
OPEN. v0 uses attachment granularity and occurrence-safe refs only.

**D. Should Frontend get explicit Recall semantics?**
OPEN. v0 does not give the Frontend mouth a Recall boundary.

**E. How does Cognitive Lifecycle constrain the unified Delivery Plan?**
OPEN. v0 records `policy_stage = base_surfacing_pre_lifecycle` and does not
compose Lifecycle into delivery.

**F. How do source priorities / conflicts work?**
OPEN. v0 records per-source facts and invents no cross-source priority.

**G. How should semantic overlap be handled?**
OPEN. v0 has no semantic dedup, no embedding overlap and no LLM judge.

**H. When can the old direct Frontend mouth be removed?**
OPEN. v0 does not remove, gate, rewrite or delete the Frontend mouth.

---

## Aggregate reporter (read-only)

`src/ombrebrain/maintenance/memory_delivery_report.py` scans the request-scoped,
privacy-safe artifacts and prints an **aggregate-only** report. It is a
maintenance / research tool, **not** a runtime authority:

- it creates no database, no rolling global aggregate and no runtime
  accumulator;
- it writes, creates, moves and deletes **nothing** (not even a malformed
  artifact);
- it validates each artifact with the **same** authoritative v0 validator the
  writer uses; an artifact that fails validation is counted as `invalid` and
  contributes **zero** to every statistic (reject-by-whole);
- a missing artifact and a malformed artifact are counted separately
  (`artifact_count` vs `invalid_artifact_count`), so "no data" is never reported
  as "zero units";
- reads are bounded by `--limit` (default `500`; a negative limit is clamped to
  `0`, never "scan everything").

Example (synthetic fixture only; never run over real user memory data for
validation):

```
PYTHONPATH=src python -m ombrebrain.maintenance.memory_delivery_report \
  --root <temporary-test-root> --limit 100 --json
```

```json
{
  "artifact_count": 1,
  "invalid_artifact_count": 0,
  "valid_artifact_count": 1,
  "unit_count": 2,
  "frontend_unit_count": 1,
  "ob_unit_count": 1,
  "direct_data_count": 1,
  "unknown_count": 1,
  "frontend_direct_data_count": 1,
  "ob_unknown_count": 1,
  "policy_unavailable_ob_count": 1,
  "report": "aggregate_only",
  "mode": "shadow_only",
  "version": "memory-delivery-plan-shadow.v1"
}
```

---

## Explicit non-goals of this experiment

Not implemented, and must not be read into this work:

- live Frontend gating; deletion of `relevant_memories.txt`; Frontend Recall;
  a Frontend item parser;
- semantic deduplication; embedding overlap; an LLM arbitration / judge;
- a Memory Delivery live renderer; cross-source scoring or reranking;
- storage unification; ArchiveCandidate; consolidation; supersession;
- Episodic / Semantic migration; a relation graph; Prospective Memory; Wake;
  Emotion / attachment.
