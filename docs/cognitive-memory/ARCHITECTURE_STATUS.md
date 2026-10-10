# Cognitive Memory Architecture Status

**STATUS: WORKING DRAFT / REALITY MAP — NOT AN ADR — NOT A SPECIFICATION**

This document describes what the cognitive-memory chain **does today** in this
repository. It is a snapshot, not a contract. It is expected to become stale as
soon as the next phase lands. Nothing here overrides the code: if this document
and the code disagree, the code is correct and this document is wrong.

- Repository source of truth: the tree at the baseline commit below.
- Baseline: `origin/feature/gateway-v2`
- Baseline HEAD: `07cc03883a4493c5388879458717fe62e18485b5`
- Baseline tree: `79883fa380daebf3fa817016052fbaec3e5a2dd5`
- Audit date context: written for the Frontend Memory Arbitration Shadow task
  (branch `trae/frontend-memory-arbitration-shadow-v1`).

Everything below is derived by reading the real modules listed next to each
stage. Where a claim could not be verified from code it is explicitly marked
`UNVERIFIED` rather than assumed.

---

## 0. The one invariant that matters most

The chain preserves a strict ladder. Each rung is a **different** thing and the
code is careful never to collapse them:

```
retrieved
  != surfaced
  != live_exposed
  != recalled
  != loaded
  != used
  != reinforced
```

No stage may derive reinforcement, usage, or lifecycle change from retrieval,
surfacing, exposure, Recall, model text, similarity, or an LLM judge. When this
document says a stage is "observation-only" it means: it writes a shadow
artifact and/or a log line, and it does not feed back into the live request, the
canonical memory, or the lifecycle.

## 0.1 Privacy convention used across the chain

Two different standards coexist and must not be conflated:

- **Memory-stage log lines** (Memory Flash / Exposure Ledger / Recall Surface /
  Live Exposure / Confidence Gate / Memory Surfacing) are privacy-safe: counts,
  booleans, enums, revisions, reason codes. No conversation id, cognitive
  request id, memory id, memref, cue, query, or memory text.
- **Pre-existing Context log lines** (Unified / Preview / Gate / Mutation /
  Real Injection / Snapshot / Compact / Semantic / Candidate) are unchanged and
  **some of them do still log `conversation_id`**. That is existing behaviour and
  this document does not propose changing it.

Persisted artifacts follow the first standard for *content* (no raw memory text
in the Exposure Ledger) but some (Memory Flash, Recall Surface, Exposure Ledger)
do persist bounded cue text and/or internal ids because a future Recall needs a
target. This asymmetry is intentional in the baseline and is recorded here so it
is not mistaken for an oversight.

---

## 1. Chain order (as actually orchestrated)

Owner of the whole chain: `src/ombrebrain/context/context_pipeline_coordinator.py`.

The write path in the web transport (`src/web/gateway.py`) is deliberately thin:

```
one request body
  -> _rewrite_upstream_body()            (cache stabilisation, preserves Frontend Memory)
  -> run_context_pipeline(forward_body)  (single Context entry point)
  -> selected_body forwarded upstream
```

`gateway.py` never decides stage ordering, freshness, or injection eligibility.
`run_context_pipeline` and `run_context_pipeline_with_binding` are the *same*
single pass; only the latter also returns the trusted `conversation_id` /
`cognitive_request_id` binding for the provider transport.

Inside `_run_context_pipeline_selection`:

```
1. mint cognitive_request_id (ctxreq_<32 hex>, internal only)
2. observe_context_sources(forward_body)             -> conversation_id
3. observe_unified_preview_gate(conversation_id, cpu) -> context_chain_fresh
     - conversation sources -> Unified -> Confidence -> Memory observer chain
       (Surfacing -> Lifecycle -> Flash -> Recall Surface -> Exposure Ledger)
       -> Preview -> Gate -> context_event
4. if fresh: observe_request_mutation(...)           (discards mutated body)
5. select_real_injection(...)                        -> selected_body
6. select_live_memory_flash_exposure(...)            -> final_body
```

---

## 2. Stage map

Field legend:

- **Owner** — the module that owns the rules. Orchestration may live elsewhere.
- **Persisted** — artifact path template under `OMBRE_CONTEXT_STATE_DIR`
  (default `/app/buckets/.context`).
- **Mutates provider request** — can the stage change the bytes sent upstream?
- **Mutates canonical memory** — can it write memory source files / metadata?
- **Lifecycle effect** — can it change derived lifecycle state?

### 2.1 Gateway canonicalization layer

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/gateway/canonicalizer.py`, `models.py`, `operit_adapter.py` |
| Input | raw JSON request bytes |
| Output | `CanonicalRequest` (system / passthrough / history / current), each message a tuple of `CanonicalBlock` tagged with `SegmentKind` |
| Persisted | none (observers log summaries only) |
| Shadow/Live | observation + cache rewriting (rewrite is live on the forward path) |
| Flag | `OMBRE_GATEWAY_CANONICAL_OBSERVE`, `OMBRE_GATEWAY_REWRITE_SHADOW` (observation); `OMBRE_GATEWAY_REWRITE` / `OMBRE_GATEWAY_CACHE_STABLE` (rewrite) |
| Mutates provider request | yes, but only for cache stabilisation (see §3). Classification itself is observation-only |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | every Context stage that needs segment structure |

`SegmentKind` values: `USER_TEXT`, `DYNAMIC_CONTEXT`, `PERCEPTION`, `FINGERTIPS`,
`FRONTEND_MEMORY`, `UNKNOWN`.

Classification is metadata-driven and never keyword-scans ordinary text:

- `DYNAMIC_CONTEXT` — Operit `message_insert_extra_bundle_*` / `Time:` prefixes
  or the legacy `extra_info_*.txt` filenames / prefixes.
- `FRONTEND_MEMORY` — attachment whose `filename` is exactly
  `relevant_memories.txt` (case-insensitive).
- `FINGERTIPS` — filename/id contains `fingertips` or `指尖`.
- everything else → `UNKNOWN`.

### 2.2 Conversation Snapshot

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/conversation_snapshot.py` |
| Input | request body + `conversation_id` + boundary prefix |
| Output | bounded stable-message snapshot (≤12 messages, ≤3000 chars each, ≤16000 total) |
| Persisted | `snapshots/{conversation_id}.json` |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_CONTEXT_SNAPSHOT_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | Compact → Semantic → Semantic State → Conversation Candidate |

Critical property: **only `SegmentKind.USER_TEXT` blocks become snapshot text.**
Every other segment kind (including `FRONTEND_MEMORY` and `DYNAMIC_CONTEXT`) is
counted in `excluded_segments` and dropped from USER_TEXT-derived semantic state.

### 2.3 Deterministic Compact

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/conversation_compact.py` |
| Input | snapshot revision |
| Output | deterministic compacted message list, with a compaction ratio |
| Persisted | `compact/{conversation_id}.json` |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_CONTEXT_COMPACT_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | Semantic, Trusted Facts, Conversation Candidate, retrieval query |

### 2.4 Conservative Semantic Frame + Persistent Semantic State

| Field | Value |
| --- | --- |
| Owner | `conversation_semantic.py` (frame), `conversation_semantic_state.py` (state) |
| Input | compact revision |
| Output | current task / decisions / constraints / open items; semantic state adds lifecycle of those items |
| Persisted | `semantic/{conversation_id}.json`, `semantic_state/{conversation_id}.json` |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_CONTEXT_SEMANTIC_SHADOW`, `OMBRE_GATEWAY_CONTEXT_SEMANTIC_STATE_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | none on memory lifecycle; it has its own item lifecycle (add/refresh/archive/purge) scoped to semantic state |
| Downstream | Conversation Candidate, Confidence Gate, Unified |

### 2.5 Explicit Trusted Facts

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/conversation_trusted_facts.py` |
| Input | compact revision |
| Output | trusted fact set derived from **explicit** commands/signals |
| Persisted | `trusted_facts/{conversation_id}.json` |
| Shadow/Live | shadow (inference is flag-gated and conservative) |
| Flag | `OMBRE_GATEWAY_CONTEXT_TRUSTED_FACTS_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | Conversation Candidate, Unified |

### 2.6 Conversation Context Candidate

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/conversation_context_candidate.py` |
| Input | semantic state + trusted facts + recent context |
| Output | candidate context bundle with dedup / budget / control rejection |
| Persisted | `context_candidate/{conversation_id}.json` |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_CONTEXT_CANDIDATE_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | Unified, Confidence Gate |

### 2.7 Unified Context Candidate

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/unified_context_candidate.py` |
| Input | Conversation Candidate + a **real** retrieval query (current user text, used only for retrieval, never copied into Unified) |
| Output | Unified artifact: sections (memories, plans, state, …), telemetry, revision; plus a request-local `memory_shadow_snapshot` returned only in-process |
| Persisted | `unified_context_candidate/{conversation_id}.json` (the artifact); the snapshot is **never persisted** |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_CONTEXT_UNIFIED_CANDIDATE_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | no (read-only retrieval) |
| Lifecycle effect | none |
| Downstream | Confidence Gate, Memory Surfacing Policy, Preview, Gate, Mutation Shadow, Real Injection |

The Unified file is **conversation-global** (overwritten per request). The
`memory_shadow_snapshot` is **request-local** and exists precisely so downstream
observers never re-read the shared file. This request-local vs
conversation-global split is load-bearing for concurrency.

### 2.8 Confidence Gate

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/context_confidence_gate.py` |
| Input | Conversation Candidate + Unified revisions |
| Output | `allow_shadow` / `deny_shadow` decision, freshness + count telemetry |
| Persisted | `confidence_gate/{conversation_id}.json` |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | Memory Surfacing Policy (as evidence), Memory Flash |

Its decision is **never a prerequisite**; a deny does not stop later stages and
does not change the forwarded body.

### 2.9 Memory Surfacing Policy

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/memory_surfacing_policy.py` |
| Input | Unified + Confidence report + conservative.v1 candidate evidence |
| Output | deterministic eligible set in canonical order (rule-based) |
| Persisted | none (in-memory report consumed by Flash) |
| Shadow/Live | shadow |
| Flag | driven by the Flash flag; no independent flag |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | Lifecycle Surfacing gate → Memory Flash |

### 2.10 Lifecycle Surfacing gate

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/memory_surfacing_lifecycle.py` |
| Input | Surfacing Policy report |
| Output | possibly-narrower eligible set (never widens; preserves primary candidate and canonical order) |
| Persisted | none |
| Shadow/Live | shadow, default OFF |
| Flag | `OMBRE_GATEWAY_CONTEXT_MEMORY_SURFACING_LIFECYCLE` (+ thresholds `..._MIN_ACCESSIBILITY`, `..._MIN_SALIENCE`) |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | reads lifecycle, does not write it |
| Downstream | Memory Flash |

Fail-open: any failure returns the base policy untouched.

### 2.11 Memory Flash

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/memory_flash.py` |
| Input | Surfacing Policy report |
| Output | bounded **cues** (not full memories), `surfaced_count`, revisions |
| Persisted | `memory_flash/{conversation_id}/{cognitive_request_id}.json` |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | none (no reinforcement) |
| Downstream | Recall Surface, Exposure Ledger, Live Exposure |

### 2.12 Recall Surface

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/memory_recall_surface.py` |
| Input | Memory Flash report |
| Output | `memref_<32 hex>` → bounded cue mapping, bound to one conversation + request |
| Persisted | `recall_surface/{conversation_id}/{cognitive_request_id}.json` |
| Shadow/Live | shadow, default OFF |
| Flag | `OMBRE_GATEWAY_CONTEXT_MEMORY_RECALL_SURFACE_SHADOW` |
| Mutates provider request | no (by itself) |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | Live Memory Flash Exposure |

### 2.13 Exposure Ledger

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/exposure_ledger.py` |
| Input | Unified + surfaced ids |
| Output | per-request `retrieved` / `surfaced_as_flash` events |
| Persisted | `exposure_ledger/{conversation_id}/{cognitive_request_id}.json` |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_CONTEXT_EXPOSURE_LEDGER_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | none yet (evidence only) |

Records **two rungs only**. Later rungs (`noticed` / `recall_requested` /
`full_memory_loaded` / `used_in_response`) are deliberately absent, not `false`.

### 2.14 Injection Preview → Injection Gate

| Field | Value |
| --- | --- |
| Owner | `context_injection_preview.py`, `context_injection_gate.py` |
| Input | Unified + Candidate revisions |
| Output | a rendered (not sent) context block and an allow/deny decision |
| Persisted | `injection_preview/{conversation_id}.json`, `injection_gate/{conversation_id}.json` |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_CONTEXT_INJECTION_PREVIEW_SHADOW`, `OMBRE_GATEWAY_CONTEXT_INJECTION_GATE_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | Mutation Shadow, Real Injection |

### 2.15 Mutation Shadow

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/context_request_mutation_shadow.py` |
| Input | `forward_body` + Preview/Gate refreshed by **this** request |
| Output | hypothetical injected body + report; the body is **discarded** |
| Persisted | none |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_CONTEXT_REQUEST_MUTATION_SHADOW` |
| Mutates provider request | no (discarded) |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | none (evidence only); Real Injection re-runs the same builder |

### 2.16 Real Context Injection

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/context_real_injection.py` |
| Input | `forward_body` + fresh Preview/Gate |
| Output | `selected_body` (may differ from input) with an `applied` report |
| Persisted | none |
| Shadow/Live | **LIVE**, default OFF |
| Flag | `OMBRE_GATEWAY_CONTEXT_REAL_INJECTION` |
| Mutates provider request | **yes** — the only earlier stage that can |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | Live Memory Flash Exposure |

Guarded by: master flag, `context_chain_fresh is True`, deny → unchanged body,
and a `applied=true` report validated against original/selected sha256.

### 2.17 Live Memory Flash Exposure

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/memory_flash_live_exposure.py` |
| Input | the already-selected body + this request's Recall Surface |
| Output | body with one small DATA-ONLY Memory Flash block, or the exact input body |
| Persisted | `memory_live_exposure/{conversation_id}/{cognitive_request_id}.json` |
| Shadow/Live | **LIVE**, default OFF |
| Flag | `OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_LIVE_EXPOSURE` (+ requires Flash + Recall Surface flags) |
| Mutates provider request | yes (bounded cue + opaque memref only) |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | provider transport (if enabled) |

Strictly post-Real-Injection and fail-open **relative to its own input**, so a
failure never reverts an already-applied Context injection.

### 2.18 Recall Authorization / Provider-bound Recall / Transport

| Field | Value |
| --- | --- |
| Owner | `live_recall_authorization.py`, `provider_live_recall_transport.py`, `live_recall_transport_registry.py`, `live_recall_transport_capability.py`, `recall_request.py` |
| Input | trusted binding (`conversation_id`, `cognitive_request_id`) + surface |
| Output | capability / MCP server attachment, receipts |
| Persisted | `provider_live_recall_transport/…`, `live_recall_transport_capability/…`, `related_recall/…`, `recall_request/…` |
| Shadow/Live | capability + transport; default OFF |
| Flags | `OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_TRANSPORT`, `..._LIVE_RECALL_PROVIDER`, `..._LIVE_RECALL_MCP_URL`, `..._LIVE_RECALL_BRIDGE`, `..._RECALL_ENABLED`, `..._RELATED_RECALL_SHADOW` |
| Mutates provider request | yes (adds MCP connector / tools) |
| Mutates canonical memory | no |
| Lifecycle effect | none directly |
| Downstream | model tool call (`recall`) |

### 2.19 Usage Surface → UseMemory attribution → Usage→Lifecycle

| Field | Value |
| --- | --- |
| Owner | `live_recall_usage_surface.py`, `live_recall_usage_attribution.py`, `live_recall_usage_ref_receipt.py`, `live_recall_usage_lifecycle_integration.py` |
| Input | model-visible usage refs; explicit `UseMemory(useref)` |
| Output | validated `used` events; reinforcement evidence |
| Persisted | `live_recall_usage_surface/…`, `memory_usage/{cid}/{reqid}/{recall_id}.json`, `live_recall_usage_ref_receipt/…` |
| Shadow/Live | shadow |
| Flags | `OMBRE_GATEWAY_CONTEXT_USAGE_ATTRIBUTION`, `OMBRE_GATEWAY_CONTEXT_USAGE_LIFECYCLE_INTEGRATION` (+ `OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW`), budgets `..._USAGE_REFS_MAX`, `..._USAGE_REFS_TOKEN_BUDGET` |
| Mutates provider request | no |
| Mutates canonical memory | no |
| Lifecycle effect | **yes, transitively** — only explicit `UseMemory` validated usage can produce reinforcement evidence |
| Downstream | Memory Lifecycle State / Events, Memory Reinforcement |

This is the only path permitted to turn `used` into reinforcement. It requires an
explicit opaque `useref` the model must have chosen; it can never be triggered by
retrieval, surfacing, exposure, or Recall alone.

### 2.20 Memory Lifecycle State / Events / Reinforcement

| Field | Value |
| --- | --- |
| Owner | `memory_lifecycle_state.py`, `memory_lifecycle_event.py`, `memory_reinforcement.py` |
| Input | validated usage / reinforcement evidence |
| Output | derived lifecycle state and event log |
| Persisted | `memory_lifecycle_state/…`, `memory_lifecycle_events/…` |
| Shadow/Live | shadow |
| Flag | `OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW` |
| Mutates provider request | no |
| Mutates canonical memory | derivations only; it does not rewrite memory source text |
| Lifecycle effect | **this is** the lifecycle authority for the new chain |
| Downstream | Lifecycle Surfacing gate (read), usage integration |

### 2.21 Legacy DecayEngine

| Field | Value |
| --- | --- |
| Owner | `src/decay_engine.py` |
| Input | memory buckets / activation metadata |
| Output | decay / forgetting over long horizons |
| Persisted | legacy bucket metadata |
| Shadow/Live | **live legacy subsystem**, independent of the new chain |
| Flag | none in the new chain |
| Mutates provider request | no |
| Mutates canonical memory | yes (legacy activation/importance fields) |
| Lifecycle effect | yes, in the **legacy** model |
| Downstream | legacy retrieval / ranking |

---

## 3. What the cache rewriters do to attachments (relevant to Frontend Memory)

`src/ombrebrain/gateway/rewriter.py` (`rewrite_history_for_cache`) and
`cache_stable_body.py` only remove **strongly identified historical
`dynamic_context` / `fingertips`** attachments from historical user messages.

They explicitly preserve:

- `frontend_memory` attachments (`relevant_memories.txt`),
- unknown attachments,
- the current user message,
- assistant messages.

Consequence: **Frontend Memory full contents travel to the model in the live
request** and are not removed by cache stabilisation. This is the asymmetry the
Frontend Memory Arbitration Shadow task measures (see §4).

---

## 4. Frontend Memory vs OB Memory — the access asymmetry (measured, not fixed)

Two memory paths reach the model with different access semantics:

- **OB / cognitive memory** —
  `retrieved → eligibility → Memory Flash → live exposure → model decides Recall
  → full memory loaded (only on explicit Recall) → optional explicit UseMemory`.
  The model normally sees only a bounded cue; full content requires an explicit
  Recall.
- **Frontend Memory** (`relevant_memories.txt`) — full content is already in the
  model-visible request; it is preserved by cache rewriting; it is excluded from
  Conversation Snapshot USER_TEXT-derived semantic state.

This asymmetry is **intentional in the baseline and not removed** by this task.
Frontend Memory Arbitration Shadow v1 only *observes* it.

### 4.1 Observation units are not equal (attachment ≠ memory item)

The repository has **no proven item-level schema** for the inner body of a
`relevant_memories.txt` attachment. The only thing the canonicalizer guarantees
is that an explicit `filename="relevant_memories.txt"` envelope is a Frontend
Memory *attachment*. Therefore v1 measures:

- a Frontend Memory **attachment / bundle** (one canonical envelope), versus
- an OB **retrieved memory candidate** (one Unified-admitted memory).

`OBSERVATION SHAPE != FUTURE CANONICAL MEMORY CONTRACT`. Do not read the
arbitration artifact as if it proved an item-level memory inventory.

### 4.2 Four size concepts must never be collapsed

The chain has four different "size" notions. Only the Frontend attachment
footprint is genuinely model-visible in the request:

1. **Frontend attachment body size** — the extracted inner text of the
   attachment (`body`). Useful for comparison, not the wire cost.
2. **Frontend model-visible attachment footprint** — the exact raw canonical
   envelope present in the request, including metadata
   (`<attachment …>BODY</attachment>`). This is the real provider-context cost.
3. **OB retrieved candidate full-content size** — the whole Unified candidate
   memory content. Normally **NOT** model-visible.
4. **OB model-visible cue / Recall size** — the bounded Flash cue, and full
   content only after an explicit Recall. This is a later, separate stage.

Because (1)/(3) are body/content sizes while (2) is genuinely model-visible and
(4) is a different stage, there is deliberately **no cross-source token share**.
A single "memory token share" would be semantically wrong.

---

## 5. Feature-flag dependency map (observation and mutation)

Flags are **not** independent in general; enabling a downstream observer
implicitly drives the upstream chain it consumes.

- `OMBRE_GATEWAY_CONTEXT_CONFIDENCE_GATE_SHADOW`,
  `..._MEMORY_FLASH_SHADOW`, `..._EXPOSURE_LEDGER_SHADOW`,
  `..._MEMORY_RECALL_SURFACE_SHADOW`, and
  `..._MEMORY_SOURCE_ARBITRATION_SHADOW` all cause
  `observe_context_sources` / `observe_unified_preview_gate` to run the
  upstream Conversation→Unified chain for this request (they are consumers, not
  observability-only toggles).
- `..._MEMORY_RECALL_SURFACE_SHADOW` is **not a standalone producer**: a Recall
  Surface derives from a Memory Flash artifact. With no Flash artifact for the
  request it has nothing to surface. Enabling it alone drives the upstream chain
  (it is part of `memory_observer_enabled`) but does **not** by itself yield a
  complete memory observation.
- `OMBRE_GATEWAY_CONTEXT_REAL_INJECTION` also drives the full upstream chain,
  even when every observability shadow flag is off.
- `..._MEMORY_FLASH_LIVE_EXPOSURE` requires the Flash flag and the Recall Surface
  flag; it is strictly post-Real-Injection.
- `..._MEMORY_SURFACING_LIFECYCLE` is only consulted while the Flash stage is
  already running; it never starts the pipeline.
- `..._USAGE_LIFECYCLE_INTEGRATION` requires `..._USAGE_ATTRIBUTION` and
  `OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW`.

Shadow flags that never mutate the provider request:
`..._UNIFIED_CANDIDATE_SHADOW`, `..._INJECTION_PREVIEW_SHADOW`,
`..._INJECTION_GATE_SHADOW`, `..._REQUEST_MUTATION_SHADOW`,
`..._CONFIDENCE_GATE_SHADOW`, `..._MEMORY_FLASH_SHADOW`,
`..._EXPOSURE_LEDGER_SHADOW`, `..._MEMORY_RECALL_SURFACE_SHADOW`,
`..._MEMORY_SOURCE_ARBITRATION_SHADOW`,
`OMBRE_GATEWAY_MEMORY_LIFECYCLE_SHADOW`.

Flags that can mutate the provider request:
`OMBRE_GATEWAY_REWRITE` / `OMBRE_GATEWAY_CACHE_STABLE` (cache form),
`OMBRE_GATEWAY_CONTEXT_REAL_INJECTION`,
`OMBRE_GATEWAY_CONTEXT_MEMORY_FLASH_LIVE_EXPOSURE`,
`OMBRE_GATEWAY_CONTEXT_LIVE_RECALL_TRANSPORT`.

---

## 6. Audit Findings — Not Changed In This Task

These are observations recorded for later phases. **None of them is fixed by the
Frontend Memory Arbitration Shadow task.** They are ordered by severity, and each
points at real code.

### P0 — correctness / authority problems

**None demonstrated at this baseline.** The access asymmetry in §4 is an
intentional design/authority gap under measurement, not a proven invariant
violation, so it is deliberately not classified as P0. P0 is reserved for
demonstrated correctness / data-integrity / authority violations.

### Open Architecture Questions

- **OA-1 — Two memory access models coexist without a shared arbitration
  authority.** OB memory is cue/Recall-gated; Frontend Memory is full-content
  model-visible (`canonicalizer.py` classifies it; `rewriter.py` preserves it).
  There is no single authority that decides what the model is allowed to see
  across both sources. This is the question the shadow phase exists to measure;
  it is **not** fixed here, and no current invariant is violated by it.

### P1 — architecture debt likely to hurt the next stages

- **P1-1 — Broad implicit flag coupling makes rollout ordering non-obvious.**
  Enabling a downstream shadow consumer silently drives the whole upstream
  Conversation→Unified chain (`context_observation_pipeline.py` and
  `observe_unified_preview_gate`). It is correct, but a future "source-neutral
  candidate pool" must decide explicitly whether it inherits this coupling.
- **P1-2 — Request-local vs conversation-global state is a repeated
  footgun.** Unified/Preview/Gate/Confidence live in conversation-scoped files
  (`unified_context_candidate/{cid}.json`, `injection_preview/{cid}.json`, …)
  while Flash/Surface/Ledger are request-scoped. Every consumer must remember to
  use `memory_shadow_snapshot` instead of re-reading the shared file
  (`build_memory_shadow_snapshot`). This is currently handled correctly, but it
  relies on discipline rather than a type boundary.
- **P1-3 — Several persisted artifacts contain redundant identity fields.**
  `exposure_ledger.py` and (following it) other request-scoped artifacts persist
  `conversation_id` + `cognitive_request_id` inside the artifact even though both
  are already in the path. Harmless today; worth a single decision before more
  request-scoped artifacts are added.
- **P1-4 — The conservative.v1 candidate policy is computed twice by design.**
  `build_candidate_evidence` (`retrieval_decision_shadow.py`) simulates a future
  filter, and `memory_surfacing_policy.py` independently decides eligibility. Two
  "keep/drop" notions exist side by side; they must not be conflated in later
  phases (simulated filter ≠ surfacing).

### P2 — cleanup / naming / maintainability

- **P2-1 — `CANONICAL OBSERVE` requires re-parsing the body.** The gateway
  canonical summary (`observer.py`) re-canonicalizes; Snapshot/Compact also
  canonicalize. This is acceptable but is duplicated work; a future phase may
  thread one canonical request through the coordinator instead of re-parsing.
- **P2-2 — Legacy `decay_engine.py` runs beside the new lifecycle chain.**
  Two lifecycle-ish authorities exist (legacy decay vs
  `memory_lifecycle_state.py`). They are not reconciled. Out of scope here.
- **P2-3 — The two suspected stale evidence modules are actually live.**
  `retrieval_shadow_metrics.py` is imported and called by
  `unified_context_candidate.py` (`record_retrieval_shadow_metrics`), and
  `live_recall_usage_ref_receipt.py` is imported and used by
  `live_recall_usage_surface.py` (`build_usage_ref_receipt`,
  `persist_usage_ref_receipt`, `read_usage_ref_receipt`,
  `receipt_matches_binding`, `useref_sha256`). Neither is unconsumed; no
  unconsumed memory-stage module was found among those checked.
- **P2-4 — Per-module `_root()` / `_atomic_write` / `_path` duplication.**
  `recall_types.py` centralises shared primitives, but several modules
  (`exposure_ledger.py`, `memory_flash.py`, `context_confidence_gate.py`) keep
  their own copies. New modules should reuse the shared primitives rather than
  add another copy.

### Explicitly NOT done in this task

- No change to `decay_engine.py` authority.
- No merge/archive/supersede of memories.
- No new retrieval subsystem, canonical request model, memory DB, exposure
  ledger, lifecycle authority, background worker, or cron.
- No Event Segmentation, Episodic Memory, Schema consolidation, Emotion /
  Attachment / Desire.
- No change to Frontend Memory liveness (it stays fully model-visible).
