# Cognitive Memory Architecture Status

**STATUS: WORKING DRAFT / REALITY MAP — NOT AN ADR — NOT A SPECIFICATION**

This document describes what the cognitive-memory chain **does today** in this
repository. It is a snapshot, not a contract. It is expected to become stale as
soon as the next phase lands. Nothing here overrides the code: if this document
and the code disagree, the code is correct and this document is wrong.

- Repository source of truth: the tree at the baseline commit below.
- Baseline: `origin/feature/gateway-v2`
- Baseline HEAD: `ab586b34d0265a713d3e3ed3ac60741246093ed2`
  (merge of PR#2, `trae/request-local-cognitive-workspace-v0`)
- Audit date context: originally written for the Frontend Memory Arbitration
  Shadow task (branch `trae/frontend-memory-arbitration-shadow-v1`), then
  updated for the Request-local Cognitive Workspace Foundation v0 task
  (branch `trae/request-local-cognitive-workspace-v0`), then updated for the
  Memory Lifecycle Authority Map + Dual-Authority Guardrails v0 task
  (branch `trae/lifecycle-authority-audit-v0`).

Companion document: **`docs/cognitive-memory/LIFECYCLE_AUTHORITY_MAP.md`** is the
exact call-site / authority matrix for the legacy and cognitive lifecycle planes
(referenced from §2.19, §2.20, §2.21 and §8). It is itself a WORKING DRAFT /
REALITY MAP, not an ADR or specification.

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

## 0.2 Request-local vs conversation-global authority

A second invariant now runs through the Context decision chain, and it must not
be collapsed either:

```
conversation-scoped persisted artifact  =  latest persisted observation
request-local workspace artifact        =  runtime authority for THIS request
```

`conversation-scoped artifact != request-local authority`. A conversation-scoped
file (`unified_context_candidate/{cid}.json`, `injection_preview/{cid}.json`,
`injection_gate/{cid}.json`, …) is a **latest persisted observation** that a
concurrent request on the same conversation may overwrite at any time. It is
**no longer the runtime authority** for a request that produced its own artifacts.
It still serves observability, debugging and compatibility, and a **successful
Preview/Gate persistence is additionally a v0 freshness / liveness prerequisite**
(§2.22 "Persistence semantics") — so it is not merely optional observability.

The pipeline now binds each request's exact stage outputs into an in-memory
**request-local Cognitive Workspace** (see §2.22). Downstream stages for THIS
request (Preview, Gate, Mutation Shadow, Real Injection) prefer the workspace
artifact over re-reading the shared file. This is a generalization of the
pre-existing `memory_shadow_snapshot` pattern (which already protected the Memory
observers from re-reading the shared Unified file), not a second, contradictory
request-identity mechanism.

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
3. create ContextRequestWorkspace(conversation_id, cognitive_request_id)
     (in-memory, one per request; conversation_id is only known now)
4. observe_unified_preview_gate(conversation_id, cpu, workspace=ws) -> context_chain_fresh
     - conversation sources -> Unified -> Confidence -> Memory observer chain
       (Surfacing -> Lifecycle -> Flash -> Recall Surface -> Exposure Ledger)
       -> Preview -> Gate -> context_event
     - binds THIS request's Candidate / Unified / memory snapshot / Confidence
       report / Preview / Gate / freshness into ws
5. if fresh: observe_request_mutation(..., workspace=ws)   (discards mutated body)
6. select_real_injection(..., workspace=ws)                -> selected_body
7. select_live_memory_flash_exposure(...)                  -> final_body
```

The workspace is created in-memory for the single pass and is never persisted,
logged, cached, or added to the request. Conversation-scoped files are still
written by each stage for observability; the workspace is the runtime authority
for the request currently being forwarded.

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
observers never re-read the shared file. As of the request-local workspace task,
the producer also attaches an in-memory-only `request_local_chain` field
(`conversation_candidate` + `unified`) to the report it returns; the coordinator
extracts it into the workspace (§2.22). This request-local vs
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

Both stages now have a request-local entry point used by the coordinator when a
workspace is present: `update_context_injection_preview_from_unified(...)` renders
the Preview from **this request's** explicit Unified object, and
`update_context_injection_gate_from_request_local(...)` evaluates the Gate from
**this request's** explicit Candidate / Unified / Preview. Persistence still
writes the conversation-scoped files. Those files are not the runtime CONTENT /
IDENTITY authority for THIS request, but persistence is **not** merely optional
observability: a successful store (`stored=true`) remains a v0 freshness
prerequisite, so a write failure blocks freshness and prevents injection
(§2.22 "Persistence semantics"). The request-local artifacts are the runtime
authority for this request.

### 2.15 Mutation Shadow

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/context_request_mutation_shadow.py` |
| Input | `forward_body` + Preview/Gate refreshed by **this** request (from the request-local workspace when available, else conversation-scoped files) |
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
| Input | `forward_body` + fresh Preview/Gate (from the request-local workspace when available, else conversation-scoped files) |
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
| Input (read-only legacy metadata) | `importance` / `activation_count` / `last_active` / `created` / `arousal` / `resolved` / `digested` |
| Output | decay / forgetting over long horizons |
| Persisted | legacy lifecycle transitions only (auto-`resolved`, archive move) |
| Shadow/Live | **live legacy subsystem**, independent of the new chain |
| Flag | none in the new chain |
| Mutates provider request | no |
| Mutates canonical memory | yes — automatic `resolved` write + archive transition; does **not** write `activation_count` / `last_active` / `importance` |
| Lifecycle effect | yes, in the **legacy** model |
| Downstream | legacy spontaneous surfacing / Dream truncation / archive decision / read-only score consumers; **not** canonical `BucketManager.search` retrieval scoring |

Authority (exact; see `LIFECYCLE_AUTHORITY_MAP.md` §3.8, §9 and §10): where it is
**running**, `DecayEngine` has real **canonical + archive authority**.
`run_decay_cycle()` auto-resolves buckets (`importance <= 4` and `> 30` days since
last activation) and archives buckets whose `calculate_score()` falls below the
threshold; it never deletes. `calculate_score()` also drives Dream candidate
truncation (`tools/dream/candidates.py`) and spontaneous-`breath` ranking, so the
legacy score has Dream-selection influence, not only an archive role. It is
started eagerly by the `streamable-http` lifespan and, under stdio, lazily on the
first use of `breath` / `dream` / `hold` / `grow` / `anchor` / `plan` / `i` via
`ensure_started()`. That start asymmetry is a real runtime authority asymmetry
and is documented, not "fixed", in the lifecycle audit.

### 2.21.1 (Deprecated label) "legacy DecayEngine vs new lifecycle"

Both subsystems exist and both are authorities, but they are **different scopes**,
not two candidates for one job. See §8 for the current reality and why the old
"two lifecycle-ish authorities coexist" framing under-stated it.

### 2.22 Request-local Cognitive Workspace (v0)

| Field | Value |
| --- | --- |
| Owner | `src/ombrebrain/context/context_request_workspace.py` (`ContextRequestWorkspace`) |
| Input | the exact artifacts THIS Context pass produced |
| Output | in-memory runtime authority: `conversation_id`, `cognitive_request_id`, `conversation_candidate`, `unified`, `memory_shadow_snapshot`, `confidence_report`, `preview`, `gate`, `context_chain_fresh` |
| Persisted | **none** — in-memory only, one per gateway request |
| Shadow/Live | runtime authority (not an observer, not a mutator itself) |
| Flag | none (always created; every consumer keeps its existing flag + fallback) |
| Mutates provider request | no (it holds artifacts; the stages that read it do) |
| Mutates canonical memory | no |
| Lifecycle effect | none |
| Downstream | Preview, Gate, Mutation Shadow, Real Injection |

Properties:

- `has_request_local_chain` — true only when THIS request produced its own
  Candidate + Unified. When false, every downstream stage keeps its existing
  conversation-scoped behaviour.
- `has_any_preview_gate_evidence` — true when EITHER Preview OR Gate evidence
  exists, including a partial binding (only one side) or a malformed binding
  (non-`None` non-dict). This is the tri-state discriminator downstream stages
  use: **no** evidence at all permits the legacy conversation-scoped fallback;
  **any** evidence means the caller must use this request's complete pair or fail
  closed.
- `has_preview_gate_evidence` — true when BOTH sides are dicts (the "complete
  pair" arm); a deny is still a complete pair.
- `request_local_preview_gate()` — returns the pair only when both artifacts are
  present and bound to this workspace's `conversation_id`; otherwise
  `(None, None)`.

The workspace is **not** canonical memory, **not** a database, **not** a persisted
artifact, **not** model-visible, **not** a cache key and **not** an event bus. It
is not a universal memory schema and carries no future-lifecycle fields. It
generalizes the pre-existing `memory_shadow_snapshot` request-local pattern to the
whole decision chain.

Fail semantics (unchanged invariants, preserved):

- Real Injection remains default OFF and fail-open (uncertainty → exact input
  body).
- Freshness alone is still never trusted: `context_chain_fresh=True` is required
  but not sufficient; revision / rendered-hash / token-recompute / DATA-ONLY
  envelope / original–selected SHA checks all still run inside the pure
  boundaries.
- Request-local identity is an **additional** guarantee, never a replacement for
  content-integrity checks.
- A **partial or malformed** request-local binding (exactly one artifact
  supplied / a non-dict value) is NOT equivalent to "no request-local evidence was
  supplied". It fails **closed** for the Mutation Shadow (mutation is skipped) and
  fails **open** for request forwarding (the exact input body is kept), and it
  **never** falls back to conversation-scoped Preview/Gate. The selector reports
  the stable reason `invalid_request_local_binding`.
- A complete-but-foreign binding fails closed the same way
  (`request_local_conversation_mismatch`).

### Persistence semantics (v0, intentionally unchanged)

Conversation-scoped persisted artifacts are **not** the runtime CONTENT /
IDENTITY authority for THIS request. But they are also **not** merely optional
observability: for v0, a **successful Preview/Gate persistence** (`stored=true`)
remains a **freshness / liveness prerequisite**. `observe_unified_preview_gate`
returns `context_chain_fresh=False` when Preview or Gate did not persist, and Real
Injection is then not attempted at all — so an actual write failure can never
inject stale conversation-scoped state. This contract is unchanged by the
request-local workspace task and may only be redefined by a future explicit task.

Do not confuse the two properties:

- removing conversation-scoped files **after** a successful decision proves the
  already-captured request-local decision no longer needs a later re-read;
- it does **not** prove that an initial write failure is ignored.

### 2.23 Request-local authority migration status

Stages that now take the request-local workspace as their runtime authority for
THIS request (when a workspace is present and the request produced its own chain):

| Stage | Request-local source | Fallback when no workspace evidence |
| --- | --- | --- |
| Memory observation (`memory_shadow_snapshot`) | `workspace.memory_shadow_snapshot` | existing behaviour (already request-local) |
| Preview | `workspace.unified` → `update_context_injection_preview_from_unified` | `update_context_injection_preview` (conversation-scoped) |
| Gate | `workspace.conversation_candidate` + `unified` + `preview` | `update_context_injection_gate` (conversation-scoped) |
| Mutation Shadow | `workspace.request_local_preview_gate()` | `build_context_request_mutation_shadow_from_runtime` (conversation-scoped) |
| Real Injection | request-local Preview/Gate forwarded to the selector | `select_context_injected_body` re-reads conversation-scoped files |

Stages that **still depend on conversation-global reads** (recorded honestly, not
fixed here):

- **Confidence Gate** — `observe_context_confidence(...)` re-reads the persisted
  Conversation Candidate / Unified files, guarded by
  `expected_unified_revision` so a concurrent overwrite is detected. It is an
  observer only, so its result never gates anything, but it is not yet fed from
  the workspace.
- **Memory Surfacing / Lifecycle / Memory Flash / Recall Surface / Exposure
  Ledger** — these consume the request-local `memory_shadow_snapshot` (already
  request-local), but their own persisted request-scoped artifacts are written to
  `.../{conversation_id}/{cognitive_request_id}.json`. They are observers and do
  not feed the forwarded body.
- **Live Memory Flash Exposure** — strictly post-Real-Injection and
  fail-open relative to its own input; it reads the already-selected body, not a
  conversation-global "latest" decision file.
- **Recall Authorization / Provider-bound Recall / Transport** — downstream of
  the forwarded body and driven by the trusted request binding.

The Frontend-Memory-observation path (`forward_body`) is intentionally passed as
this request's own bytes; it is not a conversation-global read.

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
- **OA-2 — The system has parallel lifecycle authorities with different scopes,
  not one lifecycle owner (re-prioritized by the lifecycle authority audit).**
  This was previously recorded low (as "P2-2, two lifecycle-ish modules coexist")
  but real authority is not mere coexistence. See §8 and
  `LIFECYCLE_AUTHORITY_MAP.md`. Concretely: legacy `DecayEngine` holds **real
  canonical + archive authority where it runs** (auto-resolve, auto-archive; its
  score also drives Dream truncation); the new cognitive lifecycle holds
  **supplementary Memory Flash surfacing authority when its gates are ON** and
  owns cognitive `used`/reinforcement/`strength`-`accessibility`-`salience`. The
  two planes **do not currently mutate each other's state**. Migration to a single
  owner is **unresolved** and is not attempted by the audit.

### P1 — architecture debt likely to hurt the next stages

- **P1-1 — Broad implicit flag coupling makes rollout ordering non-obvious.**
  Enabling a downstream shadow consumer silently drives the whole upstream
  Conversation→Unified chain (`context_observation_pipeline.py` and
  `observe_unified_preview_gate`). It is correct, but a future "source-neutral
  candidate pool" must decide explicitly whether it inherits this coupling.
- **P1-2 — Request-local vs conversation-global state was a repeated footgun
  (now materially reduced, not eliminated).** Unified/Preview/Gate/Confidence
  live in conversation-scoped files (`unified_context_candidate/{cid}.json`,
  `injection_preview/{cid}.json`, …) while Flash/Surface/Ledger are
  request-scoped. The request-local Cognitive Workspace task migrated the
  **decision** stages (Preview, Gate, Mutation Shadow, Real Injection) to prefer
  the workspace artifacts for THIS request, and both entry points fail closed on a
  malformed / foreign binding instead of silently re-reading the shared file. What
  remains: the **Confidence Gate** observer still re-reads conversation-scoped
  files (revision-guarded, observer-only), and the workspace is still a plain
  dataclass whose discipline is enforced by the coordinator rather than a fully
  type-enforced boundary. So this is downgraded from "relies on discipline for the
  whole decision chain" to "relies on discipline for one observer plus the
  coordinator wiring".
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
- **P2-2 — (Re-prioritized up to OA-2.)** Legacy `decay_engine.py` and the new
  cognitive lifecycle are **not** two interchangeable lifecycle modules; they are
  parallel authorities with different scopes (§8). Retained here only as a
  cross-reference; the live question is now OA-2.
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

### Explicitly NOT done in the Frontend Memory Arbitration Shadow task

- No change to `decay_engine.py` authority.
- No merge/archive/supersede of memories.
- No new retrieval subsystem, canonical request model, memory DB, exposure
  ledger, lifecycle authority, background worker, or cron.
- No Event Segmentation, Episodic Memory, Schema consolidation, Emotion /
  Attachment / Desire.
- No change to Frontend Memory liveness (it stays fully model-visible).

### Explicitly NOT done in the Request-local Cognitive Workspace Foundation v0 task

- No future memory system (Episodic, Semantic, Relation, Prospective, Wake,
  Emotion, Attachment, Desire).
- No Unified Memory Delivery Contract; no removal or rewriting of Frontend
  Memory; no source-neutral Frontend / OB candidate merging; no item-level
  parsing of `relevant_memories.txt`.
- No `AbstractMemoryProvider`, plugin registry, event bus, or generic cognitive
  framework.
- No deletion or migration of conversation-scoped persisted artifacts; no storage
  layout change. Authority separation only (see §0.2).
- No change to Frontend Memory liveness; no change to any default (Real
  Injection and Live Memory Flash Exposure remain default OFF).

---

## 7. North-star working direction (NON-NORMATIVE)

> This section is an **evolving working direction**, not a frozen
> specification and not a contract. Nothing in it is implemented except the
> request-local Cognitive Workspace (§2.22). It exists so later phases share one
> vocabulary. If this section and the code disagree, the code wins.

Sketch:

```
Federated Memory
  -> source-specific logic
  -> evidence adapters
  -> request-local Cognitive Workspace
  -> unified cognitive control (arbitration / selection)
  -> model
  -> explicit Use evidence
  -> lifecycle / consolidation
```

Expanded form:

```
Memory Domains -> source-specific logic -> Evidence Adapters
  -> Request-local Cognitive Workspace -> Arbitration / Selection
  -> Flash / Recall / Direct Data -> Model
  -> Explicit Use Evidence -> Lifecycle / Consolidation
```

Explicit non-goals of the direction:

- Memory domains **may keep independent storage and internal schemas**. The
  direction does not require one storage engine.
- The Cognitive Workspace is **not** a universal memory database. It is a
  per-request runtime authority, not canonical memory (§2.22).
- The north star is **not** a frozen implementation spec. It is expected to be
  revised as the code teaches us what is actually needed.

The v0 workspace is the smallest first step justified by the code that exists
today: it carries only the exact stage outputs the current Context chain already
produces, and it does not pre-design the arbitration / consolidation stages
sketched above.

---

## 8. Lifecycle authority reality — parallel planes (recorded, NOT solved)

> Status: WORKING DRAFT / REALITY MAP. This records what the code does at the
> baseline and does **not** claim the problem is solved. Full call-site matrix:
> `docs/cognitive-memory/LIFECYCLE_AUTHORITY_MAP.md`.

At this baseline the repository does **not** have one unified lifecycle owner. It
has **two parallel lifecycle authorities with different scopes**:

- **Legacy OB (canonical + archive) — real authority where it runs.**
  `DecayEngine.run_decay_cycle()` writes canonical `resolved` (auto-resolve) and
  moves low-score buckets into `archive/` (auto-archive); `trace(reinforce=True)`
  writes canonical activation metadata (`activation_count += 1`,
  `last_active = now`). `calculate_score()` additionally drives Dream candidate
  truncation and spontaneous-`breath` ranking. It never deletes.
- **New cognitive lifecycle (usage + surfacing) — gated and supplementary.**
  Only a validated, persisted `UseMemory(useref)` produces `used` evidence, which
  the reinforcement observer turns into immutable lifecycle events and derived
  `strength` / `accessibility` / `salience`. When its gates are ON,
  `memory_surfacing_lifecycle` adds a **supplementary Memory Flash surfacing
  constraint** — behaviour-affecting surfacing authority that does **not** mutate
  canonical storage.

Recorded facts:

- The legacy and cognitive planes **do not currently mutate each other's state.**
  The cognitive plane never calls `touch` / `touch_many` /
  `update(bump_active=True)` / `archive` / `delete` / `run_decay_cycle`; the
  legacy plane never reads `strength` / `accessibility` / `salience` / lifecycle
  event files / Use receipts.
- `trace(reinforce=True)` is an explicit legacy reinforcement judgement;
  `UseMemory(useref)` is explicit evidence that a recalled memory was actually
  used. **Neither is authorized to fabricate the other**, and no bridge is added.
- The start wiring is asymmetric: `streamable-http` starts `DecayEngine` eagerly;
  stdio starts it lazily on first use of certain tools. Documented, not changed.
- `resolved` is a semantic/task state; it is **not** an alias for forgotten /
  inaccessible / archived.

The correct framing is therefore **parallel lifecycle authorities with different
scopes**, not "one lifecycle implementation with a legacy duplicate".
**Migration remains unresolved**; the audit changes no live behaviour and adds
only guardrail tests.

### Explicitly NOT done in the Memory Lifecycle Authority Map + Dual-Authority Guardrails v0 task

- No change to live memory lifecycle behaviour.
- No `UseMemory` → `touch` bridge and no `trace(reinforce=True)` → `used` bridge.
- No disabling of `DecayEngine`, no enabling it in stdio, no startup change.
- No change to the decay formula, archive threshold, auto-resolve threshold,
  `activation_count` / `last_active` semantics, Memory Surfacing thresholds,
  retrieval ranking, or Dream ranking.
- No move / archive / restore / delete of any real user memory.
- No production abstraction added; only the authority map, guardrail tests, this
  status update, and one comment-only stale-comment correction.
