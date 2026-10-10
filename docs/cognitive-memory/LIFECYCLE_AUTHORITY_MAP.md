# Memory Lifecycle Authority Map — v0

**STATUS: WORKING DRAFT / REALITY MAP — NOT AN ADR — NOT A FROZEN SPECIFICATION**

This document is a **map of what the code does today**, written from the exact
baseline below. It is not a contract, not a design proposal and not a decision
record. The architecture is still evolving. If this document and the code
disagree, **the code is correct and this document is wrong**.

- Repository: `https://github.com/cnsj-123/vps-`
- Source of truth: `origin/feature/gateway-v2`
- Baseline HEAD: `ab586b34d0265a713d3e3ed3ac60741246093ed2`
  (merge of PR#2, `trae/request-local-cognitive-workspace-v0`)
- Audit scope: **authority audit + guardrails only.** This task changes **no**
  live memory lifecycle behaviour.

Every claim below was checked against the real modules. Where a claim could not
be verified from code it is marked `UNVERIFIED` rather than assumed. Where a code
comment and the runtime disagree, it is recorded as `COMMENT STALE / CODE
AUTHORITATIVE`.

---

## 0. Why this map exists

The repository currently carries **two different lifecycle vocabularies** that
share the word "lifecycle" but are not the same subsystem. They must not be
silently collapsed.

**Legacy OB (the "weight pool" / decay model):**

```
importance / arousal / activation_count / last_active / resolved / digested
  -> DecayEngine.calculate_score()
  -> spontaneous + Dream ranking, automatic lifecycle actions
  -> auto-resolve / archive
```

Legacy explicit reinforcement is:

```
trace(bucket_id, reinforce=True)
  -> tools/trace/_reinforce.py
  -> BucketManager.touch(bucket_id, ripple=True)
  -> activation_count + 1, last_active = now, temporal ripple
```

**New cognitive lifecycle (shadow-first, event-sourced):**

```
Recall
  -> Loaded
  -> explicit UseMemory(useref)
  -> persisted validated used
  -> lifecycle event (immutable receipt)
  -> strength / accessibility / salience (derived state)
  -> optional supplementary Memory Surfacing constraint
```

The single most important sentence in this map:

```
trace(reinforce=True)  !=  UseMemory
```

`trace(reinforce=True)` is an **explicit human/AI judgement that a memory should
be reinforced in the legacy OB model**. `UseMemory` is **evidence that a specific
recalled memory was actually used by the model in one recall/request**. Neither
signal may fabricate the other, and neither is wired to the other today.

---

## 1. Authority classification legend

For each writer/consumer, zero or more of the following are assigned. The labels
are deliberately narrow — a state can be `CANONICAL_METADATA_WRITE` without being
`COGNITIVE_USAGE_EVIDENCE_AUTHORITY`, and a module can be
`SPONTANEOUS_SURFACING_AUTHORITY` without being `STORAGE_RETENTION_AUTHORITY`.

| Label | Meaning |
| --- | --- |
| `OBSERVATION_ONLY` | Writes a shadow artifact / log line; feeds back into nothing live. |
| `CANONICAL_METADATA_WRITE` | Writes a canonical bucket Markdown `frontmatter` field. |
| `RETRIEVAL_RANKING_AUTHORITY` | Affects which/ordering of memories retrieval returns. |
| `SPONTANEOUS_SURFACING_AUTHORITY` | Affects the no-query `breath` surface / Dream candidate pool. |
| `MEMORY_FLASH_SURFACING_AUTHORITY` | Affects which base-eligible memory becomes a Memory Flash cue. |
| `DREAM_SELECTION_AUTHORITY` | Affects which buckets enter `dream`. |
| `COGNITIVE_USAGE_EVIDENCE_AUTHORITY` | Creates/validates the new-lifecycle `used` evidence. |
| `COGNITIVE_REINFORCEMENT_AUTHORITY` | Turns `used` into lifecycle reinforcement evidence. |
| `SEMANTIC_STATE_AUTHORITY` | Writes the task/semantic `resolved` state. |
| `STORAGE_RETENTION_AUTHORITY` | Decides whether canonical memory stays active vs archived. |
| `ARCHIVE_AUTHORITY` | Can move a canonical bucket into `archive/`. |
| `DELETE_AUTHORITY` | Can tombstone/soft-delete (or, for test data, physically delete) a bucket. |
| `RESTORE_AUTHORITY` | Can move an archived/tombstoned bucket back to an active channel. |
| `BACKGROUND_SCHEDULER_AUTHORITY` | Owns a background loop that periodically acts. |
| `COGNITIVE_DERIVED_STATE` | Derived, request-independent cognitive lifecycle state built from validated lifecycle evidence (`use_count` / `last_used` / `strength` / `accessibility` / `salience`). Implies **no** authority over `resolved`, canonical memory, archive or delete. |
| `PROVENANCE_EVIDENCE` | Non-mutating usage provenance that later validation depends on (e.g. `recall_requested` / `memory_loaded`). Grants **no** reinforcement or canonical authority; it only makes a later `used` event admissible. |

The point of this audit is to expose that the current code has **multiple owners**
with **different scopes**, not one unified lifecycle owner.

---

## 2. Legacy canonical metadata (per-bucket Markdown `frontmatter`)

Storage location for every field below: the bucket Markdown file under the
vault directories (`permanent/`, `dynamic/`, `feel/`, `plan/`, `letter/`,
`archive/`), read/written by `BucketManager`. Readers go through
`BucketManager.get` / `list_all` / `list_all(include_archive=...)`.

| Signal / field | Meaning in CURRENT code | Producer(s) / writer(s) | Reads into Retrieval | Reads into spontaneous Surfacing | Reads into Dream | Can cause resolve | Can cause archive | Transport/runtime dependence | Authority classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `created` | Bucket creation timestamp. Used as decay age fallback. | `BucketManager` create path | via recency scoring | yes (window filter) | yes (`is_within_window`) | no | no | none | `CANONICAL_METADATA_WRITE` |
| `importance` | Legacy "how important" 1..10. Anchors `DecayEngine.calculate_score` and auto-resolve gating. | explicit `trace(importance=...)`, import, `hold/grow` merge, `them` service, restore override | yes | yes | yes (sort) | yes (auto-resolve requires `imp <= 4`) | indirectly via low score | none | `CANONICAL_METADATA_WRITE`, `RETRIEVAL_RANKING_AUTHORITY` (indirect), `DREAM_SELECTION_AUTHORITY` (indirect) |
| `arousal` | Emotional arousal 0..1. Feeds `emotion_weight` in decay score. | explicit `trace(arousal=...)`, import | yes (scoring dim) | via decay score | via decay score | no | indirectly via score | none | `CANONICAL_METADATA_WRITE`, `RETRIEVAL_RANKING_AUTHORITY` (indirect), `SPONTANEOUS_SURFACING_AUTHORITY` (indirect) |
| `last_active` | Async "last real activation/recall" time; decay recency input. | `touch()`, `update(bump_active=True)`, `restore_archived()` | recency scoring | via decay score | window filter | yes (`_days_since_active`) | via score | none | `CANONICAL_METADATA_WRITE` |
| `activation_count` | Legacy activation counter. Decay score uses `activation_count ** 0.3`. | `touch()`, `update(bump_active=True)`, `_time_ripple` (+0.3) | yes (touch dim, capped/normalized) | via decay score | via decay score | no | via score | none | `CANONICAL_METADATA_WRITE` |
| `resolved` | **Semantic / task lifecycle state** ("this thing is dealt with"). NOT `forgotten`, NOT `archived`. | `DecayEngine.run_decay_cycle` auto-resolve; explicit `trace(resolved=...)`; plan→bucket cascade (`tools/_common.py`); `tools/i/core.py`; import API | no direct | excludes from spontaneous/`breath` surface | **excludes** from Dream candidates | this *is* the resolve write | no (accelerates score via `resolved_factor`) | `run_decay_cycle` requires running DecayEngine | `SEMANTIC_STATE_AUTHORITY` (+ auto-resolve is `BACKGROUND_SCHEDULER_AUTHORITY`) |
| `digested` | Set when a `feel` has been written for this memory. Accelerates decay. | `trace` / import / feel path via `update` | no direct | `breath` search hides from non-hit drift | Dream excludes | no | via score | none | `CANONICAL_METADATA_WRITE` |
| `dont_surface` | "Voluntary forget": hidden from active surfacing, bucket stays on disk. | `trace(dont_surface=...)`, web buckets toggle, `i` buckets | search filters non-hit drift | **excludes** from spontaneous/`breath` surface | Dream excludes | no | no | none | `SPONTANEOUS_SURFACING_AUTHORITY`, `DREAM_SELECTION_AUTHORITY` (never storage authority) |
| `pinned` | Pinned marker; `calculate_score` returns `999.0` (never archived). Counts against pinned quota. | `trace(pinned=...)`, web, merge | yes (pin boost) | yes | Dream excludes pinned | no | no (locked high score) | none | `CANONICAL_METADATA_WRITE`, `SPONTANEOUS_SURFACING_AUTHORITY` (indirect) |
| `protected` | Protected marker; never decays/archived. | `trace(protected=...)`, web, restore override | yes | yes | Dream excludes protected | no | no | none | `CANONICAL_METADATA_WRITE` |
| `anchor` | Boolean anchor flag on dynamic buckets; excluded from decay cycle. | `trace(anchor=...)`, `tools/anchor` | yes (anchor boost) | yes | Dream excludes anchored | no | no (excluded from cycle) | none | `CANONICAL_METADATA_WRITE` |
| `type` (lifecycle-relevant) | `permanent`/`feel`/`i`/`plan`/`letter` are skipped by decay and/or pinned to `_SCORE_FEEL`; `archived` is terminal-ish. | create path, `archive()` sets `type=archived`, `restore_archived()` restores `original_kind` | yes | yes | yes | plan/letter excluded from auto-resolve | archive sets `archived` | none | `STORAGE_RETENTION_AUTHORITY` (via `archive`) |
| `status` (lifecycle-relevant) | Plan status drives plan lifecycle (not time-driven). | `plan` tool | indirect | plan excluded | plan excluded | no | no | none | `SEMANTIC_STATE_AUTHORITY` (plan scope) |

---

## 3. Legacy actions (the mutation surface)

### 3.1 `BucketManager.touch(bucket_id, ripple=True)`

Owner: [bucket_manager.py](../../src/bucket_manager.py#L3148-L3196).

- Sets `last_active = now_iso()`, `activation_count += 1`; writes
  `TraceTouched` ledger event.
- With `ripple=True`, also runs `_time_ripple` which bumps up to 5 temporal
  neighbours' `activation_count` by `+0.3` (does not touch their `last_active`).
- **Mutates canonical memory: yes.** Authority:
  `CANONICAL_METADATA_WRITE`; it has **no** resolve/archive/delete authority.

Call sites (exact, whole branch):

| Call site | Why it exists | Notes |
| --- | --- | --- |
| [tools/trace/_reinforce.py#L44](../../src/tools/trace/_reinforce.py#L44) | The **only** explicit legacy reinforcement entrypoint (`trace(reinforce=True)`). | `ripple=True`. |
| [bucket_manager.py#L3208](../../src/bucket_manager.py#L3198-L3211) | `touch_many` delegates to `touch`. | See 3.2. |

### 3.2 `BucketManager.touch_many(bucket_ids, ripple=False)`

Owner: [bucket_manager.py](../../src/bucket_manager.py#L3198-L3211).

- Batch wrapper over `touch`. Docstring says "breath 浮现后一次性更新一批桶的激活，供后台任务调用".
- **Finding: no caller exists anywhere in the current branch.** Repo-wide search
  for `touch_many` finds only the definition and its internal `touch` call. The
  historical `breath` batch-touch caller was removed at 3.6.0 (see 3.7). This is
  a **dead public API**, retained but unused.
- **Authority capability:** `CANONICAL_METADATA_WRITE` — it mutates canonical
  activation metadata through `touch`.
- **Current runtime reachability:** `NO LIVE CALL SITE / DEAD PUBLIC API` at this
  baseline. An unused mutator does **not** become `OBSERVATION_ONLY`: it keeps its
  mutation capability and is simply unreachable today. `OBSERVATION_ONLY` is
  reserved for code that writes a shadow/log artifact and feeds no live mutation.

### 3.3 `BucketManager.update(..., bump_active=True)`

Owner: [bucket_manager.py](../../src/bucket_manager.py#L2166-L2205), implementation
[bucket_manager.py#L2621-L2623](../../src/bucket_manager.py#L2621-L2623).

- When `bump_active=True`: `last_active = now_iso()` and `activation_count += 1`
  (semantics identical to `touch`, plus the metadata edit). When `False` (default),
  a pure metadata/content edit that deliberately does **not** refresh activation.
- **Mutates canonical memory: yes.** Authority: `CANONICAL_METADATA_WRITE`.

Call sites with `bump_active=True` (exact):

| Call site | Why it exists |
| --- | --- |
| [tools/_common.py#L1098](../../src/tools/_common.py#L1093-L1100) inside `_merge_or_create_inner` | `hold` / `grow` merging into an existing near-neighbour bucket is treated as a real re-activation. |

`bump_active=False` is the default used by all other `update(...)` callers:
explicit `trace` metadata edits, `plan`/`anchor` edits, `DecayEngine`
auto-resolve, import, web toggles, etc. — none of which refresh activation.

### 3.4 `BucketManager.archive(bucket_id)`

Owner: [bucket_manager.py#L3826-L3885](../../src/bucket_manager.py#L3826-L3885).

- Moves the Markdown into `archive/<domain>/`, sets `type = "archived"`, records
  `TraceArchived`. **Does not delete**: the memory is retained, just no longer in
  an active channel.
- **Mutates canonical memory: yes.** Authority: `ARCHIVE_AUTHORITY` +
  `STORAGE_RETENTION_AUTHORITY`. It is **not** `DELETE_AUTHORITY`.

Archive call sites (exact):

| Call site | Why it exists |
| --- | --- |
| [decay_engine.py#L359](../../src/decay_engine.py#L357-L366) | Automatic forgetting: score `< threshold` → archive. **The primary legacy automatic archive authority.** |
| [deletion_requests.py#L31](../../src/deletion_requests.py#L30-L36) | Human web-UI "archive" action (`HumanDeleteExecutor.execute(action="archive")`). |

### 3.5 Archive restore path — `BucketManager.restore_archived(...)`

Owner: [bucket_manager.py#L2770-L2910](../../src/bucket_manager.py#L2770-L2910).

- Explicit-only restoration; docstring: "Discovery never calls this method. It is
  deliberately exposed only through an explicit `trace(..., restore=True)`."
- Clears `deleted_at` / `tombstone` / `tombstoned_at` / `erasure_mode`, restores
  `original_kind`, refreshes `last_active = now_iso()` (so a low-score bucket is
  not immediately re-archived).
- **Mutates canonical memory: yes.** Authority: `RESTORE_AUTHORITY`.

Restore call sites (exact):

| Call site | Why it exists |
| --- | --- |
| [tools/trace/core.py#L428](../../src/tools/trace/core.py#L324-L430) | Explicit `trace(restore=True)`. |
| [tools/_common.py#L577](../../src/tools/_common.py#L577) `restore_archived_letters` | Letter-specific restore; called from [web/meta.py#L1480](../../src/web/meta.py#L1411-L1480). |

### 3.6 `BucketManager.delete` / tombstone behaviour

Owner: [bucket_manager.py#L2758-L2768](../../src/bucket_manager.py#L2758-L2768) via `_delete_locked`.

- **Soft delete**: moves to `archive/` and stamps `deleted_at`; embeddings are
  discarded. Docstring: "F-10: 记忆不消失，只是淡去。不做物理删除".
- Physical removal exists only for test data:
  `hard_delete_test_bucket` ([bucket_manager.py#L2696](../../src/bucket_manager.py#L2696))
  requires a `provenance.kind == "test"` / `erasable is True` bucket and a
  non-empty reason.
- **Mutates canonical memory: yes.** Authority: `DELETE_AUTHORITY` (soft/tombstone),
  plus a narrow physical-delete path for test buckets only.

Delete / hard-delete call sites (exact):

| Call site | Why it exists |
| --- | --- |
| [tools/trace/core.py#L489](../../src/tools/trace/core.py#L488-L495) | `trace(delete=True)` → soft delete (archive). |
| [tools/trace/core.py#L470](../../src/tools/trace/core.py#L470-L486) | `trace(hard_delete=True)` → physical delete of test data only, with `delete_reason`. |
| [deletion_requests.py#L38](../../src/deletion_requests.py#L37-L45) | Human web-UI "delete" action. |

### 3.7 `trace(reinforce=True)` and explicit trace delete/restore

Owner: [tools/trace/core.py](../../src/tools/trace/core.py), dispatch at
[L258-L272](../../src/tools/trace/core.py#L258-L272) (reinforce),
[L274-L495](../../src/tools/trace/core.py#L274-L495) (restore/delete).

- `reinforce=True` is a standalone early-return: it must be called alone and
  routes to [_reinforce.py](../../src/tools/trace/_reinforce.py) →
  `BucketManager.touch(ripple=True)`.
- Authority: `CANONICAL_METADATA_WRITE` (activation), **not**
  `COGNITIVE_USAGE_EVIDENCE_AUTHORITY`. It creates no new-lifecycle `used`
  signal and no lifecycle receipt.

### 3.8 `DecayEngine`

Owner: [decay_engine.py](../../src/decay_engine.py).

| API | What it does | Authority | Mutates canonical memory |
| --- | --- | --- | --- |
| `calculate_score(metadata)` ([L180-L272](../../src/decay_engine.py#L180-L272)) | Pure scoring: `importance × activation_count^0.3 × e^(-λ·days) × combined_weight`, modulated by `resolved`/`digested`/`arousal`. Returns `999.0` for pinned/protected/permanent, `50.0` for feel/plan/letter. | `SPONTANEOUS_SURFACING_AUTHORITY`, `DREAM_SELECTION_AUTHORITY`, input to `STORAGE_RETENTION_AUTHORITY` / `ARCHIVE_AUTHORITY` (not a mutator; **not** canonical retrieval ranking) | no |
| `run_decay_cycle()` ([L280-L385](../../src/decay_engine.py#L280-L385)) | Iterates active buckets; **auto-resolves** (`imp <= 4` and `> 30` days) via `bucket_mgr.update(resolved=True)`, then **archives** buckets with `score < threshold`. Also self-heals embeddings. | `BACKGROUND_SCHEDULER_AUTHORITY`, `ARCHIVE_AUTHORITY`, `SEMANTIC_STATE_AUTHORITY` (auto-resolve) | **yes** |
| `ensure_started()` / `start()` / `stop()` ([L446-L476](../../src/decay_engine.py#L446-L476)) | Idempotent background loop control. `start()` launches `_background_loop` (first cycle immediately, then every `check_interval` hours). | `BACKGROUND_SCHEDULER_AUTHORITY` | no (scheduler only) |

`DecayEngine` **has real canonical + archive authority when it is running**
(auto-resolve writes `resolved`; low scores archive). It never deletes.

### 3.9 Stale comment found (corrected, no behaviour change)

[bucket_manager.py#L3145-L3147](../../src/bucket_manager.py#L3143-L3148) stated:

```
# Called on every recall hit; affects decay score.
# 每次检索命中时调用，影响衰减得分。
```

This was **COMMENT STALE / CODE AUTHORITATIVE**. Since OB 3.6.0,
`breath(query=...)` search and the spontaneous surface are read-only and do not
call `touch` (see §7 and §11). The only `touch` caller is `trace(reinforce=True)`.
The comment was corrected in the map commit because it is clearly incorrect and
has zero runtime effect.

---

## 4. New cognitive lifecycle (shadow-first, event-sourced)

All new-lifecycle artifacts are written under `OMBRE_CONTEXT_STATE_DIR`
(see [recall_types.py#L61](../../src/ombrebrain/context/recall_types.py#L61-L70));
none of them write a canonical memory field.

| Signal | Meaning in CURRENT code | Producer / writer | Storage | Readers / consumers | Mutates canonical memory | New-lifecycle authority |
| --- | --- | --- | --- | --- | --- | --- |
| `recall_requested` | The AI explicitly asked for an anchor; records even if nothing could be loaded. Persists the request-bound usage artifact that precedes loading. | `memory_usage_signal.record_recall_requested` called by [related_recall.py#L1184](../../src/ombrebrain/context/related_recall.py#L1181-L1190) | `memory_usage/{cid}/{reqid}/{recall_id}.json` | lifecycle/observability | no | `PROVENANCE_EVIDENCE` (usage-admissibility prerequisite) |
| `memory_loaded` | A persisted Related Recall result was returned. Derived from a persisted artifact, never from a retrieval candidate. Enforced prerequisite: `used ⊆ loaded`; a never-loaded memory is refused with `memory_not_loaded`. | `memory_usage_signal.record_memory_loaded` from [related_recall.py#L1168](../../src/ombrebrain/context/related_recall.py#L1166-L1174) and [L1262](../../src/ombrebrain/context/related_recall.py#L1257-L1268) | same Usage Signal file | lifecycle/observability | no | `PROVENANCE_EVIDENCE` (usage-admissibility prerequisite) |
| `used` (explicit) | The model declared that a **specific recalled memory** was actually used. Only reachable via `UseMemory(useref)`. | `attribute_live_recall_usage` → `memory_usage_signal.record_usage` ([attribution L363](../../src/ombrebrain/context/live_recall_usage_attribution.py#L360-L370)); tool wiring [server.py#L1500](../../src/server.py#L1487-L1510) | same Usage Signal file (+ request-scoped Usage Surface) | lifecycle reinforcement observer; surfacing gate | no | `COGNITIVE_USAGE_EVIDENCE_AUTHORITY` |
| lifecycle event / reinforcement receipt | Immutable deterministic receipt for one `used` slot. Once, ever. | `record_lifecycle_event_from_usage` ([memory_lifecycle_event.py#L580](../../src/ombrebrain/context/memory_lifecycle_event.py#L580-L732)) invoked by `observe_memory_usage_lifecycle` ([memory_reinforcement.py#L469](../../src/ombrebrain/context/memory_reinforcement.py#L468-L474)) | `memory_lifecycle_events/…` | `lifecycle_use_facts`, surfacing gate | no | `COGNITIVE_REINFORCEMENT_AUTHORITY` |
| `use_count`, `last_used` | Derived facts over the memory's `used` events at `as_of`. | `lifecycle_use_facts` ([memory_lifecycle_event.py#L793](../../src/ombrebrain/context/memory_lifecycle_event.py#L793)) | derived (not persisted standalone) | state builder, surfacing gate | no | `COGNITIVE_DERIVED_STATE` |
| `strength` | Monotone gain over `use_count` (`strength_for_use_count`, bounded). | `build_lifecycle_state` ([memory_lifecycle_state.py#L394](../../src/ombrebrain/context/memory_lifecycle_state.py#L394-L396)) | `memory_lifecycle_state/<key>.json` | surfacing gate (re-derives, does not trust snapshot) | no | `COGNITIVE_DERIVED_STATE` |
| `accessibility` | Decays with age; **existence != accessibility**; low accessibility != deletion. | `accessibility_for_age` via `build_lifecycle_state` ([L398](../../src/ombrebrain/context/memory_lifecycle_state.py#L398-L404)) | state snapshot | surfacing gate | no | `COGNITIVE_DERIVED_STATE` |
| `salience` | Decays with time-since-last-use (and use_count). | `salience_for_age` ([L406](../../src/ombrebrain/context/memory_lifecycle_state.py#L406-L414)) | state snapshot | surfacing gate | no | `COGNITIVE_DERIVED_STATE` |
| lifecycle Surfacing gate | One constraint on top of the base Surfacing Policy; can only further **constrain** supplementary eligibility. | `apply_lifecycle_surfacing_gate` ([memory_surfacing_lifecycle.py#L545](../../src/ombrebrain/context/memory_surfacing_lifecycle.py#L545)), wired at [context_pipeline_coordinator.py#L647](../../src/ombrebrain/context/context_pipeline_coordinator.py#L159-L182) | reads events, writes nothing | Memory Flash stage | **no** | `MEMORY_FLASH_SURFACING_AUTHORITY` (supplementary only, gated) |

Note on `recall_requested` / `memory_loaded`: they are **not** reinforcement and
**not** canonical mutation, but they are **not** pure `OBSERVATION_ONLY` either.
They are `PROVENANCE_EVIDENCE` / usage-admissibility prerequisites: validation
enforces `used ⊆ loaded`, so a never-loaded memory cannot become `used` (refused
with `memory_not_loaded`). They grant no reinforcement authority. The rungs stay
distinct:

```
recall_requested  !=  loaded  !=  used  !=  reinforced
```

New-lifecycle storage layout:

```
OMBRE_CONTEXT_STATE_DIR/
  memory_usage/{conversation_id}/{cognitive_request_id}/{recall_id}.json
  memory_lifecycle_events/…
  memory_lifecycle_state/<memory_key>.json
  live_recall_usage_surface/…
```

All of it is `MODE = "shadow_only"`. Default OFF flags (§9).

---

## 5. Multiple owners — the honest authority matrix

There is **no single lifecycle owner**. Authority is split across two planes.

### 5.1 Canonical metadata plane (legacy OB)

- Owner of "what is written to the bucket file": `BucketManager` (`touch`,
  `update`, `archive`, `delete`, `restore_archived`).
- Owners of the *decision* to write: `trace` (explicit), `DecayEngine`
  (automatic), `hold`/`grow` merge (`bump_active=True`), web UI
  (`deletion_requests`), import.

### 5.2 Cognitive plane (new lifecycle)

- Owner of `used` evidence: `memory_usage_signal` + `live_recall_usage_attribution`
  (only via explicit `UseMemory(useref)`).
- Owner of reinforcement evidence: `memory_reinforcement`
  (`observe_memory_usage_lifecycle` → `record_lifecycle_event_from_usage`).
- Owner of derived cognitive state: `memory_lifecycle_state`
  (`strength` / `accessibility` / `salience`).
- Owner of supplementary Flash surfacing constraint:
  `memory_surfacing_lifecycle` (only when both its gates are ON).

### 5.3 Cross-plane rules verified today

- The cognitive plane **never** calls **legacy canonical mutation APIs**:
  `touch` / `touch_many` / `update(bump_active=True)` / `archive` / `delete` /
  `hard_delete_test_bucket` / `DecayEngine.run_decay_cycle`. Its only canonical
  read is `BucketManager.get` ([memory_reinforcement.py#L59-L63](../../src/ombrebrain/context/memory_reinforcement.py#L56-L67)).
- **Inside** the cognitive plane, the intended new-lifecycle path is:
  validated persisted `used` → `memory_reinforcement.observe_memory_usage_lifecycle`
  → `record_lifecycle_event_from_usage` → immutable lifecycle reinforcement
  receipt. `record_lifecycle_event_from_usage` is a **new-lifecycle** writer
  (defined in `memory_lifecycle_event.py`, invoked by `memory_reinforcement.py`),
  **not** a legacy canonical mutator, so it is deliberately **not** in the
  forbidden list above. This in-plane `used` → reinforcement path is intended and
  must not be removed or weakened.
- `DecayEngine` **never** reads `strength` / `accessibility` / `salience` /
  lifecycle event files / Usage receipts (verified by inspection of
  [decay_engine.py](../../src/decay_engine.py) — no such import).
- `trace(reinforce=True)` writes legacy activation only; it creates no lifecycle
  `used` signal and no lifecycle event.
- The legacy and cognitive planes **do not currently mutate each other's state**.
  The only bridge is `used` → reinforcement, and *both* are inside the cognitive
  plane.

### 5.4 Authority contention summary

| Authority | Legacy owner | New-lifecycle owner |
| --- | --- | --- |
| Canonical metadata write | `BucketManager` | none |
| Canonical retrieval ranking (`BucketManager.search`) | `BucketManager.search` own weighted dims: topic / emotion / time / importance / touch frequency | none |
| `calculate_score` read-only display/sort (`anchor`, `them`, web) | `DecayEngine.calculate_score` | none |
| Spontaneous surfacing | legacy `dont_surface` / `resolved` + `calculate_score` ranking | none |
| Memory Flash surfacing | base Surfacing Policy | `memory_surfacing_lifecycle` (supplementary, gated) |
| Dream selection | legacy `calculate_score` truncation | none |
| Usage evidence | none | `memory_usage_signal` |
| Cognitive reinforcement | `trace(reinforce=True)` (legacy activation) | `memory_reinforcement` (event-sourced) |
| Semantic `resolved` | `DecayEngine` auto-resolve + explicit `trace` | none |
| Archive | `DecayEngine` (automatic) + human web | none |
| Delete | explicit `trace` / human web (+ test-only physical) | none |
| Restore | explicit `trace(restore=True)` + letters restore | none |
| Background scheduler | `DecayEngine._background_loop` | none |

**Scoring systems must not be conflated.** Canonical retrieval ranking is
`BucketManager.search`'s own weighted sum over topic / emotion / time /
importance / touch frequency
([bucket_manager.py#L3454-L3476](../../src/bucket_manager.py#L3454-L3476)).
`DecayEngine.calculate_score()` is a **different** score and is **not** the
canonical retrieval score. Legacy metadata such as `importance` / `last_active` /
`activation_count` / `arousal` may influence retrieval **through
`BucketManager.search`**, but `calculate_score()` itself only feeds spontaneous
surfacing, Dream selection and the archive decision, plus read-only display
consumers.

---

## 6. Five concepts that must stay separate

These are **different current/future concepts** and must not be equated:

1. **RETRIEVED** — a memory was found (retrieval / Recall surface).
2. **USED** — the model actually used a specifically recalled memory. Only
   validated persisted `UseMemory` evidence proves this in the new lifecycle.
3. **INTENTIONALLY REINFORCED / MARKED IMPORTANT** — an explicit decision that a
   memory deserves stronger persistence/access; in legacy OB this is
   `trace(reinforce=True)`.
4. **COGNITIVE ACCESSIBILITY** — how likely/easy a memory is to surface now. May
   decay without destroying the memory.
5. **STORAGE RETENTION** — whether the canonical memory stays active, is
   archived, restored, or explicitly deleted.

Do **not** equate any of these pairs:

```
retrieved        != used
surfaced         != used
trace(reinforce=True) != used
importance       != strength
activation_count != use_count
last_active      != last_used
resolved         != forgotten
low accessibility != archived
archived         != deleted
```

---

## 7. `resolved` must be treated separately

`resolved` is a **semantic / task lifecycle state** ("this item is dealt with").
It must not silently become an alias for `forgotten`, `inaccessible`, or
`archived`.

Current writers of `resolved` (exact audit):

| Writer | Path | Nature |
| --- | --- | --- |
| `DecayEngine.run_decay_cycle` auto-resolve | [decay_engine.py#L325-L342](../../src/decay_engine.py#L323-L342) | Automatic: `importance <= 4` AND `> 30` days since last activation AND not already resolved → `update(resolved=True)`. Runs **only while the engine is running**. |
| Explicit `trace(resolved=...)` | [tools/trace/core.py#L628](../../src/tools/trace/core.py#L626-L628) | Explicit human/AI decision. |
| Plan→bucket cascade | [tools/_common.py#L1463](../../src/tools/_common.py#L1455-L1467) | When a plan completes, related buckets are marked resolved. |
| `i` bucket promote path | [tools/i/core.py#L407](../../src/tools/i/core.py#L400-L410) | Internal. |
| Import API | [web/import_api.py#L877](../../src/web/import_api.py#L860-L880) | Imported field. |

`resolved` does **not** delete or archive anything by itself. In
`calculate_score` it only applies `_FACTOR_RESOLVED_DIGESTED` / `_FACTOR_RESOLVED_ONLY`
to accelerate score decay, and spontaneous/Dream paths hide resolved items.
Recorded honestly; **not changed** in this task.

---

## 8. `trace(reinforce=True)` vs `UseMemory(useref)`

| Dimension | `trace(reinforce=True)` | `UseMemory(useref)` |
| --- | --- | --- |
| Meaning | Explicit legacy judgement: "this memory should be reinforced." | Explicit evidence: "this recalled memory was actually used in this request." |
| Entry | [tools/trace/core.py#L258](../../src/tools/trace/core.py#L255-L272) → [_reinforce.py](../../src/tools/trace/_reinforce.py) | [server.py `_live_memory_usage_tool`](../../src/server.py#L1487-L1510) → `attribute_live_recall_usage_with_lifecycle` |
| Provenance required | none beyond a valid bucket id | trusted `conversation_id` + `cognitive_request_id` from the transport ContextVar, and an opaque `useref` of THIS request's Usage Surface |
| Persists | legacy `activation_count += 1`, `last_active = now` | `used` in the Usage Signal artifact |
| Side effects | temporal ripple over up to 5 neighbours | conflicts (`used ⊆ loaded`) are refused; then lifecycle event / state if the triple gate is ON |
| Writes canonical memory | **yes** | **no** |
| Creates new-lifecycle `used` | **no** | **yes** |
| Creates lifecycle reinforcement receipt | **no** | **yes** (only if triple gate ON) |
| Feeds `DecayEngine` | yes (via `activation_count` / `last_active`) | **no** |
| Feeds new `strength`/`accessibility`/`salience` | **no** | indirectly (via `used` → state) |

**Neither is currently authorized to fabricate the other.** There is no code
path that turns `trace(reinforce=True)` into `used`, and none that turns
`UseMemory` into `touch`. **No bridge is added in this task.**

---

## 9. Transport / runtime asymmetry (real, not fixed here)

### 9.1 Startup wiring audit (exact)

`streamable-http` ([server.py#L1719-L1742](../../src/server.py#L1719-L1742)):

```
RuntimeLifecycle(logger, decay_engine=decay_engine, embedding_outbox=…, you_service=…, …)
  -> install_runtime_lifespan()  (server_app.py)
  -> RuntimeLifecycle.start()    (server_app.py#L837-L845)
  -> await self.decay_engine.start()
```

`stdio` ([server.py#L1883-L1896](../../src/server.py#L1883-L1896)):

```
RuntimeLifecycle(logger, embedding_outbox=…, you_service=…, boot_marker_path=…)
  -> NO decay_engine is passed
```

`RuntimeLifecycle.start()` only starts a service when the attribute is not `None`
([server_app.py#L842-L845](../../src/server_app.py#L837-L845) via
`getattr(self.decay_engine, "start", None)`). With `decay_engine=None`, the stdio
lifespan starts the embedding outbox and You service but **never starts the decay
engine at server startup**.

### 9.2 Alternate startup path found (this corrects the naive reading of the task brief)

`rt.decay_engine` **is** injected into `tools._runtime` regardless of transport
([server.py#L754-L769](../../src/server.py#L750-L769), `decay_engine=decay_engine`).
Several **tool entrypoints call `await rt.decay_engine.ensure_started()` lazily**:

- [tools/breath/__init__.py#L146](../../src/tools/breath/__init__.py#L146)
- [tools/dream/__init__.py#L33](../../src/tools/dream/__init__.py#L33)
- [tools/hold/__init__.py#L196](../../src/tools/hold/__init__.py#L196)
- [tools/grow/__init__.py#L136](../../src/tools/grow/__init__.py#L136)
- [tools/anchor/core.py#L79](../../src/tools/anchor/core.py#L79)
- [tools/plan/core.py#L213](../../src/tools/plan/core.py#L213)
- [tools/i/core.py#L86](../../src/tools/i/core.py#L86)

`ensure_started()` is idempotent and calls `start()` (which launches the
background loop). **Therefore, under stdio, the DecayEngine can be started lazily
on the first invocation of any of those tools** even though it is not started at
server startup. An alternate stdio DecayEngine startup path **was found** — the
lazy `ensure_started()` path above. It is not a `start()`/import side effect; it
is an explicit first-use lazy start.

No other path was found: no import-time `DecayEngine.start()`, no module-level
background task, and no other `start()`/`ensure_started()` caller (repo-wide
search over the current branch).

### 9.3 The recorded asymmetry

- **At server startup:** `streamable-http` starts DecayEngine eagerly;
  `stdio` does not.
- **Lazily at first use:** both transports can start DecayEngine through the tool
  `ensure_started()` callers above.
- Consequence: under stdio, a session that never calls those tools never runs
  automatic decay / auto-resolve / auto-archive.

This is a **REAL runtime authority asymmetry** and is documented, not "fixed", in
this task.

---

## 10. Exact consumer audit for `DecayEngine.calculate_score()`

The decay score is **not only** an archive mechanism; it also has Dream selection
influence (baseline fact I).

| Consumer | Location | Why |
| --- | --- | --- |
| Decay archive decision | [decay_engine.py#L345](../../src/decay_engine.py#L344-L359) | `score < threshold` → archive. |
| Dream candidate truncation | [tools/dream/candidates.py#L116](../../src/tools/dream/candidates.py#L114-L119) | > 40 recent candidates → sort by score, keep 40. **Dream selection influence.** |
| `breath` search non-hit drift seed | [tools/breath/search.py#L453](../../src/tools/breath/search.py#L438-L455) | Only selects low-weight buckets for random drift; read-only. |
| `breath` spontaneous surface ranking | [tools/breath/surface.py#L311,L330,L356,L408](../../src/tools/breath/surface.py#L300-L412) | Sort / sample the spontaneous pool; read-only, no `touch`. |
| `anchor` score display | [tools/anchor/core.py#L176](../../src/tools/anchor/core.py#L176) | Display. |
| `them` person ranking | [ombrebrain/them/service.py#L535](../../src/ombrebrain/them/service.py#L535) | Person "vividness". |
| Web dashboard / buckets / search / hooks | [web/buckets.py#L169,L252,L271](../../src/web/buckets.py#L169), [web/search.py#L252,L373](../../src/web/search.py#L252), [web/hooks.py#L279](../../src/web/hooks.py#L279) | Display / sort. |

All consumers other than `run_decay_cycle` are **read-only** with respect to
canonical memory.

---

## 11. Read-only Retrieval / Surfacing confirmation

- `breath(query=...)` search: [tools/breath/search.py#L421-L434](../../src/tools/breath/search.py#L421-L434)
  explicitly documents and implements "检索不再强化" — no `touch` on hits.
- Spontaneous surface: [tools/breath/surface.py#L25](../../src/tools/breath/surface.py#L25)
  "不调用 touch()：浮现不能重置衰减计时器".
- `breath` importance path: [tools/breath/importance.py#L18](../../src/tools/breath/importance.py#L18)
  "不主动 touch（浮现行为不应重置衰减计时器）".

So baseline facts A, B and I hold; fact C holds; fact G holds; fact H is refined by
§9.2.

---

## 12. Working two-plane model (NON-NORMATIVE)

> Not implemented here. This is a working direction only, recorded so later
> phases share one vocabulary. If it and the code disagree, the code wins.

### A. Cognitive Lifecycle Plane

```
validated Used evidence
  -> strength / accessibility / salience
    -> cognitive admission / surfacing decisions
```

- Owns: usage evidence, reinforcement evidence, derived cognitive state,
  supplementary surfacing admission.
- Must **not** automatically own canonical archive/delete.

### B. Retention / Storage Lifecycle Plane

```
active / archived / restored / explicitly deleted ( + future retention policy )
```

- Owns: canonical retention, archive, restore, explicit delete.
- Storage retention must **not** be inferred solely from low cognitive
  accessibility.

The current reality (§3–§5) is that plane A exists only as a gated shadow and
plane B is still the legacy `BucketManager` + `DecayEngine`. They are **parallel
authorities with different scopes**, not one unified owner.

---

## 13. Known ambiguities / migration questions

- **MQ-1 — Two parallel lifecycle scopes.** Legacy decay (plane B) is authoritative
  today where it runs; the cognitive plane (A) is gated and supplementary. There is
  no single owner. Migration is **unresolved** and not attempted here.
- **MQ-2 — stdio vs streamable-http start asymmetry** (§9). Not fixed.
- **MQ-3 — `activation_count` vs `use_count`; `last_active` vs `last_used`.** Two
  parallel counters with different semantics; neither feeds the other today.
- **MQ-4 — `resolved` overload.** Semantic state also accelerates decay and hides
  from surfacing. Recorded; not changed.
- **MQ-5 — `touch_many` is a dead API.** Retained but uncalled.
- **MQ-6 — Stale comment at `bucket_manager.py#L3145`** (§3.9) corrected without
  behaviour change.

---

## 14. Explicitly NOT done in this task

- No change to decay formula, archive threshold, or auto-resolve threshold.
- No disabling of DecayEngine, no enabling it in stdio, no change to any startup
  behaviour.
- No change to `activation_count` / `last_active` semantics.
- No `UseMemory` → `touch` bridge and no `trace(reinforce=True)` → `used` bridge.
- No legacy `activation_count` feeding new `strength`; no new `strength` feeding
  `DecayEngine`.
- No change to Memory Surfacing thresholds, retrieval ranking, or Dream ranking.
- No move/archive/restore/delete of any real user memory.
- No ArchiveCandidate behaviour, consolidation, Episodic / Prospective Memory /
  Wake, Frontend Memory delivery change, or Emotion / Attachment.

This is a **working reality map**, not a frozen ADR or specification.