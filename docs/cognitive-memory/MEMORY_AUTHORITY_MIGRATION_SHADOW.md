# Memory Authority Migration Shadow v0

```
WORKING EXPERIMENT
MIGRATION READINESS OBSERVATION
NOT ADR
NOT LIVE AUTHORITY
NOT FROZEN SPEC
```

This document describes an **experiment**, not a contract. If this document and
the code disagree, the code is correct and this document is wrong. Nothing here
is normative, nothing here is frozen, and nothing here grants runtime authority.

It is a **readiness measurement**, not the migration. It observes how much of the
model-facing memory exposure of one request is already covered by a
single-authority control contract, and how much still bypasses it. It never
changes what the model sees.

Companion documents:

- [`ARCHITECTURE_STATUS.md`](ARCHITECTURE_STATUS.md) — the reality map of the
  whole cognitive-memory chain (this experiment is recorded there as §2.27).
- [`UNIFIED_MEMORY_DELIVERY.md`](UNIFIED_MEMORY_DELIVERY.md) — the Unified Memory
  Delivery Plan Shadow, the control contract this experiment measures coverage
  against.
- [`LIFECYCLE_AUTHORITY_MAP.md`](LIFECYCLE_AUTHORITY_MAP.md) and
  [`LIFECYCLE_SHADOW_COMPARISON.md`](LIFECYCLE_SHADOW_COMPARISON.md) — the
  lifecycle plane, which this experiment **intentionally does not consume**.

Owner modules:

- `src/ombrebrain/context/memory_authority_migration_shadow.py` — the Migration
  Readiness Shadow, artifact `memory-authority-migration-shadow.v1`.
- `src/ombrebrain/maintenance/memory_authority_migration_report.py` — read-only
  aggregate reporter.
- `src/ombrebrain/context/context_pipeline_coordinator.py` — integration
  (post-final-body observer).
- `src/ombrebrain/context/context_request_workspace.py` — the request-local
  `memory_authority_migration` field.

The observable output is a **comparison**, produced after the final provider body
has already been chosen, from evidence THIS request already produced. The artifact
records what the final body exposes `VS` what a single-authority control contract
covers. It is evidence, not a decision. **`live_cutover_ready` is always `false`
in v0.**

---

## 1. The current two-mouth architecture

Two memory paths currently reach the model, with different access semantics
(see `ARCHITECTURE_STATUS.md` §4):

```
Frontend Memory
    a ``relevant_memories.txt`` attachment is already present in the
    forwarded provider request
        -> the model sees the WHOLE attachment directly
        -> preserved by cache rewriting, never removed
        -> this is a DIRECT mouth: full content, no control gate

OB / cognitive memory
    retrieval -> Surfacing Policy -> optional Lifecycle gate
        -> Memory Flash (bounded cue)
            -> Recall Surface (memref)
                -> optional Live Memory Flash Exposure
                    -> model sees a bounded cue + opaque memref
                        -> full content only on an explicit Recall
        -> this is a CONTROLLED mouth
```

Both mouths can be active in the **same** request. The Migration Shadow records
`frontend_direct_mouth_active`, `ob_live_cue_mouth_active` and `dual_mouth_active`
as booleans derived from the request's own evidence.

The point of the "single-authority" hypothesis is **not** unified storage. It is
that each memory system may keep its own brain and storage, but must **not each
have a direct mouth to the model**: every model-facing memory occurrence should
eventually carry a controller-owned delivery decision.

## 2. Why the PR #5 Delivery Plan is insufficient for live cutover

PR #5 produced the Source-Neutral Memory Evidence Snapshot and the Unified Memory
Delivery Plan Shadow. That Delivery Plan is a **proposal**
(`authority="proposal_only"`, `live_authority=false`). It describes the channel a
unit **already** uses (`immediate_delivery_mode`: `direct_data` / `flash_cue` /
`drop` / `unknown`) and the full-content access mode (`already_present` /
`recall_required` / `none` / `unknown`), but:

- it does **not** decide the actual final provider body;
- it does **not** measure how much of the current exposure is genuinely **outside**
  any controlled contract (the Frontend direct bypass);
- it does **not** state, per request, which cases already have a controlled cue /
  withhold contract and which are still unknown;
- it is `policy_stage="base_surfacing_pre_lifecycle"` — i.e. it is explicitly
  **pre-lifecycle** and excludes Lifecycle composition.

A delivery **plan** is not a migration **readiness** measurement. The Migration
Shadow is the next observation layer: it binds the proposal to the **actual final
body** and expresses coverage/gap in one closed vocabulary.

## 3. Why this task does NOT generate a hypothetical target body

v0 deliberately **does not** build any of:

```
frontend_removed_body
unified_memory_body
target_body
```

It never removes a Frontend attachment (not even in a copy), never introduces a
Frontend renderer / memref / Recall, and never invents a replacement body. The
migration target is **not yet defined** (questions A–H below are open), so any
"target body" would fabricate a decision that no authority has made. The only
Frontend migration conclusion v0 may ever express is the **requirement** that a
`controlled_frontend_channel_required` — never a claim that such a channel exists.

The counterfactual is **numeric-only**: it reports the model-visible footprint
that would disappear if the direct mouth were removed with no replacement. It does
**not** produce the post-removal body.

## 4. Actual final-body Frontend verification

The observer must not trust the Delivery Plan's **contract** when it can measure
**actuality** on the final body. `observe_frontend_final_presence(final_body,
source_evidence_snapshot)` returns `(available, verified)`.

The final Frontend body is canonicalized **exactly once** by Migration: the
orchestration boundary performs the single observation, and the pure builder
(`build_memory_authority_migration_artifact`) receives it as a precomputed value
and never re-canonicalizes, never calls the Operit adapter and never rereads any
request-global state. The Source Evidence observation of the **original
forwarded body** is a **different** pass with its own single canonicalization;
the two passes inspect different body stages and are not collapsed into one.

- **`available`** is `false` whenever the final body cannot be canonicalized (via
  the existing canonicalizer / `OperitAdapter`) or a Frontend segment is malformed.
  A comparison is never fabricated from a broken final body.
- **`verified`** is `true` **only** when the final body's Frontend occurrence
  multiset **exactly equals** THIS request's Source Evidence Frontend multiset.

The match is an **occurrence-safe multiset match** over
`(scope, content_digest, model_visible_chars, model_visible_estimated_tokens)`:
counts, placement, digest and footprint must all agree, so **duplicate identical
attachments stay distinct** and one attachment cannot stand in for another.

Current Frontend exposure is **attachment-level**: one `relevant_memories.txt`
attachment is ONE occurrence. The inner body is never parsed, and no item-level
readiness is claimed. On a **mismatch**, the artifact records the observation as
unverified and the coverage status cannot silently become "covered" for the
Frontend path.

## 5. Actual Live Exposure observation

The OB mouth's **actual** exposure is observed from the request-scoped Live
Exposure receipt (`read_live_memory_exposure`), and only when the body actually
changed (`live_body_changed`). The receipt is **request-level aggregate
provenance evidence**:

```
actual_ob_live_exposure_applied
actual_ob_live_exposed_count
actual_ob_live_exposure_estimated_tokens
actual_ob_live_exposure_byte_delta
```

The receipt is read **at most once** per request: `observe_actual_ob_live_exposure`
is computed exactly once at the orchestration boundary. When the body did **not**
change (`pre_live_body == final_body`) no receipt is read at all; when the body
**did** change the single read is required and must re-validate against the exact
pre-live / final bytes.

If a body changed but the receipt is missing or fails re-validation, the writer
**refuses** the artifact rather than guess (`live exposure receipt mismatch`). A
memory must never reach the model without the system being able to account for it.

## 6. Controlled-contract coverage

Coverage is a **closed** mapping from the Delivery Plan row
`(source, immediate_delivery_mode)` to exactly one coverage value and one
replacement requirement. Anything outside the closed mapping invalidates the whole
artifact:

| source | delivery mode | `authority_coverage` | `replacement_requirement` |
| --- | --- | --- | --- |
| `frontend` | `direct_data` | `gap_frontend_direct_bypass` | `controlled_frontend_channel_required` |
| `ob` | `flash_cue` | `covered_ob_controlled_cue_contract` | `none` |
| `ob` | `drop` | `covered_ob_controlled_withhold_contract` | `none` |
| `ob` | `unknown` | `gap_ob_policy_unknown` | `policy_resolution_required` |

At the request level the closed status enum is `blocked` / `covered`:
`blocked` iff `contract_gap_unit_count > 0`, else `covered`.

Migration rows mirror only the **already-persisted** Delivery rows, so the full
scope is pinned exactly to the source Delivery truncation telemetry:

```
metrics.unit_count == len(rows) + source_rows_truncated_count
```

with `source_rows_truncated == false` requiring
`source_rows_truncated_count == 0` and `metrics.unit_count == len(rows)`, so
internally legal aggregate algebra can never invent arbitrary hidden rows beyond
the source Delivery truncation counters.

## 7. Frontend direct bypass gap

```
frontend unit -> direct_data -> gap_frontend_direct_bypass
```

Every Frontend unit is, today, a **direct bypass** of unified control: its full
contents are already in the request. It is recorded as a **baseline observation**,
not an endorsement, and it requires a `controlled_frontend_channel_required` before
the direct mouth could ever be removed.

## 8. OB controlled cue contract

```
ob unit -> flash_cue -> covered_ob_controlled_cue_contract  (replacement: none)
```

An OB unit whose existing channel is a bounded `flash_cue` **already** has a
controller-owned delivery decision. It is `covered`; no replacement requirement is
recorded. This says nothing about whether the cue was noticed or used (§11).

## 9. OB controlled withhold contract

```
ob unit -> drop -> covered_ob_controlled_withhold_contract  (replacement: none)
```

`drop` is a **valid controlled outcome**, not a failure. The controller decided
this unit should not be delivered; that is a decision, so it is `covered`.

## 10. OB policy unknown gap

```
ob unit -> unknown -> gap_ob_policy_unknown  (replacement: policy_resolution_required)
```

`unknown` is a **gap**, never a silent `drop`. An unbound policy yields `unknown`
and contributes to `contract_gap_unit_count`. `UNKNOWN != DROP` is preserved
end-to-end.

## 11. Request-level actual OB exposure vs per-row contract coverage

The Live Exposure receipt is **request-level**: it says "this request actually
live-exposed N cues / M estimated tokens". It does **not** identify which delivery
row was exposed. v0 therefore never invents a per-row live-exposure mapping:

- **per-row OB coverage** comes from the Delivery Plan **contract**
  (`flash_cue` → cue coverage, `drop` → withhold coverage, `unknown` → gap);
- the **request-level actual exposure** stays a separate `current_exposure` fact
  and is never projected onto individual rows.

`current_exposure` and `metrics` persist the **same** real exposure facts, so the
authoritative validator requires the two views to agree **exactly** (Frontend
direct count / chars / tokens, live exposed count / tokens, and
`actual_model_visible_memory_estimated_tokens == frontend direct tokens + live
exposure tokens`). The mouth booleans are derived from those same facts
(`frontend_direct_mouth_active == frontend_direct_unit_count > 0`,
`ob_live_cue_mouth_active == actual_ob_live_exposure_applied`,
`dual_mouth_active == both`). A duplicated factual representation that disagrees
invalidates the whole artifact — there is no "one of them is authoritative" rule.

## 12. Direct-mouth removal impact semantics

The impact of removing the Frontend direct mouth is expressed as the **current
Frontend model-visible footprint**:

```
counterfactual_loss_if_direct_mouth_removed_without_replacement_chars
counterfactual_loss_if_direct_mouth_removed_without_replacement_estimated_tokens
```

This is a **loss if removed** number — the exposure that would disappear with no
replacement — **not** a savings claim.

## 13. Why "loss if removed" != "tokens saved"

To "save" those tokens the system would have to remove the Frontend direct mouth
**and** deliver equivalent value another way. That replacement channel does not
exist (questions A–D are open). Reporting the counterfactual as "tokens saved"
would be false: it is the amount that would be **lost** if the mouth were removed
today. The field name says `loss`, and the metric is deliberately derived from the
**current** Frontend footprint, not from any imagined future cue.

## 14. Why `live_cutover_ready` is always `false` in v0

`live_cutover_ready` is **always** `false`. `live_cutover_reason` distinguishes two
situations:

- `contract_gaps_present` — this request still has coverage gaps
  (`contract_gap_unit_count > 0`: Frontend bypass and/or OB `unknown`);
- `unified_live_renderer_not_implemented` — gaps are gone, but there is still **no**
  unified live renderer / selector that decides the actual injected body with the
  content-integrity guarantees the existing Real Injection path has.

Even a request with zero gaps is **not** cutover-ready, because the renderer does
not exist. There is **no** live-ready state in v0, and the reporter must never
print `safe_to_cut_over=true`.

## 15. Concurrency / request-local authority

The observer consumes **THIS request's** already-built
`workspace.memory_source_evidence` and `workspace.memory_delivery_plan`. It never
rebuilds either, never reads a conversation-global "latest" Unified / Delivery
Plan file as runtime authority, and never re-runs retrieval / embedding /
reranking / Surfacing Policy / Lifecycle / Flash.

Because both inputs are request-local, two interleaved requests on the **same**
conversation cannot contaminate each other's evidence: request A's artifact is
built from A's own snapshot / plan even if B writes its Delivery Plan to disk in
between. Request-local evidence is the runtime authority; conversation-scoped
files remain observability.

### 15.1 Evidence readiness gate (AVAILABLE + zero units ≠ UNAVAILABLE)

The Source Evidence and Delivery contracts explicitly distinguish:

```
AVAILABLE + zero units   ->  valid EMPTY evidence
UNAVAILABLE              ->  NOT observable
```

Migration readiness preserves that distinction. For a STORED artifact, the
request-local Source Evidence Snapshot **and** the request-local Delivery Plan
must each independently report BOTH source planes available
(`frontend_observation_available is True` **and** `ob_observation_available is
True`), and the two availability views must **agree**. If either plane is
unavailable, or the availability bits disagree:

```
stored = false
decision = no_record
reason = source_observation_unavailable
```

`UNAVAILABLE` never silently becomes `contract_gap_unit_count = 0` /
`contract_coverage_status = covered`. An available source with zero units stays
valid empty evidence. No new "unavailable migration gap" row enum is invented:
if a whole source plane cannot be observed truthfully, v0 persists **no readiness
conclusion**.

Request identity is not semantic identity, so the Delivery Plan must also
describe exactly the same source occurrence pool (per-source split and full
total) as the Source Evidence Snapshot:

```
plan.metrics.frontend_unit_count == #SOURCE_FRONTEND units in snapshot
plan.metrics.ob_unit_count       == #SOURCE_OB units in snapshot
plan.metrics.unit_count          == len(snapshot.units)
```

On any disagreement:

```
stored = false
decision = no_record
reason = source_delivery_evidence_mismatch
```

## 16. Privacy

The persisted artifact and every log line carry only:

- counts, booleans, enums, revisions and `sha256` digests;
- the request identity (`conversation_id`, `cognitive_request_id`) and the source
  revision, needed for binding;
- opaque `evidence_ref` values already produced by the Delivery Plan.

No memory text, cue, attachment body, memory id, memref or candidate fingerprint
is ever emitted or persisted. `current_exposure` stores only body **sha256** and
byte lengths, never body content.

Persisted Migration rows also carry a privacy-safe `content_digest` (`sha256`
only) **for one reason only**: the authoritative validator can recompute the row's
occurrence `evidence_ref` from the row's own fields with the **existing** shared
Delivery helper (`evidence_ref_for_fields`), so a random valid-looking 64-hex
value can never become trusted occurrence provenance. It is **not** a semantic
identity claim and it is **not** raw content. The row's derived coverage /
replacement labels are also closed against the upstream Delivery row contract, so
a copied row can never be weakened into a shape the Delivery Plan could not itself
have produced.

## 17. Reporter

`src/ombrebrain/maintenance/memory_authority_migration_report.py` scans the
request-scoped, privacy-safe artifacts and prints an **aggregate-only** report. It
is a maintenance / research tool, **not** a runtime authority and **not** an
architecture decider:

- it creates no database, no rolling global aggregate and no runtime accumulator;
- it writes, creates, moves and deletes **nothing**;
- it validates each artifact with the **same** authoritative v0 validator the
  writer uses; an invalid or unsupported-version artifact is counted as `invalid`
  and contributes **zero** statistics (reject-by-whole, no partial row salvage);
- reads are bounded by `--limit` (default `500`; a negative limit is clamped to
  `0`, never "scan everything");
- it imports no server / MCP / `BucketManager` / `DecayEngine` / `EmbeddingEngine`.

It reports current exposure, control coverage and migration gaps. It never emits a
recommendation such as "remove Frontend now", "enable unified authority" or
"migrate these memories". Human review decides architecture.

Example (synthetic fixture only; never run over real user memory state for
validation):

```
PYTHONPATH=src python -m ombrebrain.maintenance.memory_authority_migration_report \
  --root <temporary-test-root> --limit 100 --json
```

## 18. What evidence would justify the NEXT phase

v0 is observation only. It would justify considering a **future** phase only when
the aggregate evidence shows, over real traffic:

- the Frontend direct bypass is measured and understood at attachment granularity;
- the OB `unknown` policy gap (`ob_policy_unknown_count`) trends to zero, so most
  OB units have an explicit controlled cue / withhold decision;
- `dual_mouth_active` requests are characterized;
- the counterfactual loss of removing the direct mouth is quantified.

Even then, a next phase would still require the **open questions A–H** to be
resolved first (§19), plus a unified live renderer with the existing
content-integrity checks and a rollout / kill-switch design. v0 itself authorizes
**nothing**.

## 19. Open questions A–H remain open

This experiment **does not** answer any of them. From
[`UNIFIED_MEMORY_DELIVERY.md`](UNIFIED_MEMORY_DELIVERY.md):

- **A.** When should Frontend `direct_data` become `flash_cue`? — OPEN.
- **B.** Should some Frontend bundle be narrow trusted Direct Data? — OPEN.
- **C.** How will Frontend obtain stable item identity, if ever? — OPEN.
- **D.** Should Frontend get explicit Recall semantics? — OPEN.
- **E.** How does Cognitive Lifecycle constrain the unified Delivery Plan? — OPEN.
- **F.** How do source priorities / conflicts work? — OPEN.
- **G.** How should semantic overlap be handled? — OPEN.
- **H.** When can the old direct Frontend mouth be removed? — OPEN.

`migration readiness evidence exists` — **not** `unified Memory authority is live`,
**not** `Frontend direct mouth removed`, **not** `cutover approved`.

## 20. Follow-on experiment — Frontend Controlled Direct Channel → Unified Memory Authority Compatibility Candidate

A later, separate experiment builds **directly on top of this Migration Shadow**
without changing any of its semantics:

```
Memory Authority Migration Shadow
    ↓
Frontend Controlled Direct Surface
    ↓
Unified Memory Authority Compatibility Candidate Shadow
```

This Migration Shadow records the CURRENT Frontend path as a **gap**:
`frontend + direct_data -> gap_frontend_direct_bypass` with
`replacement_requirement = controlled_frontend_channel_required`. That observation
is **historical fact and stays untouched** — this experiment never mutates
`gap_frontend_direct_bypass` inside a Migration artifact.

The follow-on Candidate reads THIS request's already-validated Migration artifact
(and its Delivery Plan / Source Evidence) and observes that the **same** Frontend
occurrence can be **adopted** into an explicit, controller-owned
`preserve_direct_data` compatibility lane:

```
Migrated: frontend + direct_data -> gap_frontend_direct_bypass
Candidate: the SAME occurrence -> covered_frontend_controlled_direct_candidate
```

It answers one bounded question only — *"could unified control explicitly adopt
the CURRENT Frontend direct exposure WITHOUT changing a single byte the model
sees?"* — and proves the candidate body is byte-for-byte identical to the actual
`final_body`. It is a **ZERO-BEHAVIOUR-CHANGE compatibility migration lane**, not
the live unification: `live_authority` / `candidate_applied` / `live_cutover_ready`
are always `false`, and `preserve_direct_data` does **not** mean Frontend
`direct_data` is the final policy.

OB handling is **delegation, not re-implementation**: the Candidate maps this
Migration Shadow's validated full-scope OB metrics onto the existing OB control
plane (`ob_control_strategy = delegate_existing_ob_control_plane`) and keeps OB
`unknown` as a gap. See
[`FRONTEND_CONTROLLED_DIRECT_CHANNEL.md`](FRONTEND_CONTROLLED_DIRECT_CHANNEL.md)
and `ARCHITECTURE_STATUS.md` §2.28.

---

## Explicit non-goals of this experiment

Not implemented, and must not be read into this work:

- no hypothetical target body (`frontend_removed_body` / `unified_memory_body` /
  `target_body`);
- no Frontend removal / gating / rewriting / deletion; no Frontend renderer,
  memref, Recall or stable item identity;
- no live Frontend control channel; the Frontend direct mouth stays live and
  untouched;
- no per-row projection of the request-level Live Exposure receipt;
- no semantic dedup, embedding overlap, LLM judge, cross-source scoring or
  reranking;
- no retrieval / memory mutation (no touch / reinforce / archive / delete /
  restore / record_usage / lifecycle write);
- no Lifecycle composition; no lifecycle migration;
- no live unified renderer / selector; no provider-body mutation;
- no resolution of open questions A–H.
