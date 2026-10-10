# Legacy vs Cognitive Lifecycle Shadow Comparison — v0

**STATUS: WORKING EXPERIMENT / SHADOW OBSERVATION — NOT AN ADR — NOT A FROZEN
SPECIFICATION**

This document describes an **observation-only experiment**. It does not decide
which lifecycle model is correct, it does not freeze a contract, and it does
**not** unify the two lifecycle planes. If this document and the code disagree,
**the code is correct and this document is wrong**.

- Repository: `https://github.com/cnsj-123/vps-`
- Source of truth: `origin/feature/gateway-v2`
- Base commit: `01b5948ab1b14c2a65ddebea31611f2e5321aa74`
  (merge of PR#3, `trae/lifecycle-authority-audit-v0`)
- Companion documents:
  `docs/cognitive-memory/LIFECYCLE_AUTHORITY_MAP.md` (authority matrix) and
  `docs/cognitive-memory/ARCHITECTURE_STATUS.md` (chain reality map).

The single sentence that matters most:

```
Correlation / disagreement in this artifact does NOT prove which lifecycle
policy is better. It only exposes where their current semantics differ.
```

---

## 1. Why the comparison exists

The repository carries **two parallel lifecycle vocabularies** that share the
word "lifecycle" but are different subsystems (see `LIFECYCLE_AUTHORITY_MAP.md`
§0). They are not silently bridged and they are not unified. Before any decision
about which plane should win (open question **OA-2**), we need to know **where
their current behaviour actually differs** on real request-local retrieval
candidates.

This experiment records exactly that: for the same real candidates, what the
**existing Legacy OB policy** would do versus what the **existing Cognitive
policy** would do — without executing either, without mutating anything, and
without changing live behaviour. It is a measurement, not a migration.

---

## 2. Observation unit

One **request-local retrieval candidate**. The comparison covers the bounded real
retrieval candidate set available in THIS request (`MAX_COMPARISON_ROWS = 64`),
not only the memories that ultimately become a Memory Flash cue.

Each retained candidate is bound by:

- `candidate_fingerprint` (`sha256(normalized memory_id + "\0" + normalized
  content)`, from `retrieval_decision_shadow.candidate_fingerprint`), **plus**
- canonical retrieval occurrence / order (an ordered multiset / FIFO queue per
  fingerprint, so duplicate fingerprints are matched deterministically and are
  **never** silently merged or deduplicated), **plus**
- `retrieval_index`.

A memory id alone is **not** used to align the planes: the same id may appear
twice. If no truthful binding to this request's Unified snapshot can be
established, the candidate is recorded with category `binding_unavailable` and a
**null** binding quadruple — the comparison never guesses. Likewise, if the Base
Surfacing Policy cannot be validly bound to this request (see §8.1), base
eligibility is recorded as **UNKNOWN** (`null`), never as "not eligible".

---

## 3. Legacy semantics (what is measured, not executed)

The Legacy side answers: *"what would the existing DecayEngine do?"* — using the
**real** runtime engine's side-effect-free preview API,
`DecayEngine.preview_decay_cycle_decision(metadata)`, which shares the exact same
predicates and score math as the live `run_decay_cycle()` (no copied second
formula).

Per candidate it records:

- `available` / `applicable` / `skip_reason`
- `current_resolved`
- `auto_resolve_candidate`
- `score_current`
- `score_if_auto_resolve_succeeds`
- `threshold`
- `archive_candidate_current`
- `archive_candidate_after_auto_resolve_if_successful`
- `nominal_archive_candidate`

This is a **preview**. No resolve, archive, touch, delete or activation write
ever happens. The engine is bound narrowly at server startup
(`bind_legacy_lifecycle_comparison_engine(decay_engine)`); the comparison never
starts, stops, wraps or constructs a DecayEngine.

---

## 4. Cognitive semantics (real formulas, reused)

The Cognitive side reuses the **real** lifecycle formulas and the **same**
supplementary evaluator the live gate uses
(`memory_surfacing_lifecycle.evaluate_lifecycle_surfacing_candidate`), so the
`accessibility >= x OR salience >= y` decision lives in **one** place.

Per candidate it records:

- `available`, `observed` (tri-state, see §5)
- `use_count`, `strength`, `accessibility`, `salience`,
  `age_since_last_used_hours`
- `min_accessibility`, `min_salience`
- `would_suppress_if_supplementary`, `would_keep_if_supplementary`
- `counterfactual_gate_decision` (what the gate policy *would* decide)
- `effective_live_gate_decision` (the actual current gate-path participation)
- `invalid_event_count`

If valid `used` events exist, values come from the real derived-state helpers
(`strength_for_use_count`, `accessibility_for_age`, `salience_for_age`,
`lifecycle_use_facts`, `read_lifecycle_events`). The comparison writes **no**
lifecycle event, state snapshot, reinforcement receipt or `used` evidence.

---

## 5. Cognitive tri-state — unobserved != evidence unavailable

```
unobserved != forgotten        (neutral, not suppressed)
unobserved != unavailable      (neutral is not a failed read)
```

The Cognitive side is **tri-state** and the two non-observed states are kept
**strictly separate**:

```
available=true,  observed=true   valid Used history was observed
available=true,  observed=false  evidence read OK, no valid Used history
                                 => neutral_unobserved
available=false, observed=null   evidence could not be evaluated
                                 => cognitive_evidence_unavailable
```

An **unavailable** result is **never** represented as `observed=false`. When the
cognitive evidence is unavailable, `strength` / `accessibility` / `salience` /
`age_since_last_used_hours` are all `null` and
`would_suppress_if_supplementary` is `false` (never fabricated).

A memory with **no valid** `used` lifecycle event is **neutral**. The comparison
never fabricates `strength = 0`, `accessibility = 0`, `forgotten = true` or
`would_suppress = true` for it. Invalid lifecycle event files increment
`invalid_event_count` and **fail neutral** — they never turn a memory into
"suppressed". Cold start therefore never looks like a disagreement.

The request metrics and the aggregate reporter agree, and the three
representation counts partition the rows for which a Cognitive evaluation was
attempted and represented:

```
cognitive_observed_count
  + cognitive_neutral_unobserved_count
  + cognitive_evidence_unavailable_count
  == cognitive_represented_count
```

- `cognitive_neutral_unobserved_count` counts **only**
  `available=true AND observed=false`. It must **not** count
  `cognitive_evidence_unavailable`, `binding_unavailable`, malformed rows or a
  failed evaluator.
- `cognitive_evidence_unavailable_count` counts `available=false`.
- `binding_unavailable` rows (no usable identity) are the **documented
  exclusion**: they are counted zero times in all three, so an unknown identity
  never inflates `evidence_unavailable` either.

---

## 6. Auto-resolve counterfactual caveat

`score_if_auto_resolve_succeeds` and
`archive_candidate_after_auto_resolve_if_successful` are **counterfactuals**.
The live decay cycle performs auto-resolve **before** it evaluates archive, so the
comparison must model that ordering to be honest. But the auto-resolve canonical
write is **not** executed. These fields describe what *would* happen if the normal
auto-resolve write succeeded; they are never an observed state.

---

## 7. Archive counterfactual caveat

The comparison does **not** archive anything. Legacy archive is represented by:

- `archive_candidate_current` — archive decision with the candidate's current
  state, and
- `nominal_archive_candidate` — the archive decision **assuming the normal
  auto-resolve canonical write succeeds first**.

`nominal_archive_candidate` is a **naming-precise** term: it is a nominal result,
not a performed canonical write. It must not be read as evidence that a memory was
or will be archived.

---

## 8. Request-local binding (strict)

The artifact is **request-scoped** and bound to `conversation_id`,
`cognitive_request_id` and `source_unified_revision`. It consumes **this
request's** `memory_shadow_snapshot`. It must **never** re-read the
conversation-global file `unified_context_candidate/{conversation_id}.json` to
reconstruct current-request authority, because a concurrent request on the same
conversation may overwrite that file at any time (see `ARCHITECTURE_STATUS.md`
§0.2).

The request-local Unified binding is **strict**: the snapshot's Unified must
carry the **same** `version`, the **same** `conversation_id` **and** the
**same** `revision` as the request. A foreign Unified — **even one carrying the
same numeric revision** — is never request-local authority. In that case the
observer writes **nothing** (`reason = unified_binding_invalid`) and **never**
falls back to a shared latest file or a global policy file. This failure is
fail-open for the **live request** and fail-closed for **evidence** only.

### 8.1 Base Surfacing Policy binding is strict too

The Base Surfacing Policy passed to the comparison is validated the same way. If
it is missing, malformed, from another conversation, bound to another Unified
revision, or its `decision` / `eligible` shape is not a valid policy shape, then
`base_policy_available = false` and base eligibility is **UNKNOWN**:

- `binding.base_surfacing_eligible = null` and `binding.base_eligible_rank = null`;
- the counterfactual decision is `base_policy_unavailable` (never
  `not_base_eligible`).

A **missing** policy is therefore **not** the same as a policy that evaluated and
decided `no_surface`. The closed v1 decision enum is `allow_shadow` /
`no_surface`, and each decision has an exact eligible shape:

- `allow_shadow` **must** carry an `eligible` list;
- `no_surface` **must** carry an `eligible` list that is **empty** (the real v1
  report always owns an explicit `"eligible": []` from construction onward).

A `no_surface` whose `eligible` field is missing, `null`, or non-empty is
malformed / contradictory evidence, so it is **UNKNOWN**
(`base_policy_available = false`), never "not eligible" and never silently
rewritten to an empty list. Only a genuinely bound `no_surface` policy means
"not eligible" (`base_policy_available = true`,
`base_surfacing_eligible = false`, `base_eligible_rank = null`,
`counterfactual_gate_decision = not_base_eligible`).

A conversation-global policy file is **never** consulted as a substitute.

Storage layout:

```
<state_root>/
  memory_lifecycle_comparison/
    <conversation_id>/
      <cognitive_request_id>.json
```

There is deliberately **no** conversation-global "latest comparison" file.

---

## 9. Persisted privacy model

**Persisted** (privacy-safe):

- request binding ids already used by other request-scoped Context artifacts
  (`conversation_id`, `cognitive_request_id`)
- revision
- counts, enums, booleans
- finite numeric scores / thresholds
- `candidate_fingerprint` (a digest)
- `memory_key(memory_id)` (a cryptographic, non-reversible hashed identity)

**Never persisted**: raw memory text, memory `name`, query text, user text, cue,
`memref`, `useref`, or raw memory id. Non-finite numbers (`NaN` / `Infinity`) are
rejected to `null`.

**Log lines are stricter**: they carry only aggregate counts / reason codes /
revisions / threshold snapshots — **no** `conversation_id`,
`cognitive_request_id`, `memory_key`, `candidate_fingerprint`, or raw memory id.

---

## 10. Comparison categories

Deterministic categories (at minimum):

```
binding_unavailable
legacy_unavailable
legacy_not_applicable
cognitive_evidence_unavailable

legacy_archive_cognitive_unobserved
legacy_archive_cognitive_suppress
legacy_archive_cognitive_keep
legacy_keep_cognitive_unobserved
legacy_keep_cognitive_suppress
legacy_keep_cognitive_keep
```

`cognitive_unobserved` is **its own concept** and is never called
"cognitive_keep" without that qualification. Each candidate also records a
`binding` quadruple (`unified_admitted`, `base_policy_available`,
`base_surfacing_eligible`, `base_eligible_rank`; the eligibility/rank fields are
`null` when binding or the base policy is unavailable) and **two** gate fields
that separate the **counterfactual** from the **actual current participation**:

`counterfactual_gate_decision` — what the current gate policy **would** decide if
applied (computed regardless of the live flag):

```
primary_preserved | neutral_unobserved | would_keep | would_suppress |
not_base_eligible | base_policy_unavailable | binding_unavailable |
cognitive_evidence_unavailable
```

`effective_live_gate_decision` — the actual current gate-path participation. It
proves that the live gate *path* ran, **not** that it changed live eligibility
(the gate is fail-neutral and may return the base policy on an internal runtime
failure), so this field is never presented as a proven live eligibility change.
When the live Lifecycle Surfacing gate is **not** active under its required
conditions, this is `gate_disabled`; when it is, the recorded decision mirrors
the counterfactual decision. The artifact also records the artifact-level
boolean `live_lifecycle_gate_active` — Flash running **AND** a usable
request-local Surfacing policy object **AND** `lifecycle_surfacing_enabled()`
**AND** `lifecycle_shadow_enabled()` — so a counterfactual suppression is
**never** labelled a live effect while the gate is disabled.

The existing Lifecycle Surfacing gate **never suppresses the first base-eligible
candidate**. So a primary candidate may show
`would_suppress_if_supplementary = true` while its counterfactual decision is
`primary_preserved`. That is **not** a contradiction; it is the primary
preservation rule made explicit.

---

## 11. Feature flag

```
OMBRE_GATEWAY_CONTEXT_LIFECYCLE_COMPARISON_SHADOW
```

- Unicode truthy values accepted: `1` / `true` / `yes` / `on` (case-insensitive).
- **Default OFF.** With the flag off, the observer does not run and writes nothing.
- The comparison runs independently of the Memory Flash flag: with Flash **OFF**
  it still observes the base Surfacing Policy over this request's Unified
  snapshot and writes only the comparison artifact.

Version / mode:

```
memory-lifecycle-comparison-shadow.v1
shadow_only
```

---

## 12. Reporter usage

A read-only aggregate reporter summarises many request-scoped artifacts:

```
PYTHONPATH=src python -m ombrebrain.maintenance.lifecycle_comparison_report --limit 100 --json
```

- `--limit N` bounds how many artifacts are scanned (default `500`); a negative
  limit is clamped to `0` (it never becomes "scan everything").
- `--json` prints the aggregate as JSON; without it a human-readable summary is
  printed.
- It writes / creates / moves / deletes **nothing** — not even a malformed
  artifact.
- It emits **only aggregate statistics and finite numbers**: no
  `conversation_id`, `cognitive_request_id`, `memory_key`, `candidate_fingerprint`,
  raw memory id, text, name, query or cue.
- It validates each artifact with the **same authoritative v1 validator** the
  writer uses (`is_valid_lifecycle_comparison_artifact`). An artifact that fails
  validation — wrong version/mode, malformed identity, `NaN` / `Infinity`, an
  inconsistent binding or Cognitive tri-state, an unknown category or gate
  decision, or **one** malformed row among otherwise valid rows — is counted as
  **invalid as a whole** and contributes **zero** to every statistic. A malformed
  row is never silently skipped inside a "valid" artifact.
- Output is deterministically ordered and reports category counts, **separate**
  `counterfactual_gate_decision_counts` and `live_gate_decision_counts`, bounded
  percentile distributions, and the explicit
  `legacy_auto_resolve_changed_archive_outcome_count` (with its `created` /
  `removed` directions) alongside the counts and the representation invariant
  from §5.

The reporter is a maintenance / research tool, not a runtime authority. It creates
no database, no rolling global aggregate and no runtime accumulator.

---

## 13. Interpretation caveats

- The artifact records **where current semantics differ**, not which is better.
- `nominal_archive_candidate`, `score_if_auto_resolve_succeeds` and
  `archive_candidate_after_auto_resolve_if_successful` are **counterfactuals**,
  never observed actions.
- `cognitive_unobserved` is **neutral**, never "forgotten" and never "keep".
- `would_suppress_if_supplementary = true` on a primary candidate is expected and
  does not indicate a bug.
- A high disagreement rate is **not** evidence that either policy is wrong.
- Nothing here observes live traffic unless the flag is explicitly enabled on a
  deployment; enabling it changes no live body and mutates no memory state.

---

## 14. What data may justify a future ArchiveCandidate Shadow

This experiment may produce evidence such as:

- how often a successful Legacy auto-resolve would **change** the archive outcome
  (`legacy_auto_resolve_changed_archive_outcome_count`, with `created` /
  `removed` directions);
- how often a `nominal_archive_candidate` is simultaneously a
  `would_suppress_if_supplementary` candidate (`both_archive_and_suppress_count`);
- how often Legacy would archive while Cognitive has **no** evidence
  (`legacy_archive_cognitive_unobserved`);
- the distributions of Legacy `score_current` versus Cognitive `strength` /
  `accessibility` / `salience`;
- how often the base Surfacing Policy admits a candidate the cognitive
  counterfactual would suppress.

**If and only if** such evidence shows a material, well-understood divergence
might a *separate* ArchiveCandidate Shadow task be justified. This experiment
makes **no** such decision and implements **no** ArchiveCandidate behaviour.

---

## 15. What this experiment CANNOT prove

- It cannot prove Legacy is better, Cognitive is better, or that either should be
  deleted, disabled, or made authoritative.
- It cannot prove that `activation_count` should become `use_count`, that
  `importance` should become `strength`, that cognitive accessibility should
  control archive, or that archive should become proposal-only.
- It cannot prove causation. Disagreement here is **correlation**, not cause.
- It cannot bridge the planes: `trace(reinforce=True) != UseMemory` still holds,
  and no `used` ↔ `touch` bridge is added.
- It cannot observe a memory that retrieval never returned this request.

---

## Explicitly NOT done in this task

- No change to any live lifecycle behaviour; the comparison is default OFF and
  observation-only.
- No change to the decay formula, archive threshold, auto-resolve threshold,
  `activation_count` / `last_active` semantics, Memory Surfacing thresholds,
  retrieval ranking, or Dream ranking.
- No bridging of the Legacy and Cognitive planes; no `UseMemory` → `touch` bridge
  and no `trace(reinforce=True)` → `used` bridge.
- No ArchiveCandidate behaviour, no consolidation, no migration of lifecycle
  authority.
- No mutation of any memory state, and no conversation-global "latest" comparison
  file.

**OA-2 (parallel lifecycle authorities) remains unresolved.** This task adds
*comparison evidence*, not a resolution.