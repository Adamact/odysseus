# Wave 4 effects, provenance, freshness and truthful completion

Branch: `feature/effects-provenance-wave4`.
Exact base: Wave 3 PR #60 head `80a962d96af5f85c785bd517ae6af8e90a8b0d38`
(tree `bba4adfc9ff1628d96daeee57640be46a3f5d270`), clean at admission.
Historical references: foundation `9012e208` (parent `1e3c50d2`),
`wave-4-effects-provenance-foundation.md` and
`wave-4-canonical-refresh-a80c164d.md` in the old worktree (read only).

## Foundation decision: recreated, not cherry-picked

`9012e208` was **not** cherry-picked. Its semantics were sound, but its types
encoded assumptions that final Wave 3 made wrong:

| Historical type | Problem against final Wave 3 | Recreated as |
| --- | --- | --- |
| `resource_keys: tuple[str, ...]` | Opaque string tokens; Wave 3 now has typed exact identities. Strings would make names/paths authority-shaped. | `ResourceRef`, built only by `resource_ref()` from typed Wave 3 objects; anything else is a `TypeError`. |
| `may_have_changed: bool = False` | Defaults to "no impact"; conflates known no-op with unknown. | `Impact.NONE` only with `ExecutionOutcome.NOT_EXECUTED`; everything that reached a backend is `POSSIBLE`. |
| `EffectStatus` (claimed/reported/verified/failed/unknown) | Mixes execution outcome with verification; one FAILED cannot carry "effect done, cleanup failed". | Separate `ExecutionOutcome`, `Impact`, `CleanupState`, and derived `EffectVerdict`. |
| `verification_for` attestation | An adapter label asserted that an observation checked a postcondition. | `predicate_holds()` evaluates the explicit `Postcondition` against the observed state itself. |
| `EvidenceOrigin` (3 labels) | Cannot express coverage, mechanism admission or lifecycle-only facts. | `ObservationMechanism` + `Coverage`; only admitted readback mechanisms can verify, per resource kind. |

Preserved semantics: request ≠ admission ≠ dispatch ≠ execution ≠ verification;
failed and unknown executions may have partially changed state; stale evidence
stays historical and refresh appends; the newest check wins with no fallback to
an earlier complete one; equal positions are rejected; unknown scope invalidates
conservatively; receipts are never invalidated; matching state after unknown
execution is observation, not causation.

## Runtime chain

```
ExactOperation + Wave 3 bound operation (contextvars set by the dispatcher)
  -> mark_dispatch(): durable EffectClaim (fsync) BEFORE execution_id/backend
  -> backend invocation (unchanged producers)
  -> record_action(): EffectOutcome from typed ProducerFacts (before receipt reduction)
  -> admitted reads: Observation of the exact bound resource
  -> EffectHistory: invalidation / freshness / assess()
  -> EvidenceLedger.record_effects() -> existing evaluate() -> CompletionDecision
  -> existing buffered presentation gate (completion_answer)
```

## Contracts (`src/agent_runtime/effects.py`)

- `ResourceRef(kind, role, location, incarnation, snapshot_sha256)`. Location is
  "where" including the sealed root/namespace identity; incarnation is the object
  seen there. Kinds and their Wave 3 sources:
  - filesystem: `FilesystemResource` — root scope/owner/path/device/inode + path;
    incarnation = file/dir device:inode + ancestor-chain digest, or `absent:`.
  - process: `ProcessResource` — namespace/owner/request/thread/PID/**start token**/role.
    PID reuse is a different location.
  - process_launch: `ProcessLaunchResource` — generation (the exact launch→job linkage
    validated by `job_from_record`).
  - background_job: `BackgroundJobResource` — job id + generation.
  - owned: `OwnedResource` — namespace/owner/thread/collection/record; incarnation =
    revision. `*` collection bindings overlap their records.
  - external: `ExternalResource` — namespace/owner/endpoint/server/tool; incarnation.
  - browser_session: `BrowserSessionResource` — owner/thread/session key; incarnation
    = session incarnation. `BrowserPageResource` is refused.
- `EffectClaim`: run/action identity, sequence, `OperationRef` (final normalized
  tool/action/input digest/request), `impact_scope` (empty = unknown), `dependencies`,
  `obligations` (each must target a claimed binding), `parent_run_id`, `external`.
  No status field: a claim is intent, not dispatch.
- `EffectOutcome`: `NOT_EXECUTED | REPORTED_SUCCESS | FAILED | TIMED_OUT | CANCELLED |
  RUNNING | INTERRUPTED` (`ATTEMPTED` is derived for a claim without outcome), `Impact`,
  bounded `ProducerFacts` (exact scalar types only), `CleanupState`, `replayed`.
- `Observation`: exact resource, mechanism, coverage, source action/execution, `exists`,
  complete-content digest. Admitted readbacks require their source action.
- `EffectHistory`: unique positions; RUNNING may be followed by one settled outcome;
  a settled outcome is never replaced.

### Invalidation and freshness

`invalidated_by(observation)` = later claims that may touch it (overlap or unknown
scope; a refused no-op excluded) + later observations of the same location with a
different incarnation (replacement). `freshness()` is STALE, UNSETTLED (an earlier
overlapping effect was still attempted/running at observation time) or FRESH.
Receipts/acknowledgements are never invalidated. Filesystem overlap is
ancestor-or-self within one sealed root identity (listings, parents, rename-style
dependencies); no alias discovery is attempted.

### Verification

`assess(claim)` per obligation uses the newest observation of the target **after
settlement**, through a verifying mechanism for that kind (filesystem read, owned
record read, remote readback). It must be FRESH, and the predicate must be decidable
(partial coverage cannot decide content). Results: VERIFIED only with
`REPORTED_SUCCESS`; STATE_OBSERVED for timed-out/cancelled/interrupted execution
(causality unknown); FAILED execution never becomes success; CONTRADICTED when the
fresh check is false; UNVERIFIED otherwise. Process ownership, job state, browser
session, receipts and acknowledgements can stale evidence but never verify.

## Durable persistence (`src/agent_runtime/effect_log.py`)

- One append-only JSONL file per root run lineage under `DATA_DIR/effects`
  (`0600`, directory `0700`, `O_NOFOLLOW`, `st_nlink == 1` required).
- `claim()` writes and fsyncs before returning; failure raises
  `EffectPersistenceError` (a `ResourceIdentityError`). `mark_dispatch` claims before
  assigning `execution_id`, so the dispatcher returns BLOCKED and the backend is never
  invoked; `dispatched()` closes the un-awaited coroutine.
- Outcomes/observations are appended; a failed non-claim write sets `degraded` (the
  on-disk claim then replays as unknown). Claim-free (read-only) runs create no file.
- `load()` validates every record strictly, tolerates only a torn final line, and
  fails closed on corruption, forged enum values, inconsistent history or aliasing.
  `recover_interrupted()` appends INTERRUPTED/possible-impact outcomes for unsettled
  claims, leaves RUNNING alone, and is idempotent. `open()` returns the live log or the
  recovered durable one.
- `launch-<generation>.json` maps a background launch generation to its claim so a
  later run can settle it.
- The store is a Wave 3 control-plane path (prefix check), so filesystem tools cannot
  read or write it. Existing containment/process/job stores are not reused.

## Adapters (`src/agent_runtime/effect_adapters.py`)

Inputs are only the bound operations live at `mark_dispatch` (filesystem, owned,
process, backend, browser). Classification failure claims unknown scope; it never
blocks dispatch.

| Family | Claim | Observations / settlement | Verification available |
| --- | --- | --- | --- |
| Filesystem write/edit/patch | exact bindings; `write_file` CONTENT_SHA256 of the bytes the producer commits (after fence unwrapping; EXISTS on non-`\n` platforms), `apply_patch` add=CONTENT_SHA256 / delete=ABSENT / update=EXISTS, `edit_file` EXISTS | — | via later admitted `read_file` |
| `read_file` | none (admitted read) | re-reads the exact bound source (identity checked before/after) → COMPLETE digest, or PARTIAL for offset/limit/truncation/structured extraction | decides predicates when COMPLETE |
| `ls`/`glob`/`grep` | none | PARTIAL existence of the search root | existence only |
| bash/python launch | unknown scope + launch generation dependency | outcome from containment envelope: TIMED_OUT (`timed_out`), cleanup from `teardown.dead`, RUNNING for `bg_job_id` | none (process exit is not a postcondition) |
| `manage_bg_jobs` read | none | JOB_STATE observation; settles the RUNNING launch of the exact generation | none |
| `manage_bg_jobs` kill | job + its processes | settles the launch as CANCELLED | none |
| Owned mutation | exact revisioned records (+attachments as dependencies) | — | none (no independent readback contract) |
| Owned reads (`vault_get`, ...) | none | PARTIAL OWNED_RECORD_READ per exact revision | existence only |
| External/MCP | external backend ref, `external=True`; `remote_acknowledged` on exit 0 | none | none: no independent authorized readback exists, so it stays UNVERIFIED |
| Browser `session_info` | none | BROWSER_SESSION lifecycle observation of the session incarnation | none |
| Unbound tools (incl. `manage_tasks`) | unknown scope | — | none |

Producer seams added: `job` lifecycle facts on job reads/kills
(`job_lifecycle_facts`), `timed_out` on containment timeouts, and
`mutation_attempted` when `write_file`/`edit_file` fail after their truncating open.
RUNNING is recognized only from the native launch (`bg_job_id`) or a bridge's
explicit `detached`.

## Completion integration

No second policy. `completion._ledger()` builds the single `EvidenceLedger` used for
the decision, `ask_user` filtering and prose filtering, then calls
`record_effects(entries, action_order, partial_reads)`. Effects change the existing
`evaluate()` only for declared artifacts and artifact prose:

- a fresh contradicting readback of a required artifact → FAILED;
- a required artifact is **unsettled** (BLOCKED, "a later operation may have changed a
  required artifact without settled evidence") when, after its last successful
  mutation, an effect with unresolved impact may have touched it: explicit targets
  with unknown/cancelled/timed-out outcomes or failures after `mutation_attempted`;
  unknown-scope effects that were cancelled/interrupted, still RUNNING, or failed
  teardown. Settled shell changes remain tracked by existing artifact version capture;
- partial `read_file` validation events become non-authoritative;
- `_supports_artifact_claim` applies the same rules, so prose cannot claim the write.

Ordinary conversation and read-only synthesis are unchanged (no claims, no file).
`effect_assessments` are added to terminal metrics metadata.

## Browser, scheduler and background

Browser page/document operations still fail closed before dispatch (verified through
the real dispatcher with effects enabled: no claim, never dispatched). Only
`session_info` produces session lifecycle observations; replacement stales them.

The background monitor, after its existing `job_from_record` + `validate_job`, settles
the exact launch claim from the server-owned record's typed lifecycle facts
(idempotent across retries). The delivered report remains untrusted attributed
content; it is never an observation. Scheduler triggers are unknown-scope claims
whose replies verify nothing; scheduled runs use their own journals/logs.

## Files

Production: `effects.py`, `effect_log.py`, `effect_adapters.py` (new);
`journal.py`, `completion.py`, `agent_evidence.py`, `bg_monitor.py`,
`agent_tools/{filesystem_tools,subprocess_tools,bg_job_tools}.py` (seams);
`resources.py` (effect store added to control-plane paths; strengthening only).
Not changed: `authority.py`, containment, process ownership/reaper, browser
authority, context resolution, runtime selection, agent loop.

Tests: `test_effects_foundation.py` (recreated), `test_effect_journal_persistence.py`,
`test_effect_resource_bindings.py` (real dispatcher), `test_effect_verification_adapters.py`;
`tests/conftest.py` redirects the store to a session tmp directory.

## Residual limitations (none weakens authority or manufactures success)

- **P2 durable integrity:** records carry no MAC. A writer with access to `DATA_DIR`
  outside the tool layer could forge records that a later `load()` accepts — the same
  trust class as the existing job/containment stores.
- **P2 multi-process:** two processes appending to one log could duplicate sequences;
  replay then fails closed (never success). No inter-process lock.
- **P2 unobserved writers:** freshness is relative to recorded history; an external
  change after the last observation is detected only by a new observation.
- **P2 scope of verification:** VERIFIED is reachable only for filesystem effects.
  Owned/external effects have no independent readback contract and stay UNVERIFIED;
  `edit_file` asserts existence only.
- **P2 conservatism:** unbound tools are unknown scope, so cancelling/interrupting
  even a read-only unbound tool, or a RUNNING background job, blocks later-unsettled
  required artifacts until a new successful mutation.
- **P2 replay is lazy:** interrupted claims are recovered when a log is opened (e.g.
  background settlement); there is no startup scan. Unopened claims remain on disk
  as unsettled (assessed PENDING/unknown, never success).
- **P2 retention:** no pruning of effect logs or launch index files.
