# RECON2.md — read-only recon for the Devpost writeup (follow-up)

Facts only, gathered by reading source files, git history, and running
`pytest --collect-only`, `git show`, `gh pr list` against the current
repo state. No code changes, no commits made while producing this
document.

**HEAD at time of writing:** `d34b99b` — "README: architecture, results,
limitations, contributors TODO" (Terry Yeo). `git pull origin main`
reported "Already up to date."

---

## 1. The autonomy story

**Does `controller/controller.py` drive a real loop end to end?** Yes.
`Controller.run()` (`controller/controller.py:248`) walks `STAGE_ORDER`
(`controller/state.py:86`: `INIT, REPRODUCE_BASELINE, STAGE_1_STRUCTURAL,
STAGE_2_COMBINE, STAGE_3_TUNE, FINALIZE`), calling `_run_stage` for each,
then `_finalize`. It is not a stub — it emits `RUN_START`, per-stage
`STAGE_CHANGE`, `HYPOTHESIS`/`EVAL_START`/`EVAL_RESULT`/`DECISION`/
`CONVERGENCE_CHECK` per candidate, and terminal `FINALIZE`/`RUN_END`
events, with budget-exhaustion and illegal-stage-transition guards.

`scripts/run_controller.py` is the driver/CLI around it, not the loop
itself: it parses arguments, builds `Controller(executor=..., gate=...,
generator=..., realizer=..., policy=..., journal=..., seeds=..., ...)`
(`scripts/run_controller.py:242-249`) and calls `.run()`.

**Is there an LLM anywhere in the loop?** No. `scripts/run_controller.py`
wires `generator=SlotScriptedGenerator()` and `realizer=MovesRealizer()`
(both from `autonomy/adapters.py`). `SlotScriptedGenerator`
(`autonomy/adapters.py:194`) is an adapter around
`methods.scripted.ScriptedGenerator`'s fixed ten-move table — its own
docstring: "GeneratorPort over the scripted moves, served BY SLOT."
`MovesRealizer` (`autonomy/adapters.py:259`) is, in its own words, "A
LOOKUP, NOT AN INFERENCE" — it returns the `SlotConfig` authored next to
each hypothesis in `methods/scripted.py`, not anything computed or
generated. A repo-wide search (including `vendor/`) for
`anthropic|openai|api_key|ChatCompletion|messages.create` returns no
matches in `autonomy/` or `controller/` (or anywhere else). **It is still
`ScriptedGenerator` under the hood — no LLM call exists in this loop.**

**`autonomy/INTERVENTION_POLICY.md`** exists
(`autonomy/INTERVENTION_POLICY.md`). It defines an intervention as
exactly three things, all requiring a human to have changed what the
running system does:
- `code_changed_midrun` — `code_fingerprint` (a sha256 over
  `executor/`, `harness/`, `controller/`, `methods/`, `autonomy/` — not
  `tests/` or `scripts/`) drifted between checks.
- `manual_restart` — a prior journal didn't end in `RUN_END`, and either
  no `--resume` was passed or the fingerprint changed.
- `unknown_prior` — a prior interrupted run recorded no launch
  fingerprint at all, so it can't be verified (counts against the run on
  purpose: "the tie breaks against us").

Explicitly **NOT** counted: watching/inspecting (reading logs, `git
status`, a read-only debugger), editing files the run doesn't depend on
(tests, README, notes), or autonomous crash-recovery (`--resume` over a
byte-identical fingerprint — classified in `RUN_START` metadata but not
counted as an intervention).

The counter itself: `executor/report.py` counts every `EventKind.
INTERVENTION` event regardless of `type` — the policy document's stated
job is making sure only genuine manual touches ever produce one. A
`VERIFIED AUTONOMOUS` badge additionally requires: known-clean working
tree at launch, a fingerprint that never moved, zero interventions, and
at least `min_candidates` (default 3) node-boundary integrity checks.

**Has a full Controller run been done?** One journal exists:
`artifacts/journal_controller.jsonl` (28 events, timestamped Aug 31
17:35, produced by `python scripts/run_controller.py
--max-nodes-per-stage 1 --seeds 0` — a minimal single-seed,
one-candidate-per-stage smoke run, not a comprehensive search). From
that journal directly:

| Fact | Value |
|---|---|
| Nodes | 4 (`max node` across all events) |
| Iterations used | 1 |
| Final/incumbent config_id | `406dfba347b2` |
| Final/incumbent primary | 0.6014687563529677 |
| Stop reason | `stages_complete` |
| Total wall_seconds (sum of EVAL_RESULT) | 491.2060728003271 |
| Intervention count | 0 |
| Errors | 1 (`contract` — the `objective/multitask_bce` move, unimplemented) |

`artifacts/report_controller/` holds a render of this same journal
(`results.md`, `iterations.md`, `forecast_calibration.md`,
`trajectory.csv`) — this was produced by rendering the journal above
with `executor/report.py`, not by a separate run. No larger or later
`journal_controller*.jsonl` file exists anywhere in `artifacts/`.

---

## 2. The manual ceiling

`manual/` contains `__init__.py`, `_vendor.py`, `encode.py`, `report.py`,
`run.py` (unchanged since the last recon — see `git log --oneline --
manual/`: still exactly `d22b64b` and `bb45e16`, no new commits). Per
`manual/__init__.py`'s own docstring, yes, it is the hand-built reference
pipeline: *"The MANUAL CEILING: a hand-built, best-effort pipeline whose
only job is to measure how much headroom exists above the organizers'
published FM baseline... It is NOT part of the agent."*

**Has it been run?** No prediction cache exists: `artifacts/preds/`
contains zero files matching `manual_baseline_fm_k16` or
`manual_crosses_v1` (the two config_ids `manual/run.py` defines). No
journal, log file, or results file anywhere in `artifacts/` references
either config_id. The only recorded number anywhere is in the commit
message of `bb45e16` — a **gate verdict**, not raw metrics: "MEASURED (3
seeds, gate CONFIRM): delta +0.00012, ci95 (-0.00091, +0.00112),
backtest_delta -0.00027, accept=False, reason=ci_includes_zero."

**Plainly: no raw validation GAUC / nDCG@5 / primary for either the
unit-1 baseline or the unit-2 crosses variant exists anywhere in this
repo's current state.** Not estimated here.

---

## 3. What the best result is now

The 3-seed rank-average ensemble (`config_id=ens_rank3`) is still the
best/accepted candidate. `artifacts/report/results.md` currently reads,
verbatim, in full:

```
# Results

| Metric | Value |
|---|---|
| Validation-best GAUC | 0.6683 |
| Validation-best nDCG@5 | 0.5363 |
| Validation-best primary | 0.6023 |
| Delta vs official baseline GAUC (0.6674) | +0.0009 |
| Delta vs official baseline nDCG@5 (0.5357) | +0.0006 |
| Delta vs official baseline primary (0.6016) | +0.0007 |
| Iterations used | 2 / 50 |
| Total agent wall-clock (s) | 21.7 |
| Total tokens | 0 (no LLM in the loop yet) |
| Manual interventions | 0 |
| Total training wall-clock (s), measured | 1935.8 |

_"Total agent wall-clock" above sums each EVAL_RESULT's own wall_seconds — it covers only the time this render's run spent re-evaluating, which is ~0s whenever candidates are rebuilt from cache rather than trained. Actual training wall-clock is reported in the row above, passed in separately by the caller._

## Convergence

- Candidates decided: 7
- Baseline adopted (not an improvement): 1
- Accepted as improvements: 1
- Cleared epsilon=0.002: 0
- Consequence: 1 candidate(s) were accepted as statistically real improvements, but none cleared epsilon=0.002. Under the organizers' N=3 no-improvement rule, this run would still be judged as stalled despite the real gain(s).

4 candidate(s) failed (error classes: contract); the run continued past all 4 of them and reached FINALIZE.
```

Note: this file does not carry a "Total GPU-seconds" row and still says
"Total tokens | 0 (no LLM in the loop yet)" as a literal string rather
than a computed value — the `report.py` fix that computes both from the
journal (and the fix that lets `report.py` render a Controller journal at
all) exists only on the local, unpushed branch
`w1w3/cp932-fix-and-cost-telemetry`, not on `main`. This checked-in
`results.md` predates that fix.

The Controller run (section 1) reached incumbent primary 0.6015 — lower
than the ensemble's 0.6023. No Controller run has found anything better
than the ensemble; no `ens_rank3`-equivalent (or better) config_id
appears in `artifacts/journal_controller.jsonl`.

**`artifacts/submission_test.csv`**: dated Aug 30 18:29 — chronologically
older than `artifacts/journal_run.jsonl` (Aug 31 09:22) and
`artifacts/journal_controller.jsonl` (Aug 31 17:35). It was generated
right after the ensemble's `journal_ensemble_candidate.jsonl` (Aug 30
18:19) via `scripts/make_submission.py`, which trains the same accepted
3-seed ensemble config fresh on the full train split. Neither later run
(the 10-move `run_agent.py` run, nor the Controller smoke run) found a
candidate that beats the ensemble, so although the file predates those
runs in wall-clock time, it does not predate a better candidate — it
still reflects the current best result.

---

## 4. Methods / W4

`methods/` contains only `scripted.py` besides `__init__.py` (unchanged
— confirmed via `ls methods/`). No YAML method library exists: a
repo-wide search for `*.yaml`/`*.yml` (excluding `vendor/`, `.venv/`)
returns nothing, and no `methods/library/` directory exists anywhere.
`methods/scripted.py`'s ten `Citation.library_entry` fields still point
at nonexistent paths under `methods/library/*.yaml` (unchanged from the
prior recon). All ten moves carry a `Citation` (`grep -c
"citation=Citation("` = 10).

**Realizable moves, checked directly against `executor/realize.py`'s
current dispatch** (module docstring, lines 6-9, unchanged): `model.impl
== "fm"`, `data_view.impl in {"full", "recent_window"}`, `weighting.impl
in {"none", "exp_decay"}`, `objective.impl in {"bce", "bpr"}` (bpr not
combined with non-"none" weighting). A repo-wide search for
`multitask_bce`, `duration_debias_cwm`, `lightgbm`, `popularity_blend` in
`executor/realize.py` returns zero matches — **nobody has implemented
moves 4, 6, 7, or 9.** 6 of 10 moves are realizable (1, 2, 3, 5, 8, 10 —
5 and 10 both use the already-implemented `model/fm` impl with different
hyperparameters); 4 are not (4, 6, 7, 9), matching the CONTRACT error
recorded in the Controller smoke run's journal (section 1).

---

## 5. Tooling facts

**`requirements.txt`**, exact contents:
```
numpy
pandas
scipy
pytest
```

**External APIs**: none found. A repo-wide search (including `vendor/`)
for `requests\.(get|post)|urllib\.request|http\.client|socket\.|httpx|
aiohttp|urlopen` returns zero matches. Earlier search for
`anthropic|openai|api_key|ChatCompletion|messages\.create` also returns
zero matches anywhere in the repo.

**Zero GPU, zero LLM tokens in the scored run — confirmed, not
corrected.** `contracts.CandidateResult` carries `gpu_seconds`,
`tokens_in`, `tokens_out` fields (defaulting to `0.0`/`0`/`0`), but
neither `executor/run.py`'s `run_candidate` nor `autonomy/adapters.py`'s
real-executor port ever sets them to anything else (`grep -n
"gpu_seconds\|tokens_in\|tokens_out" autonomy/adapters.py` returns no
matches). `artifacts/report/results.md` (section 3) reports "Total
tokens | 0" and the Controller journal's own `EVAL_RESULT` payloads carry
`gpu_seconds: 0.0, tokens: 0` throughout.

**Current test count**: **555** (`pytest --collect-only -q` on current
`main`), by file:

| File | Count |
|---|---|
| tests/test_controller.py | 109 |
| tests/test_controller_fakes.py | 69 |
| tests/test_policy.py | 51 |
| tests/test_manual.py | 47 |
| tests/test_autonomy_integrity.py | 37 |
| tests/test_contracts.py | 29 |
| tests/test_autonomy_adapters.py | 29 |
| tests/test_convergence.py | 26 |
| tests/test_autonomy_render.py | 26 |
| tests/test_run_controller.py | 17 |
| tests/test_realize.py | 17 |
| tests/test_gate.py | 13 |
| tests/test_gate_false_positive_rate.py | 12 |
| tests/test_errors.py | 11 |
| tests/test_report.py | 10 |
| tests/test_false_positive_rate.py | 10 |
| tests/test_scripted.py | 6 |
| tests/test_journal.py | 6 |
| tests/test_cache.py | 6 |
| tests/test_validate.py | 5 |
| tests/test_data.py | 5 |
| tests/test_invariants.py | 4 |
| tests/test_backtest.py | 4 |
| tests/test_rungs.py | 3 |
| tests/test_run.py | 3 |

By package/workstream:

| Workstream | Files | Tests |
|---|---|---|
| `controller/` | test_controller.py, test_controller_fakes.py, test_policy.py, test_convergence.py | 255 |
| `autonomy/` (+ its integration test of the CLI) | test_autonomy_integrity.py, test_autonomy_adapters.py, test_autonomy_render.py, test_run_controller.py | 109 |
| `manual/` | test_manual.py | 47 |
| `harness/` | test_gate.py, test_gate_false_positive_rate.py, test_false_positive_rate.py, test_cache.py, test_validate.py, test_data.py, test_invariants.py, test_backtest.py, test_rungs.py | 62 |
| `executor/` | test_realize.py, test_errors.py, test_report.py, test_journal.py, test_run.py | 47 |
| `methods/` | test_scripted.py | 6 |
| `contracts.py` (shared) | test_contracts.py | 29 |

Note: `test_report.py` (10, not 11) and `test_run.py` (3, not 6) are
lower here than in the unpushed `w1w3/cp932-fix-and-cost-telemetry`
branch — that branch's additional tests aren't on `main` yet (see
section 6).

---

## 6. Unmerged work

**Local branches** (`git branch -a -vv`):

| Branch | Pushed? | Merged into main? |
|---|---|---|
| `main` | yes, up to date with `origin/main` at `d34b99b` | — |
| `w1w3/cp932-fix-and-cost-telemetry` | **no** — local only, no remote tracking branch | no. 3 commits ahead of the point it branched from: `c7f9252` (cp932 fix), `9520156` (telemetry wiring), `f1408d0` (Controller-journal rendering in report.py) |

**Remote branches not in main:**

| Branch | Merged? | Detail |
|---|---|---|
| `origin/w2/contracts-v1` | no | 1 commit ("Define frozen cross-package contracts (v1)", `3b2cd2e`) not reachable from `main`; `main` has 32 commits not reachable from it — an early/stale draft, superseded by `contracts.py`'s subsequent direct history on `main`. |

**Pull requests** (`gh pr list --state all`): exactly one PR exists,
**#1** ("Define frozen cross-package contracts (v1)", branch
`w2/contracts-v1`), status **CLOSED** (not merged) as of
2026-08-28T06:43:10Z. No open PRs.
