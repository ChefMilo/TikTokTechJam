# RECON.md — read-only recon for the Devpost writeup

Facts and file references only. Gathered by reading source files, git
history, and running `pytest --collect-only` and `git show` against
specific commits. No interpretation beyond what a file/commit states
directly.

---

## 1. `manual/`

Six files (`__init__.py`, `_vendor.py`, `encode.py`, `report.py`, `run.py`,
plus `__pycache__/`). Authored by a different contributor than the rest of
this session's work — every `manual/` commit is signed `ChefMilo
<collinteo2003@gmail.com>`, co-authored `Claude Opus 5`.

| File | Lines | What it does |
|---|---|---|
| `manual/__init__.py` | 57 | Package docstring only. States the package's purpose and import boundary (see below). |
| `manual/_vendor.py` | 72 | Loads `vendor/kuairand-starter-kit/baseline.py` by file path (same `importlib.util.spec_from_file_location` pattern as `harness/data.py`/`harness/metrics.py`), exposing `vendor.FM`, `vendor.encode`, `vendor.FIELDS`, `vendor.evaluate`, `vendor.sigmoid`. |
| `manual/encode.py` | 505 | Re-expresses the vendor's `encode()` with the field list as a parameter (`FieldSpec`), so cross-column features can be added without editing the vendored kit. Defines `BASELINE_FIELDS`, `CROSS_SPECS`, `CROSS_USER_COLUMNS`, `CROSS_VIDEO_COLUMNS`, `CrossSpec`, `SideTables`. |
| `manual/report.py` | 109 | `print_candidate_report` (per-seed numbers) and `print_comparison` (hands both `CandidateResult`s to `harness.gate` and prints the returned `Verdict` — does not compute its own delta). `PUBLISHED_BASELINE_PRIMARY = 0.6016`. |
| `manual/run.py` | 460 | The standalone train/predict/score loop. `MANUAL_BASELINE_CONFIG_ID = "manual_baseline_fm_k16"`, `MANUAL_CROSSES_CONFIG_ID = "manual_crosses_v1"`. `VARIANTS = {"baseline": run_baseline, "crosses": run_crosses}` (comment: `"Unit 3 adds \"blend\""` — not present). CLI via `argparse`: `python -m manual.run --variant {baseline,crosses} --seeds 0,1,2 [--compare-to baseline]`. |

**Is it a hand-built reference pipeline (the "manual ceiling")?** Yes —
`manual/__init__.py` lines 1-19 state this explicitly: *"The MANUAL
CEILING: a hand-built, best-effort pipeline whose only job is to measure
how much headroom exists above the organizers' published FM baseline...
It is NOT part of the agent. It never goes through the Controller, it
proposes nothing, it has no ports, and no journal."* It is enforced to
import only `harness/`, `contracts.py`, and the vendored kit — never
`executor/` — and `tests/test_manual.py` parses every module here with an
AST check that fails on an `executor` import.

**Has it been run? Recorded scores?**

- No `manual_baseline_fm_k16` or `manual_crosses_v1` files exist under
  `artifacts/preds/` in this working copy (checked directly — 0 matches).
- No journal file, log file, or results file anywhere in the repo mentions
  either config_id or reports raw GAUC/nDCG@5/primary numbers for either
  variant.
- The **one place a real measurement is recorded** is the commit message
  of `bb45e16` ("manual/: user x item cross variant (unit 2) - REJECTED,
  delta +0.00012, CI includes zero"):
  > MEASURED (3 seeds, gate CONFIRM): delta +0.00012, ci95 (-0.00091,
  > +0.00112), backtest_delta -0.00027, accept=False,
  > reason=ci_includes_zero.
  This is a **gate verdict** (crosses vs. unit-1 baseline) — it is not a
  raw GAUC / nDCG@5 / primary figure for either variant, and no raw figure
  for either variant is recorded anywhere in the current repo state.
- `tests/test_manual.py`'s GAUC/nDCG@5 values (e.g. lines 416-417, 428-429,
  452-459: 0.60, 0.62, 0.64, 0.56, 0.58, 0.61...) are synthetic fixture
  data for unit tests, not measurements from a real run against the
  dataset.

**Plain statement:** the commit history records that unit 2 was run once,
for real, against the real dataset, and rejected by the gate (delta
+0.00012, ci95 includes zero). No raw GAUC/nDCG@5/primary numbers for
either the unit-1 baseline or the unit-2 crosses variant, and no cached
predictions for either, exist anywhere in this working copy today.

---

## 2. `controller/`

| File | Lines | One-line summary |
|---|---|---|
| `controller/__init__.py` | 7 | Package docstring only. |
| `controller/controller.py` | 879 | The `Controller` class (line 144) — a state machine that walks `Stage` order, asks a generator for a hypothesis, a realizer to turn it into slot code, an executor to evaluate it, a gate to compare it to the incumbent, and writes journal events at every step. Contains the circuit breaker (`_maybe_block_slot`) and `_check_convergence`. |
| `controller/convergence.py` | 296 | Pure module: two convergence rules (`RULE_ORGANIZERS`, `RULE_INTERNAL`), `EPSILON = 0.002`, `N_CONSECUTIVE = 3`, `SIGMA = 0.0008`, `is_significant(ci95)`, `assess(committed)`, `flat_streak(committed)`. |
| `controller/fakes.py` | 841 | Test doubles for every port: `FakeExecutor`, `SlotSensitiveExecutor`, `ScriptedGenerator` (controller-local, distinct from `methods.scripted.ScriptedGenerator`), `DisobedientGenerator`, `DeterministicRealizer`, `ScriptedRealizer`, `InMemoryJournal`, `_ConstantGate`/`AlwaysAcceptGate`/`AlwaysRejectGate`, `ScriptedGate`, `DeltaGate`. |
| `controller/policy.py` | 488 | Three real (non-fake) search policies: `UniformPolicy` (line 71), `FixedOrderPolicy` (line 125), `CostAwareBanditPolicy` (line 192). The module docstring (lines 9-19) describes the bandit as "the next PR adds" — the class already exists in the file. |
| `controller/ports.py` | 371 | `Protocol` definitions: `ExecutorPort`, `GatePort`, `PolicyPort`, `GeneratorPort`, `RealizerPort`, `JournalPort`. Also `PortExhausted`, `GeneratorExhausted`, `RealizerExhausted`. |
| `controller/state.py` | 852 | `Stage` enum, `HistoryEntry` (`NamedTuple`, records every attempt — accepted or not — with `config_id, primary, accepted, delta, ci95, significant, target_slot, wall_seconds, gpu_seconds, tokens`), `RunState`, `SlotStats`. `committed_revisions` property (line 501): `tuple(entry for entry in self.history if entry.accepted)`. |

**Runnable entry point?** No. `grep` for `if __name__`, `argparse`, or
`def main(` across `controller/*.py` returns nothing. No file under
`scripts/` references `controller` (checked with `grep -rl controller
scripts/*.py`) or `import controller`. It is library code with a fully
built `Controller` class, exercised only from `tests/` (see below) — no
CLI/driver in this repo instantiates it against the real `harness`/
`executor`/`methods` packages end to end.

**Was the `convergence.py` bug from CONTROLLER_AUDIT.md fixed?** Tracing
the exact mechanism:

- `harness/gate.py`'s `_confirm` only returns `accept=True` when `ci95`
  excludes zero (positive direction) and `backtest_delta > 0`.
- `controller/controller.py:617`: `significant=is_significant(verdict.ci95)`
  is stored on every `HistoryEntry`, accepted or not.
- `controller/state.py:514`: `committed_revisions` filters `self.history`
  to `entry.accepted` only.
- `controller/convergence.py:227-232` (`_internal_rule`):
  `converged = all(r.significant is False for r in window)`.

Since every entry that reaches `committed_revisions` has `accepted=True`,
and `accepted=True` under the real `harness.gate` implies
`is_significant(ci95) == True` (or `None` for the one baseline-adoption
entry, which has no gate ruling), no entry in the window this function
ever sees can have `significant is False`. `_internal_rule` therefore
cannot become `True` when driven by the real `harness.gate`.

The one test that exercises the real gate against a live `Controller`
loop — `tests/test_false_positive_rate.py` (`Controller(gate=harness_gate,
...)`, line ~178) — uses `FakeExecutor` for training and never asserts
`internal_converged is True` for a real-gate run. All existing tests that
assert `internal_converged is True` do so with a synthetic
`CommittedRevision`-shaped double (e.g. `tests/test_convergence.py` lines
52, 105, 118, 218, 262: `Rev(primary=..., significant=False)`) or with
`controller/fakes.py`'s `DeltaGate` (lines 790-829), whose constructor
takes `accept: bool` and `ci95` as independent arguments — it can return
`accept=True` with a `ci95` that straddles zero, which is a combination
`harness.gate.compare` can never produce. `tests/test_controller.py`'s
`test_convergence_check_is_emitted_after_every_commit` (line 880) uses
`DeltaGate(delta=0.0005)` for exactly this reason and asserts `by_rule in
{"organizers", "internal"}` rather than asserting which one fired.

**Does anything in `controller/` import from `harness/`?** No
`import harness` / `from harness import` statement exists inside
`controller/*.py` (checked directly — only prose mentions of "harness" in
comments/docstrings, e.g. `controller/ports.py:173`: *"RESOLVED:
harness/gate.py NOW SATISFIES THIS PORT"*). The wiring happens at the call
site instead: `tests/test_false_positive_rate.py:77` does
`from harness import gate as harness_gate` and passes it to
`Controller(gate=harness_gate, ...)` (line ~178) — `harness.gate` is a
module, and its `compare(candidate, incumbent) -> Verdict` function
satisfies `GatePort` structurally, with no inheritance needed
(`controller/ports.py`'s own stated design: *"Structural, not nominal"*).
This is the only real (non-fake) component wired into a `Controller`
instance anywhere in the repo; the `executor`, `generator`, and `realizer`
arguments in that same test are `FakeExecutor`, `controller.fakes.
ScriptedGenerator`, and `DeterministicRealizer` respectively.

**LLM-backed `GeneratorPort`/`RealizerPort` implementation?** None found.
`grep` for `def realize(self, hypothesis` and `def propose(self,
state_card` across the repo matches only `controller/ports.py` (the
Protocol definitions), `controller/fakes.py` (test doubles), and
`tests/test_controller.py`. A repo-wide search for `anthropic`, `openai`,
`api_key`, `ChatCompletion`, `messages.create` returns no matches.
`methods.scripted.ScriptedGenerator.propose(self, state)` has a different
signature than `GeneratorPort.propose(self, state_card, target_slot)` —
its own module docstring states it "deliberately does NOT implement
controller.ports.GeneratorPort as-is." No concrete, non-fake, non-test
implementation of either Protocol exists anywhere in the repo.

---

## 3. `methods/`

Contents besides `scripted.py`: only `__init__.py` (7 lines, package
docstring) and `__pycache__/`. `methods/__init__.py` states the package
"Will hold: the catalog of candidate recommender methods, prompts used to
propose/mutate methods, and the hypothesis generation logic" — this is
aspirational language, present tense for what exists today is just
`scripted.py`.

**Is there a YAML method library?** No. A repo-wide search for `*.yaml` /
`*.yml` files (excluding `vendor/` and `.venv/`) returns nothing, and no
`methods/library/` directory exists anywhere in the working tree.
`methods/scripted.py`'s ten `Citation.library_entry` fields all point to
paths under `methods/library/*.yaml` (e.g.
`"methods/library/fm.yaml#factorization_machine"`,
`"methods/library/recency_weighting.yaml#exponential_decay"`) — none of
these files exist on disk.

**How many entries, do they carry citations?** Ten entries
(`_MOVES` tuple, `methods/scripted.py`), one per scripted move. All ten
carry a `Citation` (`key`, `url`, `library_entry`) — confirmed by
`grep -c "citation=Citation("` = 10. The `library_entry` values reference
the non-existent YAML library described above.

---

## 4. Test count breakdown

**Current working tree** (`pytest --collect-only -q`, includes 5
uncommitted test functions on top of the last commit — see below):
**451 tests collected**, by file:

| File | Count |
|---|---|
| tests/test_controller.py | 109 |
| tests/test_controller_fakes.py | 69 |
| tests/test_policy.py | 51 |
| tests/test_manual.py | 47 |
| tests/test_contracts.py | 29 |
| tests/test_convergence.py | 26 |
| tests/test_realize.py | 17 |
| tests/test_gate.py | 13 |
| tests/test_report.py | 12 |
| tests/test_gate_false_positive_rate.py | 12 |
| tests/test_errors.py | 11 |
| tests/test_false_positive_rate.py | 10 |
| tests/test_scripted.py | 6 |
| tests/test_run.py | 6 |
| tests/test_journal.py | 6 |
| tests/test_cache.py | 6 |
| tests/test_validate.py | 5 |
| tests/test_data.py | 5 |
| tests/test_invariants.py | 4 |
| tests/test_backtest.py | 4 |
| tests/test_rungs.py | 3 |

**Per package/workstream** (grouping the files above):

| Workstream | Files | Tests |
|---|---|---|
| `controller/` | test_controller.py, test_controller_fakes.py, test_policy.py, test_convergence.py | 255 |
| `manual/` | test_manual.py | 47 |
| `harness/` | test_gate.py, test_gate_false_positive_rate.py, test_false_positive_rate.py, test_cache.py, test_validate.py, test_data.py, test_backtest.py, test_rungs.py, test_invariants.py | 62 |
| `executor/` | test_realize.py, test_report.py, test_errors.py, test_run.py, test_journal.py | 52 |
| `methods/` | test_scripted.py | 6 |
| `contracts.py` (shared) | test_contracts.py | 29 |

**Committed vs. working tree:** the last commit (`bb5ab28`) has **446**
collected tests. The working tree adds 5 uncommitted test functions on
top of that: `test_forecast_calibration_lists_unmeasured_hypotheses_separately`
and `test_forecast_calibration_flags_largest_same_direction_miss` in
`tests/test_report.py`, and `test_run_candidate_second_call_serves_from_cache_and_skips_training`,
`test_run_candidate_force_retrain_bypasses_the_cache`,
`test_run_candidate_cache_hit_marks_eval_result_served_from_cache` in
`tests/test_run.py` — confirmed by diffing `git show HEAD:<file>` against
the working copy. 446 + 5 = 451.

**Which are new since 377?** No commit in this repo's history was found
with exactly 377 collected tests. The nearest verifiable checkpoint,
counted directly with `git show <commit>:<file> | grep -c "^def test_"`
(a test-*function* count — lower than `pytest --collect-only`'s
collected-*item* count at the same commit, because 7 files use
`@pytest.mark.parametrize`, which one function definition can expand into
several collected cases; `test_manual.py` is one example, 46 function
defs vs. 47 collected items at HEAD), is commit `f4318b4` (the last
commit before `manual/` and the false-positive-rate suite were added):
**340 test functions**. The four commits between there and `HEAD` added,
by the same function-counting method:

| Commit | Message (short) | Test functions added |
|---|---|---|
| `d22b64b` | manual/ unit 1 (baseline runner) | +24 (test_manual.py: 0→24) |
| `bb45e16` | manual/ unit 2 (crosses) | +22 (test_manual.py: 24→46) |
| `8f1d87d` | false-positive-rate test | +10 (test_false_positive_rate.py) |
| `899f082` | production false-positive rate | +12 (test_gate_false_positive_rate.py) |
| `bb5ab28` | error taxonomy / run_agent (this session) | +16 (test_errors.py: 11, plus growth in test_journal.py/test_report.py) |

340 → 424 test functions from `f4318b4` to `HEAD` (`bb5ab28`), which
collects as 446 items under pytest; the working tree's 5 further
uncommitted tests bring today's total to 451.

---

## 5. Everything else on `origin/main` from the last 10 commits

```
bb5ab28  Terry Yeo   Ten-move run_agent with error taxonomy and circuit breaker; report fixes; honest acceptance count
899f082  ChefMilo    tests: production false-positive rate - 3/300 (1.0%) on the user-level bootstrap vs 51.7% naive
8f1d87d  ChefMilo    tests: false-positive-rate test - 1/360 acceptances against a null executor through the real gate
bb45e16  ChefMilo    manual/: user x item cross variant (unit 2) - REJECTED, delta +0.00012, CI includes zero
d22b64b  ChefMilo    manual/: standalone ceiling baseline runner + field-parameterized encoder (unit 1)
08e0c32  Terry Yeo   Ensemble candidate ACCEPTED: +0.00083 val, +0.00089 backtest, CI entirely positive; submission_test.csv validated
f4318b4  Terry Yeo   Moves 2/3/8 measured (all rejected); journal + report; ci_entirely_negative label; central int8 fix; ensemble probe shows rank-avg 0.6022
a705d0a  Terry Yeo   test_gate.py: widen the CONFIRM timing budget 5s -> 12s to stop full-suite-load flakes
d9aaf90  Terry Yeo   executor/realize.py + run.py: I1 passing — val primary 0.6014 vs 0.6015 published, user-level bootstrap confirmed live
c064dd3  Terry Yeo   CONTROLLER_AUDIT.md; methods/scripted.py: ten-move deterministic generator for I2
```

Grouped by files touched (workstream inferred from path only):

**executor/ + scripts/ + tests/ (Terry Yeo's sessions, this conversation's history):**
- `bb5ab28`: `executor/errors.py` (new), `executor/journal.py`,
  `executor/report.py`, `executor/run.py`, `scripts/ensemble_probe.py`,
  `scripts/run_agent.py` (new), `tests/test_errors.py` (new),
  `tests/test_journal.py`, `tests/test_report.py`
- `08e0c32`: `executor/realize.py`, `executor/run.py`,
  `harness/validate.py`, `scripts/ensemble_candidate.py` (new),
  `scripts/make_submission.py` (new), `scripts/populate_move1_backtest.py`
  (new)
- `f4318b4`: `executor/journal.py`, `executor/realize.py`,
  `executor/report.py`, `executor/run.py`, `harness/gate.py`,
  `harness/metrics.py`, `methods/scripted.py`,
  `scripts/compare_moves.py` (new), `scripts/ensemble_probe.py` (new),
  `tests/test_gate.py`, `tests/test_journal.py`, `tests/test_realize.py`,
  `tests/test_report.py`, `tests/test_run.py`
- `a705d0a`: `harness/gate.py`, `tests/test_gate.py`
- `d9aaf90`: `EXECUTOR_SURVEY.md` (new), `executor/realize.py`,
  `executor/run.py`, `scripts/i1_smoke.py` (new)
- `c064dd3`: `CONTROLLER_AUDIT.md` (new), `methods/scripted.py` (new),
  `tests/test_scripted.py` (new)

**manual/ (ChefMilo's sessions):**
- `d22b64b`: `manual/__init__.py`, `manual/_vendor.py`,
  `manual/encode.py`, `manual/report.py`, `manual/run.py` (all new),
  `tests/test_manual.py` (new)
- `bb45e16`: `manual/encode.py`, `manual/run.py`, `tests/test_manual.py`

**controller/ + false-positive-rate tests (ChefMilo's sessions):**
- `8f1d87d`: `controller/ports.py` (docstring only, no behaviour change
  per the commit message), `tests/test_false_positive_rate.py` (new)
- `899f082`: `tests/test_false_positive_rate.py`,
  `tests/test_gate_false_positive_rate.py` (new)

No other files (e.g. `contracts.py`, `controller/controller.py`,
`controller/convergence.py`, `controller/fakes.py`, `controller/policy.py`,
`controller/state.py`) were touched in any of the last 10 commits — the
substantial `controller/` implementation (879+296+841+488+371+852 = 3727
lines across 6 files) predates this 10-commit window and was not
inspected for further history beyond what's captured above.
