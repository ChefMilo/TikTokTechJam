# CONTROLLER_AUDIT.md — controller/ vs. harness/HANDOFF.md

Read-only audit. No code changed. All line numbers as of commit
`b3f32e2` (harness/gate.py's val_pred_path/cache fix + HANDOFF.md).

## Headline finding

**`controller/` never imports from `harness` at all** — zero matches for
`from harness` / `import harness` in any `controller/*.py` file.
`clears_convergence_epsilon` is never called; it does not appear
anywhere in `controller/`. Convergence is entirely reimplemented locally
in `controller/convergence.py`, with its own hardcoded `EPSILON = 0.002`
and `SIGMA = 0.0008` (sourced from the organizers' published JSON, not
from `harness.gate.SIGMA`, which is our own measured validation figure,
0.000353 — see HANDOFF.md).

That reimplementation has a structural bug (Check 1 below): the
"internal rule," specifically designed as the noise-aware second opinion
to the organizers' noisy rule, **can never fire** against a real
`harness.gate.compare()` Verdict. It only appears to work in tests
because the test double that exercises it (`DeltaGate`) constructs a
Verdict shape the real gate can never produce.

---

## CHECK 1: two-signal separation

### (a) Promotion to incumbent — uses `verdict.accept`. Correct.

`controller/controller.py:607-608`:
```python
if verdict.accept:
    state = state.with_incumbent(result, config)
```
This is the only place a candidate becomes the incumbent (aside from the
baseline-adoption special case at `controller.py:569`, which runs
without a gate call at all since there is nothing to compare against
yet). Correct signal.

### (b) Consecutive-non-improvement tracking — uses NEITHER `verdict.accept` NOR `clears_convergence_epsilon`. Uses a third, independently recomputed quantity.

The organizers' N=3 rule lives in `controller/convergence.py:212-224`
(`_organizers_rule`) and `convergence.py:189-209` (`_improvements`):

```python
# convergence.py:206-209
return tuple(
    committed[i].primary - committed[i - 1].primary
    for i in range(1, len(committed))
)
```

This differences each committed revision's own **absolute `.primary`**
against the previous committed revision's `.primary` — it does **not**
read `verdict.delta` (the gate's paired per-seed delta, which is a
*different number* and is separately stored on the same `HistoryEntry`
at `controller.py:613`, `delta=verdict.delta`). The comparison against
epsilon is against the local `EPSILON` constant
(`convergence.py:78, 224`), not `harness.gate.CONVERGENCE_EPSILON` or
`clears_convergence_epsilon`.

Confirmed nothing in `convergence.py` ever reads `.delta` from a
`CommittedRevision`, despite the Protocol declaring it
(`convergence.py:134`) and the module docstring explaining at length why
`delta`/`ci95`/`significant` are retained on `HistoryEntry` specifically
to feed this module (`state.py:220-228`):

```
$ grep -n "\.delta\b" controller/convergence.py
157:    measurement and live on HistoryEntry.delta and in DECISION events."""
```
— a docstring mention only. The field is carried all the way from the
gate onto `HistoryEntry` and then never read by the logic it was kept
for.

### (c) `verdict.accept` is not used as the convergence signal directly — but the "internal rule" that was built to replace it is dead code.

The internal rule (`convergence.py:227-245`, `_internal_rule`) converges
when `all(r.significant is False for r in window)`. `significant` is
computed once, at the point of decision, via:

`controller/controller.py:617`:
```python
significant=is_significant(verdict.ci95),
```

`is_significant` (`convergence.py:173-186`) returns
`not (ci95[0] <= 0 <= ci95[1])` — True whenever the interval's lower
bound is positive.

**The window this rule evaluates is `state.committed_revisions`
(`state.py:501-514`), filtered to `entry.accepted` only.** And
`harness/gate.py`'s `_confirm` only ever sets `accept=True` in the
branch reached after `ci95[0] > 0` has already been checked (`gate.py`'s
`_confirm`: `if ci95[0] <= 0: accept=False ...`, falling through to
`accept=True` only past that guard, and past the backtest checks).

**Therefore: every real, non-baseline accepted revision has
`ci95[0] > 0` by construction, which means `is_significant(ci95)` is
always `True` for it.** The one committed entry that can have
`significant=None` is the baseline (`controller.py:586-592`, gate never
called), which occupies the window only until 3 more revisions are
committed. Once the window is 3 real accepted revisions deep,
`_internal_rule` requires all three to have `significant is False` —
which is now impossible, because `significant=False` cannot coexist with
`accepted=True` under the real gate. **The internal rule can never
converge in production.**

This is confirmed by how the test suite actually exercises it. The only
test double that drives the internal rule
(`tests/test_controller.py:863-879`, `test_a_flat_run_converges_and_says_so`,
using `DeltaGate(delta=0.0005)`) works only because `DeltaGate` violates
the real gate's invariant: `controller/fakes.py:811-819` defaults
`accept=True` while `ci95=(-0.00107, +0.00207)` straddles zero —
`accept=True` with `significant=False`, a Verdict shape
`harness.gate.compare()` cannot produce. The test's own assertion
(`tests/test_controller.py:892`, `by_rule in {"organizers", "internal"}`)
is loose enough that it doesn't need to distinguish which rule actually
fired, and given `FakeExecutor`'s flat scores also satisfy the
organizers' rule (per the test's own docstring), it's likely the
organizers' rule is doing the real work in that test regardless.
`tests/test_convergence.py` unit-tests `_internal_rule` correctly and
usefully as a pure function (e.g. line 105, 118, 218 build
`significant=False` fixtures directly) — that part is fine in isolation.
The bug is entirely in the wiring between the real gate and this module,
not in the module's own arithmetic.

### (d) Does the controller import from harness.gate at all?

No. Confirmed by `grep -rn "from harness\|import harness" controller/*.py`
— zero results. Everything convergence-related is a local
reimplementation in `controller/convergence.py`, including its own
`EPSILON` and `SIGMA` constants (`convergence.py:78, 98`), independently
sourced from the vendor's published JSON rather than from
`harness.gate.CONVERGENCE_EPSILON` / `harness.gate.SIGMA`.

---

## CHECK 2: Verdict shape

Only two `Verdict(` construction sites in `controller/`, both in
`controller/fakes.py` (test doubles; `controller.py` never constructs
one, only consumes what the gate returns):

- `fakes.py:709` (`_ConstantGate.compare`, backing `AlwaysAcceptGate`/`AlwaysRejectGate`)
- `fakes.py:824` (`DeltaGate.compare`)

**`n_seeds`: not omitted.** Both sites supply it (`fakes.py:713`,
`fakes.py:832`) — this was already fixed in a prior pass.

**`backtest_delta`: never `None` in either fake.** `_ConstantGate`
always computes a real paired backtest delta (`fakes.py:702, 719`,
explicitly commented "this fake ... does not model the 'no backtest
ran' case"); `DeltaGate` sets `backtest_delta=self.delta`
(`fakes.py:835`), also always a float. Consequence: **no test double in
`controller/` ever produces a Verdict with `backtest_delta=None`**, so
the `backtest_missing` path is never exercised at the controller
integration level, even though `harness/gate.py` can produce it.

**Arithmetic/comparison on `backtest_delta` without a None check:**
none found. `controller.py:603` is the only read of
`verdict.backtest_delta` anywhere in `controller/`, and it's a bare
assignment into a journal payload dict (`"backtest_delta":
verdict.backtest_delta`) — `None` there just serializes to JSON `null`.
No crash risk.

**`verdict.reason`: no `==` comparisons anywhere.** `controller.py:604`
is the only read of `verdict.reason` in `controller/`, again a bare
assignment into the DECISION payload. No exact-match risk from the
`"; coarse_ci_seed_bootstrap"` tag suffix.

**Not asked, but adjacent and worth flagging: `verdict.n_seeds` is
never read by `controller.py` at all.** It's set on every constructed
Verdict (required field) but the DECISION payload
(`controller.py:596-606`) records `verdict` (=accept), `delta_primary`,
`ci95`, `backtest_delta`, `reason` — not `n_seeds`. `HistoryEntry`
(`state.py:211-239`) has no `n_seeds` field either. The "how much
evidence backed this verdict" signal — the entire reason `n_seeds` was
added to the contract (see `contracts.py`'s `Verdict.n_seeds` comment) —
is computed by the gate and then dropped before it reaches the journal
or any controller state.

---

## ALSO REPORT

**What does `controller/` import from `harness/`? Anything expected
that doesn't exist?**

Nothing. `controller/` imports only from `contracts` (types) and its own
submodules (`controller.ports`, `controller.state`, `controller.convergence`,
`controller.policy`). No file in `controller/` imports `harness.gate`,
`harness.cache`, `harness.data`, `harness.metrics`, or `harness.backtest`.
Nothing is missing because nothing is expected — but this also means
none of `harness/HANDOFF.md`'s public API is actually wired into the
controller yet, including `gate.compare` and `gate.clears_convergence_epsilon`
themselves: `GatePort` (`controller/ports.py`) is a structural Protocol
that `harness.gate` presumably satisfies by having a matching `compare`
method, but there is no import, instantiation, or test proving that the
real `harness.gate` module is actually plugged in anywhere — every
`GatePort` in the current test suite is a fake from `controller/fakes.py`.

One related staleness: `controller/ports.py:34-40`'s "NO CACHE PORT,
DELIBERATELY" comment says *"harness/cache.py is still
`get(*args, **kwargs)` / `put(*args, **kwargs)` with no real
signature"* — that was true when written but is no longer accurate;
`harness/cache.py` now has real signatures for `get`/`put` plus
`save_predictions`/`load_predictions`/`exists`. Doesn't affect behavior
(the architectural conclusion — no CachePort, executor owns the cache —
is still correct and is exactly what HANDOFF.md also says), but the
comment's factual claim about `cache.py`'s current state is out of date.

**Is there a scripted (non-LLM) hypothesis generator in place — is I2
runnable today?**

No. `methods/` contains only `__init__.py` (no implementation).
`controller/ports.py:266` says so directly: *"IMPLEMENTED BY W4, AND
methods/ IS CURRENTLY EMPTY."* The only things satisfying `GeneratorPort`
today are test doubles in `controller/fakes.py`
(`ScriptedGenerator`, `DisobedientGenerator`) used solely for controller
unit tests. There is no real generator — scripted or LLM — runnable
anywhere in the repo yet.

**Does anything in `controller/` call `cache.save_predictions`, or is
the executor solely responsible?**

Executor solely responsible, and this is deliberate, not an oversight:
`controller/ports.py:34-40`, *"NO CACHE PORT, DELIBERATELY ... the
Controller never touches the cache directly — the executor does,
underneath ExecutorPort.run."* No `cache.` call of any kind appears in
`controller/controller.py`, `state.py`, `policy.py`, or `convergence.py`
(only mentions are the comment above and an unrelated docstring use of
the word "cache" in `state.py:589`, `fakes.py:193` — not calls). This
matches `harness/HANDOFF.md`'s framing exactly: W3 must call
`cache.save_predictions` before returning a `CandidateResult`, and the
controller is correctly not in that path at all.
