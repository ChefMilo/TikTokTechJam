# EXECUTOR_SURVEY.md — executor/ and methods/, as of commit `c064dd3`

Read-only survey. No code changed.

## executor/

**1. What files exist, and what does each contain?**

```
executor/__init__.py   (7 lines)
```

That is the entire package. Its content is the original scaffold
docstring, unchanged since the initial commit:

```python
"""Executor package: sandboxed execution of candidate methods, error
taxonomy, automated repair, and telemetry.

Will hold: the sandbox that runs untrusted/generated method code, error
classification, retry/repair logic, and run telemetry collection. Owned
by the executor team.
"""
```

No other file, real or stub, exists anywhere under `executor/`. There is
no second module, no class, no function — just this one docstring.

**2. Sandbox (subprocess, timeout, memory caps)?** No. `grep -rin
"subprocess\|resource\.setrlimit\|multiprocessing\|sandbox"` across the
whole repo (excluding `.venv`) matches only the word "sandbox" inside
this same docstring. No process isolation, no timeout, no memory limit
exists anywhere.

**3. Error taxonomy classifier producing `contracts.ErrorClass`?** No.
`ErrorClass` is defined in `contracts.py` and referenced in three other
places, none of them a classifier:
- `controller/controller.py:451,491` — the Controller hardcodes
  `ErrorClass.CONTRACT` for its own internal port-violation detection
  (e.g. a generator or realizer breaking its own protocol). This
  classifies bugs in *other controller-side ports*, not a candidate's
  training failure.
- `controller/ports.py:297` — a comment describing the line above.
- `controller/fakes.py` / test files — test doubles and assertions.

Nothing anywhere inspects a real training failure (OOM, timeout,
NaN loss, degenerate output, a broken dependency) and maps it to one of
`ErrorClass`'s values. The taxonomy is defined; nothing classifies into it.

**4. Anything that takes a `PipelineConfig` and actually trains a model?**
No. No file outside `vendor/kuairand-starter-kit/` and `harness/`
performs training. The actual numerical work already exists and is
already tested — `vendor/kuairand-starter-kit/baseline.py`'s `FM` class
and `run_fm` (exercised directly by `tests/test_rungs.py` and
`scripts/seed_variance.py`) — but nothing turns a `contracts.PipelineConfig`
into a call to it. There is no "realizer": no code reads
`config.slots["model"].impl == "fm"` and dispatches to `run_fm`, or reads
any other slot's `impl` at all.

**5. Does anything call `cache.save_predictions`?** No, outside
`harness`'s own test suite (`tests/test_cache.py`, `tests/test_gate.py`).
`grep -rn "save_predictions"` repo-wide confirms every call site is a
test. Per `harness/HANDOFF.md`, this is the single most important
integration requirement for W3 — and nothing calls it yet.

**6. Does anything construct a `CandidateResult`? With what fields?**
Only test doubles and tests — never production code:
- `controller/fakes.py`'s `FakeExecutor.run` (the only *realistic-shaped*
  construction) populates `config_id`, `status`, `val`, `backtest`,
  `wall_seconds`, `tokens_in`, `tokens_out`, and on failure
  `error_class`/`error_excerpt`. It deliberately never sets
  `val_pred_path`, `test_pred_path`, `gpu_seconds`, or `repair_attempts`
  — consistent with it not modeling a cache at all (its own docstring
  says so).
- `controller/fakes.py`'s other gate/test doubles, `tests/test_controller.py`,
  `tests/test_controller_fakes.py`, `tests/test_gate.py` construct
  `CandidateResult` directly for synthetic test fixtures.

No file under `executor/` constructs one, because no file under
`executor/` exists beyond the docstring.

**7. Does `executor/` import from `harness/` at all?** N/A — there is no
code in `executor/` to import anything. (For contrast: nothing in
`controller/` imports from `harness/` either, per `CONTROLLER_AUDIT.md`.)

**8. Journal implementation (append-only JSONL, replay)?** No real one.
The only `Journal`-shaped thing anywhere is `controller/fakes.py`'s
`InMemoryJournal` — a test double that appends `JournalEvent`s to a Python
list (`self._events`) and offers `events_of_kind` for assertions. It
never touches disk, never writes JSONL, and has no replay/resume logic.
`contracts.JournalEvent.to_jsonl`/`from_jsonl` (with the hardened
`JournalDecodeError` handling) exist and are unit-tested in
`tests/test_contracts.py`, but nothing anywhere calls them against a real
file. `JournalPort` in `controller/ports.py` is a Protocol only.

---

## methods/

Besides `scripted.py` (the deterministic I2 generator, added last turn):

```
methods/__init__.py   (7 lines, scaffold docstring only)
```

Nothing else. No method/impl registry, no YAML library (referenced by
`Citation.library_entry` and `scripted.py`'s own citations as
`methods/library/*.yaml` — none of those paths exist), no prompt
templates, no LLM-backed generator. `controller/ports.py:266` says this
directly: *"IMPLEMENTED BY W4, AND methods/ IS CURRENTLY EMPTY."*
`scripted.py` is the only content in the package.

---

## Shortest path to running ONE real candidate (`baseline_reproduce`) end-to-end

Config in → trained model → predictions saved to cache → `CandidateResult`
out → scored by `harness.metrics`. In dependency order, what's missing:

1. **A default/baseline `PipelineConfig` for the other five slots.**
   `methods/scripted.py`'s `baseline_reproduce` move only proposes a
   fragment for `model` (`impl="fm", params={"k":16,"lr":0.001,"epochs":40}`).
   Nothing anywhere defines what `data_view`, `features`, `weighting`,
   `objective`, and `calibration` should be to reproduce the vendor's
   5-field FM baseline exactly. This has to exist before there is a
   complete `PipelineConfig` to hash or run at all.

2. **A realizer that dispatches `SlotConfig.impl` to real code.**
   Something that reads a full `PipelineConfig` and knows `impl="fm"`
   means "call `vendor.baseline.FM`/`run_fm` with these params" (and,
   eventually, what the other slots' impls mean). This is the executor's
   core job and doesn't exist in `executor/` (empty) or `methods/`
   (only a generator, not a realizer). The underlying training code this
   would call — vendor's `FM`/`run_fm` — already exists and is already
   proven correct (`tests/test_rungs.py`).

3. **A backtest training pass.** `CandidateResult.backtest` needs
   `Metrics` from training on `harness.backtest.split()`'s `fit_rows` and
   scoring on `score_rows`, using the same resolved model config. Nothing
   calls `harness.backtest.split()` outside its own tests today — this is
   a second realization of step 2, not automatically produced by it.

4. **Extracting per-seed predictions and calling `cache.save_predictions`.**
   After training (validation pass), the `(user_ids, labels, scores)`
   triple for each seed must reach
   `harness.cache.save_predictions(config_id, seed, "val", ...)`. This is
   the step `harness/HANDOFF.md` calls out explicitly — skip it and the
   noise gate silently degrades to its ~12.5%-false-positive fallback.
   Nothing calls this outside `harness`'s own tests today.

5. **Assembling the `CandidateResult`.** `config_id` (from
   `PipelineConfig.config_id`), `status=Status.OK`, `val` and `backtest`
   (from `harness.metrics.evaluate` on each seed's predictions — this
   part already works and is already correct), plus cost fields
   (`wall_seconds` from a timer, `tokens_in=tokens_out=0` since this is a
   pre-scripted move with no LLM call, `gpu_seconds=0` for CPU-only FM).
   No real code does this assembly today.

6. **The executor itself** — an `ExecutorPort`-shaped object
   (`run(config, seeds) -> CandidateResult`) that wires 1-5 together.
   `executor/` is the literal missing package; this is where it goes.

**Not required to get one trusted, hand-authored candidate through once,
but needed before this can run unsupervised or on untrusted/LLM-generated
code:** the sandbox (#2 in the executor's own docstring), the error
taxonomy classifier (#3), and gate wiring (comparing against an
incumbent needs a second candidate to compare against, which is outside
the scope of "one candidate in, one candidate out"). Two small,
adjacent items worth fixing alongside whichever of these lands first:
`controller/ports.py`'s `GatePort` docstring still claims *"harness/gate.py
exposes only `passes_gate(*args, **kwargs)`... nothing anywhere returns a
Verdict"* — no longer true, `harness.gate.compare` has matched this
Protocol's exact signature since the fixes two turns ago — and its
`ExecutorPort`/cache-signature comment (`ports.py:36`) is similarly
stale about `harness/cache.py`'s current real API.
