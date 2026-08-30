# harness/ — interface reference for W2 (controller) and W3 (executor)

Not a tutorial. This is the contract: what to call, what it returns, and
the facts about the data/metrics you need to make correct decisions.

## WHAT W3 MUST DO OR THE GATE DEGRADES SILENTLY

**Every successful candidate must call**

```python
cache.save_predictions(config_id, seed, "val", user_ids, labels, scores)
```

**before the `CandidateResult` is returned.** Without this, `gate.compare()`
falls back to a bootstrap over just the 3-5 per-seed deltas, which has a
**~12.5% false-positive rate** (three same-signed noise draws happen
1-in-8 of the time, and resampling same-signed numbers can never cross
zero). The run still completes and looks completely normal — a `UserWarning`
fires and the verdict's `reason` gets tagged `coarse_ci_seed_bootstrap`,
but nothing stops the run. This is the single most important integration
requirement in the harness. If you see that warning or that tag in the
journal, predictions are missing for some (config_id, seed) — check the
warning message, it names them.

`gate.py` decides whether the real bootstrap is available by calling
`cache.exists(config_id, seed, "val")` directly — not by checking any
field on `CandidateResult`. There is nothing else to set; saving the
predictions is sufficient and necessary.

## PUBLIC API

```python
data.load(split: str) -> list[tuple]                         # split="test" raises PermissionError
data.load_side_features() -> tuple[DataFrame, DataFrame]      # (user_features, video_features)
metrics.evaluate(user_ids, labels, scores, k=5) -> Metrics
backtest.split() -> tuple[list[tuple], list[tuple]]            # (fit_rows, score_rows)
gate.compare(candidate: CandidateResult, incumbent: CandidateResult) -> Verdict
gate.clears_convergence_epsilon(verdict: Verdict) -> bool
cache.get(slot_hash: str) -> Any | None
cache.put(slot_hash: str, artifact: Any) -> None
cache.save_predictions(config_id: str, seed: int, split: str, user_ids, labels, scores) -> None
cache.load_predictions(config_id: str, seed: int, split: str) -> tuple[ndarray, ndarray, ndarray]
cache.exists(config_id: str, seed: int, split: str) -> bool
validate.validate_submission(path: str, split: str = "test") -> bool
validate.write_submission(path: str, rows, scores) -> None
```

`data.load("test")` and `validate.validate_submission`'s hidden-test path
are the only two sanctioned places test-split structure is ever touched;
everywhere else, "test" is off-limits by construction.

## ROW TUPLE SHAPE

Every row from `data.load()` (and thus `backtest.split()`) is:

```
(date: int, user_id: str, video_id: str, author_id: str,
 tab: str, duration_ms: float, long_view: int)
```

**Index 6 is the label** (`long_view`, 0/1). This is the vendor's own row
shape, unchanged — see `harness/SCHEMA_NOTES.md` data.py Q7-Q8 for the
full derivation.

## MEASURED FACTS the controller should know

- `sigma_primary` on validation = **0.000353** (`artifacts/seed_variance.json`,
  seeds 0-4). Treat this as a **LOWER BOUND**, not an honest estimate:
  vendor `run_fm` early-stops on validation primary itself, so each run
  reports a max over ~40 epochs on the split being measured, which
  compresses variance relative to a single evaluation.
- Convergence epsilon (0.002) is **5.66x** that measured sigma — vs. the
  organizers' own ~2.5x on their (higher) test-split sigma. The gate's
  screen threshold floors at `max(3*SIGMA, 0.002)` for this reason.
- FM baseline validation primary: **0.6015** (published: 0.6016; GAUC
  0.6671/0.6674, nDCG@5 0.5358/0.5357).
- Oracle ceiling: **0.8484 on validation**, 0.8645 on hidden test (nDCG@5
  can't reach 1.0 — ~27-30% of users are all-negative/all-positive).
- Backtest windows (`harness/backtest.py`): fit = date <= 20220417
  (1,055,237 rows), score = 20220418-20220421 (85,875 rows). Chosen for
  volume *shape*, not just date arithmetic — see next point.
- **Train volume is heavily front-loaded**: peak 278,835 rows on
  20220411, decaying to a ~20-24k/day plateau by 20220418-21.
  **Validation is flat at 14-27k/day** — it looks like the plateau, not
  the burst. A model fit on the whole train window is fit partly on a
  regime (the burst) that validation doesn't resemble. **Recency
  weighting is therefore a well-motivated candidate move**, not a guess.
- `20220408` has **zero rows** despite being the documented train start
  (`train` window is nominally 20220408-20220421).

## TWO SIGNALS, NOT ONE

`verdict.accept` and `clears_convergence_epsilon(verdict)` answer
**different questions** and must be tracked separately:

- `verdict.accept` — is this improvement statistically real? (paired CI
  excludes zero AND backtest agrees in direction)
- `clears_convergence_epsilon(verdict)` — is it big enough (`delta >
  0.002`) to reset the organizers' N=3 no-improvement counter?

A run can **accept several real gains below 0.002 in a row and still be
declared converged** by the organizers' rule, because none of them
individually cleared epsilon. Track both; do not conflate "accepted" with
"counts toward not-converged."

## GATE STAGE DISPATCH

`gate.compare()` dispatches on `len(candidate.val)`:

| seeds in `candidate.val` | stage | behaviour |
|---|---|---|
| 1 | **screen** | reject-only; can never return `accept=True` |
| >= 3 | **confirm** | the real decision (paired deltas, bootstrap CI, backtest) |
| 2 (or 0) | — | raises `ValueError` — not a supported shape |

`n_seeds` on the returned `Verdict` is always the number of **matched**
seeds actually used (intersection of candidate and incumbent seeds), not
the total on either side.
