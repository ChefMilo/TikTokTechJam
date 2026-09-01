# TikTokTechJam

An autonomous ML research agent for the [KuaiRand-Pure](https://kuairand.com/) recommender-system benchmark.

## Project overview

**Architecture thesis:** a cost-aware experiment controller sits on top of a typed, six-slot pipeline. An LLM proposes hypotheses and turns them into code for one slot at a time; every other part of the loop — evaluation, caching, statistics, retry, logging — is deterministic Python. Splitting the system this way is what buys autonomy, robustness, and cost control at the same time, rather than trading one for another:

- **Autonomy** comes from the two LLM seams being narrow and swappable. `controller/ports.py` defines `GeneratorPort.propose(state, target_slot) -> HypothesisPayload` ("what should we try, and why") and `RealizerPort.realize(hypothesis) -> SlotConfig` ("turn that into a runnable slot config") as two separate model calls with two separate token budgets. Neither one ever needs to see or write more than one slot's worth of the pipeline, so a wrong or wasted call is cheap to detect and cheap to retry.
- **Robustness** comes from everything downstream of those two calls being ordinary, tested Python with no model in the loop: `harness/data.py` enforces a hard lockout on the hidden test split, `harness/metrics.py` reproduces the organizers' own metric math exactly, `harness/gate.py` is a paired bootstrap significance test (not a raw score comparison) that also requires backtest confirmation before accepting anything, and `executor/journal.py` writes an append-only, crash-resumable log of every hypothesis, result, and decision. None of that logic can hallucinate; it either passes its tests or it doesn't run.
- **Cost control** comes from the same split: an LLM is only ever asked to do two small, well-scoped things (propose, realize), so the run's token cost is the sum of exactly `2 x (number of candidates tried)` model calls, fully accounted for in the journal — not an open-ended agent loop that can spend an unbounded amount of budget deciding what to do next.

The six pipeline slots, each independently swappable (see `contracts.py`'s `SlotName`):

| Slot | Governs |
|---|---|
| `data_view` | which rows of the interaction log are used to fit |
| `features` | which input fields the model sees |
| `weighting` | per-row sample weighting during training |
| `model` | the model class and its hyperparameters |
| `objective` | the training loss |
| `calibration` | post-hoc score adjustment before submission |

The codebase is split into four independently-owned packages plus a shared contract module:

| Package | Owns |
|---|---|
| `harness/` | Data loading with a hard test-split lockout, metrics, the noise gate, backtesting, prediction caching, submission validation |
| `controller/` | The agent's state machine, search policy, port protocols, convergence tracking |
| `executor/` | Realizing a `PipelineConfig` into trained predictions, the error taxonomy, the journal writer, the report renderer |
| `methods/` | The hypothesis library — currently a deterministic ten-move script (`ScriptedGenerator`), standing in for the `GeneratorPort` |
| `autonomy/` | Adapters wiring the real `Controller` to the real `harness`/`executor` (`GeneratorPort`/`RealizerPort` implementations over the scripted moves), the code-fingerprint/manual-intervention integrity system (`INTERVENTION_POLICY.md`, `integrity.py`), and the autonomy-section renderer (`render.py`) that reads a Controller journal directly |
| `manual/` | The hand-built "manual ceiling" reference pipeline — measures headroom above the organizers' published baseline independently of the agent; never goes through the `Controller` |

`contracts.py` at the repo root is the frozen cross-package interface — dataclasses only, no behavior. Every package imports from it rather than reaching into another package's internals.

Other top-level directories: `vendor/` (the organizer's starter kit, vendored unmodified — every wrapper in `harness/`/`executor/` imports it by file path, never edits it), `scripts/` (CLI entry points), `tests/`, and `data/`/`artifacts/` (local, gitignored working directories for the dataset and run outputs).

## Setup and installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.9+ (matching the vendored starter kit's own requirement).

Download the KuaiRand-Pure dataset (Zenodo, no registration required — see `harness/SCHEMA_NOTES.md` Q21):

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

Copy the four CSVs the harness reads into this repo's `data/` directory (gitignored, not shipped):

```
data/log_standard_4_08_to_4_21_pure.csv
data/log_standard_4_22_to_5_08_pure.csv
data/video_features_basic_pure.csv
data/user_features_pure.csv
```

(`log_random_4_22_to_5_08_pure.csv` from the same tarball is not read by anything here and can be skipped.)

The organizer's starter kit already lives in `vendor/kuairand-starter-kit/` and needs no separate download.

Verify the packages import cleanly:

```bash
python -c "import harness, controller, executor, methods"
```

## Steps to reproduce

Run in order from the repo root:

```bash
pytest                              # 559 tests as of this writing
python scripts/seed_variance.py     # ~4 min
python scripts/run_agent.py         # ~12 min
python scripts/run_controller.py    # varies with --max-nodes-per-stage / --seeds
python scripts/make_submission.py   # ~2.5 min
```

- **`pytest`** — the full test suite for `harness/`, `controller/`, `executor/`, `methods/`, `autonomy/`, and the `manual/` reference runner.
- **`scripts/seed_variance.py`** — trains the FM baseline at 5 seeds on validation and measures per-seed noise (sigma), writing `artifacts/seed_variance.json`. `harness/gate.py`'s acceptance thresholds are calibrated against this file.
- **`scripts/run_agent.py`** — the fixed ten-move-plus-ensemble replay described under Results below, and **the driver that produced the submitted result**: `scripts/make_submission.py` retrains the exact candidate this run accepts. Writes `artifacts/journal_run.jsonl` (the append-only decision log) and renders `artifacts/report/{iterations,results,forecast_calibration}.md` + `trajectory.csv`. Moves 1, 2, 3, 8, and the ensemble rebuild their `CandidateResult`s from already-cached predictions (no training); moves 4, 5, 6, 7, 9, and 10 are executed for real, since nothing has run them before.
- **`scripts/run_controller.py`** — drives the real `controller.Controller` loop end to end (real hypotheses via the scripted moves through `GeneratorPort`/`RealizerPort`, the real `harness.gate`, real training through `executor.run.run_candidate`), writing `artifacts/journal_controller.jsonl` and, with `--report`, rendering it via `executor/report.py`. This is the actual unattended search loop, separate from `run_agent.py`'s fixed replay above; runtime scales with `--max-nodes-per-stage`/`--seeds` (a `--max-nodes-per-stage 1 --seeds 0` smoke run took ~8 minutes). It has not yet found a candidate that beats `run_agent.py`'s ensemble.
- **`scripts/make_submission.py`** — retrains the accepted 3-seed rank-average ensemble on the full training split, scores the hidden test split, and writes/validates `artifacts/submission_test.csv` against the organizer's own `submit.py --check`.

## Results

Source: `artifacts/report/results.md`, rendered from `artifacts/journal_run.jsonl` by `scripts/run_agent.py`.

| Metric | Validation-best | Official baseline | Delta |
|---|---|---|---|
| GAUC | 0.6683 | 0.6674 | +0.0009 |
| nDCG@5 | 0.5363 | 0.5357 | +0.0006 |
| primary (mean of GAUC, nDCG@5) | 0.6023 | 0.6016 | +0.0007 |

**Accepted candidate:** a 3-seed rank-average ensemble of the baseline FM (three disjoint groups of the nine base seeds, ranks averaged rather than scores — robust to inter-seed scale differences). `harness/gate.py`'s paired bootstrap: delta +0.00083 on validation, ci95 = (+0.00029, +0.00139) — entirely positive, excludes zero — and backtest confirms in the same direction, delta +0.00089.

**The most interesting finding in this run:** that acceptance is statistically real but **not** big enough to matter under the organizers' own convergence rule. `epsilon = 0.002` and the ensemble's delta is +0.00083 — well inside a real, gate-confirmed improvement, but short of the threshold that resets the organizers' N=3 no-improvement counter. Of 7 candidates that reached a decision, exactly 1 was accepted as an improvement (the ensemble; move 1's "adopt as initial incumbent" decision is not an improvement over anything and is excluded from this count), and 0 cleared epsilon. Under the organizers' rule, this run would still be judged as stalled despite a real, confirmed gain.

### Every move tried

| # | Move (`target_slot`/`impl`) | Outcome | delta (val) | reason |
|---|---|---|---|---|
| 1 | baseline_reproduce (`model`/`fm`, k=16) | adopted as initial incumbent | +0.0000 | control — nothing to compare against yet |
| 2 | recency_weight_exp (`weighting`/`exp_decay`) | REJECTED | +0.0003 | `ci_includes_zero` |
| 3 | recency_window (`data_view`/`recent_window`) | REJECTED | -0.0090 | `ci_entirely_negative` |
| 4 | multitask_longview_click (`objective`/`multitask_bce`) | **not realized** | — | `CONTRACT`: no realization implemented for objective impl `'multitask_bce'` |
| 5 | fm_rank_k (`model`/`fm`, k=32) | REJECTED | +0.0000 | `ci_includes_zero` |
| 6 | duration_debias (`calibration`/`duration_debias_cwm`) | **not realized** | — | `CONTRACT`: no realization implemented for calibration impl `'duration_debias_cwm'` |
| 7 | model_lightgbm (`model`/`lightgbm`) | **not realized** | — | `CONTRACT`: no realization implemented for model impl `'lightgbm'` |
| 8 | pairwise_loss (`objective`/`bpr`) | REJECTED | -0.0033 | `ci_entirely_negative` |
| 9 | popularity_prior (`calibration`/`popularity_blend`) | **not realized** | — | `CONTRACT`: no realization implemented for calibration impl `'popularity_blend'` — 2nd consecutive `calibration` failure, circuit breaker blocks that slot |
| 10 | tune_lr_epochs (`model`/`fm`, lr=0.0005/epochs=60) | REJECTED | +0.0003 | `ci_includes_zero` |
| ensemble | rank_avg_ensemble (3-seed disjoint rank-average of move 1) | **ACCEPTED** | +0.0008 | paired CI excludes zero, entirely positive; backtest confirms (+0.00089) |

Six hypotheses were actually measured (1, 2, 3, 5, 8, 10, plus the ensemble); four (4, 6, 7, 9) could not be realized — `executor/realize.py` has no implementation for their slot/impl combination — and were routed around rather than stalling the run. Each of those four still produced a HYPOTHESIS record (the reasoning is on the record even though it couldn't run), an ERROR tagged `contract`/`skip_unimplemented`, and a RECOVERY event showing the run continued to the next candidate. `scripts/run_agent.py`'s docstring has the full detail, including a correction: the run's own brief assumed all six of {4,5,6,7,9,10} would fail this way, but moves 5 and 10 both use `model.impl="fm"`, which *is* implemented regardless of its hyperparameters — both trained successfully and were gate-rejected on their own merits, not routed around.

## Limitations and what we'd improve

- **The hypothesis generator is deterministic and scripted, not an LLM.** `methods/scripted.py`'s `ScriptedGenerator` emits a fixed, hand-authored ten-move script and implements the same `propose()` shape `controller/ports.py`'s `GeneratorPort` protocol expects, so an LLM generator could sit behind that interface without changing anything downstream — but this run used the scripted one. Its LLM token cost is exactly zero, and the autonomy the architecture is designed for is demonstrated structurally (the interface exists and is exercised end-to-end), not behaviorally (no model actually chose what to try).
- **The "realize" step is not LLM-driven either.** `controller/ports.py` also defines a `RealizerPort` (LLM call #2: turn a hypothesis into a runnable `SlotConfig`), but nothing in this run calls it. `executor/realize.py` is a hand-written Python dispatch table covering six specific slot/impl combinations (`fm`, `recent_window`, `exp_decay`, `bpr`, and their composition rules) — real training code, but authored by hand, not generated per-hypothesis. Both LLM seams in the architecture thesis above are currently filled with deterministic stand-ins.
- **Four of ten scripted moves have no realizer implementation** (`multitask_bce`, `duration_debias_cwm`, `lightgbm`, `popularity_blend` — see the table above). They're real, motivated hypotheses (each cites a source), just not yet wired to training code.
- **Measured validation sigma (0.000353) is a lower bound, not an honest estimate.** The vendor FM baseline early-stops on validation primary itself, so each seed's reported score is a max over ~40 epochs on the very split being measured — that compresses variance relative to a single evaluation. `harness/gate.py` accounts for this by flooring its acceptance threshold at `max(3*sigma, 0.002)` rather than trusting `3*sigma` alone.
- **Only the additive move (the rank-average ensemble) beat the baseline.** Every component-replacing move — different weighting, a hard recency cutoff, a pairwise objective, different FM capacity, a different learning-rate schedule — either lost outright or landed inside the noise gate's rejection region. On this dataset, at this point in the search, combining seeds of the same model beat swapping any single component of it.
- **The manual ceiling (`manual/`) exists but has never been run.** It's a real, hand-built reference pipeline (`manual/run.py`, `manual/encode.py`) meant to measure how much headroom exists above the organizers' published FM baseline (0.6016) independently of the agent — but no cached predictions, journal, or results file for either of its variants (`manual_baseline_fm_k16`, `manual_crosses_v1`) exist anywhere in this repo. So the accepted ensemble's +0.00083 has no measured ceiling of our own to be judged against; only the organizers' own published baseline and oracle figures are available for comparison.
- **The error taxonomy (`executor/errors.py`) declares repair policies for `OOM`, `TIMEOUT`, and `NAN_LOSS`** (`simplify_retry`, `simplify_retry`, `lower_lr_retry` respectively) **that were never exercised.** Only `ErrorClass.CONTRACT` / policy `skip_unimplemented` actually fired in this run, because the only failure mode any scripted move produced was an unimplemented slot/impl combination. The other classes have real, narrow detection rules (see `executor/errors.py`'s `classify()`) but no retry loop has ever been built or tested behind them — they're declared for the day that work happens, not evidence it already has.

## Team member contributions

<!-- TODO: fill in names per workstream. -->

| Workstream | Owns | Contributor(s) |
|---|---|---|
| Harness (`harness/`) | Data, metrics, noise gate, backtest, cache, submission validation | TODO |
| Controller (`controller/`) | State machine, search policy, port protocols, convergence | TODO |
| Executor (`executor/`) | Realization, error taxonomy, journal, report | TODO |
| Methods (`methods/`) | Hypothesis library / scripted generator | TODO |
