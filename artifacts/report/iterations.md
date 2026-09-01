# Iterations

_`node` counts every evaluation attempted; `iteration` counts committed revisions and only advances when a DECISION accepts. Several consecutive nodes sharing one iteration number means several consecutive rejections, not a stall — see executor/journal.py's log_decision docstring._

## Errors and recovery

| node | error_class | policy | recovered |
|---|---|---|---|
| 4 | contract | skip_unimplemented | yes |
| 6 | contract | skip_unimplemented | yes |
| 7 | contract | skip_unimplemented | yes |
| 9 | contract | skip_unimplemented | yes |

4 candidate(s) failed; the run continued past every one.

## Node 1 (iteration 1)

**Hypothesis** (target_slot=`model`)

- Rationale: Reproduce the organizers' own published FM baseline exactly (k=16, lr=0.001), to confirm the harness end-to-end — data loading, encoding, training, evaluation — reproduces validation primary 0.6016 before any structural change is judged against it. This is the control, not a candidate expected to win.
- Citation: [rendle2010fm](https://ieeexplore.ieee.org/document/5694074) (methods/library/fm.yaml#factorization_machine)
- Expected gain: +0.0000
- Expected cost: 40.0s

**Config diff vs parent** (slot: `model`)

- params: {'k': 16, 'lr': 0.001} -> {'epochs': 40, 'k': 16, 'lr': 0.001}

**Result** (config_id=`bce19171850a`)

Validation:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6671 | 0.5358 | 0.6015 |
| 1 | 0.6674 | 0.5361 | 0.6018 |
| 2 | 0.6671 | 0.5351 | 0.6011 |

Backtest:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6593 | 0.5161 | 0.5877 |
| 1 | 0.6607 | 0.5165 | 0.5886 |
| 2 | 0.6583 | 0.5163 | 0.5873 |

wall_seconds: 0.0

**Verdict: ACCEPTED**

- delta: +0.0000 (n_seeds=3)
- ci95: [0.0000, 0.0000]
- backtest_delta: +0.00000
- reason: baseline_reproduce adopted as initial incumbent; no prior candidate to compare against

**Convergence check**: delta +0.00000 vs epsilon 0.002 — did NOT clear

---

## Node 2 (iteration 1)

**Hypothesis** (target_slot=`weighting`)

- Rationale: Train volume is heavily front-loaded — 278,835 rows on 20220411 decaying to ~20-24k/day by 20220418 — while validation is flat at 14-27k/day. Validation resembles the tail plateau, not the burst, so early training rows are drawn from a materially different regime. Downweighting them by recency should help the model fit the regime validation is actually drawn from.
- Citation: [koren2009temporal](https://dl.acm.org/doi/10.1145/1557019.1557072) (methods/library/recency_weighting.yaml#exponential_decay)
- Expected gain: +0.0080
- Expected cost: 45.0s

**Config diff vs parent** (slot: `weighting`)

- impl: 'none' -> 'exp_decay'; params: {} -> {'half_life_days': 5.0}

**Result** (config_id=`54794c9f54ca`)

Validation:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6675 | 0.5361 | 0.6018 |
| 1 | 0.6678 | 0.5361 | 0.6020 |
| 2 | 0.6672 | 0.5356 | 0.6014 |

wall_seconds: 0.0

**Verdict: REJECTED**

- delta: +0.0003 (n_seeds=3)
- ci95: [-0.0008, 0.0013]
- backtest_delta: n/a
- reason: ci_includes_zero

**Convergence check**: delta +0.00028 vs epsilon 0.002 — did NOT clear

---

## Node 3 (iteration 1)

**Hypothesis** (target_slot=`data_view`)

- Rationale: The blunt version of move 2: a hard cutoff so training only ever sees the plateau regime validation resembles, instead of continuously downweighting older rows. Kept separate from move 2 because a hard cutoff and a smooth decay can behave differently in practice — dropping rows outright also reduces user_id/video_id ID coverage, which the baseline's own diagnosis says carries most of the learnable signal, so this could underperform move 2 despite the same motivation.
- Citation: [kuairand_volume_shape_analysis](harness/HANDOFF.md) (methods/library/recency_window.yaml#hard_cutoff)
- Expected gain: +0.0050
- Expected cost: 35.0s

**Config diff vs parent** (slot: `data_view`)

- impl: 'full' -> 'recent_window'; params: {} -> {'days': 7}

**Result** (config_id=`a2cc8e05f859`)

Validation:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6543 | 0.5300 | 0.5922 |
| 1 | 0.6553 | 0.5307 | 0.5930 |
| 2 | 0.6543 | 0.5297 | 0.5920 |

wall_seconds: 0.0

**Verdict: REJECTED**

- delta: -0.0090 (n_seeds=3)
- ci95: [-0.0112, -0.0071]
- backtest_delta: n/a
- reason: ci_entirely_negative

**Convergence check**: delta -0.00904 vs epsilon 0.002 — did NOT clear

---

## Node 4 (iteration 1)

**Hypothesis** (target_slot=`objective`)

- Rationale: KuaiRand ships 12 feedback signals (is_click, is_like, is_follow, is_comment, is_forward, play_time_ms, ...) and only long_view is scored. click is denser than long_view (more positives per user), so an auxiliary click head shares gradient signal into the shared representation before the sparser long_view head has to do all the work alone.
- Citation: [ma2018esmm](https://dl.acm.org/doi/10.1145/3209978.3210104) (methods/library/multitask.yaml#esmm_click_longview)
- Expected gain: +0.0080
- Expected cost: 60.0s

**Error**: `contract` — NotImplementedError("executor.realize: no realization implemented for objective impl 'multitask_bce'")

**Recovery**: {'message': 'move 4 (objective/multitask_bce) failed with contract; run continues to the next candidate.', 'policy': 'skip_unimplemented'}

---

## Node 5 (iteration 1)

**Hypothesis** (target_slot=`model`)

- Rationale: Capacity ablation. The organizers' own sweep (k=8/16/32) already showed near-flat scores (0.5895/0.5902/0.5887) — the bottleneck is not capacity. This is a cheap confirmation, not a high-expectation move: kept to re-verify the finding still holds once earlier accepted changes have shifted the operating point, not because capacity is expected to matter now.
- Citation: [kuairand_capacity_ablation](harness/SCHEMA_NOTES.md) (methods/library/fm.yaml#capacity_k32)
- Expected gain: +0.0010
- Expected cost: 45.0s

**Config diff vs parent** (slot: `model`)

- params: {'epochs': 40, 'k': 16, 'lr': 0.001} -> {'epochs': 40, 'k': 32, 'lr': 0.001}

**Result** (config_id=`0700557629de`)

Validation:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6664 | 0.5354 | 0.6009 |
| 1 | 0.6682 | 0.5366 | 0.6024 |
| 2 | 0.6663 | 0.5358 | 0.6010 |

Backtest:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6602 | 0.5165 | 0.5884 |
| 1 | 0.6603 | 0.5166 | 0.5884 |
| 2 | 0.6593 | 0.5167 | 0.5880 |

wall_seconds: 4.0704226000234485 (served from cache — not real training time)

**Verdict: REJECTED**

- delta: +0.0000 (n_seeds=3)
- ci95: [-0.0007, 0.0007]
- backtest_delta: +0.00039
- reason: ci_includes_zero

**Convergence check**: delta +0.00002 vs epsilon 0.002 — did NOT clear

---

## Node 6 (iteration 1)

**Hypothesis** (target_slot=`calibration`)

- Rationale: The scored label long_view is a watch-time threshold, so it is mechanically entangled with video duration — which the baseline already encodes only crudely as dur_bucket. The organizers' own reference [4], Counterfactual Watch Time (KDD 2024), flags this duration bias explicitly and treats watch time as needing a duration-conditional (censored) correction rather than being modeled directly.
- Citation: [cwm_kdd2024](https://github.com/hyz20/CWM) (methods/library/cwm.yaml#duration_debias)
- Expected gain: +0.0100
- Expected cost: 50.0s

**Error**: `contract` — NotImplementedError("executor.realize: no realization implemented for calibration impl 'duration_debias_cwm'")

**Recovery**: {'message': 'move 6 (calibration/duration_debias_cwm) failed with contract; run continues to the next candidate.', 'policy': 'skip_unimplemented'}

---

## Node 7 (iteration 1)

**Hypothesis** (target_slot=`model`)

- Rationale: Swap FM for gradient boosting over the same 5 fields, testing whether non-linear tree splits capture interactions FM's bilinear form misses. The baseline's own diagnosis is that the user_id x video_id crossing already captures most learnable signal, so this is a genuine test of that ceiling rather than a guaranteed win.
- Citation: [ke2017lightgbm](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) (methods/library/lightgbm.yaml#gbdt_baseline)
- Expected gain: +0.0060
- Expected cost: 90.0s

**Error**: `contract` — NotImplementedError("executor.realize: no realization implemented for model impl 'lightgbm'")

**Recovery**: {'message': 'move 7 (model/lightgbm) failed with contract; run continues to the next candidate.', 'policy': 'skip_unimplemented'}

---

## Node 8 (iteration 1)

**Hypothesis** (target_slot=`objective`)

- Rationale: GAUC and nDCG@5 are both ranking metrics evaluated within a user's own impressions, but the baseline trains pointwise BCE, which optimizes calibrated probability rather than relative order. A pairwise objective directly optimizes what's measured — this is also the direction the organizers themselves rank as most likely to help, ahead of behavioural-sequence and multi-task directions.
- Citation: [rendle2009bpr](https://arxiv.org/abs/1205.2618) (methods/library/pairwise.yaml#bpr)
- Expected gain: +0.0120
- Expected cost: 55.0s

**Config diff vs parent** (slot: `objective`)

- impl: 'bce' -> 'bpr'; params: {} -> {'pairs_per_batch': 8192}

**Result** (config_id=`2665f53f972c`)

Validation:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6643 | 0.5349 | 0.5996 |
| 1 | 0.6616 | 0.5332 | 0.5974 |
| 2 | 0.6613 | 0.5335 | 0.5974 |

wall_seconds: 0.0

**Verdict: REJECTED**

- delta: -0.0033 (n_seeds=3)
- ci95: [-0.0050, -0.0015]
- backtest_delta: n/a
- reason: ci_entirely_negative

**Convergence check**: delta -0.00331 vs epsilon 0.002 — did NOT clear

---

## Node 9 (iteration 1)

**Hypothesis** (target_slot=`calibration`)

- Rationale: Item popularity alone already reaches primary 0.5807 on validation. Popularity is a marginal (video-only) statistic while FM's user_id x video_id crossing is joint — the two are not the same signal, so blending FM's score with a popularity prior is cheap (no retraining) and has a plausible mechanism, even though it's a lower-variance move than the structural changes above.
- Citation: [kuairand_item_popularity_baseline](vendor/kuairand-starter-kit/baseline.py) (methods/library/popularity_blend.yaml#item_popularity_prior)
- Expected gain: +0.0040
- Expected cost: 5.0s

**Error**: `contract` — NotImplementedError("executor.realize: no realization implemented for calibration impl 'popularity_blend'")

**Recovery**: {'message': 'move 9 (calibration/popularity_blend) failed with contract; run continues to the next candidate.', 'policy': 'skip_unimplemented'}

**Circuit breaker**: slot `calibration` blocked after 2 consecutive failures.

---

## Node 10 (iteration 1)

**Hypothesis** (target_slot=`model`)

- Rationale: Final hyperparameter sweep once the pipeline's structural shape is settled. The organizers' own choices already look well-tuned (published std 0.0008 across seeds is tight), so this is the lowest-expected-value move in the script. Deliberately last: if the structural moves above already found the real gains, self-terminating after three flat iterations here costs nothing worth having.
- Citation: [kuairand_fm_baseline_config](vendor/kuairand-starter-kit/baseline_scores.json) (methods/library/fm.yaml#hyperparameter_sweep)
- Expected gain: +0.0020
- Expected cost: 60.0s

**Config diff vs parent** (slot: `model`)

- params: {'epochs': 40, 'k': 32, 'lr': 0.001} -> {'epochs': 60, 'lr': 0.0005, 'patience': 6}

**Result** (config_id=`cac5cd2eb39e`)

Validation:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6670 | 0.5358 | 0.6014 |
| 1 | 0.6680 | 0.5361 | 0.6021 |
| 2 | 0.6679 | 0.5358 | 0.6019 |

Backtest:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6610 | 0.5165 | 0.5888 |
| 1 | 0.6605 | 0.5169 | 0.5887 |
| 2 | 0.6601 | 0.5167 | 0.5884 |

wall_seconds: 17.664408000186086 (served from cache — not real training time)

**Verdict: REJECTED**

- delta: +0.0003 (n_seeds=3)
- ci95: [-0.0003, 0.0010]
- backtest_delta: +0.00073
- reason: ci_includes_zero

**Convergence check**: delta +0.00034 vs epsilon 0.002 — did NOT clear

---

## Node 11 (iteration 2)

**Hypothesis** (target_slot=`ensemble`)

- Rationale: Every component-replacing move tried so far — 2 (recency_weight_exp), 3 (recency_window), 5 (fm k=32), 8 (pairwise_loss), and 10 (fm lr=0.0005/epochs=60) — was rejected by the noise gate against the move-1 baseline; 4, 6, 7, and 9 could not even be evaluated (unimplemented slots). Rather than keep swapping single components, try an ADDITIVE change instead: rank-average multiple independently seeded runs of the same baseline model. This does not change what the model learns, only how its variance across seeds is combined, so it is a genuinely different kind of move than anything else tried.
- Citation: [dietterich2000ensemble](https://link.springer.com/chapter/10.1007/3-540-45014-9_1) (methods/library/ensembling.yaml#rank_average)
- Expected gain: +0.0010
- Expected cost: 0.0s
- Predecessor evidence: 54794c9f54ca, a2cc8e05f859, 0700557629de, 2665f53f972c, cac5cd2eb39e

**Config diff vs parent** (slot: `ensemble`)

- impl: 'rank_avg_ensemble' (first change to this slot; baseline default shown for reference)

**Result** (config_id=`ens_rank3`)

Validation:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6680 | 0.5364 | 0.6022 |
| 1 | 0.6691 | 0.5366 | 0.6029 |
| 2 | 0.6676 | 0.5359 | 0.6018 |

Backtest:

| seed | GAUC | nDCG@5 | primary |
|---|---|---|---|
| 0 | 0.6604 | 0.5165 | 0.5885 |
| 1 | 0.6613 | 0.5167 | 0.5890 |
| 2 | 0.6609 | 0.5168 | 0.5889 |

wall_seconds: 0.0

**Verdict: ACCEPTED**

- delta: +0.0008 (n_seeds=3)
- ci95: [0.0003, 0.0014]
- backtest_delta: +0.00089
- reason: paired CI excludes zero (n=3 seeds, delta=+0.00083) and backtest confirms (backtest_delta=+0.00089)

**Convergence check**: delta +0.00083 vs epsilon 0.002 — did NOT clear

---
