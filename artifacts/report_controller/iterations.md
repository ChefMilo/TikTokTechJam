# Iterations

_`node` counts every evaluation attempted; `iteration` counts committed revisions and only advances when a DECISION accepts. Several consecutive nodes sharing one iteration number means several consecutive rejections, not a stall — see executor/journal.py's log_decision docstring._

## Errors and recovery

| node | error_class | policy | recovered |
|---|---|---|---|
| 3 | contract | n/a | yes |
| 6 | generator_exhausted | n/a | yes |

2 candidate(s) failed; the run continued past every one.

## Node 1 (iteration 0)

**Result** (config_id=`bce19171850a`)

status: ok
primary: 0.6014

wall_seconds: 2.1850921000004746
gpu_seconds: 0.0
tokens: 0

**Verdict: ACCEPTED**

- delta: +0.0000 (n_seeds=None)
- ci95: [0.0000, 0.0000]
- backtest_delta: +0.00000
- reason: first candidate adopted as incumbent; nothing to compare against

**Convergence check**: not converged — iterations_considered=1/3, epsilon=0.002, recent_deltas=[none yet]

---

## Node 2 (iteration 1)

**Hypothesis** (target_slot=`model`)

- Rationale: Reproduce the organizers' own published FM baseline exactly (k=16, lr=0.001), to confirm the harness end-to-end — data loading, encoding, training, evaluation — reproduces validation primary 0.6016 before any structural change is judged against it. This is the control, not a candidate expected to win.
- Citation: [rendle2010fm](https://ieeexplore.ieee.org/document/5694074) (methods/library/fm.yaml#factorization_machine)
- Expected gain: +0.0000
- Expected cost: 40.0s

**Result** (config_id=`bce19171850a`)

status: ok
primary: 0.6014

wall_seconds: 7.451879300002474
gpu_seconds: 0.0
tokens: 0

**Verdict: REJECTED**

- delta: +0.0000 (n_seeds=None)
- ci95: [0.0000, 0.0000]
- backtest_delta: +0.00000
- reason: ci_includes_zero

---

## Node 3 (iteration 1)

**Hypothesis** (target_slot=`objective`)

- Rationale: KuaiRand ships 12 feedback signals (is_click, is_like, is_follow, is_comment, is_forward, play_time_ms, ...) and only long_view is scored. click is denser than long_view (more positives per user), so an auxiliary click head shares gradient signal into the shared representation before the sparser long_view head has to do all the work alone.
- Citation: [ma2018esmm](https://dl.acm.org/doi/10.1145/3209978.3210104) (methods/library/multitask.yaml#esmm_click_longview)
- Expected gain: +0.0080
- Expected cost: 60.0s

**Result** (config_id=`de3e33e391bd`)

status: failed
primary: n/a

wall_seconds: 11.43701960000908
gpu_seconds: 0.0
tokens: 0

**Error**: `contract` — NotImplementedError("executor.realize: no realization implemented for objective impl 'multitask_bce'")

---

## Node 4 (iteration 1)

**Hypothesis** (target_slot=`weighting`)

- Rationale: Train volume is heavily front-loaded — 278,835 rows on 20220411 decaying to ~20-24k/day by 20220418 — while validation is flat at 14-27k/day. Validation resembles the tail plateau, not the burst, so early training rows are drawn from a materially different regime. Downweighting them by recency should help the model fit the regime validation is actually drawn from.
- Citation: [koren2009temporal](https://dl.acm.org/doi/10.1145/1557019.1557072) (methods/library/recency_weighting.yaml#exponential_decay)
- Expected gain: +0.0080
- Expected cost: 45.0s

**Result** (config_id=`54794c9f54ca`)

status: ok
primary: 0.6017

wall_seconds: 7.836777799995616
gpu_seconds: 0.0
tokens: 0

**Verdict: REJECTED**

- delta: +0.0003 (n_seeds=None)
- ci95: [-0.0008, 0.0013]
- backtest_delta: +0.00003
- reason: ci_includes_zero

---

## Node 5 (iteration 1)

**Hypothesis** (target_slot=`model`)

- Rationale: Capacity ablation. The organizers' own sweep (k=8/16/32) already showed near-flat scores (0.5895/0.5902/0.5887) — the bottleneck is not capacity. This is a cheap confirmation, not a high-expectation move: kept to re-verify the finding still holds once earlier accepted changes have shifted the operating point, not because capacity is expected to matter now.
- Citation: [kuairand_capacity_ablation](harness/SCHEMA_NOTES.md) (methods/library/fm.yaml#capacity_k32)
- Expected gain: +0.0010
- Expected cost: 45.0s

**Result** (config_id=`0700557629de`)

status: ok
primary: 0.6015

wall_seconds: 4.100457700027619
gpu_seconds: 0.0
tokens: 0

**Verdict: REJECTED**

- delta: +0.0000 (n_seeds=None)
- ci95: [-0.0007, 0.0007]
- backtest_delta: +0.00039
- reason: ci_includes_zero

---

## Node 6 (iteration 1)

**Hypothesis** (target_slot=`objective`)

- Rationale: GAUC and nDCG@5 are both ranking metrics evaluated within a user's own impressions, but the baseline trains pointwise BCE, which optimizes calibrated probability rather than relative order. A pairwise objective directly optimizes what's measured — this is also the direction the organizers themselves rank as most likely to help, ahead of behavioural-sequence and multi-task directions.
- Citation: [rendle2009bpr](https://arxiv.org/abs/1205.2618) (methods/library/pairwise.yaml#bpr)
- Expected gain: +0.0120
- Expected cost: 55.0s

**Result** (config_id=`2665f53f972c`)

status: ok
primary: 0.5981

wall_seconds: 7.692653200007044
gpu_seconds: 0.0
tokens: 0

**Verdict: REJECTED**

- delta: -0.0033 (n_seeds=None)
- ci95: [-0.0050, -0.0015]
- backtest_delta: -0.00352
- reason: ci_entirely_negative

**Error**: `generator_exhausted` — the scripted move catalog has 1 move(s) for slot 'weighting' and all of them have been proposed; no further hypothesis to offer for this slot

---
