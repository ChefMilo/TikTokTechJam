# SCHEMA_NOTES.md — vendor/kuairand-starter-kit inspection

Read-only inspection of the organizer starter kit unzipped into
`vendor/kuairand-starter-kit/`. This document is descriptive only — no
implementation code was written to produce it. All line numbers refer to
the files as they currently exist in `vendor/`; the kit is read-only and
must never be edited, only wrapped.

---

## TL;DR FOR THE TEAM

```
Metrics the CODE computes : GAUC, nDCG@5   (NOT NDCG@10/Recall@50 — see evaluate.py Q2)
Primary-score formula     : primary = (GAUC + nDCG@5) / 2.0
load() signature          : load(data_dir) -> dict[str, list[tuple]]   (data.py:12)
Submission CSV header     : row_id,user_id,video_id,score              (submit.py:25)
```

---

## evaluate.py

Full path: `vendor/kuairand-starter-kit/evaluate.py`. Module docstring
(lines 1-12) states this is the official scoring script and that its
conventions ("口径") are frozen — "写死在这里，不要改" ("hardcoded here,
do not change").

**1. Exact signature of the scoring function(s). What does it return?**

Three functions, one of which is the actual entry point:

- `def auc(labels, scores):` (line 15) — returns a single `float` (Mann-Whitney U based AUC).
- `def ndcg_at_k(labels, k):` (line 35) — returns a single `float`.
- `def evaluate(user_ids, labels, scores, k=5):` (line 43) — the entry point. Returns a **dict**, built at lines 60-61:

  ```python
  return {'GAUC': gauc, f'nDCG@{k}': ndcg, 'primary': (gauc + ndcg) / 2.0,
          'users': len(byu), 'rows': len(labels)}
  ```

  Keys: `GAUC` (float), `nDCG@5` (float, key name depends on `k`), `primary` (float), `users` (int, number of distinct users), `rows` (int, `len(labels)`).

**2. WHICH METRICS does it actually compute?**

The code computes **GAUC and nDCG@5** — matching the kit's own README/JSON description, **not** the "NDCG@10 / Recall@50" pairing mentioned in the challenge prose. Evidence:

- Module docstring line 6: `指标 : GAUC, nDCG@5 (主分 = 两者的平均)` — "Metrics: GAUC, nDCG@5 (primary = mean of both)".
- `evaluate()`'s default argument is `k=5` (line 43), and the dict key is built as `f'nDCG@{k}'` (line 60) — so with the default call signature the key literally comes out as `'nDCG@5'`.
- There is no Recall@50 computation anywhere in this file, or anywhere else in `vendor/`.
- `README.md` line 34 and `baseline_scores.json` lines 10-13 both independently confirm `["GAUC", "nDCG@5"]`.

**This is the single most load-bearing fact in this document: build the harness against GAUC/nDCG@5, not NDCG@10/Recall@50.**

**3. Is a "primary" score computed inside it, or must we compute it?**

Computed inside, at line 60: `'primary': (gauc + ndcg) / 2.0` — an unweighted arithmetic mean of GAUC and nDCG@5. Nothing to recompute; consume `result['primary']` directly.

**4. How are users with zero positive labels handled?**

Differently per metric — this is easy to misread, since it means the same user can be excluded from one metric and included (at a fixed value) in the other:

- **nDCG**: every user contributes to the average, unconditionally. Line 57: `nd.append(ndcg_at_k(labs, k))` runs inside the `for u, lst in byu.items():` loop for *all* users, with no `npos` filter. `ndcg_at_k` returns `0.0` when `idcg == 0` (line 41), which is exactly the all-negative-labels case — so a zero-positive user gets nDCG **0.0** and is kept in the mean (`sum(nd) / len(nd)`, line 59).
- **GAUC**: zero-positive users are **excluded** entirely. Line 54: `if 0 < npos < len(labs):` gates the only place `gnum`/`gden` are incremented (lines 55-56) — a user with `npos == 0` never contributes to either the numerator or denominator.

**5. For GAUC: filter for `0 < positives < impressions`? Per-user weighting?**

Yes, exactly that filter, at line 54: `if 0 < npos < len(labs):` — this excludes both all-negative (`npos == 0`) and all-positive (`npos == len(labs)`, i.e. zero negatives) users from GAUC, since AUC is undefined without both classes present.

Weighting (lines 55-56):

```python
gnum += npos * auc(labs, [s for s, _ in lst])
gden += npos
```

GAUC is a **weighted mean of per-user AUC, weighted by each user's number of positives (`npos`)** — not weighted by impression count. Final value: `gnum / gden` (line 58), falling back to `0.5` if `gden == 0` (no qualifying users at all).

**6. nDCG gain formula and discount?**

- Gain (lines 38, 40): `(2 ** t) - 1` for label `t`. Docstring line 10 notes this is "等价于 identity" ("equivalent to identity") under binary labels, since `2**0 - 1 == 0` and `2**1 - 1 == 1`.
- Discount (line 37): `disc = [math.log2(i + 2) for i in range(k)]` — standard `log2(rank + 1)` discount with `i` 0-indexed (position `i=0` → `log2(2)=1`, i.e. no discount at rank 1).
- IDCG (line 39-40) is computed from `sorted(labels, reverse=True)[:k]`, i.e. the ideal ranking truncated to `k`.
- `ndcg_at_k` documents at line 36 that `labels` must already be sorted by descending predicted score by the caller — `evaluate()` does this sort itself at line 51 (`lst.sort(key=lambda x: -x[0])`) before calling it.

---

## data.py (the data loader)

Full path: `vendor/kuairand-starter-kit/data.py`. Module docstring (line 1): "KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。" ("data loading + official split + feature encoding. Only depends on stdlib and numpy.")

Module-level constants:
- `LABEL = 'long_view'` (line 5)
- `SPLITS = {'train': (20220408, 20220421), 'valid': (20220422, 20220428), 'test': (20220429, 20220508)}` (lines 6-8)
- `FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']` (line 10), with a comment: "5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。" ("5 feature fields. Add features here if you want to — this is one of the places students should most be touching.")

**7. Exact signature and return type/shape of `load()`.**

`def load(data_dir):` (line 12) — a single positional argument, no keyword arguments, no defaults.

Return type: `dict[str, list[tuple]]`, keyed by split name (`'train'`/`'valid'`/`'test'`). Each element of a split's list is the 7-tuple built at lines 23-25:

```python
(int(r['date']), r['user_id'], r['video_id'],
 vid2author.get(r['video_id'], 'UNK'), r['tab'],
 float(r['duration_ms']), 1 if r[LABEL] != '0' else 0)
```

**8. Which columns come back, with dtypes?**

Positional tuple fields (there are no column names at runtime — this is a plain list of tuples, not a DataFrame):

| index | field | dtype (as coerced in code) |
|---|---|---|
| 0 | `date` (YYYYMMDD) | `int` |
| 1 | `user_id` | `str` |
| 2 | `video_id` | `str` |
| 3 | `author_id` | `str` (looked up via `vid2author`, default `'UNK'` if the video is missing from `video_features_basic_pure.csv`) |
| 4 | `tab` | `str` |
| 5 | `duration_ms` | `float` |
| 6 | `long_view` label | `int`, 0/1 (line 25: `1 if r[LABEL] != '0' else 0`) |

**9. Is `row_id` generated by the loader, 0-based and contiguous?**

`row_id` is **not** produced by `load()` at all — there is no such field in the tuple above. It's a submission-file concept, generated later by `submit.py`'s `write_submission()` via `for i, (x, s) in enumerate(zip(rows, scores)):` (submit.py line 31) — i.e. it's just the positional index into `load(data_dir)[split]`, which is 0-based and contiguous by construction (`enumerate` from 0). `submit.py`'s own docstring (lines 6-8) and `README.md` (lines 90) both state this correspondence explicitly and note it is deterministic because `load()` reads the two log files in a fixed order and preserves each file's row order after date filtering (see Q10).

**10. Does the loader filter by date itself, or leave splitting to the caller? Hardcoded boundaries?**

`load()` does the splitting itself — the caller gets pre-split data. Boundaries are the hardcoded `SPLITS` dict (lines 6-8):

```
train : 20220408–20220421
valid : 20220422–20220428
test  : 20220429–20220508
```

Applied at lines 27-29:

```python
out = {}
for name, (lo, hi) in SPLITS.items():
    out[name] = [x for x in rows if lo <= x[0] <= hi]
return out
```

Because this is a list comprehension over `rows` (which itself was built by appending rows file-by-file, in-file-order — lines 19-25), the relative order of rows within a split is preserved from the source CSVs. This is exactly the ordering `submit.py` relies on for `row_id` alignment.

**11. Are user-side and video-side feature tables loaded here, separately, or not at all?**

- **Video-side**: partially loaded, in `load()` itself (lines 14-17) — but *only* to build a `video_id -> author_id` lookup (`vid2author`). No other video-side columns (e.g. `music_id`, `video_type`, `upload_type`) are read by `data.py`.
- **User-side**: **not loaded at all** by `data.py`. `user_features_pure.csv` never appears in this file. It is only read separately, in the standalone ablation script `ablation_features.py` (lines 17-18) — see that file's section below.

**12. Row counts per split, if asserted/documented anywhere.**

UNKNOWN — not determinable from the kit as exact row counts. No `.py` file in `vendor/` asserts or hardcodes per-split row counts; `baseline.py` line 112 only *prints* `{k_: len(v) for k_, v in splits.items()}` at runtime, it doesn't assert against a known value. The only documented count is the **test split's user count**, 23,875, stated in `README.md` line 53 and `baseline_scores.json` line 92 (`test_set_composition.users`). `README.md` line 120 mentions "114 万行数据" (~1.14M rows) in passing while explaining a negative ablation result, but this reads as an approximate aside about training data, not an asserted/documented row count for any specific split.

---

## baseline.py

Full path: `vendor/kuairand-starter-kit/baseline.py`. Module docstring (lines 1-6) lists three baselines: `pop` (official baseline, pure statistics, no training), `fm` (starting model, "students should build up from here"), `random` (lower bound, self-check that the eval code isn't broken).

**13. Exactly which 5 categorical fields does the FM use?**

The same `FIELDS` constant imported from `data.py` (baseline.py line 9: `from data import load, encode, FIELDS`):

```
user_id, video_id, author_id, tab, dur_bucket
```

`dur_bucket` isn't a raw CSV column — it's derived inside `encode()` (data.py lines 32-42) by quantile-bucketing `duration_ms` into 10 buckets using edges fit on the **train** split only (`_bucket_edges`, data.py line 33). Cross-checked against `baseline_scores.json` lines 68-74 (`config.fields`), which lists the identical 5 names.

**14. Is the random seed settable from the CLI or only in code? How do we run 5 seeds precisely?**

Settable from the CLI via `--seed` (line 108: `ap.add_argument('--seed', type=int, default=0)`), wired to both models that use randomness (line 113-114):

```python
res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
       'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)}[a.model](splits)
```

There is **no** flag to sweep multiple seeds in one process — `run_fm`'s signature is `def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):` (line 75), single `seed` in, single result dict out. To reproduce the kit's "5 seeds" numbers you must invoke the CLI **five separate times**, once per seed value:

```bash
python3 baseline.py --model fm --seed 0
python3 baseline.py --model fm --seed 1
python3 baseline.py --model fm --seed 2
python3 baseline.py --model fm --seed 3
python3 baseline.py --model fm --seed 4
```

(`README.md` line 72 and `baseline_scores.json` lines 56-59/31 corroborate that the published std/mean figures are over seeds 0-4.)

**15. What hyperparameters are exposed as CLI flags?**

From the `argparse` block (lines 101-109):

| flag | type | default |
|---|---|---|
| `--data_dir` | str | `'./KuaiRand-Pure/data'` |
| `--model` | choice | `'fm'` (choices: `pop`, `fm`, `random`) |
| `--k` | int | `16` |
| `--lr` | float | `0.001` |
| `--epochs` | int | `40` |
| `--seed` | int | `0` |

**Not** exposed via CLI, only settable by calling `run_fm`/`run_pop` directly in code: `bs` (batch size, default `8192`), `patience` (default `4`), and `run_pop`'s `prior` (default `20.0`, line 15).

**16. Where does it write its output, and in what format?**

Nowhere — `baseline.py` writes **no files**. It only prints to stdout: per-epoch progress (lines 85-87) and a final per-split summary (lines 116-118). Writing an actual submission file is `submit.py`'s job (below), which internally reuses `baseline.FM` but not `baseline.run_fm`.

---

## submit.py

Full path: `vendor/kuairand-starter-kit/submit.py`. Module docstring (lines 1-20) defines the submission format and explains the reason for `row_id` (see Q9 above): `(user_id, video_id)` pairs are **not unique** in the eval set — "test 集有 3.06% 的重复对，最多重复 12 次" ("3.06% of pairs in the test set are duplicated, up to 12 times") — so they cannot serve as a primary key; only `row_id` (positional index) can.

**17. What exactly does `--check` validate? Enumerate every check.**

`--check` calls `read_submission(path, rows)` (defined lines 34-62) with no additional scoring. Checks, in the order they're raised:

1. **Header** must equal `HEADER = ['row_id', 'user_id', 'video_id', 'score']` exactly (line 39).
2. **Field count** — every data row must have exactly 4 fields (line 43-44).
3. **`row_id` contiguity** — must equal the running 0-based counter `n` for that row; i.e. strictly `0, 1, 2, …` with no gaps or reordering (line 46-47).
4. **No overflow** — submission cannot have more rows than the evaluation set (line 48-49: `if n >= len(rows): raise ...`).
5. **Alignment** — `user_id` and `video_id` on each row must exactly match `rows[n][1]` / `rows[n][2]` from `load(data_dir)[split]` at the same position (line 50-52).
6. **Score parses as a number** — `float(sc)` must not raise (line 53-56).
7. **Score is finite** — rejects `NaN` and both `Inf`/`-Inf` (line 57-58: `if v != v or v in (float('inf'), float('-inf')): raise ...`).
8. **Exact row count** — after reading everything, total rows read must equal `len(rows)` exactly, not just "not more than" (line 60-61).

**18. What does `--make` produce?**

Trains a fresh FM model **inline** (lines 78-99) — this duplicates (rather than calls) the training loop shape of `baseline.run_fm`, hardcoded to `k=16, lr=0.001, seed=0` (line 85), up to 40 epochs with early-stopping patience 4 on validation `primary` (lines 88-96) — then predicts on `enc[a.split]` (default split `test`) and writes the result via `write_submission()` (line 98). Note: `--make`'s seed is **not** configurable via `submit.py`'s CLI (no `--seed` flag exists there, lines 65-73) — it is always the fixed baseline config, seed 0.

**19. Exact expected CSV header and column order.**

```
HEADER = ['row_id', 'user_id', 'video_id', 'score']   # submit.py:25
```

Confirmed by the worked example in `README.md` (lines 81-86):

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
```

Score is written with `f"{float(s):.6g}"` (submit.py line 32) — 6 significant figures, general float format. No other columns are permitted (`--check`'s field-count and header checks above enforce exactly these 4, in this order).

---

## Everything else in vendor/

**20. Every other file, one line each:**

| file | description |
|---|---|
| `vendor/.gitkeep` | Placeholder that kept the (then-empty) `vendor/` directory tracked in git before the kit was unzipped into it. |
| `vendor/kuairand-starter-kit/README.md` | Organizer's usage guide: setup/deps, task definition, baseline ladder + oracle ceiling analysis, convergence rule, submission format, and a prioritized "what to try next" list for the team. |
| `vendor/kuairand-starter-kit/baseline_scores.json` | Machine-readable snapshot of the task config, split boundaries, metric names, the convergence rule (`epsilon=0.002, N=3`), and baseline/oracle scores with cross-seed std. |
| `vendor/kuairand-starter-kit/ablation_features.py` | Standalone script (not imported by `baseline.py`/`submit.py`) that re-implements feature loading with 3 modes (5-field base / +item-side / +CWM's 13 fields) to reproduce the organizers' "adding static features doesn't help" ablation result. |

(`evaluate.py`, `data.py`, `baseline.py`, `submit.py` are covered in their own sections above.)

**21. Hardcoded paths, raw-CSV assumptions, and what we must download separately.**

Default data directory, repeated identically in three places:

- `baseline.py` line 102-103: `ap.add_argument('--data_dir', default='./KuaiRand-Pure/data', ...)`
- `submit.py` line 67: `ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')`
- `ablation_features.py` line 8: `D = sys.argv[1] if len(sys.argv) > 1 else './KuaiRand-Pure/data'` (a positional CLI arg with this fallback, not argparse)

Raw CSV files the code expects inside that directory:

| file | referenced by | notes |
|---|---|---|
| `video_features_basic_pure.csv` | `data.py:15`, `ablation_features.py:20` | `data.py` only pulls `author_id` from it |
| `log_standard_4_08_to_4_21_pure.csv` | `data.py:20`, `ablation_features.py:25` | train-window interaction log |
| `log_standard_4_22_to_5_08_pure.csv` | `data.py:20`, `ablation_features.py:25` | valid+test-window interaction log |
| `user_features_pure.csv` | `ablation_features.py:17` only | **not** read by `data.py`/`baseline.py`/`submit.py` — ablation-only |
| `log_random_4_22_to_5_08_pure.csv` | mentioned only in `README.md:141` | unbiased random-exposure log (~1.18M rows per README); **not read by any `.py` file in the kit** — proposed as an optional future unbiased-validation set |

Download source (`README.md` lines 9-15): Zenodo, no registration required —

```
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

...which unpacks to `./KuaiRand-Pure/` (containing a `data/` subdirectory matching the hardcoded default above). **This tarball is not in `vendor/` and must be downloaded separately** by whoever runs the harness; it does not ship with the starter kit.

Dependencies: `README.md` lines 3-5 state the kit needs only **Python 3.9+ and numpy** — explicitly "没有别的" ("nothing else") and "不需要 torch、pandas、sklearn" ("no need for torch, pandas, sklearn"). This is a statement about the vendored kit itself; it doesn't constrain what our own `harness/` package uses (our `requirements.txt` already includes pandas/scipy/pytest).

---

## OPEN QUESTIONS FOR ORGANIZERS

- **Per-split row counts** are never asserted or printed anywhere in the kit source — only the test split's *user* count (23,875) is documented. We have no ground-truth row counts to assert our own loader against once we download the data.
- **`bs` (batch size) and `patience`** for `run_fm` are not exposed as CLI flags on `baseline.py`, unlike `k`/`lr`/`epochs`/`seed`. Was this intentional (meant to be swept only via direct calls to `run_fm`), or an oversight?
- **`submit.py --make`** always hardcodes `seed=0` for the FM baseline it trains internally, with no `--seed` flag of its own. Is that seed choice meaningful/official, or arbitrary/incidental?
- **GAUC weighting is by number of positives (`npos`) per user**, not by impression count. The README states this as fixed convention but doesn't explain the rationale — worth confirming this is the intended definition and not, e.g., a simplification versus the literature's impression-weighted GAUC.
- **`log_random_4_22_to_5_08_pure.csv`** (the unbiased exposure log) is proposed in the README as an advanced/optional direction but is never loaded by any shipped `.py` file. Is it officially in-scope for evaluation/leaderboard purposes, or purely exploratory for our own use?
