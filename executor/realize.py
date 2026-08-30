"""Turns a PipelineConfig into trained-model predictions.

SCOPE DISCIPLINE: this is the minimum spine to get real candidates
running end to end, not the full executor. No sandbox, no subprocess
isolation, no error taxonomy — see EXECUTOR_SURVEY.md for what comes
after this. Realized today: `model.impl == "fm"`, `data_view.impl in
{"full", "recent_window"}`, `weighting.impl in {"none", "exp_decay"}`,
`objective.impl in {"bce", "bpr"}` (not combined with a non-"none"
weighting — see realize()'s dispatch). Everything else (features,
calibration beyond its default, and any other model) raises
NotImplementedError naming the slot and impl — that is correct
behaviour, not a gap to silently paper over: nothing else has a
realization yet.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np

from contracts import PipelineConfig, SlotConfig, SlotName

REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_DIR = REPO_ROOT / "vendor" / "kuairand-starter-kit"

DEFAULT_SLOTS: dict[SlotName, SlotConfig] = {
    "data_view": SlotConfig(impl="full"),
    "features": SlotConfig(impl="baseline_5"),
    "weighting": SlotConfig(impl="none"),
    "model": SlotConfig(impl="fm", params={"k": 16, "lr": 0.001}),
    "objective": SlotConfig(impl="bce"),
    "calibration": SlotConfig(impl="none"),
}
"""A complete six-slot default reproducing the vendor's own FM baseline.

A hypothesis generator move (see methods/scripted.py) only ever proposes
a fragment for ONE slot; this fills in the other five so there is always
a complete, hashable PipelineConfig to run.
"""


def build_config(fragment: SlotConfig, target_slot: SlotName, seed: int) -> PipelineConfig:
    """Overlays `fragment` onto DEFAULT_SLOTS at `target_slot`. Every
    other slot keeps its default.
    """
    slots = dict(DEFAULT_SLOTS)
    slots[target_slot] = fragment
    return PipelineConfig(slots=slots, seed=seed)


def _load_vendor_baseline():
    """Imports vendor/kuairand-starter-kit/baseline.py by file path — same
    approach as harness/data.py, harness/metrics.py, and
    tests/test_rungs.py's baseline import. baseline.py's own top-level
    `from data import ...` / `from evaluate import ...` rely on Python's
    normal sys.path search, so the vendor directory is put on sys.path
    just long enough to exec it.
    """
    vendor_dir_str = str(_VENDOR_DIR)
    sys.path.insert(0, vendor_dir_str)
    try:
        spec = importlib.util.spec_from_file_location(
            "_vendor_kuairand_baseline", _VENDOR_DIR / "baseline.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(vendor_dir_str)
    return module


_vendor = _load_vendor_baseline()


def _yyyymmdd_to_date(value: int) -> dt.date:
    """Rows carry dates as plain YYYYMMDD ints (see harness/data.py's row
    shape). Naive integer arithmetic on that int is wrong across a month
    boundary (20220401 - 3 is not 20220329), so every date computation
    below goes through a real dt.date instead.
    """
    return dt.datetime.strptime(str(value), "%Y%m%d").date()


def _apply_recency_window(fit_rows: list[tuple], params: dict[str, Any]) -> list[tuple]:
    """data_view impl "recent_window": keeps only the last `days`
    calendar days of `fit_rows`, by each row's own date field (index 0)
    relative to the max date PRESENT IN fit_rows (not "today" — this runs
    on both the real train window and the backtest's earlier fit window,
    and each has its own "most recent day"). `score_rows` is never
    touched — only what the model trains on is windowed.

    "Last N days" is inclusive of the max date itself: N=7 keeps
    [max_date - 6 days, max_date], seven calendar days total.
    """
    days = params.get("days", 7)
    if not fit_rows:
        return fit_rows
    max_date = _yyyymmdd_to_date(max(row[0] for row in fit_rows))
    cutoff = max_date - dt.timedelta(days=days - 1)
    return [row for row in fit_rows if _yyyymmdd_to_date(row[0]) >= cutoff]


def _compute_exp_decay_weights(fit_rows: list[tuple], params: dict[str, Any]) -> np.ndarray:
    """weighting impl "exp_decay": per-row weight
    0.5 ** (days_before / half_life_days), where days_before is the
    number of calendar days between the row's own date and the max date
    in `fit_rows` (0 for rows on the most recent day, growing for older
    ones). Returned in the SAME order as `fit_rows`, so it lines up
    index-for-index with whatever encode() builds from it.
    """
    half_life_days = params.get("half_life_days", 5.0)
    if not fit_rows:
        return np.array([], dtype=np.float32)
    max_date = _yyyymmdd_to_date(max(row[0] for row in fit_rows))
    days_before = np.array(
        [(max_date - _yyyymmdd_to_date(row[0])).days for row in fit_rows], dtype=np.float64
    )
    weights = 0.5 ** (days_before / half_life_days)
    return weights.astype(np.float32)


class _WeightedFM(_vendor.FM):
    """The vendor's FM, with `step` accepting a per-sample weight array.

    vendor/kuairand-starter-kit/baseline.py is read-only, so this
    subclasses it rather than reimplementing FM from scratch: `__init__`,
    `logits`, and `predict` are all inherited unchanged, along with every
    attribute (self.V, self.W, self.b, the Adam moment buffers, self.t).
    Only `step` is overridden.

    EXACTLY WHAT CHANGES, and why (this is the part most likely to be
    subtly wrong, so it's spelled out in full). The vendor's own step():

        g = ((sigmoid(z) - y) / B).astype(np.float32)

    is the gradient of the logistic loss w.r.t. each sample's own logit,
    already divided by the batch size B — i.e. this IS the batch's mean
    reduction, computed per-sample before `np.add.at` accumulates each
    sample's contribution into gW/gV. Weighting has to happen to `g`
    itself, before that accumulation, not to the accumulation — a sample
    that already turned into a gW/gV update can't be reweighted
    afterwards without redoing the update. This becomes:

        g = ((sigmoid(z) - y) * weights / weights.sum()).astype(np.float32)

    Normalizing by `weights.sum()` rather than by B is the correct
    generalization of a MEAN to a WEIGHTED MEAN: for weighted empirical
    risk minimization, dL/dz_i = w_i * (sigmoid(z_i) - y_i) / sum(w).
    It's also what makes the required sanity check hold for the RIGHT
    reason: when every weight is ~1.0 (half_life_days set absurdly
    high), weights.sum() ~= B and this reduces to the vendor's own
    unweighted `g` exactly. Keeping the vendor's `/ B` while just
    multiplying by `weights` would ALSO pass that one sanity check (since
    weights~1 makes the two normalizers coincide), but would be the WRONG
    generalization for any other weight distribution — it would entangle
    the weighting scheme's absolute scale with the effective learning
    rate instead of computing a proper weighted average.

    Everything downstream of `g` — the Adam update, `self.b`'s update via
    `g.sum()` — is untouched, since weighting only changes what "the
    mean" means, not how the resulting gradient is applied. The returned
    loss is the weighted mean BCE, the same generalization applied to the
    vendor's `-np.mean(...)`.
    """

    def step(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        z, E, S = self.logits(X)
        weight_sum = weights.sum()
        g = ((_vendor.sigmoid(z) - y) * weights / weight_sum).astype(np.float32)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        probs = _vendor.sigmoid(z)
        weighted_bce = weights * (y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9))
        return float(-weighted_bce.sum() / weight_sum)


class _BprPairIndex(NamedTuple):
    """Groups encoded row indices by (user, label==1/0) so BPR pairs can
    be sampled without a per-pair Python loop. Built once per fit window,
    reused across every training step.
    """

    pos_flat_idx: np.ndarray
    pos_offsets: np.ndarray
    pos_counts: np.ndarray
    neg_flat_idx: np.ndarray
    neg_offsets: np.ndarray
    neg_counts: np.ndarray
    eligible_codes: np.ndarray


def _build_bpr_pair_index(users: np.ndarray, labels: np.ndarray) -> _BprPairIndex:
    """A user contributes pairs only if it has at least one positive AND
    at least one negative row — the SAME eligibility rule GAUC itself
    uses (0 < npos < n_impressions; see harness/gate.py's
    _per_user_metrics), so a user GAUC would ignore anyway contributes no
    pairs here either, which is correct rather than an oversight.

    `pos_counts[c] * neg_counts[c]` is the number of DISTINCT valid pairs
    user-code `c` could contribute (1 positive x 3 negatives = 3 pairs,
    for example) — this module doesn't enumerate them (infeasible at
    real dataset scale), it samples from that space instead, but the
    count itself is exactly this product.
    """
    unique_users, user_codes = np.unique(users, return_inverse=True)
    n_users = len(unique_users)

    def _flat(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_idx = np.flatnonzero(mask)
        codes = user_codes[row_idx]
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes[order], minlength=n_users)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        return row_idx[order], offsets, counts

    pos_flat_idx, pos_offsets, pos_counts = _flat(labels == 1)
    neg_flat_idx, neg_offsets, neg_counts = _flat(labels == 0)
    eligible_codes = np.flatnonzero((pos_counts > 0) & (neg_counts > 0))

    return _BprPairIndex(
        pos_flat_idx=pos_flat_idx,
        pos_offsets=pos_offsets,
        pos_counts=pos_counts,
        neg_flat_idx=neg_flat_idx,
        neg_offsets=neg_offsets,
        neg_counts=neg_counts,
        eligible_codes=eligible_codes,
    )


def _sample_bpr_pairs(
    index: _BprPairIndex, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized: draws `n` (positive_row_idx, negative_row_idx) pairs,
    each independently uniform over (an eligible user, one of its
    positives, one of its negatives). Not a per-pair Python loop —
    `pairs_per_batch` can be in the thousands and this runs once per
    training step, so a loop here would dominate wall time.
    """
    if len(index.eligible_codes) == 0:
        raise ValueError(
            "no user has both a positive and a negative row in this fit "
            "window; cannot sample BPR pairs"
        )
    chosen = index.eligible_codes[rng.integers(0, len(index.eligible_codes), size=n)]
    pos_within = (rng.random(n) * index.pos_counts[chosen]).astype(np.int64)
    neg_within = (rng.random(n) * index.neg_counts[chosen]).astype(np.int64)
    pos_row_idx = index.pos_flat_idx[index.pos_offsets[chosen] + pos_within]
    neg_row_idx = index.neg_flat_idx[index.neg_offsets[chosen] + neg_within]
    return pos_row_idx, neg_row_idx


class _PairwiseFM(_vendor.FM):
    """The vendor's FM trained with a BPR pairwise objective instead of
    pointwise BCE. Same subclassing approach as _WeightedFM above:
    `__init__`, `logits`, and `predict` are inherited unchanged; only
    `step` is overridden, and its signature changes to match what a
    pairwise step actually needs — a batch of positive rows and a batch
    of negative rows, not (X, y).

    THE GRADIENT. For one pair (i, j), BPR loss is
    L = -log(sigmoid(z_i - z_j)). Writing s = sigmoid(z_i - z_j):

        dL/dz_i = s - 1
        dL/dz_j = 1 - s

    (chain rule through z_i - z_j, coefficient +1 for z_i and -1 for
    z_j). Positive and negative rows go through ONE combined
    self.logits() call, then each row gets its own per-row gradient
    value exactly like the vendor's pointwise step() already does —
    np.add.at's accumulation into gW/gV doesn't care whether a row is
    "from a positive sample" or "from a negative sample", only that it
    receives the right per-row gradient. So the rest of step() (the
    accumulation, the Adam update) is copied from the vendor unchanged;
    only how `g` and the batch itself are built differs.

    `self.b`'s update (`self.b -= self.lr * g.sum()`) is ALSO copied
    unchanged, and is a mathematical no-op for BPR, not an oversight: a
    GLOBAL bias added identically to every logit cancels out exactly in
    z_i - z_j, so dL/db = (s-1) + (1-s) = 0 for every pair, and g.sum()
    is ~0 up to floating-point noise regardless of how many pairs are in
    the batch.
    """

    def step(self, x_pos: np.ndarray, x_neg: np.ndarray) -> float:
        n_pairs = len(x_pos)
        x_combined = np.concatenate([x_pos, x_neg], axis=0)
        z, E, S = self.logits(x_combined)
        z_pos, z_neg = z[:n_pairs], z[n_pairs:]
        s = _vendor.sigmoid(z_pos - z_neg)
        g = (np.concatenate([s - 1.0, 1.0 - s]) / n_pairs).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, x_combined, g[:, None])
        np.add.at(gV, x_combined, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(np.log(s + 1e-9)))


def _realize_fm_pairwise(
    model_slot: SlotConfig,
    fit_rows: list[tuple],
    score_rows: list[tuple],
    seed: int,
    objective_params: dict[str, Any],
):
    """Same shape as _realize_fm — same encode() call, same
    early-stopping rule (best validation primary, `patience` epochs of
    no improvement), same final scoring pass on `score_rows` — reused
    unchanged so the only difference from the baseline is the objective
    itself (see _PairwiseFM).
    """
    k = model_slot.params.get("k", 16)
    lr = model_slot.params.get("lr", 0.001)
    epochs = model_slot.params.get("epochs", 40)
    bs = model_slot.params.get("bs", 8192)
    patience = model_slot.params.get("patience", 4)
    pairs_per_batch = objective_params.get("pairs_per_batch", 8192)

    splits = {"train": fit_rows, "valid": score_rows, "test": score_rows}
    enc, dim = _vendor.encode(splits)
    x_train, y_train, users_train = enc["train"]
    x_score, y_score, user_ids_score = enc["valid"]

    pair_index = _build_bpr_pair_index(users_train, y_train)
    rng = np.random.default_rng(seed)
    model = _PairwiseFM(dim, k=k, lr=lr, seed=seed)

    # Same number of steps per epoch the pointwise loop would take over
    # this many rows (len(y_train) // bs), so an "epoch" costs a
    # comparable amount of work — BPR just spends it on sampled pairs
    # instead of on every row once.
    steps_per_epoch = max(1, len(y_train) // bs)

    best_primary, best_state, bad = -1.0, None, 0
    for _ in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            pos_idx, neg_idx = _sample_bpr_pairs(pair_index, pairs_per_batch, rng)
            model.step(x_train[pos_idx], x_train[neg_idx])
        score_result = _vendor.evaluate(user_ids_score, y_score, model.predict(x_score))
        if score_result["primary"] > best_primary + 1e-5:
            best_primary, bad = score_result["primary"], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break
    model.V, model.W, model.b = best_state

    scores = model.predict(x_score)
    return user_ids_score, y_score, scores


def _realize_fm(
    model_slot: SlotConfig,
    fit_rows: list[tuple],
    score_rows: list[tuple],
    seed: int,
    sample_weights: Optional[np.ndarray] = None,
):
    """Trains vendor's FM on `fit_rows`, early-stopping and scoring on
    `score_rows`, and returns (user_ids, labels, scores) aligned to
    `score_rows`, in `score_rows` order.

    WHAT WE FOUND ABOUT run_fm (investigated as requested, since we need
    to run this on backtest windows too, not just the official split):

    1. run_fm's dict KEYS are hardcoded ('train'/'valid'/'test' —
       vendor/kuairand-starter-kit/baseline.py does
       `Xtr,... = enc['train']`, `Xva,... = enc['valid']`,
       `Xte,... = enc['test']` verbatim) but the VALUES are not tied to
       any date range — it only ever touches whatever row lists the
       caller placed under those keys. So it CAN be reused for arbitrary
       fit/score row lists by relabeling: `fit_rows` under 'train',
       `score_rows` under 'valid'. It also unconditionally destructures
       `enc['test']`, so a 'test' entry must be present even though nothing
       reads its result; we reuse `score_rows` there rather than ever
       touching the real hidden test split (which `data.load("test")`
       would refuse to return anyway).

    2. THIS IS NOT WHY WE DON'T CALL run_fm() DIRECTLY, THOUGH. The real
       blocker: run_fm's return value only ever exposes vendor's own
       aggregated evaluate() dict —
       `return {'valid': evaluate(uva, yva, m.predict(Xva)), 'test': ...}`
       — the trained model `m` and the raw predictions `m.predict(Xva)`
       are local variables, never returned. We need the raw
       (user_ids, labels, scores) triple (harness.metrics.evaluate and
       cache.save_predictions both need it), which run_fm's signature
       cannot give us at all, regardless of the key-relabeling above. So
       this function reimplements run_fm's training loop inline —
       same hyperparameters, same early-stopping rule — using
       `encode()` and the `FM` class directly, stopping short of
       run_fm's own final `evaluate()` call so the raw scores survive.

    Using `score_rows` for BOTH early-stopping model selection AND final
    scoring is not a leak introduced by this reuse — it's the same
    pattern run_fm already uses for the official train/valid split
    (early-stop on 'valid', then score 'valid' again), which is standard
    practice. It's also exactly what the backtest is supposed to mirror:
    the same train-then-evaluate-on-a-later-window methodology real
    validation uses, just carved out of the training data itself.

    `sample_weights`, if given, must align index-for-index with
    `fit_rows` (one weight per fit row, same order) — see
    _compute_exp_decay_weights, which builds exactly that. When None,
    this uses the vendor's own unweighted FM/step unchanged, so moves
    that don't touch the weighting slot get byte-identical behaviour to
    before this function accepted a weights argument at all.
    """
    k = model_slot.params.get("k", 16)
    lr = model_slot.params.get("lr", 0.001)
    epochs = model_slot.params.get("epochs", 40)
    bs = model_slot.params.get("bs", 8192)
    patience = model_slot.params.get("patience", 4)

    splits = {"train": fit_rows, "valid": score_rows, "test": score_rows}
    enc, dim = _vendor.encode(splits)
    x_train, y_train, _ = enc["train"]
    x_score, y_score, user_ids_score = enc["valid"]

    if sample_weights is not None:
        if len(sample_weights) != len(y_train):
            raise ValueError(
                f"sample_weights length {len(sample_weights)} does not match "
                f"fit_rows length {len(y_train)}"
            )
        model = _WeightedFM(dim, k=k, lr=lr, seed=seed)
    else:
        model = _vendor.FM(dim, k=k, lr=lr, seed=seed)

    rng = np.random.default_rng(seed)
    best_primary, best_state, bad = -1.0, None, 0
    for _ in range(1, epochs + 1):
        order = rng.permutation(len(y_train))
        for i in range(0, len(order), bs):
            batch = order[i : i + bs]
            if sample_weights is not None:
                model.step(x_train[batch], y_train[batch], sample_weights[batch])
            else:
                model.step(x_train[batch], y_train[batch])
        score_result = _vendor.evaluate(user_ids_score, y_score, model.predict(x_score))
        if score_result["primary"] > best_primary + 1e-5:
            best_primary, bad = score_result["primary"], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break
    model.V, model.W, model.b = best_state

    scores = model.predict(x_score)
    return user_ids_score, y_score, scores


def _realize_fm_for_test_submission(
    model_slot: SlotConfig,
    fit_rows: list[tuple],
    val_rows: list[tuple],
    test_rows: list[tuple],
    seed: int,
):
    """Trains vendor's FM on `fit_rows`, early-stopping and MODEL-
    SELECTING via `val_rows` — the exact same procedure and early-
    stopping rule as _realize_fm's validation pass — then predicts on
    `test_rows` using the FINAL selected weights. Returns (user_ids,
    scores) aligned to `test_rows`, in `test_rows` order. No labels are
    returned: a real submission has none to report, and this function
    must not tempt a caller into reading the ones present in this
    offline copy of the dataset.

    TEST LABELS ARE NEVER READ FOR ANYTHING. `encode()` extracts a label
    column for every split it's handed, including 'test' — that's a
    byproduct of encode() not knowing or caring about this distinction —
    but this function only ever uses test_rows' user_ids and X
    (features), for alignment and prediction respectively. Which epoch's
    weights survive (`best_state`) is decided ENTIRELY from `val_rows`,
    before `test_rows` are touched at all. Scoring test_rows during
    training to pick an epoch would be exactly the leak
    harness/validate.py's "read structure, never labels" principle
    exists to prevent — it would just be a subtler version of it, since
    nothing about training with test-set early stopping ever LOOKS like
    reading a label, from the caller's side.
    """
    k = model_slot.params.get("k", 16)
    lr = model_slot.params.get("lr", 0.001)
    epochs = model_slot.params.get("epochs", 40)
    bs = model_slot.params.get("bs", 8192)
    patience = model_slot.params.get("patience", 4)

    splits = {"train": fit_rows, "valid": val_rows, "test": test_rows}
    enc, dim = _vendor.encode(splits)
    x_train, y_train, _ = enc["train"]
    x_val, y_val, user_ids_val = enc["valid"]
    x_test, _, user_ids_test = enc["test"]  # test labels deliberately discarded

    model = _vendor.FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary, best_state, bad = -1.0, None, 0
    for _ in range(1, epochs + 1):
        order = rng.permutation(len(y_train))
        for i in range(0, len(order), bs):
            batch = order[i : i + bs]
            model.step(x_train[batch], y_train[batch])
        # Model selection uses ONLY val_rows — test_rows are never scored
        # during training, at any epoch.
        val_result = _vendor.evaluate(user_ids_val, y_val, model.predict(x_val))
        if val_result["primary"] > best_primary + 1e-5:
            best_primary, bad = val_result["primary"], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break
    model.V, model.W, model.b = best_state

    test_scores = model.predict(x_test)
    return user_ids_test, test_scores


def realize_for_submission(
    config: PipelineConfig,
    fit_rows: list[tuple],
    val_rows: list[tuple],
    test_rows: list[tuple],
    seed: int,
):
    """Like realize(), but for producing a FINAL submission: trains on
    `fit_rows` with model selection via `val_rows` (never `test_rows`),
    then predicts on `test_rows` using the final selected weights.
    Returns (user_ids, scores) — no labels, unlike realize(), matching
    _realize_fm_for_test_submission's contract.

    Only model.impl == "fm" is realized, matching realize()'s own scope
    today; anything else raises NotImplementedError naming it.
    """
    model_slot = config.slots["model"]
    if model_slot.impl != "fm":
        raise NotImplementedError(
            f"executor.realize: no submission realization implemented for model impl {model_slot.impl!r}"
        )
    return _realize_fm_for_test_submission(model_slot, fit_rows, val_rows, test_rows, seed)


def realize(config: PipelineConfig, fit_rows: list[tuple], score_rows: list[tuple], seed: int):
    """Trains `config` on `fit_rows` and returns (user_ids, labels,
    scores) aligned to `score_rows`, in `score_rows` order.

    Dispatches on model/data_view/weighting impls; falls through to
    existing behaviour ("full" / "none") unchanged. Any impl not listed
    below — for any slot, including model, features, objective, and
    calibration — raises NotImplementedError naming the slot and impl.
    That is correct behaviour, not a gap to silently paper over: nothing
    else has a realization yet.
    """
    model_slot = config.slots["model"]
    if model_slot.impl != "fm":
        raise NotImplementedError(
            f"executor.realize: no realization implemented for model impl {model_slot.impl!r}"
        )

    features_slot = config.slots["features"]
    if features_slot.impl != "baseline_5":
        raise NotImplementedError(
            f"executor.realize: no realization implemented for features impl {features_slot.impl!r}"
        )

    calibration_slot = config.slots["calibration"]
    if calibration_slot.impl != "none":
        raise NotImplementedError(
            f"executor.realize: no realization implemented for calibration impl {calibration_slot.impl!r}"
        )

    data_view_slot = config.slots["data_view"]
    if data_view_slot.impl == "full":
        pass
    elif data_view_slot.impl == "recent_window":
        fit_rows = _apply_recency_window(fit_rows, data_view_slot.params)
    else:
        raise NotImplementedError(
            f"executor.realize: no realization implemented for data_view impl {data_view_slot.impl!r}"
        )

    objective_slot = config.slots["objective"]
    weighting_slot = config.slots["weighting"]

    if objective_slot.impl == "bpr":
        # BPR changes the training loop's batching unit itself (pairs,
        # not weighted rows), so it doesn't compose with the weighting
        # slot's per-row sample_weights today — nothing proposes that
        # combination yet, and silently ignoring one or the other would
        # be exactly the kind of quiet wrong-behaviour this project has
        # been careful to avoid elsewhere.
        if weighting_slot.impl != "none":
            raise NotImplementedError(
                f"executor.realize: combining weighting impl {weighting_slot.impl!r} "
                "with objective impl 'bpr' is not implemented"
            )
        return _realize_fm_pairwise(model_slot, fit_rows, score_rows, seed, objective_slot.params)

    if objective_slot.impl != "bce":
        raise NotImplementedError(
            f"executor.realize: no realization implemented for objective impl {objective_slot.impl!r}"
        )

    if weighting_slot.impl == "none":
        sample_weights = None
    elif weighting_slot.impl == "exp_decay":
        sample_weights = _compute_exp_decay_weights(fit_rows, weighting_slot.params)
    else:
        raise NotImplementedError(
            f"executor.realize: no realization implemented for weighting impl {weighting_slot.impl!r}"
        )

    return _realize_fm(model_slot, fit_rows, score_rows, seed, sample_weights=sample_weights)
