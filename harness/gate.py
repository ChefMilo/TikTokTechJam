"""Noise gate: decides whether an observed metric improvement is real or
within noise, and returns the controller's contracts.Verdict.

Two stages, dispatched on how many seeds the candidate carries in `.val`:

  SCREEN  (exactly 1 seed) — a cheap filter. Can reject outright on a
          clearly-worse single draw; can never accept, since one seed is
          not enough evidence to promote a candidate.
  CONFIRM (>= 3 seeds)     — the real decision. Paired per-seed deltas,
          a bootstrap CI, and a same-direction backtest requirement.

See contracts.py's PipelineConfig.seed comment and CandidateResult.val
comment for why seeds are paired rather than averaged before reaching
this module.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from contracts import CandidateResult, Metrics, Verdict
from harness import cache, metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
_SEED_VARIANCE_PATH = REPO_ROOT / "artifacts" / "seed_variance.json"
_FALLBACK_SIGMA = 0.0008

# The organizers' convergence rule ("stop after N=3 iterations without a
# >epsilon improvement") — see clears_convergence_epsilon below for why
# this is tracked separately from the gate's own accept/reject decision.
CONVERGENCE_EPSILON = 0.002

_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_PERCENTILES = (2.5, 97.5)


def _load_sigma() -> float:
    if not _SEED_VARIANCE_PATH.exists():
        warnings.warn(
            f"{_SEED_VARIANCE_PATH} not found; falling back to the organizers' "
            f"hidden-test sigma ({_FALLBACK_SIGMA}). Run scripts/seed_variance.py "
            "to measure our own validation sigma.",
            stacklevel=2,
        )
        return _FALLBACK_SIGMA
    with open(_SEED_VARIANCE_PATH) as fh:
        payload = json.load(fh)
    return float(payload["sigma_primary"])


# IMPORTANT CAVEAT: our measured SIGMA (0.000353, from
# artifacts/seed_variance.json) is LESS than half the organizers' hidden-
# TEST-split sigma (0.0008) — the wrong direction, since validation
# (124,909 rows) is the SMALLER split and should be noisier, not quieter,
# than test (170,588 rows). The likely cause: vendor baseline.run_fm
# early-stops on validation primary itself (patience=4), so each of our 5
# runs reports a MAX over ~40 epochs on the very split being measured, and
# taking a max compresses variance relative to a single evaluation. Treat
# SIGMA as a LOWER BOUND on the true per-seed noise, not an honest
# estimate of it — this is why the thresholds below use an explicit floor
# (max(3 * SIGMA, 0.002)) rather than trusting 3 * SIGMA alone.
SIGMA = _load_sigma()


def _mean_primary(seed_metrics: dict[int, Metrics]) -> float:
    primaries = [metrics.primary for metrics in seed_metrics.values()]
    return sum(primaries) / len(primaries)


def _backtest_delta(candidate: CandidateResult, incumbent: CandidateResult) -> float | None:
    """Mean backtest primary delta over seeds present in both. None if
    either side never backtested, or if they share no backtested seed.
    """
    if not candidate.backtest or not incumbent.backtest:
        return None
    matched = sorted(set(candidate.backtest) & set(incumbent.backtest))
    if not matched:
        return None
    return sum(
        candidate.backtest[seed].primary - incumbent.backtest[seed].primary for seed in matched
    ) / len(matched)


def _bootstrap_ci_seed_level(per_seed_deltas: list[float], rng: np.random.Generator) -> tuple[float, float]:
    """FALLBACK ONLY (see _confirm) — bootstraps over the 3-5 per-seed
    deltas themselves, used when per-user validation predictions are not
    available for candidate and incumbent both.

    This is NOT a coarse approximation of the intended test — it is
    materially too permissive, and was a real correctness bug when it was
    the only implementation. With exactly 3 matched seeds, if all three
    per-seed deltas happen to share a sign, EVERY bootstrap resample drawn
    from them also shares that sign, so the CI always excludes zero and
    the gate accepts — regardless of how small the deltas are, because
    resampling 3 same-signed numbers can never produce a resample mean of
    the opposite sign. Under pure noise (no real effect), three
    independent same-sign draws happen 1-in-8 of the time (2 * 0.5^3),
    i.e. roughly a 12.5% false-positive rate. A fluke accepted here
    becomes the new incumbent and poisons every later comparison against
    it. _bootstrap_ci_user_level below is the real test and is preferred
    whenever possible; this function only lets the gate still render a
    verdict when per-user data isn't available, and callers must mark the
    resulting reason with "coarse_ci_seed_bootstrap" so this weaker path
    is never silently indistinguishable from the real one in the journal.
    """
    deltas = np.asarray(per_seed_deltas, dtype=np.float64)
    resampled_means = np.empty(_BOOTSTRAP_RESAMPLES)
    for i in range(_BOOTSTRAP_RESAMPLES):
        resampled_means[i] = rng.choice(deltas, size=len(deltas), replace=True).mean()
    low, high = np.percentile(resampled_means, _BOOTSTRAP_PERCENTILES)
    return float(low), float(high)


def _per_user_metrics(
    user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray, k: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precomputes per-user contributions ONCE, aligned to unique_users:

        npos : int64,   positives per user
        auc  : float64, per-user AUC — NaN where the user is
               GAUC-ineligible (npos == 0 or npos == impressions), which
               doubles as the eligibility mask for callers (see
               _bootstrap_ci_user_level)
        ndcg : float64, per-user nDCG@k

    This is exact, not an approximation: per harness/SCHEMA_NOTES.md
    evaluate.py Q4-Q6, GAUC and nDCG@k already ARE per-user aggregates
    (GAUC = sum(npos_u * auc_u) / sum(npos_u) over eligible users; nDCG@k
    = mean(ndcg_u) over ALL users). Reaggregating these three arrays
    reproduces vendor evaluate()'s primary exactly — see
    test_per_user_metrics_reaggregates_to_vendor_primary in
    tests/test_gate.py, which is the actual proof, not just an assertion
    of it.

    Calls the vendor's OWN auc()/ndcg_at_k() per user rather than
    reimplementing them, so this is guaranteed byte-identical to
    harness.metrics.evaluate's per-user math rather than merely
    "mirroring" it by hand.
    """
    unique_users, codes = np.unique(user_ids, return_inverse=True)
    n_unique = len(unique_users)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes[order], minlength=n_unique)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    npos = np.zeros(n_unique, dtype=np.int64)
    auc = np.full(n_unique, np.nan, dtype=np.float64)
    ndcg = np.zeros(n_unique, dtype=np.float64)

    vendor_auc = metrics._vendor.auc
    vendor_ndcg_at_k = metrics._vendor.ndcg_at_k

    for c in range(n_unique):
        idx = order[offsets[c]:offsets[c + 1]]
        user_labels = labels[idx]
        user_scores = scores[idx]
        n_impressions = len(idx)
        pos = int(user_labels.sum())
        npos[c] = pos

        if 0 < pos < n_impressions:
            auc[c] = vendor_auc(user_labels.tolist(), user_scores.tolist())

        # ndcg_at_k expects labels already sorted by DESCENDING score —
        # vendor evaluate() does that sort itself before calling it
        # (evaluate.py: `lst.sort(key=lambda x: -x[0])`).
        desc_order = np.argsort(-user_scores, kind="stable")
        ndcg[c] = vendor_ndcg_at_k(user_labels[desc_order].tolist(), k)

    return unique_users, npos, auc, ndcg


_BOOTSTRAP_CHUNK = 50  # resamples per vectorized batch — see the timing note below


def _eligibility_weighted(npos: np.ndarray, auc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Folds GAUC eligibility into two arrays so the resample loop never
    has to look at NaN/eligibility again: `weighted_npos` is `npos` where
    the user is GAUC-eligible and 0 elsewhere, `weighted_gauc_num` is
    `npos * auc` under the same mask. Both feed straight into a sum —
    exactly `gnum`/`gden` from vendor evaluate(), computed once per
    (seed, model) instead of once per resample per batch.
    """
    eligible = ~np.isnan(auc)  # see _per_user_metrics: NaN <=> ineligible
    weighted_npos = np.where(eligible, npos, 0).astype(np.float64)
    weighted_gauc_num = np.where(eligible, npos * auc, 0.0)
    return weighted_npos, weighted_gauc_num


def _resampled_primary_batch(
    weighted_npos: np.ndarray,
    weighted_gauc_num: np.ndarray,
    ndcg: np.ndarray,
    drawn: np.ndarray,
) -> np.ndarray:
    """primary for a whole BATCH of resamples at once, from the
    eligibility-weighted arrays above — mirrors vendor evaluate()'s own
    gauc/ndcg/primary reduction exactly, just applied to resampled index
    sets instead of every user once, and to many resamples in one
    vectorized pass instead of one Python-level call per resample (the
    per-resample version cost ~0.6ms/call, dominated by numpy dispatch
    overhead on 22k-element arrays rather than by the arithmetic itself —
    batching amortizes that overhead across `drawn`'s whole first axis).

    `drawn` has shape (batch, n_users); returns primary, shape (batch,).
    """
    gnum = weighted_gauc_num[drawn].sum(axis=1)
    gden = weighted_npos[drawn].sum(axis=1)
    # matches vendor evaluate(): `gnum / gden if gden else 0.5`, per resample
    gauc = np.where(gden > 0, gnum / np.where(gden > 0, gden, 1), 0.5)
    ndcg_mean = ndcg[drawn].mean(axis=1)
    return (gauc + ndcg_mean) / 2.0


def _bootstrap_ci_user_level(
    candidate: CandidateResult,
    incumbent: CandidateResult,
    matched_seeds: list[int],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """The intended test: precompute each matched seed's per-user (npos,
    auc, ndcg) ONCE for candidate and incumbent, then for each of 1000
    resamples draw the validation user universe with replacement ONCE
    (shared across every matched seed, so seed-to-seed correlation from
    resampling the same users is preserved), recompute primary for
    candidate and incumbent straight from the precomputed arrays (no
    retraining, and no repeated full scorer passes), and average the
    resulting per-seed delta over matched seeds. Percentiles are taken
    over the 1000 resample-level means.

    This replaces an earlier version that called harness.metrics.evaluate
    (a full O(rows) scorer pass) inside the resample loop: 1000 resamples
    x n_seeds x 2 models x ~125k rows was roughly 50 minutes per gate
    decision against a 6-hour run budget — unusable. Precomputing once
    turns the resample loop into O(1000 * seeds * users) array indexing,
    with no further calls into the scorer at all. A second pass then
    batches those resamples (_resampled_primary_batch, _BOOTSTRAP_CHUNK
    at a time) rather than looping one resample at a time in Python —
    on the full real validation set (~125k rows, ~22k users, 3 seeds)
    that brings a full CONFIRM to ~4-5s (see
    test_full_confirm_on_real_validation_data_completes_in_seconds_not_minutes).
    """
    per_seed_stats: dict[int, tuple] = {}
    for seed in matched_seeds:
        c_users, c_labels, c_scores = cache.load_predictions(candidate.config_id, seed, "val")
        i_users, i_labels, i_scores = cache.load_predictions(incumbent.config_id, seed, "val")
        # cache.py stores labels as int8 (correct for the on-disk schema —
        # each value is 0/1), which is fine here since labels are only
        # touched during this one precompute pass, never resampled
        # directly. Upcast before summing regardless, so a real user with
        # an unusually large impression count still can't overflow it.
        c_labels = c_labels.astype(np.int64)
        i_labels = i_labels.astype(np.int64)
        if not np.array_equal(c_users, i_users) or not np.array_equal(c_labels, i_labels):
            raise ValueError(
                f"seed {seed}: candidate and incumbent validation predictions "
                "do not share the same rows (user_ids/labels differ) — cannot "
                "pair them for a user-level bootstrap"
            )
        unique_users, npos, c_auc, c_ndcg = _per_user_metrics(c_users, c_labels, c_scores)
        # npos only depends on labels, which candidate and incumbent
        # share (checked above) — no need to recompute it for incumbent.
        _, _, i_auc, i_ndcg = _per_user_metrics(i_users, i_labels, i_scores)
        c_weighted_npos, c_weighted_gauc_num = _eligibility_weighted(npos, c_auc)
        i_weighted_npos, i_weighted_gauc_num = _eligibility_weighted(npos, i_auc)
        # c_weighted_npos == i_weighted_npos always (eligibility depends
        # only on npos/impressions, which candidate and incumbent share).
        per_seed_stats[seed] = (
            unique_users, c_weighted_npos, c_weighted_gauc_num, c_ndcg, i_weighted_gauc_num, i_ndcg,
        )

    # Validation rows/users are identical across seeds (only scores
    # differ), so any matched seed's user set is the resampling universe.
    reference_seed = matched_seeds[0]
    n_users = len(per_seed_stats[reference_seed][0])

    resample_deltas = np.empty(_BOOTSTRAP_RESAMPLES)
    for start in range(0, _BOOTSTRAP_RESAMPLES, _BOOTSTRAP_CHUNK):
        end = min(start + _BOOTSTRAP_CHUNK, _BOOTSTRAP_RESAMPLES)
        # One draw per resample in this batch, shared across every matched
        # seed below — same "shared-user-draw" behaviour as before, just
        # generated batch_size-at-a-time instead of one-at-a-time.
        drawn = rng.integers(0, n_users, size=(end - start, n_users))

        batch_seed_deltas = np.zeros(end - start)
        for seed in matched_seeds:
            _, weighted_npos, c_weighted_gauc_num, c_ndcg, i_weighted_gauc_num, i_ndcg = per_seed_stats[seed]
            candidate_primary = _resampled_primary_batch(weighted_npos, c_weighted_gauc_num, c_ndcg, drawn)
            incumbent_primary = _resampled_primary_batch(weighted_npos, i_weighted_gauc_num, i_ndcg, drawn)
            batch_seed_deltas += candidate_primary - incumbent_primary
        resample_deltas[start:end] = batch_seed_deltas / len(matched_seeds)

    low, high = np.percentile(resample_deltas, _BOOTSTRAP_PERCENTILES)
    return float(low), float(high)


def _screen(candidate: CandidateResult, incumbent: CandidateResult) -> Verdict:
    """Cheap rejection only, never acceptance — one seed is not enough
    evidence to promote a candidate, only enough to rule one out early.
    """
    (candidate_seed, candidate_metrics), = candidate.val.items()
    if candidate_seed in incumbent.val:
        # Paired on the same seed when possible, same philosophy as CONFIRM.
        incumbent_primary = incumbent.val[candidate_seed].primary
    else:
        incumbent_primary = _mean_primary(incumbent.val)
    delta = candidate_metrics.primary - incumbent_primary

    # Floor at 0.002 rather than trusting 3 * SIGMA (~0.001) alone — see
    # the IMPORTANT CAVEAT above. Without the floor, a candidate that drew
    # one merely-unlucky seed would be killed before it ever reached
    # CONFIRM.
    screen_threshold = max(3 * SIGMA, 0.002)

    backtest_delta = _backtest_delta(candidate, incumbent)

    if delta < -screen_threshold:
        reason = "screen_rejected_delta_below_threshold"
    else:
        reason = "screen_passed_needs_confirm"

    return Verdict(
        accept=False,
        delta=delta,
        ci95=(delta, delta),
        n_seeds=1,
        backtest_delta=backtest_delta,
        reason=reason,
    )


def _confirm(candidate: CandidateResult, incumbent: CandidateResult) -> Verdict:
    matched_seeds = sorted(set(candidate.val) & set(incumbent.val))

    if len(matched_seeds) < 3:
        # 3 is the evidence bar itself (see module docstring / compare());
        # fewer matched seeds means we can't even attempt the paired test,
        # regardless of what the raw numbers happen to look like.
        delta = (
            sum(candidate.val[s].primary - incumbent.val[s].primary for s in matched_seeds)
            / len(matched_seeds)
            if matched_seeds
            else 0.0
        )
        return Verdict(
            accept=False,
            delta=delta,
            ci95=(delta, delta),
            n_seeds=len(matched_seeds),
            backtest_delta=None,
            reason="insufficient_paired_seeds",
        )

    per_seed_deltas = [candidate.val[s].primary - incumbent.val[s].primary for s in matched_seeds]
    delta = sum(per_seed_deltas) / len(per_seed_deltas)

    # Fixed seed so compare() stays a pure function of its inputs: the
    # same two CandidateResults must always produce the same Verdict.
    rng = np.random.default_rng(0)

    # The real, intended test needs per-user validation predictions on
    # BOTH sides, for EVERY matched seed. cache.exists is the ground
    # truth for that — not CandidateResult.val_pred_path, which is a
    # single field that says nothing about which specific seeds actually
    # got their predictions saved, and can disagree with what
    # _bootstrap_ci_user_level itself will try to load via
    # cache.load_predictions(config_id, seed, "val"). Checking
    # val_pred_path here was two sources of truth for one fact; probing
    # the cache directly makes this decision the only one that matters.
    missing = [
        (config_id, seed)
        for seed in matched_seeds
        for config_id in (candidate.config_id, incumbent.config_id)
        if not cache.exists(config_id, seed, "val")
    ]
    user_level_available = not missing

    if user_level_available:
        ci95 = _bootstrap_ci_user_level(candidate, incumbent, matched_seeds, rng)
        ci_method_note = None
    else:
        # Never silently degrade. "coarse_ci_seed_bootstrap" below is the
        # machine-readable signal, but a human running the search needs
        # to see this too — a run where every decision silently took this
        # path would still complete and look completely normal, and this
        # fallback has a ~12.5% false-positive rate (see
        # _bootstrap_ci_seed_level's docstring).
        warnings.warn(
            f"noise gate falling back to the weaker seed-level bootstrap "
            f"(~12.5% false-positive rate) for candidate={candidate.config_id!r} "
            f"vs incumbent={incumbent.config_id!r}: missing cached validation "
            f"predictions for (config_id, seed) = {missing}",
            stacklevel=2,
        )
        ci95 = _bootstrap_ci_seed_level(per_seed_deltas, rng)
        ci_method_note = "coarse_ci_seed_bootstrap"

    backtest_delta = _backtest_delta(candidate, incumbent)
    n_seeds = len(matched_seeds)

    if ci95[0] <= 0:
        accept, reason = False, "ci_includes_zero"
    elif backtest_delta is None:
        accept, reason = False, "backtest_missing"
    elif backtest_delta <= 0:
        # Covers both a negative backtest delta and exactly 0.0 — neither
        # clears the strict ">0" bar acceptance requires.
        accept, reason = False, "backtest_negative"
    else:
        accept = True
        reason = (
            f"paired CI excludes zero (n={n_seeds} seeds, delta={delta:+.5f}) "
            f"and backtest confirms (backtest_delta={backtest_delta:+.5f})"
        )

    if ci_method_note:
        reason = f"{reason}; {ci_method_note}"

    return Verdict(
        accept=accept,
        delta=delta,
        ci95=ci95,
        n_seeds=n_seeds,
        backtest_delta=backtest_delta,
        reason=reason,
    )


def compare(candidate: CandidateResult, incumbent: CandidateResult) -> Verdict:
    """Decides whether `candidate` beats `incumbent`. Dispatches on the
    number of seeds present in candidate.val — see module docstring.
    """
    n_candidate_seeds = len(candidate.val)
    if n_candidate_seeds == 1:
        return _screen(candidate, incumbent)
    if n_candidate_seeds >= 3:
        return _confirm(candidate, incumbent)
    raise ValueError(
        f"candidate has {n_candidate_seeds} seed(s) in .val; the noise gate "
        "expects exactly 1 (screen stage) or >= 3 (confirm stage) — this "
        "shape isn't part of the screen/confirm design."
    )


def clears_convergence_epsilon(verdict: Verdict) -> bool:
    """True iff verdict.delta > CONVERGENCE_EPSILON (0.002).

    This is a DIFFERENT test from the gate's own accept/reject rule above,
    and the two must not be conflated. The gate accepts any statistically
    real gain — a genuine +0.0005 improvement is still real and worth
    keeping, however small. But the organizers' convergence rule ("stop
    after N=3 iterations without a >epsilon improvement") is calibrated on
    raw effect size, not on statistical significance, and answers a
    different question: WHEN TO STOP SEARCHING, not whether to keep a
    result. A run can accept several small, real, statistically-confirmed
    gains in a row and still be "stalled" by the organizers' definition,
    because none of them individually cleared epsilon. The controller
    needs both signals independently: `verdict.accept` (was this real?)
    and `clears_convergence_epsilon(verdict)` (was it big enough to reset
    the no-improvement counter?).
    """
    return verdict.delta > CONVERGENCE_EPSILON
