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


def _rows_by_user_code(user_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns (unique_users, row_indices_by_code): row_indices_by_code[c]
    is the array of positions in `user_ids` belonging to unique_users[c].

    Built with one argsort rather than a per-user scan (real validation
    has ~24k users; an O(n_users * n_rows) scan would not finish).
    """
    unique_users, codes = np.unique(user_ids, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes[order], minlength=len(unique_users))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    row_indices_by_code = [order[offsets[c]:offsets[c + 1]] for c in range(len(unique_users))]
    return unique_users, row_indices_by_code


def _bootstrap_ci_user_level(
    candidate: CandidateResult,
    incumbent: CandidateResult,
    matched_seeds: list[int],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """The intended test: for each of 1000 resamples, draw the validation
    user universe with replacement ONCE (shared across every matched seed
    in that resample, so seed-to-seed correlation from resampling the
    same users is preserved rather than resampled independently per
    seed), rebuild each seed's resampled row set from the drawn users'
    own rows, recompute candidate/incumbent primary via harness.metrics
    on that resampled set (no retraining — this only reruns the vendor
    scorer), and average the resulting per-seed delta over matched seeds.
    Percentiles are taken over the 1000 resample-level means.

    Validation is not vectorized against harness.metrics.evaluate's pure-
    Python scorer, so this is O(1000 * seeds * rows) — fine for tests and
    for the synthetic scale used here, but not optimized for the full
    ~125k-row / ~24k-user validation set; a production run against the
    real split would need a faster per-user aggregation than repeatedly
    calling the vendor scorer.
    """
    per_seed_predictions: dict[int, tuple] = {}
    for seed in matched_seeds:
        c_users, c_labels, c_scores = cache.load_predictions(candidate.config_id, seed, "val")
        i_users, i_labels, i_scores = cache.load_predictions(incumbent.config_id, seed, "val")
        # cache.py stores labels as int8 (correct for the on-disk schema —
        # each value is 0/1). But a bootstrap resample can draw the same
        # user many times, stacking that user's rows into a block far
        # bigger than any single real user has, and vendor evaluate()'s
        # auc() sums labels and multiplies counts internally
        # (npos * (npos + 1)) — done in whatever dtype it's handed, so a
        # resampled per-user block large enough pushes that multiplication
        # past int8's range and silently wraps instead of raising. Upcast
        # once here, before any resampling, rather than downstream.
        c_labels = c_labels.astype(np.int64)
        i_labels = i_labels.astype(np.int64)
        if not np.array_equal(c_users, i_users) or not np.array_equal(c_labels, i_labels):
            raise ValueError(
                f"seed {seed}: candidate and incumbent validation predictions "
                "do not share the same rows (user_ids/labels differ) — cannot "
                "pair them for a user-level bootstrap"
            )
        unique_users, row_indices_by_code = _rows_by_user_code(c_users)
        per_seed_predictions[seed] = (c_users, c_labels, c_scores, i_scores, unique_users, row_indices_by_code)

    # Validation rows/users are identical across seeds (only scores
    # differ), so any matched seed's user set is the resampling universe.
    reference_seed = matched_seeds[0]
    n_users = len(per_seed_predictions[reference_seed][4])

    resample_deltas = np.empty(_BOOTSTRAP_RESAMPLES)
    for i in range(_BOOTSTRAP_RESAMPLES):
        drawn_codes = rng.integers(0, n_users, size=n_users)
        per_seed_deltas = []
        for seed in matched_seeds:
            c_users, c_labels, c_scores, i_scores, _, row_indices_by_code = per_seed_predictions[seed]
            idx = np.concatenate([row_indices_by_code[c] for c in drawn_codes])
            resampled_users = c_users[idx]
            resampled_labels = c_labels[idx]
            candidate_primary = metrics.evaluate(resampled_users, resampled_labels, c_scores[idx]).primary
            incumbent_primary = metrics.evaluate(resampled_users, resampled_labels, i_scores[idx]).primary
            per_seed_deltas.append(candidate_primary - incumbent_primary)
        resample_deltas[i] = sum(per_seed_deltas) / len(per_seed_deltas)

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
    # BOTH sides. Never silently degrade: when they aren't available, we
    # still render a verdict via the weaker seed-level bootstrap, but the
    # reason string below is tagged so the journal can tell the two
    # apart.
    user_level_available = candidate.val_pred_path is not None and incumbent.val_pred_path is not None
    if user_level_available:
        ci95 = _bootstrap_ci_user_level(candidate, incumbent, matched_seeds, rng)
        ci_method_note = None
    else:
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
