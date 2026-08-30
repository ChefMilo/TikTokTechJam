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


def _bootstrap_ci(per_seed_deltas: list[float]) -> tuple[float, float]:
    """1000-resample, 95% percentile bootstrap CI on the mean paired delta.

    GRANULARITY NOTE: the design calls for a "user-level" bootstrap —
    resampling individual users' GAUC/nDCG contributions from raw
    validation predictions, since those are already per-user aggregates.
    contracts.CandidateResult.val only carries per-seed Metrics (already
    aggregated across every user for that seed), not the underlying
    per-user rows — a true user-level bootstrap would need to read
    candidate.val_pred_path / incumbent.val_pred_path off disk, and no
    schema for that file exists yet in this project. This instead
    bootstraps over the matched PER-SEED paired deltas: the finest
    granularity actually available in the contract today. With only 3-5
    seeds this is a coarse CI, not a substitute for the real thing — when
    val_pred_path gains a defined schema, this should be upgraded to
    resample actual users instead of seeds.

    Uses a fixed-seed RNG so compare() stays a pure function of its
    inputs: the same two CandidateResults must always produce the same
    Verdict, not one that flickers between calls.
    """
    rng = np.random.default_rng(0)
    deltas = np.asarray(per_seed_deltas, dtype=np.float64)
    resampled_means = np.empty(_BOOTSTRAP_RESAMPLES)
    for i in range(_BOOTSTRAP_RESAMPLES):
        resampled_means[i] = rng.choice(deltas, size=len(deltas), replace=True).mean()
    low, high = np.percentile(resampled_means, _BOOTSTRAP_PERCENTILES)
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
    ci95 = _bootstrap_ci(per_seed_deltas)
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
