"""Tests for harness.gate — synthetic CandidateResults only, no training.

SIGMA is whatever harness.gate loaded from artifacts/seed_variance.json
at import time (see that module's IMPORTANT CAVEAT comment on why it's
treated as a lower bound, not an honest estimate). Tests build deltas as
multiples of gate.SIGMA rather than hardcoding a number, so they stay
correct if the measured sigma is ever refreshed.
"""

from contracts import CandidateResult, Metrics, Status, Verdict
from harness import gate


def _metrics(primary: float) -> Metrics:
    # Two equal values so Metrics.primary (their mean) is exactly `primary`
    # — gives synthetic tests exact control without caring about real
    # GAUC/nDCG semantics.
    return Metrics(values={"GAUC": primary, "nDCG@5": primary})


def _candidate(val: dict[int, float], backtest: dict[int, float] | None = None) -> CandidateResult:
    backtest = backtest or {}
    return CandidateResult(
        config_id="cand",
        status=Status.OK,
        val={seed: _metrics(p) for seed, p in val.items()},
        backtest={seed: _metrics(p) for seed, p in backtest.items()},
    )


def test_identical_candidate_is_rejected_with_ci_spanning_zero():
    incumbent = _candidate({0: 0.60, 1: 0.60, 2: 0.60}, {0: 0.55, 1: 0.55, 2: 0.55})
    candidate = _candidate({0: 0.60, 1: 0.60, 2: 0.60}, {0: 0.55, 1: 0.55, 2: 0.55})

    verdict = gate.compare(candidate, incumbent)

    assert verdict.accept is False
    assert verdict.ci95[0] <= 0.0 <= verdict.ci95[1]


def test_candidate_better_by_ten_sigma_on_every_seed_is_accepted():
    bump = 10 * gate.SIGMA
    incumbent = _candidate({0: 0.60, 1: 0.60, 2: 0.60}, {0: 0.55, 1: 0.55, 2: 0.55})
    candidate = _candidate(
        {0: 0.60 + bump, 1: 0.60 + bump, 2: 0.60 + bump},
        {0: 0.55 + bump, 1: 0.55 + bump, 2: 0.55 + bump},
    )

    verdict = gate.compare(candidate, incumbent)

    assert verdict.accept is True
    assert verdict.ci95[0] > 0
    assert verdict.backtest_delta is not None and verdict.backtest_delta > 0


def test_better_validation_negative_backtest_is_rejected_as_backtest_negative():
    bump = 10 * gate.SIGMA
    incumbent = _candidate({0: 0.60, 1: 0.60, 2: 0.60}, {0: 0.55, 1: 0.55, 2: 0.55})
    candidate = _candidate(
        {0: 0.60 + bump, 1: 0.60 + bump, 2: 0.60 + bump},
        {0: 0.50, 1: 0.50, 2: 0.50},  # worse on backtest
    )

    verdict = gate.compare(candidate, incumbent)

    assert verdict.accept is False
    assert verdict.reason == "backtest_negative"


def test_empty_backtest_is_rejected_as_backtest_missing():
    bump = 10 * gate.SIGMA
    incumbent = _candidate({0: 0.60, 1: 0.60, 2: 0.60}, {0: 0.55, 1: 0.55, 2: 0.55})
    candidate = _candidate({0: 0.60 + bump, 1: 0.60 + bump, 2: 0.60 + bump}, {})

    verdict = gate.compare(candidate, incumbent)

    assert verdict.accept is False
    assert verdict.reason == "backtest_missing"
    assert verdict.backtest_delta is None


def test_one_seed_candidate_never_accepts():
    bump = 100 * gate.SIGMA  # deliberately huge — should still never accept
    incumbent = _candidate({0: 0.60}, {0: 0.55})
    candidate = _candidate({0: 0.60 + bump}, {0: 0.55 + bump})

    verdict = gate.compare(candidate, incumbent)

    assert verdict.accept is False
    assert verdict.n_seeds == 1


def test_only_two_matching_seeds_is_insufficient():
    incumbent = _candidate({0: 0.60, 1: 0.60}, {0: 0.55, 1: 0.55})
    candidate = _candidate({0: 0.61, 1: 0.61, 2: 0.61}, {0: 0.56, 1: 0.56, 2: 0.56})

    verdict = gate.compare(candidate, incumbent)

    assert verdict.accept is False
    assert verdict.reason == "insufficient_paired_seeds"
    assert verdict.n_seeds == 2


def test_n_seeds_reflects_matched_seeds_not_total_on_either_side():
    bump = 10 * gate.SIGMA
    # incumbent has 4 seeds, candidate has 5, only {0, 1, 2} overlap.
    incumbent = _candidate(
        {0: 0.60, 1: 0.60, 2: 0.60, 3: 0.60},
        {0: 0.55, 1: 0.55, 2: 0.55, 3: 0.55},
    )
    candidate = _candidate(
        {0: 0.60 + bump, 1: 0.60 + bump, 2: 0.60 + bump, 4: 0.60 + bump, 5: 0.60 + bump},
        {0: 0.55 + bump, 1: 0.55 + bump, 2: 0.55 + bump, 4: 0.55 + bump, 5: 0.55 + bump},
    )

    verdict = gate.compare(candidate, incumbent)

    assert verdict.n_seeds == 3


def test_clears_convergence_epsilon_thresholds():
    below = Verdict(
        accept=True, delta=0.0005, ci95=(0.0001, 0.0009), n_seeds=3,
        backtest_delta=0.0004, reason="ok",
    )
    above = Verdict(
        accept=True, delta=0.003, ci95=(0.001, 0.005), n_seeds=3,
        backtest_delta=0.002, reason="ok",
    )

    assert gate.clears_convergence_epsilon(below) is False
    assert gate.clears_convergence_epsilon(above) is True
