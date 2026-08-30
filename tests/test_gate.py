"""Tests for harness.gate — synthetic CandidateResults only, no training.

SIGMA is whatever harness.gate loaded from artifacts/seed_variance.json
at import time (see that module's IMPORTANT CAVEAT comment on why it's
treated as a lower bound, not an honest estimate). Tests build deltas as
multiples of gate.SIGMA rather than hardcoding a number, so they stay
correct if the measured sigma is ever refreshed.
"""

import dataclasses
import time

import numpy as np
import pytest

from contracts import CandidateResult, Metrics, Status, Verdict
from harness import cache, data, gate, metrics as hmetrics


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
    assert verdict.reason.startswith("ci_includes_zero")


def test_candidate_worse_by_ten_sigma_is_rejected_as_ci_entirely_negative():
    """Distinct from the identical-candidate case above: this interval
    doesn't straddle zero, it sits entirely below it — the candidate
    isn't statistically indistinguishable from the incumbent, it's
    worse. The two must not share a reason string (judges read these
    directly from the journal)."""
    bump = 10 * gate.SIGMA
    incumbent = _candidate({0: 0.60, 1: 0.60, 2: 0.60}, {0: 0.55, 1: 0.55, 2: 0.55})
    candidate = _candidate(
        {0: 0.60 - bump, 1: 0.60 - bump, 2: 0.60 - bump},
        {0: 0.55 - bump, 1: 0.55 - bump, 2: 0.55 - bump},
    )

    verdict = gate.compare(candidate, incumbent)

    assert verdict.accept is False
    assert verdict.ci95[1] < 0.0
    assert verdict.reason.startswith("ci_entirely_negative")


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
    # No val_pred_path set on these synthetic candidates, so the gate
    # falls back to the seed-level bootstrap and tags the reason with it
    # (see harness/gate.py's "Never silently degrade" comment).
    assert verdict.reason.startswith("backtest_negative")


def test_empty_backtest_is_rejected_as_backtest_missing():
    bump = 10 * gate.SIGMA
    incumbent = _candidate({0: 0.60, 1: 0.60, 2: 0.60}, {0: 0.55, 1: 0.55, 2: 0.55})
    candidate = _candidate({0: 0.60 + bump, 1: 0.60 + bump, 2: 0.60 + bump}, {})

    verdict = gate.compare(candidate, incumbent)

    assert verdict.accept is False
    assert verdict.reason.startswith("backtest_missing")
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


def test_tiny_same_sign_deltas_are_not_accepted_with_real_per_user_predictions(tmp_path, monkeypatch):
    """Demonstrates the bug the seed-level bootstrap had: three matched
    seeds with tiny, same-signed per-seed deltas that a pure per-seed
    bootstrap can never see cross zero (any resample of same-signed
    numbers stays same-signed), no matter how small they are or how
    little real per-user signal backs them.

    Both candidate and incumbent scores here are independent noise,
    uncorrelated with the label — there is no real effect at all. The
    three seeds below (offsets 17, 45, 30 into a fixed RNG stream) were
    picked because, purely by chance, all three happen to land on the
    positive side (deltas ~0.0027, ~0.0035, ~0.0040) — exactly the kind
    of 1-in-8 fluke harness/gate.py's seed-level fallback comment
    describes. With real per-user predictions available, the user-level
    bootstrap must reveal this as noise and reject.
    """
    monkeypatch.setattr(cache, "_PREDS_DIR", tmp_path / "preds")

    n_users = 150
    rows_per_user = 4
    n_rows = n_users * rows_per_user
    offsets = {0: 17, 1: 45, 2: 30}

    val_for_candidate = {}
    val_for_incumbent = {}
    for seed, offset in offsets.items():
        rng = np.random.default_rng(2000 + offset)
        user_ids = np.repeat(np.arange(n_users), rows_per_user)
        labels = rng.integers(0, 2, size=n_rows)
        incumbent_scores = rng.normal(size=n_rows)
        candidate_scores = rng.normal(size=n_rows)  # independent of incumbent_scores: no real effect

        cache.save_predictions("cand", seed, "val", user_ids, labels, candidate_scores)
        cache.save_predictions("incu", seed, "val", user_ids, labels, incumbent_scores)

        val_for_candidate[seed] = hmetrics.evaluate(user_ids, labels, candidate_scores)
        val_for_incumbent[seed] = hmetrics.evaluate(user_ids, labels, incumbent_scores)

    # Same tiny, same-signed deltas a pure seed-level bootstrap would
    # always accept (assert the premise, so this test fails loudly if the
    # chosen offsets ever stop producing it).
    per_seed_deltas = [
        val_for_candidate[s].primary - val_for_incumbent[s].primary for s in offsets
    ]
    assert all(d > 0 for d in per_seed_deltas)
    assert all(d < 0.01 for d in per_seed_deltas)

    backtest = {s: _metrics(0.6) for s in offsets}
    candidate = CandidateResult(
        config_id="cand", status=Status.OK, val=val_for_candidate, backtest=backtest,
        val_pred_path="artifacts/preds/cand.npz",
    )
    incumbent = CandidateResult(
        config_id="incu", status=Status.OK, val=val_for_incumbent, backtest=backtest,
        val_pred_path="artifacts/preds/incu.npz",
    )

    verdict = gate.compare(candidate, incumbent)

    assert verdict.accept is False
    assert verdict.reason == "ci_includes_zero"
    assert verdict.ci95[0] <= 0.0
    assert "coarse_ci_seed_bootstrap" not in verdict.reason


def test_per_user_metrics_reaggregates_to_vendor_primary_on_real_validation_data():
    """CRITICAL: if this fails, _per_user_metrics's per-user math disagrees
    with the vendor's, and the entire user-level bootstrap built on top of
    it is wrong regardless of what any other gate test shows.

    Uses real validation user_ids/labels (real label distribution: real
    zero-positive/all-positive users, real per-user impression counts,
    real ties) with fixed-seed random scores standing in for a model —
    the helper's aggregation math must reproduce the vendor's primary
    exactly regardless of what scores it's handed.
    """
    val_rows = data.load("val")
    user_ids = np.array([row[1] for row in val_rows])
    labels = np.array([row[6] for row in val_rows], dtype=np.int64)
    rng = np.random.default_rng(0)
    scores = rng.normal(size=len(val_rows))

    unique_users, npos, auc, ndcg = gate._per_user_metrics(user_ids, labels, scores)

    eligible = ~np.isnan(auc)
    gauc = (npos[eligible] * auc[eligible]).sum() / npos[eligible].sum()
    ndcg_mean = ndcg.mean()
    reaggregated_primary = (gauc + ndcg_mean) / 2.0

    vendor_primary = hmetrics.evaluate(user_ids, labels, scores).primary

    assert len(unique_users) == len(set(user_ids))
    assert abs(reaggregated_primary - vendor_primary) < 1e-9, (
        f"helper reaggregation {reaggregated_primary} vs vendor {vendor_primary}"
    )


def test_full_confirm_on_real_validation_data_completes_in_seconds_not_minutes(tmp_path, monkeypatch):
    """Target from the fix: a full 1000-resample CONFIRM on the real
    ~125k-row / ~24k-user validation set must run in seconds, not the old
    ~50-minutes-per-decision cost. Prints the elapsed time so this stays
    visible in test output, not just behind a pass/fail.

    The budget (12s) is deliberately loose. The point being verified is
    "seconds, not ~50 minutes" — a bound tight enough to distinguish 4s
    from 8s adds nothing to that and only buys flakiness: this measured
    4.07-4.72s in isolation but 5.32-5.41s under full-suite load, so 5s
    flaked twice for a distinction the test was never meant to make.
    """
    monkeypatch.setattr(cache, "_PREDS_DIR", tmp_path / "preds")

    val_rows = data.load("val")
    user_ids = np.array([row[1] for row in val_rows])
    labels = np.array([row[6] for row in val_rows], dtype=np.int64)

    val_metrics_candidate = {}
    val_metrics_incumbent = {}
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        incumbent_scores = rng.normal(size=len(val_rows))
        candidate_scores = incumbent_scores + rng.normal(scale=0.01, size=len(val_rows))

        cache.save_predictions("cand", seed, "val", user_ids, labels, candidate_scores)
        cache.save_predictions("incu", seed, "val", user_ids, labels, incumbent_scores)

        val_metrics_candidate[seed] = hmetrics.evaluate(user_ids, labels, candidate_scores)
        val_metrics_incumbent[seed] = hmetrics.evaluate(user_ids, labels, incumbent_scores)

    backtest = {s: _metrics(0.6) for s in (0, 1, 2)}
    candidate = CandidateResult(
        config_id="cand", status=Status.OK, val=val_metrics_candidate, backtest=backtest,
        val_pred_path="artifacts/preds/cand.npz",
    )
    incumbent = CandidateResult(
        config_id="incu", status=Status.OK, val=val_metrics_incumbent, backtest=backtest,
        val_pred_path="artifacts/preds/incu.npz",
    )

    start = time.perf_counter()
    verdict = gate.compare(candidate, incumbent)
    elapsed = time.perf_counter() - start

    print(f"\nfull CONFIRM on real validation data ({len(val_rows):,} rows, "
          f"{len(set(user_ids)):,} users, 3 seeds, 1000 resamples): {elapsed:.2f}s")

    assert "coarse_ci_seed_bootstrap" not in verdict.reason
    assert elapsed < 12.0, f"took {elapsed:.2f}s, target is under 12s"


def test_missing_cached_predictions_falls_back_and_warns(tmp_path, monkeypatch):
    """cache.exists must be the ground truth for the user-level-bootstrap
    decision, not CandidateResult.val_pred_path. Both candidates below
    set val_pred_path (proving the field alone is NOT enough to trigger
    the real bootstrap) but have nothing actually saved in the cache, so
    the gate must fall back, warn, and tag the reason.
    """
    monkeypatch.setattr(cache, "_PREDS_DIR", tmp_path / "preds")

    bump = 10 * gate.SIGMA
    incumbent = _candidate({0: 0.60, 1: 0.60, 2: 0.60}, {0: 0.55, 1: 0.55, 2: 0.55})
    candidate = _candidate(
        {0: 0.60 + bump, 1: 0.60 + bump, 2: 0.60 + bump},
        {0: 0.55 + bump, 1: 0.55 + bump, 2: 0.55 + bump},
    )
    # val_pred_path set on both, but no cache.save_predictions was ever
    # called — nothing actually exists under tmp_path / "preds".
    candidate = dataclasses.replace(candidate, val_pred_path="artifacts/preds/cand__0__val.npz")
    incumbent = dataclasses.replace(incumbent, val_pred_path="artifacts/preds/incu__0__val.npz")

    with pytest.warns(UserWarning, match="cand"):
        verdict = gate.compare(candidate, incumbent)

    assert "coarse_ci_seed_bootstrap" in verdict.reason
