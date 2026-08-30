"""THE PRODUCTION FALSE-POSITIVE RATE: the noise gate on its real CI path.

WHY THIS EXISTS ALONGSIDE tests/test_false_positive_rate.py
------------------------------------------------------------
That file drives the whole Controller against a null executor and is the
INTEGRATION proof: the loop is wired to the real gate and does not accept
noise. But `FakeExecutor` caches no per-user prediction vectors, so the
gate there degrades to its seed-level fallback — the weaker of its two
confidence intervals, whose own docstring puts the nominal false-positive
rate near 12.5%. The number that file reports is therefore a bound
measured on the wrong path.

This file measures the path production actually takes. It seeds synthetic
null per-user prediction vectors into the cache, so `gate.compare` runs
`_bootstrap_ci_user_level` — resampling the user universe with
replacement, 1000 times, recomputing GAUC and nDCG@5 from precomputed
per-user contributions. That is the real test, and its false-positive rate
is the robustness number worth quoting.

The two files answer different questions and neither replaces the other:
one asks "is the loop wired to the gate and does it hold up end to end",
this one asks "how good is the gate's judgement when it has the evidence
it was designed for".

WHAT MAKES THIS A VALID NULL
-----------------------------
Candidate and incumbent scores are drawn from the SAME data-generating
process: a shared latent relevance signal plus independent noise of equal
scale. So they are exchangeable given the labels — swap their names and
the joint distribution is unchanged. Neither systematically ranks better,
and therefore every acceptance is a false positive with no appeal.

Both are deliberately SKILFUL rather than random: the latent signal is
correlated with the label, so GAUC lands well above 0.5 and nDCG@5 is
non-degenerate. A null built from pure noise would be a null the gate
never faces in practice — the real question is whether it can tell two
equally good models apart, not whether it can spot two useless ones.

THE CONTRAST IS THE POINT
--------------------------
Alongside the gate's verdict, every trial records what a NAIVE agent would
have decided: accept whenever the candidate's mean validation primary is
higher. Under exchangeability that is a coin flip, and it lands near 50%.
Reporting the two side by side is what turns "our gate rejects noise" from
an assertion into a measurement — the gate is not merely conservative, it
is dozens of times more discriminating than the obvious alternative, on
identical data.

CACHE HYGIENE. `harness.cache` writes under `artifacts/preds/` at repo
root with no injectable root, so the module-scoped fixture redirects
`cache._PREDS_DIR` at a pytest temp directory and restores it afterwards.
Two fixed config_ids are reused and overwritten every trial, so the temp
directory holds six files rather than six per trial. Nothing is ever
written under `artifacts/`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from contracts import CandidateResult, Metrics, Status
from harness import cache
from harness import gate as harness_gate
from harness import metrics as harness_metrics

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

TRIALS = 300
"""Independent null comparisons.

At the ~1% rate a well-calibrated gate produces here, 300 trials put the
standard error near 0.006, so the 5% bound below sits roughly seven
standard errors away — a real regression detector rather than a
formality.

300 is the floor rather than a larger number because the cost is about a
third of a second per trial and is dominated by the 1000-resample
bootstrap, which is FIXED work: shrinking the synthetic population barely
moves it (200 users and 150 users time within 15% of each other). More
trials would tighten the estimate without changing any conclusion, and
would push this module past two minutes on its own.
"""

SEEDS = (0, 1, 2)
"""Three, because `gate.compare` dispatches to its CONFIRM path only at
three or more. One seed reaches the SCREEN prefilter, which hardcodes
`accept=False` and would report a meaningless perfect score."""

N_USERS = 200
MIN_IMPRESSIONS = 4
MAX_IMPRESSIONS = 9
POSITIVE_FRACTION = 0.35
ALL_NEGATIVE_EVERY = 10
"""Every tenth user gets no positives at all.

Realistic, and it exercises a code path that would otherwise be dead here:
GAUC counts only users with `0 < npos < impressions`, so these users are
ineligible and must be masked out by `_eligibility_weighted` while still
contributing an nDCG of zero. A population where every user was eligible
would never test that the mask is applied.
"""

LATENT_LABEL_WEIGHT = 0.6
"""How strongly the shared latent signal tracks the label.

Makes both models genuinely skilful rather than random. Applied to the
SHARED latent term only, so it lifts candidate and incumbent identically
and cannot break exchangeability.
"""

BACKTEST_BASE = 0.58
BACKTEST_SIGMA = 0.0008
"""Backtest primaries are drawn as independent, identically distributed
scalars around a common base — an exchangeable null, so `backtest_delta`
is symmetric about zero and clears the gate's `> 0` bar about half the
time. The gate consumes only the point estimate here (see
`_backtest_delta`), so no cached vectors are needed for this half.
"""

POPULATION_SEED = 20260830
TRIAL_SEED_BASE = 900_000

CANDIDATE_CONFIG_ID = "gate_null_candidate"
INCUMBENT_CONFIG_ID = "gate_null_incumbent"
"""Two fixed ids, overwritten every trial, so the temp cache holds six
files total instead of six per trial."""

PRODUCTION_FP_BOUND = 0.05
"""Deliberately looser than the ~1-2% a calibrated 95% interval produces.

Zero would be the wrong assertion: a 95% CI is DEFINED to exclude the
truth 5% of the time, so a gate that never accepted noise would be one
whose statistics were broken in our favour. Acceptance additionally needs
the interval to be entirely POSITIVE (about half of that 5%) and the
backtest to agree (about half again), which is where ~1.25% comes from.
"""

NAIVE_FP_RANGE = (0.40, 0.60)
"""The naive rate is a check on the NULL, not on the gate.

If the two arms were not really exchangeable — if one had drifted
skilful — this would slide away from 0.5 and reveal that the experiment
had stopped being a null. Its width absorbs binomial noise at this
trial count (standard error ~0.029).
"""

FALLBACK_WARNING_FRAGMENT = "falling back to the weaker seed-level bootstrap"
"""The exact thing this file exists to NOT measure.

If cached predictions are missing for any (config_id, seed), the gate
warns with this text and silently uses the weak CI. That is precisely the
bug this test corrects, so its absence is asserted rather than assumed:
without the guard, a broken cache redirect would leave this file quietly
re-measuring the fallback under a heading that claims otherwise.
"""


# ---------------------------------------------------------------------------
# The synthetic population and the null
# ---------------------------------------------------------------------------


def _build_population() -> tuple[np.ndarray, np.ndarray]:
    """A fixed user population: user_ids and binary labels, row-aligned.

    Deterministic from POPULATION_SEED and identical for every trial and
    every seed — only the SCORES vary. That mirrors reality (validation
    rows are the same rows however the model changes) and it is also
    required: `_bootstrap_ci_user_level` raises if candidate and incumbent
    do not share identical user_ids and labels.
    """
    rng = np.random.default_rng(POPULATION_SEED)
    impressions = rng.integers(MIN_IMPRESSIONS, MAX_IMPRESSIONS + 1, size=N_USERS)

    user_ids = np.repeat(np.arange(N_USERS), impressions)
    label_blocks = []
    for user_index, n_impressions in enumerate(impressions):
        if user_index % ALL_NEGATIVE_EVERY == 0:
            n_positive = 0
        else:
            # At least one positive and at least one negative, so the user
            # is GAUC-eligible.
            n_positive = max(1, int(n_impressions * POSITIVE_FRACTION))
        block = np.array([1] * n_positive + [0] * (n_impressions - n_positive))
        label_blocks.append(rng.permutation(block))

    return user_ids, np.concatenate(label_blocks).astype(np.int64)


def _exchangeable_scores(
    rng: np.random.Generator, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate and incumbent score vectors drawn from ONE process.

    `latent` is the shared relevance signal both models partially capture;
    each then gets independent noise of the SAME scale. Swapping the two
    return values leaves the joint distribution unchanged, which is the
    formal statement of "no candidate is better" — and the reason every
    acceptance downstream is a false positive rather than a judgement call.
    """
    latent = rng.normal(size=len(labels)) + LATENT_LABEL_WEIGHT * labels
    candidate = latent + rng.normal(size=len(labels))
    incumbent = latent + rng.normal(size=len(labels))
    return candidate, incumbent


def _flat_metrics(primary: float) -> Metrics:
    """Metrics whose primary is exactly `primary`.

    `Metrics.primary` is the unweighted mean of the values, so setting both
    components equal makes the primary exact — a test can then reason about
    the primary directly instead of modelling how GAUC and nDCG@5 move
    against each other, which this null is not trying to simulate.
    """
    return Metrics(values={"GAUC": primary, "nDCG@5": primary})


def _exchangeable_backtest(rng: np.random.Generator) -> tuple[Metrics, Metrics]:
    """One seed's backtest Metrics for candidate and incumbent.

    Two i.i.d. draws around a common base: same distribution, so the
    resulting `backtest_delta` is symmetric about zero.
    """
    return (
        _flat_metrics(BACKTEST_BASE + rng.normal(scale=BACKTEST_SIGMA)),
        _flat_metrics(BACKTEST_BASE + rng.normal(scale=BACKTEST_SIGMA)),
    )


def _run_one_trial(
    trial: int, user_ids: np.ndarray, labels: np.ndarray
) -> dict:
    """One null comparison through the real gate on its production path.

    Per seed: draw exchangeable score vectors, cache BOTH so the gate finds
    per-user predictions for candidate and incumbent alike, and compute the
    validation Metrics from those SAME vectors — so the verdict's `delta`
    (from `.val`) and its interval (from the cached vectors) describe one
    consistent world rather than two.
    """
    rng = np.random.default_rng(TRIAL_SEED_BASE + trial)

    val_candidate: dict[int, Metrics] = {}
    val_incumbent: dict[int, Metrics] = {}
    backtest_candidate: dict[int, Metrics] = {}
    backtest_incumbent: dict[int, Metrics] = {}

    for seed in SEEDS:
        candidate_scores, incumbent_scores = _exchangeable_scores(rng, labels)

        cache.save_predictions(
            CANDIDATE_CONFIG_ID, seed, "val", user_ids, labels, candidate_scores
        )
        cache.save_predictions(
            INCUMBENT_CONFIG_ID, seed, "val", user_ids, labels, incumbent_scores
        )

        val_candidate[seed] = harness_metrics.evaluate(
            user_ids, labels, candidate_scores
        )
        val_incumbent[seed] = harness_metrics.evaluate(
            user_ids, labels, incumbent_scores
        )
        backtest_candidate[seed], backtest_incumbent[seed] = _exchangeable_backtest(rng)

    candidate = CandidateResult(
        config_id=CANDIDATE_CONFIG_ID,
        status=Status.OK,
        val=val_candidate,
        backtest=backtest_candidate,
    )
    incumbent = CandidateResult(
        config_id=INCUMBENT_CONFIG_ID,
        status=Status.OK,
        val=val_incumbent,
        backtest=backtest_incumbent,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verdict = harness_gate.compare(candidate, incumbent)
    fell_back = any(
        FALLBACK_WARNING_FRAGMENT in str(warning.message) for warning in caught
    )

    # What an agent with no statistics would have decided on the same data.
    naive_accept = _mean_primary(val_candidate) > _mean_primary(val_incumbent)

    return {
        "accept": verdict.accept,
        "delta": verdict.delta,
        "ci95": verdict.ci95,
        "n_seeds": verdict.n_seeds,
        "reason": verdict.reason,
        "fell_back": fell_back,
        "naive_accept": naive_accept,
    }


def _mean_primary(per_seed: dict[int, Metrics]) -> float:
    return sum(m.primary for m in per_seed.values()) / len(per_seed)


@pytest.fixture(scope="module")
def gate_null_experiment(tmp_path_factory):
    """Every trial, run once and shared across the assertions below.

    Module-scoped because the bootstrap is the expensive part and every
    test here interrogates the SAME evidence — re-running per test would
    multiply a ~100s cost and, worse, let two tests quote two different
    numbers for the same claim.

    `pytest.MonkeyPatch.context()` rather than the `monkeypatch` fixture:
    that fixture is function-scoped and cannot be used from a module-scoped
    one. The redirect is undone on exit, so `harness.cache` is left pointing
    at its real directory for every other test in the suite.
    """
    preds_dir = tmp_path_factory.mktemp("gate_null_preds")
    user_ids, labels = _build_population()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cache, "_PREDS_DIR", preds_dir)
        trials = [_run_one_trial(t, user_ids, labels) for t in range(TRIALS)]

    return {
        "trials": trials,
        "preds_dir": preds_dir,
        "user_ids": user_ids,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# The headline: production FP rate, and the contrast that gives it meaning
# ---------------------------------------------------------------------------


def test_the_gate_almost_never_accepts_noise_on_its_production_path(
    gate_null_experiment, capsys
):
    """THE HEADLINE. Candidate and incumbent are exchangeable by
    construction, so every acceptance is a false positive, counted."""
    trials = gate_null_experiment["trials"]
    gate_false_positives = sum(1 for t in trials if t["accept"])
    naive_false_positives = sum(1 for t in trials if t["naive_accept"])
    gate_rate = gate_false_positives / len(trials)
    naive_rate = naive_false_positives / len(trials)

    with capsys.disabled():
        print(
            f"\n  FALSE-POSITIVE RATE — production, user-level bootstrap CI "
            f"(cached preds): {gate_false_positives}/{len(trials)} = {gate_rate:.4f}"
            f"\n  FALSE-POSITIVE RATE — naive 'higher mean primary wins' baseline: "
            f"{naive_false_positives}/{len(trials)} = {naive_rate:.4f}"
            f"\n  contrast: the gate is {naive_rate / max(gate_rate, 1 / len(trials)):.0f}x "
            f"more discriminating on identical data "
            f"[{len(trials)} trials x {len(SEEDS)} seeds x {N_USERS} users]"
        )

    assert gate_rate < PRODUCTION_FP_BOUND, (
        f"the gate accepted {gate_false_positives} of {len(trials)} candidates "
        f"({gate_rate:.4f}) drawn from the same distribution as the incumbent"
    )


def test_the_naive_baseline_accepts_about_half_of_the_same_null_trials(
    gate_null_experiment,
):
    """Two things at once, and both matter.

    It is the CONTRAST that makes the headline meaningful — without it,
    "the gate rejects almost everything" is equally consistent with a gate
    that rejects everything unconditionally.

    It is also the check that the null is genuinely symmetric. If candidate
    and incumbent had stopped being exchangeable, this would drift off 0.5
    and say so.
    """
    trials = gate_null_experiment["trials"]
    naive_rate = sum(1 for t in trials if t["naive_accept"]) / len(trials)

    low, high = NAIVE_FP_RANGE
    assert low < naive_rate < high, naive_rate


def test_the_gate_is_dramatically_more_discriminating_than_the_naive_rule(
    gate_null_experiment,
):
    """The comparison stated as an assertion, not left to the eye."""
    trials = gate_null_experiment["trials"]
    gate_rate = sum(1 for t in trials if t["accept"]) / len(trials)
    naive_rate = sum(1 for t in trials if t["naive_accept"]) / len(trials)

    assert gate_rate * 5 < naive_rate


# ---------------------------------------------------------------------------
# THE GUARD: this measured the production path, not the fallback
# ---------------------------------------------------------------------------


def test_no_trial_fell_back_to_the_weak_seed_level_bootstrap(gate_null_experiment):
    """The guard this whole file turns on.

    The gate degrades silently-but-loudly to its weak CI when cached
    predictions are missing, and that fallback is what the Controller-level
    test already measures. If the cache redirect broke, or a config_id
    stopped matching, this file would keep passing while quietly reporting
    the wrong number under the right heading. So the warning's ABSENCE is
    asserted on every single trial.
    """
    fell_back = [index for index, t in enumerate(gate_null_experiment["trials"]) if t["fell_back"]]
    assert fell_back == [], (
        f"{len(fell_back)} trial(s) used the seed-level fallback; this test "
        "is supposed to measure the user-level bootstrap"
    )


def test_no_verdict_is_marked_with_the_coarse_ci_note(gate_null_experiment):
    """The same fact from the other side: the gate's own machine-readable
    marker. A verdict computed with the weak CI carries
    'coarse_ci_seed_bootstrap' in its reason, and none here may."""
    for trial in gate_null_experiment["trials"]:
        assert "coarse_ci_seed_bootstrap" not in trial["reason"]


def test_every_trial_reached_the_confirm_path_on_three_seeds(gate_null_experiment):
    """One seed would land in the SCREEN prefilter, which can only reject —
    a null test there reports a perfect score while measuring nothing."""
    for trial in gate_null_experiment["trials"]:
        assert trial["n_seeds"] == len(SEEDS)
        assert not trial["reason"].startswith("screen_")


# ---------------------------------------------------------------------------
# The world really is null
# ---------------------------------------------------------------------------


def test_the_paired_deltas_scatter_around_zero(gate_null_experiment):
    """Under exchangeability the mean paired delta must sit at zero.

    Unlike the Controller-level test — where every comparison in a run
    shares one fixed incumbent and the deltas are correlated through it —
    these trials are independent, so their mean IS a valid null check and
    is asserted as one.
    """
    deltas = [t["delta"] for t in gate_null_experiment["trials"]]
    mean_delta = sum(deltas) / len(deltas)
    spread = float(np.std(deltas))

    assert any(d > 0 for d in deltas)
    assert any(d < 0 for d in deltas)
    # Independent trials, so the standard error of this mean is
    # spread/sqrt(TRIALS); three of those is a wide, non-flaky bound.
    assert abs(mean_delta) < 3 * spread / np.sqrt(len(deltas)), (mean_delta, spread)


def test_both_arms_are_genuinely_skilful_not_two_random_rankers(
    gate_null_experiment,
):
    """A null of two USELESS models is not the null the gate faces.

    The shared latent signal tracks the label, so both arms should rank
    well above chance. If this ever collapsed to 0.5 the experiment would
    still be exchangeable but would no longer resemble the decision the
    gate is asked to make in production.
    """
    user_ids = gate_null_experiment["user_ids"]
    labels = gate_null_experiment["labels"]
    rng = np.random.default_rng(TRIAL_SEED_BASE)
    candidate_scores, incumbent_scores = _exchangeable_scores(rng, labels)

    candidate = harness_metrics.evaluate(user_ids, labels, candidate_scores)
    incumbent = harness_metrics.evaluate(user_ids, labels, incumbent_scores)

    assert candidate.values["GAUC"] > 0.6
    assert incumbent.values["GAUC"] > 0.6


def test_the_population_has_both_eligible_and_ineligible_users(
    gate_null_experiment,
):
    """The GAUC eligibility mask must actually be exercised: GAUC counts
    only users with 0 < npos < impressions, and a population where every
    user qualified would never test that the others are excluded."""
    user_ids = gate_null_experiment["user_ids"]
    labels = gate_null_experiment["labels"]

    eligible = ineligible = 0
    for user in np.unique(user_ids):
        user_labels = labels[user_ids == user]
        n_positive = int(user_labels.sum())
        if 0 < n_positive < len(user_labels):
            eligible += 1
        else:
            ineligible += 1

    assert eligible > 0 and ineligible > 0
    assert eligible > ineligible  # ineligible users are the minority


# ---------------------------------------------------------------------------
# Reproducibility and hygiene
# ---------------------------------------------------------------------------


def test_the_experiment_is_reproducible(tmp_path):
    """Every seed is fixed, so a re-run of one trial must reproduce the
    verdict exactly. A rate that moved between runs would be a number
    nobody could quote."""
    user_ids, labels = _build_population()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cache, "_PREDS_DIR", tmp_path / "preds")
        first = _run_one_trial(0, user_ids, labels)
        second = _run_one_trial(0, user_ids, labels)

    assert first["accept"] == second["accept"]
    assert first["delta"] == second["delta"]
    assert first["ci95"] == second["ci95"]


def test_the_population_is_identical_across_calls():
    """The rows must not move between trials — `_bootstrap_ci_user_level`
    raises outright if candidate and incumbent disagree about user_ids or
    labels, and a drifting population would also make trials
    incomparable."""
    first_users, first_labels = _build_population()
    second_users, second_labels = _build_population()

    assert np.array_equal(first_users, second_users)
    assert np.array_equal(first_labels, second_labels)


def test_predictions_were_written_to_the_temp_dir_and_not_to_artifacts(
    gate_null_experiment,
):
    """Cache hygiene, asserted rather than hoped for.

    Six files — two config_ids x three seeds — overwritten every trial
    rather than accumulating, and all of them under pytest's temp
    directory. `harness.cache` has no injectable root, so without the
    redirect this experiment would drop files into the same `artifacts/`
    directory real runs use.
    """
    preds_dir = gate_null_experiment["preds_dir"]
    written = sorted(p.name for p in preds_dir.glob("*.npz"))

    assert len(written) == len(SEEDS) * 2, written
    for seed in SEEDS:
        assert f"{CANDIDATE_CONFIG_ID}__{seed}__val.npz" in written
        assert f"{INCUMBENT_CONFIG_ID}__{seed}__val.npz" in written

    # The redirect was undone on fixture exit, so the module is no longer
    # pointing anywhere near this experiment's files...
    assert cache._PREDS_DIR != preds_dir

    # ...and nothing this module writes ever landed in the real directory.
    if cache._PREDS_DIR.exists():
        littered = [
            path.name
            for path in cache._PREDS_DIR.glob("*.npz")
            if path.name.startswith((CANDIDATE_CONFIG_ID, INCUMBENT_CONFIG_ID))
        ]
        assert littered == []
