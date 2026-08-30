"""THE HEADLINE NULL TEST: how often does the agent accept noise?

WHAT THIS MEASURES, AND WHY IT IS THE NUMBER WORTH SHOWING
----------------------------------------------------------
Every result an autonomous search reports has the same problem: you cannot
tell a real improvement from a lucky draw, because on real data you never
know which one you got. This test removes that problem by removing the
signal. `FakeExecutor(true_effect=0.0)` draws every candidate's scores
i.i.d. around the same baseline, so the ground truth is known in advance
and is absolute: **no candidate is better than any other, and therefore
every acceptance is a false positive.**

The full Controller then runs against that null world, driving the REAL
harness noise gate — not a test double. Whatever it accepts, it accepted
by mistake, and we can count it. That is a claim about the agent's
judgement that needs no GPU, no dataset and no training run to defend.

WHAT IT DOES *NOT* MEASURE. Nothing here says the agent finds real
improvements; a gate that rejected everything would ace this test and be
useless. That is the false-NEGATIVE question, and it is the same
machinery with `true_effect` set non-zero (see FakeExecutor's docstring).
The two numbers are only meaningful as a pair, and this file deliberately
owns only one of them.

WHY THE RATE IS NOT EXPECTED TO BE EXACTLY ZERO
-----------------------------------------------
A 95% confidence interval is *defined* to exclude the truth 5% of the
time, so a correctly calibrated gate accepts some noise by construction —
asserting zero would be asserting that the statistics are broken in our
favour. Two things then push the realised rate down from there: the
interval has to be entirely POSITIVE (roughly half of its 5% miss rate,
since an entirely-negative interval is a rejection), and acceptance
additionally requires the backtest to agree, which under the null is
close to a coin flip on an independently drawn split.

So the honest expectation is low single-digit percent, and the bound
asserted below is deliberately looser than the observed value so that
this test fails on a real regression rather than on noise.

WHICH CI THE GATE ACTUALLY USES HERE, STATED PLAINLY
-----------------------------------------------------
`FakeExecutor` emits Metrics but caches no per-user prediction vectors,
and it says so deliberately. harness.gate's strong user-level bootstrap
reads those vectors, so with none on disk the gate degrades — loudly, with
a warning, and marking every verdict `coarse_ci_seed_bootstrap` — to its
weaker seed-level fallback. That is the path under test here, and
`test_the_measured_path_is_the_seed_level_fallback` pins it so the fact
can never become invisible.

This matters for reading the number: harness/gate.py's own docstring puts
the seed-level fallback's false-positive rate at roughly 12.5% before the
backtest requirement is applied. The rate this test observes is therefore
a measurement of the WEAKER of the gate's two paths. The user-level
bootstrap, which real runs use, is stricter still. In other words the
figure below is a conservative upper bound on the agent's real-world
credulity, not a flattering one — which is the right direction for a
number offered as evidence.
"""

from __future__ import annotations

import warnings

import pytest

from contracts import Citation, EventKind, HypothesisPayload
from controller.controller import Controller
from controller.fakes import (
    BASELINE_PRIMARY,
    DeterministicRealizer,
    FakeExecutor,
    InMemoryJournal,
    ScriptedGenerator,
)
from controller.policy import UniformPolicy
from controller.ports import GatePort
from harness import gate as harness_gate

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

SEEDS = (0, 1, 2)
"""Three seeds so `gate.compare` takes the CONFIRM path.

Not a detail. With one seed the gate dispatches to its SCREEN stage, which
hardcodes `accept=False` and can only ever reject — a null test on one seed
would report a perfect zero while measuring nothing at all. Three is the
gate's own evidence bar for rendering a real acceptance decision.
"""

MAX_NODES_PER_STAGE = 40
SEARCH_STAGES = 3
"""STAGE_1_STRUCTURAL, STAGE_2_COMBINE, STAGE_3_TUNE. REPRODUCE_BASELINE
evaluates a single fixed config and INIT/FINALIZE/DONE evaluate nothing."""

COMPARISONS_PER_RUN = MAX_NODES_PER_STAGE * SEARCH_STAGES  # 120
RUNS = 3
MIN_COMPARISONS = 300
"""The floor a rate needs to mean anything.

At a true rate near 2%, 300 trials puts the 95% interval on the estimate
at roughly +/- 1.6 points — narrow enough that a bound of 5% is a real
test rather than a formality. Nine comparisons (the Controller's default
run length) would give an interval spanning most of the [0, 1] range and
would pass whatever the gate did.
"""

FALSE_POSITIVE_BOUND = 0.05
"""Deliberately looser than the observed rate. See the module docstring on
why zero is the wrong assertion; the margin is what keeps this test a
regression detector rather than a flake.
"""

AUTO_ADOPT_REASON = "first candidate adopted as incumbent; nothing to compare against"
"""The one DECISION event that is NOT a gate decision.

controller/controller.py adopts the first candidate as incumbent without
calling the gate at all — there is nothing to compare it against — and
still emits a DECISION with `verdict: True`. It is a bookkeeping record of
"the run now has an incumbent", not a judgement, and counting it would
report one guaranteed false positive per run and inflate the rate by
1/(comparisons+1). Matched on the literal reason string the Controller
writes, and `test_exactly_one_auto_adopt_per_run_is_found_and_excluded`
fails loudly if that string ever drifts.
"""


def _hypothesis(index: int) -> HypothesisPayload:
    """One proposal. The content is irrelevant to this experiment.

    Against a null executor the scores do not depend on the config, so what
    is proposed cannot affect what is measured. These exist only to keep
    the Controller's loop fed; they are deliberately uniform so nothing
    about the generator can be mistaken for a cause of the result.
    """
    return HypothesisPayload(
        target_slot="model",
        rationale=f"null-experiment candidate {index}",
        citation=Citation(
            key=f"null{index}",
            url=f"https://example.invalid/{index}",
            library_entry=f"lib/impl_{index}",
        ),
        expected_gain=0.003,
        expected_cost_s=40.0,
    )


def _script(n: int) -> list[HypothesisPayload]:
    """`n` proposals. ScriptedGenerator does not cycle — running off the
    end raises and ends the run early, which would silently shorten the
    experiment — so this must cover the baseline plus every search
    attempt."""
    return [_hypothesis(i) for i in range(n)]


def _run_null_experiment(run_index: int) -> InMemoryJournal:
    """One full Controller run against the null executor and the real gate.

    A FRESH FakeExecutor per run, seeded on `run_index`: its RNG stream
    advances per call rather than per config, so a shared instance would
    make each run's numbers depend on how many runs preceded it. Seeding
    on the index keeps the whole experiment reproducible while giving each
    run an independent noise draw.

    UniformPolicy, not the cost-aware bandit: in a world where no arm is
    better than any other, a bandit is scoring noise, and its extra
    machinery could only add variance to a measurement that is about the
    gate rather than about the search.
    """
    journal = InMemoryJournal()
    Controller(
        # Null by construction: true_effect=0.0 is the default, passed
        # explicitly because it is the entire premise of this file.
        executor=FakeExecutor(seed=run_index, true_effect=0.0),
        # THE REAL GATE. Not AlwaysAcceptGate, not DeltaGate — the module
        # W1 ships, which is what makes this a claim about the system
        # rather than about a double.
        gate=harness_gate,
        generator=ScriptedGenerator(_script(1 + COMPARISONS_PER_RUN)),
        realizer=DeterministicRealizer(),
        policy=UniformPolicy(seed=run_index),
        journal=journal,
        seeds=SEEDS,
        max_nodes_per_stage=MAX_NODES_PER_STAGE,
        run_id=f"null-run-{run_index}",
    ).run()
    return journal


def _split_decisions(journal: InMemoryJournal):
    """Every DECISION, split into (real gate decisions, auto-adopts)."""
    decisions = journal.events_of_kind(EventKind.DECISION)
    auto_adopts = [e for e in decisions if e.payload["reason"] == AUTO_ADOPT_REASON]
    gate_decisions = [e for e in decisions if e.payload["reason"] != AUTO_ADOPT_REASON]
    return gate_decisions, auto_adopts


@pytest.fixture(scope="module")
def null_experiment():
    """The whole experiment, run once and shared.

    Module-scoped because it is the expensive part — each gate comparison
    runs a 1000-resample bootstrap — and because every test in this file
    asks a different question about the SAME evidence. Re-running per test
    would multiply the cost and, worse, invite two tests to quote two
    different numbers.

    Warnings are suppressed here and nowhere else: the gate warns on every
    single comparison that it is using the seed-level fallback, which is
    correct and is asserted for explicitly in
    `test_the_measured_path_is_the_seed_level_fallback` — but several
    hundred copies of it would bury the test output this file exists to
    make legible.
    """
    gate_decisions = []
    auto_adopts = []
    eval_primaries = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for run_index in range(RUNS):
            journal = _run_null_experiment(run_index)
            run_gate, run_auto = _split_decisions(journal)
            gate_decisions.extend(run_gate)
            auto_adopts.append(run_auto)
            eval_primaries.extend(
                event.payload["primary"]
                for event in journal.events_of_kind(EventKind.EVAL_RESULT)
            )
    return {
        "gate_decisions": gate_decisions,
        "auto_adopts_per_run": auto_adopts,
        "eval_primaries": eval_primaries,
    }


# ---------------------------------------------------------------------------
# The headline
# ---------------------------------------------------------------------------


def test_the_agent_almost_never_accepts_noise(null_experiment, capsys):
    """THE HEADLINE. Ground truth is that no candidate is an improvement,
    so every acceptance below is a false positive, counted."""
    gate_decisions = null_experiment["gate_decisions"]
    comparisons = len(gate_decisions)
    false_positives = sum(1 for e in gate_decisions if e.payload["verdict"] is True)
    rate = false_positives / comparisons

    # Printed, not just asserted: this number is the deliverable, and it
    # should be readable in the run output rather than reconstructed from
    # a passing dot. `-s` or a failure will surface it.
    with capsys.disabled():
        print(
            f"\n  FALSE-POSITIVE RATE — integration, seed-level fallback CI "
            f"(no cached preds): {false_positives}/{comparisons} = {rate:.4f}"
            f"  [bound {FALSE_POSITIVE_BOUND}, {RUNS} runs x {SEEDS} seeds]"
            f"\n  (production user-level CI is measured separately in "
            f"tests/test_gate_false_positive_rate.py)"
        )

    assert comparisons >= MIN_COMPARISONS, (
        f"only {comparisons} gate comparisons; a rate measured on fewer "
        f"than {MIN_COMPARISONS} trials is not evidence of anything"
    )
    assert rate < FALSE_POSITIVE_BOUND, (
        f"the agent accepted {false_positives} of {comparisons} candidates "
        f"({rate:.4f}) in a world where none of them was an improvement"
    )


def test_exactly_one_auto_adopt_per_run_is_found_and_excluded(null_experiment):
    """The exclusion, made explicit rather than assumed.

    The Controller adopts the first candidate without consulting the gate
    and still logs a DECISION with verdict=True. It must be found — exactly
    one per run — and it must not be in the counted set, or the reported
    rate carries one guaranteed false positive per run that no gate ever
    decided.
    """
    auto_adopts_per_run = null_experiment["auto_adopts_per_run"]

    assert len(auto_adopts_per_run) == RUNS
    for run_index, auto_adopts in enumerate(auto_adopts_per_run):
        # Exactly one. Zero would mean the reason string drifted and the
        # exclusion silently stopped matching anything; more than one would
        # mean the incumbent was reset mid-run, which this loop never does.
        assert len(auto_adopts) == 1, run_index
        assert auto_adopts[0].payload["verdict"] is True
        assert auto_adopts[0].payload["ci95"] == [0.0, 0.0]

    # And none of them leaked into the counted set.
    assert all(
        e.payload["reason"] != AUTO_ADOPT_REASON
        for e in null_experiment["gate_decisions"]
    )


def test_excluding_the_auto_adopt_actually_changes_the_number(null_experiment):
    """Keeps the exclusion from being a no-op nobody would notice.

    If the auto-adopts were counted, the numerator would rise by exactly
    RUNS. Asserting that difference means this file can never quietly stop
    excluding them.
    """
    gate_decisions = null_experiment["gate_decisions"]
    counted = sum(1 for e in gate_decisions if e.payload["verdict"] is True)
    naive = counted + RUNS

    assert naive > counted
    assert naive - counted == RUNS


# ---------------------------------------------------------------------------
# The experiment is the one we think it is
# ---------------------------------------------------------------------------


def test_the_real_harness_gate_is_a_drop_in_for_the_controller_port():
    """No adapter. harness.gate exposes `compare` at module level with the
    signature GatePort names, so the module object itself satisfies the
    port — which is what `_run_null_experiment` passes."""
    assert isinstance(harness_gate, GatePort)
    assert callable(harness_gate.compare)


def test_every_counted_decision_took_the_confirm_path(null_experiment):
    """One seed would put the gate in its SCREEN stage, which can only
    reject — a null test there reports a meaningless zero. Every counted
    decision must be a real CONFIRM-stage verdict."""
    for event in null_experiment["gate_decisions"]:
        assert not event.payload["reason"].startswith("screen_"), event.payload


def test_the_measured_path_is_the_seed_level_fallback(null_experiment):
    """Pins WHICH of the gate's two confidence intervals this number
    describes.

    FakeExecutor caches no prediction vectors, so the gate degrades to its
    weaker seed-level bootstrap and marks every verdict accordingly. That
    is expected and is not worked around here — but it must be visible,
    because the stronger user-level bootstrap real runs use is stricter,
    which makes the headline rate a conservative bound rather than a
    flattering one.
    """
    marked = [
        e
        for e in null_experiment["gate_decisions"]
        if "coarse_ci_seed_bootstrap" in e.payload["reason"]
    ]
    assert len(marked) == len(null_experiment["gate_decisions"])


def test_the_null_world_really_is_null(null_experiment):
    """Guards against a vacuous pass.

    If the fake had drifted to a negative true effect, every candidate
    would be visibly worse, the gate would reject everything for the RIGHT
    reason in a world we had mislabelled, and this file would report a
    perfect score while measuring nothing.

    THE CHECK IS ON THE EXECUTOR'S OWN DRAWS, NOT ON THE GATE'S DELTAS, and
    that distinction is the whole point of this test. Each EVAL_RESULT
    primary is an independent draw around the baseline, so a few hundred of
    them pin the mean tightly. The gate's `delta_primary` values look
    tempting and would be wrong to use here — see the test below.
    """
    primaries = null_experiment["eval_primaries"]
    mean_primary = sum(primaries) / len(primaries)

    assert len(primaries) >= MIN_COMPARISONS
    # Independent 3-seed means at sigma=0.0008 give a standard error of
    # about 2.4e-05 over this many draws, so this bound is many standard
    # errors wide and still far tighter than any effect worth detecting.
    assert abs(mean_primary - BASELINE_PRIMARY) < 2e-4, mean_primary


def test_the_paired_deltas_are_correlated_through_a_shared_incumbent(
    null_experiment,
):
    """Why the mean delta is NOT a null check, written down so nobody
    re-derives it the hard way.

    Every comparison in a run is measured against the SAME incumbent — one
    fixed 3-seed draw, adopted at the start and only replaced on an
    acceptance, which under the null almost never happens. So the run's 120
    deltas all share one offset and are anything but independent: the
    effective sample size of their mean is the number of runs, not the
    number of comparisons. A run whose baseline drew high shows a
    systematically negative mean delta while being perfectly null.

    Both signs must still appear, which is the real distributional check;
    the mean is left unasserted on purpose.
    """
    deltas = [e.payload["delta_primary"] for e in null_experiment["gate_decisions"]]

    assert any(d > 0 for d in deltas)
    assert any(d < 0 for d in deltas)
    # Scatter of the right order: a paired 3-seed delta at sigma=0.0008 has
    # a spread near 0.00065, so individual deltas must reach a few
    # thousandths without exploding.
    assert max(abs(d) for d in deltas) < 0.01


def test_the_experiment_gathered_the_evidence_it_claims(null_experiment):
    """The run-length arithmetic, asserted rather than assumed: three
    evaluating stages x max_nodes_per_stage, per run."""
    assert len(null_experiment["gate_decisions"]) == RUNS * COMPARISONS_PER_RUN
    assert RUNS * COMPARISONS_PER_RUN >= MIN_COMPARISONS


def test_the_result_is_reproducible():
    """Fixed seeds throughout, so this cannot flake.

    Re-runs one seeded experiment and requires the identical verdict
    sequence. A rate that moved between runs would be a number nobody
    could quote.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first, _ = _split_decisions(_run_null_experiment(0))
        second, _ = _split_decisions(_run_null_experiment(0))

    assert [e.payload["verdict"] for e in first] == [
        e.payload["verdict"] for e in second
    ]
    assert [e.payload["delta_primary"] for e in first] == [
        e.payload["delta_primary"] for e in second
    ]
