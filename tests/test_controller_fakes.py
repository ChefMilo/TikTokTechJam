"""Tests for controller/ports.py and controller/fakes.py.

These pin down the properties the Controller's future tests will lean on:
that the doubles satisfy their ports, that FakeExecutor is reproducible
and correctly calibrated, and that the journal and generator doubles
behave predictably.

Every statistical assertion here is on an interval, never an exact float.
They are additionally deterministic — each draws from a fixed-seed
`random.Random` — but writing them as intervals means a change to the
draw mechanism produces a real signal rather than a spurious failure.
"""

from __future__ import annotations

import math
import statistics

import pytest

from contracts import (
    SLOT_ORDER,
    CandidateResult,
    Citation,
    ErrorClass,
    EventKind,
    HypothesisPayload,
    JournalEvent,
    Metrics,
    PipelineConfig,
    SlotConfig,
    Status,
)
from controller.fakes import (
    BASELINE_GAUC,
    BASELINE_NDCG5,
    BASELINE_PRIMARY,
    BASELINE_SIGMA,
    AlwaysAcceptGate,
    AlwaysRejectGate,
    FakeExecutor,
    InMemoryJournal,
    ScriptedGenerator,
    ScriptExhaustedError,
    mean_primary,
    metrics_from_delta,
)
from controller.ports import ExecutorPort, GatePort, GeneratorPort, JournalPort

# Sample size for the statistical tests. 400 draws puts the standard error
# of a mean difference at sigma*sqrt(2/400) ~ 5.7e-5, two orders of
# magnitude below the effect sizes asserted below.
N_DRAWS = 400


def _config(**overrides: SlotConfig) -> PipelineConfig:
    """A fully-occupied pipeline.

    Every slot must be filled: PipelineConfig.slot_hash walks SLOT_ORDER
    and indexes self.slots directly, so a partial config raises KeyError
    the moment config_id is read.
    """
    slots = {name: SlotConfig(impl=f"{name}_impl") for name in SLOT_ORDER}
    slots.update(overrides)
    return PipelineConfig(slots=slots)


def _hypothesis(method: str) -> HypothesisPayload:
    return HypothesisPayload(
        target_slot="model",
        rationale=f"try {method}",
        citation=Citation(key=method, url=f"https://example.invalid/{method}", library_entry=f"lib#{method}"),
        expected_gain=0.003,
        expected_cost_s=40.0,
    )


def _event(kind: EventKind, seq: int, run_id: str = "run-1") -> JournalEvent:
    return JournalEvent(
        ts=f"2026-08-28T00:00:{seq:02d}Z",
        run_id=run_id,
        iteration=seq,
        node=seq,
        kind=kind,
        payload={"seq": seq},
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "double, port",
    [
        (FakeExecutor(), ExecutorPort),
        (ScriptedGenerator([]), GeneratorPort),
        (InMemoryJournal(), JournalPort),
        (AlwaysAcceptGate(), GatePort),
        (AlwaysRejectGate(), GatePort),
    ],
)
def test_each_fake_satisfies_its_port(double: object, port: type) -> None:
    assert isinstance(double, port)


def test_ports_do_not_match_unrelated_objects() -> None:
    """Guards that the isinstance checks above mean something — a
    runtime_checkable Protocol only looks for method names, so a port that
    matched anything would make the tests above vacuous."""
    assert not isinstance(object(), ExecutorPort)
    assert not isinstance(InMemoryJournal(), ExecutorPort)
    assert not isinstance(FakeExecutor(), GatePort)


# ---------------------------------------------------------------------------
# FakeExecutor: reproducibility
# ---------------------------------------------------------------------------


def test_same_construction_seed_gives_identical_results() -> None:
    config = _config()
    left = FakeExecutor(seed=42).run(config, [0, 1, 2])
    right = FakeExecutor(seed=42).run(config, [0, 1, 2])

    assert left == right
    assert left.val == right.val
    assert left.backtest == right.backtest


def test_different_construction_seeds_give_different_results() -> None:
    config = _config()
    left = FakeExecutor(seed=1).run(config, [0, 1, 2])
    right = FakeExecutor(seed=2).run(config, [0, 1, 2])

    assert left.val != right.val


def test_results_advance_with_the_call_sequence() -> None:
    """Documented sharp edge: the stream advances per call, so the same
    config asked twice gives two different answers. This fake models
    re-running, not caching."""
    executor = FakeExecutor(seed=7)
    config = _config()

    first = executor.run(config, [0])
    second = executor.run(config, [0])

    assert first.config_id == second.config_id
    assert first.val[0].primary != second.val[0].primary


# ---------------------------------------------------------------------------
# FakeExecutor: shape of what it returns
# ---------------------------------------------------------------------------


def test_populates_val_and_backtest_for_every_requested_seed() -> None:
    seeds = [0, 3, 11, 1337]
    result = FakeExecutor(seed=5).run(_config(), seeds)

    assert result.status is Status.OK
    assert set(result.val) == set(seeds)
    assert set(result.backtest) == set(seeds)
    assert result.error_class is ErrorClass.NONE


def test_val_and_backtest_are_drawn_independently() -> None:
    """A backtest that merely echoed val could never catch a candidate
    that wins on validation and loses out of sample."""
    result = FakeExecutor(seed=5).run(_config(), list(range(20)))

    assert result.val != result.backtest
    for seed in result.val:
        assert result.val[seed].primary != result.backtest[seed].primary


def test_metrics_use_the_keys_harness_metrics_actually_emits() -> None:
    """harness/metrics.py:evaluate returns Metrics(values={"GAUC", "nDCG@5"}).
    If the fake emitted different keys, every Controller test would be
    exercising a shape that never occurs in production."""
    result = FakeExecutor(seed=0).run(_config(), [0])

    assert set(result.val[0].values) == {"GAUC", "nDCG@5"}
    assert set(result.backtest[0].values) == {"GAUC", "nDCG@5"}


def test_metrics_from_delta_lands_primary_exactly_on_target() -> None:
    assert metrics_from_delta(0.0).primary == pytest.approx(BASELINE_PRIMARY)
    assert metrics_from_delta(0.01).primary == pytest.approx(BASELINE_PRIMARY + 0.01)
    assert BASELINE_PRIMARY == pytest.approx((BASELINE_GAUC + BASELINE_NDCG5) / 2)
    assert BASELINE_PRIMARY == pytest.approx(0.6016, abs=1e-4)


def test_cost_fields_are_synthetic_and_never_a_clock_reading() -> None:
    one = FakeExecutor(seed=0).run(_config(), [0])
    three = FakeExecutor(seed=0).run(_config(), [0, 1, 2])

    assert three.wall_seconds == pytest.approx(3 * one.wall_seconds)
    assert one.tokens_in > 0 and one.tokens_out > 0


def test_calls_are_recorded_for_later_assertions() -> None:
    executor = FakeExecutor(seed=0)
    config = _config()
    executor.run(config, [0, 1])
    executor.run(config, [2])

    assert executor.calls == [
        (config.config_id, (0, 1)),
        (config.config_id, (2,)),
    ]


# ---------------------------------------------------------------------------
# FakeExecutor: calibration (interval assertions only)
# ---------------------------------------------------------------------------


def test_per_seed_spread_is_consistent_with_baseline_sigma() -> None:
    result = FakeExecutor(seed=99).run(_config(), list(range(N_DRAWS)))
    primaries = [m.primary for m in result.val.values()]

    spread = statistics.stdev(primaries)
    assert 0.5 * BASELINE_SIGMA < spread < 2.0 * BASELINE_SIGMA

    centre = statistics.fmean(primaries)
    assert abs(centre - BASELINE_PRIMARY) < 5 * BASELINE_SIGMA / math.sqrt(N_DRAWS)


def test_null_ground_truth_two_candidates_differ_only_by_noise() -> None:
    """With true_effect=0.0 no candidate is genuinely better than any
    other, so the mean difference between independently drawn candidates
    must be indistinguishable from zero. Every acceptance a Controller
    makes against this executor is therefore a false positive."""
    executor = FakeExecutor(seed=3)
    config = _config()

    diffs = [
        executor.run(config, [0]).val[0].primary
        - executor.run(config, [0]).val[0].primary
        for _ in range(N_DRAWS)
    ]

    standard_error = BASELINE_SIGMA * math.sqrt(2.0 / N_DRAWS)
    assert abs(statistics.fmean(diffs)) < 4 * standard_error


def test_injected_true_effect_is_recovered_within_tolerance() -> None:
    """The mirror image: with a genuine effect present, the same machinery
    measures how reliably it is detected, i.e. false negatives."""
    effect = 0.004
    config = _config()
    treated = FakeExecutor(seed=21, true_effect=effect)
    control = FakeExecutor(seed=22, true_effect=0.0)

    treated_scores = [treated.run(config, [0]).val[0].primary for _ in range(N_DRAWS)]
    control_scores = [control.run(config, [0]).val[0].primary for _ in range(N_DRAWS)]

    recovered = statistics.fmean(treated_scores) - statistics.fmean(control_scores)
    standard_error = BASELINE_SIGMA * math.sqrt(2.0 / N_DRAWS)
    assert abs(recovered - effect) < 6 * standard_error


def test_true_effect_shifts_the_backtest_too() -> None:
    config = _config()
    treated = FakeExecutor(seed=31, true_effect=0.01).run(config, list(range(N_DRAWS)))

    backtest_centre = statistics.fmean(m.primary for m in treated.backtest.values())
    assert abs(backtest_centre - (BASELINE_PRIMARY + 0.01)) < 0.001


# ---------------------------------------------------------------------------
# FakeExecutor: the failure hook
# ---------------------------------------------------------------------------


def test_failure_hook_produces_failed_status_with_requested_error_class() -> None:
    def always_oom(_config: PipelineConfig) -> tuple[ErrorClass, str]:
        return ErrorClass.OOM, "MemoryError: unable to allocate 8.4 GiB"

    result = FakeExecutor(seed=0, fail_on=always_oom).run(_config(), [0, 1])

    assert result.status is Status.FAILED
    assert result.error_class is ErrorClass.OOM
    assert "MemoryError" in result.error_excerpt
    assert result.val == {}
    assert result.backtest == {}


def test_failure_hook_can_target_specific_configs() -> None:
    """The Controller's robustness path needs a world where some configs
    fail and others do not — a blanket failure cannot exercise recovery."""

    def fail_only_explosive(config: PipelineConfig) -> tuple[ErrorClass, str] | None:
        if config.slots["model"].impl == "explodes":
            return ErrorClass.NAN_LOSS, "loss became nan at epoch 3"
        return None

    executor = FakeExecutor(seed=0, fail_on=fail_only_explosive)

    bad = executor.run(_config(model=SlotConfig(impl="explodes")), [0])
    good = executor.run(_config(model=SlotConfig(impl="lightgbm")), [0])

    assert bad.status is Status.FAILED
    assert bad.error_class is ErrorClass.NAN_LOSS
    assert good.status is Status.OK
    assert good.error_class is ErrorClass.NONE
    assert set(good.val) == {0}


def test_failed_candidates_still_report_cost() -> None:
    """A candidate that died halfway still burned budget getting there."""

    def always_timeout(_config: PipelineConfig) -> tuple[ErrorClass, str]:
        return ErrorClass.TIMEOUT, "exceeded 600s"

    result = FakeExecutor(seed=0, fail_on=always_timeout).run(_config(), [0, 1])

    assert result.wall_seconds > 0
    assert result.tokens_in > 0


# ---------------------------------------------------------------------------
# ScriptedGenerator
# ---------------------------------------------------------------------------


def test_scripted_generator_returns_payloads_in_supplied_order() -> None:
    script = [_hypothesis("fm"), _hypothesis("lightgbm"), _hypothesis("dcn")]
    generator = ScriptedGenerator(script)

    returned = [generator.propose({}) for _ in range(3)]

    assert returned == script
    assert [h.citation.key for h in returned] == ["fm", "lightgbm", "dcn"]
    assert generator.remaining == 0


def test_scripted_generator_raises_rather_than_cycling_when_exhausted() -> None:
    generator = ScriptedGenerator([_hypothesis("fm")])
    generator.propose({})

    with pytest.raises(ScriptExhaustedError):
        generator.propose({})


def test_scripted_generator_records_state_cards_defensively() -> None:
    generator = ScriptedGenerator([_hypothesis("fm")])
    card = {"incumbent_primary": 0.6016, "blocked_slots": []}

    generator.propose(card)
    card["incumbent_primary"] = 999.0  # caller mutates afterwards

    assert generator.state_cards[0]["incumbent_primary"] == 0.6016


# ---------------------------------------------------------------------------
# InMemoryJournal
# ---------------------------------------------------------------------------


def test_journal_records_and_replays_in_append_order() -> None:
    journal = InMemoryJournal()
    events = [
        _event(EventKind.RUN_START, 0),
        _event(EventKind.HYPOTHESIS, 1),
        _event(EventKind.EVAL_RESULT, 2),
        _event(EventKind.DECISION, 3),
    ]
    for event in events:
        journal.append(event)

    assert list(journal.replay("run-1")) == events
    assert journal.events == tuple(events)


def test_journal_replay_isolates_runs() -> None:
    journal = InMemoryJournal()
    journal.append(_event(EventKind.RUN_START, 0, run_id="run-a"))
    journal.append(_event(EventKind.RUN_START, 1, run_id="run-b"))
    journal.append(_event(EventKind.DECISION, 2, run_id="run-a"))

    assert [e.run_id for e in journal.replay("run-a")] == ["run-a", "run-a"]
    assert [e.iteration for e in journal.replay("run-a")] == [0, 2]
    assert list(journal.replay("run-missing")) == []


def test_journal_filters_by_event_kind() -> None:
    journal = InMemoryJournal()
    journal.append(_event(EventKind.DECISION, 0, run_id="run-a"))
    journal.append(_event(EventKind.ERROR, 1, run_id="run-a"))
    journal.append(_event(EventKind.DECISION, 2, run_id="run-b"))

    assert len(journal.events_of_kind(EventKind.DECISION)) == 2
    assert len(journal.events_of_kind(EventKind.DECISION, run_id="run-a")) == 1
    assert journal.events_of_kind(EventKind.INTERVENTION) == ()


def test_journal_events_view_cannot_mutate_the_journal() -> None:
    journal = InMemoryJournal()
    journal.append(_event(EventKind.RUN_START, 0))

    assert isinstance(journal.events, tuple)
    assert len(journal.events) == 1


# ---------------------------------------------------------------------------
# Boundary gates
# ---------------------------------------------------------------------------


def _pair() -> tuple[CandidateResult, CandidateResult]:
    candidate = CandidateResult(
        config_id="cand",
        status=Status.OK,
        val={0: metrics_from_delta(0.005), 1: metrics_from_delta(0.003)},
        backtest={0: metrics_from_delta(0.002), 1: metrics_from_delta(0.004)},
    )
    incumbent = CandidateResult(
        config_id="inc",
        status=Status.OK,
        val={0: metrics_from_delta(0.0), 1: metrics_from_delta(0.0)},
        backtest={0: metrics_from_delta(0.0), 1: metrics_from_delta(0.0)},
    )
    return candidate, incumbent


def test_always_accept_gate_accepts() -> None:
    candidate, incumbent = _pair()
    verdict = AlwaysAcceptGate().compare(candidate, incumbent)

    assert verdict.accept is True
    assert verdict.reason


def test_always_reject_gate_rejects() -> None:
    candidate, incumbent = _pair()
    verdict = AlwaysRejectGate().compare(candidate, incumbent)

    assert verdict.accept is False
    assert verdict.reason


def test_gates_report_a_real_paired_delta_and_required_backtest_delta() -> None:
    candidate, incumbent = _pair()
    verdict = AlwaysAcceptGate().compare(candidate, incumbent)

    assert verdict.delta == pytest.approx(0.004)  # mean of +0.005 and +0.003
    assert verdict.backtest_delta == pytest.approx(0.003)  # mean of +0.002, +0.004
    assert isinstance(verdict.backtest_delta, float)
    low, high = verdict.ci95
    assert low < verdict.delta < high


def test_gates_pair_only_on_shared_seeds() -> None:
    candidate = CandidateResult(
        config_id="cand",
        status=Status.OK,
        val={0: metrics_from_delta(0.01), 9: metrics_from_delta(1.0)},
        backtest={0: metrics_from_delta(0.01)},
    )
    incumbent = CandidateResult(
        config_id="inc",
        status=Status.OK,
        val={0: metrics_from_delta(0.0)},
        backtest={0: metrics_from_delta(0.0)},
    )

    verdict = AlwaysRejectGate().compare(candidate, incumbent)

    # Seed 9 is unpaired and must be ignored, not averaged in.
    assert verdict.delta == pytest.approx(0.01)


def test_mean_primary_handles_the_failed_candidate_case() -> None:
    assert mean_primary({}) == 0.0
    assert mean_primary({0: metrics_from_delta(0.0)}) == pytest.approx(BASELINE_PRIMARY)
    assert mean_primary(
        {0: metrics_from_delta(0.0), 1: metrics_from_delta(0.002)}
    ) == pytest.approx(BASELINE_PRIMARY + 0.001)


def test_metrics_helper_matches_contracts_metrics_semantics() -> None:
    """Sanity-check the fake against the real type it is imitating."""
    direct = Metrics(values={"GAUC": BASELINE_GAUC, "nDCG@5": BASELINE_NDCG5})
    assert metrics_from_delta(0.0) == direct
    assert direct.primary == pytest.approx(BASELINE_PRIMARY)
