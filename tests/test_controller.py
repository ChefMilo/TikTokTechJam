"""Tests for controller/state.py and controller/controller.py.

Everything here runs against the deterministic doubles in
controller/fakes.py, so the whole experiment loop is exercised end to end
with no dataset, no GPU, no LLM and no clock dependence beyond the event
timestamps themselves.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from contracts import (
    Budget,
    BudgetCounter,
    Citation,
    ErrorClass,
    EventKind,
    HypothesisPayload,
    PipelineConfig,
    Status,
)
from controller.controller import BASELINE_SLOTS, Controller, baseline_pipeline
from controller.fakes import (
    AlwaysAcceptGate,
    AlwaysRejectGate,
    FakeExecutor,
    InMemoryJournal,
    ScriptedGenerator,
)
from controller.state import (
    STAGE_ORDER,
    STATE_CARD_HISTORY,
    HistoryEntry,
    RunState,
    Stage,
    build_state_card,
    next_stage,
)

# A default run is 1 baseline evaluation plus nodes_per_stage candidates in
# each of the three search stages.
NODES_PER_STAGE = 3
SEARCH_STAGES = 3
SEARCH_CANDIDATES = NODES_PER_STAGE * SEARCH_STAGES  # 9
TOTAL_CANDIDATES = SEARCH_CANDIDATES + 1  # 10, including the baseline


def _hypothesis(index: int, slot: str = "model") -> HypothesisPayload:
    return HypothesisPayload(
        target_slot=slot,
        rationale=f"candidate {index}",
        citation=Citation(
            key=f"paper{index}",
            url=f"https://example.invalid/{index}",
            library_entry=f"lib/impl_{index}",
        ),
        expected_gain=0.003,
        expected_cost_s=40.0,
    )


def _script(n: int = SEARCH_CANDIDATES) -> list[HypothesisPayload]:
    return [_hypothesis(i) for i in range(n)]


def _controller(
    *,
    gate=None,
    executor=None,
    generator=None,
    journal=None,
    budget=None,
    run_id="run-test",
    nodes_per_stage=NODES_PER_STAGE,
) -> tuple[Controller, InMemoryJournal]:
    journal = journal if journal is not None else InMemoryJournal()
    controller = Controller(
        executor=executor if executor is not None else FakeExecutor(seed=1),
        gate=gate if gate is not None else AlwaysAcceptGate(),
        generator=generator if generator is not None else ScriptedGenerator(_script()),
        journal=journal,
        budget=budget,
        nodes_per_stage=nodes_per_stage,
        run_id=run_id,
    )
    return controller, journal


def _kinds(journal: InMemoryJournal) -> list[str]:
    return [e.kind.value for e in journal.events]


# ---------------------------------------------------------------------------
# Stage vocabulary
# ---------------------------------------------------------------------------


def test_stage_order_covers_every_stage_except_done():
    """A stage added to the enum but forgotten in STAGE_ORDER would simply
    never run, silently. This is the guard against that."""
    assert set(STAGE_ORDER) == set(Stage) - {Stage.DONE}
    assert Stage.DONE not in STAGE_ORDER
    assert STAGE_ORDER[0] is Stage.INIT
    assert STAGE_ORDER[-1] is Stage.FINALIZE


def test_stage_order_puts_structural_before_tuning():
    """Load-bearing ordering: tuning early would burn the convergence
    budget on sub-epsilon gains before the structural moves are tried."""
    order = list(STAGE_ORDER)
    assert order.index(Stage.STAGE_1_STRUCTURAL) < order.index(Stage.STAGE_3_TUNE)
    assert order.index(Stage.REPRODUCE_BASELINE) < order.index(Stage.STAGE_1_STRUCTURAL)


def test_next_stage_walks_the_progression_and_ends_at_none():
    assert next_stage(Stage.INIT) is Stage.REPRODUCE_BASELINE
    assert next_stage(Stage.REPRODUCE_BASELINE) is Stage.STAGE_1_STRUCTURAL
    assert next_stage(Stage.STAGE_1_STRUCTURAL) is Stage.STAGE_2_COMBINE
    assert next_stage(Stage.STAGE_2_COMBINE) is Stage.STAGE_3_TUNE
    assert next_stage(Stage.STAGE_3_TUNE) is Stage.FINALIZE
    assert next_stage(Stage.FINALIZE) is Stage.DONE
    assert next_stage(Stage.DONE) is None


def test_stage_is_str_valued_and_json_safe():
    assert Stage.STAGE_1_STRUCTURAL == "stage_1_structural"
    assert json.loads(json.dumps({"s": Stage.FINALIZE}))["s"] == "finalize"


# ---------------------------------------------------------------------------
# RunState
# ---------------------------------------------------------------------------


def test_run_state_is_frozen():
    state = RunState(run_id="r", stage=Stage.INIT)
    with pytest.raises(FrozenInstanceError):
        state.stage = Stage.FINALIZE


def test_run_state_derivations_are_pure():
    """Each with_* returns a new state and leaves the original untouched —
    that is what lets a journal event pin the state as it was."""
    state = RunState(run_id="r", stage=Stage.INIT)

    advanced = state.with_stage(Stage.REPRODUCE_BASELINE).with_node_started()

    assert state.stage is Stage.INIT and state.node == 0
    assert advanced.stage is Stage.REPRODUCE_BASELINE and advanced.node == 1


def test_with_outcome_separates_iteration_from_node():
    state = RunState(run_id="r", stage=Stage.STAGE_1_STRUCTURAL)

    rejected = state.with_node_started().with_outcome("cfg", 0.6, accepted=False)
    accepted = rejected.with_node_started().with_outcome("cfg2", 0.61, accepted=True)

    assert (rejected.node, rejected.iteration) == (1, 0)
    assert (accepted.node, accepted.iteration) == (2, 1)
    assert accepted.history == (
        HistoryEntry("cfg", 0.6, False),
        HistoryEntry("cfg2", 0.61, True),
    )


def test_with_spend_charges_every_counter_from_the_result():
    executor = FakeExecutor(seed=0)
    result = executor.run(baseline_pipeline(), [0, 1])
    state = RunState(
        run_id="r",
        stage=Stage.INIT,
        budget=Budget(
            wall_seconds=BudgetCounter(limit=1000.0),
            tokens=BudgetCounter(limit=10_000.0),
            evaluations=BudgetCounter(limit=10.0),
            gpu_seconds=BudgetCounter(limit=100.0),
        ),
    )

    spent = state.with_spend(result)

    assert spent.budget.wall_seconds.consumed == result.wall_seconds
    assert spent.budget.tokens.consumed == result.tokens_in + result.tokens_out
    assert spent.budget.evaluations.consumed == 1.0
    assert state.budget.evaluations.consumed == 0.0  # original untouched


# ---------------------------------------------------------------------------
# build_state_card — the interface W4 codes against
# ---------------------------------------------------------------------------


def test_state_card_has_the_documented_keys_and_is_strict_json():
    controller, _ = _controller()
    state = controller.run()

    card = build_state_card(state)

    assert set(card) == {
        "run_id",
        "stage",
        "iteration",
        "node",
        "incumbent_config_id",
        "incumbent_primary",
        "recent_history",
        "blocked_slots",
        "budget_remaining",
    }
    # allow_nan=False proves it is RFC-valid JSON, not just Python-parseable:
    # an unmetered budget counter must not leak an `Infinity` token.
    assert json.loads(json.dumps(card, allow_nan=False)) == card


def test_state_card_reports_stage_as_a_string_not_an_enum():
    state = RunState(run_id="r", stage=Stage.STAGE_2_COMBINE)
    card = build_state_card(state)
    assert card["stage"] == "stage_2_combine"
    # `type(...) is str`, not isinstance: Stage members ARE str instances,
    # so isinstance would pass even if the enum member leaked through. W3's
    # renderer reads this string and must not need the enum to interpret it.
    assert type(card["stage"]) is str


def test_state_card_truncates_history_but_run_state_keeps_it_all():
    state = RunState(run_id="r", stage=Stage.STAGE_3_TUNE)
    for i in range(STATE_CARD_HISTORY + 4):
        state = state.with_node_started().with_outcome(f"cfg{i}", 0.6, accepted=False)

    card = build_state_card(state)

    assert len(state.history) == STATE_CARD_HISTORY + 4
    assert len(card["recent_history"]) == STATE_CARD_HISTORY
    assert card["recent_history"][-1]["config_id"] == f"cfg{STATE_CARD_HISTORY + 3}"


def test_state_card_reports_none_for_unmetered_budget_counters():
    card = build_state_card(RunState(run_id="r", stage=Stage.INIT))
    assert card["budget_remaining"] == {
        "wall_seconds": None,
        "tokens": None,
        "evaluations": None,
        "gpu_seconds": None,
    }


def test_state_card_omits_raw_results():
    controller, _ = _controller()
    state = controller.run()
    card = build_state_card(state)

    blob = json.dumps(card)
    assert "CandidateResult" not in blob
    assert "Metrics" not in blob
    assert isinstance(card["incumbent_primary"], float)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_full_run_completes_in_stage_done():
    controller, _ = _controller()

    state = controller.run()

    assert state.stage is Stage.DONE
    assert state.incumbent is not None
    assert state.incumbent_config is not None


def test_event_sequence_is_exactly_as_expected():
    controller, journal = _controller()
    controller.run()

    kinds = _kinds(journal)

    assert kinds[0] == EventKind.RUN_START.value
    assert kinds[-1] == EventKind.RUN_END.value
    assert kinds.count(EventKind.RUN_START.value) == 1
    assert kinds.count(EventKind.RUN_END.value) == 1
    assert kinds.count(EventKind.FINALIZE.value) == 1

    # One STAGE_CHANGE per stage, in STAGE_ORDER.
    stages = [e.payload["stage"] for e in journal.events_of_kind(EventKind.STAGE_CHANGE)]
    assert stages == [s.value for s in STAGE_ORDER]

    # One EVAL_START / EVAL_RESULT / DECISION per candidate. HYPOTHESIS is
    # one fewer: REPRODUCE_BASELINE evaluates a fixed published config and
    # never asks the generator to invent it.
    assert kinds.count(EventKind.EVAL_START.value) == TOTAL_CANDIDATES
    assert kinds.count(EventKind.EVAL_RESULT.value) == TOTAL_CANDIDATES
    assert kinds.count(EventKind.DECISION.value) == TOTAL_CANDIDATES
    assert kinds.count(EventKind.HYPOTHESIS.value) == SEARCH_CANDIDATES


def test_baseline_is_adopted_unconditionally_without_calling_the_gate():
    class ExplodingGate:
        def compare(self, candidate, incumbent):
            raise AssertionError("gate must not be called with a None incumbent")

    journal = InMemoryJournal()
    controller = Controller(
        executor=FakeExecutor(seed=1),
        gate=ExplodingGate(),
        generator=ScriptedGenerator([]),  # exhausts right after the baseline
        journal=journal,
        run_id="run-baseline",
    )

    state = controller.run()

    assert state.iteration == 1  # the baseline became the incumbent
    first_decision = journal.events_of_kind(EventKind.DECISION)[0]
    assert first_decision.payload["verdict"] is True
    assert "nothing to compare against" in first_decision.payload["reason"]


def test_baseline_evaluates_the_published_fm_configuration():
    controller, journal = _controller()
    controller.run()

    first_eval = journal.events_of_kind(EventKind.EVAL_START)[0]
    assert first_eval.payload["config_id"] == baseline_pipeline().config_id
    assert BASELINE_SLOTS["model"].impl == "fm"


# ---------------------------------------------------------------------------
# The two counters mean different things
# ---------------------------------------------------------------------------


def test_reject_gate_leaves_iteration_at_the_baseline_while_node_climbs():
    """The load-bearing test for iteration-vs-node: with nothing ever
    accepted, committed revisions stay at 1 (the baseline) while attempted
    evaluations reach every candidate."""
    controller, journal = _controller(gate=AlwaysRejectGate())

    state = controller.run()

    assert state.iteration == 1
    assert state.node == TOTAL_CANDIDATES
    assert len(journal.events_of_kind(EventKind.DECISION)) == TOTAL_CANDIDATES
    rejected = [
        e for e in journal.events_of_kind(EventKind.DECISION)
        if e.payload["verdict"] is False
    ]
    assert len(rejected) == SEARCH_CANDIDATES


def test_accept_gate_increments_iteration_once_per_candidate():
    controller, _ = _controller(gate=AlwaysAcceptGate())

    state = controller.run()

    assert state.iteration == TOTAL_CANDIDATES
    assert state.node == TOTAL_CANDIDATES


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_a_failed_candidate_does_not_stop_the_run():
    def fail_impl_3(config: PipelineConfig):
        if config.slots["model"].impl == "lib/impl_3":
            return ErrorClass.NAN_LOSS, "loss became nan at epoch 3"
        return None

    controller, journal = _controller(
        executor=FakeExecutor(seed=1, fail_on=fail_impl_3)
    )

    state = controller.run()

    assert state.stage is Stage.DONE
    assert state.node == TOTAL_CANDIDATES  # the failure still consumed a node

    errors = journal.events_of_kind(EventKind.ERROR)
    assert len(errors) == 1
    assert errors[0].payload["error_class"] == ErrorClass.NAN_LOSS.value
    assert "nan" in errors[0].payload["error_excerpt"]

    # The failed candidate got an EVAL_RESULT but no DECISION: there is
    # nothing to decide about a run that produced no scores.
    statuses = [
        e.payload["status"] for e in journal.events_of_kind(EventKind.EVAL_RESULT)
    ]
    assert statuses.count(Status.FAILED.value) == 1
    assert len(journal.events_of_kind(EventKind.DECISION)) == TOTAL_CANDIDATES - 1
    assert _kinds(journal)[-1] == EventKind.RUN_END.value


def test_short_script_ends_the_run_cleanly_without_escaping():
    """ScriptExhaustedError must never reach the caller — a run that ends
    early still has to leave a terminated journal behind."""
    controller, journal = _controller(generator=ScriptedGenerator(_script(4)))

    state = controller.run()  # must not raise

    assert state.stage is Stage.DONE
    kinds = _kinds(journal)
    assert kinds[-1] == EventKind.RUN_END.value
    assert kinds.count(EventKind.FINALIZE.value) == 1

    exhausted = [
        e for e in journal.events_of_kind(EventKind.ERROR)
        if e.payload.get("reason") == "generator_exhausted"
    ]
    assert len(exhausted) == 1
    run_end = journal.events_of_kind(EventKind.RUN_END)[0]
    assert run_end.payload["stop_reason"] == "generator_exhausted"


def test_budget_exhaustion_jumps_to_finalize_without_exceeding_the_limit():
    budget = Budget(evaluations=BudgetCounter(limit=2.0))
    controller, journal = _controller(budget=budget)

    state = controller.run()

    assert state.stage is Stage.DONE
    assert state.budget.evaluations.consumed <= 2.0
    assert state.node == 2

    warnings = journal.events_of_kind(EventKind.BUDGET_WARNING)
    assert len(warnings) == 1
    assert warnings[0].payload["tripped"] == ["evaluations"]

    kinds = _kinds(journal)
    assert kinds.count(EventKind.FINALIZE.value) == 1
    assert kinds[-1] == EventKind.RUN_END.value
    assert journal.events_of_kind(EventKind.RUN_END)[0].payload["stop_reason"] == (
        "budget_exhausted"
    )
    # The stages after the abort never ran.
    stages = [e.payload["stage"] for e in journal.events_of_kind(EventKind.STAGE_CHANGE)]
    assert Stage.STAGE_3_TUNE.value not in stages
    assert stages[-1] == Stage.FINALIZE.value


def test_budget_defaults_to_unlimited_when_none_is_passed():
    controller, _ = _controller(budget=None)
    state = controller.run()

    assert state.budget.wall_seconds.limit == float("inf")
    assert state.stage is Stage.DONE  # nothing tripped


@pytest.mark.parametrize(
    "expected_reason, kwargs",
    [
        ("stages_complete", {}),
        ("generator_exhausted", {"generator": ScriptedGenerator(_script(2))}),
        ("budget_exhausted", {"budget": Budget(evaluations=BudgetCounter(limit=1.0))}),
    ],
)
def test_run_end_is_emitted_on_every_termination_path(expected_reason, kwargs):
    """However a run ends, replay must be able to see that it *did* end —
    that is the whole reason RUN_END was added to EventKind."""
    controller, journal = _controller(**kwargs)

    controller.run()

    assert _kinds(journal)[-1] == EventKind.RUN_END.value
    run_end = journal.events_of_kind(EventKind.RUN_END)[0]
    assert run_end.payload["stop_reason"] == expected_reason


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_identical_runs_emit_identical_event_sequences():
    """Determinism is what makes the loop testable at all. `ts` is excluded
    because it is the one clock-derived field, by design."""

    def fingerprint() -> list[tuple]:
        journal = InMemoryJournal()
        Controller(
            executor=FakeExecutor(seed=7),
            gate=AlwaysAcceptGate(),
            generator=ScriptedGenerator(_script()),
            journal=journal,
            run_id="run-determinism",
        ).run()
        return [
            (e.kind, e.run_id, e.iteration, e.node, json.dumps(e.payload, sort_keys=True))
            for e in journal.events
        ]

    assert fingerprint() == fingerprint()


def test_different_executor_seeds_change_the_numbers_but_not_the_shape():
    def run(seed: int):
        journal = InMemoryJournal()
        Controller(
            executor=FakeExecutor(seed=seed),
            gate=AlwaysAcceptGate(),
            generator=ScriptedGenerator(_script()),
            journal=journal,
            run_id="run-shape",
        ).run()
        return journal

    a, b = run(1), run(2)

    assert _kinds(a) == _kinds(b)
    primaries_a = [e.payload["primary"] for e in a.events_of_kind(EventKind.EVAL_RESULT)]
    primaries_b = [e.payload["primary"] for e in b.events_of_kind(EventKind.EVAL_RESULT)]
    assert primaries_a != primaries_b


def test_controller_records_the_seeds_it_asked_for():
    executor = FakeExecutor(seed=1)
    controller, _ = _controller(executor=executor)
    controller.run()

    assert all(seeds == (0,) for _config_id, seeds in executor.calls)
    assert len(executor.calls) == TOTAL_CANDIDATES
