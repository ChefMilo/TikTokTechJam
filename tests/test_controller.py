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
    CandidateResult,
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
    DeltaGate,
    DeterministicRealizer,
    FakeExecutor,
    InMemoryJournal,
    ScriptedGenerator,
    ScriptedRealizer,
    metrics_from_delta,
)
from controller.ports import RealizerExhausted
from controller import convergence
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
MAX_NODES_PER_STAGE = 3
SEARCH_STAGES = 3
SEARCH_CANDIDATES = MAX_NODES_PER_STAGE * SEARCH_STAGES  # 9
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


class _ClimbingExecutor:
    """Primaries that climb by a fixed step, so neither rule converges.

    FakeExecutor draws i.i.d. around a constant, which is exactly right for
    testing acceptance logic and exactly wrong for testing loop mechanics:
    differenced primaries hover near zero, so the organizers' rule reads
    every iteration as flat and the run converges after four commits.
    Tests that need a run to go the distance need scores that genuinely
    improve by more than epsilon each time.
    """

    def __init__(self, step: float = 0.01, fail_on=None) -> None:
        self.step = step
        self._fail_on = fail_on
        self._committed = 0
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def run(self, config: PipelineConfig, seeds) -> CandidateResult:
        seeds = tuple(seeds)
        self.calls.append((config.config_id, seeds))
        failure = self._fail_on(config) if self._fail_on is not None else None
        if failure is not None:
            error_class, excerpt = failure
            return CandidateResult(
                config_id=config.config_id,
                status=Status.FAILED,
                val={},
                backtest={},
                error_class=error_class,
                error_excerpt=excerpt,
                wall_seconds=40.0 * len(seeds),
                tokens_in=900,
                tokens_out=300,
            )
        metrics = metrics_from_delta(self.step * self._committed)
        self._committed += 1
        return CandidateResult(
            config_id=config.config_id,
            status=Status.OK,
            val={seed: metrics for seed in seeds},
            backtest={seed: metrics for seed in seeds},
            wall_seconds=40.0 * len(seeds),
            tokens_in=900,
            tokens_out=300,
        )


def _controller(
    *,
    gate=None,
    executor=None,
    generator=None,
    realizer=None,
    journal=None,
    budget=None,
    run_id="run-test",
    max_nodes_per_stage=MAX_NODES_PER_STAGE,
) -> tuple[Controller, InMemoryJournal]:
    journal = journal if journal is not None else InMemoryJournal()
    controller = Controller(
        # Climbing by default so the mechanics tests still see a full run:
        # with convergence live, a flat executor stops the loop after four
        # commits and every count below would be measuring convergence
        # instead of the thing it names.
        executor=executor if executor is not None else _ClimbingExecutor(),
        gate=gate if gate is not None else AlwaysAcceptGate(),
        generator=generator if generator is not None else ScriptedGenerator(_script()),
        # DeterministicRealizer by default: a test about the loop should not
        # have to maintain a config script in lockstep with its hypothesis
        # script just to get a candidate out the other end.
        realizer=realizer if realizer is not None else DeterministicRealizer(),
        journal=journal,
        budget=budget,
            max_nodes_per_stage=max_nodes_per_stage,
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
        "convergence",
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
    # One CODE_EMITTED per realized candidate — the baseline is not
    # realized, so it matches HYPOTHESIS rather than EVAL_START.
    assert kinds.count(EventKind.CODE_EMITTED.value) == SEARCH_CANDIDATES


def test_baseline_is_adopted_unconditionally_without_calling_the_gate():
    class ExplodingGate:
        def compare(self, candidate, incumbent):
            raise AssertionError("gate must not be called with a None incumbent")

    journal = InMemoryJournal()
    controller = Controller(
        executor=FakeExecutor(seed=1),
        gate=ExplodingGate(),
        generator=ScriptedGenerator([]),  # exhausts right after the baseline
        realizer=DeterministicRealizer(),
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
        executor=_ClimbingExecutor(fail_on=fail_impl_3)
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
            realizer=DeterministicRealizer(),
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
            realizer=DeterministicRealizer(),
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
    executor = _ClimbingExecutor()
    controller, _ = _controller(executor=executor)
    controller.run()

    assert all(seeds == (0,) for _config_id, seeds in executor.calls)
    assert len(executor.calls) == TOTAL_CANDIDATES


# ---------------------------------------------------------------------------
# The realizer seam — appended. Nothing above this line is modified.
# ---------------------------------------------------------------------------

import ast
import pathlib

from contracts import SlotConfig
import controller.controller as controller_module


class _RecordingExecutor:
    """Delegates to a FakeExecutor while keeping the PipelineConfig objects.

    FakeExecutor.calls records only (config_id, seeds); the lineage
    assertions below need the config itself.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.configs: list[PipelineConfig] = []

    def run(self, config: PipelineConfig, seeds):
        self.configs.append(config)
        return self._inner.run(config, seeds)


class _FlakyRealizer:
    """Realizes everything except one nominated hypothesis."""

    def __init__(self, fail_on_key: str) -> None:
        self._fail_on_key = fail_on_key
        self._inner = DeterministicRealizer()

    def realize(self, hypothesis: HypothesisPayload) -> SlotConfig:
        if hypothesis.citation.key == self._fail_on_key:
            raise RealizerExhausted(
                f"no library entry realizable for {hypothesis.citation.key}"
            )
        return self._inner.realize(hypothesis)


def test_controller_module_imports_nothing_from_the_test_doubles():
    """Structural, not a comment: parses controller.py's own import
    statements so the layering wart cannot silently come back."""
    source = pathlib.Path(controller_module.__file__).read_text(encoding="utf-8")
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)

    assert "controller.fakes" not in modules
    assert not any(m.endswith("fakes") for m in modules), modules
    assert "controller.ports" in modules  # it does still depend on the seams


def test_realizer_is_required_and_has_no_default():
    """A default would silently supply the very wiring this seam exists to
    make explicit."""
    with pytest.raises(TypeError):
        Controller(
            executor=FakeExecutor(seed=0),
            gate=AlwaysAcceptGate(),
            generator=ScriptedGenerator(_script()),
            journal=InMemoryJournal(),
        )


def test_code_emitted_sits_between_hypothesis_and_eval_start():
    controller, journal = _controller()
    controller.run()

    kinds = _kinds(journal)
    positions = [i for i, k in enumerate(kinds) if k == EventKind.CODE_EMITTED.value]

    assert len(positions) == SEARCH_CANDIDATES
    for i in positions:
        assert kinds[i - 1] == EventKind.HYPOTHESIS.value
        assert kinds[i + 1] == EventKind.EVAL_START.value


def test_baseline_emits_no_code_emitted():
    """REPRODUCE_BASELINE evaluates a published config directly — nothing
    is proposed and nothing is realized."""
    controller, journal = _controller()
    controller.run()

    kinds = _kinds(journal)
    first_eval = kinds.index(EventKind.EVAL_START.value)
    assert EventKind.CODE_EMITTED.value not in kinds[:first_eval]


def test_code_emitted_payload_records_the_realized_slot():
    controller, journal = _controller()
    controller.run()

    payload = journal.events_of_kind(EventKind.CODE_EMITTED)[0].payload

    assert set(payload) == {
        "target_slot",
        "impl",
        "has_code_blob",
        "code_blob_chars",
    }
    assert payload["target_slot"] == "model"
    assert payload["impl"] == "lib/impl_0"
    assert payload["has_code_blob"] is False
    assert payload["code_blob_chars"] == 0


def test_code_emitted_never_carries_the_code_blob_itself():
    """A journal that inlines generated source stops being readable. Record
    that code exists and how much of it, never the text."""
    blob = "def score(x):" + ("\n    pass" * 400)
    realizer = ScriptedRealizer(
        [SlotConfig(impl="custom", code_blob=blob) for _ in range(SEARCH_CANDIDATES)]
    )
    controller, journal = _controller(realizer=realizer)
    controller.run()

    assert len(blob) > 1000  # long enough that inlining would be obvious
    for event in journal.events_of_kind(EventKind.CODE_EMITTED):
        assert event.payload["has_code_blob"] is True
        assert event.payload["code_blob_chars"] == len(blob)
        rendered = json.dumps(event.payload)
        assert blob not in rendered
        assert "pass" not in rendered


def test_realized_candidate_records_lineage_and_differs_in_one_slot():
    """parent_id is what makes the search tree reconstructable from the
    journal alone."""
    executor = _RecordingExecutor(_ClimbingExecutor())
    controller, _ = _controller(executor=executor, gate=AlwaysAcceptGate())
    controller.run()

    baseline = executor.configs[0]
    first = executor.configs[1]
    second = executor.configs[2]

    assert baseline.parent_id is None  # nothing preceded it
    assert first.parent_id == baseline.config_id
    # AlwaysAcceptGate promotes every candidate, so the next one descends
    # from the one immediately before it.
    assert second.parent_id == first.config_id

    differing = [s for s in baseline.slots if baseline.slots[s] != first.slots[s]]
    assert differing == ["model"]  # exactly the target slot, nothing else


def test_lineage_does_not_perturb_identity():
    """parent_id is not part of the content hash, so recording ancestry
    never changes a candidate's cache key."""
    slots = dict(BASELINE_SLOTS)
    without = PipelineConfig(slots=slots)
    with_parent = PipelineConfig(slots=slots, parent_id="some-ancestor")

    assert without.config_id == with_parent.config_id


def test_realizer_failure_kills_the_candidate_not_the_run():
    realizer = _FlakyRealizer(fail_on_key="paper4")
    controller, journal = _controller(realizer=realizer)

    state = controller.run()

    assert state.stage is Stage.DONE
    assert state.node == TOTAL_CANDIDATES  # the dead candidate still counted

    errors = [
        e
        for e in journal.events_of_kind(EventKind.ERROR)
        if e.payload.get("reason") == "realizer_exhausted"
    ]
    assert len(errors) == 1
    assert errors[0].payload["error_class"] == ErrorClass.CONTRACT.value
    assert errors[0].payload["target_slot"] == "model"
    assert "paper4" in errors[0].payload["error_excerpt"]

    # That candidate never reached the executor, so it produced no
    # CODE_EMITTED, no EVAL_START and no DECISION — every other one did.
    kinds = _kinds(journal)
    assert kinds.count(EventKind.HYPOTHESIS.value) == SEARCH_CANDIDATES
    assert kinds.count(EventKind.CODE_EMITTED.value) == SEARCH_CANDIDATES - 1
    assert kinds.count(EventKind.EVAL_START.value) == TOTAL_CANDIDATES - 1
    assert kinds.count(EventKind.DECISION.value) == TOTAL_CANDIDATES - 1
    assert kinds[-1] == EventKind.RUN_END.value
    assert journal.events_of_kind(EventKind.RUN_END)[0].payload["stop_reason"] == (
        "stages_complete"
    )


def test_unrealized_candidate_shows_up_in_history_as_a_sentinel():
    realizer = _FlakyRealizer(fail_on_key="paper0")
    controller, _ = _controller(realizer=realizer)

    state = controller.run()

    sentinels = [
        h for h in state.history if h.config_id == controller_module.UNREALIZED_CONFIG_ID
    ]
    assert len(sentinels) == 1
    assert sentinels[0].primary is None
    assert sentinels[0].accepted is False


def test_realizer_receives_every_non_baseline_hypothesis():
    realizer = ScriptedRealizer(
        [SlotConfig(impl=f"scripted_{i}") for i in range(SEARCH_CANDIDATES)]
    )
    controller, _ = _controller(realizer=realizer)
    controller.run()

    assert len(realizer.calls) == SEARCH_CANDIDATES
    assert [h.citation.key for h in realizer.calls] == [
        f"paper{i}" for i in range(SEARCH_CANDIDATES)
    ]


# ---------------------------------------------------------------------------
# Convergence — appended. Nothing above this line is modified.
# ---------------------------------------------------------------------------


def _convergence_events(journal: InMemoryJournal):
    return journal.events_of_kind(EventKind.CONVERGENCE_CHECK)


def test_a_flat_run_converges_and_says_so():
    """DeltaGate(0.0005) gives a ci95 straddling zero, so no candidate is
    significantly better; FakeExecutor's flat scores also make the
    organizers' rule fire. Either way the run must stop early and say why."""
    controller, journal = _controller(
        executor=FakeExecutor(seed=1), gate=DeltaGate(delta=0.0005)
    )

    state = controller.run()

    assert state.stage is Stage.DONE
    run_end = journal.events_of_kind(EventKind.RUN_END)[0]
    assert run_end.payload["stop_reason"] == "converged"
    assert journal.events_of_kind(EventKind.FINALIZE)[0].payload["stop_reason"] == (
        "converged"
    )
    assert state.node < TOTAL_CANDIDATES  # stopped before the cap


def test_convergence_check_is_emitted_after_every_commit():
    controller, journal = _controller(
        executor=FakeExecutor(seed=1), gate=DeltaGate(delta=0.0005)
    )
    state = controller.run()

    checks = _convergence_events(journal)

    assert len(checks) == state.iteration  # one per committed revision
    assert checks[-1].payload["converged"] is True
    assert checks[-1].payload["by_rule"] in {"organizers", "internal"}
    for earlier in checks[:-1]:
        assert earlier.payload["converged"] is False


def test_convergence_check_records_the_iteration_definition():
    """So a journal reader can see which reading of "iteration" produced
    the number, and can re-render under the other using `node`."""
    controller, journal = _controller(
        executor=FakeExecutor(seed=1), gate=DeltaGate(delta=0.0005)
    )
    controller.run()

    payload = _convergence_events(journal)[0].payload

    assert payload["iteration_definition"] == convergence.ITERATION_DEFINITION
    assert payload["iteration_definition"] == "committed_revision"
    assert set(payload) == {
        "iteration_definition",
        "converged",
        "by_rule",
        "organizers_converged",
        "internal_converged",
        "recent_deltas",
        "recent_significant",
        "iterations_considered",
        "epsilon",
        "n_required",
    }
    assert payload["epsilon"] == convergence.EPSILON
    assert payload["n_required"] == convergence.N_CONSECUTIVE


def test_convergence_check_comes_after_decision():
    controller, journal = _controller(
        executor=FakeExecutor(seed=1), gate=DeltaGate(delta=0.0005)
    )
    controller.run()

    kinds = _kinds(journal)
    for i, kind in enumerate(kinds):
        if kind == EventKind.CONVERGENCE_CHECK.value:
            assert kinds[i - 1] == EventKind.DECISION.value


def test_a_climbing_run_never_converges_and_completes_its_stages():
    """Real improvements above epsilon, and a ci95 strictly above zero, so
    neither rule can fire. The run must go the distance."""
    controller, journal = _controller(
        executor=_ClimbingExecutor(step=0.01), gate=DeltaGate(delta=0.01)
    )

    state = controller.run()

    assert state.stage is Stage.DONE
    assert journal.events_of_kind(EventKind.RUN_END)[0].payload["stop_reason"] == (
        "stages_complete"
    )
    assert all(not e.payload["converged"] for e in _convergence_events(journal))
    assert state.node == TOTAL_CANDIDATES


def test_the_cap_still_bounds_a_run_that_never_converges():
    """max_nodes_per_stage is the backstop. Without it, an accepting gate
    and an unlimited default Budget could run forever."""
    controller, _ = _controller(
        executor=_ClimbingExecutor(step=0.01),
        gate=DeltaGate(delta=0.01),
        max_nodes_per_stage=2,
    )

    state = controller.run()

    # 1 baseline + 3 search stages * 2 attempts.
    assert state.node == 1 + SEARCH_STAGES * 2
    assert state.node <= 1 + SEARCH_STAGES * 2  # an upper bound, explicitly


def test_the_baseline_alone_cannot_converge():
    """One committed revision, no predecessor to difference against and no
    gate ruling to judge — neither rule may fire on it."""
    journal = InMemoryJournal()
    controller = Controller(
        executor=_ClimbingExecutor(),
        gate=AlwaysAcceptGate(),
        generator=ScriptedGenerator([]),  # exhausts right after the baseline
        realizer=DeterministicRealizer(),
        journal=journal,
        run_id="run-baseline-only",
    )

    state = controller.run()

    checks = _convergence_events(journal)
    assert len(checks) == 1
    assert checks[0].payload["converged"] is False
    assert checks[0].payload["iterations_considered"] == 1
    assert state.iteration == 1
    assert journal.events_of_kind(EventKind.RUN_END)[0].payload["stop_reason"] == (
        "generator_exhausted"
    )


def test_only_committed_revisions_enter_the_convergence_window():
    """DECISION 1, pinned: rejected, failed and unrealized candidates are
    not iterations. If they counted, three dead ends in a row would end the
    search before it had searched."""

    def fail_impl_1(config: PipelineConfig):
        if config.slots["model"].impl == "lib/impl_1":
            return ErrorClass.OOM, "out of memory"
        return None

    controller, _ = _controller(
        executor=_ClimbingExecutor(fail_on=fail_impl_1),
        gate=DeltaGate(delta=0.01, accept=False),  # every gated candidate rejected
        realizer=_FlakyRealizer(fail_on_key="paper2"),
    )

    state = controller.run()

    committed = state.committed_revisions
    # Only the baseline was ever committed; everything else was rejected,
    # failed or unrealized.
    assert len(committed) == 1
    assert state.iteration == 1
    assert len(state.history) == TOTAL_CANDIDATES  # they all still happened

    rejected = [h for h in state.history if not h.accepted]
    assert len(rejected) == TOTAL_CANDIDATES - 1
    assert all(entry not in committed for entry in rejected)


def test_history_entry_verdict_fields_are_none_where_no_gate_ruled():
    def fail_impl_1(config: PipelineConfig):
        if config.slots["model"].impl == "lib/impl_1":
            return ErrorClass.OOM, "out of memory"
        return None

    controller, _ = _controller(
        executor=_ClimbingExecutor(fail_on=fail_impl_1),
        gate=DeltaGate(delta=0.01),
        realizer=_FlakyRealizer(fail_on_key="paper2"),
    )

    state = controller.run()
    history = list(state.history)

    baseline = history[0]
    assert (baseline.delta, baseline.ci95, baseline.significant) == (None, None, None)

    failed = [h for h in history if h.primary is None and h.config_id != "<unrealized>"]
    assert failed and all(
        (h.delta, h.ci95, h.significant) == (None, None, None) for h in failed
    )

    unrealized = [h for h in history if h.config_id == "<unrealized>"]
    assert unrealized and all(
        (h.delta, h.ci95, h.significant) == (None, None, None) for h in unrealized
    )

    gated = [h for h in history if h.delta is not None]
    assert gated
    for entry in gated:
        assert entry.delta == pytest.approx(0.01)
        assert entry.significant is True  # ci95 strictly above zero
        assert entry.ci95[0] < entry.delta < entry.ci95[1]


def test_state_card_convergence_summary_round_trips_as_json():
    controller, _ = _controller(
        executor=_ClimbingExecutor(step=0.01), gate=DeltaGate(delta=0.01)
    )
    state = controller.run()

    card = build_state_card(state)

    assert set(card["convergence"]) == {
        "iterations_considered",
        "epsilon",
        "n_required",
        "flat_streak",
        "converged",
        "by_rule",
    }
    assert card["convergence"]["converged"] is False
    assert card["convergence"]["flat_streak"] == 0  # every step beat epsilon
    assert json.loads(json.dumps(card, allow_nan=False)) == card


def test_state_card_reports_a_flat_streak_when_the_run_stalls():
    controller, _ = _controller(
        executor=FakeExecutor(seed=1), gate=DeltaGate(delta=0.0005)
    )
    state = controller.run()

    card = build_state_card(state)

    assert card["convergence"]["flat_streak"] >= 1
    assert card["convergence"]["iterations_considered"] == state.iteration


def test_run_start_records_the_renamed_cap():
    controller, journal = _controller(
        executor=_ClimbingExecutor(), gate=DeltaGate(delta=0.01)
    )
    controller.run()

    payload = journal.events_of_kind(EventKind.RUN_START)[0].payload

    assert payload["max_nodes_per_stage"] == MAX_NODES_PER_STAGE
    assert "nodes_per_stage" not in payload
