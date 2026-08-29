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
    SlotSensitiveExecutor,
    metrics_from_delta,
)
from controller.policy import CostAwareBanditPolicy, FixedOrderPolicy, UniformPolicy
from controller.ports import RealizerExhausted
from controller import convergence
from controller.state import (
    STAGE_ORDER,
    STAGE_SLOTS,
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


def _model_policy() -> FixedOrderPolicy:
    """A policy that always selects `model`.

    The default for the mechanics tests below, and a deliberate choice
    rather than a convenience. Every hypothesis in `_script()` is about
    `model`, and several assertions here name that slot directly (the
    CODE_EMITTED payload, the one-slot lineage diff). Pinning the policy to
    `model` keeps those tests measuring what they are named after - event
    ordering, lineage, budget, convergence - instead of quietly becoming
    tests of whichever arm a random policy happened to draw.

    `model` is a member of STAGE_SLOTS for all three search stages, so this
    never runs out of a slot to return.
    """
    return FixedOrderPolicy(("model",))


def _controller(
    *,
    gate=None,
    executor=None,
    generator=None,
    realizer=None,
    policy=None,
    journal=None,
    budget=None,
    run_id="run-test",
    max_nodes_per_stage=MAX_NODES_PER_STAGE,
    failures_before_block=2,
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
        policy=policy if policy is not None else _model_policy(),
        journal=journal,
        budget=budget,
            max_nodes_per_stage=max_nodes_per_stage,
        failures_before_block=failures_before_block,
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
        # FLAGGED ADDITION (bandit PR): the per-slot aggregate the
        # cost-aware policy scores arms on. An addition, not a rename —
        # every key above and below is unchanged, so nothing coded against
        # the old shape breaks. See build_state_card for why it goes on the
        # card rather than through a side channel.
        "slot_stats",
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
        policy=_model_policy(),
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
            policy=_model_policy(),
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
            policy=_model_policy(),
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
        policy=_model_policy(),
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


# ---------------------------------------------------------------------------
# Slot selection — appended. Nothing above this line is modified except the
# mechanical addition of `policy=` to the Controller constructions, which is
# now a required argument.
# ---------------------------------------------------------------------------

from contracts import SLOT_ORDER, Verdict
from controller.controller import MISPROPOSED_CONFIG_ID, UNREALIZED_CONFIG_ID
from controller.fakes import BASELINE_SIGMA, DisobedientGenerator
from controller.ports import PolicyContractError, PortExhausted
from controller.state import SlotStats, slot_stats, slot_stats_as_json

STRUCTURAL_ORDER = ("features", "weighting", "model", "objective")


class _AlternatingGate:
    """Accepts every other candidate, with a fixed significant delta.

    Needed because the mixed-outcome fixture below has to produce accepted
    AND rejected entries in one run: DeltaGate is all-or-nothing on
    `accept`, and ScriptedGate would have to predict exactly how many
    candidates survive the executor and the realizer to reach it.

    delta 0.01 with a 1.96-sigma half width puts ci95 strictly above zero,
    so the internal convergence rule never fires and the run goes the
    distance.
    """

    def __init__(self) -> None:
        self._n = 0
        self.delta = 0.01
        self._half_width = 1.96 * BASELINE_SIGMA

    def compare(self, candidate, incumbent) -> Verdict:
        self._n += 1
        accept = self._n % 2 == 1
        return Verdict(
            accept=accept,
            delta=self.delta,
            ci95=(self.delta - self._half_width, self.delta + self._half_width),
            backtest_delta=self.delta,
            reason=f"_AlternatingGate: accept={accept} (test double)",
        )


class _BadPolicy:
    """Returns a slot it was never offered. Models a bug in our own code."""

    def __init__(self, slot: str = "data_view") -> None:
        self.slot = slot

    def select_slot(self, state_card, candidate_slots):
        return self.slot


def _mixed_outcome_run() -> RunState:
    """One run containing all four outcome kinds: accepted, rejected,
    FAILED and unrealized.

    Built once and reused by the four cost assertions below so they are
    reading the same run rather than four subtly different ones.
    """

    def fail_impl_3(config: PipelineConfig):
        if config.slots["model"].impl == "lib/impl_3":
            return ErrorClass.NAN_LOSS, "loss became nan at epoch 3"
        return None

    controller, _ = _controller(
        executor=_ClimbingExecutor(fail_on=fail_impl_3),
        gate=_AlternatingGate(),
        realizer=_FlakyRealizer(fail_on_key="paper5"),
    )
    return controller.run()


# ---------------------------------------------------------------------------
# STAGE_SLOTS
# ---------------------------------------------------------------------------


def test_stage_slots_has_an_entry_for_every_stage():
    """A stage added to the enum but forgotten here would raise KeyError
    deep inside the first attempt that reached it. state.py also guards
    this at import time; this is the test that says so out loud."""
    assert set(STAGE_SLOTS) == set(Stage)


def test_stage_slots_only_ever_names_real_slots():
    """A typo'd slot name would pass silently into the policy and then
    KeyError on the splice. SLOT_ORDER is the vocabulary."""
    for stage, slots in STAGE_SLOTS.items():
        for slot in slots:
            assert slot in SLOT_ORDER, f"{stage.value} names unknown slot {slot!r}"


def test_search_stages_cover_everything_the_published_baseline_ignores():
    """The four slots where the baseline takes the default are exactly
    where the headroom is, so both structural stages must offer all four."""
    baseline_defaults = {"features", "weighting", "model", "objective"}

    assert set(STAGE_SLOTS[Stage.STAGE_1_STRUCTURAL]) == baseline_defaults
    assert set(STAGE_SLOTS[Stage.STAGE_2_COMBINE]) == baseline_defaults


def test_tuning_stage_opens_every_slot_and_the_others_open_none():
    assert set(STAGE_SLOTS[Stage.STAGE_3_TUNE]) == set(SLOT_ORDER)
    for stage in (Stage.INIT, Stage.REPRODUCE_BASELINE, Stage.FINALIZE, Stage.DONE):
        assert STAGE_SLOTS[stage] == (), stage


def test_stage_slots_ordering_follows_slot_order():
    """Order is load-bearing: it is the order the policy is handed, so a
    policy that depends on it (FixedOrderPolicy) must behave identically
    across runs and processes."""
    for slots in STAGE_SLOTS.values():
        assert list(slots) == [s for s in SLOT_ORDER if s in slots]


# ---------------------------------------------------------------------------
# The Controller owns slot selection now
# ---------------------------------------------------------------------------


def test_policy_is_required_and_has_no_default():
    """Same reasoning as the realizer: a silent default would supply the
    very wiring this change exists to make explicit."""
    with pytest.raises(TypeError):
        Controller(
            executor=FakeExecutor(seed=0),
            gate=AlwaysAcceptGate(),
            generator=ScriptedGenerator(_script()),
            realizer=DeterministicRealizer(),
            journal=InMemoryJournal(),
        )


def test_controller_passes_the_policy_selected_slot_to_the_generator():
    """THE load-bearing test for this change. The generator no longer picks
    the slot; it is told, and what it is told is exactly what the policy
    returned."""
    generator = ScriptedGenerator(_script())
    policy = FixedOrderPolicy(STRUCTURAL_ORDER)
    controller, journal = _controller(generator=generator, policy=policy)

    controller.run()

    # Nine search candidates, cycling the four structural arms in order.
    assert generator.requested_slots == [
        "features",
        "weighting",
        "model",
        "objective",
        "features",
        "weighting",
        "model",
        "objective",
        "features",
    ]
    # And the slot that was requested is the slot that got realized.
    emitted = [
        e.payload["target_slot"]
        for e in journal.events_of_kind(EventKind.CODE_EMITTED)
    ]
    assert emitted == generator.requested_slots


def test_the_generator_never_sees_the_slot_twice_via_the_state_card():
    """The card's top-level key set is unchanged by this PR — the slot is a
    parameter, not a key. A second copy in the card would be a channel the
    constraint could be honoured or ignored through with no way to tell."""
    generator = ScriptedGenerator(_script())
    controller, _ = _controller(generator=generator)

    controller.run()

    card = generator.state_cards[0]
    assert set(card) == {
        "run_id",
        "stage",
        "iteration",
        "node",
        "incumbent_config_id",
        "incumbent_primary",
        "recent_history",
        "slot_stats",  # FLAGGED ADDITION (bandit PR) — see the test above.
        "blocked_slots",
        "budget_remaining",
        "convergence",
    }
    assert "target_slot" not in card


def test_the_candidate_differs_from_its_parent_in_the_selected_slot_only():
    """Selecting a different slot must move that slot and nothing else —
    the one-slot diff is what makes the cascading slot_hash reuse pay off."""
    executor = _RecordingExecutor(_ClimbingExecutor())
    controller, _ = _controller(
        executor=executor, policy=FixedOrderPolicy(("weighting",))
    )
    controller.run()

    baseline, first = executor.configs[0], executor.configs[1]
    differing = [s for s in baseline.slots if baseline.slots[s] != first.slots[s]]

    assert differing == ["weighting"]


# ---------------------------------------------------------------------------
# The generator wandered: a contract violation, logged, survived
# ---------------------------------------------------------------------------


def test_a_disobedient_generator_produces_a_contract_error_and_the_run_goes_on():
    generator = DisobedientGenerator(_script(), wrong_slot="calibration")
    controller, journal = _controller(generator=generator)

    state = controller.run()

    assert state.stage is Stage.DONE
    assert state.node == TOTAL_CANDIDATES  # every mismatch still cost a node

    mismatches = [
        e
        for e in journal.events_of_kind(EventKind.ERROR)
        if e.payload.get("reason") == "generator_slot_mismatch"
    ]
    assert len(mismatches) == SEARCH_CANDIDATES
    payload = mismatches[0].payload
    assert payload["error_class"] == ErrorClass.CONTRACT.value
    assert payload["requested_slot"] == "model"
    assert payload["proposed_slot"] == "calibration"
    assert payload["config_id"] == MISPROPOSED_CONFIG_ID
    assert _kinds(journal)[-1] == EventKind.RUN_END.value


def test_the_controller_does_not_obey_the_slot_the_generator_named():
    """The whole point of the guard. A payload for the wrong slot is
    discarded, never spliced — otherwise the LLM takes the search decision
    back through the back door."""
    generator = DisobedientGenerator(_script(), wrong_slot="calibration")
    controller, journal = _controller(generator=generator)

    state = controller.run()

    # The incumbent is still the untouched baseline: no candidate ever got
    # as far as being evaluated.
    assert state.incumbent_config is not None
    assert state.incumbent_config.slots == BASELINE_SLOTS
    assert state.incumbent_config.slots["calibration"] == BASELINE_SLOTS["calibration"]
    assert state.iteration == 1  # the baseline, and nothing else

    kinds = _kinds(journal)
    # Proposed nine times, realized and evaluated exactly none of them.
    assert kinds.count(EventKind.HYPOTHESIS.value) == SEARCH_CANDIDATES
    assert kinds.count(EventKind.CODE_EMITTED.value) == 0
    assert kinds.count(EventKind.EVAL_START.value) == 1  # the baseline only


def test_the_mismatch_sentinel_is_distinct_from_the_unrealized_one():
    """Two different failures with two different owners — a prompting
    problem and a library problem — and the log must be able to count them
    apart."""
    generator = DisobedientGenerator(_script(), wrong_slot="calibration")
    controller, _ = _controller(generator=generator)

    state = controller.run()

    assert MISPROPOSED_CONFIG_ID != UNREALIZED_CONFIG_ID
    sentinels = [h for h in state.history if h.config_id == MISPROPOSED_CONFIG_ID]
    assert len(sentinels) == SEARCH_CANDIDATES
    assert all(h.config_id != UNREALIZED_CONFIG_ID for h in sentinels)
    # Charged to the slot the policy asked for, not the one the LLM named.
    assert all(h.target_slot == "model" for h in sentinels)


# ---------------------------------------------------------------------------
# The policy misbehaved: loud, immediately
# ---------------------------------------------------------------------------


def test_a_policy_returning_an_unoffered_slot_raises_and_is_not_swallowed():
    """`data_view` is not in STAGE_1_STRUCTURAL's slots. Unlike an LLM, the
    policy is our own deterministic code, so this is a defect and must stop
    the run rather than being logged and absorbed."""
    controller, _ = _controller(policy=_BadPolicy("data_view"))

    with pytest.raises(PolicyContractError) as excinfo:
        controller.run()

    message = str(excinfo.value)
    assert "data_view" in message
    assert "_BadPolicy" in message
    assert Stage.STAGE_1_STRUCTURAL.value in message


def test_the_policy_contract_error_is_not_catchable_as_an_exhausted_port():
    """A handler written to shrug off an exhausted generator must not also
    shrug off a policy bug."""
    assert not issubclass(PolicyContractError, PortExhausted)
    assert issubclass(PolicyContractError, RuntimeError)


def test_a_blocked_slot_returned_by_the_policy_is_also_a_violation():
    """The candidate set is stage slots MINUS blocked slots, so re-offering
    a blocked arm is caught by the same check."""
    controller, _ = _controller(policy=_BadPolicy("model"))
    state = RunState(
        run_id="r",
        stage=Stage.STAGE_1_STRUCTURAL,
        blocked_slots=frozenset({"model"}),
    )

    with pytest.raises(PolicyContractError):
        controller._attempt(
            state,
            Stage.STAGE_1_STRUCTURAL,
            controller._candidate_slots(state, Stage.STAGE_1_STRUCTURAL),
        )


# ---------------------------------------------------------------------------
# blocked_slots narrow the candidate set
# ---------------------------------------------------------------------------


def test_blocked_slots_are_excluded_from_the_candidate_set():
    controller, _ = _controller()
    state = RunState(
        run_id="r",
        stage=Stage.STAGE_1_STRUCTURAL,
        blocked_slots=frozenset({"model", "features"}),
    )

    candidates = controller._candidate_slots(state, Stage.STAGE_1_STRUCTURAL)

    assert candidates == ("weighting", "objective")
    assert "model" not in candidates
    assert "features" not in candidates


def test_blocking_narrows_the_tuning_stage_without_touching_the_others():
    controller, _ = _controller()
    state = RunState(
        run_id="r", stage=Stage.STAGE_3_TUNE, blocked_slots=frozenset({"data_view"})
    )

    assert controller._candidate_slots(state, Stage.STAGE_3_TUNE) == (
        "features",
        "weighting",
        "model",
        "objective",
        "calibration",
    )
    # An unblocked state is unchanged by the filter.
    clean = RunState(run_id="r", stage=Stage.STAGE_3_TUNE)
    assert controller._candidate_slots(clean, Stage.STAGE_3_TUNE) == tuple(SLOT_ORDER)


def test_a_fully_blocked_stage_ends_cleanly_without_calling_the_generator():
    """Reached through `_run_stage` directly because nothing in the current
    Controller calls `with_blocked_slot` yet — slot blocking arrives with
    the repair policy. The path exists now and must be covered now, or it
    is one refactor away from being deleted as dead code.
    """
    generator = ScriptedGenerator(_script())
    controller, journal = _controller(generator=generator)
    state = RunState(
        run_id="run-test",
        stage=Stage.STAGE_1_STRUCTURAL,
        blocked_slots=frozenset(STAGE_SLOTS[Stage.STAGE_1_STRUCTURAL]),
    )

    ended, stop_reason = controller._run_stage(state, Stage.STAGE_1_STRUCTURAL)

    # The stage ended; the RUN did not — a later stage has a different set.
    assert stop_reason is None
    assert generator.requested_slots == []  # the generator was never asked
    assert ended.node == 0  # no candidate, so no node consumed
    assert ended.history == ()

    blocked = journal.events_of_kind(EventKind.SLOT_BLOCKED)
    assert len(blocked) == 1
    assert blocked[0].payload["stage"] == Stage.STAGE_1_STRUCTURAL.value
    assert blocked[0].payload["action"] == "end_stage"
    assert sorted(blocked[0].payload["blocked_slots"]) == sorted(
        STAGE_SLOTS[Stage.STAGE_1_STRUCTURAL]
    )


def test_a_fully_blocked_stage_emits_no_hypothesis_and_does_not_crash():
    generator = ScriptedGenerator(_script())
    controller, journal = _controller(generator=generator)
    state = RunState(
        run_id="run-test",
        stage=Stage.STAGE_2_COMBINE,
        blocked_slots=frozenset(SLOT_ORDER),
    )

    controller._run_stage(state, Stage.STAGE_2_COMBINE)  # must not raise

    kinds = _kinds(journal)
    assert EventKind.HYPOTHESIS.value not in kinds
    assert EventKind.EVAL_START.value not in kinds
    assert EventKind.ERROR.value not in kinds


# ---------------------------------------------------------------------------
# HistoryEntry now carries the arm and its cost — for every outcome
# ---------------------------------------------------------------------------


def test_an_accepted_attempt_records_its_slot_and_its_cost():
    state = _mixed_outcome_run()

    accepted = [h for h in state.history if h.accepted and h.target_slot is not None]

    assert accepted
    for entry in accepted:
        assert entry.target_slot == "model"
        assert entry.wall_seconds == 40.0
        assert entry.tokens == 1200
        assert entry.gpu_seconds == 0.0


def test_a_rejected_attempt_records_its_slot_and_its_cost():
    state = _mixed_outcome_run()

    rejected = [
        h
        for h in state.history
        if not h.accepted and h.primary is not None and h.delta is not None
    ]

    assert rejected
    for entry in rejected:
        assert entry.target_slot == "model"
        assert entry.wall_seconds == 40.0
        assert entry.tokens == 1200


def test_a_failed_attempt_records_its_slot_and_a_NON_ZERO_cost():
    """The one that matters most. A candidate that died halfway still burned
    what it took to get there; recording zero would teach a cost-aware
    policy that fragile slots are cheap, which is exactly backwards."""
    state = _mixed_outcome_run()

    failed = [
        h
        for h in state.history
        if h.primary is None
        and h.config_id not in (UNREALIZED_CONFIG_ID, MISPROPOSED_CONFIG_ID)
    ]

    assert failed
    for entry in failed:
        assert entry.target_slot == "model"
        assert entry.wall_seconds > 0.0
        assert entry.tokens > 0
        assert entry.wall_seconds == 40.0
        assert entry.tokens == 1200


def test_an_unrealized_attempt_records_its_slot_and_an_honest_zero_cost():
    """Zero because it never reached the executor, not because cost was
    dropped: the realizer's own tokens are real spend but no port reports
    them yet."""
    state = _mixed_outcome_run()

    unrealized = [h for h in state.history if h.config_id == UNREALIZED_CONFIG_ID]

    assert unrealized
    for entry in unrealized:
        assert entry.target_slot == "model"
        assert entry.wall_seconds == 0.0
        assert entry.gpu_seconds == 0.0
        assert entry.tokens == 0


def test_the_baseline_records_no_slot_because_no_policy_chose_it():
    state = _mixed_outcome_run()

    baseline = state.history[0]

    assert baseline.target_slot is None
    assert baseline.accepted is True
    assert baseline.wall_seconds == 40.0  # it still cost an evaluation


def test_the_new_history_fields_survive_the_state_card_json_hop():
    """FLAGGED CHANGE: `recent_history` entries gained four keys. They are
    additions, not renames, and must stay RFC-valid JSON — W4 reads them."""
    state = _mixed_outcome_run()

    card = build_state_card(state)

    for entry in card["recent_history"]:
        assert {"target_slot", "wall_seconds", "gpu_seconds", "tokens"} <= set(entry)
        # Every key that was there before is still there, unchanged.
        assert {"config_id", "primary", "accepted", "delta", "ci95", "significant"} <= (
            set(entry)
        )
    assert json.loads(json.dumps(card, allow_nan=False)) == card


# ---------------------------------------------------------------------------
# slot_stats — the bookkeeping the bandit PR will consume
# ---------------------------------------------------------------------------


def test_slot_stats_aggregates_attempts_acceptances_and_cost_per_slot():
    state = RunState(run_id="r", stage=Stage.STAGE_1_STRUCTURAL)
    state = state.with_outcome(
        "a", 0.60, True, delta=0.01, target_slot="model",
        wall_seconds=40.0, gpu_seconds=1.0, tokens=1200,
    )
    state = state.with_outcome(
        "b", 0.59, False, delta=-0.005, target_slot="model",
        wall_seconds=20.0, gpu_seconds=0.5, tokens=800,
    )
    state = state.with_outcome(
        "c", None, False, target_slot="model",
        wall_seconds=5.0, tokens=100,  # a failure: no delta, real cost
    )
    state = state.with_outcome(
        "d", 0.61, True, delta=0.02, target_slot="weighting",
        wall_seconds=10.0, tokens=300,
    )

    stats = slot_stats(state)

    assert set(stats) == {"model", "weighting"}
    assert stats["model"] == SlotStats(
        attempts=3,
        accepted=1,
        total_tokens=2100,
        total_wall_seconds=65.0,
        deltas=(0.01, -0.005),  # the failure contributes no delta
    )
    assert stats["weighting"] == SlotStats(
        attempts=1, accepted=1, total_tokens=300, total_wall_seconds=10.0,
        deltas=(0.02,),
    )


def test_slot_stats_skips_the_baseline_because_no_policy_chose_it():
    state = RunState(run_id="r", stage=Stage.REPRODUCE_BASELINE)
    state = state.with_outcome("base", 0.6016, True, wall_seconds=40.0, tokens=1200)

    assert slot_stats(state) == {}


def test_slot_stats_omits_untried_slots_entirely():
    """An absent key means 'no data'. A zero-filled entry would erase the
    difference between an untried arm and one tried once to no effect."""
    state = RunState(run_id="r", stage=Stage.STAGE_1_STRUCTURAL)
    state = state.with_outcome("a", 0.6, False, target_slot="model")

    stats = slot_stats(state)

    assert set(stats) == {"model"}
    assert "features" not in stats
    assert "calibration" not in stats


def test_slot_stats_is_a_pure_function_of_history():
    state = _mixed_outcome_run()
    history_before = state.history

    first = slot_stats(state)
    second = slot_stats(state)

    assert first == second
    assert state.history == history_before  # nothing mutated
    assert first is not second  # a fresh fold, not a cached object


def test_slot_stats_over_a_real_run_reconciles_with_the_history():
    """Cross-check against the raw history, so a future refactor of the
    fold cannot drift from what it is folding."""
    state = _mixed_outcome_run()

    stats = slot_stats(state)
    attempted = [h for h in state.history if h.target_slot is not None]

    assert sum(s.attempts for s in stats.values()) == len(attempted)
    assert sum(s.accepted for s in stats.values()) == sum(
        1 for h in attempted if h.accepted
    )
    assert sum(s.total_tokens for s in stats.values()) == sum(
        h.tokens for h in attempted
    )
    assert stats["model"].attempts == SEARCH_CANDIDATES


def test_slot_stats_attributes_cost_across_several_arms_in_one_run():
    """The shape the bandit reads: distinct arms, each with its own cost and
    hit count, from a run where the policy actually moved between them."""
    controller, _ = _controller(policy=FixedOrderPolicy(STRUCTURAL_ORDER))

    state = controller.run()
    stats = slot_stats(state)

    assert set(stats) == set(STRUCTURAL_ORDER)
    # Nine candidates over four arms, cycled: 3/2/2/2.
    assert stats["features"].attempts == 3
    assert stats["weighting"].attempts == 2
    assert all(s.total_tokens > 0 for s in stats.values())
    assert all(s.total_wall_seconds > 0 for s in stats.values())


# ---------------------------------------------------------------------------
# Nothing else changed
# ---------------------------------------------------------------------------


def test_the_full_run_still_completes_exactly_as_before():
    """Existing behaviour is unchanged apart from who picks the slot: same
    stages, same candidate count, same terminal event."""
    controller, journal = _controller()

    state = controller.run()

    assert state.stage is Stage.DONE
    assert state.node == TOTAL_CANDIDATES
    assert _kinds(journal)[-1] == EventKind.RUN_END.value
    assert journal.events_of_kind(EventKind.RUN_END)[0].payload["stop_reason"] == (
        "stages_complete"
    )
    assert journal.events_of_kind(EventKind.SLOT_BLOCKED) == ()


def test_a_run_driven_by_a_seeded_uniform_policy_is_reproducible():
    """The ablation arm, end to end. Two runs of the same seeded policy must
    attack the same slots in the same order, or a comparison against the
    bandit measures run-to-run variance instead of the policy."""

    def slots_of_a_run(seed: int) -> list[str]:
        generator = ScriptedGenerator(_script())
        controller, _ = _controller(
            generator=generator, policy=UniformPolicy(seed=seed)
        )
        controller.run()
        return list(generator.requested_slots)

    assert slots_of_a_run(4) == slots_of_a_run(4)
    assert len(slots_of_a_run(4)) == SEARCH_CANDIDATES
    assert set(slots_of_a_run(4)) <= set(SLOT_ORDER)


def test_a_uniform_policy_run_still_reaches_done():
    controller, journal = _controller(policy=UniformPolicy(seed=2))

    state = controller.run()

    assert state.stage is Stage.DONE
    assert state.node == TOTAL_CANDIDATES
    assert _kinds(journal)[-1] == EventKind.RUN_END.value


# ---------------------------------------------------------------------------
# The circuit breaker, and the cost-aware bandit's ablation
# ---------------------------------------------------------------------------


class _StickyPolicy:
    """Selects `preferred` whenever it is offered, else the first candidate.

    FixedOrderPolicy raises when its order shares no slot with the offered
    set, which is exactly what happens the moment the circuit breaker
    blocks the slot these tests are pinned to - so a blocking test written
    against it would fail on the policy rather than on the mechanism under
    test. This falls back instead, which is what makes "kept attacking one
    slot until it was blocked, then moved on" expressible.

    `offered` records every candidate set it was handed, in order. That is
    the assertion surface for both halves of the contract: a blocked slot
    stops being offered, and the Controller never calls a policy with an
    empty set.
    """

    def __init__(self, preferred: str) -> None:
        self.preferred = preferred
        self.offered: list[tuple[str, ...]] = []

    def select_slot(self, state_card, candidate_slots):
        self.offered.append(tuple(candidate_slots))
        if self.preferred in candidate_slots:
            return self.preferred
        return candidate_slots[0]


def _fails_on_slot(slot: str, times: int = 10**6):
    """A FailureHook that fails candidates whose `slot` left the baseline.

    Compared against BASELINE_SLOTS rather than against the incumbent: a
    candidate for a failing slot never gets accepted, so the incumbent
    never carries that slot's change and the two readings coincide - but
    naming the baseline says what is meant without depending on that.

    `times` caps how many failures it produces, so a test can make a slot
    fail, recover, and fail again.
    """
    fired = {"n": 0}

    def hook(config):
        if config.slots.get(slot) == BASELINE_SLOTS[slot]:
            return None
        if fired["n"] >= times:
            return None
        fired["n"] += 1
        return (ErrorClass.OOM, f"synthetic OOM realizing {slot}")

    return hook


def _fails_on_any_search_candidate():
    """Fails every candidate that differs from the baseline anywhere.

    The fixture for "a whole stage's arms go down one after another",
    which is the only way the fully-blocked path is reached from a real
    run rather than from a hand-built RunState.
    """

    def hook(config):
        if any(config.slots.get(s) != BASELINE_SLOTS[s] for s in BASELINE_SLOTS):
            return (ErrorClass.OOM, "synthetic OOM")
        return None

    return hook


def _blocks(journal: InMemoryJournal) -> list:
    """SLOT_BLOCKED events that blocked a slot, not the ones that ended a
    stage because everything was already blocked."""
    return [
        e
        for e in journal.events_of_kind(EventKind.SLOT_BLOCKED)
        if e.payload.get("action") == "block_slot"
    ]


# -- DECISION 4: the ablation ------------------------------------------


def test_the_bandit_pulls_the_good_arm_far_more_often_than_uniform():
    """THE ABLATION. A bandit that cannot be shown to differ from its own
    control is not evidence of anything, so this runs both policies over
    the same seeded configuration and compares.

    GROUND TRUTH: SlotSensitiveExecutor makes `weighting` genuinely worth
    +0.02 primary and every other slot pure noise at the measured sigma of
    0.0008 - a 25-sigma effect, so the ranking is not a coin flip.

    max_nodes_per_stage = 20, and the number matters. Stage one offers four
    arms, so the bandit spends its first four attempts on the mandatory
    initialisation pass; at the default of 3 it would never finish that
    pass and would be a differently-seeded uniform by construction. 20
    leaves 16 informed pulls per stage, which is where the two policies can
    actually separate.

    exploration_c = 0.005, and this number matters too - see
    CostAwareBanditPolicy on why the default of 1.0 does not work here. The
    exploitation term is delta-per-1k-tokens, about 0.017 for the good arm;
    sqrt(ln T / n) is order 1. At c = 1.0 the bonus outruns the signal by
    two orders of magnitude and the bandit round-robins. 0.005 puts the two
    terms on the same footing at this scale. That is a finding about the
    formula, not a thumb on the scale for the test: the same c is what a
    real run on this benchmark would need.

    AlwaysRejectGate so the incumbent never moves, which keeps every
    candidate measured against the same reference and stops convergence
    ending one policy's run earlier than the other's. Rejected attempts
    still carry a gate delta, so the bandit still learns.
    """
    def run(policy) -> int:
        generator = ScriptedGenerator(_script(80))
        Controller(
            executor=SlotSensitiveExecutor("weighting", effect=0.02, seed=7),
            gate=AlwaysRejectGate(),
            generator=generator,
            realizer=DeterministicRealizer(),
            policy=policy,
            journal=InMemoryJournal(),
            seeds=(0, 1, 2),
            max_nodes_per_stage=20,
            run_id="ablation",
        ).run()
        return generator.requested_slots.count("weighting")

    bandit = run(CostAwareBanditPolicy(seed=0, exploration_c=0.005))
    uniform = run(UniformPolicy(seed=0))

    # A margin, not an exact count. Measured over a grid of executor and
    # uniform seeds the smallest gap was 29; 20 sits well inside that
    # without being so loose that a regression to round-robin would pass.
    assert bandit >= uniform + 20, (
        f"bandit pulled the good arm {bandit} times, uniform {uniform} — "
        f"the bandit is not distinguishable from its own control"
    )
    # And the absolute claim, so a future change that made BOTH bad still
    # fails: the bandit spends most of the run on the arm that works.
    assert bandit > 40


def test_at_the_textbook_exploration_constant_the_bandit_matches_uniform():
    """THE HONEST COUNTERPART to the ablation above, pinned so nobody
    over-claims from it.

    At the default exploration_c = 1.0 the bonus dominates the
    delta-per-1k-token signal and the bandit degenerates into round-robin —
    it pulls the good arm about as often as uniform does. The bandit is a
    result only when c is chosen against the reward scale, and this test
    exists so that caveat cannot quietly rot out of the docstring.
    """
    def run(policy) -> int:
        generator = ScriptedGenerator(_script(80))
        Controller(
            executor=SlotSensitiveExecutor("weighting", effect=0.02, seed=7),
            gate=AlwaysRejectGate(),
            generator=generator,
            realizer=DeterministicRealizer(),
            policy=policy,
            journal=InMemoryJournal(),
            seeds=(0, 1, 2),
            max_nodes_per_stage=20,
            run_id="ablation-default-c",
        ).run()
        return generator.requested_slots.count("weighting")

    default_c = run(CostAwareBanditPolicy(seed=0, exploration_c=1.0))
    uniform = run(UniformPolicy(seed=0))

    assert abs(default_c - uniform) <= 5
    # Far short of what the tuned constant achieves in the test above.
    assert default_c < 20


# -- the breaker fires on executor failures -----------------------------


def test_k_consecutive_executor_failures_block_the_slot():
    policy = _StickyPolicy("model")
    controller, journal = _controller(
        executor=_ClimbingExecutor(fail_on=_fails_on_slot("model")),
        policy=policy,
        max_nodes_per_stage=4,
        failures_before_block=2,
    )
    state = RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL)

    ended, stop_reason = controller._run_stage(state, Stage.STAGE_1_STRUCTURAL)

    assert stop_reason is None  # a blocked slot ends nothing
    blocked = _blocks(journal)
    assert len(blocked) == 1
    assert blocked[0].payload["slot"] == "model"
    assert blocked[0].payload["consecutive_failures"] == 2
    assert blocked[0].payload["threshold"] == 2
    assert blocked[0].payload["stage"] == Stage.STAGE_1_STRUCTURAL.value
    assert ended.blocked_slots == frozenset({"model"})


def test_a_blocked_slot_is_removed_from_every_later_candidate_set():
    policy = _StickyPolicy("model")
    controller, _ = _controller(
        executor=_ClimbingExecutor(fail_on=_fails_on_slot("model")),
        policy=policy,
        max_nodes_per_stage=5,
        failures_before_block=2,
    )
    state = RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL)

    controller._run_stage(state, Stage.STAGE_1_STRUCTURAL)

    # Two pulls on `model`, then it is gone for the rest of the stage.
    assert "model" in policy.offered[0]
    assert "model" in policy.offered[1]
    assert all("model" not in offer for offer in policy.offered[2:])
    assert len(policy.offered) == 5


def test_the_slot_blocked_event_is_distinguishable_from_the_stage_ending_one():
    """Two different facts share the event kind; `action` separates them,
    and a reader counting blocks must not accidentally count stage ends."""
    policy = _StickyPolicy("model")
    controller, journal = _controller(
        executor=_ClimbingExecutor(fail_on=_fails_on_slot("model")),
        policy=policy,
        max_nodes_per_stage=3,
        failures_before_block=2,
    )

    controller._run_stage(
        RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL),
        Stage.STAGE_1_STRUCTURAL,
    )

    actions = [
        e.payload["action"] for e in journal.events_of_kind(EventKind.SLOT_BLOCKED)
    ]
    assert actions == ["block_slot"]


def test_the_failure_error_event_still_precedes_the_block():
    """Cause then consequence: the ERROR for the failure that tripped the
    breaker is written before the SLOT_BLOCKED it caused."""
    controller, journal = _controller(
        executor=_ClimbingExecutor(fail_on=_fails_on_slot("model")),
        policy=_StickyPolicy("model"),
        max_nodes_per_stage=2,
        failures_before_block=2,
    )

    controller._run_stage(
        RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL),
        Stage.STAGE_1_STRUCTURAL,
    )

    kinds = _kinds(journal)
    assert kinds.index(EventKind.ERROR.value) < kinds.index(
        EventKind.SLOT_BLOCKED.value
    )


def test_the_threshold_is_configurable_and_rejects_a_nonsense_value():
    """A tunable, not a law — and one that must not be set below 1, which
    would block a slot before it had failed at all."""
    controller, journal = _controller(
        executor=_ClimbingExecutor(fail_on=_fails_on_slot("model")),
        policy=_StickyPolicy("model"),
        max_nodes_per_stage=4,
        failures_before_block=3,
    )

    controller._run_stage(
        RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL),
        Stage.STAGE_1_STRUCTURAL,
    )

    assert _blocks(journal)[0].payload["consecutive_failures"] == 3

    with pytest.raises(ValueError, match="failures_before_block"):
        _controller(failures_before_block=0)
    with pytest.raises(ValueError, match="failures_before_block"):
        _controller(failures_before_block=-1)


def test_consecutive_failures_reset_after_a_success_on_that_slot():
    """The counter measures a current run of failures, not a lifetime
    total: a slot that failed, recovered and failed again is not one
    attempt from being blocked."""
    controller, journal = _controller(
        # Fails the first `model` candidate only; every later one works.
        executor=_ClimbingExecutor(fail_on=_fails_on_slot("model", times=1)),
        policy=_StickyPolicy("model"),
        max_nodes_per_stage=5,
        failures_before_block=2,
    )

    ended, _ = controller._run_stage(
        RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL),
        Stage.STAGE_1_STRUCTURAL,
    )

    assert _blocks(journal) == []
    assert ended.blocked_slots == frozenset()
    stats = slot_stats(ended)
    assert stats["model"].attempts == 5
    assert stats["model"].consecutive_failures == 0


def test_the_counter_is_per_slot_not_run_wide():
    """Two different slots failing once each must not add up to a block on
    either. The run has two failures; no arm has two."""
    model_hook = _fails_on_slot("model", times=1)
    features_hook = _fails_on_slot("features", times=1)

    def hook(config):
        return model_hook(config) or features_hook(config)

    controller, journal = _controller(
        executor=_ClimbingExecutor(fail_on=hook),
        # Alternates between the two arms; each fails on its first pull and
        # works thereafter, so the run accumulates two failures spread over
        # two arms rather than two on one.
        policy=FixedOrderPolicy(("model", "features")),
        max_nodes_per_stage=4,
        failures_before_block=2,
    )

    ended, _ = controller._run_stage(
        RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL),
        Stage.STAGE_1_STRUCTURAL,
    )

    assert _blocks(journal) == []
    assert sum(1 for e in ended.history if e.executor_failed) == 2
    stats = slot_stats(ended)
    assert stats["model"].consecutive_failures == 0  # reset by the success
    assert stats["features"].consecutive_failures == 0


def test_failures_on_one_slot_stay_consecutive_across_other_slots_pulls():
    """"Consecutive" is within the ARM's own attempts, not within the run's.
    Pulling a different slot in between does not launder a failing arm's
    record — the fold is per-slot, and an interleaved good result on some
    other slot says nothing about this one."""
    controller, journal = _controller(
        executor=_ClimbingExecutor(fail_on=_fails_on_slot("model")),
        # model, features, model, ... — the two model failures are not
        # adjacent in history, but they are adjacent in `model`'s history.
        policy=FixedOrderPolicy(("model", "features")),
        max_nodes_per_stage=4,
        failures_before_block=2,
    )

    ended, _ = controller._run_stage(
        RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL),
        Stage.STAGE_1_STRUCTURAL,
    )

    blocked = _blocks(journal)
    assert len(blocked) == 1
    assert blocked[0].payload["slot"] == "model"
    assert ended.blocked_slots == frozenset({"model"})
    assert slot_stats(ended)["features"].consecutive_failures == 0


# -- DECISION 1: what pointedly does NOT count --------------------------


def test_generator_slot_mismatches_never_block_however_many_there_are():
    """PINS DECISION 1. A generator that keeps naming the wrong slot is
    W4's bug; blocking `model` over it would delete a good arm from this
    run because another workstream's component misbehaved."""
    generator = DisobedientGenerator(_script(12), wrong_slot="calibration")
    controller, journal = _controller(
        generator=generator,
        policy=_StickyPolicy("model"),
        max_nodes_per_stage=8,
        failures_before_block=2,
    )

    ended, _ = controller._run_stage(
        RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL),
        Stage.STAGE_1_STRUCTURAL,
    )

    assert _blocks(journal) == []
    assert ended.blocked_slots == frozenset()
    # Eight misproposals, all charged to the requested arm, none of them a
    # failure — so the arm's record is bad but the arm is still in play.
    assert slot_stats(ended)["model"].attempts == 8
    assert slot_stats(ended)["model"].consecutive_failures == 0
    assert all(e.executor_failed is False for e in ended.history)


def test_realizer_exhaustions_never_block_however_many_there_are():
    """PINS DECISION 1. A hypothesis that cannot be turned into code is the
    realizer failing its side of the port, not evidence about the slot."""
    controller, journal = _controller(
        realizer=ScriptedRealizer([]),  # exhausted on the first call
        policy=_StickyPolicy("model"),
        max_nodes_per_stage=8,
        failures_before_block=2,
    )

    ended, _ = controller._run_stage(
        RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL),
        Stage.STAGE_1_STRUCTURAL,
    )

    assert _blocks(journal) == []
    assert ended.blocked_slots == frozenset()
    assert slot_stats(ended)["model"].attempts == 8
    assert slot_stats(ended)["model"].consecutive_failures == 0


def test_clean_gate_rejections_never_block():
    """PINS DECISION 1. A rejection is a working arm reporting no
    improvement — the negative result the search exists to collect, and
    exactly the signal the bandit consumes. Blocking on it would stop the
    search the moment it learned something."""
    controller, journal = _controller(
        gate=AlwaysRejectGate(),
        policy=_StickyPolicy("model"),
        max_nodes_per_stage=8,
        failures_before_block=2,
    )

    ended, _ = controller._run_stage(
        RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL),
        Stage.STAGE_1_STRUCTURAL,
    )

    assert _blocks(journal) == []
    assert ended.blocked_slots == frozenset()
    stats = slot_stats(ended)
    assert stats["model"].attempts == 8
    # One acceptance, and it is not the gate's doing: the first candidate of
    # a stage entered with no incumbent is adopted without the gate being
    # called at all. The other seven went to AlwaysRejectGate and were
    # rejected — seven rulings, seven deltas, and not one of them a failure.
    assert stats["model"].accepted == 1
    assert len(stats["model"].deltas) == 7
    assert not any(e.executor_failed for e in ended.history)


def test_only_the_executor_failure_path_marks_the_history_entry():
    """The flag the breaker counts is set on exactly one path. Asserted on
    a run that produces an acceptance, a rejection and a failure, so a
    future edit that set it too eagerly is caught here."""
    controller, _ = _controller(
        executor=_ClimbingExecutor(fail_on=_fails_on_slot("model", times=1)),
        gate=_AlternatingGate(),
        policy=_StickyPolicy("model"),
        max_nodes_per_stage=4,
    )

    ended, _ = controller._run_stage(
        RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL),
        Stage.STAGE_1_STRUCTURAL,
    )

    flagged = [e for e in ended.history if e.executor_failed]
    assert len(flagged) == 1
    assert flagged[0].primary is None
    assert flagged[0].accepted is False
    assert flagged[0].tokens > 0  # a failure that cost real evaluation time
    assert any(e.accepted for e in ended.history)  # and the run did commit


def test_the_baseline_failing_blocks_nothing_because_no_policy_chose_it():
    """REPRODUCE_BASELINE evaluates a fixed published config. A baseline
    that will not run is a broken harness, not a bad arm, and there is no
    slot to hold responsible."""
    def hook(config):
        return (ErrorClass.DEPENDENCY, "synthetic baseline failure")

    controller, journal = _controller(
        executor=_ClimbingExecutor(fail_on=hook), failures_before_block=1
    )

    ended, _ = controller._run_stage(
        RunState(run_id="run-test", stage=Stage.REPRODUCE_BASELINE),
        Stage.REPRODUCE_BASELINE,
    )

    assert _blocks(journal) == []
    assert ended.blocked_slots == frozenset()
    assert ended.history[0].target_slot is None
    assert ended.history[0].executor_failed is True  # still recorded honestly


# -- DECISION 2: blocking is stage-scoped -------------------------------


def test_with_stage_clears_blocked_slots():
    """PINS DECISION 2 at the state level. Blocking is a budget decision
    scoped to one stage, and `with_blocked_slot` only ever adds — without
    this there is no unblock path anywhere and a transient OOM would delete
    an arm for the rest of the run."""
    state = RunState(
        run_id="r",
        stage=Stage.STAGE_1_STRUCTURAL,
        blocked_slots=frozenset({"model", "features"}),
    )

    moved = state.with_stage(Stage.STAGE_2_COMBINE)

    assert moved.blocked_slots == frozenset()
    assert state.blocked_slots == frozenset({"model", "features"})  # still pure


def test_clearing_the_blocks_keeps_the_history_that_justified_them():
    """The block goes; the evidence does not. A slot that failed twice
    comes back eligible but with a poor record, which is the right amount
    of memory for the bandit to carry across a stage boundary."""
    state = RunState(run_id="r", stage=Stage.STAGE_1_STRUCTURAL)
    state = state.with_outcome(
        "a", None, False, target_slot="model", tokens=1200, executor_failed=True
    )
    state = state.with_blocked_slot("model")

    moved = state.with_stage(Stage.STAGE_2_COMBINE)

    assert moved.blocked_slots == frozenset()
    assert slot_stats(moved)["model"].consecutive_failures == 1
    assert slot_stats(moved)["model"].attempts == 1


def test_a_slot_blocked_in_one_stage_is_offered_again_in_the_next():
    """PINS DECISION 2 end to end, through the real stage machinery.

    AND THE NON-OBVIOUS HALF OF IT, pinned because it surprised us. The
    BLOCK clears at the stage boundary; the EVIDENCE does not.
    `consecutive_failures` is folded over the whole history, so a slot that
    failed twice in stage one returns in stage two already at two, and one
    more failure re-blocks it immediately rather than buying it a fresh
    allowance of k.

    That is the intended reading of the two decisions together, not a gap
    between them. Stage entry does not make a failure un-happen, and a
    genuinely dead arm should not cost k evaluations in every stage. What
    the slot gets back is a real chance: one pull, and if it SUCCEEDS the
    counter resets to zero and its full allowance returns with it. So the
    arm below is offered twice in stage one - the two pulls that earned the
    block - and once in each stage after.
    """
    policy = _StickyPolicy("model")
    controller, journal = _controller(
        executor=_ClimbingExecutor(fail_on=_fails_on_slot("model")),
        generator=ScriptedGenerator(_script(40)),
        policy=policy,
        max_nodes_per_stage=4,
        failures_before_block=2,
    )

    final = controller.run()

    # Blocked once per search stage: cleared on entry, earned again.
    blocked = _blocks(journal)
    assert len(blocked) == SEARCH_STAGES
    assert {e.payload["slot"] for e in blocked} == {"model"}
    assert [e.payload["stage"] for e in blocked] == [
        Stage.STAGE_1_STRUCTURAL.value,
        Stage.STAGE_2_COMBINE.value,
        Stage.STAGE_3_TUNE.value,
    ]
    # Two pulls in stage one to earn the block, then one in each later
    # stage: offered again, but on a record that followed it across.
    assert sum(1 for offer in policy.offered if "model" in offer) == 2 + 1 + 1
    # The point of DECISION 2 all the same — every stage began with the arm
    # available rather than deleted for the rest of the run.
    assert "model" in policy.offered[0]
    assert "model" in policy.offered[4]  # first attempt of stage two
    assert final.stage is Stage.DONE
    assert final.blocked_slots == frozenset()  # cleared on the way to FINALIZE


# -- a stage whose arms all go down -------------------------------------


def test_a_stage_whose_arms_all_get_blocked_ends_cleanly_and_the_run_finishes():
    """The fully-blocked path, reached from a real run rather than a
    hand-built state: every arm fails twice, the candidate set empties, and
    the stage ends through the existing check instead of crashing or asking
    a policy to choose from nothing."""
    policy = _StickyPolicy("model")
    controller, journal = _controller(
        executor=_ClimbingExecutor(fail_on=_fails_on_any_search_candidate()),
        generator=ScriptedGenerator(_script(80)),
        policy=policy,
        max_nodes_per_stage=12,
        failures_before_block=2,
    )

    final = controller.run()

    # THE POINT: the policy was never handed an empty set.
    assert policy.offered  # it was called at all
    assert all(offer for offer in policy.offered)
    # Four structural arms, two failures each, then the stage ends.
    assert len(_blocks(journal)) == 4 + 4 + 6  # stages 1, 2 and 3's arms
    stage_ends = [
        e
        for e in journal.events_of_kind(EventKind.SLOT_BLOCKED)
        if e.payload["action"] == "end_stage"
    ]
    assert len(stage_ends) == SEARCH_STAGES
    # And the run still terminates through the ordinary path.
    assert final.stage is Stage.DONE
    assert _kinds(journal)[-1] == EventKind.RUN_END.value


def test_a_fully_blocked_stage_consumes_no_node_and_writes_no_history():
    controller, _ = _controller(
        executor=_ClimbingExecutor(fail_on=_fails_on_any_search_candidate()),
        generator=ScriptedGenerator(_script(40)),
        policy=_StickyPolicy("model"),
        max_nodes_per_stage=12,
        failures_before_block=2,
    )
    state = RunState(run_id="run-test", stage=Stage.STAGE_1_STRUCTURAL)

    ended, _ = controller._run_stage(state, Stage.STAGE_1_STRUCTURAL)

    # Eight attempts (four arms, two failures each), then the stage ends —
    # the remaining four nodes of the budget are simply not spent.
    assert ended.node == 8
    assert len(ended.history) == 8
    assert ended.blocked_slots == frozenset(STAGE_SLOTS[Stage.STAGE_1_STRUCTURAL])


# -- the card carries what the bandit reads -----------------------------


def test_the_state_card_carries_slot_stats_for_the_policy_and_survives_json():
    """FLAGGED ADDITION: `slot_stats` is a new top-level card key, and the
    card is W4's interface. An addition, not a rename."""
    controller, _ = _controller(policy=FixedOrderPolicy(STRUCTURAL_ORDER))
    state = controller.run()

    card = build_state_card(state)

    assert set(card["slot_stats"]) == set(STRUCTURAL_ORDER)
    assert card["slot_stats"]["features"]["attempts"] == 3
    assert card["slot_stats"] == slot_stats_as_json(slot_stats(state))
    assert json.loads(json.dumps(card, allow_nan=False)) == card


def test_the_policy_and_the_generator_see_the_same_slot_stats():
    """One card, built once, handed to both — so "what did the run look
    like when the slot was picked" and "what did the generator know" cannot
    become two questions with two answers."""
    class _RecordingPolicy:
        def __init__(self) -> None:
            self.cards: list[dict] = []

        def select_slot(self, state_card, candidate_slots):
            self.cards.append(dict(state_card))
            return candidate_slots[0]

    policy = _RecordingPolicy()
    generator = ScriptedGenerator(_script())
    controller, _ = _controller(policy=policy, generator=generator)

    controller.run()

    assert len(policy.cards) == len(generator.state_cards)
    for seen_by_policy, seen_by_generator in zip(
        policy.cards, generator.state_cards
    ):
        assert seen_by_policy["slot_stats"] == seen_by_generator["slot_stats"]


def test_the_card_shows_the_bandit_a_growing_record_of_the_arm_it_pulled():
    """The feedback loop, end to end: what the policy did last time is on
    the card it reads next time."""
    generator = ScriptedGenerator(_script())
    controller, _ = _controller(
        policy=FixedOrderPolicy(("model",)), generator=generator
    )

    controller.run()

    attempts = [
        (card["slot_stats"].get("model") or {}).get("attempts", 0)
        for card in generator.state_cards
    ]
    assert attempts == list(range(len(attempts)))  # 0, 1, 2, ... one per pull
