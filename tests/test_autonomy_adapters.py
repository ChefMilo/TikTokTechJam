"""Tests for autonomy.adapters — the four port adapters that let the real
Controller drive the real executor.

FAST BY CONSTRUCTION: no test here trains a factorization machine, loads
the KuaiRand CSVs, or touches harness.cache. The executor adapter takes an
injectable `runner`, so its delegation and the end-to-end Controller loop
are both exercised against a stub that returns synthetic CandidateResults
in microseconds. What is under test is the WIRING — that the Controller
can drive these adapters to completion, and that each adapter translates
what it claims to translate. Whether an FM trains is executor/run.py's
question, and tests/test_realize.py already asks it.
"""

import pytest

from contracts import (
    SLOT_ORDER,
    Budget,
    CandidateResult,
    Citation,
    ErrorClass,
    EventKind,
    HypothesisPayload,
    Metrics,
    PipelineConfig,
    SlotConfig,
    Status,
)
from controller.controller import BASELINE_SLOTS, Controller, baseline_pipeline
from controller.fakes import AlwaysAcceptGate, AlwaysRejectGate, InMemoryJournal
from controller.policy import FixedOrderPolicy
from controller.ports import (
    ExecutorPort,
    GeneratorExhausted,
    GeneratorPort,
    JournalPort,
    RealizerExhausted,
    RealizerPort,
)
from executor.journal import Journal
from executor.realize import DEFAULT_SLOTS
from methods.scripted import _MOVES

from autonomy.adapters import (
    DurableJournal,
    ExecutorAdapterError,
    MovesRealizer,
    RunCandidateExecutor,
    ScriptedMoves,
    SlotScriptedGenerator,
    resolve_fragment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics(primary: float) -> Metrics:
    """A Metrics whose `primary` (the unweighted mean) is `primary`."""
    return Metrics(values={"GAUC": primary, "nDCG@5": primary})


def _stub_runner(results=None, primary: float = 0.60):
    """A stand-in for executor.run.run_candidate that never trains.

    Returns a callable with run_candidate's signature plus a `.calls`
    list. `results` optionally maps a fragment impl to a canned
    CandidateResult, so a test can make one specific move fail.
    """
    results = results or {}

    def runner(fragment, target_slot, seeds=(0, 1, 2), journal=None):
        runner.calls.append((fragment, target_slot, tuple(seeds), journal))
        if fragment.impl in results:
            return results[fragment.impl]
        return CandidateResult(
            config_id=f"stub_{fragment.impl}",
            status=Status.OK,
            val={seed: _metrics(primary) for seed in seeds},
            backtest={seed: _metrics(primary) for seed in seeds},
            wall_seconds=0.0,
        )

    runner.calls = []
    return runner


def _hypothesis(target_slot: str = "model") -> HypothesisPayload:
    """A payload no scripted move authored."""
    return HypothesisPayload(
        target_slot=target_slot,
        rationale="not from the script",
        citation=Citation(key="nobody2026", url="https://example.com", library_entry="x#y"),
        expected_gain=0.01,
        expected_cost_s=1.0,
    )


# ---------------------------------------------------------------------------
# The two-vocabulary discovery this module is built around
# ---------------------------------------------------------------------------


def test_controller_and_executor_baselines_use_different_slot_vocabularies():
    """THE REASON resolve_fragment DIFFS AGAINST BASELINE_SLOTS.

    controller.BASELINE_SLOTS and executor.realize.DEFAULT_SLOTS describe
    the same published FM baseline in different words. Diffing a
    Controller-built candidate against DEFAULT_SLOTS would report five or
    six changed slots on EVERY candidate and the adapter would reject all
    of them. Pinned as a test so that if the two vocabularies are ever
    reconciled, this fails and tells the next reader the workaround can
    go — rather than the workaround quietly outliving its reason.
    """
    differing = [
        slot for slot in SLOT_ORDER if BASELINE_SLOTS[slot] != DEFAULT_SLOTS[slot]
    ]
    assert differing == ["data_view", "features", "weighting", "model", "objective"]
    assert BASELINE_SLOTS["calibration"] == DEFAULT_SLOTS["calibration"]


# ---------------------------------------------------------------------------
# Port conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter, port",
    [
        (SlotScriptedGenerator(), GeneratorPort),
        (MovesRealizer(), RealizerPort),
        (RunCandidateExecutor(runner=_stub_runner()), ExecutorPort),
    ],
)
def test_adapters_satisfy_their_ports(adapter, port):
    assert isinstance(adapter, port)


def test_durable_journal_satisfies_journal_port(tmp_path):
    journal = Journal(str(tmp_path / "j.jsonl"), run_id="run-1")
    assert isinstance(DurableJournal(journal), JournalPort)


# ---------------------------------------------------------------------------
# ScriptedMoves — the index
# ---------------------------------------------------------------------------


def test_catalog_indexes_every_scripted_move_by_slot():
    catalog = ScriptedMoves()

    assert len(catalog) == len(_MOVES) == 10
    assert len(catalog.for_slot("model")) == 4
    assert len(catalog.for_slot("objective")) == 2
    assert len(catalog.for_slot("calibration")) == 2
    assert len(catalog.for_slot("weighting")) == 1
    assert len(catalog.for_slot("data_view")) == 1
    # The script never proposes anything for `features` — see
    # scripts/run_controller.py's default policy order.
    assert catalog.for_slot("features") == ()


def test_catalog_baseline_fragment_is_the_scripts_own_move_one():
    """REPRODUCE_BASELINE delegates move 1, so the config it produces is
    the one scripts/run_agent.py calls move 1 — same config_id, same
    cache key."""
    fragment, target_slot = ScriptedMoves().baseline_fragment()

    assert target_slot == "model"
    assert fragment == _MOVES[0][0]
    assert fragment.impl == "fm"
    assert fragment.params == {"k": 16, "lr": 0.001, "epochs": 40}


# ---------------------------------------------------------------------------
# SlotScriptedGenerator
# ---------------------------------------------------------------------------


def test_generator_serves_moves_for_the_requested_slot_in_script_order():
    generator = SlotScriptedGenerator()

    first = generator.propose({}, "model")
    second = generator.propose({}, "model")

    model_moves = ScriptedMoves().for_slot("model")
    assert first is model_moves[0][1]
    assert second is model_moves[1][1]
    assert first != second


def test_generator_returns_a_payload_for_exactly_the_requested_slot():
    """The Controller treats a mismatch here as a CONTRACT violation and
    kills the candidate (controller.py's slot-mismatch guard). A table
    lookup keyed by slot cannot wander, and this pins that."""
    generator = SlotScriptedGenerator()

    for slot in ("model", "objective", "weighting", "data_view", "calibration"):
        payload = generator.propose({}, slot)
        assert isinstance(payload, HypothesisPayload)
        assert payload.target_slot == slot


def test_generator_tracks_slots_independently():
    generator = SlotScriptedGenerator()

    generator.propose({}, "model")
    objective_first = generator.propose({}, "objective")
    model_second = generator.propose({}, "model")

    assert objective_first.target_slot == "objective"
    assert model_second.target_slot == "model"
    assert generator.served == {"model": 2, "objective": 1}


def test_generator_raises_generator_exhausted_when_a_slot_is_drained():
    generator = SlotScriptedGenerator()

    # `weighting` has exactly one scripted move.
    generator.propose({}, "weighting")

    with pytest.raises(GeneratorExhausted, match="weighting"):
        generator.propose({}, "weighting")


def test_generator_raises_immediately_for_a_slot_the_script_never_targets():
    with pytest.raises(GeneratorExhausted, match="features"):
        SlotScriptedGenerator().propose({}, "features")


# ---------------------------------------------------------------------------
# MovesRealizer
# ---------------------------------------------------------------------------


def test_realizer_maps_a_scripted_payload_to_an_executor_runnable_fragment():
    """The fakes' DeterministicRealizer produces
    impl="methods/library/fm.yaml#factorization_machine", which
    executor/realize.py rejects. This must produce the real impl."""
    realizer = MovesRealizer()

    for fragment, hypothesis in _MOVES:
        assert realizer.realize(hypothesis) is fragment

    model_payload = ScriptedMoves().for_slot("model")[0][1]
    assert realizer.realize(model_payload).impl == "fm"


def test_realizer_produces_impls_the_executor_actually_implements():
    """Not every scripted move is implemented — four deliberately are
    not, and that is the error-taxonomy path. But the ones the executor
    does implement must come through under the names it dispatches on."""
    realizer = MovesRealizer()
    impls = {realizer.realize(h).impl for _, h in _MOVES}

    assert {"fm", "exp_decay", "recent_window", "bpr"} <= impls
    # The four that raise NotImplementedError by design.
    assert {"multitask_bce", "duration_debias_cwm", "lightgbm", "popularity_blend"} <= impls


def test_realizer_raises_realizer_exhausted_for_an_unauthored_hypothesis():
    with pytest.raises(RealizerExhausted, match="lookup"):
        MovesRealizer().realize(_hypothesis())


# ---------------------------------------------------------------------------
# resolve_fragment — the diff half of diff-and-delegate
# ---------------------------------------------------------------------------


def test_resolve_fragment_maps_the_baseline_config_to_the_baseline_move():
    fragment, target_slot = resolve_fragment(baseline_pipeline())

    assert (fragment, target_slot) == ScriptedMoves().baseline_fragment()


def test_resolve_fragment_recovers_a_single_changed_slot():
    changed = SlotConfig(impl="exp_decay", params={"half_life_days": 5.0})
    slots = dict(BASELINE_SLOTS)
    slots["weighting"] = changed

    fragment, target_slot = resolve_fragment(PipelineConfig(slots=slots))

    assert target_slot == "weighting"
    assert fragment is changed


def test_resolve_fragment_raises_when_more_than_one_slot_differs():
    """executor.realize.build_config overlays exactly ONE slot onto
    DEFAULT_SLOTS, so a two-slot candidate is not expressible as a
    run_candidate call. Reachable once an acceptance moves the incumbent
    off the baseline."""
    slots = dict(BASELINE_SLOTS)
    slots["weighting"] = SlotConfig(impl="exp_decay", params={"half_life_days": 5.0})
    slots["objective"] = SlotConfig(impl="bpr", params={"pairs_per_batch": 8192})

    with pytest.raises(ExecutorAdapterError, match="differs from the controller baseline in 2 slots"):
        resolve_fragment(PipelineConfig(slots=slots))


def test_resolve_fragment_raises_on_a_malformed_config():
    slots = dict(BASELINE_SLOTS)
    del slots["calibration"]

    with pytest.raises(ExecutorAdapterError, match="missing=\\['calibration'\\]"):
        resolve_fragment(PipelineConfig(slots=slots))


# ---------------------------------------------------------------------------
# RunCandidateExecutor — the delegate half
# ---------------------------------------------------------------------------


def test_executor_adapter_delegates_the_recovered_fragment_and_slot():
    runner = _stub_runner()
    adapter = RunCandidateExecutor(runner=runner)
    changed = SlotConfig(impl="bpr", params={"pairs_per_batch": 8192})
    slots = dict(BASELINE_SLOTS)
    slots["objective"] = changed

    result = adapter.run(PipelineConfig(slots=slots), seeds=(0, 1, 2))

    assert result.status is Status.OK
    assert len(runner.calls) == 1
    fragment, target_slot, seeds, journal = runner.calls[0]
    assert fragment is changed
    assert target_slot == "objective"
    assert seeds == (0, 1, 2)
    # The Controller is the sole journaller for a Controller-driven run.
    assert journal is None
    assert adapter.calls == [(changed, "objective", (0, 1, 2))]


def test_executor_adapter_delegates_the_baseline_move_for_the_baseline_config():
    runner = _stub_runner()
    adapter = RunCandidateExecutor(runner=runner)

    adapter.run(baseline_pipeline(), seeds=(0,))

    fragment, target_slot, seeds, _ = runner.calls[0]
    assert (fragment, target_slot) == ScriptedMoves().baseline_fragment()
    assert seeds == (0,)


def test_executor_adapter_returns_failed_rather_than_raising_on_a_bad_config():
    """ExecutorPort is explicit that raising here "would take the whole
    run down with it and lose the journal". A config this executor cannot
    express must be a dead candidate, not a dead run."""
    runner = _stub_runner()
    adapter = RunCandidateExecutor(runner=runner)
    slots = dict(BASELINE_SLOTS)
    slots["weighting"] = SlotConfig(impl="exp_decay", params={"half_life_days": 5.0})
    slots["objective"] = SlotConfig(impl="bpr", params={"pairs_per_batch": 8192})

    result = adapter.run(PipelineConfig(slots=slots), seeds=(0, 1, 2))

    assert result.status is Status.FAILED
    assert result.error_class is ErrorClass.CONTRACT
    assert "2 slots" in result.error_excerpt
    assert runner.calls == []  # never reached the executor


# ---------------------------------------------------------------------------
# DurableJournal
# ---------------------------------------------------------------------------


def test_durable_journal_appends_through_and_replays_by_run_id(tmp_path):
    path = tmp_path / "journal.jsonl"
    underlying = Journal(str(path), run_id="run-a")
    adapter = DurableJournal(underlying)

    underlying.log_run_start(seeds=[0])
    underlying.log_finalize(stop_reason="cap")

    replayed = list(adapter.replay("run-a"))
    assert [e.kind for e in replayed] == [EventKind.RUN_START, EventKind.FINALIZE]
    # One file can hold several runs; a reader asking for one must not be
    # handed another's.
    assert list(adapter.replay("run-b")) == []
    assert adapter.journal is underlying


# ---------------------------------------------------------------------------
# End to end: the REAL Controller driving these adapters
# ---------------------------------------------------------------------------


def _run_controller(gate, *, max_nodes_per_stage=2, runner=None, journal=None):
    runner = runner or _stub_runner()
    journal = journal if journal is not None else InMemoryJournal()
    executor = RunCandidateExecutor(runner=runner)
    controller = Controller(
        executor=executor,
        gate=gate,
        generator=SlotScriptedGenerator(),
        realizer=MovesRealizer(),
        policy=FixedOrderPolicy(("model", "objective", "weighting")),
        journal=journal,
        budget=Budget(),
        seeds=(0, 1, 2),
        max_nodes_per_stage=max_nodes_per_stage,
        run_id="wiring-test",
    )
    return controller.run(), journal, executor, runner


def test_controller_drives_the_adapters_to_a_clean_finish():
    """THE WIRING PROOF. The real Controller, the real state machine, our
    three adapters, and a stub executor — start to RUN_END with no
    exception escaping."""
    state, journal, executor, runner = _run_controller(AlwaysRejectGate())

    kinds = [e.kind for e in journal.events]
    assert kinds[0] is EventKind.RUN_START
    assert kinds[-1] is EventKind.RUN_END
    assert EventKind.FINALIZE in kinds
    # The baseline plus at least one search candidate actually reached the
    # executor through the adapter.
    assert len(executor.calls) >= 2
    assert len(runner.calls) == len(executor.calls)
    assert state.stage.value == "done"


def test_controller_run_evaluates_the_baseline_first_through_the_adapter():
    _, journal, executor, _ = _run_controller(AlwaysRejectGate())

    first_fragment, first_slot, _ = executor.calls[0]
    assert (first_fragment, first_slot) == ScriptedMoves().baseline_fragment()
    # And the baseline is adopted with no gate call, per the Controller's
    # first-candidate rule.
    decisions = [e for e in journal.events if e.kind is EventKind.DECISION]
    assert decisions[0].payload["reason"].startswith("first candidate adopted")


def test_controller_records_hypotheses_and_emitted_code_for_search_candidates():
    _, journal, _, _ = _run_controller(AlwaysRejectGate())

    hypotheses = [e for e in journal.events if e.kind is EventKind.HYPOTHESIS]
    emitted = [e for e in journal.events if e.kind is EventKind.CODE_EMITTED]

    assert hypotheses, "no HYPOTHESIS reached the journal"
    assert len(emitted) == len(hypotheses)
    # Every realized impl is a real executor impl name, not a library path.
    for event in emitted:
        assert "/" not in event.payload["impl"]


def test_controller_survives_an_executor_failure_and_keeps_going():
    """A move the executor cannot realize comes back FAILED, is
    classified, and the run continues — the robustness path, driven by
    the real state machine rather than run_agent.py's own dict."""
    failing = CandidateResult(
        config_id="cfg_unimplemented",
        status=Status.FAILED,
        val={},
        backtest={},
        error_class=ErrorClass.CONTRACT,
        error_excerpt="NotImplementedError('no realization implemented for objective impl')",
        wall_seconds=0.1,
    )
    runner = _stub_runner(results={"multitask_bce": failing})

    state, journal, _, _ = _run_controller(AlwaysRejectGate(), runner=runner)

    errors = [e for e in journal.events if e.kind is EventKind.ERROR]
    assert any(e.payload.get("error_class") == ErrorClass.CONTRACT.value for e in errors)
    assert journal.events[-1].kind is EventKind.RUN_END
    assert state.stage.value == "done"


def test_multi_slot_candidate_degrades_to_a_failed_candidate_not_a_dead_run():
    """With an accepting gate the incumbent moves off the baseline, so a
    later candidate differs in two slots. The adapter must turn that into
    a FAILED CandidateResult and let the run finish."""
    state, journal, executor, _ = _run_controller(
        AlwaysAcceptGate(), max_nodes_per_stage=3
    )

    assert journal.events[-1].kind is EventKind.RUN_END
    assert state.stage.value == "done"
    # Whether a two-slot candidate actually arises depends on how many
    # acceptances the stages produce, so this asserts the invariant that
    # matters either way: nothing escaped, and the run finalised.
    errors = [e for e in journal.events if e.kind is EventKind.ERROR]
    for event in errors:
        assert event.payload.get("error_class") in {
            ErrorClass.CONTRACT.value,
            None,
        }


def test_controller_writes_a_durable_journal_through_the_adapter(tmp_path):
    """The same loop, but journalling to disk — the shape an unattended
    run actually uses."""
    path = tmp_path / "journal_controller.jsonl"
    underlying = Journal(str(path), run_id="wiring-test")

    state, _, _, _ = _run_controller(AlwaysRejectGate(), journal=DurableJournal(underlying))

    replayed = Journal.replay(str(path))
    assert replayed, "nothing reached disk"
    assert replayed[0].kind is EventKind.RUN_START
    assert replayed[-1].kind is EventKind.RUN_END
    assert all(e.run_id == "wiring-test" for e in replayed)
    assert state.stage.value == "done"
