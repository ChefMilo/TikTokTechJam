"""Port adapters that let controller.Controller drive the real executor.

Four seams, four adapters, all of them thin. Nothing here decides
anything about the search — the Controller already does that, and the
entire point of this module is to stop reimplementing its loop by hand
(compare scripts/run_agent.py, which rebuilds node counting, circuit
breaking and baseline adoption from scratch).

SCOPE DISCIPLINE: lookup and translation only. The generator adapter does
not invent hypotheses, the realizer does not infer configs, and the
executor adapter does not re-implement executor.run.run_candidate's seed
loop — it recovers the arguments run_candidate wants and delegates, so
the two critical cache.save_predictions calls in executor/run.py stay on
the one code path that has been exercised.


THE TWO SLOT VOCABULARIES, AND WHY THIS MODULE HAS TO KNOW
----------------------------------------------------------
This is the non-obvious part, and getting it wrong silently breaks every
candidate, so it is written down rather than left to a reader to
rediscover.

The repo describes the SAME published FM baseline in two different slot
vocabularies:

    slot          controller.BASELINE_SLOTS    executor.realize.DEFAULT_SLOTS
    data_view     "full_log"                   "full"
    features      "five_field_categorical"     "baseline_5"
    weighting     "uniform"                    "none"
    model         fm{k,lr,epochs}              fm{k,lr}
    objective     "logloss"                    "bce"
    calibration   "none"                       "none"

Five of six differ. They are not in conflict — they are two names for one
baseline, owned by two workstreams — but they mean an adapter cannot
diff an incoming PipelineConfig against DEFAULT_SLOTS to find "the one
changed slot": every slot would look changed, on every candidate.

The baseline that matters here is the CONTROLLER's, because the
Controller is what builds the configs this adapter receives.
Controller._realize splices one realized SlotConfig onto
`state.incumbent_config.slots` or, before there is an incumbent, onto
BASELINE_SLOTS. So a candidate arriving at ExecutorPort.run is
"BASELINE_SLOTS with exactly one slot replaced", and diffing against
BASELINE_SLOTS recovers that one slot exactly.

Delegation then re-splices the recovered fragment onto the EXECUTOR's
vocabulary, because executor.run.run_candidate calls
executor.realize.build_config, which overlays onto DEFAULT_SLOTS. That
translation is the useful work this adapter does, and it has a
side-benefit worth naming: the resulting config_id is byte-identical to
the one scripts/run_agent.py produces for the same move, so an unattended
Controller run reuses the predictions already in harness.cache rather
than retraining what has been trained.

It also makes one composition legal that would otherwise fail. Move 8
(objective="bpr") refuses to compose with any weighting other than
"none" (executor/realize.py's `realize`). BASELINE_SLOTS says
weighting="uniform", which is not "none" — but the fragment is
re-spliced onto DEFAULT_SLOTS, where weighting IS "none", so bpr runs.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Optional

from contracts import (
    SLOT_ORDER,
    CandidateResult,
    ErrorClass,
    HypothesisPayload,
    JournalEvent,
    PipelineConfig,
    SlotConfig,
    SlotName,
    Status,
)
from controller.controller import BASELINE_SLOTS
from controller.ports import GeneratorExhausted, RealizerExhausted
from executor.journal import Journal
from executor.run import run_candidate as _run_candidate
from methods.scripted import _MOVES

__all__ = [
    "DurableJournal",
    "ExecutorAdapterError",
    "MovesRealizer",
    "RunCandidateExecutor",
    "ScriptedMoves",
    "SlotScriptedGenerator",
    "resolve_fragment",
]

MoveCatalog = Sequence[tuple[SlotConfig, HypothesisPayload]]


class ExecutorAdapterError(RuntimeError):
    """The adapter cannot express an incoming PipelineConfig as a
    (fragment, target_slot) pair that executor.run.run_candidate accepts.

    Deliberately NOT a PortExhausted. Every exception in
    controller.ports means "a collaborator has nothing left to give",
    which the Controller absorbs and carries on from. This one means the
    config it handed over is outside what this executor can run at all,
    which is either a wiring bug (a malformed config) or a real
    capability limit (a candidate differing from the baseline in more
    than one slot — build_config overlays exactly one).

    NOT ALLOWED TO ESCAPE ExecutorPort.run. ports.ExecutorPort is
    explicit that an exception there "would take the whole run down with
    it and lose the journal, which is the one artifact that makes the run
    auditable". So RunCandidateExecutor.run catches this and returns a
    FAILED CandidateResult classified CONTRACT, which is exactly the
    shape the error taxonomy and the circuit breaker already handle. The
    exception type still exists, and is still raised by
    `resolve_fragment`, so the assertion is directly testable and reads
    loudly in a stack trace during development.
    """


class ScriptedMoves:
    """Read-only index over methods.scripted's ten scripted moves.

    `_MOVES` is a tuple of (SlotConfig, HypothesisPayload) pairs — the
    fragment and the proposal that motivates it, authored together. Both
    adapters below are lookups into this index rather than anything
    cleverer, and that is the point: the mapping from "hypothesis" to
    "runnable config" was written by hand in methods/scripted.py and does
    not need to be inferred.

    Imports `_MOVES` (a private name) rather than calling
    ScriptedGenerator.propose() ten times. The private read is the more
    honest of the two: it is a pure data table, it is what
    scripts/run_agent.py effectively reconstructs anyway, and reaching
    for it makes the coupling visible here instead of hiding it behind a
    generator whose own signature this module exists to replace. Pass
    `moves=` to use a different catalog (every test does).
    """

    def __init__(self, moves: Optional[MoveCatalog] = None) -> None:
        catalog: MoveCatalog = _MOVES if moves is None else tuple(moves)
        self._moves: tuple[tuple[SlotConfig, HypothesisPayload], ...] = tuple(catalog)

        by_slot: dict[SlotName, list[tuple[SlotConfig, HypothesisPayload]]] = {}
        for fragment, hypothesis in self._moves:
            by_slot.setdefault(hypothesis.target_slot, []).append((fragment, hypothesis))
        self._by_slot = {slot: tuple(entries) for slot, entries in by_slot.items()}

        # HypothesisPayload is a frozen dataclass whose every field is
        # hashable (Citation is frozen too, predecessor_evidence is a
        # tuple), so the payload itself is the lookup key. Keying on the
        # object rather than on an index means the realizer cannot drift
        # out of step with the generator: it answers "what config was
        # authored for THIS proposal", not "what is the Nth config".
        self._fragment_by_hypothesis: dict[HypothesisPayload, SlotConfig] = {
            hypothesis: fragment for fragment, hypothesis in self._moves
        }

    def for_slot(self, target_slot: SlotName) -> tuple[tuple[SlotConfig, HypothesisPayload], ...]:
        """Every move targeting `target_slot`, in script order. Empty
        tuple for a slot the script never proposes anything for —
        `features` is the real example: the ten moves leave it alone.
        """
        return self._by_slot.get(target_slot, ())

    def fragment_for(self, hypothesis: HypothesisPayload) -> Optional[SlotConfig]:
        """The SlotConfig authored alongside `hypothesis`, or None."""
        return self._fragment_by_hypothesis.get(hypothesis)

    def baseline_fragment(self) -> tuple[SlotConfig, SlotName]:
        """The (fragment, slot) pair to delegate for the BASELINE config.

        Stage REPRODUCE_BASELINE evaluates `baseline_pipeline()`
        unchanged, so nothing differs from BASELINE_SLOTS and there is no
        "changed slot" to recover. The script's first `model` move IS the
        published baseline — move 1, "baseline_reproduce", fm k=16
        lr=0.001 — so delegating it reproduces exactly the config
        scripts/run_agent.py calls move 1, down to the config_id and
        therefore down to the cache key.
        """
        model_moves = self.for_slot("model")
        if not model_moves:
            raise ExecutorAdapterError(
                "move catalog has no 'model' move to stand in for the baseline "
                "config; cannot delegate REPRODUCE_BASELINE"
            )
        return model_moves[0][0], "model"

    def __len__(self) -> int:
        return len(self._moves)


class SlotScriptedGenerator:
    """GeneratorPort over the scripted moves, served BY SLOT.

    methods.scripted.ScriptedGenerator predates the current port and
    still has the old shape — `propose(state)` returning a
    `(SlotConfig, dict)` 2-tuple and raising StopIteration — and its own
    module docstring says wiring it to GeneratorPort "is a thin adapter
    around this class; it is not this class's job". This is that adapter,
    written against the move table directly.

    THE ADAPTATION THAT MATTERS is not the signature, it is who chooses
    the slot. The scripted generator dictates its own slot order, one
    move after another. GeneratorPort hands the slot IN, because the
    search policy owns that decision (see PolicyPort's docstring for why
    it was taken away from the generator). So this adapter keeps a cursor
    PER SLOT and serves the next unused move for whichever slot it is
    asked about. The script's content is preserved; only its ordering
    authority is given back to the policy.

    Consequently `payload.target_slot` always equals the requested
    `target_slot`, so the Controller's slot-mismatch guard never fires
    for this generator — the guard exists for an LLM that wanders, and a
    table lookup cannot.

    `state_card` is accepted and never read. This generator does not
    adapt to the run, by design; that is what makes an integration run
    against it a test of the wiring rather than of a model.
    """

    def __init__(self, moves: Optional[MoveCatalog] = None) -> None:
        self._catalog = ScriptedMoves(moves)
        self._served: dict[SlotName, int] = {}

    def propose(
        self, state_card: Mapping[str, Any], target_slot: SlotName
    ) -> HypothesisPayload:
        """The next unused scripted move for `target_slot`.

        Raises GeneratorExhausted once this slot's moves are used up, or
        immediately for a slot the script never targets. That ENDS THE
        RUN — the Controller treats generator exhaustion as a normal
        termination and finalises through its ordinary path, so the
        journal still gets FINALIZE and RUN_END. It is not a crash, but
        it is a real constraint on how long an unattended run lasts, and
        scripts/run_controller.py picks its policy order with that in
        mind.
        """
        available = self._catalog.for_slot(target_slot)
        served = self._served.get(target_slot, 0)
        if served >= len(available):
            raise GeneratorExhausted(
                f"the scripted move catalog has {len(available)} move(s) for "
                f"slot {target_slot!r} and all of them have been proposed; "
                "no further hypothesis to offer for this slot"
            )
        self._served[target_slot] = served + 1
        return available[served][1]

    @property
    def served(self) -> Mapping[SlotName, int]:
        """How many moves have been served per slot. For assertions and
        for a run summary; not read by the Controller."""
        return dict(self._served)


class MovesRealizer:
    """RealizerPort: the SlotConfig authored alongside a hypothesis.

    A LOOKUP, NOT AN INFERENCE, and the distinction is the whole reason
    this class exists. controller.fakes.DeterministicRealizer builds
    `SlotConfig(impl=hypothesis.citation.library_entry)`, which yields
    impls like "methods/library/fm.yaml#factorization_machine" — a
    perfectly reasonable stand-in for a registry-backed realizer, and one
    that executor/realize.py rejects with NotImplementedError for every
    single candidate, because the impls it implements are "fm",
    "exp_decay", "recent_window" and "bpr". Pairing the payload back to
    the fragment written next to it in methods/scripted.py is what makes
    a candidate actually runnable.
    """

    def __init__(self, moves: Optional[MoveCatalog] = None) -> None:
        self._catalog = ScriptedMoves(moves)

    def realize(self, hypothesis: HypothesisPayload) -> SlotConfig:
        """The scripted fragment for `hypothesis`.

        Raises RealizerExhausted for a payload this catalog never
        authored. That is a dead CANDIDATE, not a dead run: the
        Controller logs an ERROR classified CONTRACT, counts the node and
        moves on.
        """
        fragment = self._catalog.fragment_for(hypothesis)
        if fragment is None:
            raise RealizerExhausted(
                "no scripted SlotConfig was authored for this hypothesis "
                f"(target_slot={hypothesis.target_slot!r}, "
                f"citation={hypothesis.citation.key!r}); this realizer is a "
                "lookup over methods.scripted's move table and cannot invent one"
            )
        return fragment


def resolve_fragment(
    config: PipelineConfig,
    *,
    baseline_slots: Optional[Mapping[SlotName, SlotConfig]] = None,
    catalog: Optional[ScriptedMoves] = None,
) -> tuple[SlotConfig, SlotName]:
    """Recover the (fragment, target_slot) pair behind a PipelineConfig.

    The diff half of "diff and delegate". See the module docstring for
    why the diff is taken against the CONTROLLER's baseline vocabulary
    and not the executor's.

    Three outcomes:

      0 slots differ  -> the baseline itself (stage REPRODUCE_BASELINE).
                         Delegates the script's own baseline move; see
                         ScriptedMoves.baseline_fragment.
      1 slot  differs -> that slot and its fragment. The normal case:
                         Controller._realize splices exactly one slot.
      2+ differ       -> ExecutorAdapterError. Not a defensive assert but
                         a real limit: executor.realize.build_config
                         overlays ONE slot onto DEFAULT_SLOTS, so a
                         two-slot candidate cannot be expressed as a
                         run_candidate call at all. Reachable only after
                         an acceptance moves the incumbent off the
                         baseline (a stage-2 recombination), which is why
                         it must degrade rather than explode — see
                         ExecutorAdapterError.

    A config missing a slot, or carrying one SLOT_ORDER does not name,
    also raises: that is a malformed config rather than an unsupported
    one, and it means something upstream is wrong.
    """
    base = BASELINE_SLOTS if baseline_slots is None else baseline_slots
    catalog = ScriptedMoves() if catalog is None else catalog

    expected = set(SLOT_ORDER)
    actual = set(config.slots)
    if actual != expected:
        raise ExecutorAdapterError(
            "PipelineConfig does not name exactly the six slots in "
            f"SLOT_ORDER: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    # SLOT_ORDER rather than dict order, so the reported slot list is
    # stable across runs and processes.
    differing = [slot for slot in SLOT_ORDER if config.slots[slot] != base[slot]]

    if not differing:
        return catalog.baseline_fragment()
    if len(differing) == 1:
        slot = differing[0]
        return config.slots[slot], slot
    raise ExecutorAdapterError(
        f"candidate differs from the controller baseline in {len(differing)} "
        f"slots {differing}, but executor.realize.build_config overlays "
        "exactly one slot onto DEFAULT_SLOTS; this candidate cannot be "
        "expressed as a run_candidate call"
    )


class RunCandidateExecutor:
    """ExecutorPort backed by the real executor.run.run_candidate.

    DIFF AND DELEGATE, rather than re-implementing the seed loop.
    run_candidate does more than train: for every seed it writes both
    prediction splits into harness.cache, and executor/run.py calls one
    of those saves "THE SINGLE MOST IMPORTANT LINE IN THIS FILE" —
    without it harness.gate silently degrades to a seed-level bootstrap
    with a ~12.5% false-positive rate, and the run still looks completely
    normal. Copying that loop to avoid an argument mismatch would put a
    second, unexercised copy of it in the tree. So this adapter recovers
    the arguments and calls the real thing.

    NEVER RAISES, per ExecutorPort. A failure inside run_candidate is
    already returned as CandidateResult(status=FAILED) with a classified
    error_class — that is its own documented contract. The one thing this
    adapter adds is catching ExecutorAdapterError and returning the same
    shape, so a config this executor cannot express is a dead candidate
    the circuit breaker can act on rather than a dead run.

    `journal` DEFAULTS TO None, and that is deliberate. run_candidate
    will happily log EVAL_START/EVAL_RESULT/ERROR itself, but the
    Controller already emits its own events for the same node, and
    run_candidate numbers its node independently (journal.current_node +
    1). Passing a journal to both puts two overlapping accounts of one
    attempt in the log at disagreeing node numbers. The Controller is the
    authority on the journal for a Controller-driven run, so the executor
    stays quiet.
    """

    def __init__(
        self,
        *,
        journal: Optional[Journal] = None,
        baseline_slots: Optional[Mapping[SlotName, SlotConfig]] = None,
        catalog: Optional[ScriptedMoves] = None,
        runner: Any = None,
    ) -> None:
        self._journal = journal
        self._baseline_slots = BASELINE_SLOTS if baseline_slots is None else baseline_slots
        self._catalog = ScriptedMoves() if catalog is None else catalog
        # Injectable so a test can prove delegation without training an
        # FM. Defaults to the real function; nothing in production passes
        # this.
        self._runner = _run_candidate if runner is None else runner
        self.calls: list[tuple[SlotConfig, SlotName, tuple[int, ...]]] = []
        """Every delegation, in order — (fragment, target_slot, seeds).
        A run summary reads this; so do the delegation tests."""

    def run(self, config: PipelineConfig, seeds: Sequence[int]) -> CandidateResult:
        try:
            fragment, target_slot = resolve_fragment(
                config, baseline_slots=self._baseline_slots, catalog=self._catalog
            )
        except ExecutorAdapterError as exc:
            return CandidateResult(
                config_id=config.config_id,
                status=Status.FAILED,
                val={},
                backtest={},
                # CONTRACT, matching how the Controller classifies its own
                # realizer/generator port violations: the candidate never
                # reached training, so nothing about the METHOD failed.
                error_class=ErrorClass.CONTRACT,
                error_excerpt=repr(exc)[:2000],
            )

        self.calls.append((fragment, target_slot, tuple(seeds)))
        return self._runner(
            fragment, target_slot, seeds=tuple(seeds), journal=self._journal
        )


class DurableJournal:
    """JournalPort over the durable executor.journal.Journal.

    Journal.append already matches the port exactly, and the Controller
    calls nothing else — so a bare Journal would work today by accident.
    `replay` is where they diverge: the port declares an instance method
    `replay(run_id) -> Iterator`, while Journal.replay is a CLASSMETHOD
    taking a PATH and returning a list. Both are reasonable; they are
    just not the same method. Since Protocols are structural and
    @runtime_checkable only checks that a name exists, an isinstance
    assertion would pass on the bare Journal and then `replay(run_id)`
    would try to open a run id as a filename.

    Wrapping is ~15 lines and makes the conformance real rather than
    incidental. Filtering by run_id also matters for a durable journal in
    a way it does not for the in-memory double: one file can accumulate
    several runs, and a reader asking for one run's events should not be
    handed another's.
    """

    def __init__(self, journal: Journal) -> None:
        self._journal = journal

    @property
    def journal(self) -> Journal:
        """The wrapped Journal, for callers that want its path or its
        log_* helpers (log_intervention, in particular)."""
        return self._journal

    def append(self, event: JournalEvent) -> None:
        self._journal.append(event)

    def replay(self, run_id: str) -> Iterator[JournalEvent]:
        """This run's events, in append order, streamed.

        Reads from disk on every call rather than caching: the file is
        the source of truth, and a replay during a live run should see
        what has actually been fsynced.
        """
        return iter(
            [
                event
                for event in Journal.replay(str(self._journal.path))
                if event.run_id == run_id
            ]
        )
