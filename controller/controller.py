"""The Controller: a deliberately boring, fully auditable state machine.

It walks STAGE_ORDER, and within each stage attempts a fixed number of
candidates. For each candidate it asks the generator for a hypothesis,
asks the realizer to turn that hypothesis into slot code, splices the
result into the incumbent pipeline, has the executor evaluate it, asks the
gate whether it beat the incumbent, and writes an event to the journal at
every step.

THE CONTROLLER CONTAINS NO ACCEPTANCE LOGIC.
--------------------------------------------
It obeys `verdict.accept` and nothing else. There is no threshold here,
no significance test, no convergence check, no comparison of deltas
against epsilon — and none may be added. All of that lives behind
GatePort, where it can be tested against known ground truth in isolation.

This is not fastidiousness. The moment a threshold appears in this file
there are two places that decide what counts as an improvement, they
disagree under some input, and the journal stops being an explanation of
why a candidate was kept. If you are reading this because you want to add
"just a quick check that delta > 0" — that belongs in the gate.

WHAT COUNTS AS AN ITERATION
---------------------------
A committed revision — a candidate the gate accepted — not every attempt.
The organizers' rule ends a run after N consecutive iterations that improve
the primary by no more than epsilon; counting rejected dead ends toward N
would end a search the moment it tried three bad ideas in a row, which is
the opposite of exploring. Every CONVERGENCE_CHECK payload stamps
`iteration_definition` so the journal says which reading produced its
numbers, and `node` is tracked alongside `iteration` so the log can be
re-rendered under the other reading without re-running anything. See
controller/convergence.py.

Determinism: given the same doubles, the same seeds and the same run_id,
two runs emit an identical sequence of events apart from `ts`. Nothing
here sleeps, writes files, touches the network, or reads the clock except
to timestamp an event.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from contracts import (
    Budget,
    CandidateResult,
    ErrorClass,
    EventKind,
    HypothesisPayload,
    JournalEvent,
    PipelineConfig,
    SlotConfig,
    SLOT_ORDER,
    SlotName,
    Status,
)
from controller.ports import (
    ExecutorPort,
    GatePort,
    GeneratorExhausted,
    GeneratorPort,
    JournalPort,
    PolicyContractError,
    PolicyPort,
    RealizerExhausted,
    RealizerPort,
)
from controller.convergence import ITERATION_DEFINITION, assess, is_significant
from controller.state import (
    RunState,
    STAGE_ORDER,
    STAGE_SLOTS,
    Stage,
    build_state_card,
    next_stage,
    slot_stats,
)

__all__ = [
    "BASELINE_SLOTS",
    "MISPROPOSED_CONFIG_ID",
    "UNREALIZED_CONFIG_ID",
    "Controller",
    "baseline_pipeline",
]

UNREALIZED_CONFIG_ID = "<unrealized>"
"""Stands in for a candidate the realizer could not produce.

`HistoryEntry.config_id` is a str, and a hypothesis that never became a
config has no content to hash. A visible sentinel beats an empty string:
in the state card's `recent_history` it reads as an explicit "this attempt
produced nothing", rather than as a value someone forgot to fill in.
"""


MISPROPOSED_CONFIG_ID = "<slot-mismatch>"
"""Stands in for a hypothesis the generator aimed at the wrong slot.

Kept distinct from UNREALIZED_CONFIG_ID rather than folded into it, even
though both mark "this attempt produced no config". They are different
failures with different owners: an unrealized candidate means the realizer
could not write the code, while this one means the generator ignored an
explicit instruction. Those want different fixes and should be countable
separately in the log - a run with ten of these has a prompting problem,
a run with ten unrealized candidates has a library problem.
"""


BASELINE_SLOTS: dict[str, SlotConfig] = {
    "data_view": SlotConfig(impl="full_log"),
    "features": SlotConfig(impl="five_field_categorical"),
    "weighting": SlotConfig(impl="uniform"),
    "model": SlotConfig(impl="fm", params={"k": 16, "lr": 0.001, "epochs": 40}),
    "objective": SlotConfig(impl="logloss"),
    "calibration": SlotConfig(impl="none"),
}
"""The organizers' published FM baseline, expressed in our slot vocabulary.

Mirrors vendor/kuairand-starter-kit/baseline_scores.json: the five
categorical fields the FM consumes, and k=16 / lr=0.001 / max 40 epochs.
REPRODUCE_BASELINE evaluates exactly this, so that the very first number
the run produces is one we can check against a published figure (validation
primary 0.6016). If it does not reproduce, everything downstream is
measured against a broken ruler and no amount of search will save it.
"""


def baseline_pipeline() -> PipelineConfig:
    """A fully-occupied PipelineConfig for the published FM baseline.

    Every slot must be filled: PipelineConfig.slot_hash indexes
    `self.slots` directly while walking SLOT_ORDER, so a partial config
    raises KeyError the moment `config_id` is read.
    """
    return PipelineConfig(slots=dict(BASELINE_SLOTS))


class Controller:
    """Drives one experiment run from INIT to DONE.

    Collaborators arrive as ports (see controller/ports.py) so the whole
    loop can be exercised against deterministic doubles long before W1's
    gate, W3's executor or W4's generator exist.
    """

    def __init__(
        self,
        *,
        executor: ExecutorPort,
        gate: GatePort,
        generator: GeneratorPort,
        realizer: RealizerPort,
        policy: PolicyPort,
        journal: JournalPort,
        budget: Optional[Budget] = None,
        seeds: Sequence[int] = (0,),
        max_nodes_per_stage: int = 3,
        failures_before_block: int = 2,
        run_id: Optional[str] = None,
    ) -> None:
        self._executor = executor
        self._gate = gate
        self._generator = generator
        # Required, never defaulted. A default realizer would silently
        # supply the very wiring this seam exists to make explicit, and a
        # caller who forgot to pass one would get a working run built on a
        # component they never chose.
        self._realizer = realizer
        # Required, never defaulted - same reasoning as the realizer above,
        # and this seam is the entire point of the change that introduced
        # it. Slot selection used to happen implicitly: the Controller read
        # `payload.target_slot` off whatever the generator returned and
        # spliced there. Defaulting to, say, UniformPolicy would swap one
        # invisible choice for another and a caller who never thought about
        # the search policy would still get one, silently. Making it
        # required forces the wiring into every call site, where it is
        # visible in a diff and in a stack trace.
        self._policy = policy
        self._journal = journal
        # An unlimited Budget by default. No agreed wall-clock, token or
        # evaluation ceiling exists anywhere in the repo yet, and inventing
        # one here would be fiction that later reads as a decision someone
        # made on purpose.
        self._budget = budget if budget is not None else Budget()
        self._seeds = tuple(seeds)
        # Renamed from `nodes_per_stage` to say what it is: a backstop, not
        # the thing that decides when a stage is done. Convergence ends the
        # run; this caps a stage so that a run which never converges — with
        # an unlimited default Budget and an accepting gate — still
        # terminates. It is the only hard termination guarantee left.
        self._max_nodes_per_stage = max_nodes_per_stage
        if failures_before_block < 1:
            # Zero or less would block a slot before it had failed at all,
            # emptying the candidate set on the first attempt of a stage
            # and ending every stage without evaluating anything. A raise
            # rather than a clamp: silently correcting it would hide a
            # wiring mistake behind a run that looked merely unlucky.
            raise ValueError(
                "failures_before_block must be at least 1, got "
                f"{failures_before_block}"
            )
        # A TUNABLE, NOT A LAW. Two consecutive executor failures on one
        # slot is a defensible default and nothing more: it stops the agent
        # re-debugging one bad idea for a whole stage, while still giving
        # a slot a second chance, since a single failure is as likely to be
        # a flaky machine as a dead end. Nobody has measured the right
        # number - it is exposed here so a run can change it without
        # editing this file, and so the journal can record what was used.
        self._failures_before_block = failures_before_block
        # Callers who want a byte-reproducible journal should pass run_id;
        # the generated default is deliberately unique per run so two runs
        # cannot be confused for one another in a shared log.
        self._run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"

    # -- public API ----------------------------------------------------

    def run(self) -> RunState:
        """Drive the loop to completion and return the final state."""
        state = RunState(
            run_id=self._run_id, stage=Stage.INIT, budget=self._budget
        )
        self._emit(
            state,
            EventKind.RUN_START,
            {
                "seeds": list(self._seeds),
                "max_nodes_per_stage": self._max_nodes_per_stage,
                "stage_order": [s.value for s in STAGE_ORDER],
            },
        )

        stop_reason: Optional[str] = None
        for stage in STAGE_ORDER:
            if stage is Stage.FINALIZE:
                # Entered by _finalize, so that the abort path and the
                # normal path converge on exactly one FINALIZE entry.
                break
            state = self._enter(state, stage)
            if stage is Stage.INIT:
                continue  # nothing to evaluate; the stage exists to mark the start
            state, stop_reason = self._run_stage(state, stage)
            if stop_reason is not None:
                break

        return self._finalize(state, stop_reason)

    # -- stage machinery -----------------------------------------------

    def _enter(self, state: RunState, stage: Stage) -> RunState:
        """Transition into `stage` and log it.

        Guards the progression: except for FINALIZE, a stage may only be
        entered if `next_stage` says it follows the current one. FINALIZE
        is the single sanctioned non-linear transition, because an abort
        (out of budget, out of hypotheses) has to be able to reach the
        terminal path from wherever it happened. Walking the intervening
        stages instead would emit STAGE_CHANGE events for stages that
        never ran, which is a worse lie than the jump.
        """
        if stage is not state.stage:
            if stage is not Stage.FINALIZE and stage is not next_stage(state.stage):
                raise RuntimeError(
                    f"illegal stage transition {state.stage.value} -> {stage.value}"
                )
            state = state.with_stage(stage)
        self._emit(state, EventKind.STAGE_CHANGE, {"stage": stage.value})
        return state

    def _run_stage(
        self, state: RunState, stage: Stage
    ) -> tuple[RunState, Optional[str]]:
        """Attempt this stage's candidates. Returns (state, stop_reason)."""
        # REPRODUCE_BASELINE is a single fixed evaluation, not a search.
        attempts = 1 if stage is Stage.REPRODUCE_BASELINE else self._max_nodes_per_stage

        for _ in range(attempts):
            if state.budget.exhausted:
                self._emit(
                    state,
                    EventKind.BUDGET_WARNING,
                    {
                        "tripped": list(state.budget.tripped),
                        "stage": stage.value,
                        "action": "finalize",
                    },
                )
                return state, "budget_exhausted"

            # Recomputed every attempt rather than once per stage:
            # blocked_slots only ever grows, so a slot blocked partway
            # through a stage must stop being offered for the rest of it.
            candidate_slots = self._candidate_slots(state, stage)
            if stage is not Stage.REPRODUCE_BASELINE and not candidate_slots:
                # Nothing left to attack in this stage. That ends the
                # STAGE, not the run: a later stage has a different slot
                # set (STAGE_3_TUNE permits all six), so slots exhausted
                # here says nothing about what is available there.
                #
                # No node is consumed and the generator is never called.
                # `node` counts evaluations attempted, and there is no
                # candidate here to attempt - inventing one so the loop
                # has something to count would put a phantom attempt in
                # the journal.
                self._emit(
                    state,
                    EventKind.SLOT_BLOCKED,
                    {
                        "stage": stage.value,
                        "stage_slots": list(STAGE_SLOTS[stage]),
                        "blocked_slots": sorted(state.blocked_slots),
                        "reason": "every slot this stage may attack is blocked",
                        "action": "end_stage",
                    },
                )
                return state, None

            try:
                state, attempt_stop = self._attempt(state, stage, candidate_slots)
            except GeneratorExhausted as exc:
                # Running out of scripted hypotheses is a normal end to a
                # test run, not a crash. Ending through the ordinary
                # finalisation path means the journal still gets its
                # terminal event, so a replay can tell this run finished.
                self._emit(
                    state,
                    EventKind.ERROR,
                    {
                        "reason": "generator_exhausted",
                        "stage": stage.value,
                        "detail": str(exc),
                    },
                )
                return state, "generator_exhausted"

            if attempt_stop is not None:
                # Convergence ends the RUN, not just this stage: the
                # organizers' rule is written about the agent's iteration
                # loop as a whole, and reinterpreting it per-stage would be
                # a stretch we would then have to defend.
                return state, attempt_stop

        return state, None

    def _candidate_slots(
        self, state: RunState, stage: Stage
    ) -> tuple[SlotName, ...]:
        """This stage's permitted slots, minus the ones the run has blocked.

        The set the policy is allowed to choose from, and the set the
        Controller validates its answer against. Order follows
        STAGE_SLOTS - which follows SLOT_ORDER - rather than set iteration
        order, so a policy that depends on the order it is handed (like
        FixedOrderPolicy) behaves identically across runs and across
        Python processes.

        Two filters, deliberately composed here rather than left to the
        policy: the stage constraint protects the run's plan from a greedy
        policy (see STAGE_SLOTS) and blocked_slots keeps the search off
        arms already known to be broken. Neither is something a policy
        should be trusted to remember to apply.
        """
        return tuple(
            slot for slot in STAGE_SLOTS[stage] if slot not in state.blocked_slots
        )

    def _attempt(
        self, state: RunState, stage: Stage, candidate_slots: tuple[SlotName, ...]
    ) -> tuple[RunState, Optional[str]]:
        """Evaluate one candidate, end to end.

        Returns (state, stop_reason). The stop_reason is non-None only when
        this candidate's commit tripped convergence — a dead candidate
        never ends the run.

        `candidate_slots` is computed by the caller and is guaranteed
        non-empty for every stage except REPRODUCE_BASELINE, which selects
        no slot at all.
        """
        state = state.with_node_started()
        is_baseline = stage is Stage.REPRODUCE_BASELINE
        # None for the baseline: it evaluates a fixed published config that
        # no policy chose, so there is no arm to charge it to.
        target_slot: Optional[SlotName] = None

        if is_baseline:
            # No HYPOTHESIS event: the baseline is a known published
            # configuration, not something an LLM proposes. Asking a
            # generator to invent the thing we are trying to reproduce
            # would defeat the point of reproducing it.
            config = baseline_pipeline()
        else:
            # ONE card, built once, handed to both. If the policy and the
            # generator saw different cards, "what did the run look like
            # when this slot was picked" and "what did the generator know"
            # would be two different questions with two different answers,
            # and the journal could not answer either from one event.
            state_card = build_state_card(state)
            target_slot = self._policy.select_slot(state_card, candidate_slots)
            if target_slot not in candidate_slots:
                # Loud, immediately, at the point of the bug. The policy is
                # our own deterministic code - not an LLM - so this is not
                # a runtime condition to absorb, it is a defect. Carrying
                # on would attribute every subsequent cost and delta to the
                # wrong arm, and the run would look fine while measuring
                # something else. See ports.PolicyContractError.
                raise PolicyContractError(
                    f"{type(self._policy).__name__}.select_slot returned "
                    f"{target_slot!r}, which is not among the candidate slots "
                    f"{candidate_slots} offered in stage {stage.value}"
                )

            payload = self._generator.propose(state_card, target_slot)
            # Emitted before the check below, and unconditionally: the
            # journal records what the generator actually said, including
            # when what it said was wrong. A HYPOTHESIS event suppressed on
            # mismatch would leave an ERROR referring to a proposal that
            # appears nowhere in the log.
            self._emit(state, EventKind.HYPOTHESIS, asdict(payload))

            if payload.target_slot != target_slot:
                # THE DETERMINISTIC GUARD ON A GENERATOR THAT WANDERED.
                # The Controller does not obey the payload's choice. It was
                # asked for a hypothesis about one slot and returned one
                # about another, so what came back does not answer the
                # question the search policy asked - splicing it in would
                # hand slot selection back to the LLM through the back door,
                # which is the exact behaviour this PR removed.
                #
                # Classified CONTRACT for the same reason the realizer path
                # is: the collaborator failed to honour its side of the
                # port, rather than the candidate failing to train.
                #
                # A dead candidate, not a dead run. An LLM ignoring an
                # instruction is an expected operating condition; the node
                # is counted, the attempt is recorded, and the loop moves
                # on to the next candidate.
                self._emit(
                    state,
                    EventKind.ERROR,
                    {
                        "config_id": MISPROPOSED_CONFIG_ID,
                        "reason": "generator_slot_mismatch",
                        "requested_slot": target_slot,
                        "proposed_slot": payload.target_slot,
                        "error_class": ErrorClass.CONTRACT.value,
                        "error_excerpt": (
                            f"generator was asked for {target_slot!r} and "
                            f"proposed {payload.target_slot!r}"
                        ),
                    },
                )
                return (
                    state.with_outcome(
                        MISPROPOSED_CONFIG_ID,
                        None,
                        accepted=False,
                        # Charged to the slot that was REQUESTED, not the
                        # one the generator named. The requested slot is
                        # the arm the policy pulled, and an arm whose
                        # proposals keep coming back malformed is genuinely
                        # performing badly - crediting the attempt to a
                        # slot nobody selected would hide that.
                        target_slot=target_slot,
                        **_attempt_cost(None),
                    ),
                    None,
                )

            try:
                config = self._realize(state, payload)
            except RealizerExhausted as exc:
                # A hypothesis that cannot be turned into runnable code is
                # a dead candidate, not a dead run — absorbing exactly this
                # kind of localized failure is what the architecture is
                # for. Classified CONTRACT because the realizer failed to
                # honour its side of RealizerPort: it was asked for a
                # SlotConfig and could not produce one.
                self._emit(
                    state,
                    EventKind.ERROR,
                    {
                        "config_id": UNREALIZED_CONFIG_ID,
                        "reason": "realizer_exhausted",
                        "target_slot": payload.target_slot,
                        "error_class": ErrorClass.CONTRACT.value,
                        "error_excerpt": str(exc),
                    },
                )
                return (
                    state.with_outcome(
                        UNREALIZED_CONFIG_ID,
                        None,
                        accepted=False,
                        target_slot=target_slot,
                        # Genuinely zero: this candidate never reached the
                        # executor, so it consumed no evaluation cost. The
                        # realizer's own tokens are real spend but no port
                        # reports them yet - see HistoryEntry's cost docs.
                        **_attempt_cost(None),
                    ),
                    None,
                )

        self._emit(
            state,
            EventKind.EVAL_START,
            {"config_id": config.config_id, "seeds": list(self._seeds)},
        )
        result = self._executor.run(config, self._seeds)
        state = state.with_spend(result)
        primary = _mean_primary(result)
        self._emit(
            state,
            EventKind.EVAL_RESULT,
            {
                "config_id": result.config_id,
                "status": result.status.value,
                "primary": primary,
                "wall_seconds": result.wall_seconds,
                "gpu_seconds": result.gpu_seconds,
                "tokens": result.tokens_in + result.tokens_out,
            },
        )

        if result.status is Status.FAILED:
            # A failed candidate must never stop the run. Log what broke,
            # count the node, move on — robustness is a graded axis and a
            # loop that dies on the first bad config scores zero on it.
            self._emit(
                state,
                EventKind.ERROR,
                {
                    "config_id": result.config_id,
                    "error_class": result.error_class.value,
                    "error_excerpt": result.error_excerpt,
                },
            )
            state = state.with_outcome(
                result.config_id,
                None,
                accepted=False,
                target_slot=target_slot,
                # NON-ZERO, and that is the whole point. A candidate
                # that died halfway still burned what it took to get
                # there; recording zero would make the slot that
                # produced it look cheap. See HistoryEntry's cost docs.
                **_attempt_cost(result),
                # THE ONLY PLACE THIS IS SET. The circuit breaker counts
                # executor failures and nothing else - see
                # HistoryEntry.executor_failed for why the generator
                # mismatch and realizer exhaustion paths above must not.
                executor_failed=True,
            )
            # Recorded first, then judged: the entry just written is part
            # of the run this decision is about, so blocking has to see it.
            return self._maybe_block_slot(state, stage, target_slot), None

        if state.incumbent is None:
            # Nothing to compare against yet, so the gate is not called —
            # GatePort.compare has no meaningful behaviour with a None
            # incumbent, and passing one would force every future gate
            # implementation to special-case it.
            state = state.with_incumbent(result, config)
            self._emit(
                state,
                EventKind.DECISION,
                {
                    "verdict": True,
                    "delta_primary": 0.0,
                    "ci95": [0.0, 0.0],
                    "backtest_delta": 0.0,
                    "reason": "first candidate adopted as incumbent; nothing to compare against",
                },
            )
            # A committed revision, so it counts toward `iteration` and is
            # assessed like any other — but it can never converge anything
            # on its own: with no predecessor there is no improvement to
            # difference, and with no gate ruling its `significant` is None,
            # which breaks the internal rule's streak rather than feeding it.
            state = state.with_outcome(
                result.config_id,
                primary,
                accepted=True,
                target_slot=target_slot,
                **_attempt_cost(result),
            )
            return self._check_convergence(state)

        verdict = self._gate.compare(result, state.incumbent)
        self._emit(
            state,
            EventKind.DECISION,
            {
                "verdict": verdict.accept,
                "delta_primary": verdict.delta,
                "ci95": list(verdict.ci95),
                "backtest_delta": verdict.backtest_delta,
                "reason": verdict.reason,
            },
        )
        if verdict.accept:
            state = state.with_incumbent(result, config)
        state = state.with_outcome(
            result.config_id,
            primary,
            accepted=verdict.accept,
            delta=verdict.delta,
            ci95=verdict.ci95,
            # Computed once, here, so the stored flag and the DECISION event
            # can never disagree about what the same interval meant.
            significant=is_significant(verdict.ci95),
            target_slot=target_slot,
            **_attempt_cost(result),
        )
        if not verdict.accept:
            # Only committed revisions are iterations, so a rejection
            # changes nothing convergence looks at.
            return state, None
        return self._check_convergence(state)

    def _maybe_block_slot(
        self, state: RunState, stage: Stage, target_slot: Optional[SlotName]
    ) -> RunState:
        """Block `target_slot` for this stage once its failures reach k.

        THE CIRCUIT BREAKER. The agent's job is to spend a fixed budget of
        evaluations well, and re-attacking a slot whose last k candidates
        all died in the executor spends it on re-debugging one bad idea.
        After k, the slot leaves the candidate set and the search moves on.

        WHAT COUNTS, AND WHAT POINTEDLY DOES NOT. Only executor failures,
        counted consecutively per slot by `SlotStats.consecutive_failures`.
        A generator that proposed the wrong slot and a realizer that could
        not produce code are contract breaches by other workstreams'
        collaborators, and a clean gate rejection is a working arm honestly
        reporting no improvement. Blocking on any of those would delete
        good arms over somebody else's bug, or over exactly the negative
        result the search exists to collect. See
        HistoryEntry.executor_failed.

        STAGE-SCOPED. `RunState.with_stage` clears blocked_slots on entry
        to the next stage, so a block costs a slot the remainder of one
        stage and never the run - a transient OOM must not permanently
        delete an arm. The failures stay in history either way, so the
        bandit still scores the arm on its record when it returns.

        Returns the state unchanged when there is nothing to block, so the
        caller can use it unconditionally. `target_slot` is None for the
        REPRODUCE_BASELINE evaluation: that runs a fixed published config
        no policy chose, and a baseline that will not run is a broken
        harness rather than a bad arm - there is no slot to hold
        responsible and blocking one would be a guess.
        """
        if target_slot is None:
            return state
        stats = slot_stats(state).get(target_slot)
        if stats is None or stats.consecutive_failures < self._failures_before_block:
            return state

        state = state.with_blocked_slot(target_slot)
        # Emitted AFTER the ERROR event for the failure that tripped it,
        # and after the state already carries the block, so a reader sees
        # the cause, then the consequence, then a `blocked_slots` that
        # agrees with the run from this point on.
        self._emit(
            state,
            EventKind.SLOT_BLOCKED,
            {
                "slot": target_slot,
                "consecutive_failures": stats.consecutive_failures,
                "threshold": self._failures_before_block,
                "stage": stage.value,
                "blocked_slots": sorted(state.blocked_slots),
                "reason": "consecutive executor failures reached the threshold",
                # Distinguishes this from the other SLOT_BLOCKED emission,
                # which reports a stage ending because everything was
                # already blocked rather than a slot being blocked now.
                "action": "block_slot",
            },
        )
        return state

    def _check_convergence(self, state: RunState) -> tuple[RunState, Optional[str]]:
        """Assess both stopping rules after a commit and log the result.

        Emitted after DECISION, never instead of it: the decision and the
        consequence of that decision are two separate facts and a reader
        should see both. The payload carries the whole ConvergenceStatus so
        the event is self-contained — someone auditing the run can re-check
        the arithmetic without reconstructing the window from surrounding
        events — plus `iteration_definition`, so it is unambiguous which
        reading of "iteration" produced the numbers.
        """
        status = assess(state.committed_revisions)
        self._emit(
            state,
            EventKind.CONVERGENCE_CHECK,
            {
                "iteration_definition": ITERATION_DEFINITION,
                "converged": status.converged,
                "by_rule": status.by_rule,
                "organizers_converged": status.organizers_converged,
                "internal_converged": status.internal_converged,
                "recent_deltas": list(status.recent_deltas),
                "recent_significant": list(status.recent_significant),
                "iterations_considered": status.iterations_considered,
                "epsilon": status.epsilon,
                "n_required": status.n_required,
            },
        )
        return state, ("converged" if status.converged else None)

    def _finalize(self, state: RunState, stop_reason: Optional[str]) -> RunState:
        """Enter FINALIZE, log the terminal events, land in DONE."""
        state = self._enter(state, Stage.FINALIZE)
        self._emit(
            state,
            EventKind.FINALIZE,
            {
                "incumbent_config_id": (
                    state.incumbent_config.config_id
                    if state.incumbent_config is not None
                    else None
                ),
                "incumbent_primary": _mean_primary(state.incumbent),
                "stop_reason": stop_reason or "stages_complete",
            },
        )
        state = state.with_stage(Stage.DONE)
        self._emit(
            state,
            EventKind.RUN_END,
            {
                "stop_reason": stop_reason or "stages_complete",
                "iteration": state.iteration,
                "node": state.node,
            },
        )
        return state

    # -- helpers -------------------------------------------------------

    def _realize(
        self, state: RunState, payload: HypothesisPayload
    ) -> PipelineConfig:
        """Realize the hypothesis into slot code, then splice it in.

        Two steps, deliberately separate. RealizerPort produces a
        SlotConfig for ONE slot; the Controller decides where that slot
        sits, because only the Controller knows the current incumbent and
        only it is answerable for candidate lineage.

        WHY parent_id MATTERS: it records which incumbent this candidate
        was derived from, so a reader holding nothing but the journal can
        reconstruct the whole search tree - which experiment branched from
        which, and where a winning line actually started. Without it the
        log is a flat list of configs with no ancestry and "how did we get
        here" is unanswerable after the fact. It is deliberately NOT part
        of the content hash (config_id hashes slots only), so recording
        lineage never perturbs a candidate's identity or its cache key:
        two runs reaching the same config by different routes still share
        cached artifacts.

        The candidate is the incumbent with exactly one slot swapped. That
        one-slot diff is what makes the cascading slot_hash reuse pay off -
        everything upstream of the changed slot keeps its cached artifact.
        """
        slot_config = self._realizer.realize(payload)

        self._emit(
            state,
            EventKind.CODE_EMITTED,
            {
                "target_slot": payload.target_slot,
                "impl": slot_config.impl,
                # Presence and size only, never the blob. A journal that
                # inlines generated source stops being readable and can
                # balloon one JSONL line past anything a human will scroll
                # through - and the code is already content-addressed
                # inside the config, so the log does not need a second copy.
                "has_code_blob": slot_config.code_blob is not None,
                "code_blob_chars": len(slot_config.code_blob or ""),
            },
        )

        base = (
            state.incumbent_config.slots
            if state.incumbent_config is not None
            else BASELINE_SLOTS
        )
        slots = dict(base)
        slots[payload.target_slot] = slot_config
        return PipelineConfig(
            slots=slots,
            parent_id=(
                state.incumbent_config.config_id
                if state.incumbent_config is not None
                else None
            ),
        )

    def _emit(
        self, state: RunState, kind: EventKind, payload: dict[str, Any]
    ) -> None:
        """Write one journal event, stamped with the state as it is now.

        `ts` is the only value here that reads the clock, and it never
        enters a payload — every other field is derived from state, so two
        runs with the same inputs differ in nothing but their timestamps.
        """
        self._journal.append(
            JournalEvent(
                ts=datetime.now(timezone.utc).isoformat(),
                run_id=state.run_id,
                iteration=state.iteration,
                node=state.node,
                kind=kind,
                payload=payload,
            )
        )


def _attempt_cost(result: Optional[CandidateResult]) -> dict[str, Any]:
    """This attempt's cost, in the keyword shape `with_outcome` takes.

    A helper rather than four inline literals because it is written at four
    call sites - accepted, rejected, failed and unrealized - and the one
    thing that must be true of all four is that they agree. If the failure
    path ever spelled its cost differently from the success path, a
    cost-aware policy would read a systematic bias between slots that fail
    often and slots that do not, which is precisely the signal it is meant
    to be measuring.

    `None` means the attempt never reached the executor (unrealized, or a
    slot mismatch), so there is no measured spend to record. That is an
    honest zero, not a placeholder: no evaluation was run. It is NOT the
    same as a FAILED result, which has real numbers on it and must carry
    them.

    `tokens` is summed in+out to match what `with_spend` charges the budget
    and what EVAL_RESULT logs, so the per-slot totals and the run-level
    counters are the same quantity.
    """
    if result is None:
        return {"wall_seconds": 0.0, "gpu_seconds": 0.0, "tokens": 0}
    return {
        "wall_seconds": result.wall_seconds,
        "gpu_seconds": result.gpu_seconds,
        "tokens": result.tokens_in + result.tokens_out,
    }


def _mean_primary(result: Optional[CandidateResult]) -> Optional[float]:
    """Mean validation primary across seeds — for the log only.

    Never an acceptance input: that is the gate's job, and it needs the
    per-seed pairing this average discards. None when there is nothing to
    average (no result, or a candidate that failed before scoring).
    """
    if result is None or not result.val:
        return None
    return sum(m.primary for m in result.val.values()) / len(result.val)


# Import-time check that the baseline fills every slot the hash walks; a
# missing slot would otherwise only surface as a KeyError deep inside the
# first evaluation. A raise rather than an assert, because `python -O`
# strips asserts and this guard is worth more than the nanosecond it costs.
if set(BASELINE_SLOTS) != set(SLOT_ORDER):
    raise RuntimeError(
        "BASELINE_SLOTS must fill every slot in contracts.SLOT_ORDER; "
        f"missing {set(SLOT_ORDER) - set(BASELINE_SLOTS)}"
    )
