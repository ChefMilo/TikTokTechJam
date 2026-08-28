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
    Status,
)
from controller.ports import (
    ExecutorPort,
    GatePort,
    GeneratorExhausted,
    GeneratorPort,
    JournalPort,
    RealizerExhausted,
    RealizerPort,
)
from controller.state import (
    RunState,
    STAGE_ORDER,
    Stage,
    build_state_card,
    next_stage,
)

__all__ = ["BASELINE_SLOTS", "UNREALIZED_CONFIG_ID", "Controller", "baseline_pipeline"]

UNREALIZED_CONFIG_ID = "<unrealized>"
"""Stands in for a candidate the realizer could not produce.

`HistoryEntry.config_id` is a str, and a hypothesis that never became a
config has no content to hash. A visible sentinel beats an empty string:
in the state card's `recent_history` it reads as an explicit "this attempt
produced nothing", rather than as a value someone forgot to fill in.
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
        journal: JournalPort,
        budget: Optional[Budget] = None,
        seeds: Sequence[int] = (0,),
        nodes_per_stage: int = 3,
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
        self._journal = journal
        # An unlimited Budget by default. No agreed wall-clock, token or
        # evaluation ceiling exists anywhere in the repo yet, and inventing
        # one here would be fiction that later reads as a decision someone
        # made on purpose.
        self._budget = budget if budget is not None else Budget()
        self._seeds = tuple(seeds)
        self._nodes_per_stage = nodes_per_stage
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
                "nodes_per_stage": self._nodes_per_stage,
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
        attempts = 1 if stage is Stage.REPRODUCE_BASELINE else self._nodes_per_stage

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

            try:
                state = self._attempt(state, stage)
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

        return state, None

    def _attempt(self, state: RunState, stage: Stage) -> RunState:
        """Evaluate one candidate, end to end."""
        state = state.with_node_started()
        is_baseline = stage is Stage.REPRODUCE_BASELINE

        if is_baseline:
            # No HYPOTHESIS event: the baseline is a known published
            # configuration, not something an LLM proposes. Asking a
            # generator to invent the thing we are trying to reproduce
            # would defeat the point of reproducing it.
            config = baseline_pipeline()
        else:
            payload = self._generator.propose(build_state_card(state))
            self._emit(state, EventKind.HYPOTHESIS, asdict(payload))
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
                return state.with_outcome(
                    UNREALIZED_CONFIG_ID, None, accepted=False
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
            return state.with_outcome(result.config_id, None, accepted=False)

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
            return state.with_outcome(result.config_id, primary, accepted=True)

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
        return state.with_outcome(result.config_id, primary, accepted=verdict.accept)

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
